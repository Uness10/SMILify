"""Numpy re-implementation of the SMIL/SMAL forward kinematics used by SMILify.

Mirrors smal_model/smal_torch.py + smal_model/batch_lbs.py exactly for the joint
chain (we never need the skinned mesh here, only J_transformed), so the angles
computed downstream are the angles of the rig that was actually rendered.

Verified against the repo's own torch implementation by tests/test_fk.py.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- io
class _Unpickler(pickle.Unpickler):
    """Tolerates chumpy-pickled SMAL files without importing chumpy."""

    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Ch
        if module == "scipy.sparse.csc" or module.startswith("scipy.sparse"):
            import scipy.sparse

            return getattr(scipy.sparse, name, dict)
        return super().find_class(module, name)


class _Ch(np.ndarray):
    def __new__(cls, *a, **k):
        return np.zeros(0).view(cls)

    def __array__(self, *a, **k):
        return np.asarray(getattr(self, "x", np.zeros(0)))

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


def _undo_chumpy(x):
    return np.asarray(x) if not hasattr(x, "r") else np.asarray(x.r)


@dataclass
class SmilModel:
    v_template: np.ndarray      # (V, 3)
    shapedirs: np.ndarray       # (V, 3, B)
    J_regressor: np.ndarray     # (J, V)
    J_rest: np.ndarray          # (J, 3)  rest joints of the mean shape
    parents: np.ndarray         # (J,)
    joint_names: list
    static_joint_locs: bool

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    def joint(self, name: str) -> int:
        return self.joint_names.index(name)

    def rest_joints(self, betas: Optional[np.ndarray]) -> np.ndarray:
        """Shape-dependent rest joint locations, exactly as smal_torch does."""
        if self.static_joint_locs or betas is None:
            return self.J_rest.copy()
        b = np.asarray(betas, float).reshape(-1)
        nb = min(len(b), self.shapedirs.shape[2])
        v = self.v_template + self.shapedirs[:, :, :nb] @ b[:nb]
        return self.J_regressor @ v


def load_model(path: str) -> SmilModel:
    with open(path, "rb") as f:
        u = _Unpickler(f)
        u.encoding = "latin1"
        dd = u.load()

    Jr = dd["J_regressor"]
    Jr = np.asarray(Jr.todense()) if hasattr(Jr, "todense") else np.asarray(_undo_chumpy(Jr))
    if Jr.shape[0] != len(dd["J_names"]):
        Jr = Jr.T
    return SmilModel(
        v_template=np.asarray(_undo_chumpy(dd["v_template"]), float),
        shapedirs=np.asarray(_undo_chumpy(dd["shapedirs"]), float),
        J_regressor=np.asarray(Jr, float),
        J_rest=np.asarray(_undo_chumpy(dd["J"]), float),
        parents=np.asarray(dd["kintree_table"][0], int),
        joint_names=[str(n) for n in dd["J_names"]],
        static_joint_locs=bool(dd.get("static_joint_locs", False)),
    )


# --------------------------------------------------------------------- rotations
def rodrigues(theta: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 3, 3). Same 1e-8 guard as batch_lbs.py."""
    t = np.asarray(theta, float)
    flat = t.reshape(-1, 3)
    angle = np.linalg.norm(flat + 1e-8, axis=1, keepdims=True)
    r = flat / angle
    c, s = np.cos(angle)[:, :, None], np.sin(angle)[:, :, None]
    outer = r[:, :, None] * r[:, None, :]
    K = np.zeros((len(flat), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -r[:, 2], r[:, 1]
    K[:, 1, 0], K[:, 1, 2] = r[:, 2], -r[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -r[:, 1], r[:, 0]
    R = c * np.eye(3)[None] + (1 - c) * outer + s * K
    return R.reshape(t.shape[:-1] + (3, 3))


# ------------------------------------------------------------------------- lbs
def global_rigid_transformation(
    Rs: np.ndarray,                       # (F, J, 3, 3)
    J: np.ndarray,                        # (F, J, 3) rest joints
    parents: np.ndarray,
    betas_logscale: Optional[np.ndarray] = None,   # (F, J, 3)
    betas_trans: Optional[np.ndarray] = None,      # (F, J, 3)
    propagate_scaling: bool = False,
):
    """Port of batch_lbs.batch_global_rigid_transformation (joint positions only)."""
    F, JJ = Rs.shape[0], Rs.shape[1]
    scale = np.ones((F, JJ, 3)) if betas_logscale is None else np.exp(betas_logscale)
    S = np.zeros((F, JJ, 3, 3))
    S[..., 0, 0], S[..., 1, 1], S[..., 2, 2] = scale[..., 0], scale[..., 1], scale[..., 2]

    # batch_lbs flips y on the translation offsets ("needed for Unreal data")
    toff = None if betas_trans is None else betas_trans * np.array([1.0, -1.0, 1.0])

    Rg = np.empty((F, JJ, 3, 3))
    tg = np.empty((F, JJ, 3))
    Rg[:, 0] = Rs[:, 0]
    tg[:, 0] = J[:, 0]

    for i in range(1, JJ):
        p = parents[i]
        j_here = J[:, i] - J[:, p]
        if toff is not None:
            j_here = j_here + toff[:, i]
        s_par_inv = np.eye(3)[None].repeat(F, 0) if propagate_scaling else np.linalg.inv(S[:, p])
        rot_new = s_par_inv @ Rs[:, i] @ S[:, i]
        Rg[:, i] = Rg[:, p] @ rot_new
        tg[:, i] = (Rg[:, p] @ j_here[..., None])[..., 0] + tg[:, p]
    return tg, Rg


def forward_kinematics(
    model: SmilModel,
    poses: np.ndarray,                     # (F, J, 3) axis-angle, index 0 = global
    betas: Optional[np.ndarray] = None,    # (B,) or (F, B)
    log_beta_scales: Optional[np.ndarray] = None,
    betas_trans: Optional[np.ndarray] = None,
    propagate_scaling: bool = False,
    zero_global: bool = True,
):
    """-> (joints (F,J,3), global joint orientations (F,J,3,3), rest joints (F,J,3)).

    zero_global=True removes the predicted global rotation, so the output is in the
    animal's own body frame (+x anterior, +y left, +z dorsal) with the root at the
    origin after subtracting joints[:, 0]. That is the frame every angle and every
    tarsus trajectory in this analysis is expressed in.
    """
    poses = np.asarray(poses, float)
    F, JJ = poses.shape[:2]
    if betas is None:
        Jrest = np.repeat(model.J_rest[None], F, 0)
    else:
        b = np.asarray(betas, float)
        if b.ndim == 1:
            Jrest = np.repeat(model.rest_joints(b)[None], F, 0)
        else:
            Jrest = np.stack([model.rest_joints(b[i]) for i in range(F)])

    p = poses.copy()
    if zero_global:
        p[:, 0] = 0.0
    Rs = rodrigues(p)
    tg, Rg = global_rigid_transformation(
        Rs, Jrest, model.parents, log_beta_scales, betas_trans, propagate_scaling
    )
    return tg, Rg, Jrest
