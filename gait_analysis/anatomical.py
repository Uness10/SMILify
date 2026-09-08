"""Anatomical joint angles for the SMILy_STICK rig, in the animal's body frame.

Conventions are the ones the repo already validated against the stick-insect
literature (see diagnostics/probe_stick_joint_axes_PROBE.py and
diagnostics/joint_dof_VERIFY_PROBE.log):

  model frame   +x anterior, +y left, +z dorsal
  ThC (l_N_co_*)   3-DoF ball-and-socket.  alpha = protraction/retraction (azimuth
                   of the coxa->FTi line in the horizontal plane, 0 = perpendicular
                   to the body axis, + = anterior);  beta = levation/depression
                   (elevation of the same line above horizontal, + = dorsal).
  CTr (l_N_tr_*)   1-DoF hinge ("C-TF" in Guschlbauer et al. 2022, because the
                   trochanter and femur are fused).  delta = interior angle at the
                   trochanter between the coxa and the trochanterofemur.
  fused (l_N_fe_*) locked to zero on all three axes (Theunissen et al. 2015).
  FTi (l_N_ti_*)   1-DoF hinge.  gamma = interior angle between femur and tibia,
                   180 deg = fully extended, small = fully flexed.
  TiTa (l_N_ta_*)  tarsus; reported for completeness.

Every angle is computed from the *posed joint positions*, not from the raw
axis-angle components, so it is free of Euler-order / axis-angle ambiguity and is
directly comparable to angles measured on real animals.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

LEGS = [(n, s) for n in (1, 2, 3) for s in ("r", "l")]
LEG_LABEL = {(1, "r"): "R1 front", (2, "r"): "R2 middle", (3, "r"): "R3 hind",
             (1, "l"): "L1 front", (2, "l"): "L2 middle", (3, "l"): "L3 hind"}
SEGS = ("co", "tr", "fe", "ti", "ta", "pt")


def jname(n, s, seg):
    return f"l_{n}_{seg}_{s}"


def _unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def _interior_angle(a, b):
    """Angle (deg) between two stacks of vectors."""
    return np.degrees(np.arccos(np.clip((_unit(a) * _unit(b)).sum(-1), -1, 1)))


def rest_angles(model, n: int, s: str) -> dict:
    """Rest-pose alpha/beta/gamma of the rig - the absolute<->relative offsets."""
    P = {seg: model.J_rest[model.joint(jname(n, s, seg))] for seg in SEGS}
    v = P["ti"] - P["co"]
    lat = -v[1] if s == "r" else v[1]
    return {
        "alpha": float(np.degrees(np.arctan2(v[0], lat))),
        "beta": float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))),
        "gamma": float(_interior_angle(P["fe"] - P["ti"], P["ta"] - P["ti"])),
        "delta": float(_interior_angle(P["co"] - P["tr"], P["ti"] - P["tr"])),
    }


def thc_intrinsic(poses: np.ndarray, model, n: int, s: str) -> dict:
    """Joint-intrinsic ThC angles, straight from the pose parameter of l_N_co_*.

    The ThC relative rotation (model frame, relative to the rig's rest pose) is
    decomposed ZXY:  R = Rz(a) Rx(b) Ry(c).  From the rig's own axis probe,
    +z is protraction on the right and retraction on the left, and +x is
    depression on the right and levation on the left, so the signs are flipped
    per side to give one convention for both.  Adding the rest-pose values makes
    these absolute anatomical angles, directly comparable to the literature.

    Unlike the position-derived angles these depend only on this joint, not on
    the state of the CTr/FTi joints downstream of it.
    """
    R = Rotation.from_rotvec(poses[:, model.joint(jname(n, s, "co")), :])
    a, b, c = Rotation.as_euler(R, "ZXY", degrees=True).T
    sgn = 1.0 if s == "r" else -1.0
    rest = rest_angles(model, n, s)
    return {
        "ThC_alpha_protraction_rel": sgn * a,
        "ThC_beta_levation_rel": -sgn * b,
        "ThC_axial_rotation_rel": sgn * c,
        "ThC_alpha_protraction_abs": rest["alpha"] + sgn * a,
        "ThC_beta_levation_abs": rest["beta"] - sgn * b,
    }


def hinge_rotation(poses: np.ndarray, model, n: int, s: str, seg: str) -> dict:
    """Signed rotation of a 1-DoF hinge about the leg-plane normal, plus the
    off-axis residual that a perfect hinge would leave at zero."""
    r = poses[:, model.joint(jname(n, s, seg)), :]
    nrm = hinge_axis(model, n, s, seg)
    on = r @ nrm
    off = np.linalg.norm(r - on[:, None] * nrm[None, :], axis=1)
    return {f"{seg}_hinge_flexion_rel": np.degrees(on), f"{seg}_offaxis_residual": np.degrees(off)}


def leg_angles(joints: np.ndarray, model, n: int, s: str) -> dict:
    """joints: (F, J, 3) posed joint positions in the body frame.

    Returns dict of (F,) arrays in degrees.
    """
    P = {seg: joints[:, model.joint(jname(n, s, seg))] for seg in SEGS}
    v = P["ti"] - P["co"]                       # coxa -> distal end of trochanterofemur
    lat = -v[:, 1] if s == "r" else v[:, 1]     # outward-lateral component
    return {
        "ThC_alpha_protraction": np.degrees(np.arctan2(v[:, 0], lat)),
        "ThC_beta_levation": np.degrees(np.arctan2(v[:, 2], np.hypot(v[:, 0], v[:, 1]))),
        "CTr_delta": _interior_angle(P["co"] - P["tr"], P["ti"] - P["tr"]),
        "FTi_gamma": _interior_angle(P["fe"] - P["ti"], P["ta"] - P["ti"]),
        "TiTa": _interior_angle(P["ti"] - P["ta"], P["pt"] - P["ta"]),
    }


def all_leg_angles(joints, model):
    return {leg: leg_angles(joints, model, *leg) for leg in LEGS}


# ------------------------------------------------------- dominant rotation axes
def dominant_axes(poses: np.ndarray, joint_idx: int, frames: np.ndarray | None = None):
    """PCA of a joint's axis-angle vectors -> the axes it actually rotates about.

    `poses` is (F, J, 3) axis-angle in the model frame (index 0 = global rotation).
    Returns (axes (3,3) as rows e1,e2,e3 ordered by variance, sd_deg (3,),
    projections (F,3) in degrees, mean rotation vector).

    This is the empirical answer to "which two axes dominate": e1 and e2 span the
    plane the joint's rotation vector actually lives in, and sd_deg says how much
    of the motion each carries.
    """
    r = poses[:, joint_idx, :]
    if frames is not None:
        r_win = r[frames]
    else:
        r_win = r
    mu = r_win.mean(0)
    X = r_win - mu
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    sd = S / np.sqrt(max(len(X) - 1, 1))
    proj = np.degrees((r - mu) @ Vt.T)
    return Vt, np.degrees(sd), proj, mu


ANATOMICAL_AXES = {
    # model-frame unit axes, from probe_stick_joint_axes_PROBE.py section 2
    "protraction(+z, sign flips L/R)": np.array([0.0, 0.0, 1.0]),
    "levation(+x, sign flips L/R)": np.array([1.0, 0.0, 0.0]),
    "lateral(+y)": np.array([0.0, 1.0, 0.0]),
}


def leg_plane_normal(model, n, s):
    """SVD-fitted normal of the resting leg plane = the CTr/FTi hinge axis.

    Sign is arbitrary here; use hinge_axis() when the sign has to mean something.
    A least-squares plane is used rather than a cross product because the femur
    and tibia are nearly collinear in the rest pose (probe section 3).
    """
    P = np.array([model.J_rest[model.joint(jname(n, s, seg))] for seg in SEGS])
    Q = P - P.mean(0)
    nrm = np.linalg.svd(Q)[2][-1]
    return nrm / np.linalg.norm(nrm)


_HINGE_CACHE = {}


def hinge_axis(model, n, s, seg, eps_deg=5.0):
    """Leg-plane normal, signed so that a POSITIVE rotation about it is FLEXION.

    The sign is determined the same way the rig probe determines its axis signs:
    apply a small test rotation to that joint alone, run the model's own forward
    kinematics, and see which way the interior angle moves. Nothing is assumed.
    """
    key = (id(model), n, s, seg)
    if key in _HINGE_CACHE:
        return _HINGE_CACHE[key]
    import smil_fk

    nrm = leg_plane_normal(model, n, s)
    poses = np.zeros((2, model.n_joints, 3))
    poses[1, model.joint(jname(n, s, seg))] = np.radians(eps_deg) * nrm
    J, _, _ = smil_fk.forward_kinematics(model, poses, betas=None, zero_global=True)
    key_angle = "FTi_gamma" if seg == "ti" else "CTr_delta"
    a0, a1 = (leg_angles(J, model, n, s)[key_angle][i] for i in (0, 1))
    out = nrm if a1 < a0 else -nrm          # flexion = interior angle decreases
    _HINGE_CACHE[key] = out
    return out
