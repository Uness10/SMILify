"""
Flat-vector layout for SMIL parameter heads.

``SMILImageRegressor`` computes the per-group output widths in
``_calculate_output_dims()`` and then parses the head's flat output vector back
into a parameter dict.  That parsing currently exists twice (once in
``SMILImageRegressor._forward_mlp``, once in
``MultiViewSMILImageRegressor._predict_body_params``), which is exactly the kind
of duplicate-by-development-stage code ``CLAUDE.md`` asks to unify.

This module holds a single, side-effect-free description of that layout so the
multi-animal MLP head has one implementation to depend on.  It deliberately does
*not* recompute the widths: it reads them off an already-configured regressor,
so there is only ever one source of truth for the dimension math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass(frozen=True)
class ParameterLayout:
    """Widths of each parameter group inside a head's flat output vector.

    The field order below *is* the layout order and must match
    ``SMILImageRegressor._forward_mlp``.
    """

    global_rot_dim: int
    joint_rot_dim: int
    betas_dim: int
    trans_dim: int
    fov_dim: int
    cam_rot_dim: int
    cam_trans_dim: int
    scales_dim: int
    joint_trans_dim: int

    # Reshaping hints
    n_pose: int
    rotation_representation: str = "axis_angle"
    scale_trans_mode: str = "separate"
    scales_use_pca: bool = True
    joint_trans_use_pca: bool = True

    @property
    def rotation_width(self) -> int:
        """Number of scalars used per rotation (3 for axis-angle, 6 for 6D)."""
        return 6 if self.rotation_representation == "6d" else 3

    @property
    def total_dim(self) -> int:
        """Width of the head's flat output vector."""
        return (
            self.global_rot_dim
            + self.joint_rot_dim
            + self.betas_dim
            + self.trans_dim
            + self.fov_dim
            + self.cam_rot_dim
            + self.cam_trans_dim
            + self.scales_dim
            + self.joint_trans_dim
        )

    @classmethod
    def from_regressor(cls, regressor: Any) -> "ParameterLayout":
        """Read the layout off a configured ``SMILImageRegressor``.

        ``_calculate_output_dims()`` must already have run (it does, in
        ``SMILImageRegressor.__init__``).  Reading rather than recomputing keeps
        the dimension math in exactly one place.
        """
        import config as smil_config
        from smal_fitter.neuralSMIL.training_config import TrainingConfig

        use_pca = True
        if getattr(regressor, "scale_trans_mode", "separate") == "separate":
            scale_trans_config = TrainingConfig.get_scale_trans_config()
            use_pca = scale_trans_config.get("separate", {}).get("use_pca_transformation", True)

        return cls(
            global_rot_dim=regressor.global_rot_dim,
            joint_rot_dim=regressor.joint_rot_dim,
            betas_dim=regressor.betas_dim,
            trans_dim=regressor.trans_dim,
            fov_dim=regressor.fov_dim,
            cam_rot_dim=regressor.cam_rot_dim,
            cam_trans_dim=regressor.cam_trans_dim,
            scales_dim=regressor.scales_dim,
            joint_trans_dim=regressor.joint_trans_dim,
            n_pose=smil_config.N_POSE,
            rotation_representation=getattr(regressor, "rotation_representation", "axis_angle"),
            scale_trans_mode=getattr(regressor, "scale_trans_mode", "separate"),
            scales_use_pca=use_pca,
            joint_trans_use_pca=use_pca,
        )


def parse_flat_parameter_vector(
    output: torch.Tensor,
    layout: ParameterLayout,
    include_camera: bool = True,
) -> Dict[str, torch.Tensor]:
    """Split a head's flat output ``(B, total_dim)`` into a parameter dict.

    Args:
        output: Flat head output.
        layout: Group widths, in layout order.
        include_camera: When ``False`` the camera slots are still *consumed*
            (so the offsets of the groups after them stay correct) but are not
            emitted.  That is the multi-animal case: the camera is scene level
            and comes from a dedicated head, never from a specimen head.

    Returns:
        Dict with ``global_rot``, ``joint_rot``, ``betas``, ``trans`` and,
        depending on configuration, ``fov`` / ``cam_rot`` / ``cam_trans`` /
        ``log_beta_scales`` / ``betas_trans``.
    """
    if output.dim() != 2:
        raise ValueError(f"expected a 2-D (B, D) tensor, got shape {tuple(output.shape)}")
    if output.shape[1] != layout.total_dim:
        raise ValueError(f"head output width {output.shape[1]} does not match layout total_dim {layout.total_dim}")

    batch_size = output.shape[0]
    params: Dict[str, torch.Tensor] = {}
    idx = 0

    params["global_rot"] = output[:, idx : idx + layout.global_rot_dim]
    idx += layout.global_rot_dim

    joint_rot_flat = output[:, idx : idx + layout.joint_rot_dim]
    params["joint_rot"] = joint_rot_flat.view(batch_size, layout.n_pose, layout.rotation_width)
    idx += layout.joint_rot_dim

    params["betas"] = output[:, idx : idx + layout.betas_dim]
    idx += layout.betas_dim

    params["trans"] = output[:, idx : idx + layout.trans_dim]
    idx += layout.trans_dim

    fov = output[:, idx : idx + layout.fov_dim]
    idx += layout.fov_dim
    cam_rot = output[:, idx : idx + layout.cam_rot_dim]
    idx += layout.cam_rot_dim
    cam_trans = output[:, idx : idx + layout.cam_trans_dim]
    idx += layout.cam_trans_dim
    if include_camera:
        params["fov"] = fov
        params["cam_rot"] = cam_rot.view(batch_size, 3, 3) if layout.cam_rot_dim == 9 else cam_rot
        params["cam_trans"] = cam_trans

    if layout.scales_dim > 0:
        scales_flat = output[:, idx : idx + layout.scales_dim]
        params["log_beta_scales"] = _maybe_reshape_per_joint(scales_flat, layout.scales_use_pca, layout.scale_trans_mode)
        idx += layout.scales_dim

    if layout.joint_trans_dim > 0:
        trans_flat = output[:, idx : idx + layout.joint_trans_dim]
        params["betas_trans"] = _maybe_reshape_per_joint(
            trans_flat, layout.joint_trans_use_pca, layout.scale_trans_mode
        )
        idx += layout.joint_trans_dim

    return params


def _maybe_reshape_per_joint(flat: torch.Tensor, use_pca: bool, scale_trans_mode: str) -> torch.Tensor:
    """Keep PCA weights flat, reshape raw per-joint values to ``(B, J, 3)``.

    Mirrors the branching in ``SMILImageRegressor._forward_mlp``: in
    ``separate`` mode with PCA transformation the head emits PCA weights (same
    width as betas) and the loss is computed on those directly; otherwise the
    values are per-joint XYZ triples.
    """
    if scale_trans_mode == "separate" and use_pca:
        return flat
    n_joints = flat.shape[1] // 3
    return flat.view(flat.shape[0], n_joints, 3)


def strip_camera(params: Dict[str, torch.Tensor], inplace: bool = False) -> Dict[str, torch.Tensor]:
    """Drop camera entries from a parameter dict.

    Specimen heads must not carry camera predictions: the camera belongs to the
    scene/view, not to an animal (design doc §6).  Applied to the transformer
    decoder head, whose output width is fixed and always includes camera slots.
    """
    target = params if inplace else dict(params)
    for key in ("fov", "cam_rot", "cam_trans"):
        target.pop(key, None)
    return target


def camera_only(params: Dict[str, torch.Tensor]) -> Dict[str, Optional[torch.Tensor]]:
    """Extract just the camera entries from a parameter dict."""
    return {key: params[key] for key in ("fov", "cam_rot", "cam_trans") if key in params}
