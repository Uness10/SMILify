"""Shared inference helpers for the single-view and multi-view entrypoints.

``run_singleview_inference.py`` and ``run_multiview_inference.py`` used to keep
near-duplicate copies of the temporal smoother, the parameter device shuffling,
the sub-clip splitting and — most importantly — the rules that decide *how a
prediction is turned back into a rendered mesh*:

* which camera intrinsics / extrinsics the renderer is given (this differs
  between the ``model_centric`` and ``camera_centric`` frame conventions),
* how the shape-space outputs are interpreted (``scale_trans_mode`` =
  ``separate`` with PCA weights vs. per-joint values, vs.
  ``entangled_with_betas``, vs. ``ignore``),
* whether the mesh is placed with the legacy 10x UE scaling or the predicted
  per-sample ``mesh_scale``.

The single-view copies had drifted out of sync with the multi-view ones (see
issue #100), which is exactly the class of bug this module exists to prevent:
there is now one implementation of each rule and both entrypoints import it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Parameter plumbing
# --------------------------------------------------------------------------- #


class PredictionSmoother:
    """Temporal smoother that applies a moving average over predicted parameters.

    Maintains a ring buffer of the last ``window_size`` predictions and returns
    their element-wise mean. Metadata keys (non-tensor values and lists whose
    length varies across frames) are passed through from the latest frame.

    The same smoother is used by both entrypoints so a single-view and a
    multi-view run with the same ``--smoothing_window`` smooth identically.
    """

    _PER_VIEW_KEYS = {"fov_per_view", "cam_rot_per_view", "cam_trans_per_view"}
    _METADATA_KEYS = {"num_views", "view_mask", "camera_indices"}

    def __init__(self, window_size: int):
        self.window_size = window_size
        self._buffer: List[Dict[str, Any]] = []

    def __call__(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add *params* to the buffer and return the smoothed result."""
        if self.window_size <= 0:
            return params

        self._buffer.append(params)
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

        if len(self._buffer) == 1:
            return params

        smoothed: Dict[str, Any] = {}
        for key in params:
            if key in self._METADATA_KEYS:
                smoothed[key] = params[key]
            elif key in self._PER_VIEW_KEYS:
                num_views = len(params[key])
                smoothed_list = []
                for v in range(num_views):
                    tensors = [buf[key][v] for buf in self._buffer if key in buf and v < len(buf[key])]
                    if tensors:
                        smoothed_list.append(torch.stack(tensors).mean(dim=0))
                    else:
                        smoothed_list.append(params[key][v])
                smoothed[key] = smoothed_list
            elif isinstance(params[key], torch.Tensor):
                tensors = [buf[key] for buf in self._buffer if key in buf]
                smoothed[key] = torch.stack(tensors).mean(dim=0)
            else:
                smoothed[key] = params[key]

        return smoothed


def params_to_cpu(params: Dict[str, Any]) -> Dict[str, Any]:
    """Detach and move all tensors in a ``predicted_params`` dict to CPU."""
    out: Dict[str, Any] = {}
    for key, val in params.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.detach().cpu()
        elif isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
            out[key] = [t.detach().cpu() for t in val]
        else:
            out[key] = val
    return out


def params_to_device(params: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Move all tensors in a ``predicted_params`` dict to *device*."""
    out: Dict[str, Any] = {}
    for key, val in params.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.to(device)
        elif isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
            out[key] = [t.to(device) for t in val]
        else:
            out[key] = val
    return out


class InMemoryImageExporter:
    """Image exporter that keeps the last collage in memory instead of on disk."""

    def __init__(self):
        self.image = None

    def export(self, collage_np, batch_id, global_id, img_parameters, vertices, faces, img_idx=0, epoch=None):
        self.image = collage_np


# --------------------------------------------------------------------------- #
# Frame / clip bookkeeping
# --------------------------------------------------------------------------- #


def pad_or_resize(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Fit *frame* into ``(target_w, target_h)`` by resizing down or padding up."""
    import cv2

    target_w, target_h = target_size
    if frame.shape[1] == target_w and frame.shape[0] == target_h:
        return frame
    if frame.shape[1] > target_w or frame.shape[0] > target_h:
        return cv2.resize(frame, (target_w, target_h))
    padded = np.ones((target_h, target_w, 3), dtype=np.uint8) * 40
    h = min(target_h, frame.shape[0])
    w = min(target_w, frame.shape[1])
    padded[:h, :w] = frame[:h, :w]
    return padded


def compute_subclip_ranges(
    dataset_size: int,
    max_frames: Optional[int],
    num_subclips: int,
    rank: int = 0,
) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` index ranges (end exclusive) for each subclip.

    With ``num_subclips > 1`` the dataset is divided into evenly-spaced slots of
    size ``dataset_size // num_subclips``; each subclip starts at slot ``i`` and
    runs for ``max_frames`` frames. Falls back to a single full-dataset clip when
    subclips can't fit (``max_frames`` not set, or per-slot space smaller than
    ``max_frames``).
    """
    if num_subclips <= 1:
        end = min(max_frames, dataset_size) if max_frames is not None else dataset_size
        return [(0, end)]

    if max_frames is None:
        if rank == 0:
            print(
                f"WARNING: --generate_num_subclips={num_subclips} requires --max_frames; "
                f"falling back to a single full-dataset clip."
            )
        return [(0, dataset_size)]

    slot_size = dataset_size // num_subclips
    if slot_size < max_frames:
        if rank == 0:
            print(
                f"WARNING: dataset has {dataset_size} frames; {num_subclips} subclips of "
                f"{max_frames} frames each don't fit (only {slot_size} frames per slot). "
                f"Falling back to a single full-dataset clip."
            )
        return [(0, dataset_size)]

    ranges: List[Tuple[int, int]] = []
    for i in range(num_subclips):
        start = i * dataset_size // num_subclips
        end = min(start + max_frames, dataset_size)
        ranges.append((start, end))
    return ranges


# --------------------------------------------------------------------------- #
# Checkpoint -> dataset/render convention resolution
# --------------------------------------------------------------------------- #


def resolve_frame_convention(ckpt_config: Optional[Dict[str, Any]], state_dict: Optional[Dict[str, Any]] = None) -> Dict:
    """Resolve the frame-convention / camera / mesh-scale flags from a checkpoint.

    These live at the TOP level of ``checkpoint["config"]`` (the trainers persist
    them there, not inside ``model_config``). This mirrors
    ``benchmark_model._create_singleview_model`` exactly so inference, benchmark
    and training agree on what a checkpoint means.

    Returns a dict with ``frame_convention``, ``camera_centric``,
    ``from_multiview``, ``fixed_camera``, ``use_ue_scaling``,
    ``allow_mesh_scaling`` and ``mesh_scale_init``.
    """
    ckpt_config = ckpt_config or {}
    state_dict = state_dict or {}

    frame_convention = ckpt_config.get("frame_convention", "model_centric")
    camera_centric = frame_convention == "camera_centric"
    from_multiview = bool(ckpt_config.get("from_multiview", False))
    fixed_camera = bool(ckpt_config.get("fixed_camera", camera_centric))
    use_ue_scaling = bool(ckpt_config.get("use_ue_scaling", not fixed_camera))

    # Mesh-scale: prefer the persisted flag; fall back to detecting the
    # mesh_scale head in the state dict so older checkpoints (saved before the
    # flag was persisted) still rebuild with the head instead of dropping it
    # (a dropped head renders the mesh at native size, ~35x too large).
    has_mesh_scale_head = any("mesh_scale_head" in k for k in state_dict)
    allow_mesh_scaling = bool(ckpt_config.get("allow_mesh_scaling", has_mesh_scale_head))
    mesh_scale_init = float(ckpt_config.get("init_mesh_scale", 1.0))

    return {
        "frame_convention": frame_convention,
        "camera_centric": camera_centric,
        "from_multiview": from_multiview,
        "fixed_camera": fixed_camera,
        "use_ue_scaling": use_ue_scaling,
        "allow_mesh_scaling": allow_mesh_scaling,
        "mesh_scale_init": mesh_scale_init,
    }


def hdf5_is_multiview(dataset_path: str) -> bool:
    """True when *dataset_path* is a multi-view HDF5 (``/metadata.is_multiview``)."""
    import h5py

    try:
        with h5py.File(str(dataset_path), "r") as f:
            if "metadata" not in f:
                return False
            return bool(f["metadata"].attrs.get("is_multiview", False))
    except OSError:
        return False


def resolve_singleview_dataset_kwargs(dataset_path: str, conventions: Dict[str, Any]) -> Dict[str, Any]:
    """Extra ``UnifiedSMILDataset.from_path`` kwargs for single-view consumption.

    A multi-view HDF5 must be opened in single-view mode for a single-view
    checkpoint, otherwise ``__getitem__`` yields a list of per-view images the
    single-view regressor cannot consume. ``camera_centric`` follows the
    checkpoint's frame convention, so the camera intrinsics/extrinsics the
    dataset hands back match the frame the model was trained in:

    * ``camera_centric``: the sampled view IS the world origin, so the dataset
      returns the PyTorch3D identity camera plus that view's calibrated vertical
      FOV, and re-expresses the 3D keypoints / root pose in that frame.
    * ``model_centric``: the dataset returns the view's real ``(R, t)`` and FOV
      in the shared world frame.

    ``expand_all_views`` makes every valid view of every sample its own item, so
    a run covers all cameras rather than only the preferred one.

    A genuine single-view HDF5 (``SLEAPDataset`` / ``OptimizedSMILDataset``)
    needs none of these — it returns ``{}``.
    """
    if not hdf5_is_multiview(dataset_path):
        return {}
    return {
        "return_single_view": True,
        "camera_centric": bool(conventions.get("camera_centric", False)),
        "expand_all_views": True,
    }


# --------------------------------------------------------------------------- #
# Prediction -> renderer parameter application
# --------------------------------------------------------------------------- #


def apply_shape_space_params(
    temp_fitter,
    model,
    predicted_params: Dict[str, Any],
    index: int = 0,
    disable_scaling: bool = False,
    disable_translation: bool = False,
) -> None:
    """Write the predicted shape-space variation onto *temp_fitter*.

    The network's ``log_beta_scales`` / ``betas_trans`` outputs are NOT always
    per-joint values — their meaning depends on ``scale_trans_mode``:

    * ``ignore`` — the heads do not exist; nothing to apply.
    * ``separate`` + ``use_pca_transformation`` (the default) — the heads emit
      ``(B, N_BETAS)`` PCA weights that must be expanded through
      ``_transform_separate_pca_weights_to_joint_values`` into ``(B, J, 3)``
      before the SMAL model can use them. Assigning the raw ``(B, N_BETAS)``
      weights straight onto ``temp_fitter.log_beta_scales`` (what the single-view
      inference script used to do) silently reshapes the parameter and produces
      a mesh whose limb scaling has nothing to do with the prediction.
    * ``separate`` without PCA, and ``entangled_with_betas`` — already
      ``(B, J, 3)`` per-joint values, applied directly.

    Scaling and translation are gated independently so ``--disable_scaling`` /
    ``--disable_translation`` behave the same in both entrypoints.
    """
    if "log_beta_scales" not in predicted_params or "betas_trans" not in predicted_params:
        return

    mode = getattr(model, "scale_trans_mode", "separate")
    if mode == "ignore":
        return

    sl = slice(index, index + 1)
    scales = predicted_params["log_beta_scales"][sl].detach()
    trans = predicted_params["betas_trans"][sl].detach()

    if mode == "separate":
        from smal_fitter.neuralSMIL.training_config import TrainingConfig

        scale_trans_config = TrainingConfig.get_scale_trans_config()
        use_pca = scale_trans_config.get("separate", {}).get("use_pca_transformation", True)
        # dim() == 2 is the PCA-weight layout (B, N_BETAS); per-joint values are
        # (B, J, 3) and must not be pushed through the PCA expansion again.
        if use_pca and scales.dim() == 2:
            try:
                scales, trans = model._transform_separate_pca_weights_to_joint_values(scales, trans)
            except Exception as e:  # noqa: BLE001 - visualization must not kill inference
                print(f"Warning: failed to expand PCA limb scales for visualization: {e}")
                return

    device = temp_fitter.device
    if not disable_scaling:
        temp_fitter.log_beta_scales.data = scales.to(device)
    if not disable_translation:
        temp_fitter.betas_trans.data = trans.to(device)


def resolve_render_camera(
    model,
    predicted_params: Dict[str, Any],
    y_data: Optional[Dict[str, Any]],
    device: str,
    index: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[float]]:
    """Return ``(R, T, fov, aspect_ratio)`` for the render camera of one sample.

    This is where the model-centric / camera-centric split lives:

    * ``fixed_camera`` (camera-centric checkpoints): the sampled camera IS the
      world origin, so the render camera is the PyTorch3D identity and the
      vertical FOV comes from the dataset's per-sample calibration
      (``y_data['cam_fov']``). The model's camera heads are unsupervised in this
      convention and must NOT be used — mirroring the fixed-camera override in
      ``SMILImageRegressor.predict_from_batch`` and the training visualization.
    * otherwise (model-centric): the predicted ``cam_rot`` / ``cam_trans`` /
      ``fov`` are used, which is what the reprojection loss was evaluated
      through during training.

    ``aspect_ratio`` comes from the dataset's intrinsics when available
    (``cam_aspect``); non-square sensors / ``fx != fy`` calibrations need it or
    the projection is stretched. ``None`` means square (1.0).
    """
    aspect: Optional[float] = None
    if y_data is not None:
        raw_aspect = y_data.get("cam_aspect", None)
        if raw_aspect is not None:
            try:
                aspect = float(np.asarray(raw_aspect).reshape(-1)[0])
            except (TypeError, ValueError):
                aspect = None

    if getattr(model, "fixed_camera", False):
        fov_val: Optional[float] = None
        if y_data is not None and y_data.get("cam_fov", None) is not None:
            try:
                fov_val = float(np.asarray(y_data["cam_fov"]).reshape(-1)[0])
            except (TypeError, ValueError):
                fov_val = None
        if fov_val is None:
            # predict_from_batch injects the GT FOV into predicted_params when the
            # dataset supplies it; falling back to it keeps a raw-image run (which
            # has no calibration) on the resolved --fov instead of crashing.
            fov_val = float(np.asarray(predicted_params["fov"].detach().cpu()).reshape(-1)[0])
        fov = torch.tensor([fov_val], dtype=torch.float32, device=device)
        cam_rot = torch.eye(3, device=device).unsqueeze(0)
        cam_trans = torch.zeros(1, 3, device=device)
        return cam_rot, cam_trans, fov, aspect

    sl = slice(index, index + 1)
    fov = predicted_params["fov"][sl].detach().to(device).reshape(-1)[:1]
    cam_rot = predicted_params["cam_rot"][sl].detach().to(device)
    cam_trans = predicted_params["cam_trans"][sl].detach().to(device)
    return cam_rot, cam_trans, fov, aspect


def place_mesh(
    use_ue_scaling: bool,
    verts: torch.Tensor,
    joints: torch.Tensor,
    trans: torch.Tensor,
    mesh_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the checkpoint's mesh-placement convention to ``(verts, joints)``.

    Mirrors ``SMALFitter.generate_visualization`` exactly, including its
    precedence:

    1. ``use_ue_scaling`` — legacy replicAnt placement: recentre on the root
       joint, scale 10x (the Unreal Engine model scale), then translate.
    2. otherwise a predicted per-sample ``mesh_scale`` — recentre, scale,
       translate. Without this branch a camera-centric checkpoint renders the
       mesh at native size (~35x too large against the metric 3D).
    3. otherwise translation only.

    The two scaling branches are mutually exclusive by construction:
    ``fixed_camera`` (camera-centric) requires ``use_ue_scaling=False``.

    Args:
        use_ue_scaling: the model's ``use_ue_scaling`` flag.
        verts: ``(B, V, 3)`` vertices straight out of the SMAL model.
        joints: ``(B, J, 3)`` joints straight out of the SMAL model.
        trans: ``(B, 3)`` root translation.
        mesh_scale: optional ``(B, 1)`` (or scalar / ``(B,)``) predicted scale.
    """
    root = joints[:, 0:1, :]
    trans_b = trans.unsqueeze(1)

    if use_ue_scaling:
        return (verts - root) * 10 + trans_b, (joints - root) * 10 + trans_b

    if mesh_scale is not None:
        scale_val = mesh_scale
        if not isinstance(scale_val, torch.Tensor):
            scale_val = torch.tensor([[float(scale_val)]], dtype=torch.float32, device=verts.device)
        scale_val = scale_val.to(device=verts.device, dtype=torch.float32)
        if scale_val.dim() == 0:
            scale_val = scale_val.unsqueeze(0).unsqueeze(0)
        elif scale_val.dim() == 1:
            scale_val = scale_val.unsqueeze(1)
        scale_val = scale_val.unsqueeze(-1)
        return (verts - root) * scale_val + trans_b, (joints - root) * scale_val + trans_b

    return verts + trans_b, joints + trans_b


def resolve_mesh_scale(model, predicted_params: Dict[str, Any], index: int = 0) -> Optional[torch.Tensor]:
    """Per-sample mesh scale for the renderer, or ``None``.

    Only meaningful for checkpoints trained with ``allow_mesh_scaling``. Note
    that ``SMALFitter.generate_visualization`` ignores ``mesh_scale`` when
    ``apply_UE_transform`` is set — the legacy 10x UE placement and the learned
    mesh scale are mutually exclusive by construction (``fixed_camera`` requires
    ``use_ue_scaling=False``).
    """
    if not getattr(model, "allow_mesh_scaling", False):
        return None
    if "mesh_scale" not in predicted_params:
        return None
    return predicted_params["mesh_scale"][index : index + 1].detach()


def apply_pose_and_shape(
    temp_fitter,
    model,
    predicted_params: Dict[str, Any],
    index: int = 0,
    disable_scaling: bool = False,
    disable_translation: bool = False,
) -> None:
    """Write rotations, betas and translation onto *temp_fitter*, then the shape space.

    Rotations are converted from the model's ``rotation_representation`` to the
    axis-angle the SMAL model expects.
    """
    from smal_fitter.neuralSMIL.smil_image_regressor import rotation_6d_to_axis_angle

    sl = slice(index, index + 1)
    if model.rotation_representation == "6d":
        global_rot_aa = rotation_6d_to_axis_angle(predicted_params["global_rot"][sl].detach())
        joint_rot_aa = rotation_6d_to_axis_angle(predicted_params["joint_rot"][sl].detach())
    else:
        global_rot_aa = predicted_params["global_rot"][sl].detach()
        joint_rot_aa = predicted_params["joint_rot"][sl].detach()

    device = temp_fitter.device
    temp_fitter.global_rotation.data = global_rot_aa.to(device)
    temp_fitter.joint_rotations.data = joint_rot_aa.to(device)
    temp_fitter.betas.data = predicted_params["betas"][index].detach().to(device)
    temp_fitter.trans.data = predicted_params["trans"][sl].detach().to(device)

    apply_shape_space_params(
        temp_fitter,
        model,
        predicted_params,
        index=index,
        disable_scaling=disable_scaling,
        disable_translation=disable_translation,
    )


# --------------------------------------------------------------------------- #
# Video helpers
# --------------------------------------------------------------------------- #


def write_video(frames: List[np.ndarray], out_path: Path, fps: float, size: Tuple[int, int], fourcc: str = "mp4v"):
    """Write BGR *frames* to *out_path*. Returns the number of frames written."""
    import cv2

    if not frames:
        return 0
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_path} at {size[0]}x{size[1]}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return len(frames)
