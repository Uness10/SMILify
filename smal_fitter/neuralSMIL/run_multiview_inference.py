#!/usr/bin/env python3
"""
Simple multi-view inference + visualization using a preprocessed SLEAP dataset.

This script mirrors the visualization logic used during training in
`train_multiview_regressor.py`, but runs over all samples in a preprocessed
multi-view HDF5 dataset and writes two videos in the current working directory:

  - "<DATASET>_multiview_inference.avi" (multi-view grid visualization, AVI+MJPG
    so wide grids exceed MPEG-4's 8192px width cap)
  - "<DATASET>_smultiview_first_camera_render.mp4" (single-view render for view 0)
"""

# ===== CRITICAL: Force IPv4 BEFORE any other imports =====
# This prevents "Address family not supported by protocol" (errno: 97) errors
# on HPC systems that don't have full IPv6 support. The patch itself lives in
# inference_ddp so the single-view entrypoint applies exactly the same one.
from smal_fitter.neuralSMIL.inference_ddp import force_ipv4_getaddrinfo

force_ipv4_getaddrinfo()
# ===== End IPv4 forcing =====

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Tuple


import numpy as np
import cv2
import torch
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp

from smal_fitter.neuralSMIL.multiview_smil_regressor import create_multiview_regressor, MultiViewSMILImageRegressor
from smal_fitter.fitter import SMALFitter
from smal_fitter.sleap_data.sleap_multiview_dataset import SLEAPMultiViewDataset
from smal_fitter.neuralSMIL.configs import apply_smal_file_override
import config
from smal_fitter.neuralSMIL.animation_export import build_recorder_from_config, build_multiview_cameras
from smal_fitter.neuralSMIL.multiview_visualization import (
    compute_multiview_grid_layout,
    create_multiview_visualization,
)
from smal_fitter.neuralSMIL.inference_common import (
    InMemoryImageExporter,
    PredictionSmoother,
    apply_pose_and_shape,
    compute_subclip_ranges,
    pad_or_resize,
    params_to_cpu,
    params_to_device,
    resolve_mesh_scale,
)
from smal_fitter.neuralSMIL.inference_ddp import (
    cleanup_ddp,
    cleanup_temp_dir,
    compute_rank_indices,
    gather_predictions,
    is_torchrun_launched,
    merge_frame_streams_from_temp,
    resolve_launch,
    setup_ddp,
    validate_num_gpus,
    write_frame_streams_to_temp,
    write_video_from_manifest,
)


DEFAULT_CHECKPOINTS = [
    "multiview_checkpoints/best_model.pth",
    "multiview_checkpoints/final_model.pth",
]


def run_forward_multiview(
    model: MultiViewSMILImageRegressor, x_data: dict, y_data: dict, device: str
) -> Optional[dict]:
    """Run a single forward pass and return predicted_params (or None if no views)."""
    images = x_data.get("images", [])
    num_views = len(images)
    if num_views == 0:
        return None

    cam_indices = x_data.get("camera_indices", list(range(num_views)))
    if isinstance(cam_indices, np.ndarray):
        cam_indices = cam_indices.tolist()
    if len(cam_indices) != num_views:
        if len(cam_indices) > num_views:
            cam_indices = cam_indices[:num_views]
        else:
            cam_indices = list(cam_indices) + list(range(len(cam_indices), num_views))

    images_per_view = []
    for img in images:
        img_tensor = model.preprocess_image(img).to(device)
        images_per_view.append(img_tensor.squeeze(0))

    images_tensors = [img.unsqueeze(0) for img in images_per_view]
    camera_indices_tensor = torch.tensor([cam_indices], device=device)
    view_mask = torch.ones(1, num_views, dtype=torch.bool, device=device)

    with torch.no_grad():
        predicted_params = model.forward_multiview(
            images_tensors, camera_indices_tensor, view_mask, target_data=[y_data]
        )

    return predicted_params


def _find_default_checkpoint() -> Path:
    for rel_path in DEFAULT_CHECKPOINTS:
        path = Path(rel_path)
        if path.exists():
            return path
    return Path(DEFAULT_CHECKPOINTS[0])


def load_multiview_model_from_checkpoint(
    checkpoint_path: Path, device: str, max_views: int = None, canonical_camera_order: List[str] = None
) -> MultiViewSMILImageRegressor:
    """
    Load a trained MultiViewSMILImageRegressor model from checkpoint.

    CRITICAL: max_views and canonical_camera_order are inferred from the checkpoint,
    not from the dataset. The model architecture (view_embeddings, camera_heads) is
    determined by max_views used during training. The model can still handle samples
    with fewer views than max_views through the view_mask mechanism.

    Args:
        checkpoint_path: Path to checkpoint file
        device: PyTorch device
        max_views: Optional max_views (if None, inferred from checkpoint)
        canonical_camera_order: Optional canonical camera order (if None, loaded from checkpoint)

    Returns:
        MultiViewSMILImageRegressor model
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    ckpt_config = checkpoint.get("config", {})

    # Get state dict for inferring model structure
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    backbone_name = ckpt_config.get("backbone_name", "vit_large_patch16_224")
    head_type = ckpt_config.get("head_type", "transformer_decoder")
    hidden_dim = ckpt_config.get("hidden_dim", 512)
    rotation_representation = ckpt_config.get("rotation_representation", "6d")
    scale_trans_mode = ckpt_config.get("scale_trans_mode", "separate")
    freeze_backbone = ckpt_config.get("freeze_backbone", True)
    use_unity_prior = ckpt_config.get("use_unity_prior", False)
    use_ue_scaling = ckpt_config.get("use_ue_scaling", False)
    allow_mesh_scaling = ckpt_config.get("allow_mesh_scaling", False)
    mesh_scale_init = ckpt_config.get("mesh_scale_init", 1.0)

    cross_attention_layers = ckpt_config.get("cross_attention_layers", 2)
    cross_attention_heads = ckpt_config.get("cross_attention_heads", 8)
    cross_attention_dropout = ckpt_config.get("cross_attention_dropout", 0.1)
    transformer_config = ckpt_config.get("transformer_config", {})
    use_gt_camera_init = ckpt_config.get("use_gt_camera_init", False)

    # CRITICAL: Infer max_views from checkpoint state dict to ensure model architecture matches
    # The view_embeddings.weight shape determines the number of camera positions
    if max_views is None:
        if "view_embeddings.weight" in state_dict:
            max_views = state_dict["view_embeddings.weight"].shape[0]
            print(f"Inferred max_views={max_views} from checkpoint view_embeddings.weight shape")
        else:
            # Fall back to config or default
            max_views = ckpt_config.get("max_views", 4)
            print(f"Using max_views={max_views} from checkpoint config or default")

    # Get canonical_camera_order from checkpoint if not provided
    if canonical_camera_order is None:
        canonical_camera_order = ckpt_config.get("canonical_camera_order", None)
        if canonical_camera_order is None:
            # Create placeholder list - indices are what matter, not names
            canonical_camera_order = [f"Camera{i}" for i in range(max_views)]
            print(f"Created placeholder canonical camera order (indices 0-{max_views - 1})")
        else:
            print(f"Loaded canonical camera order from checkpoint: {canonical_camera_order}")

    print(
        f"Model architecture: max_views={max_views}, canonical_camera_order has {len(canonical_camera_order)} cameras"
    )
    print("Note: Model can handle samples with fewer views than max_views via view_mask")

    from smal_fitter.neuralSMIL.backbone_factory import BackboneFactory

    input_resolution = BackboneFactory.get_default_input_resolution(backbone_name)

    model = create_multiview_regressor(
        device=device,
        batch_size=1,
        shape_family=config.SHAPE_FAMILY,
        use_unity_prior=use_unity_prior,
        max_views=max_views,
        canonical_camera_order=canonical_camera_order,
        cross_attention_layers=cross_attention_layers,
        cross_attention_heads=cross_attention_heads,
        cross_attention_dropout=cross_attention_dropout,
        backbone_name=backbone_name,
        freeze_backbone=freeze_backbone,
        head_type=head_type,
        hidden_dim=hidden_dim,
        rotation_representation=rotation_representation,
        scale_trans_mode=scale_trans_mode,
        use_ue_scaling=use_ue_scaling,
        input_resolution=input_resolution,
        transformer_config=transformer_config,
        allow_mesh_scaling=allow_mesh_scaling,
        mesh_scale_init=mesh_scale_init,
        use_gt_camera_init=use_gt_camera_init,
    ).to(device)

    if use_gt_camera_init:
        print(
            "  Note: Model trained with GT camera initialization - will use GT camera params as base for delta predictions"
        )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    smal_optimization_params = [
        "global_rotation",
        "joint_rotations",
        "trans",
        "log_beta_scales",
        "betas_trans",
        "betas",
        "fov",
        "target_joints",
        "target_visibility",
    ]
    nn_state_dict = {
        k: v
        for k, v in state_dict.items()
        if not any(k == param or k.startswith(param + ".") for param in smal_optimization_params)
    }
    model.load_state_dict(nn_state_dict, strict=False)
    model.eval()
    return model


def render_singleview_collage(
    model: MultiViewSMILImageRegressor,
    x_data: dict,
    y_data: dict,
    device: str,
    view_idx: int = 0,
    disable_scaling: bool = False,
    disable_translation: bool = False,
    predicted_params: Optional[dict] = None,
    render_resolution: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Render single-view mesh visualization matching training visualization exactly.

    CRITICAL: This function must match the training visualization in train_multiview_regressor.py
    exactly, especially for PCA transformation and parameter application order.

    Args:
        model: MultiViewSMILImageRegressor model
        x_data: Input data dictionary
        y_data: Target data dictionary
        device: PyTorch device
        view_idx: Which view to render (must be < num_views for this sample)
        disable_scaling: If True, skip applying log_beta_scales (testing/debugging)
        disable_translation: If True, skip applying betas_trans (testing/debugging)
        predicted_params: Pre-computed predicted parameters. If None, runs forward pass internally.
        render_resolution: If set, render the mesh + composite the footage at this square
            pixel resolution instead of the model's native ``renderer.image_size`` (224).
            Footage is interpolated up via PIL bilinear when this exceeds the native footage
            resolution (512). Does not affect model inference.
    """
    images = x_data.get("images", [])
    num_views = len(images)
    if num_views == 0 or view_idx >= num_views:
        return None

    # Run forward pass if predicted_params not provided
    if predicted_params is None:
        predicted_params = run_forward_multiview(model, x_data, y_data, device)
        if predicted_params is None:
            return None

    fov_per_view = predicted_params.get("fov_per_view", None)
    cam_rot_per_view = predicted_params.get("cam_rot_per_view", None)
    cam_trans_per_view = predicted_params.get("cam_trans_per_view", None)

    target_size = int(render_resolution) if render_resolution else int(getattr(model.renderer, "image_size", 224))

    original_image = images[view_idx]
    from PIL import Image

    pil_img = Image.fromarray((original_image * 255).astype(np.uint8))
    pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
    resized_image = np.array(pil_img).astype(np.float32) / 255.0

    resized_image = np.clip(resized_image, 0.0, 1.0)
    rgb = torch.from_numpy(resized_image).permute(2, 0, 1).unsqueeze(0).float()

    keypoints_2d = y_data.get("keypoints_2d", None)
    visibility = y_data.get("keypoint_visibility", None)

    view_keypoints = None
    view_visibility = None
    if keypoints_2d is not None:
        if len(keypoints_2d.shape) == 3:
            view_keypoints = keypoints_2d[view_idx] if view_idx < keypoints_2d.shape[0] else None
            if visibility is not None and view_idx < visibility.shape[0]:
                view_visibility = visibility[view_idx]
        else:
            view_keypoints = keypoints_2d if view_idx == 0 else None
            view_visibility = visibility if view_idx == 0 else None

    sil = torch.zeros(1, 1, target_size, target_size)
    if view_keypoints is not None and view_visibility is not None:
        pixel_coords = view_keypoints.copy()
        pixel_coords[:, 0] = pixel_coords[:, 0] * target_size
        pixel_coords[:, 1] = pixel_coords[:, 1] * target_size
        num_joints = len(view_keypoints)
        joints = torch.tensor(pixel_coords.reshape(1, num_joints, 2), dtype=torch.float32)
        vis = torch.tensor(view_visibility.reshape(1, num_joints), dtype=torch.float32)
        temp_batch = (rgb, sil, joints, vis)
        rgb_only = False
    else:
        temp_batch = rgb
        rgb_only = True

    temp_fitter = SMALFitter(
        device=device,
        data_batch=temp_batch,
        batch_size=1,
        shape_family=config.SHAPE_FAMILY,
        use_unity_prior=False,
        rgb_only=rgb_only,
    )

    # CRITICAL: Match propagate_scaling to the training model's setting.
    # The model learns scales with propagate_scaling=True (set in SMILImageRegressor.__init__),
    # so visualization must also use propagate_scaling=True for consistent geometry.
    temp_fitter.propagate_scaling = model.propagate_scaling

    if view_keypoints is not None and view_visibility is not None:
        pixel_coords = view_keypoints.copy()
        pixel_coords[:, 0] = pixel_coords[:, 0] * target_size
        pixel_coords[:, 1] = pixel_coords[:, 1] * target_size
        temp_fitter.target_joints = torch.tensor(pixel_coords, dtype=torch.float32, device=device).unsqueeze(0)
        temp_fitter.target_visibility = torch.tensor(view_visibility, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        n_joints = temp_fitter.joint_rotations.shape[1] + 1
        temp_fitter.target_joints = torch.zeros((1, n_joints, 2), device=device)
        temp_fitter.target_visibility = torch.zeros((1, n_joints), device=device)

    # Pose, betas, translation and the shape-space variation (scale_trans_mode
    # aware: PCA weights are expanded to per-joint values before they reach the
    # SMAL model). Shared with run_singleview_inference so both entrypoints
    # interpret the same prediction identically — see inference_common.
    apply_pose_and_shape(
        temp_fitter,
        model,
        predicted_params,
        index=0,
        disable_scaling=disable_scaling,
        disable_translation=disable_translation,
    )

    if fov_per_view is not None and view_idx < len(fov_per_view):
        fov_val = fov_per_view[view_idx][0, 0].detach().to(device)
        temp_fitter.fov.data = fov_val.unsqueeze(0)
    elif "fov" in predicted_params:
        temp_fitter.fov.data = predicted_params["fov"][0:1].detach().to(device)

    if cam_rot_per_view is not None and cam_trans_per_view is not None and view_idx < len(cam_rot_per_view):
        cam_rot = cam_rot_per_view[view_idx][0:1].detach().to(device)
        cam_trans = cam_trans_per_view[view_idx][0:1].detach().to(device)
        if fov_per_view is not None and view_idx < len(fov_per_view):
            view_fov_val = fov_per_view[view_idx][0, 0].detach().to(device)
            view_fov = view_fov_val.unsqueeze(0)
            temp_fitter.fov.data = view_fov
        else:
            view_fov = temp_fitter.fov.data

        aspect = None
        try:
            if y_data.get("cam_aspect_per_view") is not None:
                aspect = float(np.array(y_data["cam_aspect_per_view"][view_idx]).reshape(-1)[0])
        except Exception:
            aspect = None

        temp_fitter.renderer.set_camera_parameters(R=cam_rot, T=cam_trans, fov=view_fov, aspect_ratio=aspect)

    exporter = InMemoryImageExporter()
    vis_mesh_scale = resolve_mesh_scale(model, predicted_params, index=0)
    temp_fitter.generate_visualization(
        exporter,
        apply_UE_transform=model.use_ue_scaling,  # MUST match model setting for consistency with 3D keypoints!
        img_idx=view_idx,
        mesh_scale=vis_mesh_scale,
    )
    return exporter.image


def _export_animation(
    raw_predictions: List[Tuple[int, dict]],
    rank: int,
    world_size: int,
    dataset: SLEAPMultiViewDataset,
    model: MultiViewSMILImageRegressor,
    checkpoint_path: Path,
    dataset_path: Path,
    export_path: str,
    fps: float,
) -> None:
    """Gather predictions to rank 0 and write an AMASS-style .npz + .json clip.

    The export captures the *raw*, pre-smoothing network predictions so downstream
    consumers (Blender addon, etc.) can apply their own smoothing if desired.
    """
    # Temp dir name derives from the export path so it stays unique when several
    # subclips are exported in the same run.
    export_temp_base = Path.cwd() / f".animation_export_temp_{Path(export_path).name}"
    if world_size > 1 and rank == 0:
        cleanup_temp_dir(export_temp_base)

    all_predictions = gather_predictions(
        raw_predictions,
        rank=rank,
        world_size=world_size,
        temp_dir=export_temp_base,
        all_ranks=False,
    )

    if world_size > 1 and rank == 0:
        cleanup_temp_dir(export_temp_base)

    if rank != 0 or not all_predictions:
        return

    recorder = build_recorder_from_config(
        output_path=export_path,
        rotation_representation=getattr(model, "rotation_representation", "6d"),
        fps=fps,
        source_checkpoint=str(checkpoint_path),
        source_input=str(dataset_path),
        model_id=getattr(model, "model_id", None),
    )

    cameras = build_multiview_cameras(all_predictions, dataset.get_canonical_camera_order())
    if cameras:
        recorder.set_cameras(cameras)

    for _, params in all_predictions:
        recorder.record(params)

    written = recorder.write()
    print(f"Animation export written: {written['npz']} + {written['json']} ({recorder.num_frames()} frames)")


def run_inference_phase(
    dataset: SLEAPMultiViewDataset,
    model: MultiViewSMILImageRegressor,
    device: str,
    indices: List[int],
    rank: int,
) -> List[Tuple[int, dict]]:
    """Run forward passes on the assigned indices and return raw predictions on CPU.

    Returns:
        List of ``(global_idx, predicted_params_cpu)`` tuples.
    """
    model.eval()
    raw_predictions: List[Tuple[int, dict]] = []

    iterator = tqdm(indices, desc="Running inference", disable=(rank != 0))
    for global_idx in iterator:
        try:
            x_data, y_data = dataset[global_idx]
            if len(x_data.get("images", [])) == 0:
                continue
            predicted_params = run_forward_multiview(model, x_data, y_data, device)
            if predicted_params is None:
                continue
            raw_predictions.append((global_idx, params_to_cpu(predicted_params)))
        except Exception as e:
            print(f"[Rank {rank}] Error in inference for sample {global_idx}: {e}")
            continue

    print(f"[Rank {rank}] Inference complete: {len(raw_predictions)} predictions")
    return raw_predictions


def run_render_phase(
    dataset: SLEAPMultiViewDataset,
    model: MultiViewSMILImageRegressor,
    device: str,
    smoothed_params: Dict[int, dict],
    indices: List[int],
    rank: int,
    grid_width: int,
    grid_height: int,
    singleview_size: Tuple[int, int],
    disable_scaling: bool = False,
    disable_translation: bool = False,
    view_indices: Optional[List[int]] = None,
    total_view_slots: Optional[int] = None,
    render_resolution: Optional[int] = None,
) -> Tuple[List[np.ndarray], Dict[int, List[np.ndarray]], List[int], Dict[int, List[int]]]:
    """Render visualizations for the assigned indices using pre-computed smoothed params.

    Returns the same tuple as the old ``process_dataset_portion``.
    """
    if view_indices is None:
        view_indices = [0]

    multiview_frames: List[np.ndarray] = []
    mv_frame_indices: List[int] = []
    singleview_frames_per_view: Dict[int, List[np.ndarray]] = {v: [] for v in view_indices}
    sv_frame_indices_per_view: Dict[int, List[int]] = {v: [] for v in view_indices}

    iterator = tqdm(indices, desc="Rendering visualizations", disable=(rank != 0))
    for global_idx in iterator:
        if global_idx not in smoothed_params:
            continue
        try:
            x_data, y_data = dataset[global_idx]
            num_available_views = len(x_data.get("images", []))
            if num_available_views == 0:
                continue

            params = params_to_device(smoothed_params[global_idx], device)

            mv_frame = create_multiview_visualization(
                model,
                x_data,
                y_data,
                device,
                disable_scaling=disable_scaling,
                disable_translation=disable_translation,
                predicted_params=params,
                total_view_slots=total_view_slots,
            )
            if mv_frame is not None:
                mv_frame = pad_or_resize(mv_frame, (grid_width, grid_height))
                multiview_frames.append(cv2.cvtColor(mv_frame, cv2.COLOR_RGB2BGR))
                mv_frame_indices.append(global_idx)

            for view_idx in view_indices:
                if view_idx >= num_available_views:
                    continue
                sv_frame = render_singleview_collage(
                    model,
                    x_data,
                    y_data,
                    device,
                    view_idx=view_idx,
                    disable_scaling=disable_scaling,
                    disable_translation=disable_translation,
                    predicted_params=params,
                    render_resolution=render_resolution,
                )
                if sv_frame is not None:
                    sv_frame = pad_or_resize(sv_frame, singleview_size)
                    singleview_frames_per_view[view_idx].append(cv2.cvtColor(sv_frame, cv2.COLOR_RGB2BGR))
                    sv_frame_indices_per_view[view_idx].append(global_idx)

        except Exception as e:
            print(f"[Rank {rank}] Error rendering sample {global_idx}: {e}")
            continue

    total_sv = sum(len(f) for f in singleview_frames_per_view.values())
    print(f"[Rank {rank}] Rendering complete: {len(multiview_frames)} multiview, {total_sv} singleview frames")
    return multiview_frames, singleview_frames_per_view, mv_frame_indices, sv_frame_indices_per_view


def _sv_stream_name(view_idx: int) -> str:
    """Temp-storage stream name for the single-view render of one camera."""
    return f"sv_view{view_idx}"


def write_frames_to_temp_storage(
    multiview_frames: List[np.ndarray],
    singleview_frames_per_view: Dict[int, List[np.ndarray]],
    mv_frame_indices: List[int],
    sv_frame_indices_per_view: Dict[int, List[int]],
    temp_dir: Path,
    rank: int,
) -> Path:
    """
    Write frames to temporary storage on disk to avoid memory issues with all_gather.

    Thin adapter over ``inference_ddp.write_frame_streams_to_temp``: the
    multi-view grid becomes the ``"mv"`` stream and each rendered camera becomes
    an ``"sv_view<N>"`` stream. The single-view entrypoint uses the same
    machinery with a single ``"sv"`` stream.

    Args:
        multiview_frames: List of multiview grid frames
        singleview_frames_per_view: Dict mapping view_idx -> list of singleview frames
        mv_frame_indices: Original dataset indices for multiview frames
        sv_frame_indices_per_view: Dict mapping view_idx -> list of dataset indices
        temp_dir: Directory to store temporary files
        rank: Current process rank

    Returns:
        Path to rank directory containing the frames and manifest
    """
    streams = {"mv": (multiview_frames, mv_frame_indices)}
    for view_idx, frames in singleview_frames_per_view.items():
        streams[_sv_stream_name(view_idx)] = (frames, sv_frame_indices_per_view[view_idx])
    return write_frame_streams_to_temp(streams, temp_dir, rank)


def merge_frames_and_write_videos(
    temp_dir: Path,
    world_size: int,
    multiview_out: Path,
    singleview_out_base: Path,
    fps: int,
    grid_width: int,
    grid_height: int,
    singleview_size: Tuple[int, int],
    view_indices: List[int],
):
    """
    Merge frames from all ranks and write final videos (called only by rank 0).

    Reads manifests from all ranks, sorts frames by original index, and writes videos.
    Creates separate singleview videos for each view index.
    """
    print(f"\nMerging frames from {world_size} ranks...")

    stream_names = ["mv"] + [_sv_stream_name(v) for v in view_indices]
    merged = merge_frame_streams_from_temp(temp_dir, world_size, stream_names)

    all_mv_entries = merged.get("mv", [])
    print(f"Writing {len(all_mv_entries)} multiview frames...")

    # Write multiview video. MPEG-4 caps frame dimensions at 8192 px, so wide
    # multi-camera grids fall over silently with mp4v. AVI+MJPG has no such
    # cap and stays well-supported everywhere we play these back.
    if all_mv_entries:
        write_video_from_manifest(
            all_mv_entries,
            multiview_out,
            fps,
            (grid_width, grid_height),
            fourcc="MJPG",
        )
        print(f"Wrote {multiview_out}")

    # Write singleview video for each view
    for view_idx in view_indices:
        sv_entries = merged.get(_sv_stream_name(view_idx), [])
        if not sv_entries:
            continue
        if len(view_indices) == 1:
            # Single view: use original naming
            singleview_out = singleview_out_base
        else:
            # Multiple views: add view index to filename
            stem = singleview_out_base.stem
            singleview_out = singleview_out_base.parent / f"{stem}_view{view_idx}.mp4"

        written = write_video_from_manifest(sv_entries, singleview_out, fps, singleview_size, fourcc="mp4v")
        print(f"Wrote {singleview_out} ({written} frames)")


def main_inference(
    args,
    rank: int = 0,
    world_size: int = 1,
    device_override: Optional[str] = None,
):
    """
    Main inference function that can run in single-GPU or multi-GPU mode.

    Args:
        args: Parsed command line arguments
        rank: Process rank (0 for single-GPU)
        world_size: Total number of processes (1 for single-GPU)
        device_override: Optional device string override
    """
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Set device
    if device_override:
        device = device_override
    else:
        if world_size > 1:
            # Multi-GPU: use rank-specific device
            device = f"cuda:{rank}"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    # Apply SMAL model override if provided (similar to training script)
    # This must be done before loading the dataset/model to ensure config.dd, config.N_POSE, etc. are correct
    if args.smal_file:
        if rank == 0:
            print(f"Applying SMAL file override: {args.smal_file}")
        shape_family = args.shape_family if args.shape_family is not None else config.SHAPE_FAMILY
        apply_smal_file_override(args.smal_file, shape_family=shape_family)
        if rank == 0:
            print(f"  Shape family: {config.SHAPE_FAMILY}")
            print(f"  N_POSE: {config.N_POSE}")
            print(f"  N_BETAS: {config.N_BETAS}")

    # Parse view indices from comma-separated string
    view_indices = [int(x.strip()) for x in args.view_indices.split(",")]

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("MULTI-VIEW INFERENCE")
        print(f"{'=' * 60}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Dataset: {dataset_path}")
        print(f"Checkpoint: {args.checkpoint if args.checkpoint else '(auto-detect)'}")
        if args.max_frames is not None:
            print(f"Max frames: {args.max_frames} (testing mode)")
        if args.generate_num_subclips > 1:
            print(f"Subclips: {args.generate_num_subclips} (per-clip length: {args.max_frames})")
        if args.disable_scaling:
            print("Part scaling: DISABLED (comparison mode)")
        if args.disable_translation:
            print("Part translation: DISABLED (comparison mode)")
        print(f"View indices for singleview: {view_indices}")
        if args.smoothing_window > 0:
            print(f"Temporal smoothing: {args.smoothing_window} frames")
        print(f"{'=' * 60}\n")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else _find_default_checkpoint()

    # Load dataset
    # Use num_views_to_use=None to use all available views per sample (same as training)
    # This allows the dataset to have variable numbers of views per sample
    dataset = SLEAPMultiViewDataset(
        hdf5_path=str(dataset_path),
        rotation_representation="6d",
        num_views_to_use=None,  # Use all available views (handles variable view counts)
        random_view_sampling=True,
    )

    dataset_max_views = dataset.get_max_views_in_dataset()
    dataset_canonical_camera_order = dataset.get_canonical_camera_order()

    if rank == 0:
        print(f"Dataset size: {len(dataset)}")
        print(f"Max views in dataset: {dataset_max_views}")
        print(f"Dataset canonical camera order: {dataset_canonical_camera_order}")
        print("Note: Samples may have fewer views than dataset max_views\n")

    # Load model
    # CRITICAL: max_views and canonical_camera_order are inferred from the checkpoint,
    # not from the dataset. The model architecture must match what was used during training.
    # The model can still handle samples with fewer views than max_views via view_mask.
    model = load_multiview_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        max_views=None,  # Infer from checkpoint
        canonical_camera_order=None,  # Load from checkpoint
    )

    # Get model's max_views (from checkpoint architecture)
    model_max_views = model.max_views

    if rank == 0:
        print(f"Model max_views (from checkpoint): {model_max_views}")
        print(f"Dataset max_views: {dataset_max_views}")
        if model_max_views > dataset_max_views:
            print(f"Note: Model supports {model_max_views} views, dataset has up to {dataset_max_views} views")
            print("      Model will handle samples with fewer views via view_mask\n")
        elif model_max_views < dataset_max_views:
            print(f"WARNING: Model supports {model_max_views} views but dataset has up to {dataset_max_views} views")
            print(f"         Samples with >{model_max_views} views will be truncated\n")

    # Optional higher-resolution single-view visualization. When set, the single-view
    # mesh collage is rendered (and footage interpolated up) at this square resolution.
    # The multi-view grid is left at the native renderer resolution on purpose.
    render_resolution = getattr(args, "render_resolution", None)
    if render_resolution is not None:
        if render_resolution <= 0:
            raise ValueError(f"--render_resolution must be a positive integer, got {render_resolution}")
        if rank == 0:
            native_footage = 512  # decoded JPEG crops stored in the HDF5 are 512x512
            if render_resolution < native_footage:
                print(
                    f"WARNING: --render_resolution={render_resolution} is below the native "
                    f"footage resolution ({native_footage}); rendering below native footage "
                    f"wastes available detail."
                )
            if render_resolution > 4096:
                print(
                    f"WARNING: --render_resolution={render_resolution} is very large; "
                    f"render time and single-view file size grow ~quadratically and "
                    f"memory use may be high."
                )
            print(
                f"Single-view render resolution: {render_resolution} "
                f"(native renderer image_size={int(model.renderer.image_size)})"
            )

    # Calculate frame dimensions — derive from the model's renderer resolution.
    # Layout (single-row vs wrapped 6-per-row block layout for >12 views) is
    # decided once from model.max_views so every frame in the output video has
    # identical dimensions, even when individual samples have fewer views.
    # NOTE: the multi-view grid intentionally stays at the native renderer
    # resolution; only the single-view collage honors --render_resolution.
    img_size = int(model.renderer.image_size)
    mv_layout = compute_multiview_grid_layout(model_max_views, img_size)
    grid_width = mv_layout["grid_width"]
    grid_height = mv_layout["grid_height"]
    if rank == 0:
        print(
            f"Multiview grid layout: {mv_layout['num_blocks']} block(s) × "
            f"{mv_layout['cols']} col(s) → {grid_width}×{grid_height}"
        )

    # Determine singleview frame size (use first sample to get actual size).
    # Fallback default reflects the effective single-view resolution so a failed
    # test render still yields sane dimensions.
    sv_img_size = render_resolution if render_resolution else img_size
    singleview_size = (sv_img_size, sv_img_size)  # Default, overridden below by test render
    if len(dataset) > 0:
        try:
            test_frame = render_singleview_collage(
                model,
                dataset[0][0],
                dataset[0][1],
                device,
                view_idx=0,
                render_resolution=render_resolution,
            )
            if test_frame is not None:
                singleview_size = (test_frame.shape[1], test_frame.shape[0])
        except Exception:
            pass  # Use default

    # Determine subclip ranges (single full-dataset clip by default).
    subclip_ranges = compute_subclip_ranges(
        dataset_size=len(dataset),
        max_frames=args.max_frames,
        num_subclips=args.generate_num_subclips,
        rank=rank,
    )
    multi_subclip = len(subclip_ranges) > 1

    dataset_name = dataset_path.stem
    smoothing_window = args.smoothing_window
    export_animation_path = getattr(args, "export_animation", None)

    for clip_idx, (start_idx, end_idx) in enumerate(subclip_ranges):
        if multi_subclip and rank == 0:
            print(f"\n{'#' * 60}")
            print(
                f"# SUBCLIP {clip_idx + 1}/{len(subclip_ranges)}: "
                f"frames [{start_idx}, {end_idx}) ({end_idx - start_idx} frames)"
            )
            print(f"{'#' * 60}")

        range_suffix = f"_frames{start_idx:06d}-{end_idx:06d}" if multi_subclip else ""

        # Compute which dataset indices this rank is responsible for in this subclip
        assigned_indices = compute_rank_indices(
            len(dataset),
            rank,
            world_size,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        # ── Phase 1: Inference (all ranks in parallel) ──────────────────────
        if rank == 0:
            print("\n── Phase 1: Running inference ──")
        raw_predictions = run_inference_phase(dataset, model, device, assigned_indices, rank)

        # Free GPU memory after inference — rendering will reload params to device as needed
        torch.cuda.empty_cache()

        # ── Phase 1b: Optional animation export (raw, pre-smoothing) ────────
        if export_animation_path:
            clip_export_path = f"{export_animation_path}{range_suffix}"
            _export_animation(
                raw_predictions=raw_predictions,
                rank=rank,
                world_size=world_size,
                dataset=dataset,
                model=model,
                checkpoint_path=checkpoint_path,
                dataset_path=dataset_path,
                export_path=clip_export_path,
                fps=float(args.fps),
            )

        # ── Phase 2: Gather + smooth predictions ────────────────────────────
        temp_base = Path.cwd() / f".inference_temp_{dataset_name}{range_suffix}"

        if world_size > 1 and smoothing_window > 0:
            # Multi-GPU with smoothing: gather all predictions so every rank
            # can build the full temporally-ordered sequence for correct smoothing.
            # (Smoothing over a rank's striped subset would average frames that
            # are world_size apart in time, not adjacent ones.)
            if rank == 0:
                print(
                    f"\n── Phase 2: Gathering predictions across {world_size} ranks for smoothing (window={smoothing_window}) ──"
                )
                cleanup_temp_dir(temp_base)

            # cleanup=True removes the pickles but keeps temp_base itself, which
            # is reused for frame storage in Phase 4.
            all_predictions = gather_predictions(
                raw_predictions,
                rank=rank,
                world_size=world_size,
                temp_dir=temp_base,
                all_ranks=True,
            )

            # Apply smoothing over the full sorted sequence
            smoother = PredictionSmoother(smoothing_window)
            smoothed_params: Dict[int, dict] = {}
            iterator = tqdm(all_predictions, desc="Applying temporal smoothing", disable=(rank != 0))
            for global_idx, params in iterator:
                smoothed_params[global_idx] = smoother(params)

            del raw_predictions, all_predictions

        elif smoothing_window > 0:
            # Single GPU with smoothing
            if rank == 0:
                print(f"\n── Phase 2: Applying temporal smoothing (window={smoothing_window}) ──")
            raw_predictions.sort(key=lambda x: x[0])
            smoother = PredictionSmoother(smoothing_window)
            smoothed_params = {}
            for global_idx, params in tqdm(raw_predictions, desc="Applying temporal smoothing", disable=(rank != 0)):
                smoothed_params[global_idx] = smoother(params)
            del raw_predictions

        else:
            # No smoothing: use raw predictions directly
            if rank == 0:
                print("\n── Phase 2: No smoothing (window=0) ──")
            smoothed_params = {idx: params for idx, params in raw_predictions}
            del raw_predictions

        # ── Phase 3: Render visualizations (all ranks in parallel) ──────────
        if rank == 0:
            print("\n── Phase 3: Rendering visualizations ──")
        multiview_frames, singleview_frames_per_view, mv_frame_indices, sv_frame_indices_per_view = run_render_phase(
            dataset=dataset,
            model=model,
            device=device,
            smoothed_params=smoothed_params,
            indices=assigned_indices,
            rank=rank,
            grid_width=grid_width,
            grid_height=grid_height,
            singleview_size=singleview_size,
            disable_scaling=args.disable_scaling,
            disable_translation=args.disable_translation,
            view_indices=view_indices,
            total_view_slots=model_max_views,
            render_resolution=render_resolution,
        )
        del smoothed_params

        # ── Phase 4: Write output videos ────────────────────────────────────
        if rank == 0:
            print("\n── Phase 4: Writing output videos ──")
        multiview_out = Path(f"{dataset_name}{range_suffix}_multiview_inference.avi")
        singleview_out_base = Path(f"{dataset_name}{range_suffix}_singleview_inference.mp4")

        if world_size > 1:
            if rank == 0:
                if not temp_base.exists():
                    temp_base.mkdir(parents=True, exist_ok=True)
            dist.barrier()

            write_frames_to_temp_storage(
                multiview_frames=multiview_frames,
                singleview_frames_per_view=singleview_frames_per_view,
                mv_frame_indices=mv_frame_indices,
                sv_frame_indices_per_view=sv_frame_indices_per_view,
                temp_dir=temp_base,
                rank=rank,
            )
            del multiview_frames, singleview_frames_per_view, mv_frame_indices, sv_frame_indices_per_view
            torch.cuda.empty_cache()

            dist.barrier()

            if rank == 0:
                merge_frames_and_write_videos(
                    temp_dir=temp_base,
                    world_size=world_size,
                    multiview_out=multiview_out,
                    singleview_out_base=singleview_out_base,
                    fps=args.fps,
                    grid_width=grid_width,
                    grid_height=grid_height,
                    singleview_size=singleview_size,
                    view_indices=view_indices,
                )
                print(f"Cleaning up temporary directory: {temp_base}")
                shutil.rmtree(temp_base)

            dist.barrier()
        else:
            # Single GPU: write directly. See note in merge_frames_and_write_videos
            # for why multiview uses AVI+MJPG instead of MP4.
            if len(multiview_frames) > 0:
                multiview_writer = cv2.VideoWriter(
                    str(multiview_out),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    args.fps,
                    (grid_width, grid_height),
                )
                if not multiview_writer.isOpened():
                    raise RuntimeError(
                        f"Failed to open multiview VideoWriter for {multiview_out} at {grid_width}x{grid_height}"
                    )
                for frame in multiview_frames:
                    multiview_writer.write(frame)
                multiview_writer.release()
                print(f"Wrote {multiview_out}")

            for view_idx in view_indices:
                frames = singleview_frames_per_view.get(view_idx, [])
                if len(frames) > 0:
                    if len(view_indices) == 1:
                        singleview_out = singleview_out_base
                    else:
                        stem = singleview_out_base.stem
                        singleview_out = singleview_out_base.parent / f"{stem}_view{view_idx}.mp4"

                    singleview_writer = cv2.VideoWriter(
                        str(singleview_out),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        args.fps,
                        singleview_size,
                    )
                    for frame in frames:
                        singleview_writer.write(frame)
                    singleview_writer.release()
                    print(f"Wrote {singleview_out} ({len(frames)} frames)")


def ddp_main_inference(rank: int, world_size: int, args, master_port: str):
    """
    DDP wrapper around main_inference function.

    Supports two launch modes:
    1. mp.spawn (single-node): rank is passed by spawn, local_rank == rank
    2. torchrun/SLURM (multi-node): environment variables are auto-detected and used
    """
    # Under torchrun/SLURM the environment is authoritative; under mp.spawn the
    # local rank equals the rank.
    rank, world_size, gpu_rank = resolve_launch(rank, world_size)

    setup_ddp(rank, world_size, master_port, local_rank=gpu_rank, timeout_s=getattr(args, "dist_timeout", None))

    # Set device override for this rank
    device_override = f"cuda:{gpu_rank}"

    try:
        main_inference(args, rank=rank, world_size=world_size, device_override=device_override)
    finally:
        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="Run simple multi-view inference on a preprocessed dataset")
    parser.add_argument("--dataset", required=True, type=str, help="Path to preprocessed SLEAP HDF5 dataset")
    parser.add_argument("--fps", type=int, default=60, help="Output video FPS (default: 60)")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all frames). Useful for quick testing. With --generate_num_subclips > 1, this is the length of each subclip.",
    )
    parser.add_argument(
        "--generate_num_subclips",
        type=int,
        default=1,
        help="Generate N subclips evenly spaced across the dataset, each --max_frames "
        "long. Subclip i starts at index i * len(dataset) / N. Each output "
        "video and exported animation file is suffixed with the frame range. "
        "Falls back to a single full-dataset clip if subclips don't fit. "
        "Default: 1 (single clip).",
    )
    parser.add_argument(
        "--disable_scaling", action="store_true", help="Disable part scaling (log_beta_scales) for comparison/debugging"
    )
    parser.add_argument(
        "--disable_translation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable part translation (betas_trans) for comparison/debugging. Pass --no-disable_translation to keep translation enabled.",
    )
    parser.add_argument(
        "--view_indices",
        type=str,
        default="0",
        help="Comma-separated list of camera view indices to render for singleview output (default: '0'). E.g., '0,4,11' renders views 0, 4, and 11.",
    )
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=0,
        help="Number of frames to average predictions over for temporal smoothing (default: 0, disabled)",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="Number of GPUs to use (default: 1, ignored when using torchrun)"
    )
    parser.add_argument(
        "--master-port",
        type=str,
        default=None,
        help="Master port for distributed processing (default: from MASTER_PORT env var or 12355)",
    )
    parser.add_argument(
        "--smal_file", type=str, default=None, help="Path to SMAL model file to override config.py SMAL_FILE (optional)"
    )
    parser.add_argument(
        "--shape_family",
        type=int,
        default=None,
        help="Shape family to use with --smal_file (optional, defaults to config.SHAPE_FAMILY)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pth) to use for inference (default: auto-detected from multiview_checkpoints/)",
    )
    parser.add_argument(
        "--export_animation",
        type=str,
        default=None,
        help="Optional output path stem for SMIL animation export. "
        "Writes <stem>.npz + <stem>.json with raw (pre-smoothing) parameters "
        "and per-view cameras. Gathered to rank 0 in multi-GPU runs. "
        'NOTE: any string is accepted as-is (e.g. "True" writes True.npz) — '
        "no validation is performed, so pass a real path/filename stem.",
    )
    parser.add_argument(
        "--dist_timeout",
        type=int,
        default=None,
        help="Distributed process-group timeout in SECONDS for multi-GPU runs "
        "(default: SMILIFY_DIST_TIMEOUT_S env var, else 14400 = 4 h). Must "
        "exceed the longest rank-0-only phase (prediction gather, animation "
        "export, video merge) or the NCCL watchdog kills the ranks waiting at "
        "the next barrier — on ~100k-frame datasets the old 1800 s default "
        "aborted the job after inference had already finished.",
    )
    parser.add_argument(
        "--render_resolution",
        type=int,
        default=None,
        help="Square pixel resolution for the single-view mesh visualization. "
        "The mesh is rendered and the background footage interpolated up to "
        "match (native footage is 512). Default: None = renderer's native "
        "image_size (224). Does NOT affect model inference / backbone input, "
        "and does NOT change the multi-view grid. Note: render time and "
        "single-view file size scale ~quadratically with this value.",
    )
    args = parser.parse_args()

    # Get master port from args or environment variable
    master_port = args.master_port or os.environ.get("MASTER_PORT", "12355")

    # Check if launched via torchrun/torch.distributed.launch (HPC environment)
    if is_torchrun_launched():
        # Launched via torchrun - processes are already spawned by the launcher
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Only print from rank 0 to avoid duplicate output
        if rank == 0:
            local_rank = int(os.environ["LOCAL_RANK"])
            print("Detected torchrun/HPC launch environment:")
            print(f"  Global rank: {rank}")
            print(f"  Local rank (GPU): {local_rank}")
            print(f"  World size: {world_size}")
            print(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'not set')}")
            print(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'not set')}")

        # Call ddp_main_inference directly
        ddp_main_inference(rank, world_size, args, master_port)

    elif args.num_gpus > 1:
        # Manual multi-GPU launch using mp.spawn
        try:
            validate_num_gpus(args.num_gpus)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            exit(1)

        print(f"Launching multi-GPU inference on {args.num_gpus} GPUs (using mp.spawn)...")
        print(f"Master port: {master_port}")

        # Launch multi-GPU processing using spawn
        mp.spawn(ddp_main_inference, args=(args.num_gpus, args, master_port), nprocs=args.num_gpus, join=True)
    else:
        # Single GPU processing (existing path)
        print("Launching single-GPU inference...")
        main_inference(args, rank=0, world_size=1, device_override=None)


if __name__ == "__main__":
    main()
