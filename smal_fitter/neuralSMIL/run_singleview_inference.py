#!/usr/bin/env python3
"""
SMIL Image Regressor Inference Script

Loads a trained ``SMILImageRegressor`` checkpoint and runs inference on a
**preprocessed HDF5 dataset**, a folder of images, or a video, writing
visualizations / videos / animation exports to an output folder.

Usage:
    # For a preprocessed dataset (preferred - mirrors run_multiview_inference.py)
    python -m smal_fitter.neuralSMIL.run_singleview_inference \
        --checkpoint path/to/checkpoint.pth --dataset path/to/dataset.h5 \
        --output_folder path/to/output

    # For images
    python -m smal_fitter.neuralSMIL.run_singleview_inference \
        --checkpoint path/to/checkpoint.pth --input_folder path/to/images --output_folder path/to/output

    # For video
    python -m smal_fitter.neuralSMIL.run_singleview_inference \
        --checkpoint path/to/checkpoint.pth --input_video path/to/video.mp4 --output_folder path/to/output

Dataset mode (issue #100)
-------------------------
Historically the single-view entrypoint only accepted raw images / video while
``run_multiview_inference.py`` ran over a preprocessed dataset, and the two
render paths had drifted apart. ``--dataset`` closes that gap: it consumes the
same HDF5 datasets the trainer and ``benchmark_model.py`` use and follows the
identical conventions, namely

* **Camera intrinsics / extrinsics.** A ``camera_centric`` checkpoint renders
  through the FIXED PyTorch3D identity camera with the vertical FOV (and aspect
  ratio) taken from the sample's calibration, exactly like
  ``SMILImageRegressor.predict_from_batch``. A ``model_centric`` checkpoint
  renders through its own predicted ``cam_rot`` / ``cam_trans`` / ``fov``. A
  multi-view HDF5 is opened with ``return_single_view=True`` and the matching
  ``camera_centric`` flag so the dataset re-anchors the world onto the sampled
  camera before handing back keypoints and 3D.
* **Shape-space variation.** ``scale_trans_mode`` decides whether
  ``log_beta_scales`` / ``betas_trans`` are PCA weights (``separate`` +
  ``use_pca_transformation``, which must be expanded to per-joint values) or
  already per-joint (``separate`` without PCA, ``entangled_with_betas``), or
  absent (``ignore``).
* **Mesh scaling and cropping.** Placement uses the legacy 10x UE scaling or the
  predicted per-sample ``mesh_scale``, whichever the checkpoint was trained
  with, and the dataset's own crop is used as-is (``--crop_mode`` applies only
  to raw image / video input, where no preprocessing has happened yet).

Features:
    - Loads trained model from checkpoint
    - Processes a preprocessed dataset, images, or video files
    - Supports center-crop preprocessing for raw input (matching training)
    - Generates SMIL model visualizations
    - Saves predicted parameters and visualizations
    - For videos / datasets: generates output video and per-frame results
    - Optional temporal smoothing and AMASS-style animation export
"""

# ===== CRITICAL: Force IPv4 before torch/distributed resolves anything =====
# Prevents "Address family not supported by protocol" (errno: 97) on HPC systems
# without full IPv6 support. Shared with run_multiview_inference.py.
from smal_fitter.neuralSMIL.inference_ddp import force_ipv4_getaddrinfo

force_ipv4_getaddrinfo()
# ===== End IPv4 forcing =====

import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import h5py
import json
import pickle as pkl

import torch
import torch.multiprocessing as mp
import numpy as np
import cv2
import imageio
from tqdm import tqdm

# Set matplotlib backend BEFORE any other imports to prevent tkinter issues
import matplotlib

matplotlib.use("Agg")


from smal_fitter.neuralSMIL.smil_image_regressor import SMILImageRegressor
from smal_fitter.neuralSMIL.training_config import TrainingConfig
from smal_fitter.fitter import SMALFitter
import config
from smal_fitter.neuralSMIL.animation_export import AnimationRecorder, build_recorder_from_config
from smal_fitter.neuralSMIL.inference_common import (
    InMemoryImageExporter,
    PredictionSmoother,
    apply_pose_and_shape,
    compute_subclip_ranges,
    pad_or_resize,
    params_to_cpu,
    params_to_device,
    place_mesh,
    resolve_frame_convention,
    resolve_mesh_scale,
    resolve_render_camera,
    resolve_singleview_dataset_kwargs,
    write_video,
)
from smal_fitter.neuralSMIL.inference_ddp import (
    barrier,
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
from sleap_data_loader import SLEAPDataLoader
import importlib.util
import importlib


def _import_sleap_preprocessor():
    try:
        module = importlib.import_module("smal_fitter.sleap_data.preprocess_sleap_dataset")
        return module.SLEAPDatasetPreprocessor
    except ModuleNotFoundError:
        module_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "sleap_data", "preprocess_sleap_dataset.py")
        )
        spec = importlib.util.spec_from_file_location("preprocess_sleap_dataset", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not locate preprocess_sleap_dataset at {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SLEAPDatasetPreprocessor


SLEAPDatasetPreprocessor = _import_sleap_preprocessor()


class SLEAPCroppingHelper:
    """Leverage preprocessing logic from preprocess_sleap_dataset for bbox cropping."""

    def __init__(
        self,
        project_path: str,
        crop_mode: str,
        target_resolution: int,
        backbone_name: str,
        use_reprojections: bool = True,
    ):
        self.project_path = Path(project_path)
        self.crop_mode = crop_mode
        self.preprocessor = SLEAPDatasetPreprocessor(
            joint_lookup_table_path=None,
            shape_betas_table_path=None,
            target_resolution=target_resolution,
            backbone_name=backbone_name,
            crop_mode=crop_mode,
            use_reprojections=use_reprojections,
        )

        session_paths = self.preprocessor.discover_sleap_sessions(project_path)
        if not session_paths:
            raise ValueError(f"No SLEAP sessions found in {project_path}")

        self.session_map: Dict[str, Path] = {Path(s).name: Path(s) for s in session_paths}
        self.single_session_name = next(iter(self.session_map)) if len(self.session_map) == 1 else None

        self.session_loaders: Dict[str, SLEAPDataLoader] = {}
        self.camera_data_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.reprojection_cache: Dict[str, Tuple[Optional[h5py.File], Dict[str, h5py.File]]] = {}
        self._warned_missing_camera: Set[str] = set()
        self._warned_missing_frame: Set[Tuple[str, int]] = set()

    def close(self):
        """Close any open reprojection file handles."""
        for root_handle, sub_handles in self.reprojection_cache.values():
            try:
                if root_handle is not None:
                    root_handle.close()
            except Exception:
                pass
            for handle in sub_handles.values():
                try:
                    handle.close()
                except Exception:
                    pass

    def list_cameras(self) -> List[str]:
        cameras = set()
        for session_name in self.session_map:
            loader = self._get_loader(session_name)
            cameras.update(loader.camera_views)
        return sorted(cameras)

    def _resolve_session(self, media_path: str) -> Tuple[str, Path]:
        for session_name, session_path in self.session_map.items():
            if session_name in media_path:
                return session_name, session_path
        if self.single_session_name is not None:
            session_path = self.session_map[self.single_session_name]
            return self.single_session_name, session_path
        raise ValueError(f"Could not infer SLEAP session for path: {media_path}")

    def _get_loader(self, session_name: str) -> SLEAPDataLoader:
        if session_name not in self.session_loaders:
            session_path = self.session_map[session_name]
            self.session_loaders[session_name] = SLEAPDataLoader(
                project_path=str(session_path),
                lookup_table_path=self.preprocessor.joint_lookup_table_path,
                shape_betas_path=self.preprocessor.shape_betas_table_path,
            )
        return self.session_loaders[session_name]

    def _get_reprojection_handles(self, session_name: str) -> Tuple[Optional[h5py.File], Dict[str, h5py.File]]:
        if session_name in self.reprojection_cache:
            return self.reprojection_cache[session_name]

        session_path = self.session_map[session_name]
        root_handle = None
        root_reproj = session_path / "reprojections.h5"
        if root_reproj.exists():
            try:
                root_handle = h5py.File(str(root_reproj), "r")
            except Exception as exc:
                print(f"Warning: Failed to open reprojections file {root_reproj}: {exc}")
                root_handle = None

        sub_handles = (
            self.preprocessor._find_all_reprojection_files(str(session_path))
            if self.preprocessor.use_reprojections
            else {}
        )
        self.reprojection_cache[session_name] = (root_handle, sub_handles)
        return self.reprojection_cache[session_name]

    def _get_camera_data(self, session_name: str, camera_name: str, loader: SLEAPDataLoader) -> Dict[str, Any]:
        key = (session_name, camera_name)
        if key not in self.camera_data_cache:
            self.camera_data_cache[key] = loader.load_camera_data(camera_name)
        return self.camera_data_cache[key]

    def _infer_camera(self, media_path: str, loader: SLEAPDataLoader, explicit_camera: Optional[str]) -> Optional[str]:
        if explicit_camera:
            return explicit_camera

        path_lower = str(media_path).lower()
        stem_lower = Path(media_path).stem.lower()
        parent_names = {p.name.lower() for p in Path(media_path).parents}
        for camera in loader.camera_views:
            cam_lower = camera.lower()
            if f"_cam{cam_lower}" in path_lower or f"-cam{cam_lower}" in path_lower:
                return camera
            if cam_lower in stem_lower.split("_"):
                return camera
            # Also check directory names in the path (e.g., .../Camera2/...)
            if cam_lower in parent_names:
                return camera
        if len(loader.camera_views) == 1:
            return loader.camera_views[0]
        return None

    def _get_reprojection_handle_for_camera(self, session_name: str, camera_name: str) -> Optional[h5py.File]:
        root_handle, sub_handles = self._get_reprojection_handles(session_name)
        if root_handle is not None:
            return root_handle
        session_path = self.session_map[session_name]
        camera_subdir = self.preprocessor._find_camera_subdir(str(session_path), camera_name)
        if camera_subdir and camera_subdir in sub_handles:
            return sub_handles[camera_subdir]
        return None

    def _extract_keypoints(
        self, loader: SLEAPDataLoader, camera_data: Dict[str, Any], camera_name: str, frame_idx: int, session_name: str
    ) -> Optional[np.ndarray]:
        reproj_handle = None
        if self.preprocessor.use_reprojections:
            reproj_handle = self._get_reprojection_handle_for_camera(session_name, camera_name)
        try:
            keypoints_2d, visibility = self.preprocessor._extract_2d_keypoints_for_frame(
                loader=loader,
                camera_data=camera_data,
                camera_name=camera_name,
                frame_idx=frame_idx,
                reproj_handle=reproj_handle,
            )
        except Exception as exc:
            key = (camera_name, frame_idx)
            if key not in self._warned_missing_frame:
                print(f"Warning: Failed to extract keypoints for camera '{camera_name}', frame {frame_idx}: {exc}")
                self._warned_missing_frame.add(key)
            return None

        keypoints_2d = self.preprocessor._sanitize_array(keypoints_2d, default_value=np.nan)
        if keypoints_2d is None:
            return None

        keypoints = np.asarray(keypoints_2d, dtype=np.float32)
        vis_mask = None
        if visibility is not None:
            visibility = np.asarray(visibility)
            if visibility.dtype == bool:
                vis_mask = visibility
            else:
                vis_mask = visibility > 0.5
        if vis_mask is not None and vis_mask.shape[0] == keypoints.shape[0]:
            keypoints = keypoints[vis_mask]

        if keypoints.size == 0:
            return None

        valid_mask = np.isfinite(keypoints).all(axis=1)
        keypoints = keypoints[valid_mask]
        keypoints = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
        return keypoints if keypoints.size > 0 else None

    def preprocess_image(
        self, image: np.ndarray, media_path: str, frame_idx: int, explicit_camera: Optional[str] = None
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        session_name, _ = self._resolve_session(media_path)
        loader = self._get_loader(session_name)
        camera_name = self._infer_camera(media_path, loader, explicit_camera)
        if camera_name is None:
            if media_path not in self._warned_missing_camera:
                print(f"Warning: Could not infer SLEAP camera for '{media_path}'")
                self._warned_missing_camera.add(media_path)
            return None

        camera_data = self._get_camera_data(session_name, camera_name, loader)
        keypoints = self._extract_keypoints(loader, camera_data, camera_name, frame_idx, session_name)
        if keypoints is None or len(keypoints) == 0:
            return None

        if image.max() <= 1.0:
            work_image = (image * 255.0).astype(np.uint8)
        else:
            work_image = image.astype(np.uint8)

        processed_image, transform_info = self.preprocessor._preprocess_image(work_image, keypoints)
        return processed_image, transform_info


class InferenceImageExporter:
    """Enhanced image exporter for inference results."""

    def __init__(self, output_dir: str):
        """
        Initialize the image exporter.

        Args:
            output_dir: Directory to save visualization images
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export(
        self,
        collage_np: np.ndarray,
        batch_id: int,
        global_id: int,
        img_parameters: Dict[str, Any],
        vertices: torch.Tensor,
        faces: np.ndarray,
        img_idx: int = 0,
        image_name: str = "image",
        **kwargs,  # tolerate extra kwargs (e.g. epoch) passed by SMALFitter.generate_visualization
    ):
        """
        Export visualization image and parameters.

        Args:
            collage_np: Visualization collage as numpy array
            batch_id: Batch ID
            global_id: Global ID
            img_parameters: Dictionary of SMIL parameters
            vertices: Model vertices
            faces: Model faces
            img_idx: Image index
            image_name: Base name for the image
        """
        # Save visualization image
        vis_filename = f"{image_name}_visualization.png"
        vis_path = os.path.join(self.output_dir, vis_filename)
        imageio.imsave(vis_path, collage_np)

        # Save parameters as JSON (for human readability)
        params_filename = f"{image_name}_parameters.json"
        params_path = os.path.join(self.output_dir, params_filename)

        # Convert numpy arrays / tensors to lists for JSON serialization, recursing
        # into nested dicts/lists (a nested tensor previously escaped the top-level
        # check and raised "Object of type Tensor is not JSON serializable").
        def _to_jsonable(v):
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().numpy().tolist()
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
            if isinstance(v, dict):
                return {k: _to_jsonable(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_to_jsonable(x) for x in v]
            return v

        json_parameters = _to_jsonable(img_parameters)

        with open(params_path, "w") as f:
            json.dump(json_parameters, f, indent=2, default=str)

        # Save parameters as pickle (for exact reproduction)
        pkl_filename = f"{image_name}_parameters.pkl"
        pkl_path = os.path.join(self.output_dir, pkl_filename)
        with open(pkl_path, "wb") as f:
            pkl.dump(img_parameters, f)

        print(f"Saved results for {image_name}:")
        print(f"  Visualization: {vis_path}")
        print(f"  Parameters (JSON): {params_path}")
        print(f"  Parameters (PKL): {pkl_path}")


def load_model_from_checkpoint(
    checkpoint_path: str, device: str
) -> Tuple[SMILImageRegressor, Dict[str, Any], Dict[str, Any]]:
    """
    Load a trained SMILImageRegressor model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: PyTorch device ('cuda' or 'cpu')

    Returns:
        Tuple of (loaded_model, model_config, conventions) where *conventions* is
        the frame-convention / camera / mesh-scale dict resolved by
        ``inference_common.resolve_frame_convention`` — the same resolution
        ``benchmark_model.py`` performs, so inference and benchmarking agree on
        what a checkpoint means.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("Checkpoint loaded successfully")

        # Prefer config from checkpoint so inference matches training; fall back to training_config if missing
        ckpt_config = checkpoint.get("config", {})
        training_config_fallback = TrainingConfig.get_all_config()
        fallback_model = training_config_fallback["model_config"].copy()
        fallback_params = training_config_fallback["training_params"]

        if ckpt_config:
            model_config = {**fallback_model, **ckpt_config.get("model_config", {})}
            rotation_representation = (ckpt_config.get("training_params") or {}).get(
                "rotation_representation"
            ) or fallback_params.get("rotation_representation", "6d")
            scale_trans_mode = ckpt_config.get("scale_trans_mode") or TrainingConfig.get_scale_trans_mode()
            shape_family = ckpt_config.get("shape_family", config.SHAPE_FAMILY)
            config_source = "checkpoint (fallback: training_config for missing keys)"
        else:
            model_config = fallback_model
            rotation_representation = fallback_params["rotation_representation"]
            scale_trans_mode = TrainingConfig.get_scale_trans_mode()
            shape_family = config.SHAPE_FAMILY
            config_source = "training_config.py (no config in checkpoint)"

        # Frame convention + camera flags (persisted at the TOP level of
        # checkpoint["config"] by the trainer, not inside model_config).
        # camera_centric checkpoints use a fixed identity camera and were
        # trained without the 10x UE scaling. Resolved by the shared helper so
        # this script, benchmark_model.py and the multi-view entrypoint cannot
        # drift apart again (issue #100).
        conventions = resolve_frame_convention(ckpt_config, checkpoint.get("model_state_dict", {}))
        frame_convention = conventions["frame_convention"]
        fixed_camera = conventions["fixed_camera"]
        use_ue_scaling = conventions["use_ue_scaling"]
        allow_mesh_scaling = conventions["allow_mesh_scaling"]
        mesh_scale_init = conventions["mesh_scale_init"]
        model_config["frame_convention"] = frame_convention
        model_config["fixed_camera"] = fixed_camera
        conventions["scale_trans_mode"] = scale_trans_mode
        conventions["shape_family"] = shape_family
        conventions["rotation_representation"] = rotation_representation

        print(f"Configuration from {config_source}:")
        print(f"  frame_convention: {frame_convention} (fixed_camera={fixed_camera}, use_ue_scaling={use_ue_scaling})")
        print(f"  from_multiview: {conventions['from_multiview']}")
        print(f"  allow_mesh_scaling: {allow_mesh_scaling} (init {mesh_scale_init})")
        print(f"  backbone_name: {model_config['backbone_name']}")
        print(f"  head_type: {model_config.get('head_type', 'mlp')}")
        print(f"  rotation_representation: {rotation_representation}")
        print(f"  scale_trans_mode: {scale_trans_mode}")
        print(f"  shape_family: {shape_family}")

        # If the checkpoint specifies a SMAL/SMIL model file, re-derive
        # config.dd, N_POSE, N_BETAS, joint_names, etc. from that file.
        if ckpt_config and ckpt_config.get("smal_file"):
            from smal_fitter.neuralSMIL.configs import apply_smal_file_override

            apply_smal_file_override(
                ckpt_config["smal_file"],
                shape_family=shape_family,
            )

        # Verify this matches the checkpoint by checking state dict keys
        state_dict = checkpoint["model_state_dict"]

        # Check for transformer head
        has_transformer_head = any("transformer_head" in key for key in state_dict.keys())

        # Infer backbone type from feature dimensions in the checkpoint
        # Note: Backbone weights are NOT saved (frozen pretrained weights are re-downloaded on load)
        # So we detect backbone type from the input dimension of the regression head
        inferred_backbone = None
        if has_transformer_head:
            # Check transformer head input dimension from token_embedding weight
            token_emb_key = "transformer_head.token_embedding.weight"
            if token_emb_key in state_dict:
                feature_dim = state_dict[token_emb_key].shape[0]
                if feature_dim == 1024:
                    inferred_backbone = "vit_large"
                elif feature_dim == 768:
                    inferred_backbone = "vit_base"
                elif feature_dim == 2048:
                    inferred_backbone = "resnet"
                print(f"Inferred backbone from checkpoint feature dim ({feature_dim}): {inferred_backbone}")

        # Validate config matches inferred backbone
        config_backbone = model_config["backbone_name"]
        if inferred_backbone:
            config_is_vit_large = "vit_large" in config_backbone
            config_is_vit_base = "vit_base" in config_backbone
            config_is_resnet = config_backbone.startswith("resnet")

            if inferred_backbone == "vit_large" and not config_is_vit_large:
                print(f"WARNING: Checkpoint was trained with ViT-Large but config specifies {config_backbone}")
            elif inferred_backbone == "vit_base" and not config_is_vit_base:
                print(f"WARNING: Checkpoint was trained with ViT-Base but config specifies {config_backbone}")
            elif inferred_backbone == "resnet" and not config_is_resnet:
                print(f"WARNING: Checkpoint was trained with ResNet but config specifies {config_backbone}")
            else:
                print(f"Backbone configuration matches checkpoint: {config_backbone}")

        if model_config["head_type"] == "transformer_decoder" and not has_transformer_head:
            print("WARNING: Config specifies transformer_decoder but checkpoint doesn't contain transformer_head")

        print("Checkpoint verification:")
        print(f"  Inferred backbone: {inferred_backbone or 'unknown'}")
        print(f"  Contains transformer_head: {has_transformer_head}")

        print("Model configuration:")
        for key, value in model_config.items():
            if key != "transformer_config":
                print(f"  {key}: {value}")

        print(f"Using rotation representation: {rotation_representation}")

        # For inference, always use batch_size=1
        # The checkpoint may have been saved with a different batch size during training,
        # but for inference we process one image at a time
        batch_size = 1
        print(f"Using batch size: {batch_size} (inference mode)")

        # Create placeholder data for model initialization
        placeholder_data = torch.zeros((batch_size, 3, 512, 512))

        # Determine input resolution from the centralized backbone factory
        from smal_fitter.neuralSMIL.backbone_factory import BackboneFactory

        input_resolution = BackboneFactory.get_default_input_resolution(model_config["backbone_name"])

        print(f"Creating model with input resolution: {input_resolution}")

        # Initialize model with detected configuration
        model = SMILImageRegressor(
            device=device,
            data_batch=placeholder_data,
            batch_size=batch_size,
            shape_family=shape_family,
            use_unity_prior=model_config.get("use_unity_prior", False),
            rgb_only=model_config.get("rgb_only", True),
            freeze_backbone=model_config.get("freeze_backbone", True),
            hidden_dim=model_config.get("hidden_dim", 1024),
            # Legacy replicAnt single-view uses 10x UE scaling; camera-centric
            # (multi-view-derived) checkpoints do not (scale baked via world_scale).
            use_ue_scaling=use_ue_scaling,
            rotation_representation=rotation_representation,
            input_resolution=input_resolution,
            backbone_name=model_config["backbone_name"],
            head_type=model_config.get("head_type", "mlp"),
            transformer_config=model_config.get("transformer_config", {}),
            scale_trans_mode=scale_trans_mode,  # Critical for correct output dimensions
            fixed_camera=fixed_camera,  # camera-centric: pin camera to identity
            allow_mesh_scaling=allow_mesh_scaling,  # rebuild the mesh_scale head
            mesh_scale_init=mesh_scale_init,
        ).to(device)

        # Load model state, handling batch size differences
        # For inference, we need the neural network weights, but skip SMAL optimization parameters
        state_dict = checkpoint["model_state_dict"]

        # Filter out SMAL optimization parameters that have batch size dependencies
        # These are specific to the SMALFitter optimization process, not the neural network
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

        # Keep all neural network parameters (backbone, transformer_head, fc layers, etc.)
        nn_state_dict = {}
        skipped_params = []

        for k, v in state_dict.items():
            # Skip SMAL optimization parameters that are specific to the optimization process
            if any(k == param or k.startswith(param + ".") for param in smal_optimization_params):
                skipped_params.append(k)
            else:
                nn_state_dict[k] = v

        print(f"Loading {len(nn_state_dict)} neural network parameters")
        print(
            f"Skipping {len(skipped_params)} SMAL optimization parameters: {skipped_params[:5]}{'...' if len(skipped_params) > 5 else ''}"
        )

        # Load the neural network weights
        missing_keys, unexpected_keys = model.load_state_dict(nn_state_dict, strict=False)

        if missing_keys:
            print(
                f"Missing keys (will use random initialization): {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''}"
            )
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {unexpected_keys[:3]}{'...' if len(unexpected_keys) > 3 else ''}")
        model.eval()

        print("Model loaded and set to evaluation mode")

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Model statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Head type: {model.head_type}")
        print(f"  Backbone: {model.backbone_name}")
        print(f"  Input resolution: {input_resolution}")

        return model, model_config, conventions

    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")


def find_image_files(input_folder: str, supported_extensions: List[str] = None) -> List[str]:
    """
    Find all image files in the input folder.

    Args:
        input_folder: Path to folder containing images
        supported_extensions: List of supported file extensions

    Returns:
        List of image file paths
    """
    if supported_extensions is None:
        supported_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".tif",
            ".JPG",
            ".JPEG",
            ".PNG",
            ".BMP",
            ".TIFF",
            ".TIF",
        ]

    image_files = []
    input_path = Path(input_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_folder}")

    # Find all image files
    for ext in supported_extensions:
        pattern = f"*{ext}"
        image_files.extend(input_path.glob(pattern))

    # Convert to strings and sort
    image_files = [str(f) for f in image_files]
    image_files.sort()

    print(f"Found {len(image_files)} image files in {input_folder}")

    return image_files


def preprocess_frame(
    image: np.ndarray, target_resolution: int, crop_mode: str = "centred"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Preprocess a frame using centred or default resize.

    Args:
        image: Input image (H, W, C) in range [0, 255] or [0, 1]
        target_resolution: Target resolution for model input
        crop_mode: 'centred' or 'default'

    Returns:
        Tuple of (preprocessed_image, transform_info)
    """
    if crop_mode == "bbox_crop":
        raise ValueError(
            "bbox_crop should be handled via SLEAPCroppingHelper; preprocess_frame only supports 'centred' or 'default'."
        )

    # Ensure image is in [0, 255] range
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    original_h, original_w = image.shape[:2]
    transform_info = {
        "original_size": (original_h, original_w),
        "crop_offset": (0, 0),
        "crop_size": (original_h, original_w),
        "scale_factor": 1.0,
        "mode": crop_mode,
    }

    if crop_mode == "centred":
        crop_size = min(original_h, original_w)
        y_offset = (original_h - crop_size) // 2
        x_offset = (original_w - crop_size) // 2
        image = image[y_offset : y_offset + crop_size, x_offset : x_offset + crop_size]
        transform_info["crop_offset"] = (y_offset, x_offset)
        transform_info["crop_size"] = (crop_size, crop_size)
        scale_factor = target_resolution / crop_size
        transform_info["scale_factor"] = scale_factor
        image = cv2.resize(image, (target_resolution, target_resolution))

    else:
        scale_y = target_resolution / original_h
        scale_x = target_resolution / original_w
        transform_info["scale_factor"] = (scale_y, scale_x)
        image = cv2.resize(image, (target_resolution, target_resolution))

    image = image.astype(np.float32) / 255.0
    return image, transform_info


def load_and_preprocess_image(
    image_path: str, model: SMILImageRegressor, crop_mode: str = "centred", keypoints_2d: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, torch.Tensor, Dict[str, Any]]:
    """
    Load and preprocess an image for inference.

    Args:
        image_path: Path to the image file
        model: SMILImageRegressor model for preprocessing
        crop_mode: Cropping mode ('centred' or 'default')

    Returns:
        Tuple of (original_image_array, preprocessed_tensor, transform_info)
    """
    try:
        # Load image
        image_data = imageio.v2.imread(image_path)

        # Keep original for visualization
        original_image = image_data.copy()

        # Preprocess with proper cropping
        target_resolution = model.input_resolution
        preprocessed_image, transform_info = preprocess_frame(image_data, target_resolution, crop_mode)

        # Convert to tensor (C, H, W) format
        preprocessed_tensor = torch.from_numpy(preprocessed_image).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

        return original_image, preprocessed_tensor, transform_info

    except Exception as e:
        raise RuntimeError(f"Failed to load/preprocess image {image_path}: {e}")


def run_inference_on_image(
    model: SMILImageRegressor, image_tensor: torch.Tensor, device: str
) -> Dict[str, torch.Tensor]:
    """
    Run inference on a preprocessed image tensor.

    Args:
        model: SMILImageRegressor model
        image_tensor: Preprocessed image tensor (1, C, H, W)
        device: PyTorch device

    Returns:
        Dictionary of predicted SMIL parameters
    """
    try:
        with torch.no_grad():
            # Move to device
            image_tensor = image_tensor.to(device)

            # Get batch size from tensor
            image_tensor.shape[0]

            # Run inference through the model's forward pass
            # The model's forward() method handles batches correctly
            predicted_params = model.forward(image_tensor)

            # Camera-centric checkpoints: pin the camera to the PyTorch3D identity
            # and inject the chosen FOV. Inference calls forward() directly (not
            # predict_from_batch), and there is no GT calibration for a raw image,
            # so we re-implement the fixed_camera override here, sourcing the FOV
            # from model._inference_fov (resolved in main: --fov or 60.0 default).
            if getattr(model, "fixed_camera", False):
                bs = predicted_params["global_rot"].shape[0]
                predicted_params["cam_rot"] = torch.eye(3, device=device).unsqueeze(0).expand(bs, 3, 3).contiguous()
                predicted_params["cam_trans"] = torch.zeros(bs, 3, device=device)
                fov_deg = float(getattr(model, "_inference_fov", 60.0))
                predicted_params["fov"] = torch.full((bs, 1), fov_deg, device=device)

            # Move results back to CPU for visualization
            cpu_params = {}
            for key, value in predicted_params.items():
                if isinstance(value, torch.Tensor):
                    cpu_params[key] = value.cpu()
                else:
                    cpu_params[key] = value

            return cpu_params

    except Exception as e:
        import traceback

        print("\n" + "=" * 60)
        print("INFERENCE ERROR DEBUG INFO")
        print("=" * 60)
        print(f"Input tensor shape: {image_tensor.shape}")
        print(f"Device: {device}")
        print(f"Model batch_size attribute: {model.batch_size}")
        print(f"Model head_type: {model.head_type}")
        print(f"Model rotation_representation: {model.rotation_representation}")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        raise RuntimeError(f"Inference failed: {e}")


def _build_render_fitter(
    model: SMILImageRegressor,
    predicted_params: Dict[str, torch.Tensor],
    rgb_tensor: torch.Tensor,
    device: str,
    y_data: Optional[Dict[str, Any]] = None,
    disable_scaling: bool = False,
    disable_translation: bool = False,
) -> SMALFitter:
    """Build a throwaway SMALFitter loaded with *predicted_params*, ready to render.

    Centralises the three rules that used to be re-implemented (inconsistently)
    at every render site in this file:

    1. ``propagate_scaling`` must match the training model's setting.
    2. ``log_beta_scales`` / ``betas_trans`` must be interpreted through
       ``scale_trans_mode`` (PCA weights expanded to per-joint values) — see
       ``inference_common.apply_shape_space_params``.
    3. The camera comes from the checkpoint's frame convention: fixed identity +
       calibrated FOV for ``camera_centric``, predicted camera otherwise — see
       ``inference_common.resolve_render_camera``.
    """
    temp_fitter = SMALFitter(
        device=device,
        data_batch=rgb_tensor,
        batch_size=1,
        shape_family=config.SHAPE_FAMILY,
        use_unity_prior=False,
        rgb_only=True,
    )

    # CRITICAL: Match propagate_scaling to the training model's setting.
    # The model learns scales with propagate_scaling=True (set in SMILImageRegressor.__init__),
    # so visualization must also use propagate_scaling=True for consistent geometry.
    temp_fitter.propagate_scaling = model.propagate_scaling

    apply_pose_and_shape(
        temp_fitter,
        model,
        predicted_params,
        index=0,
        disable_scaling=disable_scaling,
        disable_translation=disable_translation,
    )

    cam_rot, cam_trans, fov, aspect = resolve_render_camera(model, predicted_params, y_data, device)
    temp_fitter.fov.data = fov.reshape(-1)[:1].clone()
    temp_fitter.renderer.set_camera_parameters(R=cam_rot, T=cam_trans, fov=fov, aspect_ratio=aspect)

    return temp_fitter


def _render_mesh_rgb(
    model: SMILImageRegressor,
    predicted_params: Dict[str, torch.Tensor],
    rgb_tensor: torch.Tensor,
    device: str,
    y_data: Optional[Dict[str, Any]] = None,
    disable_scaling: bool = False,
    disable_translation: bool = False,
) -> np.ndarray:
    """Render the predicted mesh over *rgb_tensor* and return float RGB in [0, 1]."""
    temp_fitter = _build_render_fitter(
        model,
        predicted_params,
        rgb_tensor,
        device,
        y_data=y_data,
        disable_scaling=disable_scaling,
        disable_translation=disable_translation,
    )
    mesh_scale = resolve_mesh_scale(model, predicted_params)

    with torch.no_grad():
        # SMAL reads the beta count as ``beta.shape[1]``, so betas must be 2-D
        # here. ``apply_pose_and_shape`` writes the (N_BETAS,) layout SMALFitter
        # declares (its own generate_visualization expands it); this call site
        # bypasses that expansion, so add the batch dim explicitly.
        verts, joints, Rs, v_shaped = temp_fitter.smal_model(
            temp_fitter.betas.reshape(1, -1),
            torch.cat([temp_fitter.global_rotation.unsqueeze(1), temp_fitter.joint_rotations], dim=1),
            betas_logscale=temp_fitter.log_beta_scales,
            betas_trans=temp_fitter.betas_trans,
            propagate_scaling=temp_fitter.propagate_scaling,
        )

        verts, joints = place_mesh(
            model.use_ue_scaling, verts, joints, temp_fitter.trans, mesh_scale=mesh_scale
        )

        canonical_joints = joints[:, config.CANONICAL_MODEL_JOINTS]
        faces_batch = temp_fitter.smal_model.faces.unsqueeze(0).expand(verts.shape[0], -1, -1)

        _, _, rendered_image = temp_fitter.renderer(
            verts.float(), canonical_joints.float(), faces_batch, render_texture=True
        )

    rendered_np = rendered_image[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    return np.clip(rendered_np, 0, 1)


def render_model_only(
    model: SMILImageRegressor,
    predicted_params: Dict[str, torch.Tensor],
    device: str,
    render_size: int,
    y_data: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Render only the predicted 3D model without any background.

    Args:
        model: SMILImageRegressor model
        predicted_params: Dictionary of predicted SMIL parameters
        device: PyTorch device
        render_size: Target render resolution
        y_data: Optional target dict supplying camera calibration (FOV / aspect)
                for camera-centric checkpoints

    Returns:
        Rendered model image (render_size, render_size, 3) in RGB, range [0, 255]
    """
    try:
        rgb_tensor = torch.zeros((1, 3, render_size, render_size), device=device)
        rendered_np = _render_mesh_rgb(model, predicted_params, rgb_tensor, device, y_data=y_data)
        return (rendered_np * 255).astype(np.uint8)
    except Exception as e:
        print(f"Warning: Failed to render model: {e}")
        return np.zeros((render_size, render_size, 3), dtype=np.uint8)


def render_prediction_on_frame(
    model: SMILImageRegressor,
    predicted_params: Dict[str, torch.Tensor],
    original_frame: np.ndarray,
    device: str,
    transform_info: Optional[Dict[str, Any]] = None,
    y_data: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Render the predicted 3D model onto the original frame.

    Args:
        model: SMILImageRegressor model
        predicted_params: Dictionary of predicted SMIL parameters
        original_frame: Original frame (H, W, 3) in RGB, range [0, 255]
        device: PyTorch device
        transform_info: Crop/resize bookkeeping from ``preprocess_frame`` so the
                        render is composited back onto the region it was cropped
                        from (identity for already-cropped dataset frames)
        y_data: Optional target dict supplying camera calibration (FOV / aspect)

    Returns:
        Rendered frame with 3D model overlay (H, W, 3) in RGB, range [0, 255]
    """
    try:
        frame_h, frame_w = original_frame.shape[:2]
        render_size = model.input_resolution

        if original_frame.max() > 1.0:
            rgb_image = original_frame.astype(np.float32) / 255.0
        else:
            rgb_image = original_frame.astype(np.float32)

        rgb_resized = cv2.resize(rgb_image, (render_size, render_size))
        rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).unsqueeze(0)

        rendered_np = _render_mesh_rgb(model, predicted_params, rgb_tensor, device, y_data=y_data)

        alpha = 0.6  # Transparency of the overlay
        overlay_base = original_frame.astype(np.float32) / 255.0

        placed_overlay = overlay_base.copy()

        if transform_info is not None and transform_info.get("mode") in ("centred", "bbox_crop"):
            crop_height, crop_width = transform_info.get("crop_size", (frame_h, frame_w))
            y_offset, x_offset = transform_info.get("crop_offset", (0, 0))

            crop_height = int(round(crop_height))
            crop_width = int(round(crop_width))
            y_offset = int(round(y_offset))
            x_offset = int(round(x_offset))

            if crop_height <= 0 or crop_width <= 0:
                rendered_resized = cv2.resize(rendered_np, (frame_w, frame_h))
                blended = alpha * rendered_resized + (1 - alpha) * overlay_base
                return (blended * 255).astype(np.uint8)

            overlay_y_end = min(y_offset + crop_height, frame_h)
            overlay_x_end = min(x_offset + crop_width, frame_w)

            if overlay_y_end <= y_offset or overlay_x_end <= x_offset:
                rendered_resized = cv2.resize(rendered_np, (frame_w, frame_h))
                blended = alpha * rendered_resized + (1 - alpha) * overlay_base
                return (blended * 255).astype(np.uint8)

            target_height = overlay_y_end - y_offset
            target_width = overlay_x_end - x_offset

            rendered_resized = cv2.resize(rendered_np, (target_width, target_height))

            base_region = overlay_base[y_offset:overlay_y_end, x_offset:overlay_x_end]
            blended_region = alpha * rendered_resized + (1 - alpha) * base_region
            placed_overlay[y_offset:overlay_y_end, x_offset:overlay_x_end] = blended_region
            return (placed_overlay * 255).astype(np.uint8)
        else:
            rendered_resized = cv2.resize(rendered_np, (frame_w, frame_h))
            blended = alpha * rendered_resized + (1 - alpha) * overlay_base
            return (blended * 255).astype(np.uint8)

    except Exception as e:
        print(f"Warning: Failed to render prediction: {e}")
        # Return original frame on error
        return original_frame


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Normalise an RGB array to uint8 [0, 255] whether it arrived as [0,1] or [0,255]."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _set_target_joints(temp_fitter, keypoints_2d, keypoint_visibility, device, target_size: int) -> None:
    """Attach GT keypoints (normalised [y, x]) to *temp_fitter* at render scale.

    ``SMALFitter`` draws ``target_joints`` in pixels of the rendered image, so
    the normalised dataset keypoints are scaled by the fitter's own image size,
    NOT by ``model.input_resolution`` — scaling by the backbone input resolution
    shrinks the GT markers by ``input_res / native`` and misaligns them (the same
    fix already applied in ``train_smil_regressor.visualize_training_progress``).
    """
    if keypoints_2d is not None and keypoint_visibility is not None:
        pixel_coords = np.asarray(keypoints_2d, dtype=np.float32).copy()
        pixel_coords[:, 0] = pixel_coords[:, 0] * target_size  # y
        pixel_coords[:, 1] = pixel_coords[:, 1] * target_size  # x
        vis = np.asarray(keypoint_visibility, dtype=np.float32).reshape(-1)
        temp_fitter.target_joints = torch.tensor(pixel_coords, dtype=torch.float32, device=device).unsqueeze(0)
        temp_fitter.target_visibility = torch.tensor(vis, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        # No GT keypoints: draw nothing. The marker drawer unpacks
        # ``(bs, nj, 2)`` and indexes config.MARKER_COLORS / MARKER_TYPE by joint
        # id, and the predicted joints it draws are config.CANONICAL_MODEL_JOINTS,
        # so the placeholder must have exactly that many rows (this equals
        # N_POSE + 1 for SMIL models but NOT for the legacy hard-coded body).
        n_joints = len(config.CANONICAL_MODEL_JOINTS)
        temp_fitter.target_joints = torch.zeros((1, n_joints, 2), device=device)
        temp_fitter.target_visibility = torch.zeros((1, n_joints), device=device)


def generate_visualization(
    model: SMILImageRegressor,
    predicted_params: Dict[str, torch.Tensor],
    original_image: np.ndarray,
    image_exporter: InferenceImageExporter,
    image_name: str,
    device: str,
    y_data: Optional[Dict[str, Any]] = None,
    disable_scaling: bool = False,
    disable_translation: bool = False,
) -> None:
    """
    Generate visualization using the SMIL model and predicted parameters.

    Args:
        model: SMILImageRegressor model
        predicted_params: Dictionary of predicted SMIL parameters
        original_image: Original input image
        image_exporter: Image exporter for saving results
        image_name: Base name for the image
        device: PyTorch device
        y_data: Optional target dict (dataset mode) supplying GT keypoints and
                the camera calibration used by camera-centric checkpoints
        disable_scaling: Skip applying log_beta_scales (comparison/debugging)
        disable_translation: Skip applying betas_trans (comparison/debugging)
    """
    try:
        # Create a simplified SMALFitter for visualization
        # Use the original image as RGB input
        if original_image.max() > 1.0:
            rgb_image = original_image.astype(np.float32) / 255.0
        else:
            rgb_image = original_image.astype(np.float32)
        rgb_image = np.clip(rgb_image, 0.0, 1.0)

        # Resize to model's expected input size
        target_size = (model.input_resolution, model.input_resolution)

        if rgb_image.shape[:2] != target_size:
            rgb_image = cv2.resize(rgb_image, target_size)

        # Convert to tensor format expected by SMALFitter
        rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        temp_fitter = _build_render_fitter(
            model,
            predicted_params,
            rgb_tensor,
            device,
            y_data=y_data,
            disable_scaling=disable_scaling,
            disable_translation=disable_translation,
        )

        keypoints_2d = y_data.get("keypoints_2d") if y_data is not None else None
        keypoint_visibility = y_data.get("keypoint_visibility") if y_data is not None else None
        _set_target_joints(temp_fitter, keypoints_2d, keypoint_visibility, device, int(temp_fitter.image_size))

        # Generate visualization with custom image exporter wrapper
        class NamedImageExporter:
            def __init__(self, base_exporter, image_name):
                self.base_exporter = base_exporter
                self.image_name = image_name

            def export(self, collage_np, batch_id, global_id, img_parameters, vertices, faces, img_idx=0, **kwargs):
                # Call the base exporter with the specific image name; forward any
                # extra kwargs (e.g. epoch) from SMALFitter.generate_visualization.
                self.base_exporter.export(
                    collage_np,
                    batch_id,
                    global_id,
                    img_parameters,
                    vertices,
                    faces,
                    img_idx=img_idx,
                    image_name=self.image_name,
                    **kwargs,
                )

        named_exporter = NamedImageExporter(image_exporter, image_name)
        # Apply the predicted per-sample mesh scale (camera_centric); without it the
        # mesh renders at native size (~35x too large vs the metric 3D).
        temp_fitter.generate_visualization(
            named_exporter,
            apply_UE_transform=model.use_ue_scaling,
            img_idx=0,
            mesh_scale=resolve_mesh_scale(model, predicted_params),
        )

        print(f"Generated visualization for {image_name}")

    except Exception as e:
        print(f"Warning: Failed to generate visualization for {image_name}: {e}")
        # Save just the parameters without visualization
        img_parameters = {
            k: v.cpu().data.numpy() if isinstance(v, torch.Tensor) else v for k, v in predicted_params.items()
        }

        # Create a simple visualization showing the original image
        simple_vis = original_image.copy()
        if simple_vis.max() > 1.0:
            simple_vis = (simple_vis).astype(np.uint8)
        else:
            simple_vis = (simple_vis * 255).astype(np.uint8)

        image_exporter.export(
            simple_vis,
            0,
            0,
            img_parameters,
            torch.zeros(1, 1000, 3),
            np.zeros((1000, 3), dtype=int),
            img_idx=0,
            image_name=image_name,
        )


def process_images_batch(
    model: SMILImageRegressor,
    image_files: List[str],
    output_folder: str,
    device: str,
    crop_mode: str = "centred",
    batch_size: int = 1,
    sleap_helper: Optional[SLEAPCroppingHelper] = None,
    sleap_camera: Optional[str] = None,
    disable_scaling: bool = False,
    disable_translation: bool = False,
) -> None:
    """
    Process a batch of images for inference.

    Args:
        model: SMILImageRegressor model
        image_files: List of image file paths
        output_folder: Output folder for results
        device: PyTorch device
        crop_mode: Cropping mode ('centred', 'default', or 'bbox_crop')
        batch_size: Batch size for processing (currently only supports 1)
        sleap_helper: Optional helper for bbox_crop using SLEAP keypoints
        sleap_camera: Optional camera override for bbox_crop
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Create image exporter
    image_exporter = InferenceImageExporter(output_folder)

    print(f"Processing {len(image_files)} images...")
    print(f"Crop mode: {crop_mode}")

    # Process images with progress bar
    for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            # Get image name for output files
            image_name = Path(image_path).stem

            print(f"\nProcessing image {i + 1}/{len(image_files)}: {image_name}")

            if crop_mode == "bbox_crop" and sleap_helper is not None:
                original_image = imageio.v2.imread(image_path)
                if original_image is None:
                    raise RuntimeError(f"Failed to read image: {image_path}")
                preprocess_result = sleap_helper.preprocess_image(
                    original_image, image_path, frame_idx=i, explicit_camera=sleap_camera
                )
                if preprocess_result is None:
                    print("Warning: bbox_crop requested but no keypoints available; falling back to centred crop")
                    preprocessed_image, transform_info = preprocess_frame(
                        original_image, model.input_resolution, crop_mode="centred"
                    )
                else:
                    preprocessed_image, transform_info = preprocess_result
                preprocessed_tensor = torch.from_numpy(preprocessed_image).permute(2, 0, 1).unsqueeze(0)
            else:
                original_image, preprocessed_tensor, transform_info = load_and_preprocess_image(
                    image_path, model, crop_mode
                )

            # Run inference
            predicted_params = run_inference_on_image(model, preprocessed_tensor, device)

            # Generate visualization with unique image name
            generate_visualization(
                model,
                predicted_params,
                original_image,
                image_exporter,
                image_name,
                device,
                disable_scaling=disable_scaling,
                disable_translation=disable_translation,
            )

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    print(f"\nProcessing complete! Results saved to: {output_folder}")


def smooth_camera_parameters(
    predicted_params: Dict[str, torch.Tensor], camera_buffer: Dict[str, List], window_size: int
) -> Dict[str, torch.Tensor]:
    """
    Apply moving average smoothing to camera parameters.

    Args:
        predicted_params: Dictionary of predicted parameters
        camera_buffer: Buffer storing recent camera parameters
        window_size: Size of the moving average window

    Returns:
        Smoothed parameters dictionary
    """
    smoothed_params = predicted_params.copy()

    # Add current predictions to buffers
    camera_buffer["cam_rot"].append(predicted_params["cam_rot"].clone())
    camera_buffer["cam_trans"].append(predicted_params["cam_trans"].clone())
    camera_buffer["fov"].append(predicted_params["fov"].clone())

    # Keep only the last window_size frames
    if len(camera_buffer["cam_rot"]) > window_size:
        camera_buffer["cam_rot"].pop(0)
        camera_buffer["cam_trans"].pop(0)
        camera_buffer["fov"].pop(0)

    # Compute moving average
    if len(camera_buffer["cam_rot"]) > 0:
        smoothed_params["cam_rot"] = torch.stack(camera_buffer["cam_rot"]).mean(dim=0)
        smoothed_params["cam_trans"] = torch.stack(camera_buffer["cam_trans"]).mean(dim=0)
        smoothed_params["fov"] = torch.stack(camera_buffer["fov"]).mean(dim=0)

    return smoothed_params


def process_video(
    model: SMILImageRegressor,
    video_path: str,
    output_folder: str,
    device: str,
    crop_mode: str = "centred",
    fps: Optional[int] = None,
    save_frames: bool = False,
    max_frames: int = -1,
    camera_smoothing_window: int = 10,
    sleap_helper: Optional[SLEAPCroppingHelper] = None,
    sleap_camera: Optional[str] = None,
    video_export_mode: str = "overlay",
    animation_recorder: Optional[AnimationRecorder] = None,
    smoothing_window: int = 0,
    disable_scaling: bool = False,
    disable_translation: bool = False,
) -> None:
    """
    Process a video file for inference.

    Args:
        model: SMILImageRegressor model
        video_path: Path to input video file
        output_folder: Output folder for results
        device: PyTorch device
        crop_mode: Cropping mode ('centred', 'default', or 'bbox_crop')
        fps: Output video FPS (None = same as input)
        save_frames: Whether to save individual frame results
        max_frames: Maximum number of frames to process (-1 for all frames)
        camera_smoothing_window: Number of frames for moving average of camera parameters (default: 10)
        sleap_helper: Optional helper for bbox_crop using SLEAP keypoints
        sleap_camera: Optional camera override when using bbox_crop
        video_export_mode: Export mode ('overlay' or 'side_by_side')
        animation_recorder: Optional recorder for the raw (pre-smoothing) parameters
        smoothing_window: Moving-average window over ALL predicted parameters
            (``PredictionSmoother``, the same smoother run_multiview_inference.py
            uses). When > 0 it supersedes ``camera_smoothing_window``, which only
            averages the camera parameters.
        disable_scaling: Skip applying log_beta_scales when rendering
        disable_translation: Skip applying betas_trans when rendering
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_fps = fps if fps is not None else input_fps

    # Determine how many frames to process
    if max_frames > 0:
        frames_to_process = min(max_frames, total_frames)
    else:
        frames_to_process = total_frames

    print("Video properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  Frames to process: {frames_to_process}")
    print(f"  Input FPS: {input_fps}")
    print(f"  Output FPS: {output_fps}")
    print(f"  Resolution: {frame_width}x{frame_height}")
    print(f"  Crop mode: {crop_mode}")
    print(f"  Camera smoothing window: {camera_smoothing_window} frames")
    print(f"  Video export mode: {video_export_mode}")

    # Determine output video dimensions based on export mode
    if video_export_mode == "side_by_side":
        # Determine render size based on model's input resolution
        render_size = model.input_resolution

        # For side-by-side: input video will be rescaled to match render_size height
        # Output width will be 2 * render_size
        output_height = render_size
        output_width = render_size * 2
    else:
        # For overlay mode: keep original video dimensions
        output_height = frame_height
        output_width = frame_width

    # Create video writer for output
    output_video_path = os.path.join(output_folder, Path(video_path).stem + "_inference.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (output_width, output_height))

    # Initialize moving average buffers for camera parameters
    camera_buffer = {"cam_rot": [], "cam_trans": [], "fov": []}
    # Full-parameter smoother (shared with the multi-view entrypoint). When
    # active it replaces the camera-only moving average so a --smoothing_window
    # run smooths identically on both paths.
    param_smoother = PredictionSmoother(smoothing_window) if smoothing_window > 0 else None
    if param_smoother is not None and camera_smoothing_window > 0:
        print(
            "Note: --smoothing_window is set; it supersedes --camera_smoothing "
            "(all parameters are smoothed, not just the camera)."
        )

    # Optionally create frame exporter
    if save_frames:
        frames_folder = os.path.join(output_folder, "frames")
        os.makedirs(frames_folder, exist_ok=True)
        frame_exporter = InferenceImageExporter(frames_folder)
    else:
        frame_exporter = None

    # Process frames
    frame_idx = 0

    try:
        pbar = tqdm(total=frames_to_process, desc="Processing video")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Stop if we've reached max_frames limit
            if max_frames > 0 and frame_idx >= max_frames:
                break

            try:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Preprocess frame
                target_resolution = model.input_resolution
                if crop_mode == "bbox_crop" and sleap_helper is not None:
                    preprocess_result = sleap_helper.preprocess_image(
                        frame_rgb, video_path, frame_idx=frame_idx, explicit_camera=sleap_camera
                    )
                    if preprocess_result is None:
                        print(
                            f"Warning: Missing SLEAP keypoints for frame {frame_idx}; "
                            "falling back to centred crop for this frame"
                        )
                        preprocessed_image, transform_info = preprocess_frame(
                            frame_rgb, target_resolution, crop_mode="centred"
                        )
                    else:
                        preprocessed_image, transform_info = preprocess_result
                else:
                    preprocessed_image, transform_info = preprocess_frame(frame_rgb, target_resolution, crop_mode)

                # Convert to tensor
                preprocessed_tensor = torch.from_numpy(preprocessed_image).permute(2, 0, 1).unsqueeze(0)

                # Run inference
                predicted_params = run_inference_on_image(model, preprocessed_tensor, device)

                # Record raw (pre-smoothing) parameters for Phase 1 animation export.
                if animation_recorder is not None:
                    animation_recorder.record(predicted_params)

                # Apply smoothing: full-parameter moving average when
                # --smoothing_window is set, otherwise the legacy camera-only one.
                if param_smoother is not None:
                    smoothed_params = param_smoother(predicted_params)
                elif camera_smoothing_window > 0:
                    smoothed_params = smooth_camera_parameters(predicted_params, camera_buffer, camera_smoothing_window)
                else:
                    smoothed_params = predicted_params

                # Process frame based on export mode
                if video_export_mode == "side_by_side":
                    # Render model only
                    rendered_model = render_model_only(model, smoothed_params, device, render_size)

                    # Resize input frame to match render_size
                    input_resized = cv2.resize(frame_rgb, (render_size, render_size))

                    # Create side-by-side visualization
                    side_by_side = np.hstack([input_resized, rendered_model])

                    # Convert RGB to BGR for OpenCV
                    output_frame_bgr = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)
                else:
                    # Render prediction onto frame (using smoothed camera parameters)
                    rendered_frame = render_prediction_on_frame(
                        model, smoothed_params, frame_rgb, device, transform_info=transform_info
                    )

                    # Convert RGB back to BGR for OpenCV
                    output_frame_bgr = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)

                # Write output frame to video
                out.write(output_frame_bgr)

                # Optionally save frame results
                if save_frames and frame_idx % 10 == 0:  # Save every 10th frame
                    frame_name = f"frame_{frame_idx:06d}"
                    try:
                        generate_visualization(
                            model,
                            predicted_params,
                            frame_rgb,
                            frame_exporter,
                            frame_name,
                            device,
                            disable_scaling=disable_scaling,
                            disable_translation=disable_translation,
                        )
                    except Exception as e:
                        print(f"Warning: Failed to save frame {frame_idx}: {e}")

                frame_idx += 1
                pbar.update(1)

            except Exception as e:
                print(f"Warning: Failed to process frame {frame_idx}: {e}")
                # Write original frame on error
                out.write(frame)
                frame_idx += 1
                pbar.update(1)
                continue

        pbar.close()

    finally:
        cap.release()
        out.release()

    print("\nVideo processing complete!")
    print(f"  Output video: {output_video_path}")
    print(f"  Processed {frame_idx} frames")
    if save_frames:
        print(f"  Frame results: {frames_folder}")


# --------------------------------------------------------------------------- #
# Dataset mode (issue #100) - mirrors run_multiview_inference.py
# --------------------------------------------------------------------------- #


def load_inference_dataset(dataset_path: str, model: SMILImageRegressor, conventions: Dict[str, Any]):
    """Open *dataset_path* for single-view inference under the checkpoint's convention.

    ``UnifiedSMILDataset.from_path`` dispatches on the HDF5 ``/metadata`` attrs:
    a multi-view file is opened with ``return_single_view=True`` (plus the
    checkpoint's ``camera_centric`` flag and ``expand_all_views=True``) so every
    calibrated view becomes its own single-view item and the camera the model
    renders through is the one the dataset re-anchored to. A genuine single-view
    file (``SLEAPDataset`` / ``OptimizedSMILDataset``) needs no extra kwargs.

    This is the same resolution ``benchmark_model._run_singleview_benchmark``
    performs, so a benchmark run and an inference run consume identical items.
    """
    from smal_fitter.neuralSMIL.smil_datasets import UnifiedSMILDataset

    extra_kwargs = resolve_singleview_dataset_kwargs(dataset_path, conventions)
    dataset = UnifiedSMILDataset.from_path(
        dataset_path,
        rotation_representation=model.rotation_representation,
        backbone_name=model.backbone_name,
        **extra_kwargs,
    )
    return dataset, extra_kwargs


def check_dataset_model_coherence(dataset, model: SMILImageRegressor, conventions: Dict[str, Any]) -> List[str]:
    """Cross-check the dataset's conventions against the checkpoint's.

    Returns the list of warnings emitted (also printed). These are exactly the
    axes issue #100 calls out: camera intrinsics/extrinsics, the shape-space
    scale/translation parameterisation, and mesh scaling / cropping.
    """
    warnings: List[str] = []

    def warn(msg: str):
        warnings.append(msg)
        print(f"WARNING: {msg}")

    print("\n" + "-" * 60)
    print("DATASET / CHECKPOINT COHERENCE")
    print("-" * 60)

    # --- Mesh scaling -------------------------------------------------------
    dataset_ue = None
    if hasattr(dataset, "get_ue_scaling_flag"):
        try:
            dataset_ue = bool(dataset.get_ue_scaling_flag())
        except Exception:
            dataset_ue = None
    print(f"  mesh placement: use_ue_scaling={model.use_ue_scaling}, allow_mesh_scaling={model.allow_mesh_scaling}")
    if dataset_ue is not None:
        print(f"  dataset UE-scaling convention: {dataset_ue}")
        if dataset_ue != bool(model.use_ue_scaling):
            warn(
                f"dataset expects use_ue_scaling={dataset_ue} but the checkpoint was trained with "
                f"use_ue_scaling={model.use_ue_scaling}; the rendered mesh will be misplaced/mis-scaled."
            )
    if not model.use_ue_scaling and not model.allow_mesh_scaling:
        print("  note: neither UE scaling nor a learned mesh_scale - mesh is placed by translation only.")

    # --- Cropping -----------------------------------------------------------
    crop_mode = getattr(dataset, "crop_mode", None)
    target_resolution = None
    if hasattr(dataset, "get_target_resolution"):
        try:
            target_resolution = int(dataset.get_target_resolution())
        except Exception:
            target_resolution = None
    print(f"  dataset crop_mode: {crop_mode if crop_mode is not None else '(not recorded)'}")
    print(f"  dataset target_resolution: {target_resolution}, model input_resolution: {model.input_resolution}")
    print("  note: dataset frames are already cropped by the preprocessor - --crop_mode is NOT applied here.")

    # --- Camera intrinsics / extrinsics -------------------------------------
    has_cam = bool(getattr(dataset, "has_camera_parameters", False))
    if conventions.get("fixed_camera", False):
        print("  camera: camera_centric - FIXED PyTorch3D identity, vertical FOV from per-sample calibration.")
        if not has_cam:
            warn(
                "camera-centric checkpoint but the dataset carries no camera calibration; "
                "the render falls back to --fov, which will not match the footage."
            )
    else:
        print("  camera: model_centric - the network's predicted cam_rot / cam_trans / fov are used.")
        if has_cam:
            print("  note: dataset ships GT camera parameters; they are used only for the aspect ratio here.")

    world_scale = getattr(dataset, "world_scale", None)
    if world_scale is not None:
        print(f"  dataset world_scale: {world_scale}")

    # --- Shape space --------------------------------------------------------
    mode = getattr(model, "scale_trans_mode", "separate")
    use_pca = None
    if mode == "separate":
        use_pca = TrainingConfig.get_scale_trans_config().get("separate", {}).get("use_pca_transformation", True)
    print(f"  scale_trans_mode: {mode}" + (f" (use_pca_transformation={use_pca})" if use_pca is not None else ""))
    if mode == "separate" and use_pca:
        print("  note: log_beta_scales / betas_trans are PCA weights and are expanded to per-joint values.")
    elif mode == "ignore":
        print("  note: shape-space scale/translation is disabled for this checkpoint.")

    print("-" * 60 + "\n")
    return warnings


def run_forward_singleview(
    model: SMILImageRegressor, x_data: Dict[str, Any], y_data: Dict[str, Any]
) -> Optional[Dict[str, torch.Tensor]]:
    """Run one dataset sample through the model, returning ``predicted_params``.

    Uses ``predict_from_batch`` rather than a bare ``forward`` so the
    camera-centric fixed-camera override sources its FOV from the sample's own
    calibration (``y_data['cam_fov']``) - the same code path training and
    ``benchmark_model.py`` use. Calling ``forward`` directly (as the raw-image
    path must, having no calibration) would silently substitute the CLI ``--fov``.
    """
    if x_data.get("input_image_data") is None:
        return None
    with torch.no_grad():
        result = model.predict_from_batch([x_data], [y_data])
    if result is None or result[0] is None:
        return None
    return result[0]


def render_dataset_sample(
    model: SMILImageRegressor,
    x_data: Dict[str, Any],
    y_data: Dict[str, Any],
    device: str,
    predicted_params: Dict[str, torch.Tensor],
    disable_scaling: bool = False,
    disable_translation: bool = False,
    render_resolution: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Render the SMALFitter collage for one dataset sample (RGB uint8).

    Deliberately mirrors ``run_multiview_inference.render_singleview_collage``:
    same footage handling, same GT-keypoint scaling, same shared parameter
    application, same camera resolution, same mesh placement - so a single-view
    checkpoint and a multi-view checkpoint produce comparable frames.

    The dataset image is used as-is: it is already the crop the preprocessor
    produced, and ``keypoints_2d`` are normalised to that crop, so applying a
    second crop here would break the correspondence.
    """
    image = x_data.get("input_image_data")
    if image is None:
        return None

    target_size = int(render_resolution) if render_resolution else int(getattr(model.renderer, "image_size", 224))

    from PIL import Image

    pil_img = Image.fromarray(_to_uint8_rgb(image))
    pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
    resized_image = np.asarray(pil_img).astype(np.float32) / 255.0
    resized_image = np.clip(resized_image, 0.0, 1.0)
    rgb = torch.from_numpy(resized_image).permute(2, 0, 1).unsqueeze(0).float()

    keypoints_2d = y_data.get("keypoints_2d", None)
    visibility = y_data.get("keypoint_visibility", None)

    if keypoints_2d is not None and visibility is not None:
        pixel_coords = np.asarray(keypoints_2d, dtype=np.float32).copy()
        pixel_coords[:, 0] = pixel_coords[:, 0] * target_size
        pixel_coords[:, 1] = pixel_coords[:, 1] * target_size
        num_joints = pixel_coords.shape[0]
        joints = torch.tensor(pixel_coords.reshape(1, num_joints, 2), dtype=torch.float32)
        vis = torch.tensor(np.asarray(visibility, dtype=np.float32).reshape(1, num_joints), dtype=torch.float32)
        sil = torch.zeros(1, 1, target_size, target_size)
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
    temp_fitter.propagate_scaling = model.propagate_scaling

    _set_target_joints(temp_fitter, keypoints_2d, visibility, device, target_size)

    apply_pose_and_shape(
        temp_fitter,
        model,
        predicted_params,
        index=0,
        disable_scaling=disable_scaling,
        disable_translation=disable_translation,
    )

    cam_rot, cam_trans, fov, aspect = resolve_render_camera(model, predicted_params, y_data, device)
    temp_fitter.fov.data = fov.reshape(-1)[:1].clone()
    temp_fitter.renderer.set_camera_parameters(R=cam_rot, T=cam_trans, fov=fov, aspect_ratio=aspect)

    exporter = InMemoryImageExporter()
    temp_fitter.generate_visualization(
        exporter,
        apply_UE_transform=model.use_ue_scaling,
        img_idx=0,
        mesh_scale=resolve_mesh_scale(model, predicted_params),
    )
    return exporter.image


SV_FRAME_STREAM = "sv"
"""Temp-storage stream name for the single-view render (see inference_ddp)."""


def _probe_frame_size(
    model: SMILImageRegressor,
    dataset,
    device: str,
    render_resolution: Optional[int],
) -> Tuple[int, int]:
    """Determine the output frame size once, identically on every rank.

    Every rank must agree on the video dimensions before any frame is written,
    otherwise ``cv2.VideoWriter`` silently drops the frames whose size differs
    from the one it was opened with. Probing the SAME sample (index 0) on every
    rank guarantees agreement without a collective.
    """
    fallback = int(render_resolution) if render_resolution else int(getattr(model.renderer, "image_size", 224))
    if len(dataset) == 0:
        return (fallback, fallback)
    try:
        x_data, y_data = dataset[0]
        predicted_params = run_forward_singleview(model, x_data, y_data)
        if predicted_params is not None:
            frame = render_dataset_sample(
                model,
                x_data,
                y_data,
                device,
                predicted_params,
                render_resolution=render_resolution,
            )
            if frame is not None:
                return (frame.shape[1], frame.shape[0])
    except Exception as e:
        print(f"Warning: frame-size probe failed ({e}); falling back to {fallback}x{fallback}")
    return (fallback, fallback)


def _export_animation_singleview(
    raw_predictions: List[Tuple[int, Dict[str, Any]]],
    rank: int,
    world_size: int,
    model: SMILImageRegressor,
    checkpoint_path: str,
    dataset_path: str,
    export_path: str,
    fps: float,
) -> None:
    """Gather predictions to rank 0 and write an AMASS-style .npz + .json clip.

    Captures the *raw*, pre-smoothing predictions so downstream consumers
    (Blender addon, etc.) can apply their own smoothing. Mirrors
    ``run_multiview_inference._export_animation``; the recorder builds the
    averaged single-view camera block itself, so there is no per-view camera
    list to assemble here.
    """
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
        rotation_representation=model.rotation_representation,
        fps=fps,
        source_checkpoint=str(checkpoint_path),
        source_input=str(dataset_path),
        model_id=getattr(model, "model_id", None),
    )
    for _, params in all_predictions:
        recorder.record(params)

    written = recorder.write()
    print(f"Animation export written: {written['npz']} + {written['json']} ({recorder.num_frames()} frames)")


def run_dataset_inference_phase(
    model: SMILImageRegressor,
    dataset,
    indices: List[int],
    rank: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    """Run forward passes on the assigned indices; return raw predictions on CPU."""
    model.eval()
    raw_predictions: List[Tuple[int, Dict[str, Any]]] = []

    for global_idx in tqdm(indices, desc="Running inference", disable=(rank != 0)):
        try:
            x_data, y_data = dataset[global_idx]
            predicted_params = run_forward_singleview(model, x_data, y_data)
            if predicted_params is None:
                continue
            raw_predictions.append((global_idx, params_to_cpu(predicted_params)))
        except Exception as e:
            print(f"[Rank {rank}] Error in inference for sample {global_idx}: {e}")
            continue

    print(f"[Rank {rank}] Inference complete: {len(raw_predictions)} predictions")
    return raw_predictions


def run_dataset_render_phase(
    model: SMILImageRegressor,
    dataset,
    device: str,
    smoothed_params: Dict[int, Dict[str, Any]],
    indices: List[int],
    rank: int,
    frame_size: Tuple[int, int],
    args,
    render_resolution: Optional[int],
    frame_exporter: Optional[InferenceImageExporter],
) -> Tuple[List[np.ndarray], List[int]]:
    """Render the assigned indices; return ``(bgr_frames, global_indices)``."""
    frames: List[np.ndarray] = []
    frame_indices: List[int] = []

    for global_idx in tqdm(indices, desc="Rendering visualizations", disable=(rank != 0)):
        if global_idx not in smoothed_params:
            continue
        try:
            x_data, y_data = dataset[global_idx]
            params = params_to_device(smoothed_params[global_idx], device)
            frame = render_dataset_sample(
                model,
                x_data,
                y_data,
                device,
                params,
                disable_scaling=args.disable_scaling,
                disable_translation=args.disable_translation,
                render_resolution=render_resolution,
            )
            if frame is None:
                continue
            frame = pad_or_resize(frame, frame_size)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame_indices.append(global_idx)

            if frame_exporter is not None and global_idx % 10 == 0:
                generate_visualization(
                    model,
                    params,
                    _to_uint8_rgb(x_data["input_image_data"]),
                    frame_exporter,
                    f"sample_{global_idx:06d}",
                    device,
                    y_data=y_data,
                    disable_scaling=args.disable_scaling,
                    disable_translation=args.disable_translation,
                )
        except Exception as e:
            print(f"[Rank {rank}] Error rendering sample {global_idx}: {e}")
            continue

    print(f"[Rank {rank}] Rendering complete: {len(frames)} frames")
    return frames, frame_indices


def process_dataset(
    model: SMILImageRegressor,
    dataset_path: str,
    output_folder: str,
    device: str,
    conventions: Dict[str, Any],
    args,
    checkpoint_path: str,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """Run inference over a preprocessed dataset and write video(s) + exports.

    Phase structure mirrors ``run_multiview_inference.main_inference`` exactly,
    including its multi-GPU behaviour:

      1. inference over this rank's striped index slice,
      1b. optional animation export (raw predictions, gathered to rank 0),
      2. temporal smoothing — gathered across ranks first when smoothing is on,
         because smoothing a rank's striped subset would average frames that are
         ``world_size`` apart in time rather than adjacent ones,
      3. rendering this rank's slice,
      4. video writing (rank 0 merges the per-rank frames back into clip order).
    """
    os.makedirs(output_folder, exist_ok=True)

    dataset, extra_kwargs = load_inference_dataset(dataset_path, model, conventions)
    if rank == 0:
        if extra_kwargs:
            print(f"Opened multi-view dataset in single-view mode: {extra_kwargs}")
        print(f"Dataset size: {len(dataset)} item(s)")
        print(f"World size: {world_size}")
        check_dataset_model_coherence(dataset, model, conventions)

    render_resolution = getattr(args, "render_resolution", None)
    if render_resolution is not None and render_resolution <= 0:
        raise ValueError(f"--render_resolution must be a positive integer, got {render_resolution}")

    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else None
    subclip_ranges = compute_subclip_ranges(
        dataset_size=len(dataset),
        max_frames=max_frames,
        num_subclips=args.generate_num_subclips,
        rank=rank,
    )
    multi_subclip = len(subclip_ranges) > 1

    dataset_name = Path(dataset_path).stem
    smoothing_window = args.smoothing_window
    fps = float(args.fps) if args.fps else 30.0

    # Probed identically on every rank so all frames share one size.
    frame_size = _probe_frame_size(model, dataset, device, render_resolution)
    if rank == 0:
        print(f"Output frame size: {frame_size[0]}x{frame_size[1]}")

    frame_exporter = None
    if args.save_frames:
        frames_folder = os.path.join(output_folder, "frames")
        os.makedirs(frames_folder, exist_ok=True)
        frame_exporter = InferenceImageExporter(frames_folder)

    for clip_idx, (start_idx, end_idx) in enumerate(subclip_ranges):
        if multi_subclip and rank == 0:
            print(f"\n{'#' * 60}")
            print(
                f"# SUBCLIP {clip_idx + 1}/{len(subclip_ranges)}: "
                f"frames [{start_idx}, {end_idx}) ({end_idx - start_idx} frames)"
            )
            print(f"{'#' * 60}")

        range_suffix = f"_frames{start_idx:06d}-{end_idx:06d}" if multi_subclip else ""
        assigned_indices = compute_rank_indices(
            len(dataset),
            rank,
            world_size,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        # -- Phase 1: inference (all ranks in parallel) -----------------------
        if rank == 0:
            print("\n-- Phase 1: Running inference --")
        raw_predictions = run_dataset_inference_phase(model, dataset, assigned_indices, rank)

        # Free GPU memory after inference - rendering reloads params as needed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -- Phase 1b: animation export (raw, pre-smoothing) ------------------
        if args.export_animation:
            _export_animation_singleview(
                raw_predictions=raw_predictions,
                rank=rank,
                world_size=world_size,
                model=model,
                checkpoint_path=checkpoint_path,
                dataset_path=dataset_path,
                export_path=f"{args.export_animation}{range_suffix}",
                fps=fps,
            )

        # -- Phase 2: gather + smooth -----------------------------------------
        temp_base = Path.cwd() / f".sv_inference_temp_{dataset_name}{range_suffix}"

        if world_size > 1 and smoothing_window > 0:
            if rank == 0:
                print(
                    f"\n-- Phase 2: Gathering predictions across {world_size} ranks "
                    f"for smoothing (window={smoothing_window}) --"
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
            smoother = PredictionSmoother(smoothing_window)
            smoothed_params: Dict[int, Dict[str, Any]] = {}
            for global_idx, params in tqdm(
                all_predictions, desc="Applying temporal smoothing", disable=(rank != 0)
            ):
                smoothed_params[global_idx] = smoother(params)
            del raw_predictions, all_predictions

        elif smoothing_window > 0:
            if rank == 0:
                print(f"\n-- Phase 2: Applying temporal smoothing (window={smoothing_window}) --")
            raw_predictions.sort(key=lambda kv: kv[0])
            smoother = PredictionSmoother(smoothing_window)
            smoothed_params = {}
            for global_idx, params in tqdm(
                raw_predictions, desc="Applying temporal smoothing", disable=(rank != 0)
            ):
                smoothed_params[global_idx] = smoother(params)
            del raw_predictions

        else:
            if rank == 0:
                print("\n-- Phase 2: No smoothing (window=0) --")
            raw_predictions.sort(key=lambda kv: kv[0])
            smoothed_params = dict(raw_predictions)
            del raw_predictions

        # -- Phase 3: rendering (all ranks in parallel) ------------------------
        if rank == 0:
            print("\n-- Phase 3: Rendering visualizations --")
        frames, frame_indices = run_dataset_render_phase(
            model=model,
            dataset=dataset,
            device=device,
            smoothed_params=smoothed_params,
            indices=assigned_indices,
            rank=rank,
            frame_size=frame_size,
            args=args,
            render_resolution=render_resolution,
            frame_exporter=frame_exporter,
        )
        del smoothed_params

        # -- Phase 4: write video ---------------------------------------------
        if rank == 0:
            print("\n-- Phase 4: Writing output video --")
        out_path = Path(output_folder) / f"{dataset_name}{range_suffix}_singleview_inference.mp4"

        if world_size > 1:
            if rank == 0:
                temp_base.mkdir(parents=True, exist_ok=True)
            barrier()

            write_frame_streams_to_temp({SV_FRAME_STREAM: (frames, frame_indices)}, temp_base, rank)
            del frames, frame_indices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            barrier()

            if rank == 0:
                merged = merge_frame_streams_from_temp(temp_base, world_size, [SV_FRAME_STREAM])
                entries = merged.get(SV_FRAME_STREAM, [])
                if entries:
                    written = write_video_from_manifest(entries, out_path, fps, frame_size)
                    print(f"Wrote {out_path} ({written} frames)")
                else:
                    print("No frames rendered; no video written.")
                print(f"Cleaning up temporary directory: {temp_base}")
                cleanup_temp_dir(temp_base)

            barrier()
        else:
            if frames:
                written = write_video(frames, out_path, fps, frame_size)
                print(f"Wrote {out_path} ({written} frames)")
            else:
                print("No frames rendered; no video written.")


def prepare_model(args, device: str):
    """Load the checkpoint and apply the CLI overrides every input mode needs."""
    print("\n" + "=" * 40)
    print("Loading model...")
    model, model_config, conventions = load_model_from_checkpoint(args.checkpoint, device)

    # Optional SMAL/SMIL model override on top of the one recorded in the
    # checkpoint. Must run before any dataset construction so config.dd /
    # N_POSE / N_BETAS are correct; load_model_from_checkpoint has already
    # applied the checkpoint's own smal_file at this point.
    if args.smal_file:
        from smal_fitter.neuralSMIL.configs import apply_smal_file_override

        shape_family = args.shape_family if args.shape_family is not None else conventions.get("shape_family")
        print(f"Applying SMAL file override: {args.smal_file} (shape_family={shape_family})")
        apply_smal_file_override(args.smal_file, shape_family=shape_family)

    # Resolve the inference FOV for camera-centric checkpoints. A raw image
    # carries no GT calibration, so the fallback chain is: --fov -> 60.0
    # (the pytorch3d / codebase default). Stashed on the model for the
    # fixed-camera override in run_inference_on_image. In dataset mode the
    # per-sample calibration takes precedence over this value.
    chosen_fov = args.fov if args.fov is not None else 60.0
    model._inference_fov = chosen_fov
    if getattr(model, "fixed_camera", False):
        src = "from --fov" if args.fov is not None else "default"
        print(f"Camera-centric checkpoint: fixed identity camera, FOV={chosen_fov} deg ({src})")

    return model, model_config, conventions


def dataset_main(args, rank: int = 0, world_size: int = 1, device_override: Optional[str] = None) -> None:
    """Load the model and run dataset inference for one rank."""
    if device_override:
        device = device_override
    elif world_size > 1:
        device = f"cuda:{rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _model_config, conventions = prepare_model(args, device)

    print("\n" + "=" * 40)
    print("Running inference on dataset...")
    process_dataset(
        model=model,
        dataset_path=args.dataset,
        output_folder=args.output_folder,
        device=device,
        conventions=conventions,
        args=args,
        checkpoint_path=args.checkpoint,
        rank=rank,
        world_size=world_size,
    )


def ddp_dataset_main(rank: int, world_size: int, args, master_port: str) -> None:
    """DDP wrapper around :func:`dataset_main`.

    Supports two launch modes:
    1. mp.spawn (single-node): rank is passed by spawn, local_rank == rank
    2. torchrun/SLURM (multi-node): environment variables are auto-detected
    """
    rank, world_size, gpu_rank = resolve_launch(rank, world_size)
    setup_ddp(rank, world_size, master_port, local_rank=gpu_rank, timeout_s=getattr(args, "dist_timeout", None))
    try:
        dataset_main(args, rank=rank, world_size=world_size, device_override=f"cuda:{gpu_rank}")
    finally:
        cleanup_ddp()


def main():
    """Main function for the inference script."""
    parser = argparse.ArgumentParser(
        description="Run SMIL single-view inference on a preprocessed dataset, images, or video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a preprocessed dataset (mirrors run_multiview_inference.py)
  python -m smal_fitter.neuralSMIL.run_singleview_inference -c model.pth -d dataset.h5 -o output/
  python -m smal_fitter.neuralSMIL.run_singleview_inference -c model.pth -d dataset.h5 -o output/ \
      --max_frames 300 --smoothing_window 5 --render_resolution 512 --export_animation out/clip

  # Multi-GPU (single node), or via torchrun / SLURM for multi-node
  python -m smal_fitter.neuralSMIL.run_singleview_inference -c model.pth -d dataset.h5 -o output/ --num_gpus 4
  torchrun --nproc_per_node=4 -m smal_fitter.neuralSMIL.run_singleview_inference -c model.pth -d dataset.h5 -o output/

  # Process images
  python -m smal_fitter.neuralSMIL.run_singleview_inference -c model.pth -i images/ -o output/ --crop_mode centred

  # Process video
  python -m smal_fitter.neuralSMIL.run_singleview_inference -c model.pth -v video.mp4 -o output/ --save_frames --fps 30

Supported image formats: jpg, jpeg, png, bmp, tiff, tif (case-insensitive)
Supported video formats: mp4, avi, mov, mkv (anything supported by OpenCV)
Supported dataset formats: .h5 / .hdf5 produced by the SLEAP or replicAnt preprocessors
        """,
    )

    parser.add_argument(
        "-c", "--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pth file)"
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-i", "--input_folder", type=str, help="Path to folder containing input images")
    input_group.add_argument("-v", "--input_video", type=str, help="Path to input video file")
    input_group.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Path to a preprocessed HDF5 dataset (.h5/.hdf5). Mirrors run_multiview_inference.py: "
        "a multi-view dataset is opened in single-view mode under the checkpoint's frame convention, "
        "so camera intrinsics/extrinsics, shape-space scaling, mesh scaling and cropping all follow "
        "the dataset's own convention instead of being re-derived from raw pixels.",
    )

    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to folder for saving results")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for image-folder inference (default: 1).",
    )

    parser.add_argument(
        "--fov",
        type=float,
        default=None,
        help="Vertical field-of-view in degrees for camera-centric checkpoints (fixed identity camera). "
        "Fallback chain: --fov -> 60.0 (pytorch3d / codebase default). "
        "Ignored for legacy model-centric checkpoints (which predict their own camera).",
    )

    # Preprocessing options
    parser.add_argument(
        "--crop_mode",
        type=str,
        default="centred",
        choices=["centred", "default", "bbox_crop"],
        help="Image preprocessing mode: centred=center crop (preserves aspect ratio), "
        "default=direct resize (may distort), bbox_crop=SLEAP-driven bounding box crop. "
        "Should match training preprocessing. (default: centred)",
    )

    # Processing options
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference (default: auto), using cuda if available",
    )

    # Video-specific options
    parser.add_argument("--fps", type=int, default=None, help="Output video FPS (default: same as input)")
    parser.add_argument(
        "--save_frames", action="store_true", help="Save individual frame results when processing video"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="Maximum number of frames to process from video (default: -1 for all frames)",
    )
    parser.add_argument(
        "--camera_smoothing",
        type=int,
        default=0,
        help="Moving average window size for camera parameter smoothing (default: 0, set to 0 to disable)",
    )
    parser.add_argument(
        "--video_export_mode",
        type=str,
        default="overlay",
        choices=["overlay", "side_by_side"],
        help="Video export mode: overlay=blend model onto input (default), "
        "side_by_side=display input and rendered model side by side at same resolution",
    )

    # Dataset-mode options (mirroring run_multiview_inference.py)
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=0,
        help="Number of frames to average ALL predicted parameters over for temporal smoothing "
        "(default: 0, disabled). Same semantics as run_multiview_inference.py --smoothing_window. "
        "Applies to --dataset and --input_video. Distinct from --camera_smoothing, which smooths "
        "only the camera parameters on the legacy video path.",
    )
    parser.add_argument(
        "--generate_num_subclips",
        type=int,
        default=1,
        help="Dataset mode: generate N subclips evenly spaced across the dataset, each --max_frames "
        "long. Each output video / animation export is suffixed with the frame range. "
        "Falls back to a single full-dataset clip if subclips do not fit. Default: 1.",
    )
    parser.add_argument(
        "--disable_scaling",
        action="store_true",
        help="Disable part scaling (log_beta_scales) when rendering, for comparison/debugging",
    )
    parser.add_argument(
        "--disable_translation",
        action="store_true",
        help="Disable part translation (betas_trans) when rendering, for comparison/debugging",
    )
    parser.add_argument(
        "--render_resolution",
        type=int,
        default=None,
        help="Dataset mode: square pixel resolution for the mesh visualization. The mesh is rendered "
        "and the footage interpolated up to match. Default: the renderer's native image_size. "
        "Does NOT affect model inference / backbone input.",
    )
    parser.add_argument(
        "--smal_file",
        type=str,
        default=None,
        help="Path to a SMAL/SMIL model file overriding the one recorded in the checkpoint (optional)",
    )
    parser.add_argument(
        "--shape_family",
        type=int,
        default=None,
        help="Shape family to use with --smal_file (optional, defaults to the checkpoint / config value)",
    )

    # SLEAP-specific options
    parser.add_argument(
        "--sleap_project", type=str, default=None, help="Path to SLEAP project directory (required for bbox_crop)"
    )
    parser.add_argument(
        "--sleap_camera",
        type=str,
        default=None,
        help="Optional camera name override when using bbox_crop with SLEAP data",
    )

    # Animation export (Phase 1)
    parser.add_argument(
        "--export_animation",
        type=str,
        default=None,
        help="Optional output path stem for SMIL animation export. "
        "Writes <stem>.npz + <stem>.json alongside the MP4. "
        "Active for --input_video and --dataset. "
        'NOTE: any string is accepted as-is (e.g. "True" writes True.npz) — '
        "no validation is performed, so pass a real path/filename stem.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SMIL Image Regressor - Inference Script")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    if args.input_folder:
        print(f"Input folder: {args.input_folder}")
    if args.input_video:
        print(f"Input video: {args.input_video}")
    if args.dataset:
        print(f"Dataset: {args.dataset}")
    print(f"Output folder: {args.output_folder}")
    if args.dataset:
        print("Crop mode: (from dataset - --crop_mode is ignored for preprocessed datasets)")
    else:
        print(f"Crop mode: {args.crop_mode}")
    if args.smoothing_window > 0:
        print(f"Temporal smoothing: {args.smoothing_window} frames")
    if args.disable_scaling:
        print("Part scaling: DISABLED (comparison mode)")
    if args.disable_translation:
        print("Part translation: DISABLED (comparison mode)")
    if args.sleap_project:
        print(f"SLEAP project: {args.sleap_project}")
        if args.sleap_camera:
            print(f"SLEAP camera override: {args.sleap_camera}")
    if args.input_video or args.dataset:
        print(f"Save frames: {args.save_frames}")
        if args.fps:
            print(f"Output FPS: {args.fps}")
        if args.max_frames > 0:
            print(f"Max frames: {args.max_frames}")
        else:
            print("Max frames: All frames")
        if args.generate_num_subclips > 1:
            print(f"Subclips: {args.generate_num_subclips} (per-clip length: {args.max_frames})")

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = "cpu"

    print(f"Device: {device}")

    if args.crop_mode == "bbox_crop" and not args.sleap_project and not args.dataset:
        print("Error: bbox_crop mode requires --sleap-project to supply keypoints.")
        return 1

    if args.dataset and not os.path.exists(args.dataset):
        print(f"Error: dataset not found: {args.dataset}")
        return 1

    # ---- Multi-GPU dataset inference -------------------------------------
    # Must branch BEFORE the model is loaded: each rank builds its own model on
    # its own GPU. Only dataset mode is distributed - the raw image/video paths
    # stream from a single decoder and gain nothing from extra ranks.
    master_port = args.master_port or os.environ.get("MASTER_PORT", "12355")

    if args.dataset and is_torchrun_launched():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if rank == 0:
            print("Detected torchrun/HPC launch environment:")
            print(f"  Global rank: {rank}")
            print(f"  Local rank (GPU): {os.environ['LOCAL_RANK']}")
            print(f"  World size: {world_size}")
            print(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'not set')}")
            print(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'not set')}")
        ddp_dataset_main(rank, world_size, args, master_port)
        return 0

    if args.dataset and args.num_gpus > 1:
        try:
            validate_num_gpus(args.num_gpus)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return 1
        print(f"Launching multi-GPU dataset inference on {args.num_gpus} GPUs (using mp.spawn)...")
        print(f"Master port: {master_port}")
        mp.spawn(ddp_dataset_main, args=(args.num_gpus, args, master_port), nprocs=args.num_gpus, join=True)
        print("\n" + "=" * 60)
        print("Inference completed successfully!")
        print(f"Results saved to: {args.output_folder}")
        print("=" * 60)
        return 0

    if args.num_gpus > 1 and not args.dataset:
        print("Warning: --num_gpus > 1 only applies to --dataset mode; running single-process.")

    sleap_helper = None
    try:
        # Load model from checkpoint (shared with the distributed dataset path)
        model, model_config, conventions = prepare_model(args, device)

        if args.crop_mode == "bbox_crop":
            sleap_helper = SLEAPCroppingHelper(
                project_path=args.sleap_project,
                crop_mode=args.crop_mode,
                target_resolution=model.input_resolution,
                backbone_name=model.backbone_name,
            )
            print(f"SLEAP project loaded from {args.sleap_project}")
            print(f"Available SLEAP cameras: {sleap_helper.list_cameras()}")

        # Process based on input type
        if args.dataset:
            print("\n" + "=" * 40)
            print("Running inference on dataset...")
            process_dataset(
                model=model,
                dataset_path=args.dataset,
                output_folder=args.output_folder,
                device=device,
                conventions=conventions,
                args=args,
                checkpoint_path=args.checkpoint,
                rank=0,
                world_size=1,
            )

        elif args.input_folder:
            # Find image files
            print("\n" + "=" * 40)
            print("Finding images...")
            image_files = find_image_files(args.input_folder)

            if len(image_files) == 0:
                print("No image files found in the input folder!")
                return 1

            # Process images
            print("\n" + "=" * 40)
            print("Running inference on images...")
            process_images_batch(
                model,
                image_files,
                args.output_folder,
                device,
                args.crop_mode,
                args.batch_size,
                sleap_helper=sleap_helper,
                sleap_camera=args.sleap_camera,
                disable_scaling=args.disable_scaling,
                disable_translation=args.disable_translation,
            )

        elif args.input_video:
            # Process video
            print("\n" + "=" * 40)
            print("Running inference on video...")

            animation_recorder: Optional[AnimationRecorder] = None
            if args.export_animation:
                output_fps = (
                    args.fps if args.fps is not None else cv2.VideoCapture(args.input_video).get(cv2.CAP_PROP_FPS)
                )
                animation_recorder = build_recorder_from_config(
                    output_path=args.export_animation,
                    rotation_representation=model.rotation_representation,
                    fps=float(output_fps),
                    source_checkpoint=args.checkpoint,
                    source_input=args.input_video,
                    model_id=getattr(model, "model_id", None),
                )
                print(f"Animation export enabled: {args.export_animation}.[npz|json]")

            process_video(
                model,
                args.input_video,
                args.output_folder,
                device,
                args.crop_mode,
                args.fps,
                args.save_frames,
                args.max_frames,
                args.camera_smoothing,
                sleap_helper=sleap_helper,
                sleap_camera=args.sleap_camera,
                video_export_mode=args.video_export_mode,
                animation_recorder=animation_recorder,
                smoothing_window=args.smoothing_window,
                disable_scaling=args.disable_scaling,
                disable_translation=args.disable_translation,
            )

            if animation_recorder is not None and animation_recorder.num_frames() > 0:
                written = animation_recorder.write()
                print(
                    f"Animation export written: {written['npz']} + {written['json']} "
                    f"({animation_recorder.num_frames()} frames)"
                )

        print("\n" + "=" * 60)
        print("Inference completed successfully!")
        print(f"Results saved to: {args.output_folder}")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        print("\nInference interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if sleap_helper is not None:
            sleap_helper.close()


if __name__ == "__main__":
    exit(main())
