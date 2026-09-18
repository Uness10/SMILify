#!/usr/bin/env python3
"""
Diagnose "keypoints look right but the mesh does not render" in single-view viz.

For a few samples it runs the checkpoint exactly the way
train_smil_regressor.visualize_training_progress renders them (fixed identity
camera + GT FOV for camera_centric, predicted mesh_scale, predicted per-joint
scales/translations) and prints, per sample:

  * mesh_scale, trans, FOV actually used
  * the largest |log_beta_scales| entries (joint, axis, value) and |betas|
  * 3D extent of the vertices vs. the joints (rest model: ~1.06)
  * fraction of vertices at or behind the camera plane (z <= 0)
  * silhouette coverage of the rendered image (1.0 == mesh fills the frame)
  * the same numbers with log_beta_scales / betas_trans / betas zeroed

If the "zeroed" row renders a sane silhouette while the "predicted" row fills
the frame, the render path is fine and the model's predicted per-joint scales
(which the 2D keypoint loss cannot see) are what blew the mesh up.

Usage (repo root, same env as training):
    python smal_fitter/neuralSMIL/diagnose_mesh_render.py \
        --checkpoint $RUNS_ROOT/singleview_2d_lam0/checkpoints/checkpoint_epoch_0178.pth \
        --data_path SMILySTICKS_centred_reprojected_FIXED.h5 --num_samples 5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch

from smal_fitter.neuralSMIL.run_singleview_inference import load_model_from_checkpoint
from smal_fitter.neuralSMIL.smil_image_regressor import rotation_6d_to_axis_angle
from smal_fitter.neuralSMIL.smil_datasets import UnifiedSMILDataset
import config


def _per_joint(model, params):
    """Return (log_beta_scales, betas_trans) as (B, J, 3) whatever the head outputs."""
    s = params.get("log_beta_scales")
    t = params.get("betas_trans")
    if s is None or t is None:
        return s, t
    if model.scale_trans_mode == "separate" and s.dim() == 2:
        s, t = model._transform_separate_pca_weights_to_joint_values(s, t)
    return s, t


@torch.no_grad()
def build_and_render(model, params, log_s, trans_j, betas, fov, device):
    if model.rotation_representation == "6d":
        g = rotation_6d_to_axis_angle(params["global_rot"])
        j = rotation_6d_to_axis_angle(params["joint_rot"])
    else:
        g, j = params["global_rot"], params["joint_rot"]
    verts, joints, _, _ = model.smal_model(
        betas,
        torch.cat([g.unsqueeze(1), j], dim=1),
        betas_logscale=log_s,
        betas_trans=trans_j,
        propagate_scaling=model.propagate_scaling,
    )
    trans = params["trans"]
    if model.use_ue_scaling:
        root = joints[:, 0:1]
        verts = (verts - root) * 10 + trans.unsqueeze(1)
        joints = (joints - root) * 10 + trans.unsqueeze(1)
    elif getattr(model, "allow_mesh_scaling", False) and "mesh_scale" in params:
        root = joints[:, 0:1]
        ms = params["mesh_scale"].reshape(-1, 1, 1)
        verts = (verts - root) * ms + trans.unsqueeze(1)
        joints = (joints - root) * ms + trans.unsqueeze(1)
    else:
        verts = verts + trans.unsqueeze(1)
        joints = joints + trans.unsqueeze(1)

    model.renderer.set_camera_parameters(R=params["cam_rot"], T=params["cam_trans"], fov=fov)
    faces = model.smal_model.faces.unsqueeze(0).expand(verts.shape[0], -1, -1)
    sil, proj = model.renderer(verts.float(), joints[:, config.CANONICAL_MODEL_JOINTS].float(), faces)

    # depth in camera view space
    z = model.renderer.cameras.get_world_to_view_transform().transform_points(verts.float())[..., 2]
    ext = lambda x: (x.amax(1) - x.amin(1)).amax(-1)  # noqa: E731
    return {
        "vert/joint extent": (ext(verts) / ext(joints).clamp_min(1e-12)).item(),
        "verts z<=0": (z <= 0).float().mean().item(),
        "sil coverage": (sil > 0.5).float().mean().item(),
        "proj joints in frame": (
            ((proj >= 0) & (proj <= model.renderer.image_size)).all(-1).float().mean().item()
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, _ = load_model_from_checkpoint(args.checkpoint, args.device)
    model.eval()
    ckpt_cfg = torch.load(args.checkpoint, map_location="cpu").get("config", {})
    camera_centric = ckpt_cfg.get("frame_convention", "model_centric") == "camera_centric"
    print(
        f"\nscale_trans_mode={model.scale_trans_mode}  fixed_camera={model.fixed_camera}  "
        f"allow_mesh_scaling={getattr(model, 'allow_mesh_scaling', False)}  "
        f"use_ue_scaling={model.use_ue_scaling}  propagate_scaling={model.propagate_scaling}  "
        f"positive_depth={getattr(model, 'positive_depth', False)} "
        f"depth_bounds={getattr(model, 'min_depth', None)}..{getattr(model, 'max_depth', None)}"
    )

    kw = {}
    if camera_centric or ckpt_cfg.get("from_multiview"):
        kw = dict(return_single_view=True, camera_centric=camera_centric, expand_all_views=True, augment=False)
    ds = UnifiedSMILDataset.from_path(
        args.data_path,
        rotation_representation=model.rotation_representation,
        backbone_name=model.backbone_name,
        **kw,
    )
    names = list(config.dd["J_names"]) if hasattr(config, "dd") and "J_names" in config.dd else None
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=min(args.num_samples, len(ds)), replace=False)

    for n, idx in enumerate(idxs):
        x, y = ds[int(idx)]
        img = model.preprocess_image(x["input_image_data"]).to(args.device)
        with torch.no_grad():
            p = model(img)
        bs = p["global_rot"].shape[0]
        if model.fixed_camera:
            p["cam_rot"] = torch.eye(3, device=args.device).unsqueeze(0).expand(bs, 3, 3).contiguous()
            p["cam_trans"] = torch.zeros(bs, 3, device=args.device)
            gt_fov = y.get("cam_fov")
            gt_fov = gt_fov[0] if isinstance(gt_fov, (list, tuple, np.ndarray)) else gt_fov
            fov = torch.tensor([float(gt_fov) if gt_fov is not None else 60.0], device=args.device)
        else:
            fov = p["fov"].reshape(-1)

        log_s, trans_j = _per_joint(model, p)
        print(f"\n=== sample {n} (dataset idx {idx}) ===")
        ms = p.get("mesh_scale")
        print(f"  mesh_scale={None if ms is None else ms.flatten().tolist()}  trans={p['trans'].flatten().tolist()}"
              f"  fov={fov.tolist()}")
        if model.fixed_camera and float(p["trans"][0, 2]) <= 0:
            print("  !! depth <= 0: the prediction is BEHIND the camera (docs/ISSUE_2d_only_depth_flip.md)")
        print(f"  |betas| max={p['betas'].abs().max().item():.3f}  "
              f"|betas_trans| max={0 if trans_j is None else trans_j.abs().max().item():.4f}")
        if log_s is not None:
            flat = log_s[0].reshape(-1)
            top = flat.abs().argsort(descending=True)[:6]
            desc = [
                f"{names[int(t) // 3] if names else int(t) // 3}.{'xyz'[int(t) % 3]}={flat[t].item():+.2f}"
                for t in top
            ]
            print(f"  |log_beta_scales| max={flat.abs().max().item():.2f} rms={flat.pow(2).mean().sqrt().item():.2f}"
                  f"  top: {', '.join(desc)}")

        rows = {
            "predicted": (log_s, trans_j, p["betas"]),
            "scales=0": (torch.zeros_like(log_s) if log_s is not None else None, trans_j, p["betas"]),
            "scales,trans,betas=0": (
                torch.zeros_like(log_s) if log_s is not None else None,
                torch.zeros_like(trans_j) if trans_j is not None else None,
                torch.zeros_like(p["betas"]),
            ),
        }
        for label, (s_, t_, b_) in rows.items():
            r = build_and_render(model, p, s_, t_, b_, fov, args.device)
            print(f"  {label:>22}: " + "  ".join(f"{k}={v:.3f}" for k, v in r.items()))


if __name__ == "__main__":
    main()
