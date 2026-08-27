#!/usr/bin/env python3
"""Render an arm's exported poses (``clip_<arm>.npz``) to MP4, with no inference.

Why render from the .npz rather than from video
-----------------------------------------------
``export_poses.py`` already ran every arm over the test split and wrote the
predicted parameters. Rendering those directly means:

  * **the frames you watch are the frames that were scored** — same split, same
    order, same predictions that produced the MPJPE/PCK and violation numbers;
  * every arm renders the identical frame window, so the comparison is honest;
  * no GPU inference, no checkpoint loading, no dataset — just SMAL forward plus
    rasterisation, which is minutes rather than hours.

Re-running inference on raw clips would answer a *different* question (how the
model behaves on those clips) and would not line up with any number in the
tables.

Usage
-----
    python scripts/prior_study/render_clip_npz.py \\
        --npz prior_study_results/sv_reference/clip_sv_reference.npz \\
        --smal-file 3D_model_prep/SMILy_STICK_limits_authored.pkl \\
        --segments-file prior_study_results/renders/segments.json \\
        --out-dir prior_study_results/renders/sv_reference \\
        --label "reference (no limit prior)"

Writes ``<out-dir>/<segment name>.mp4`` per segment.

Cameras
-------
``--camera orbit`` (default) puts a fixed camera on the root-centred mesh: the
predicted translation is dropped, so the animal stays in frame and only its
ARTICULATION varies between arms. That is what a joint-limit prior changes, and
holding the body still is what makes two arms comparable frame by frame.

``--camera predicted`` uses the sidecar's camera and the predicted translation —
the model's own view. Use it to check that the fit still lands on the animal;
it is worse for judging pose because translation drift moves the subject around.

The HUD
-------
Each frame is captioned with the arm label, the frame index, and how many
authored joint-axes are out of range **in that frame** — plus the worst
offender. Without it, two five-panel grids of a stick insect are very hard to
tell apart, and the eye tends to invent differences that the numbers do not
support.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

RAD2DEG = 180.0 / np.pi


# --------------------------------------------------------------------------- io


def load_clip(npz_path: Path, json_path: Optional[Path]) -> Tuple[Dict[str, np.ndarray], dict]:
    sidecar_path = json_path or npz_path.with_suffix(".json")
    if not npz_path.is_file():
        raise SystemExit(f"ERROR: npz not found: {npz_path}")
    if not sidecar_path.is_file():
        raise SystemExit(f"ERROR: sidecar not found: {sidecar_path}")
    with np.load(npz_path) as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, json.loads(sidecar_path.read_text())


def load_limits(smal_file: Path, n_joints: int) -> Optional[np.ndarray]:
    with open(smal_file, "rb") as f:
        dd = pickle.load(f, encoding="latin1")
    jl = dd.get("joint_limits")
    if jl is None:
        return None
    jl = np.asarray(jl, dtype=np.float64)
    if jl.shape != (n_joints, 3, 2):
        print(f"  [warn] joint_limits shape {jl.shape} != {(n_joints, 3, 2)}; HUD violations disabled")
        return None
    return jl


def violation_hud(
    poses: np.ndarray, limits: Optional[np.ndarray], joint_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """Per frame: (#axes out of range, worst overshoot deg, worst joint name, #authored axes)."""
    F, J = poses.shape[0], poses.shape[1]
    if limits is None:
        return np.zeros(F, int), np.zeros(F), [""] * F, 0

    lo, hi = limits[..., 0], limits[..., 1]
    free = (lo <= -np.pi + 1e-4) & (hi >= np.pi - 1e-4)
    mask = ~free
    mask[0] = False
    n_authored = int(mask.sum())
    if n_authored == 0:
        return np.zeros(F, int), np.zeros(F), [""] * F, 0

    jj, aa = np.nonzero(mask)
    p = poses[:, jj, aa]  # (F, n_authored)
    over = np.maximum(np.maximum(lo[jj, aa] - p, 0.0), np.maximum(p - hi[jj, aa], 0.0))
    n_viol = (over > 0).sum(axis=1)
    worst_idx = over.argmax(axis=1)
    worst_deg = over[np.arange(F), worst_idx] * RAD2DEG
    axes = "xyz"
    worst_name = [
        f"{joint_names[jj[k]]}.{axes[aa[k]]}" if worst_deg[i] > 0 else "" for i, k in enumerate(worst_idx)
    ]
    return n_viol, worst_deg, worst_name, n_authored


# ----------------------------------------------------------------------- render


class ClipRenderer:
    """SMAL forward + PyTorch3D colour render, driven by exported parameters.

    The vertex construction mirrors
    ``smil_image_regressor.py:2826-2861`` exactly — same SMAL call with
    ``betas_logscale`` / ``betas_trans`` / ``propagate_scaling``, same
    root-centred mesh_scale branch — so what is rendered is what was scored.
    """

    def __init__(self, smal_file: str, image_size: int, device: str, shape_family: Optional[int] = None) -> None:
        from smal_fitter.neuralSMIL.configs.config_utils import apply_smal_file_override

        apply_smal_file_override(smal_file, shape_family=shape_family)

        import torch
        import config as smil_config
        from smal_model.smal_torch import SMAL
        from smal_fitter.p3d_renderer import Renderer

        self.torch = torch
        self.config = smil_config
        self.device = torch.device(device)
        self.smal = SMAL(self.device, shape_family_id=shape_family if shape_family is not None else -1)
        self.smal.eval()
        self.renderer = Renderer(image_size, self.device)
        self.image_size = image_size
        self.faces = self.smal.faces.long()

    def forward_batch(
        self,
        poses: np.ndarray,  # (B, J, 3) axis-angle, row 0 = global rotation
        betas: np.ndarray,  # (B, n_betas)
        trans: np.ndarray,  # (B, 3)
        log_beta_scales: Optional[np.ndarray],
        betas_trans: Optional[np.ndarray],
        mesh_scale: Optional[np.ndarray],
        scaling: str,
        root_centred: bool,
    ):
        torch = self.torch
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=self.device)  # noqa: E731

        theta = t(poses)  # (B, J, 3) — SMAL takes root + posable joints together
        beta_t = t(betas)
        with torch.no_grad():
            verts, joints, _, _ = self.smal(
                beta_t,
                theta,
                betas_logscale=t(log_beta_scales) if log_beta_scales is not None else None,
                betas_trans=t(betas_trans) if betas_trans is not None else None,
                propagate_scaling=True,
            )

        root = joints[:, 0:1, :]
        if root_centred:
            # Drop the predicted translation entirely: the body sits at the
            # origin and only articulation moves. Scale still applies so the
            # animal does not change size between frames.
            if scaling == "mesh_scale" and mesh_scale is not None:
                verts = (verts - root) * t(mesh_scale).reshape(-1, 1, 1)
            elif scaling == "ue":
                verts = (verts - root) * 10.0
            else:
                verts = verts - root
        else:
            trans_t = t(trans).unsqueeze(1)
            if scaling == "mesh_scale" and mesh_scale is not None:
                verts = (verts - root) * t(mesh_scale).reshape(-1, 1, 1) + trans_t
            elif scaling == "ue":
                verts = (verts - root) * 10.0 + trans_t
            else:
                verts = verts + trans_t
        return verts

    def render_batch(self, verts, R, T, fov) -> np.ndarray:
        """(B, H, W, 3) uint8 RGB."""
        torch = self.torch
        B = verts.shape[0]
        self.renderer.set_camera_parameters(
            R=R.expand(B, 3, 3).contiguous(),
            T=T.expand(B, 3).contiguous(),
            fov=fov.expand(B).contiguous(),
        )
        faces = self.faces.unsqueeze(0).expand(B, -1, -1)
        with torch.no_grad():
            _, _, color = self.renderer(verts, verts, faces, render_texture=True)
        img = color.permute(0, 2, 3, 1).clamp(0, 1).detach().cpu().numpy()
        return (img * 255.0).astype(np.uint8)


def orbit_camera(radius: float, elev: float, azim: float, fov: float, device, torch):
    from pytorch3d.renderer import look_at_view_transform

    R, T = look_at_view_transform(dist=radius, elev=elev, azim=azim, device=device)
    return R, T, torch.tensor([fov], dtype=torch.float32, device=device)


def draw_hud(frame: np.ndarray, lines: List[str]) -> np.ndarray:
    """Burn a small text block into the top-left corner. cv2 only, no fonts needed."""
    import cv2

    out = frame.copy()
    pad, lh, fs, th = 6, 18, 0.5, 1
    w = max((cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][0] for s in lines), default=0)
    box_h = pad * 2 + lh * len(lines)
    box_w = min(out.shape[1], w + pad * 2)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    for i, s in enumerate(lines):
        cv2.putText(
            out, s, (pad, pad + lh * (i + 1) - 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA
        )
    return out


# ------------------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True, type=Path, help="This arm's clip_*.npz")
    p.add_argument("--json", default=None, type=Path, help="Its sidecar (default: same stem, .json)")
    p.add_argument("--smal-file", default="3D_model_prep/SMILy_STICK_limits_authored.pkl", type=Path)
    p.add_argument("--shape-family", type=int, default=None)
    p.add_argument("--segments-file", required=True, type=Path, help="From pick_segments.py — SHARED by every arm")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--label", default=None, help="HUD caption for this arm (default: the npz stem)")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16, help="Frames per SMAL/raster call")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--fps", type=float, default=None, help="Override the sidecar fps")
    p.add_argument(
        "--camera",
        choices=["orbit", "predicted"],
        default="orbit",
        help="orbit = fixed camera on the root-centred mesh (default; isolates articulation). "
        "predicted = the sidecar camera with the predicted translation (the model's own view).",
    )
    p.add_argument("--elev", type=float, default=15.0, help="orbit camera elevation, degrees")
    p.add_argument("--azim", type=float, default=45.0, help="orbit camera azimuth, degrees")
    p.add_argument(
        "--orbit-fov", type=float, default=45.0, help="orbit camera vertical FOV, degrees (default: 45)"
    )
    p.add_argument(
        "--margin", type=float, default=1.35, help="orbit distance = mesh radius x this / tan(fov/2) (default: 1.35)"
    )
    p.add_argument(
        "--scaling",
        choices=["auto", "mesh_scale", "ue", "none"],
        default="auto",
        help="How the trainer turned SMAL output into world vertices. auto = mesh_scale when the "
        "npz carries it, else none. Use 'ue' for a use_ue_scaling checkpoint (10x about the root) "
        "— the npz cannot record which branch was used, so if a 'predicted' render comes out at "
        "visibly the wrong size, this is the knob.",
    )
    p.add_argument("--no-hud", action="store_true", help="Do not burn the caption/violation block in")
    args = p.parse_args()

    import cv2
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    arrays, sidecar = load_clip(args.npz, args.json)
    poses = arrays["poses"]  # (F, J, 3)
    F, J = poses.shape[0], poses.shape[1]
    betas_pf = arrays.get("betas_per_frame")
    betas_avg = arrays["betas"]
    trans = arrays["trans"]
    lbs = arrays.get("log_beta_scales")
    btr = arrays.get("betas_trans")
    msc = arrays.get("mesh_scale")
    fps = float(args.fps or sidecar.get("fps", 30.0))
    joint_names = sidecar.get("joint_names", [f"J{i}" for i in range(J)])
    label = args.label or args.npz.stem

    scaling = args.scaling
    if scaling == "auto":
        scaling = "mesh_scale" if msc is not None else "none"

    segs_doc = json.loads(args.segments_file.read_text())
    segments = segs_doc["segments"]

    print("=" * 66)
    print(f" render {args.npz.name}  ->  {args.out_dir}")
    print(f"   label     : {label}")
    print(f"   frames    : {F}, joints {J}, fps {fps}")
    print(f"   source    : {sidecar.get('source_checkpoint')}")
    print(f"   device    : {device}   image_size {args.image_size}   camera {args.camera}")
    print(f"   scaling   : {scaling}" + ("  (mesh_scale present in npz)" if msc is not None else ""))
    print(f"   segments  : {len(segments)} from {args.segments_file}")
    print("=" * 66)

    # The windows come from the reference arm's clip. If this arm is shorter,
    # the windows do not address the same frames and the comparison is void.
    if segs_doc.get("n_frames") not in (None, F):
        raise SystemExit(
            f"ERROR: segments.json was built on a clip of {segs_doc['n_frames']} frames but this\n"
            f"       npz has {F}. The two arms did not export the same test split, so frame i is\n"
            f"       not the same instant in both. Re-run preflight_study.py: the arms' seed /\n"
            f"       train_ratio / val_ratio disagree."
        )

    limits = load_limits(args.smal_file, J)
    n_viol, worst_deg, worst_name, n_authored = violation_hud(poses, limits, joint_names)

    renderer = ClipRenderer(str(args.smal_file), args.image_size, device, args.shape_family)

    # --- fixed orbit camera, framed on the whole selected range ---------------
    # One distance for every segment and every arm, computed from the union of
    # the rendered frames: a per-frame or per-arm auto-fit would zoom differently
    # between panels and make identical poses look different.
    R = T = fov = None
    if args.camera == "orbit":
        probe_idx = []
        for s in segments:
            st, ln = int(s["start"]), int(s["length"])
            probe_idx.extend(range(st, min(st + ln, F), max(ln // 16, 1)))
        probe_idx = probe_idx[:256] or [0]
        v = renderer.forward_batch(
            poses[probe_idx],
            (betas_pf[probe_idx] if betas_pf is not None else np.repeat(betas_avg[None], len(probe_idx), 0)),
            trans[probe_idx],
            lbs[probe_idx] if lbs is not None else None,
            btr[probe_idx] if btr is not None else None,
            msc[probe_idx] if msc is not None else None,
            scaling,
            root_centred=True,
        )
        radius = float(v.norm(dim=-1).max().item())
        dist = radius * args.margin / np.tan(np.deg2rad(args.orbit_fov) / 2.0)
        R, T, fov = orbit_camera(dist, args.elev, args.azim, args.orbit_fov, renderer.device, torch)
        print(f"   orbit     : mesh radius {radius:.4f}, distance {dist:.4f}, fov {args.orbit_fov}")
    else:
        cams = sidecar.get("cameras") or []
        if not cams:
            raise SystemExit("ERROR: --camera predicted but the sidecar has no cameras block.")
        cam = cams[0]
        R = torch.tensor(cam["R"], dtype=torch.float32, device=renderer.device).reshape(1, 3, 3)
        T = torch.tensor(cam["t"], dtype=torch.float32, device=renderer.device).reshape(1, 3)
        fov = torch.tensor([float(cam["fov"])], dtype=torch.float32, device=renderer.device)
        print(f"   camera    : {cam.get('view_name')} fov {cam['fov']:.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for seg in segments:
        name, start, length = seg["name"], int(seg["start"]), int(seg["length"])
        end = min(start + length, F)
        out_path = args.out_dir / f"{name}.mp4"
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.image_size, args.image_size)
        )
        if not writer.isOpened():
            raise SystemExit(f"ERROR: could not open VideoWriter for {out_path}")

        print(f"\n  {name}: frames {start}..{end}")
        for b0 in range(start, end, args.batch_size):
            b1 = min(b0 + args.batch_size, end)
            sl = slice(b0, b1)
            verts = renderer.forward_batch(
                poses[sl],
                (betas_pf[sl] if betas_pf is not None else np.repeat(betas_avg[None], b1 - b0, 0)),
                trans[sl],
                lbs[sl] if lbs is not None else None,
                btr[sl] if btr is not None else None,
                msc[sl] if msc is not None else None,
                scaling,
                root_centred=(args.camera == "orbit"),
            )
            frames = renderer.render_batch(verts, R, T, fov)
            for k, f_idx in enumerate(range(b0, b1)):
                frame = cv2.cvtColor(frames[k], cv2.COLOR_RGB2BGR)
                if not args.no_hud:
                    lines = [label, f"frame {f_idx}"]
                    if n_authored:
                        lines.append(f"out of range: {int(n_viol[f_idx])}/{n_authored} axes")
                        if worst_name[f_idx]:
                            lines.append(f"worst: {worst_name[f_idx]} +{worst_deg[f_idx]:.1f} deg")
                    frame = draw_hud(frame, lines)
                writer.write(frame)
            print(f"    {b1 - start}/{end - start} frames", end="\r", flush=True)
        writer.release()
        seg_viol = n_viol[start:end]
        print(f"\n    -> {out_path}  (mean {seg_viol.mean():.2f} of {n_authored} axes out of range)")
        written.append(out_path)

    # Provenance: a folder of MP4s is indistinguishable from any other folder of
    # MP4s three weeks later.
    (args.out_dir / "render.json").write_text(
        json.dumps(
            {
                "label": label,
                "npz": str(args.npz),
                "source_checkpoint": sidecar.get("source_checkpoint"),
                "segments_file": str(args.segments_file),
                "camera": args.camera,
                "scaling": scaling,
                "image_size": args.image_size,
                "fps": fps,
                "n_authored_axes": n_authored,
                "segments": [
                    {
                        "name": s["name"],
                        "start": int(s["start"]),
                        "length": int(s["length"]),
                        "mean_axes_out_of_range": float(
                            n_viol[int(s["start"]) : min(int(s["start"]) + int(s["length"]), F)].mean()
                        ),
                    }
                    for s in segments
                ],
            },
            indent=2,
        )
    )

    print(f"\n  wrote {len(written)} clip(s) + render.json into {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
