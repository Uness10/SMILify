#!/usr/bin/env python3
"""Export predicted poses from a SINGLE-VIEW *or* MULTI-VIEW checkpoint over an
HDF5 test split.

Why this exists
---------------
``scripts/prior_study/analyze_baseline_pose.py`` eats the ``.npz`` + ``.json``
pair produced by ``AnimationRecorder``. Neither inference entry point can
produce that pair from an HDF5 test split:

  * ``run_singleview_inference --export_animation`` only accepts
    ``--input_folder`` / ``--input_video``;
  * ``run_multiview_inference --export_animation`` likewise takes a folder, not
    an HDF5.

This script closes that gap for both modalities, so the study analyses **exactly
the frames the benchmark scored**. It reuses ``benchmark_model``'s model
builders and split logic rather than reimplementing them, so the exported poses
correspond 1:1 to the MPJPE/PCK numbers in the benchmark report.

The modality is detected from the checkpoint (``view_embeddings.weight`` in the
state dict => multi-view), never guessed from the filename.

Usage
-----
    python scripts/prior_study/export_poses.py \
        --checkpoint runs/singleview_constrained/checkpoints/best_model.pth \
        --dataset_path SMILySTICKS_centred_reprojected_FIXED.h5 \
        --smal-file 3D_model_prep/SMILy_STICK_limits_authored.pkl \
        --out prior_study_results/singleview_constrained/clip \
        --max-frames 0        # 0 = whole test split

Writes ``<out>.npz`` (poses (F,J,3) axis-angle, trans, betas, ...) and
``<out>.json`` (joint names, parents, provenance).

Determinism note (multi-view)
-----------------------------
``SLEAPMultiViewDataset`` can sample which views it hands the model. Random view
sampling would make the exported poses depend on RNG state, so the two arms of
the study would differ for a reason that has nothing to do with joint limits.
This script forces ``random_view_sampling=False`` unless ``--random-view-sampling``
is passed explicitly.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import config  # noqa: E402

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", config.GPU_IDS)

import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from smal_fitter.neuralSMIL.animation_export import build_recorder_from_config  # noqa: E402
from smal_fitter.neuralSMIL.benchmark_model import (  # noqa: E402
    _create_multiview_model,
    _create_singleview_model,
    _detect_model_type,
)
from smal_fitter.neuralSMIL.smil_datasets import UnifiedSMILDataset  # noqa: E402
from smal_fitter.neuralSMIL.train_smil_regressor import (  # noqa: E402
    custom_collate_fn,
    set_random_seeds,
)


# --------------------------------------------------------------------- splits


def build_singleview_test_split(dataset, sv_config, camera_centric: bool):
    """Reproduce ``benchmark_model``'s single-view test split exactly.

    Two paths, mirroring ``_run_singleview_benchmark``:
      * camera-centric + ``item_sample_indices`` present -> sample-grouped split
        (so all views of a sample land in the same split, as in training);
      * otherwise -> plain item-level ``random_split``.
    Both use ``sv_config["seed"]`` and the config's train/val ratios.
    """
    if camera_centric and getattr(dataset, "item_sample_indices", None) is not None:
        n_samples = int(dataset.num_samples)
        n_train = int(n_samples * sv_config["train_ratio"])
        n_val = int(n_samples * sv_config["val_ratio"])
        n_test = n_samples - n_train - n_val
        _, _, sample_test = torch.utils.data.random_split(
            range(n_samples),
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(sv_config["seed"]),
        )
        test_samples = {int(s) for s in sample_test}
        isi = dataset.item_sample_indices
        test_idx = [i for i, s in enumerate(isi) if int(s) in test_samples]
        print(f"  split: camera-centric sample-grouped, seed={sv_config['seed']}")
        print(f"         {n_train}/{n_val}/{n_test} samples -> {len(test_idx)} test view-items")
        return Subset(dataset, test_idx)

    total = len(dataset)
    n_train = int(total * sv_config["train_ratio"])
    n_val = int(total * sv_config["val_ratio"])
    n_test = total - n_train - n_val
    _, _, test_set = torch.utils.data.random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(sv_config["seed"]),
    )
    # random_split hands back indices in shuffled order. Same *set* of frames as
    # the benchmark either way, but sorting them restores dataset (i.e. temporal)
    # order so the trajectory plots in analyze_baseline_pose.py mean something.
    test_set = Subset(dataset, sorted(int(i) for i in test_set.indices))
    print(f"  split: item-level, seed={sv_config['seed']} -> {len(test_set)} test items (re-sorted)")
    return test_set


def build_multiview_test_split(dataset, mv_config):
    """Reproduce ``_run_multiview_benchmark``'s test split exactly.

    That function does a plain three-way ``random_split`` over the whole dataset
    with ``torch.Generator().manual_seed(config["seed"])`` — same call, same
    seed, same ratios, so the same *set* of samples comes back. Indices are
    re-sorted afterwards for temporal order in the trajectory plots.
    """
    total = len(dataset)
    n_train = int(total * mv_config["train_ratio"])
    n_val = int(total * mv_config["val_ratio"])
    n_test = total - n_train - n_val
    _, _, test_set = torch.utils.data.random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(mv_config["seed"]),
    )
    test_set = Subset(dataset, sorted(int(i) for i in test_set.indices))
    print(f"  split: multi-view sample-level, seed={mv_config['seed']} -> {len(test_set)} test samples (re-sorted)")
    return test_set


def slice_params(predicted_params: dict, i: int) -> dict:
    """Take sample ``i`` out of a batched param dict, keeping a leading dim of 1.

    ``AnimationRecorder.record`` indexes ``[0]`` of every entry, i.e. it records
    a single frame per call, so batched predictions must be fed one at a time.
    """
    out = {}
    for key, value in predicted_params.items():
        if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] > i:
            out[key] = value[i : i + 1].detach().cpu()
        else:
            out[key] = value
    return out


# ------------------------------------------------------------------ modality


def setup_singleview(args, ckpt, device):
    """Return ``(model, loader, run_config)`` for a single-view checkpoint."""
    model, sv_config = _create_singleview_model(
        ckpt,
        device,
        smal_file_override=args.smal_file,
        shape_family_override=args.shape_family,
        log_fn=print,
    )
    set_random_seeds(sv_config["seed"])

    camera_centric = bool(sv_config.get("camera_centric", False)) and bool(sv_config.get("from_multiview", False))
    sv_from_mv_kwargs = (
        dict(return_single_view=True, camera_centric=True, expand_all_views=True) if camera_centric else {}
    )
    dataset = UnifiedSMILDataset.from_path(
        args.dataset_path,
        rotation_representation=sv_config["rotation_representation"],
        backbone_name=sv_config["backbone_name"],
        **sv_from_mv_kwargs,
    )
    print(f"Dataset size: {len(dataset)}  (camera_centric={camera_centric})")

    if args.split == "all":
        eval_set = dataset
        print("  split: ALL items (not comparable to the benchmark test numbers)")
    else:
        eval_set = build_singleview_test_split(dataset, sv_config, camera_centric)

    batch_size = args.batch_size if args.batch_size is not None else sv_config["batch_size"]
    loader = DataLoader(
        eval_set,
        batch_size=batch_size,
        shuffle=False,  # keep dataset order so trajectory plots are meaningful
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    return model, loader, sv_config


def setup_multiview(args, ckpt, device):
    """Return ``(model, loader, run_config)`` for a multi-view checkpoint."""
    # Imported lazily: these pull in the multi-view dataset stack, which the
    # single-view path has no reason to load.
    from smal_fitter.neuralSMIL.configs import apply_smal_file_override
    from smal_fitter.neuralSMIL.train_multiview_regressor import MultiViewTrainingConfig
    from smal_fitter.sleap_data.sleap_multiview_dataset import (
        SLEAPMultiViewDataset,
        multiview_collate_fn,
    )

    mv_config = dict(ckpt.get("config") or {})
    if not mv_config:
        mv_config = dict(MultiViewTrainingConfig.get_config())
        print("  [warn] checkpoint carries no config block; falling back to MultiViewTrainingConfig defaults")

    smal_file = args.smal_file or mv_config.get("smal_file")
    if not smal_file or not os.path.exists(smal_file):
        raise SystemExit(
            f"ERROR: cannot resolve the SMAL model file.\n"
            f"  --smal-file      : {args.smal_file or '(not given)'}\n"
            f"  checkpoint config: {mv_config.get('smal_file', '(not stored)')}\n"
            f"Pass --smal-file explicitly."
        )
    shape_family = args.shape_family if args.shape_family is not None else mv_config.get("shape_family")
    apply_smal_file_override(smal_file, shape_family=shape_family)
    print(f"  smal file: {smal_file}")

    if args.batch_size is not None:
        mv_config["batch_size"] = args.batch_size
    mv_config["num_workers"] = args.num_workers
    set_random_seeds(int(mv_config.get("seed", 0)))

    dataset = SLEAPMultiViewDataset(
        hdf5_path=args.dataset_path,
        rotation_representation=mv_config["rotation_representation"],
        num_views_to_use=mv_config.get("num_views_to_use"),
        random_view_sampling=bool(args.random_view_sampling),
    )
    print(f"Dataset size: {len(dataset)}  (random_view_sampling={bool(args.random_view_sampling)})")

    if args.split == "all":
        eval_set = dataset
        print("  split: ALL samples (not comparable to the benchmark test numbers)")
    else:
        eval_set = build_multiview_test_split(dataset, mv_config)

    model, _input_resolution = _create_multiview_model(
        checkpoint=ckpt,
        checkpoint_path=args.checkpoint,
        dataset=dataset,
        config_from_ckpt=mv_config,
        device=device,
        log_fn=print,
    )

    loader = DataLoader(
        eval_set,
        batch_size=mv_config["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=mv_config.get("pin_memory", True),
        collate_fn=multiview_collate_fn,
    )
    return model, loader, mv_config


# ---------------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_path", required=True, help="HDF5 dataset (same one the benchmark used)")
    p.add_argument("--smal-file", dest="smal_file", default=None, help="Override the model .pkl")
    p.add_argument("--shape-family", dest="shape_family", type=int, default=None)
    p.add_argument("--out", required=True, help="Output stem; writes <out>.npz and <out>.json")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=0, help="0 = all test frames")
    p.add_argument("--device", default=None, help="e.g. cuda:0 (default: cuda if available)")
    p.add_argument("--fps", type=float, default=30.0, help="Recorded in the sidecar; only affects time axes in plots")
    p.add_argument(
        "--split",
        default="test",
        choices=["test", "all"],
        help="'test' (default) matches the benchmark; 'all' exports every item in the HDF5",
    )
    p.add_argument(
        "--random-view-sampling",
        action="store_true",
        help="Multi-view only: let the dataset sample views randomly. Off by default so the two arms "
        "of the study see identical view sets.",
    )
    args = p.parse_args()

    os.chdir(REPO_ROOT)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_type = _detect_model_type(ckpt)
    print(f"Checkpoint kind : {model_type}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', '(unknown)')}")

    if model_type == "singleview":
        model, loader, _run_config = setup_singleview(args, ckpt, device)
        predict = lambda xb, yb: model.predict_from_batch(xb, yb)  # noqa: E731
    else:
        model, loader, _run_config = setup_multiview(args, ckpt, device)
        predict = lambda xb, yb: model.predict_from_multiview_batch(xb, yb)  # noqa: E731

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    recorder = build_recorder_from_config(
        output_path=str(out_stem),
        rotation_representation=model.rotation_representation,
        fps=args.fps,
        source_checkpoint=args.checkpoint,
        source_input=f"{args.dataset_path}[{args.split}:{model_type}]",
        model_id=getattr(model, "model_id", None),
    )

    limit = args.max_frames if args.max_frames > 0 else None
    n_recorded = 0
    n_skipped = 0
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            result = predict(x_batch, y_batch)
            if result[0] is None:
                n_skipped += len(x_batch)
                continue
            predicted_params, _, _ = result
            bs = predicted_params["global_rot"].shape[0]
            for i in range(bs):
                recorder.record(slice_params(predicted_params, i))
                n_recorded += 1
                if limit is not None and n_recorded >= limit:
                    break
            if limit is not None and n_recorded >= limit:
                break

    if n_recorded == 0:
        raise SystemExit("ERROR: no frames recorded — every batch returned None (no image data?).")
    if n_skipped:
        print(f"  note: {n_skipped} sample(s) skipped (no image data)")

    written = recorder.write()
    print(f"\nExported {n_recorded} frames:")
    print(f"  {written['npz']}")
    print(f"  {written['json']}")

    # Empirical probe: confirm the export is analysable before the study script
    # tries to read it, and surface the angle range so an obviously broken run
    # (all zeros, NaNs) is caught here rather than in a plot.
    import numpy as np

    data = np.load(written["npz"])
    poses = np.asarray(data["poses"])
    nonroot_deg = np.degrees(np.linalg.norm(poses[:, 1:], axis=2))
    print("\nPROBE:")
    print(f"  poses shape      : {poses.shape}  (F, J, 3 axis-angle)")
    print(f"  finite           : {bool(np.isfinite(poses).all())}")
    print(
        f"  |aa| deg  min/med/max: "
        f"{nonroot_deg.min():.2f} / {np.median(nonroot_deg):.2f} / {nonroot_deg.max():.2f}"
    )
    if not np.isfinite(poses).all():
        print("  WARNING: non-finite values in poses — analysis will be unreliable.", file=sys.stderr)
        return 4
    if np.allclose(poses[:, 1:], 0.0):
        print("  WARNING: all non-root poses are zero — the model likely failed to load.", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
