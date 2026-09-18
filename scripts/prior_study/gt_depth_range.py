#!/usr/bin/env python3
"""
Report the ground-truth depth distribution of a camera_centric dataset.

Use it to choose depth.min_depth / depth.max_depth / depth.init_depth for a
2D-only run. In camera_centric mode the sampled camera IS the PyTorch3D identity
camera, so the z of the GT 3D keypoints is the depth in front of the camera, in
the same units the model's `trans` uses.

    python scripts/prior_study/gt_depth_range.py \
        --data_path SMILySTICKS_centred_reprojected_FIXED.h5 --num_samples 2000

Prints percentiles of the per-sample root depth (the mean z over visible joints)
and of all joint depths. Sensible config values:

    min_depth  a bit below the 0.1st percentile
    max_depth  a bit above the 99.9th percentile
    init_depth the median
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from smal_fitter.neuralSMIL.smil_datasets import UnifiedSMILDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--num_samples", type=int, default=2000)
    ap.add_argument("--backbone_name", default="vit_large_patch16_224")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = UnifiedSMILDataset.from_path(
        args.data_path,
        rotation_representation="6d",
        backbone_name=args.backbone_name,
        return_single_view=True,
        camera_centric=True,
        expand_all_views=True,
        augment=False,
    )
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(ds), size=min(args.num_samples, len(ds)), replace=False)

    root_depths, joint_depths, skipped = [], [], 0
    for idx in idxs:
        _, y = ds[int(idx)]
        kp3d = y.get("keypoints_3d")
        if kp3d is None:
            skipped += 1
            continue
        kp3d = np.asarray(kp3d, dtype=np.float64).reshape(-1, 3)
        # all-zero rows are the "no GT for this joint" sentinel
        valid = ~np.all(kp3d == 0, axis=1)
        if not valid.any():
            skipped += 1
            continue
        z = kp3d[valid, 2]
        joint_depths.append(z)
        root_depths.append(float(z.mean()))

    if not root_depths:
        raise SystemExit(f"No GT 3D keypoints found in {args.data_path} (skipped {skipped} samples)")

    root = np.array(root_depths)
    joints = np.concatenate(joint_depths)
    print(f"\nsamples used: {len(root)}  (skipped without GT 3D: {skipped})")
    for name, arr in (("per-sample root depth", root), ("all joint depths", joints)):
        qs = np.percentile(arr, [0.1, 1, 50, 99, 99.9])
        print(
            f"  {name:>22}: min {arr.min():.4f}  p0.1 {qs[0]:.4f}  p1 {qs[1]:.4f}  "
            f"median {qs[2]:.4f}  p99 {qs[3]:.4f}  p99.9 {qs[4]:.4f}  max {arr.max():.4f}"
        )
    if (joints <= 0).any():
        print(f"  WARNING: {(joints <= 0).mean():.1%} of GT joint depths are <= 0 (behind the camera)")

    lo, med, hi = np.percentile(root, [0.1, 50, 99.9])
    print(
        "\nSuggested config:\n"
        f'  "depth": {{"positive_depth": true, "init_depth": {med:.3f}, '
        f'"min_depth": {max(lo * 0.5, 1e-4):.3f}, "max_depth": {hi * 1.5:.3f}}}\n'
        "  (the bounds are deliberately wider than the data; they exist to stop a collapse,\n"
        "   not to encode the answer)"
    )


if __name__ == "__main__":
    main()
