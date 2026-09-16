# 2D-only single-view runs: the predicted insect is behind the camera

**Affects:** the 2D-only study runs (`configs_runs/singleview_2d_lam*.json`): single-view, `camera_centric`, trained from scratch without 3D supervision.
**Confirmed on:** `singleview_2d_lam1e-1`, checkpoint epoch 193. The λ = 0 and λ = 1e-4 runs have not been checked yet.
**Not affected:** the 3D-supervised reference (`singleview_SMILySTICKS_3D_ViT`).

## Issue

In the training visualizations (e.g. `img_0004_epoch_0179.png`):

- The predicted 2D keypoints overlay the image correctly.
- The rendered mesh is wrong: the render panel is filled with the mesh colour, the silhouette panel is black, and the rotated view shows scattered keypoints.

## Diagnosis

`smal_fitter/neuralSMIL/diagnose_mesh_render.py` was run on 5 validation samples:

| | 3D-supervised reference | 2D-only, λ = 1e-1 |
|---|---|---|
| predicted depth (`trans` z) | +0.18 to +0.31 | **−1.7 to −2.2** |
| vertices at or behind camera (z ≤ 0) | 0 % | **93–100 %** |
| silhouette coverage of the frame | 1–9 % | **23–100 %** |
| `mesh_scale` | 0.02–0.06 | 0.88–1.15 (≈ its initial value, 1.0) |
| largest per-joint log-scale | 0.25–0.47 | 1.2–1.7 |
| projected keypoints inside the frame | 100 % | 98–100 % |
| mesh size / keypoint spread | ≈ 1.05 | ≈ 1.05 (no mesh explosion) |

Checks that rule out the pipeline:

- In camera-centric mode the camera is the PyTorch3D identity, which looks along **+z**. The dataset conversion (`recanonicalize_single_view` followed by `_sleap_to_pytorch3d_camera`) flips x and y only, so ground-truth insects have **positive** depth. Verified numerically.
- The reference model places the insect at z > 0 with every vertex in front of the camera, so the render, camera and mesh-scale code paths are correct.

## Cause

**The 2D keypoint loss cannot tell whether the insect is in front of the camera or behind it.**

A pinhole camera maps a 3D point (x, y, z) to (x/z, y/z). The point (−x, −y, −z), i.e. the same point flipped through the camera centre, maps to exactly the same pixel. So for every correct pose in front of the camera there is a flipped pose behind it with **identical 2D keypoints and identical 2D loss**. Verified numerically: the 2D difference is 0.0.

- **In the 3D-supervised run,** the `trans` and `keypoint_3d` targets force the solution in front of the camera.
- **In the 2D-only runs,** those terms are set to 0, and nothing else constrains the sign of depth. The network converged to the behind-camera solution. Nothing requires depth to be positive, and training started from `mesh_scale = 1.0` (≈ 20–40× the correct size), far from the true solution.
- **Why the render breaks:** the rasterizer does not draw faces behind the camera. With every vertex behind, nothing is drawn. The predicted per-joint scales, which are also loosely constrained here, push about 7 % of the vertices in front of the camera. Triangles that straddle the camera plane project to huge shapes and fill the frame.
- **Why the keypoints look fine:** they are projected with the same x/z, y/z formula, so the flipped solution puts them exactly where the labels are.

The run cannot correct itself. Moving from z < 0 to z > 0 would require passing through z = 0, where the projection is undefined.

## Consequences for the study

- **Suspect:** all 3D outputs of these runs (3D joints, MPJPE / N-MPJPE, joint angles). The flipped body is a mirror image, which ordinary joint rotations cannot produce exactly. The joint-limit prior is therefore acting on the angles of a physically impossible configuration.
- **Still valid:** 2D metrics (keypoint reprojection error).
- **Runs may not be comparable:** whether a run flips depends on early training dynamics, so a difference between λ values could come from the flip rather than the prior.

## Fix (implemented, no 3D supervision reintroduced)

1. **Positive depth, by construction.** New config section `depth`:
   ```json
   "depth": {"positive_depth": true, "init_depth": 0.25}
   ```
   With it on, the model treats the predicted depth as a log-depth: `z = init_depth · exp(raw)`. So z > 0 always, and the behind-camera solution cannot be reached. This encodes only that the insect is in front of the camera; it is not supervision. Log-depth was chosen over softplus because its gradient never vanishes. It is the same form as the log `mesh_scale`.
   - Implemented in `SMILImageRegressor.forward` (`positive_depth`, `init_depth` arguments). It requires `camera_centric` and fails loudly otherwise.
   - The trainer reads the new section and saves it in every checkpoint. `run_singleview_inference.py` and `benchmark_model.py` rebuild the model with it.
   - The default is off, so existing configs and checkpoints behave exactly as before.
2. **Sensible starting point.** `mesh_scaling.init_mesh_scale: 0.04` and `init_depth: 0.25`, matching the 3D-supervised reference (scale 0.02–0.06, depth 0.18–0.31).
3. **Config generation.** `prepare_scratch_config.py` now switches on (1) and (2) automatically whenever `trans` and `keypoint_3d` are dropped (`--positive-depth auto|on|off`, `--init-depth`, `--init-mesh-scale`). `prepare_2d_only_sweep.sh` passes them through (`POSITIVE_DEPTH`, `INIT_DEPTH`, `INIT_MESH_SCALE`). `preflight_2d_only.py` checks that `depth` is identical across runs. The three `configs_runs/singleview_2d_lam*.json` files are updated to match.
4. **Not changed:** the part-size regularizer schedule. The render failure came from the flip, not from exploding part sizes, and changing the schedule would add a second difference from the reference. Revisit it if part sizes still drift after the restart.
5. **Still open:** positive depth fixes the *sign* of depth only. The size-versus-distance ambiguity (a larger insect farther away) remains, as `prepare_scratch_config.py` warns. Compare runs on N-MPJPE, as planned.

## Restarting the runs

The training job **auto-resumes** from the newest checkpoint in each run's `checkpoint_dir`. The old run directories must be moved aside, or the "restart" will continue the flipped runs:

```bash
scancel <old job ids>
for L in 0 1e-4 1e-1; do
  mv "$RUNS_ROOT/singleview_2d_lam$L" "$RUNS_ROOT/singleview_2d_lam${L}_FLIPPED"
done
bash scripts/prior_study/prepare_2d_only_sweep.sh   # regenerates configs with depth on
python scripts/prior_study/preflight_2d_only.py --config configs_runs/singleview_2d_lam0.json \
    --config configs_runs/singleview_2d_lam1e-4.json --config configs_runs/singleview_2d_lam1e-1.json
```

Then launch as before. At the first checkpoints, run `diagnose_mesh_render.py`. Expect depth > 0, `verts z<=0` = 0 and small silhouette coverage.

## Related code changes (already made)

- `smal_fitter/fitter.py`: the rotated back-view panel now rotates the mesh about its own centroid. Previously it rotated about the world origin, which in camera-centric mode is the camera itself.
- `smal_fitter/neuralSMIL/run_singleview_inference.py`: PCA scale/translation outputs are now converted to per-joint values before rendering. This affects only `separate` mode with PCA enabled.
- `smal_fitter/neuralSMIL/diagnose_mesh_render.py`: new diagnostic script, used for the numbers above. It flags predictions behind the camera.
- `scripts/prior_study/prepare_2d_only_sweep.sh`: fixed a pre-existing syntax error (unterminated quote on the `DATASET` default).
