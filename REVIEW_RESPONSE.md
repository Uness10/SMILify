# PR review response — joint limits (#56)

All changes are packaged in `review-fixes.bundle` (6 commits on top of your current
`base` tip `f470942`, including the merge of upstream `master`; new tip `6b56f63`).

## How to apply

```powershell
cd C:\Users\youne\OneDrive\Desktop\FZJ\Stage\SMILify
git status                              # should be clean, on base
git pull .\review-fixes.bundle base     # fast-forwards base
conda activate pytorch3d
pytest                                  # full suite incl. new tests/test_joint_limits_prior.py
git push origin base
del review-fixes.bundle; del REVIEW_RESPONSE.md; del _sync_probe.txt
```

Then two manual steps only you can do:

1. **Blender 4.2 LTS test** — `python 3D_model_prep/build_addon.py`, reinstall the
   add-on, and check the new *Visualization → Show Joint Limit Overlay* checkbox
   plus the standalone `3D_model_prep/joint_rot_limit_vis.py` (supervisor asked
   for a quick 4.2 sanity pass; the code only uses long-stable API).
2. Run `pytest` — I could not run the torch-dependent tests in my sandbox.

## What each commit does

- `b3d2a35` **Merge upstream/master** — one conflict (`core_mesh.py` imports:
  master's `require_scipy_kdtree` vs. our `axis_remap` import) resolved keeping
  both. Numpy pin, Blender 4.2 notes, and panel rename all survive.
- `da7b381` **Exporter + neural penalty**
  - Muted Limit Rotation constraints are skipped (IK fallback applies).
  - `owner_space != 'LOCAL'` → warning + skip (falls back to IK/default).
  - `export_joint_limits` default=True documented as a deliberate opt-out.
  - Neural penalty deduplicated into `SMILImageRegressor._joint_limit_penalty`;
    now **raises** when the weight is > 0 but the limit set is unusable
    (no more silent per-batch print). Multiview applies it to the whole batch
    because that loss path has no per-sample validity mask by construction —
    documented at the call site.
  - Dangling `probe_joint_limits_axis_remap.py` references removed.
  - Housekeeping: removed accidentally committed `Untitled.blend` and empty
    `scripts/prior_study/rn`; restored upstream's `3D_model_prep/SMPL_fit.pkl`
    (the branch was unintentionally reverting it to a pre-#24 version).
- `fcac6b1` **Docs** — single canonical `docs/joint_limits_user_guide.md`
  (design copy deleted, images point at `design/images/`), new "Visualize your
  authored limits" section, #56 ticked + linked in `README.md`, the two panel
  options documented in `smil_importer/README.md`.
- `8e5aa94` **Tests** — `tests/test_joint_limits_prior.py` owns the LimitPrior
  read path / wide-open fallback / validation / fitter hinge guarantees /
  neural-penalty checks (neural ones marked `slow`). All six `diagnostics/`
  scripts dropped; results remain recorded in `docs/design/issue56_implementation.md`.
- `72dcbea` **Visualizer** — standalone `3D_model_prep/joint_rot_limit_vis.py`
  + `smil_importer/visualization.py` with a "Visualization" sub-panel checkbox
  (draw handler registered/removed on toggle, cleaned up on unregister, shader
  created lazily for background mode). Both skip muted and non-local-space
  constraints, so preview and export agree.
- `6b56f63` **Lint** — ruff format on the new modules, one unused import dropped
  (keeps the lint CI green).

## Suggested PR-comment reply to the supervisor

> Merged current `master` (one small conflict in `core_mesh.py` imports since
> #89/v2.1.0 landed after your review — resolved keeping `require_scipy_kdtree`).
> The #24 `apply_pose_correctives` commit no longer rides in this PR: it was
> merged separately as #89, so after the master merge the diff only shows #56 work.
>
> Exporter: muted constraints are now skipped, and non-LOCAL `owner_space`
> constraints warn and fall back to the IK/default path. `export_joint_limits`
> stays default-on deliberately: authored constraints always land in the `.pkl`,
> and the wide-open default keeps the prior inactive for untouched rigs.
>
> Neural penalty: deduplicated into a shared helper that raises when the weight
> is > 0 but the limit set can't be loaded. The multiview loss path has no
> per-sample validity mask by construction (every sample in `target_data` is
> valid), so the penalty covers the whole batch there; the single-view path
> masks first — documented at both call sites. Noted in the docs: per-axis
> bounds on axis-angle components are non-unique for |θ| > π, same convention
> as the fitter.
>
> Docs collapsed to one canonical guide with working image refs; README #56
> ticked and linked; panel options documented. `diagnostics/` is gone —
> durable assertions now live in `tests/test_joint_limits_prior.py` (+ the
> existing `tests/test_axis_remap.py`), results stay recorded in
> `docs/design/issue56_implementation.md`.
>
> Visualizer: landed standalone under `3D_model_prep/joint_rot_limit_vis.py`
> and integrated as a "Visualization" sub-panel checkbox (draw handler
> registered/removed on toggle; removed on unregister; shader created lazily so
> headless registration works). It skips muted and non-LOCAL constraints, so
> the preview matches the export exactly. Tested in Blender 4.2 LTS. *(← after
> you actually run the 4.2 check)*

## Open question

`test_joint_regressor.csv` at the repo root looks like a leftover artifact from
testing the `export_J_regressor_to_npy` CSV-path fix. I left it in; delete it
(`git rm test_joint_regressor.csv`) before pushing if it wasn't intentional.
