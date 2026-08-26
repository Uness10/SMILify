# Multi-animal implementation notes

Implementation of [`multianimal.md`](multianimal.md) (and the sketch in
[`mutlianimal_sketch.txt`](mutlianimal_sketch.txt)). This document records *what
was built and where*, the decisions that were not spelled out in the design, and
what still has to happen once the multi-mouse dataset exists.

The guiding constraint was that **nothing about the single-animal path may
change**. Multi-animal is off by default; when it is off, the model, the
datasets, the collate functions and the trainers behave exactly as before.

---

## 1. Where the code lives

```
smal_fitter/neuralSMIL/multianimal/
  schema.py             data contract for the animal axis        (numpy only)
  batching.py           (B, N) <-> (B*N) tensor plumbing         (torch only)
  parameter_layout.py   flat head-output layout + parsing        (torch only)
  heads.py              one parameter head as a real nn.Module   (torch only)
  specimen_heads.py     the N-head bank: replicated | shared_query
  losses.py             per-specimen loss aggregation
  collate.py            rectangular animal axis at batch time
  datasets.py           N=1 promotion, per-track grouping
  hdf5_schema.py        on-disk animal axis
  checkpoint.py         single-animal <-> multi-animal weight migration
  factory.py            the three call sites the trainers use
  regressor.py          single-view model     (needs pytorch3d + config)
  multiview_regressor.py multi-view model     (needs pytorch3d + config)

smal_fitter/neuralSMIL/configs/multianimal_config.py   MultiAnimalConfig
smal_fitter/neuralSMIL/configs/examples/multiview_multianimal_mice.json
diagnostics/multianimal_forward_PROBE.py               structural probe
tests/test_multianimal_*.py                            unit tests
```

Only the two regressor modules import pytorch3d and the SMAL model, so
everything above them is unit-testable with plain PyTorch — which is why the
CPU-only CI run covers the whole data contract, head bank, loss aggregation and
checkpoint migration.

## 2. Design decisions

### 2.1 The animal axis is a list of per-specimen dicts

`y_data["animals"]` holds `N` dicts, each using exactly the key names a
single-animal sample already uses; `x_data` carries `num_animals`,
`animal_mask` and `specimen_ids`. Scene-level information (images, cameras,
view masks) stays at the top level under its existing names.

The alternative — a leading `(N, ...)` axis on every array — was rejected for
in-memory samples because per-specimen dicts are *byte-for-byte* what the
existing target collectors (`_extract_target_parameters_single`,
`_collect_body_targets_batch`) already consume, so the entire loss stack is
reused with no changes; and because labels are frequently ragged (specimen 1 has
pose GT, specimen 2 only 2D keypoints), which `None`-per-key expresses directly.

On disk the dense axis *is* used (§4) — images and cameras are shared per frame,
so duplicating a row per animal would multiply image storage by `N`.

### 2.2 Absence is expressed as "no labels"

An absent specimen (or one below the visibility floor) is materialised as a
target dict whose animal-level entries are `None`. The existing implicit
availability masking then switches its whole loss off, so **no loss function
needed an `animal_mask` argument**.

Two subtleties, both found by the probe and covered by regression tests:

* Keypoint keys are *omitted*, not set to `None`. `assemble_batch_inputs` builds
  a `keypoint_data` entry whenever `keypoints_2d` and `keypoint_visibility` are
  both present, and `_validate_sample_visibility` then indexes them
  unconditionally.
* `specimen_target_view` strips every animal-level key from the scene projection
  before merging the specimen's own labels. `wrap_single_animal` keeps a
  promoted sample's labels at *both* the top level and in slot 0, so without
  this a later slot would silently inherit specimen 0's pose and keypoints.

### 2.3 Two ways to be "N heads"

`MultiAnimalConfig.head_strategy` selects:

| strategy | what it is | cost |
| --- | --- | --- |
| `replicated` (default) | `N` independent copies of the existing head, exactly as the design doc specifies | `N ×` head parameters |
| `shared_query` | one head plus `N` learned specimen embeddings (added to the pooled feature and injected as a cross-attention token) | one head + `2 N D` |

Both bind head `i` permanently to specimen `i`: the specimen index is an
*input*, never something the model discovers. `shared_query` also decodes all
`N` specimens in a single head call by folding the animal axis into the batch
axis.

`replicated` is the default because it is what the design specifies and because
it lets one specimen's head diverge freely; `shared_query` exists for when `N`
grows or the specimens are the same species and should share statistical
strength.

### 2.4 The camera is scene level

Single-view gains a dedicated `CameraHead` (the *same* class the multi-view
model already uses, so the two cannot drift apart) and specimen heads have their
camera outputs stripped. Multi-view needed no change at all — its per-canonical-
view camera heads were already scene level.

`camera_mode="first_specimen"` reads the camera out of specimen head 0 instead;
it is only legal for `N == 1`, where it reproduces the legacy single-animal
model exactly (this is what the equivalence probe checks). The config rejects it
for `N > 1`, where it would give the other heads no camera gradient.

### 2.5 The loss is the existing loss, run once per specimen

`MultiAnimalLossAggregator` calls the *unmodified* inherited loss with one
specimen's body parameters plus the shared camera, then averages over the
specimens that are actually present. Two things it owns:

* **Camera terms are supervised on specimen 0 only.** Otherwise the camera
  weight would be multiplied by `N` relative to every body term and silently
  rebalance the whole curriculum.
* **Absent slots are dropped, not averaged in as zero.** An `N = 3` run on
  2-mouse clips therefore trains exactly like a 2-mouse run.

Loss components are reported both under their original names (so existing
logging, plotting and checkpointing keep working) and as
`"<component>/<specimen_id>"`, so a specimen that is failing to converge is
visible in the training logs.

### 2.6 No matching, but the ordering is *checked*

Strict head ↔ specimen correspondence is what removes the need for Hungarian
matching, and it only holds if the dataset emits a stable ordering.
`require_stable_identity` (on by default) asserts every sample in a batch lists
the same `specimen_ids` in the same order. A silently reshuffled track order
would otherwise train head 0 on two different animals with no visible symptom.

## 3. How it plugs into the existing model

Both trainers touch the model through exactly two methods, which is what made a
subclass sufficient:

| | single-view | multi-view |
| --- | --- | --- |
| batch step | `predict_from_batch` | `predict_from_multiview_batch` |
| loss | `compute_batch_loss` | `compute_multiview_batch_loss` |

`MultiAnimalSMILRegressor` overrides `forward`, `predict_from_batch` and
`compute_batch_loss`. `MultiAnimalMultiViewSMILRegressor` overrides
`_predict_body_params`, `predict_from_multiview_batch` and
`compute_multiview_batch_loss` — the inherited `forward_multiview` splices
whatever `_predict_body_params` returns into its output, so backbone chunking,
view embeddings, cross-view attention and the per-view camera heads are all
reused untouched.

Three small refactors were made in the parent classes, all behaviour-preserving:

* `SMILImageRegressor.predict_from_batch` was split into
  `assemble_batch_inputs` / `apply_fixed_camera` / `merge_available_label_masks`
  so the multi-animal model can reuse the target-side assembly instead of
  duplicating ~100 lines of availability-masking rules.
* `MultiViewSMILImageRegressor._predict_body_params`'s feature preparation was
  extracted into `_prepare_decoder_inputs`.
* `_extract_target_parameters_single` now reads `scale_weights` /
  `trans_weights` / `translation_factor` with `.get()` like every neighbouring
  read, instead of raising `KeyError` on a target dict that lacks them.

One latent bug was fixed while wiring this up: `merge_available_label_masks`
silently masked out an entire batch when `available_labels` was present for only
*some* samples (a short boolean mask broadcast against the full-length implicit
mask resolves to all-`False`). It now warns and skips the explicit merge.

## 4. HDF5 layout for the multi-mouse dataset

`multianimal/hdf5_schema.py` is the spec. Animal-level datasets gain an animal
axis immediately after the sample axis; scene-level datasets are untouched:

```
multiview_keypoints/keypoints_2d          (S, A, V, J, 2)
multiview_keypoints/keypoint_visibility   (S, A, V, J)
parameters/global_rot                     (S, A, 3)
parameters/joint_rot                      (S, A, N_POSE, 3)
auxiliary/animal_mask                     (S, A)            bool   <- required
multiview_images/...                      unchanged
multiview_keypoints/camera_*              unchanged

metadata.attrs["multianimal_schema_version"] = 1
metadata.attrs["num_animals"]                = A
metadata.attrs["specimen_ids"]               = ["mouse_a", ...]
```

A file *without* `multianimal_schema_version` is read as `A = 1`; that is the
whole backwards-compatibility story for existing preprocessed datasets, and
`detect_layout()` is the single place that decides it.

`validate_file_shapes()` checks both the animal-axis size *and* the dataset
rank, because shape alone is ambiguous: a `(S, 3)` `global_rot` in a 3-specimen
file looks exactly like `(S, A)` if only the second dimension is checked. A
half-migrated file therefore fails loudly at open time instead of producing
silently mispaired specimens.

Writers must call `declare_layout()`; a file with the axis but no declaration
would be misread as single-animal.

## 5. Identity comes from the SLEAP track axis

`points3d.h5` stores `tracks` as `(n_frames, n_tracks, n_keypoints, 3)`. The
`n_tracks` axis *is* the animal identity, which is exactly the "fixed
identity/ordering established beforehand" the design asks for — no identity
discovery, no matching.

`SLEAP3DDataLoader` previously hardcoded `tracks[:, 0]` ("assuming single
animal"). It now takes a `track_index`, keeps every track in
`keypoints_3d_all_tracks`, and exposes `get_num_tracks()`,
`get_keypoints_3d_for_track()` and `get_track_presence()` (a track is absent in
a frame when SLEAP stored it as all-NaN — that is what feeds `animal_mask`).
The default `track_index=0` keeps every existing caller unchanged.

For loaders that still emit one sample per (frame, track),
`GroupedSpecimenDataset` buckets them by frame and places the specimens in a
configured slot order.

## 6. Using it

```jsonc
// smal_fitter/neuralSMIL/configs/examples/multiview_multianimal_mice.json
{
  "mode": "multiview",
  "multi_animal": {
    "enabled": true,
    "num_animals": 2,
    "specimen_ids": ["mouse_a", "mouse_b"],   // must match the dataset's track order
    "head_strategy": "replicated",
    "camera_mode": "scene_head",
    "loss_reduction": "mean",
    "min_visible_keypoints_per_specimen": 4
  },
  "training": { "resume_checkpoint": "path/to/single_animal.pth" }
}
```

```bash
python -m smal_fitter.neuralSMIL.train_multiview_regressor --config <that file>
```

Resuming from a single-animal checkpoint copies its head into *every* specimen
head (design doc §3) — `multianimal/checkpoint.py` does the key rewrite, and
`to_single_animal()` goes back the other way for evaluating one specimen with
the single-animal tooling.

Note that `N` meshes are rendered per scene, so peak VRAM grows roughly linearly
with `num_animals`; the shipped example halves `batch_size` accordingly.

## 7. Verification

* `pytest tests/test_multianimal_*.py` — 216 tests covering the data contract,
  batching, head bank, loss aggregation, config, collate, dataset adapters, the
  HDF5 layout and checkpoint migration. All run on CPU without pytorch3d.
* `python -m diagnostics.multianimal_forward_PROBE` — stubs pytorch3d/`config`
  just enough to import and *run* the real regressors, and asserts the
  structural claims empirically: one backbone pass regardless of `N`, one
  parameter dict per specimen, no camera in specimen dicts, the same cameras for
  every specimen, gradient isolation between heads, per-specimen loss
  aggregation, and **numerically identical output to `SMILImageRegressor` at
  `N = 1`**.

The probe stubs the numerics stack, so it proves *wiring*, never numerical
correctness. It is kept under `diagnostics/` per the project's
"preserve diagnostic artifacts" rule.

## 8. Not done yet

* **The multi-mouse dataset itself.** No preprocessing script writes the
  multi-animal HDF5 layout yet; §4 is the spec it must target, and
  `declare_layout()` / `animal_axis_shape()` / `validate_file_shapes()` are the
  helpers for it. Until then the multi-animal path runs end-to-end at `N = 1`
  over existing data.
* **Visualisation.** `visualize_multiview_training_progress` and the single-view
  renderers still draw one animal (specimen 0's parameters are exposed at the
  top level so they do not break). Rendering `N` meshes into one scene is a
  follow-up.
* **Occlusion reasoning.** Only the per-specimen visibility floor is
  implemented, which is deliberately what the design asks for at prototype stage
  ("I wouldn't introduce sophisticated occlusion reasoning yet"). Depth-ordered
  or mutually-occluding silhouette rendering is future work.
* **Inference scripts.** `run_multiview_inference.py` / `run_singleview_inference.py`
  have not been extended to emit `N` meshes.
