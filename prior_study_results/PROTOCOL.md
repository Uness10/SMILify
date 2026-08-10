# Joint-limit prior study — pre-registered protocol

**Status:** LOCKED
**Date locked:** 2026-08-10
**Roadmap:** `scripts/prior_study/ROADMAP.md` (Phase 0 + Phase 0.5)
**Scope:** two arms — multi-view (MV) and single-view (SV) — on the stick-insect model
`3D_model_prep/SMILy_STICK.pkl` and its authored-limits successor
`3D_model_prep/SMILy_STICK_limits.pkl`.

> This document is written **before any run of the constrained model**. Everything below
> — endpoints, thresholds, hypotheses — is fixed in advance so the conclusion cannot be
> chosen after seeing the numbers. Amendments are allowed but must be **appended** to the
> Amendment log at the bottom, dated, with a reason. Nothing above the log gets edited.

---

## 0. Research question

Does adding an authored per-joint rotation prior to the neural stick-insect model produce
more *anatomically plausible* poses without losing 3D/2D accuracy — and does it help
**more when the pose is underdetermined** (single-view) than when geometry already
constrains it (multi-view)?

The prior enters training as the existing soft hinge penalty, applied identically in both
pipelines (`multiview_smil_regressor.py:893`, `smil_image_regressor.py:1985`):

```
joint_limit_regularization = λ · mean( relu(θ − θ_max) + relu(θ_min − θ) )
```

on `predicted_params["joint_rot"]`, per axis-angle component. No new loss code is written
for this study.

---

## 1. Frozen evaluation split

**Decision: the canonical split seed is `42`.** The multi-view stick config
(`configs/examples/multiview_sticks_UNET_optimal.json`) already uses it; the single-view
example uses `1234`, and the SV study config built in Phase 1b **must adopt 42**.

| Parameter | Value |
|---|---|
| `seed` | **42** |
| `train_ratio` | 0.85 |
| `val_ratio` | 0.05 |
| `test_ratio` | 0.10 |
| Dataset | `SMILySTICKS_centred_reprojected_FIXED.h5` |
| Split granularity | **sample level** (all camera views of a sample stay in one split) |

Why one seed is sufficient to make the arms comparable — probed, not assumed:

- MV splits with `random_split(dataset, [n_train, n_val, n_test], Generator().manual_seed(seed))`
  where `len(dataset) == metadata.attrs["num_samples"]`
  (`train_multiview_regressor.py:2390-2397`).
- SV-from-multiview splits with `random_split(range(n_samples), [...], Generator().manual_seed(seed))`
  and then expands each split's samples into view-items via `item_sample_indices`
  (`train_smil_regressor.py:1666-1687`).
- `random_split` draws `randperm(sum(lengths), generator=...)`, which depends only on the
  seed and the total — not on the dataset object. Same seed + same ratios + same
  `n_samples` ⇒ **identical sample partitions**, hence identical underlying frames.

**Artifact:** `prior_study_results/eval_split.json`, produced once by

```bash
python scripts/prior_study/freeze_eval_split.py \
    --dataset SMILySTICKS_centred_reprojected_FIXED.h5 --seed 42
```

It stores the train/val/test **sample** indices (MV arm) and the corresponding
**view-item** indices (SV arm), plus a `view_mask` fingerprint of the HDF5. Every study
run re-verifies against it:

```bash
python scripts/prior_study/freeze_eval_split.py \
    --dataset SMILySTICKS_centred_reprojected_FIXED.h5 --verify
```

A non-zero exit means the split moved — the run is not comparable and must not be
reported. `tests/test_prior_study_eval_split.py` pins the equivalence between the tool and
both trainers' inline logic.

**Held constant within each arm** (never changed between that arm's baseline, control and
prior runs): `joint_importance`, `ignored_joint_names`, backbone, IEF iterations,
LR schedule, augmentation, and every loss weight except λ.

**Known cross-arm differences** (why we compare Δ *within* arm, never SV absolute vs MV
absolute): `ignored_joint_names` is `["b_t","b_a_1..5"]` in SV and `[]` in MV; backbones
differ (ViT-L vs UNet-EffB3/B5); IEF iteration counts differ. `from_multiview: true` holds
the *data* constant, not the architecture.

---

## 2. Endpoints

### 2.1 Primary — realism

> ⚠️ **Superseded by Amendment 1 (2026-08-10) — see §8.** The joint set below was found to
> be unconstrainable under §5.4 before any run took place. The original text is retained
> unedited; §8 defines the endpoint actually in force.

**Per-axis limit-violation rate.** Percentage of test frames in which a joint's axis-angle
component falls outside its authored `[min, max]`, averaged over the six
`joint_importance` leg-tip joints:

```
l_1_pt_l, l_2_pt_l, l_3_pt_l, l_1_pt_r, l_2_pt_r, l_3_pt_r
```

Measured per axis (x, y, z) because the hinge loss clamps each component independently;
reported as the mean over the constrained axes of those six joints.
Source: `limit_violations.csv` from `scripts/prior_study/analyze_baseline_pose.py`.

### 2.2 Secondary — realism

- Mean and max **overshoot** in degrees, per joint/axis (same CSV).
- **Range of motion**: per-joint ‖axis-angle‖ distribution (`magnitude_stats.csv`,
  `range_of_motion.png`).
- Violation rate over **all constrained joints**, not just the six — reported alongside
  the primary so a gain concentrated in one joint is visible.

### 2.3 Guardrail — accuracy

- **MPJPE** mean and median, in **mm**.
- **PCK@5px** at native resolution (1530 px) and at input resolution (224 px).

Source: `benchmark_model.py` (`_detect_model_type` auto-detects checkpoint type; the SV
path uses `_compute_mpjpe_mm_singleview`). No changes needed for either arm.

Accuracy is a **guardrail, not a headline**. MPJPE and PCK measure keypoint position, not
plausibility: a more anatomically realistic pose can score identically. A realism gain
that is invisible in MPJPE is the *expected* outcome, not a failure.

---

## 3. Pre-registered decision rule

> **The prior is judged beneficial if, within an arm, the violation rate drops by
> ≥ 50 % relative to that arm's λ=0 control, while MPJPE increases by ≤ 2 % relative and
> PCK@5px (1530 px) drops by ≤ 1 percentage point — consistently across ≥ 2 seeds.**

Clarifications, fixed now:

- **The comparator is the λ=0 fine-tune control, not the unconstrained baseline.** Both
  are reported, but the verdict is computed against the control, so "extra 30 epochs of
  fine-tuning" cannot be mistaken for "the prior".
- "Consistently across ≥ 2 seeds" means every seed individually satisfies all three
  conditions. If seeds disagree, the verdict is **inconclusive**, not "beneficial on
  average".
- If between-seed spread exceeds the between-arm difference on an endpoint, the honest
  conclusion for that endpoint is **"no detectable effect"**.
- λ\* is selected per arm in Phase 5 at the knee of the violation-rate-vs-MPJPE curve —
  the largest violation reduction still inside this guardrail. λ\*_SV ≠ λ\*_MV is
  expected: the SV path masks invalid samples before the penalty and the MV path does not,
  so equal λ is not equal pressure.

### 3.1 Verdict template (fill in Phase 7, do not edit above)

| | MV | SV |
|---|---|---|
| λ\* | | |
| Δ violation rate vs control (pp / % rel.) | | |
| Δ MPJPE vs control (mm / % rel.) | | |
| Δ PCK@5px vs control (pp) | | |
| Rule satisfied? | | |

---

## 4. Hypotheses (Phase 0.5)

- **H1.** The prior reduces the violation rate in **both** pipelines.
- **H2 — the interesting one.** The reduction is **larger in single-view than
  multi-view**, because multi-view already constrains 3D through `keypoint_3d` and
  `triangulation_consistency`, leaving the prior less headroom.
- **H3.** The accuracy cost of the prior is **no worse in single-view** — i.e. the prior
  buys realism where it is cheapest to buy.

**H2 is the study's headline.** The reported quantity is the *interaction*,
Δviolation(SV) − Δviolation(MV), with seed spread on both — not the individual columns.
An interaction effect is a far stronger result than a main effect, and it rescues the
study from a null: if MV shows nothing but SV shows a clear gain, that is a clean,
mechanistically sensible finding rather than a failed experiment.

**First, free test of H2 — Phase 4 viability gate.** Scoring the authored limits against
both *unconstrained* baselines costs no GPU and should already show
**SV violation rate > MV violation rate**. If they violate equally, H2 is in doubt and
that must be recorded here **before** Phase 5 is run, not after.

---

## 5. Committed in advance

1. **The null outcome is publishable.** "The prior costs nothing and buys nothing on this
   dataset" is a valid finding and will be reported as such. We will not tune until
   something moves.
2. **A λ=0 control is run in every arm** — same authored `.pkl`, same epoch count, same
   seed, λ = 0.0. Without it, neither arm has a claim. If GPU time runs short, seeds are
   dropped before controls, and the MV arm is dropped before the SV arm (SV carries H2).
3. **Per-joint results are reported, not only the aggregate.** The prior likely helps a
   handful of joints a lot and the rest not at all; the aggregate hides this.
4. **Undocumented joints stay wide open (±π).** The rig has 55 joints; the literature
   covers the 18 leg joints well and the rest barely. A prior on joints we cannot justify
   is noise and would make the study uninterpretable.
5. **Sign/axis errors are treated as construction errors, not results.** A Phase-4
   violation rate > ~70 % on target joints means the limits contradict the data — almost
   certainly a flipped sign — and is fixed before any GPU time is spent. A rate < ~2 %
   means the hinge is inert and training with it changes nothing; that also sends us back
   to Phase 3 before it is reported as a null.
6. **`joint_angle_regularization` is a known confound** — it is already active in both
   curricula (MV 1e-3 → 1e-8, SV 1e-3 → 1e-6) and is an implicit shrink-to-rest prior. It
   is held identical across every run within an arm. One ablation with it zeroed is
   optional and, if run, is reported as exploratory.
7. **Fine-tuning, not from-scratch.** The prior only reshapes the last 30 epochs, which
   may understate its true effect. Stated as a limitation; the λ=0 control makes the
   comparison fair *within* the fine-tune regime.

---

## 6. Phase 4 record — SV vs MV viability (fill before Phase 5)

| | MV unconstrained | SV unconstrained |
|---|---|---|
| Violation rate, six target joints | | |
| Violation rate, all constrained joints | | |
| Mean overshoot (deg) | | |
| Gate verdict (<2 % / 5–40 % / >70 %) | | |

**H2 early read:** _(SV > MV as predicted? record the answer here before Phase 5 starts.)_

---

## 7. Amendment log

| Date | Change | Reason |
|---|---|---|
| 2026-08-10 | Protocol locked (Phase 0 + 0.5). | Initial pre-registration. |
| 2026-08-10 | **Amendment 1** (§8): primary endpoint moves from the six `*_pt_*` pretarsus joints to the 18 documented leg hinges (`co`/`tr`/`ti` × 6 legs). | §2.1 and §5.4 were mutually unsatisfiable: the pretarsi are undocumented in the literature and would be left wide open, making the primary endpoint identically zero in every column. Amended **before any constrained run**, on rig-audit evidence alone. |

---

## 8. Amendment 1 — primary endpoint joint set

**Date:** 2026-08-10 **Status:** in force, supersedes §2.1
**Evidence:** rig audit of `3D_model_prep/SMILy_STICK.pkl` (55 joints), Phase 3a.
**Runs affected:** none — no constrained model had been trained when this was written.

### Why

Each leg in the rig is the chain

```
b_t → co → tr → fe → ti → ta → pt          (× 6 legs, suffix _r / _l)
```

A joint's rotation is that of the named bone relative to its parent, so the anatomical
mapping is:

| Rig segment | Rotates relative to | Anatomical joint | Literature |
|---|---|---|---|
| `co` coxa | thorax | **ThC** — protraction/retraction (α) | well covered |
| `tr` trochanter | coxa | **CTr** — levation/depression (β) | well covered |
| `fe` femur | trochanter | trochantero-femoral | fused/immobile in stick insects |
| `ti` tibia | femur | **FTi** — flexion/extension (γ) | best documented |
| `ta` tarsus | tibia | tibia–tarsus | sparse, often treated as passive |
| `pt` pretarsus | tarsus | tarsus–pretarsus (claw) | effectively absent |

The six `joint_importance` joints named in §2.1 are `l_{1,2,3}_pt_{l,r}` — the **pretarsi**,
the bottom row of that table. §5.4 commits to leaving undocumented joints wide open at ±π.
Both cannot hold: with the pretarsi unconstrained the hinge loss is inert there, so the
violation rate is identically 0 in the unconstrained, control **and** prior columns. The
primary endpoint would measure nothing, and the Phase-4 gate would read "< 2 %, return to
Phase 3" for a reason that is not a sign or axis error.

Note that §2.1 inherited this joint set from `joint_importance`, whose purpose is
*keypoint weighting* — the leg tips are where positional error matters most. That is a
good reason to weight them in MPJPE and no reason at all to expect a rotation prior on
them.

### The endpoint now in force

**Primary (realism): per-axis limit-violation rate over the 18 documented leg hinges.**
Percentage of test frames in which a joint's axis-angle component falls outside its
authored `[min, max]`, averaged over the constrained axes of:

```
l_{1,2,3}_co_{l,r}   (6 × ThC)
l_{1,2,3}_tr_{l,r}   (6 × CTr)
l_{1,2,3}_ti_{l,r}   (6 × FTi)
```

Measured per axis (x, y, z), since the hinge loss clamps each component independently.
Source: `limit_violations.csv` from `scripts/prior_study/analyze_baseline_pose.py`, invoked
with `--important-joints` set to the 18 names above.

Reported alongside, unchanged in role:

- Violation rate over **all** constrained joints (secondary, §2.2) — catches a gain
  concentrated in one hinge.
- Per-joint breakdown (§5.3).
- The six `*_pt_*` joints stay in the **accuracy guardrail** (§2.3), where they belong:
  they are the leg tips MPJPE and PCK are weighted on.

### Consequential notes

- `fe` (trochantero-femoral) is fused in stick insects. Per the Risks table it should be
  **locked** (Min = Max = 0) rather than loosely bounded, and it is excluded from the
  primary endpoint — a locked joint that never moves cannot show a violation reduction.
- `ta` and `pt` remain wide open unless Phase 2 turns up a defensible source. If it does,
  they join the secondary "all constrained joints" figure, not the primary.
- §3's decision rule, §4's hypotheses and §6's viability table are unchanged in form; they
  now read against the 18-joint set.
