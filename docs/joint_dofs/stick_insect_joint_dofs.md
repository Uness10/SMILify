# Stick insect (Phasmatodea) joint DOFs and rotation ranges

Extracted from three papers, mapped onto the 55-joint skeleton in
`3D_model_prep/SMILy_STICK.pkl`. Machine-readable version:
[`stick_insect_joint_limits.csv`](stick_insect_joint_limits.csv).

## Sources

| Key | Reference |
|-----|-----------|
| **T15** | Theunissen, Bekemeier & Dürr (2015) *Comparative whole-body kinematics of closely related insect species with different body morphology*. J Exp Biol 218:340–352. doi:10.1242/jeb.114173 |
| **G22** | Guschlbauer, Hooper, Mantziaris, Schwarz, Szczecinski & Büschges (2022) *Correlation between ranges of leg walking angles and passive rest angles among leg types in stick insects*. Curr Biol 32:2334–2340. doi:10.1016/j.cub.2022.04.013 |
| **D16** | Dallmann, Dürr & Schmitz (2016) *Joint torques in a freely walking insect reveal distinct functions of leg joints in propulsion and posture control*. Proc R Soc B 283:20151708. doi:10.1098/rspb.2015.1708 |

All three study *Carausius morosus*; T15 additionally covers *Aretaon asperrimus* and
*Medauroidea extradentata*. Numbers below are Carausius unless stated.

## What the papers actually say about DOFs

**D16 §2c** is the only explicit DOF statement:

> "Motion of each leg is driven by three joints: the thorax–coxa (ThC) joint and the
> coxa–trochanter (CTr) joint, which together act as the 'hip', and the femur–tibia (FTi)
> joint, which acts as the 'knee'. […] **The CTr and FTi joints are approximated as hinges
> with 1 d.f. each.** They provide elevation–depression and extension–flexion of the leg,
> respectively. Both joints move in the same plane, the leg plane."

D16 further models **ThC as a single slanted axis (1 d.f.)**, tilted θ = 30° from the
vertical body axis, so that one rotation produces coupled protraction + supination.

**T15 disagrees and measures ThC as ≥2 independent DOFs:**

> "our kinematic analysis differed from several earlier studies in that we did not assume a
> fixed, single DoF joint axis of the thorax–coxa joint, but measured both the
> protraction/retraction and the pronation/supination of the leg plane independently"

and observes that pro/supination curves differ between swing and stance,
"indicating **slack of the rotation axis in the thorax–coxa joint**, which is commonly
considered fixed in stick insects."

→ For a rig, T15's 3-DOF ThC is the right choice; D16's slanted 1-DOF axis is a
simplification for inverse dynamics. The CSV follows T15.

**Trochanter–femur is fused** (T15, Fig. 9 legend):

> "Note that the femur is fused with the trochanter in these species, **without a movable
> joint in between**."

This is why `l_*_fe_*` is locked on all three axes.

## Per-joint values

### ThC — protraction / retraction (`l_*_co_*`, free)

0° = leg perpendicular to its thorax segment; positive = protraction (T15 Fig. 9A convention).

Two independent datasets, converted to a common zero:

| Leg | T15 Fig. 9A/10 plotted range | G22 ThF range (0–180°, 90° = perpendicular) | G22 re-zeroed (90 − ThF) | Adopted |
|-----|------|------|------|------|
| Front  | −20 … +80° | walk 22–104°, passive 2–104° | −14 … +88° | **−20 … +88°** |
| Middle | −40 … +60° | walk 58–124°, passive 19–139° | −49 … +71° | **−49 … +71°** |
| Hind   | −80 … +20° | walk 103–157°, passive 77–164° | −74 … +13° | **−80 … +20°** |

The two papers agree to within ~6° on the hind leg and ~8° on the front leg despite
completely different methods — a useful sanity check. Adopted values are the union
(walking + passive resting envelope).

G22's headline result is that these ranges shift **progressively more retracted from front
to hind**, and that walking ranges are nested inside the passive rest ranges. T15 Table 2
independently gives a *front leg protraction range of 45°* and *middle leg protraction mean
of 0°* for Carausius.

### ThC — pronation / supination (`l_*_co_*`, free)

**−60 … +60°**, all legs (T15 Fig. 10 supination axis; Fig. 9B time courses span −40…+40°).
T15: "the largest amplitudes of protraction/retraction and pronation/supination were
observed in the middle legs."

### CTr — levation / depression (`l_*_tr_*`, free, 1-DOF hinge)

**−40 … +60°**, all legs (T15 Fig. 10 levation axis; Fig. 9C time courses −20…+60°).
T15: "the levation angles were generally similar among species and all legs."
D16: CTr torques are directed toward depression *throughout* stance and are the joint that
carries body weight and generates propulsion.

Note T15 lumps ThC and CTr levation into one measured angle ("Levation/depression of the
thorax–coxa joint **and** the coxa–trochanter joint"). The CSV assigns the whole range to
CTr, since that is the joint D16 identifies as the dedicated 1-DOF levator/depressor, and
leaves ThC levation at the ±10° give.

### FTi — flexion / extension (`l_*_ti_*`, free, 1-DOF hinge)

Absolute joint angle **40 … 160°** (T15 Fig. 9D; Fig. 10 axis spans 40–140°).
D16 reports extension reaching 150° late in stance, and identifies **Ext = 90° as the
neutral tibia posture**: "torques tended to counteract a deviation from an angle of 90°
relative to the femur (Ext = 90°), which is the neutral posture of the tibia."

→ CSV stores **−50 … +70°** = 40–160° expressed relative to that 90° neutral.
**If the `SMILy_STICK.pkl` rest pose is not at FTi = 90°, re-zero as `min = 40 − rest`,
`max = 160 − rest`.**

T15 Table 2: hind leg flexion mean 100° (Carausius). Hind legs extend during stance
(pushing), front legs flex during stance (pulling).

### Head / neck (`b_h`)

T15 Table 2, "Head levation range": **30°** (Carausius), 15° (Medauroidea), 45° (Aretaon)
→ ±15° about rest in pitch. T15 Fig. 5D shows head orientation is *not* gaze-stabilised:
"these joints do not stabilise gaze in space but appear to adapt head pitch." Yaw and roll
were computed but no ranges are reported.

### Thorax segments — no bone to map onto

T15 Table 2 also gives, for Carausius: **T1 (prothorax) levation range 35°**, **T2
(mesothorax) levation range 25°**, thorax inclination 1.2 deg mm⁻¹. The SMILify stick
skeleton has a **single `b_t` root** for the whole thorax, so these have nowhere to go. If
inter-thoracic flexibility matters for your fits, this is the concrete argument for adding
T1/T2 bones — Carausius moves the mesothorax most, Aretaon the head most.

### Not covered by these papers

`b_a_1…5` (abdomen), `an_1/2/3_*` (antennae), `ma_*` (mandibles), `l_*_ta_*` (tibia–tarsus),
`l_*_pt_*` (pretarsus), `w_*` (wings — the species studied are apterous).

D16 explicitly excludes the tarsus: markers "cannot be placed on the tarsus without
restraining movements"; TiTa is used only as a tarsus-position estimate. Antennal joint
angles (HS 2-DOF, SP 1-DOF) would need the Dürr-lab antennal-movement literature — T15
reports antennal *lengths* only. Mandible values in the CSV are carried over from the
existing ant prior (`OmniAnt_25PCs_joint_limited.pkl`) with secondary axes reduced to ±10°.

These rows are flagged `NOT MEASURED` / `ASSUMED` in the `source` column. Don't cite them.

## Axis convention

Model frame, probed from `SMILy_STICK.pkl['J']`: **+x anterior, +y left, +z dorsal** —
identical to T15's body-fixed CS ("the x-axis of the resulting body CS pointed from
posterior to anterior, the y-axis to the left and the z-axis upward"), so paper-frame
angles carry over without a handedness flip.

Bone-local ↔ global mapping (verified in Blender): **XL = XG, YL = ZG, ZL = −YG**.
This is a right-handed 90° rotation about X (`XL × YL = ZL` ✓, det = +1). For rotation
limits it gives:

| local | global | interval transform |
|---|---|---|
| XL | XG | `[min, max]` unchanged |
| YL | ZG | `[min, max]` unchanged |
| ZL | −YG | **`[−max, −min]`** (sign flipped) |

The CSV carries **both frames side by side**. `local_*` is what you type into Blender's
Limit Rotation constraint; `global_*` is what lands in `joint_limits` in the `.pkl`.
Regenerate with [`build_stick_joint_limits.py`](build_stick_joint_limits.py).

## ⚠ The rig axes are not the anatomical axes (except for protraction)

This is the main caveat and it changed the numbers. Because the mapping is uniform across
bones, the rig axes are fixed — but the legs point in different directions, so the
anatomical hinge axes are *not* fixed relative to them. Measured from the rest pose
(horizontal limb direction, coxa→tarsus):

| Leg | limb direction (global) | splay from the lateral (Y) axis |
|-----|------|------|
| Front  | (0.70, −0.71, 0) | **44.7°** |
| Middle | (0.21, −0.98, 0) | **12.3°** |
| Hind   | (−0.79, −0.61, 0) | **52.4°** |

Consequences:

- **Protraction/retraction is exact.** It is a rotation about the dorsoventral axis =
  global Z = **local Y**, for every leg, independent of splay. Highest-confidence rows in
  the file. (D16 notes the real ThC axis is slanted ~30° from vertical; pure Z is the
  standard approximation.)
- **Levation/depression and pronation/supination are not axis-aligned** for front and hind
  legs — their true axes sit ~45–52° between the rig's X and Y. A per-axis box therefore
  cannot represent them tightly. The CSV stores the **axis-aligned envelope** (each DOF's
  range projected onto the rig axes and summed), which is *conservative*: it permits some
  non-anatomical combinations but never forbids a real pose.
- **So do not lock XG or YG to ±10° on the leg joints.** My first draft did exactly that,
  and it would have blocked legitimate levation and supination on the front and hind legs.
  Only the middle legs (12°/10° splay) are close enough to axis-aligned for that to be safe.

If you want tight rather than conservative leg limits, the axis-aligned box is the wrong
representation and you'd need per-joint rotated frames — worth knowing before you spend
time tuning `w_limit`.

**Left/right signs are derived, not assumed.** Rotating about +Z moves a right leg
(pointing −y) anteriorly but a left leg (+y) posteriorly, so protraction is +Z on the right
and −Z on the left. Left rows are mirrored `[min, max] → [−max, −min]` accordingly. This
matches the ant prior's precedent (`l_2_tr_r = [−50, 100]` vs `l_2_tr_l = [−100, 50]`).

**One sign still unverified: FTi extension direction.** CTr and FTi share the leg plane
(D16 §2c), so they share an axis, but which rotation sense is "extension" isn't recoverable
from the rest pose — the legs are nearly straight at rest (femur–tibia angle 4.7–9.8°),
which also made the leg-plane normal numerically ill-conditioned. Confirm with `R X X` on
one tibia bone and flip if needed.

`SMILy_STICK.pkl` currently has **no `joint_limits` key at all**, so nothing is being
overwritten — these are net-new.

## Locked axes

Per the ±10° "give" rule, every locked axis is written as **−10 … +10°** rather than
0/0 — real cuticular joints have slack, and a hard zero makes the hinge loss brittle.
The root `b_t` is the exception: it stays 0/0/0, as required by the `joint_limits`
convention (`joint_limits[0]` is all-zero in the ant prior).

Note the ±10° rule applies to axes that are *genuinely* locked: the fused
trochanter–femur, the bone-roll axis of a true hinge, the flagellum, the wings. It does
**not** apply to the leg joints' XG/YG rows, which look like secondary axes but are really
projections of a diagonal anatomical axis — see the section above.

## Target model

Limits are authored into **`3D_model_prep/SMILy_STICK_authored.pkl`** (backup at
`SMILy_STICK_authored.pkl.bak`). Note this is **not** `SMILy_STICK.pkl` + limits — it is a
different mesh and rest pose:

| | `SMILy_STICK.pkl` | `SMILy_STICK_authored.pkl` |
|---|---|---|
| vertices | 3020 | 3015 |
| faces | 6019 | 6009 |
| leg splay (front / middle / hind, right) | 44.7° / 12.3° / 52.4° | **25.2° / 2.9° / 44.5°** |

All splay-dependent envelopes are computed from the **authored** file's `J`. The middle legs
are now almost exactly lateral (2.9° / 0.5°), so their limits come out nearly axis-aligned
and therefore much tighter than the front and hind legs'.

The generator only replaces the `joint_limits` key; mesh, `J`, `J_regressor`, `weights`,
`shapedirs` and everything else are round-tripped untouched and asserted afterwards.

### `b_a_1` correction

The row that was in the file read `Y = [-180°, -180°]` — a zero-width interval pinned at a
half-turn, which is neither locked (`0/0`) nor free (`±180`). Confirmed as a clobbered Max
field; corrected to `[-180°, +180°]`, so the whole abdomen stays free. That matches the
literature position anyway: none of the three papers measure abdominal kinematics.

## ⚠ Blender cannot see these limits, and a re-export will delete them

The add-on's joint-limit path is **export-only**. `core_mesh.export_joint_limits_to_npy`
writes bone constraints out to the `.pkl`, but importing a `.pkl` never turns
`joint_limits` back into Limit Rotation constraints. So after running the generator, the
Bone Constraint panel is empty even though the array is present and correct.

That is cosmetic. This is not:

| Export Joint Limits | what happens to the authored array on re-export |
|---|---|
| unticked | `model_build.py:483` drops it as "stale `joint_limits` inherited from the loaded .pkl" |
| ticked | regenerated from the constraints Blender doesn't have → **every joint wide open** |

Either way the literature limits are lost. Run
[`3D_model_prep/apply_joint_limits_from_pkl.py`](../../3D_model_prep/apply_joint_limits_from_pkl.py)
in Blender's Text Editor **before** touching the model there: it reads `joint_limits` and
materialises it as `owner_space='LOCAL'` Limit Rotation constraints, so the panel, the
viewport overlay and a re-export all agree with the `.pkl`. Set `DRY_RUN = True` to report
without modifying anything.

It applies the exact inverse of `axis_remap.remap_bounds_to_model_frame`, using each bone's
own `B = rot3(bone.matrix_local)` rather than assuming one shared mapping — so it stays
correct if the rig turns out not to be uniform, and it prints any bone whose rest
orientation is a mixed-axis rotation (the exporter's issue-#56 caveat, where no axis-aligned
box is exact in either frame). That report is also the cheapest way to confirm whether
`XL=XG, YL=ZG, ZL=−YG` really does hold for every bone.

Verified in the sandbox against the repo's own exporter function: local → model → local
round-trips with **0 mismatches over all 48 signed-permutation matrices × 54 joints**
(2592 pairs), and the recovered bone-local values match the CSV's `local_*` columns
exactly. The Blender-side behaviour itself is untested — no Blender in the sandbox.

## Verification performed

`build_stick_joint_limits.py` asserts on every run:

- every `J_names` entry authored exactly once; array shape `(55, 3, 2)`
- root row all-zero; `min ≤ max` everywhere; all values finite and within ±π
- **not** all-free — the guide flags an all-±180° key as a hard error in 6D-rotation mode,
  since the penalty would then be identically zero for every possible prediction
- no axis narrower than 2·GIVE = 20° (catches the zero-width lock described below)
- `b_a_1` correction actually applied
- protraction/retraction lands exactly on ZG with the literature values, and left-side
  ranges are the exact negated mirror of the right
- the written `.pkl` is re-loaded and compared: `joint_limits` matches, and `v_template`
  (3015, 3), `f` (6009, 3), `J_names` and `J` are bit-identical to before the write

It also replicates the validation in `_ranges_from_joint_limits` (shape, `min ≤ max`,
finiteness) so failures surface here rather than at model-construction time.

### One defect this caught

The first generated pass produced `Z = [0°, 0°]` on every CTr/FTi/TiTa joint. The cause is
real, not a typo: the levation axis is exactly horizontal, so its projection onto the
dorsoventral axis is identically zero, and the envelope collapsed to a hard zero-width lock
— exactly the brittle case the ±10° give rule exists to avoid. The generator now applies a
give floor to every non-root axis after the projection step, and asserts on it.
