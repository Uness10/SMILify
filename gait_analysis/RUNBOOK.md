# Runbook — producing the `.npz` exports and the gait figures on the cluster

Everything below is copy-paste once **Step 0** is filled in. Nothing here changes
model code; the only new inference flag is `--export_animation`.

---

## Step 0 — set the paths once

Fill these in and `source` them; every later step uses them.

```bash
cat > ~/smilify_gait.env <<'EOF'
export REPO=$HOME/SMILify                       # SMILify checkout on the cluster
export DATASET=/path/to/SMILySTICKS_centred_reprojected_FIXED.h5
export OUT=$REPO/inference_out
export FIGS=$REPO/gait_figs

# checkpoints
export CKPT_SV_REF=/path/to/sv_ref.pth
export CKPT_1E4=/path/to/1e-4.pth
export CKPT_1E1=/path/to/1e-1.pth

# rigs — sv_ref used the plain stick model, the two constrained runs the authored one
export STICK_PLAIN=$REPO/3D_model_prep/SMILy_STICK.pkl
export STICK_LIMITS=$REPO/3D_model_prep/SMILy_STICK_limits_authored.pkl

# true acquisition rate of the SMILySTICKS recording, NOT the MP4 playback rate.
# In dataset mode --fps only sets the output video rate and is what gets stamped
# into the .npz, so pass the real one or the ms/Hz numbers will be wrong.
export ACQ_FPS=60
EOF

source ~/smilify_gait.env
mkdir -p "$OUT" "$FIGS"
ls -l "$DATASET" "$CKPT_SV_REF" "$CKPT_1E4" "$CKPT_1E1" "$STICK_PLAIN" "$STICK_LIMITS"
```

That last `ls` is the check — if any line errors, fix it before going on.

---

## Step 1 — get `gait_analysis/` onto the cluster

It currently only exists in your local checkout. Either commit and pull:

```bash
# local machine
cd "C:\Users\youne\OneDrive\Desktop\FZJ\Stage\SMILify"
git add gait_analysis && git commit -m "Add gait analysis for SMILySTICKS runs" && git push

# cluster
cd $REPO && git pull
```

or copy it straight across:

```bash
# local machine
scp -r "C:\Users\youne\OneDrive\Desktop\FZJ\Stage\SMILify\gait_analysis" user@cluster:~/SMILify/
```

---

## Step 2 — check the analysis code before spending GPU time

Needs no checkpoint and no dataset; runs in seconds on a login node.

```bash
source ~/smilify_gait.env
cd $REPO
conda activate smilify            # or whatever env you use for inference

python gait_analysis/test_fk.py .
python gait_analysis/test_roundtrip.py "$STICK_PLAIN"
```

Expected: `all FK checks passed`, then eight `PASS` lines. `test_fk.py` checks the
analysis' forward kinematics against the repo's own
`smal_model/batch_lbs.batch_global_rigid_transformation`; `test_roundtrip.py`
synthesises a walk with known joint angles and checks they come back out.

Also confirm the two rigs share a rest pose — if they don't, absolute angles are
not on a common baseline across runs (see Step 6):

```bash
python - <<'PY'
import sys, os; sys.path.insert(0, "gait_analysis")
import smil_fk, anatomical as ana
for p in (os.environ["STICK_PLAIN"], os.environ["STICK_LIMITS"]):
    m = smil_fk.load_model(p)
    print(f"\n{os.path.basename(p)}  ({m.n_joints} joints, {m.shapedirs.shape[2]} betas)")
    for leg in ana.LEGS:
        r = ana.rest_angles(m, *leg)
        print(f"  {ana.LEG_LABEL[leg]:<11}"
              + "".join(f"{r[k]:8.1f}" for k in ("alpha", "beta", "gamma", "delta")))
PY
```

---

## Step 3 — export the pose for each run

One command per run. Every flag except `--export_animation` should match what
produced the existing MP4s — if you used `--smoothing_window` or
`--generate_num_subclips` then, use them here too (see the notes below).

```bash
source ~/smilify_gait.env
cd $REPO

# --- sv_ref (plain rig) ---
python smal_fitter/neuralSMIL/run_singleview_inference.py \
    --checkpoint "$CKPT_SV_REF" \
    --smal_file  "$STICK_PLAIN" \
    --dataset    "$DATASET" \
    --output_folder "$OUT" \
    --view_indices 0 \
    --max_frames 500 \
    --generate_num_subclips 1 \
    --fps $ACQ_FPS \
    --export_animation "$OUT/SMILySTICKS_centred_reprojected_FIXED_singleview_inference_sv_ref"

# --- 1e-4 (authored-limits rig) ---
python smal_fitter/neuralSMIL/run_singleview_inference.py \
    --checkpoint "$CKPT_1E4" \
    --smal_file  "$STICK_LIMITS" \
    --dataset    "$DATASET" \
    --output_folder "$OUT" \
    --view_indices 0 \
    --max_frames 500 \
    --generate_num_subclips 1 \
    --fps $ACQ_FPS \
    --export_animation "$OUT/SMILySTICKS_centred_reprojected_FIXED_singleview_inference_1e-4"

# --- 1e-1 (authored-limits rig) ---
python smal_fitter/neuralSMIL/run_singleview_inference.py \
    --checkpoint "$CKPT_1E1" \
    --smal_file  "$STICK_LIMITS" \
    --dataset    "$DATASET" \
    --output_folder "$OUT" \
    --view_indices 0 \
    --max_frames 500 \
    --generate_num_subclips 1 \
    --fps $ACQ_FPS \
    --export_animation "$OUT/SMILySTICKS_centred_reprojected_FIXED_singleview_inference_1e-1"
```

Each run prints a line like

```
  Animation export: .../..._sv_ref.npz + .../..._sv_ref.json (500 frames)
```

Three things worth knowing about this export path:

* **Frame numbering.** `--view_indices 0` + `--generate_num_subclips 1` means no
  `_view*` / `_frames*` suffix is added, and `compute_subclip_ranges` returns
  `(0, 500)` — so npz frame *i* is dataset frame *i*, the same numbering the
  recommended windows use. If you pass `--generate_num_subclips > 1`, the clip
  does **not** start at 0 and you must pass `--frame-offset <start>` in Step 5.
* **The export is pre-smoothing.** The recorder runs in Phase 1b, before
  `PredictionSmoother`. The MP4 is rendered after it. So with
  `--smoothing_window > 0` the npz is the raw regressor output while the video is
  smoothed — same frames, slightly different values. That is usually what you
  want for joint angles, but say which you used when you show the plots.
* **`--fps` is cosmetic for the maths.** It only sets the video rate and the `fps`
  field in the npz. Angles and trajectories are unaffected; only the time axis and
  the printed step periods use it. You can also override it later with
  `analyse_gait.py --fps`.

### As a Slurm job

```bash
sbatch --job-name=smil_export --gres=gpu:1 --time=02:00:00 --mem=32G \
       --output=$OUT/export_%j.log \
       --wrap="source ~/smilify_gait.env && cd \$REPO && conda run -n smilify bash gait_analysis/run_on_cluster.sh"
```

`run_on_cluster.sh` runs Step 3 and Step 5 for all three checkpoints; edit the
`RUNS` map at its top with your checkpoint paths first.

---

## Step 4 — check the exports before plotting

```bash
source ~/smilify_gait.env
python - <<'PY'
import json, glob, os, numpy as np
for f in sorted(glob.glob(os.path.join(os.environ["OUT"], "*inference_*.npz"))):
    d = np.load(f); m = json.load(open(f[:-4] + ".json"))
    print(f"\n{os.path.basename(f)}")
    print(f"  frames {d['poses'].shape[0]}  joints {d['poses'].shape[1]}  fps {float(d['fps'])}")
    print(f"  arrays: {', '.join(sorted(d.files))}")
    print(f"  rotation: {m['rotation_representation']}   root joint: {m['joint_names'][0]}")
    print(f"  checkpoint: {m.get('source_checkpoint')}")
    print(f"  max |pose| = {np.abs(d['poses']).max():.3f} rad   any NaN: {bool(np.isnan(d['poses']).any())}")
PY
```

Want: 500 frames, 55 joints, `poses` present, no NaNs, and the `source_checkpoint`
line matching the run you think it is.

---

## Step 5 — run the gait analysis

```bash
source ~/smilify_gait.env
cd $REPO
S="$OUT/SMILySTICKS_centred_reprojected_FIXED_singleview_inference"

python gait_analysis/analyse_gait.py --npz "${S}_sv_ref.npz" --smal-file "$STICK_PLAIN" \
    --windows 32-196 372-495 --fps $ACQ_FPS --label sv_ref --out "$FIGS"

python gait_analysis/analyse_gait.py --npz "${S}_1e-4.npz"  --smal-file "$STICK_LIMITS" \
    --windows 32-196 372-495 --fps $ACQ_FPS --label 1e-4 --out "$FIGS"

python gait_analysis/analyse_gait.py --npz "${S}_1e-1.npz"  --smal-file "$STICK_LIMITS" \
    --windows 32-196 372-495 --fps $ACQ_FPS --label 1e-1 --out "$FIGS"
```

Add `--propagate-scaling` to all three if the checkpoints were trained with
`propagate_scaling=True` — it changes how per-joint scales compose down the leg.

Each run prints its rig, its rest pose, and a per-leg step summary, then writes
six figures per window plus four CSVs into `$FIGS`.

---

## Step 6 — read the printout before you trust the figures

Three checks, in order:

1. **Rest-pose block.** Compare it across the three runs (and against Step 2). If
   `sv_ref`'s rest pose differs from the other two, their `*_abs` angle columns
   sit on different baselines — compare the `*_rel` columns in
   `angles_<run>.csv` instead, which are relative to each rig's own rest pose.
2. **Step periods.** All six legs should land near **41 frames**, which is what
   the footage gives. A run that disagrees is tracking the animal differently,
   not walking differently.
3. **Do the model's steps land where the animal's do?** Compare the predicted
   liftoffs against the ones measured from the footage:

   ```bash
   python gait_analysis/find_gait_windows.py \
       "$OUT/..._sv_ref.mp4" --panel 0 --exclude l_1_l --out "$FIGS/footage_check.png"
   head -20 "$FIGS/steps_sv_ref.csv"
   ```

   `--panel 0` reads the ground-truth reprojected markers, `--panel 1` the model's
   own. Liftoff frames should agree to within a few frames per leg.

---

## Step 7 — bring the results back

```bash
# local machine
scp -r user@cluster:~/SMILify/gait_figs "C:\Users\youne\OneDrive\Desktop\FZJ\Stage\SMILify\Claude outputs\gait_analysis\"
```

Per run and window: `fig1` (ThC protraction + levation), `fig2` (CTr and FTi hinge
angles), `fig3` (which joints move, about which axes), `fig4`/`fig5` (tarsus
trajectory relative to the root, one trace per cycle), `fig6` (gait diagram), plus
`angles_*.csv`, `tarsi_*.csv`, `steps_*.csv`, `axis_dominance_*.csv`.
