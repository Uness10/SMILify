# Defining Joint Limits in Blender — User Guide

This guide shows, step by step, how to tell SMILify how far each joint of your model is allowed to rotate — e.g. "this knee bends between −30° and +45°". No coding needed: you author the limits in Blender, preview them in the viewport, export, and both fitting pipelines respect them automatically.

Technical details live in [issue56_implementation.md](design/issue56_implementation.md).

> 📷 **Screenshots.** Every image below is a numbered placeholder (`SCREENSHOT 01` … `SCREENSHOT 12`) with a caption describing exactly what to capture. The full shot list is in the [appendix](#appendix-screenshot-shot-list). Save the files under `docs/design/images/joint_limits/` with the exact filenames given.

---

## 1. Why joint limits?

When SMILify fits your model to images, it searches for a pose that matches. Without limits, it can find *impossible* poses — a leg bent backwards through the body. Joint limits fence off those poses: any joint that goes past its range gets penalised and is pulled back.

If you never set a limit on a joint, it stays "wide open" (free to rotate). You only need to author limits where anatomy demands them.

## 2. The one concept you must get right: axes and signs

Every joint can rotate around three axes — **X, Y, Z** — and each axis gets its own range `[Min, Max]`:

- **Min** = how far it can rotate in the *negative* direction (a negative number, e.g. −30°).
- **Max** = how far in the *positive* direction (e.g. +45°).
- `Min = Max = 0` means the axis is **locked** (no rotation at all).

**Which direction is "positive"?** That depends on the bone's own local axes — every bone carries its own little XYZ frame. Before typing numbers, always *look* at the bone's axes and *test-rotate* it (§4). Never guess the sign.

![SCREENSHOT 01 — Bone local axes](design/images/joint_limits/01_bone_local_axes.png)
> 📷 **SCREENSHOT 01** (`01_bone_local_axes.png`): Viewport in Pose Mode, one leg bone selected, with per-bone axes displayed (the small coloured X/Y/Z lines on the bone). Zoom close enough that the three axis lines are clearly readable. Annotate which colour is which axis (X red, Y green, Z blue).

To display bone axes: select the armature → **Object Data Properties** (green stick-figure tab) → **Viewport Display** → tick **Axes**.

![SCREENSHOT 02 — Enabling axes display](design/images/joint_limits/02_enable_axes_display.png)
> 📷 **SCREENSHOT 02** (`02_enable_axes_display.png`): Properties editor, Object Data Properties tab open, Viewport Display panel expanded, the **Axes** checkbox ticked. Draw a box or arrow around the checkbox.

## 3. Setup (once)

1. Build and install the add-on:

   ```bash
   cd 3D_model_prep
   python build_addon.py
   ```

   In Blender: **Edit → Preferences → Add-ons → Install…**, pick `smil_importer.zip`, enable it.
2. Open your rigged model (or import a `.pkl` model via the add-on), then **save the `.blend` file to a normal, writable folder**. An unsaved file makes the exporter fail with "Permission denied" (pitfall #1).

![SCREENSHOT 03 — Add-on enabled](design/images/joint_limits/03_addon_enabled.png)
> 📷 **SCREENSHOT 03** (`03_addon_enabled.png`): Blender Preferences → Add-ons window, searched for "SMIL", showing the SMIL Model Importer installed with its checkbox **ticked** (no warning triangle).

![SCREENSHOT 04 — SMIL panel with model loaded](design/images/joint_limits/04_smil_panel_model_loaded.png)
> 📷 **SCREENSHOT 04** (`04_smil_panel_model_loaded.png`): The 3D viewport with your model imported and the sidebar (**N**) open on the **SMIL** tab, so the whole panel is visible. This orients the reader: "this is what your screen should look like before you start."

## 4. Find the right axis and sign (the sign-posting step)

For the joint you want to limit:

1. Select the **armature**, switch to **Pose Mode** (top-left dropdown or `Ctrl+Tab`).
2. Click the bone.
3. Test-rotate it around one *local* axis at a time: press `R`, then the axis letter **twice** (e.g. `R` `X` `X` — pressing twice selects the bone's *local* axis, once gives the global axis, which is not what you want). Move the mouse and watch which way the limb swings.
4. Note the direction: the way it swings when you move toward *positive* angles is your **Max** direction; the opposite is **Min**. Press `Esc` to cancel the rotation without keeping it.
5. Repeat for `R Y Y` and `R Z Z` until you know what each axis does for this bone.

![SCREENSHOT 05 — Test-rotating around a local axis](design/images/joint_limits/05_test_rotate_local_axis.png)
> 📷 **SCREENSHOT 05** (`05_test_rotate_local_axis.png`): Pose Mode, a leg bone mid-rotation using `R X X`, with the angle readout visible in the viewport header (top-left corner shows e.g. "Rot: −37.5° along local X"). Capture it at a clearly negative angle. Caption in the image: "R, X, X — rotating around the bone's LOCAL X; the header shows the current angle and its sign."

Tip: the angle readout in the viewport corner shows the current angle *with its sign* while you rotate — this tells you directly whether "knee bends forward" is positive or negative on that axis.

## 5. Author the limit

Still in Pose Mode with the bone selected:

1. Open the **Bone Constraint** tab in the Properties editor — the icon is a **bone with a wrench**. Check the panel header says **"Add Bone Constraint"**, *not* "Add Object Constraint" (that one is a different tab and the exporter never reads it — pitfall #2).
2. Click **Add Bone Constraint → Limit Rotation**.
3. For each axis you want to limit:
   - **Tick the axis checkbox** (Limit X / Limit Y / Limit Z). An unticked axis is ignored and exports as wide-open, even if you typed numbers into its fields (pitfall #3).
   - Enter **Min** and **Max** in **degrees** (e.g. Min = −90, Max = 0).
4. Leave axes you don't want to limit unticked.
5. To **lock** an axis completely, tick it and set Min = Max = 0.
6. Set **Owner = Local Space** (see below). The default is World Space, which the exporter (and the viewport preview) will *skip* with a warning.

![SCREENSHOT 06 — Bone Constraint tab vs Object Constraint tab](design/images/joint_limits/06_bone_vs_object_constraint_tab.png)
> 📷 **SCREENSHOT 06** (`06_bone_vs_object_constraint_tab.png`): Properties editor with BOTH constraint tabs visible in the icon column. Circle the correct **Bone Constraint** tab (bone + wrench icon) in green; cross out the wrong **Object Constraint** tab (plain wrench) in red.

![SCREENSHOT 07 — A filled-in Limit Rotation constraint](design/images/joint_limits/07_limit_rotation_panel.png)
> 📷 **SCREENSHOT 07** (`07_limit_rotation_panel.png`): The added Limit Rotation constraint panel with **Limit X ticked, Min = −90°, Max = 0°**, Y and Z unticked. Arrows pointing at (a) the tick-box and (b) the Min/Max fields. Use these same values in all later screenshots so the guide tells one consistent story.

### The "Owner" space setting (important)

At the bottom of the Limit Rotation panel there's an **Owner** dropdown. It's the coordinate space Blender measures the bone's rotation in before clamping it to your Min/Max:

- **World Space** (default) — measured against the global scene axes.
- **Pose Space** — relative to the armature's pose.
- **Local With Parent** — the bone's local frame, including its parent's rest orientation.
- **Local Space** — the bone's own local rest frame, ignoring the parent.
- **Custom Space** — relative to another object you pick.

**Set Owner = Local Space.** The exporter reads the raw Min/Max values and treats them as bone-local, then converts them into the model frame itself. Local Space is what makes "tick X, −90/0" mean "this bone rotates around *its own* X", which matches the test-rotate step in §4 (`R X X` uses the bone's local axis). Constraints left in any other Owner space are **skipped with a warning** at export — they will not land in the model at all.

![SCREENSHOT 08 — Owner set to Local Space](design/images/joint_limits/08_owner_local_space.png)
> 📷 **SCREENSHOT 08** (`08_owner_local_space.png`): The Limit Rotation panel scrolled to the bottom, the **Owner** dropdown open, **Local Space** highlighted/selected. Add a note in the image: "Owner = Local Space (NOT the default World Space)".

Sanity check: with the constraint in place, test-rotate the bone again (`R X X`) — it should now visibly stop at your limits. If it stops in the wrong place, your sign is flipped: swap and negate (e.g. wrong `[0, 90]` → right `[-90, 0]`).

## 6. Visualize your authored limits (optional but recommended)

Before exporting you can preview every authored limit directly in the viewport: for each enabled axis of each bone's Limit Rotation constraint, a translucent wedge is drawn showing the allowed range (X = red, Y = green, Z = blue; the Y/twist wedge is an approximation by nature). The preview reads exactly the constraint fields the exporter reads, so what you see is what will land in `joint_limits`.

Two ways to enable it:

1. **Add-on panel (recommended):** in the SMIL panel, expand the **Visualization** sub-panel and tick **Show Joint Limit Overlay**. Untick to remove the overlay.
2. **Standalone script:** open `3D_model_prep/joint_rot_limit_vis.py` in Blender's Text Editor and press **Run**. Re-run to refresh after editing constraints; restart Blender to clear.

![SCREENSHOT 09 — Limit overlay in the viewport](design/images/joint_limits/09_limit_overlay.png)
> 📷 **SCREENSHOT 09** (`09_limit_overlay.png`): Viewport in Pose Mode showing the constrained leg bone with its **red translucent wedge** sweeping the −90°→0° range around the bone's local X, AND the SMIL panel's Visualization sub-panel visible with **Show Joint Limit Overlay** ticked. This is the guide's hero image — frame it so both the wedge and the checkbox are readable.

![SCREENSHOT 10 — Muted constraint draws nothing](design/images/joint_limits/10_muted_constraint_no_wedge.png)
> 📷 **SCREENSHOT 10** (`10_muted_constraint_no_wedge.png`): Same viewpoint as SCREENSHOT 09, but with the constraint **muted** (click the mute toggle in the constraint header) — the wedge is gone. Show the constraint panel with the mute toggle highlighted. Caption: "Muted constraints are skipped by the preview AND by the exporter."

Caveats:

- Only explicit, enabled (non-muted), local-space **Limit Rotation** constraints are drawn — exactly the constraints the exporter reads, so preview and export agree.
- The IK-limit fallback and the wide-open default range are **not** visualised; a bone with no wedge simply has no explicit constraint.
- If a wedge points the wrong way, your sign is flipped — fix it now, before export (swap and negate, see §5).

## 7. Export

1. Select the **mesh object** (not the armature — pitfall #6).
2. In the SMIL panel, keep **Export Joint Limits** ticked (default), set the **Output Filename**, click **Export SMIL Model**.
3. Your limits are now stored inside the `.pkl` under the `joint_limits` key.

![SCREENSHOT 11 — Export panel](design/images/joint_limits/11_export_panel.png)
> 📷 **SCREENSHOT 11** (`11_export_panel.png`): The SMIL panel with **Export Joint Limits** ticked, the **Default Joint Limit Range (rad)** field visible below it, the Output Filename field, and the **Export SMIL Model** button. Draw a highlight around the Export Joint Limits toggle.

You do **not** need to worry about the bone's rest orientation: the exporter automatically converts your bone-local limits into the model's frame (for standard axis-aligned rigs the conversion is exact; a tilted, mixed-axis bone prints a warning and exports the numbers as-is).

## 8. Verify the export (optional but recommended)

```python
import numpy as np
from smal_model.smal_torch import load_smal_model   # chumpy-safe loader
dd = load_smal_model("your_model.pkl")
jl = np.asarray(dd["joint_limits"])
print(jl.shape)                          # -> (J, 3, 2)
i = dd["J_names"].index("your_bone_name")
print(jl[i])                             # your limits, in RADIANS
print(jl[0])                             # root -> all zeros
```

The `.pkl` stores **radians**: −90° appears as `-1.5708`, and unconstrained axes as `[-3.1416, 3.1416]`. That's expected, not a bug (pitfall #4).

![SCREENSHOT 12 — Verified .pkl output](design/images/joint_limits/12_pkl_verification.png)
> 📷 **SCREENSHOT 12** (`12_pkl_verification.png`): A terminal running the snippet above against your exported model, output visible: the `(J, 3, 2)` shape, the constrained bone's row showing `[-1.5708, 0.]` on X and `[-3.1416, 3.1416]` on Y/Z, and the all-zero root row. Use the same bone as the earlier screenshots.

## 9. Use the limits

- **Optimisation fitter:** point `config.SMAL_FILE` at your exported `.pkl` and run with the limit weight on (`w_limit > 0`). Nothing else to configure.
- **Neural training (optional):** add `"joint_limit_regularization": 1e-3` (start small) to `loss_weights` in your training config. Default is `0.0` = off. If you enable it and the model has no usable `joint_limits`, training stops with a clear error instead of silently ignoring it.

## 10. Common pitfalls (all seen in real testing)

1. **Unsaved `.blend`** → "Permission denied" on export. Save to a writable folder first.
2. **Object constraint instead of Bone constraint** → limit silently ignored. Use the bone+wrench tab.
3. **Unticked axis** → Min/Max fields are ignored; the axis exports wide-open. Tick the checkbox.
4. **Degrees vs radians** → Blender UI is degrees; the `.pkl` is radians.
5. **Wrong file checked** → export writes to the Output Filename on the machine running Blender; verify *that* file.
6. **Armature active at export** → "No valid mesh object selected." Select the mesh first.
7. **Flipped sign** → limit stops the bone on the wrong side. Test-rotate (§4/§5), check the wedge (§6), and swap-negate the bounds.
8. **Owner left at World Space** → the constraint is skipped with a warning and never exported. Set Owner = Local Space (§5). The overlay (§6) makes this obvious: a skipped constraint draws no wedge.
9. **Muted constraint** → skipped by both the preview and the exporter; the bone falls back to its IK limits or the wide-open default.

---

## Appendix: screenshot shot list

Take all shots with the **same model and the same constrained bone** (Limit X, Min = −90°, Max = 0°, Owner = Local Space) so the guide tells one consistent story. Save as PNG under `docs/design/images/joint_limits/`.

| # | Filename | What to capture |
|---|----------|-----------------|
| 01 | `01_bone_local_axes.png` | Pose Mode, bone selected, per-bone XYZ axes visible; annotate axis colours |
| 02 | `02_enable_axes_display.png` | Object Data Properties → Viewport Display → **Axes** ticked, highlighted |
| 03 | `03_addon_enabled.png` | Preferences → Add-ons, SMIL Model Importer enabled, no warning icon |
| 04 | `04_smil_panel_model_loaded.png` | Viewport with model + sidebar open on the SMIL tab |
| 05 | `05_test_rotate_local_axis.png` | Bone mid `R X X` rotation, signed angle readout in the header |
| 06 | `06_bone_vs_object_constraint_tab.png` | Bone Constraint tab circled green, Object Constraint tab crossed red |
| 07 | `07_limit_rotation_panel.png` | Limit Rotation: Limit X ticked, −90°/0°, arrows at tick-box and fields |
| 08 | `08_owner_local_space.png` | Owner dropdown open, **Local Space** selected, annotated |
| 09 | `09_limit_overlay.png` | Red wedge on the bone + Visualization sub-panel with overlay ticked |
| 10 | `10_muted_constraint_no_wedge.png` | Same view, constraint muted, wedge gone, mute toggle highlighted |
| 11 | `11_export_panel.png` | Export Joint Limits ticked, default range field, export button |
| 12 | `12_pkl_verification.png` | Terminal output of the §8 snippet: shape, bone row, zero root row |
