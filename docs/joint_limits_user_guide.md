# Defining Joint Limits in Blender — User Guide

This guide shows, step by step, how to tell SMILify how far each joint of your model is allowed to rotate — e.g. "this knee bends between −30° and +45°". No coding needed: you draw the limits in Blender, export, and both fitting pipelines respect them automatically.

Technical details live in [issue56_implementation.md](design/issue56_implementation.md).

> 📷 All images below are placeholders — screenshots to be added. Each placeholder describes exactly what the photo should show.

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

![PHOTO: Blender viewport, one bone selected in Pose Mode with its local axes displayed (colored X/Y/Z axis lines visible on the bone). Annotate which color is which axis.](design/images/1.png)

To display bone axes: select the armature → **Object Data Properties** (green stick-figure tab) → **Viewport Display** → tick **Axes**.

![PHOTO: Properties editor, Object Data Properties tab, Viewport Display panel with the "Axes" checkbox ticked, highlighted with a box or arrow.](design/images/joint_limits/02_enable_axes_display.png)

## 3. Setup (once)

1. Build and install the add-on:

   ```bash
   cd 3D_model_prep
   python build_addon.py
   ```

   In Blender: **Edit → Preferences → Add-ons → Install…**, pick `smil_importer.zip`, enable it.
2. Open your rigged model (or import a `.pkl` model via the add-on), then **save the `.blend` file to a normal, writable folder**. An unsaved file makes the exporter fail with "Permission denied" (pitfall #1).

![PHOTO: Blender Preferences > Add-ons window with the SMIL importer add-on installed and its checkbox enabled.](design/images/3.png)
![](design/images/4.png)
![](design/images/5.png)

## 4. Find the right axis and sign (the sign-posting step)

For the joint you want to limit:

1. Select the **armature**, switch to **Pose Mode** (top-left dropdown or `Ctrl+Tab`).
2. Click the bone.
3. Test-rotate it around one *local* axis at a time: press `R`, then the axis letter **twice** (e.g. `R` `X` `X` — pressing twice selects the bone's *local* axis, once gives the global axis, which is not what you want). Move the mouse and watch which way the limb swings.
4. Note the direction: the way it swings when you move toward *positive* angles is your **Max** direction; the opposite is **Min**. Press `Esc` to cancel the rotation without keeping it.
5. Repeat for `R Y Y` and `R Z Z` until you know what each axis does for this bone.

![PHOTO: Pose Mode, a leg bone mid-rotation using R Z Z, with the rotation angle readout visible in the viewport header/corner. Caption: "R, Z, Z — rotating around the bone's LOCAL Z; the header shows the current angle and its sign."](design/images/2.png)

Tip: the angle readout in the viewport corner shows the current angle *with its sign* while you rotate — this tells you directly whether "knee bends forward" is positive or negative on that axis.

## 5. Author the limit

Still in Pose Mode with the bone selected:

1. Open the **Bone Constraint** tab in the Properties editor — the icon is a **bone with a wrench**. Check the panel header says **"Add Bone Constraint"**, *not* "Add Object Constraint" (that one is a different tab and the exporter never reads it — pitfall #2).
2. Click **Add Bone Constraint → Limit Rotation**.
3. For each axis you want to limit:
   - **Tick the axis checkbox** (Limit X / Limit Y / Limit Z). An unticked axis is ignored and exports as wide-open, even if you typed numbers into its fields (pitfall #3).
   - Enter **Min** and **Max** in **degrees** (e.g. Min = −30, Max = 45).
4. Leave axes you don't want to limit unticked.
5. To **lock** an axis completely, tick it and set Min = Max = 0.
6. Set **Owner = Local Space** (see below). The default is World Space, which will export the wrong numbers.

### The "Owner" space setting (important)

At the bottom of the Limit Rotation panel there's an **Owner** dropdown. It's the coordinate space Blender measures the bone's rotation in before clamping it to your Min/Max:

- **World Space** (default) — measured against the global scene axes.
- **Pose Space** — relative to the armature's pose.
- **Local With Parent** — the bone's local frame, including its parent's rest orientation.
- **Local Space** — the bone's own local rest frame, ignoring the parent.
- **Custom Space** — relative to another object you pick.

**Set Owner = Local Space.** The exporter reads the raw Min/Max values and treats them as bone-local, then converts them into the model frame itself. Local Space is what makes "tick Z, −30/+45" mean "this bone rotates around *its own* Z", which matches the test-rotate step in §4 (`R Z Z` uses the bone's local axis). Any other Owner space measures against different axes, so your exported limits will fence off the wrong rotations.

![PHOTO: The Limit Rotation constraint panel scrolled to the bottom, "Owner" dropdown open with "Local Space" highlighted/selected. Add a note: "Owner = Local Space (NOT the default World Space)".](design/images/joint_limits/09_owner_local_space.png)

![PHOTO: Properties editor with BOTH constraint tabs visible; the correct "Bone Constraint" tab (bone+wrench icon) circled in green, the wrong "Object Constraint" tab crossed out in red.](design/images/joint_limits/05_bone_vs_object_constraint_tab.png)

![PHOTO: An added Limit Rotation constraint panel: "Limit Z" ticked with Min = -30°, Max = 45°, X and Y unticked. Arrows pointing at the tick-box and the Min/Max fields.](design/images/joint_limits/06_limit_rotation_panel.png)

Sanity check: with the constraint in place, test-rotate the bone again (`R Z Z`) — it should now visibly stop at your limits. If it stops in the wrong place, your sign is flipped: swap and negate (e.g. wrong `[-45, 30]` → right `[-30, 45]`).

![PHOTO: Two side-by-side viewport shots of the same leg: left at the Min limit (-30°), right at the Max limit (+45°), showing the bone stopping at each end.](design/images/joint_limits/07_limit_stops_min_max.png)

## 6. Visualize your authored limits (optional)

Before exporting you can preview every authored limit directly in the viewport: for each enabled axis of each bone's Limit Rotation constraint, a translucent wedge is drawn showing the allowed range (X = red, Y = green, Z = blue; the Y/twist wedge is an approximation by nature). The preview reads exactly the constraint fields the exporter reads, so what you see is what will land in `joint_limits`.

Two ways to enable it:

1. **Add-on panel (recommended):** in the SMIL panel, open the **Visualization** box and tick **Show Joint Limit Overlay**. Untick to remove the overlay.
2. **Standalone script:** open `3D_model_prep/joint_rot_limit_vis.py` in Blender's Text Editor and press **Run**. Re-run to refresh after editing constraints; restart Blender to clear.

![PHOTO: Viewport in Pose Mode showing a leg bone with a red translucent wedge spanning -30 deg to +45 deg around its local X axis, plus the SMIL panel's Visualization box with "Show Joint Limit Overlay" ticked.](design/images/joint_limits/10_limit_overlay.png)

Caveats:

- Only explicit, enabled (non-muted) **Limit Rotation** constraints are drawn. Muted constraints are skipped - and the exporter skips them too, so preview and export agree.
- The IK-limit fallback and the wide-open default range are **not** visualized; a bone with no wedge simply has no explicit constraint.
- If a wedge points the wrong way, your sign is flipped - fix it now, before export (swap and negate, see §5).

## 7. Export

1. Select the **mesh object** (not the armature — pitfall #6).
2. In the SMIL panel, keep **Export Joint Limits** ticked (default), set the **Output Filename**, click **Export SMIL Model**.
3. Your limits are now stored inside the `.pkl` under the `joint_limits` key.

![PHOTO: The SMIL add-on panel in the sidebar with "Export Joint Limits" ticked, the "Default Joint Limit Range" field below it, the output filename field, and the "Export SMIL Model" button. Highlight the toggle.](design/images/6.png)

You do **not** need to worry about the bone's rest orientation: the exporter automatically converts your bone-local limits into the model's frame (for standard axis-aligned rigs the conversion is exact; a tilted, mixed-axis bone prints a warning and exports the numbers as-is).

### Verify the export (optional but recommended)

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

The `.pkl` stores **radians**: −30° appears as `-0.5236`, +45° as `0.7854`. That's expected, not a bug (pitfall #4).

## 8. Use the limits

- **Optimisation fitter:** point `config.SMAL_FILE` at your exported `.pkl` and run with the limit weight on (`w_limit > 0`). Nothing else to configure.
- **Neural training (optional):** add `"joint_limit_regularization": 1e-3` (start small) to `loss_weights` in your training config. Default is `0.0` = off.

## 9. Common pitfalls (all seen in real testing)

1. **Unsaved `.blend`** → "Permission denied" on export. Save to a writable folder first.
2. **Object constraint instead of Bone constraint** → limit silently ignored. Use the bone+wrench tab.
3. **Unticked axis** → Min/Max fields are ignored; the axis exports wide-open. Tick the checkbox.
4. **Degrees vs radians** → Blender UI is degrees; the `.pkl` is radians.
5. **Wrong file checked** → export writes to the Output Filename on the machine running Blender; verify *that* file.
6. **Armature active at export** → "No valid mesh object selected." Select the mesh first.
7. **Flipped sign** → limit stops the bone on the wrong side. Test-rotate (§4/§5) and swap-negate the bounds.
8. **Owner left at World Space** → limits are measured against global axes, not the bone, so the exported bounds fence off the wrong rotations. Set Owner = Local Space (§5).
