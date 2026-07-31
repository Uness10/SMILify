"""
Minimal bone rotation-limit visualiser for Blender.

Paste into the Text Editor and press Run. For every armature in the scene, it
draws a translucent wedge for each enabled axis of each bone's Limit Rotation
constraint. Bones without a Limit Rotation constraint draw nothing.

Run the script again to refresh (it removes the previous overlay so handlers
do not stack). Restart Blender to clear it entirely.

Notes:
- Muted constraints and constraints with owner_space != 'LOCAL' are skipped,
  matching what the SMIL exporter writes into ``joint_limits`` - the overlay
  is a faithful preview of the export.
- Only explicit Limit Rotation constraints are shown; the exporter's IK-limit
  fallback and the wide-open default range are not visualised.
- Uses only long-stable API (``gpu.shader`` ``UNIFORM_COLOR``,
  ``draw_handler_add``); targets Blender 4.2 LTS (the blessed export
  environment) and also runs on newer versions (written against 5.0).

The installable add-on exposes the same overlay as a checkbox:
SMIL panel > Visualization > Show Joint Limit Overlay
(see smil_importer/visualization.py).
"""

import math

import bpy
import gpu
from gpu_extras.batch import batch_for_shader

SEG = 40  # arc resolution
COLOUR = {
    0: (0.90, 0.20, 0.20, 0.30),  # X red
    1: (0.20, 0.80, 0.20, 0.25),  # Y green (twist: approximate by nature)
    2: (0.20, 0.45, 0.90, 0.30),  # Z blue
}

shader = gpu.shader.from_builtin("UNIFORM_COLOR")


def limit_frame(obj, pb):
    """World frame the local limits are measured against: the parent's current
    pose plus this bone's rest offset, so the wedge stays anchored as the bone
    rotates. Root bones fall back to the rest matrix."""
    mw = obj.matrix_world
    if pb.parent:
        offset = pb.parent.bone.matrix_local.inverted() @ pb.bone.matrix_local
        return mw @ pb.parent.matrix @ offset
    return mw @ pb.bone.matrix_local


def wedge(centre, ref, sweep, a0, a1, r):
    """Triangle list for a filled fan sweeping `ref` toward `sweep`."""
    rim = []
    for i in range(SEG + 1):
        t = a0 + (a1 - a0) * i / SEG
        rim.append(centre + (math.cos(t) * ref + math.sin(t) * sweep) * r)
    tris = []
    for i in range(len(rim) - 1):
        tris += [tuple(centre), tuple(rim[i]), tuple(rim[i + 1])]
    return tris


def draw():
    # Save the GPU state this handler touches and restore it in a try/finally,
    # so later POST_VIEW handlers never inherit our blend/depth state — even if
    # an exception fires mid-loop (issue #56 review; kept in sync with
    # smil_importer/visualization.py).
    prev_blend = gpu.state.blend_get()
    prev_depth = gpu.state.depth_test_get()
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("NONE")  # always on top
    try:
        _draw_wedges()
    finally:
        gpu.state.blend_set(prev_blend)
        gpu.state.depth_test_set(prev_depth)


def _draw_wedges():
    for obj in bpy.context.scene.objects:
        if obj.type != "ARMATURE":
            continue
        for pb in obj.pose.bones:
            # Same selection rule as the exporter: first enabled (non-muted,
            # influence > 0), local-space Limit Rotation constraint.
            con = next(
                (
                    c
                    for c in pb.constraints
                    if c.type == "LIMIT_ROTATION" and not c.mute and c.influence > 0.0 and c.owner_space == "LOCAL"
                ),
                None,
            )
            if con is None:
                continue
            m = limit_frame(obj, pb)
            x = m.col[0].xyz.normalized()
            y = m.col[1].xyz.normalized()  # along the bone
            z = m.col[2].xyz.normalized()
            head = m.translation
            r = pb.length * 0.5

            jobs = []
            if con.use_limit_x:
                jobs.append((0, con.min_x, con.max_x, y, z, head, r))
            if con.use_limit_z:
                jobs.append((2, con.min_z, con.max_z, y, -x, head, r))
            if con.use_limit_y:  # twist: fan about the bone, offset so it reads
                jobs.append((1, con.min_y, con.max_y, x, -z, head, r * 0.6))

            for axis, a0, a1, ref, sweep, centre, rr in jobs:
                batch = batch_for_shader(shader, "TRIS", {"pos": wedge(centre, ref, sweep, a0, a1, rr)})
                shader.bind()
                shader.uniform_float("color", COLOUR[axis])
                batch.draw(shader)


# Swap out any handler left by a previous run, then register a fresh one.
ns = bpy.app.driver_namespace
old = ns.get("_bone_limit_handle")
if old is not None:
    bpy.types.SpaceView3D.draw_handler_remove(old, "WINDOW")
ns["_bone_limit_handle"] = bpy.types.SpaceView3D.draw_handler_add(draw, (), "WINDOW", "POST_VIEW")

for area in bpy.context.screen.areas:
    if area.type == "VIEW_3D":
        area.tag_redraw()

print("Bone limit overlay active. Re-run to refresh, restart Blender to clear.")
