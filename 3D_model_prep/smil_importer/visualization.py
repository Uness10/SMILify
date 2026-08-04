"""Viewport visualization overlays (SMIL panel > Visualization).

Currently one overlay: per-bone rotation-limit wedges. For every armature in
the scene, a translucent wedge is drawn for each enabled axis of each bone's
Limit Rotation constraint (X red, Y green, Z blue; the Y/twist wedge is an
approximation by nature).

The overlay selects exactly the constraints the exporter reads
(:func:`core_mesh.export_joint_limits_to_npy`): muted, influence-0 and
``owner_space != 'LOCAL'`` constraints are skipped, so preview and export
always agree on WHICH limits apply. Wedges show the authored bone-local
angles; the export additionally remaps them into the model frame (exact for
signed-permutation rest orientations, verbatim + notice otherwise). The
IK-limit fallback and the wide-open default range are NOT visualised - a bone
without a wedge simply has no explicit constraint.

Also available as a self-contained Text Editor script:
``3D_model_prep/joint_rot_limit_vis.py`` (kept in sync with this module).

Uses only long-stable API (``gpu.shader`` ``UNIFORM_COLOR``,
``draw_handler_add``); targets Blender 4.2 LTS.
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

# Draw-handler handle; module-level so toggling and unregister() can remove it.
_handle = None
# Shader is created lazily: gpu.shader is unavailable in background mode, and
# the add-on must be able to register there (e.g. headless exports, tests).
_shader = None


def _get_shader():
    global _shader
    if _shader is None:
        _shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    return _shader


def _limit_frame(obj, pb):
    """World frame the local limits are measured against: the parent's current
    pose plus this bone's rest offset, so the wedge stays anchored as the bone
    rotates. Root bones fall back to the rest matrix."""
    mw = obj.matrix_world
    if pb.parent:
        offset = pb.parent.bone.matrix_local.inverted() @ pb.bone.matrix_local
        return mw @ pb.parent.matrix @ offset
    return mw @ pb.bone.matrix_local


def _wedge(centre, ref, sweep, a0, a1, r):
    """Triangle list for a filled fan sweeping `ref` toward `sweep`."""
    rim = []
    for i in range(SEG + 1):
        t = a0 + (a1 - a0) * i / SEG
        rim.append(centre + (math.cos(t) * ref + math.sin(t) * sweep) * r)
    tris = []
    for i in range(len(rim) - 1):
        tris += [tuple(centre), tuple(rim[i]), tuple(rim[i + 1])]
    return tris


def _draw():
    # The handler is session-global but the checkbox is per-scene, and undo /
    # scene switches can flip the property without firing its update= callback.
    # Gating here keeps what is DRAWN correct in every scene regardless of
    # handler state; the undo/redo sync below then reconciles the handler.
    scene = getattr(bpy.context, "scene", None)
    tool = getattr(scene, "smpl_tool", None) if scene is not None else None
    if tool is None or not tool.show_joint_limit_overlay:
        return
    shader = _get_shader()
    # Issue #56 (review): save the GPU state this handler touches and restore
    # it in a try/finally. Previously depth_test was never restored and blend
    # was reset to a hard-coded "NONE" (not the prior state), so any POST_VIEW
    # handler running after this one inherited broken state — and an exception
    # mid-loop skipped the restore entirely, leaking ALPHA blending for the
    # rest of the redraw.
    prev_blend = gpu.state.blend_get()
    prev_depth = gpu.state.depth_test_get()
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("NONE")  # always on top
    try:
        _draw_wedges(shader)
    finally:
        gpu.state.blend_set(prev_blend)
        gpu.state.depth_test_set(prev_depth)


def _draw_wedges(shader):
    for obj in bpy.context.scene.objects:
        if obj.type != "ARMATURE" or not obj.visible_get():
            # depth test is off, so wedges of hidden rigs would paint straight
            # over the viewport - skip anything not visible in this view layer.
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
            m = _limit_frame(obj, pb)
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
                batch = batch_for_shader(shader, "TRIS", {"pos": _wedge(centre, ref, sweep, a0, a1, rr)})
                shader.bind()
                shader.uniform_float("color", COLOUR[axis])
                batch.draw(shader)


def _tag_redraw_all_view3d():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def enable_overlay():
    global _handle
    if _handle is not None:
        return
    _handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_VIEW")
    _tag_redraw_all_view3d()


def disable_overlay():
    global _handle
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, "WINDOW")
        _handle = None
        _tag_redraw_all_view3d()


def toggle_overlay_update(self, context):
    """update= callback for SMPLProperties.show_joint_limit_overlay."""
    if self.show_joint_limit_overlay:
        enable_overlay()
    else:
        disable_overlay()


def _sync_overlay_to_property():
    """Align the draw-handler state with the saved checkbox (issue #56 review).

    ``show_joint_limit_overlay`` is a Scene property, so it is saved in the
    ``.blend`` — but the draw handler is session state and is not. Without this
    sync, opening a file saved with the checkbox ON shows a ticked checkbox
    that draws nothing, and disabling/re-enabling the add-on mid-session
    desyncs checkbox and handler.
    """
    if bpy.app.background:
        return  # no viewport; gpu draw handlers are pointless/unavailable
    scene = getattr(bpy.context, "scene", None)
    tool = getattr(scene, "smpl_tool", None) if scene is not None else None
    if tool is None:
        return
    if tool.show_joint_limit_overlay:
        enable_overlay()
    else:
        disable_overlay()


@bpy.app.handlers.persistent
def _load_post_restore_overlay(_filepath):
    """load_post handler: restore the overlay after opening a .blend."""
    _sync_overlay_to_property()


@bpy.app.handlers.persistent
def _undo_redo_restore_overlay(_scene):
    """undo_post/redo_post handler: Blender does not fire property update=
    callbacks on undo, so undoing across a toggle desyncs checkbox and
    handler - re-sync after every undo/redo."""
    _sync_overlay_to_property()


def register_handlers():
    """Called from the add-on's register(): install the load_post and
    undo/redo handlers and recover the overlay state for the already-loaded
    scene (covers disabling and re-enabling the add-on mid-session)."""
    if _load_post_restore_overlay not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_load_post_restore_overlay)
    if _undo_redo_restore_overlay not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(_undo_redo_restore_overlay)
    if _undo_redo_restore_overlay not in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.append(_undo_redo_restore_overlay)
    _sync_overlay_to_property()


def unregister_handlers():
    """Called from the add-on's unregister(): remove the app handlers."""
    if _load_post_restore_overlay in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_load_post_restore_overlay)
    if _undo_redo_restore_overlay in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.remove(_undo_redo_restore_overlay)
    if _undo_redo_restore_overlay in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.remove(_undo_redo_restore_overlay)
