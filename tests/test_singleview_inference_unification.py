"""Tests for the unified single-view / multi-view inference conventions (issue #100).

``run_singleview_inference.py`` used to be a raw-image/video-only entrypoint whose
render path had drifted away from ``run_multiview_inference.py``. Both now share
``smal_fitter/neuralSMIL/inference_common.py``, which owns the three rules that
must not diverge again:

* **camera intrinsics / extrinsics** — a ``camera_centric`` checkpoint renders
  through the FIXED PyTorch3D identity camera with the FOV (and aspect ratio)
  from the sample's calibration; a ``model_centric`` one renders through its own
  predicted camera.
* **shape-space variation** — ``log_beta_scales`` / ``betas_trans`` are PCA
  weights ``(B, N_BETAS)`` under ``separate`` + ``use_pca_transformation`` and
  must be expanded to per-joint ``(B, J, 3)`` values, but are already per-joint
  under ``separate``-without-PCA and ``entangled_with_betas``, and absent under
  ``ignore``. Assigning the raw PCA weights straight onto the fitter (the old
  single-view behaviour) silently produced a mesh whose limb scaling had nothing
  to do with the prediction.
* **mesh scaling** — legacy 10x UE placement vs. the predicted per-sample
  ``mesh_scale`` vs. translation only.

These tests use lightweight stand-ins for the model / fitter so they run on CPU
without a checkpoint, a dataset or pytorch3d.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from smal_fitter.neuralSMIL.inference_common import (  # noqa: E402
    PredictionSmoother,
    apply_shape_space_params,
    compute_subclip_ranges,
    hdf5_is_multiview,
    place_mesh,
    resolve_frame_convention,
    resolve_mesh_scale,
    resolve_render_camera,
    resolve_singleview_dataset_kwargs,
)
from smal_fitter.neuralSMIL.inference_ddp import (  # noqa: E402
    compute_rank_indices,
    force_ipv4_getaddrinfo,
    gather_predictions,
    is_torchrun_launched,
    merge_frame_streams_from_temp,
    resolve_dist_timeout,
    resolve_launch,
    write_frame_streams_to_temp,
)
from smal_fitter.neuralSMIL.training_config import TrainingConfig  # noqa: E402


N_JOINTS = 7
N_BETAS = 5


# --------------------------------------------------------------------------- #
# Lightweight stand-ins
# --------------------------------------------------------------------------- #
class FakeModel:
    """Minimal duck-type of SMILImageRegressor for the convention helpers."""

    def __init__(
        self,
        scale_trans_mode="separate",
        fixed_camera=False,
        allow_mesh_scaling=False,
        use_ue_scaling=False,
        rotation_representation="6d",
    ):
        self.scale_trans_mode = scale_trans_mode
        self.fixed_camera = fixed_camera
        self.allow_mesh_scaling = allow_mesh_scaling
        self.use_ue_scaling = use_ue_scaling
        self.rotation_representation = rotation_representation
        self.pca_calls = 0

    def _transform_separate_pca_weights_to_joint_values(self, scale_weights, trans_weights, translation_factor=0.01):
        self.pca_calls += 1
        batch = scale_weights.shape[0]
        return (
            torch.full((batch, N_JOINTS, 3), 0.5),
            torch.full((batch, N_JOINTS, 3), 0.25),
        )


def fake_fitter(device="cpu"):
    """Stand-in exposing just the attributes ``apply_shape_space_params`` writes."""
    return SimpleNamespace(
        device=device,
        log_beta_scales=SimpleNamespace(data=torch.zeros(1, N_JOINTS, 3)),
        betas_trans=SimpleNamespace(data=torch.zeros(1, N_JOINTS, 3)),
    )


def pca_params():
    """predicted_params carrying PCA weights (the `separate` + PCA layout)."""
    return {
        "log_beta_scales": torch.arange(N_BETAS, dtype=torch.float32).reshape(1, N_BETAS),
        "betas_trans": torch.arange(N_BETAS, dtype=torch.float32).reshape(1, N_BETAS) * 2.0,
    }


def per_joint_params():
    """predicted_params carrying per-joint values (no PCA expansion needed)."""
    return {
        "log_beta_scales": torch.full((1, N_JOINTS, 3), 0.75),
        "betas_trans": torch.full((1, N_JOINTS, 3), -0.125),
    }


@pytest.fixture
def separate_pca_config(monkeypatch):
    """Force ``scale_trans_mode='separate'`` WITH the PCA transformation on."""
    cfg = dict(TrainingConfig.SCALE_TRANS_BETA_CONFIG)
    cfg["separate"] = {**cfg["separate"], "use_pca_transformation": True}
    monkeypatch.setattr(TrainingConfig, "SCALE_TRANS_BETA_CONFIG", cfg)
    return cfg


@pytest.fixture
def separate_no_pca_config(monkeypatch):
    """Force ``scale_trans_mode='separate'`` WITHOUT the PCA transformation."""
    cfg = dict(TrainingConfig.SCALE_TRANS_BETA_CONFIG)
    cfg["separate"] = {**cfg["separate"], "use_pca_transformation": False}
    monkeypatch.setattr(TrainingConfig, "SCALE_TRANS_BETA_CONFIG", cfg)
    return cfg


# --------------------------------------------------------------------------- #
# Shape-space coherence
# --------------------------------------------------------------------------- #
def test_separate_pca_weights_are_expanded_to_per_joint(separate_pca_config):
    """PCA weights must never reach the fitter unexpanded.

    This is the concrete single-view bug: ``(1, N_BETAS)`` weights were written
    straight onto ``log_beta_scales``, which the SMAL model then consumed as if
    they were ``(1, J, 3)`` per-joint values.
    """
    model = FakeModel(scale_trans_mode="separate")
    fitter = fake_fitter()

    apply_shape_space_params(fitter, model, pca_params())

    assert model.pca_calls == 1, "PCA expansion was not invoked"
    assert tuple(fitter.log_beta_scales.data.shape) == (1, N_JOINTS, 3)
    assert tuple(fitter.betas_trans.data.shape) == (1, N_JOINTS, 3)
    assert torch.allclose(fitter.log_beta_scales.data, torch.full((1, N_JOINTS, 3), 0.5))
    assert torch.allclose(fitter.betas_trans.data, torch.full((1, N_JOINTS, 3), 0.25))


def test_separate_per_joint_values_are_not_re_expanded(separate_pca_config):
    """A ``(B, J, 3)`` payload is already per-joint - do NOT push it through PCA.

    Guards the shape check: ``use_pca_transformation`` is a global
    ``TrainingConfig`` setting, so a checkpoint whose heads emit per-joint values
    can still be loaded while the flag says PCA. The renderer must key on the
    actual tensor layout, not on the flag alone.
    """
    model = FakeModel(scale_trans_mode="separate")
    fitter = fake_fitter()

    apply_shape_space_params(fitter, model, per_joint_params())

    assert model.pca_calls == 0, "per-joint values were incorrectly PCA-expanded"
    assert torch.allclose(fitter.log_beta_scales.data, torch.full((1, N_JOINTS, 3), 0.75))
    assert torch.allclose(fitter.betas_trans.data, torch.full((1, N_JOINTS, 3), -0.125))


def test_separate_without_pca_applies_values_directly(separate_no_pca_config):
    model = FakeModel(scale_trans_mode="separate")
    fitter = fake_fitter()

    apply_shape_space_params(fitter, model, per_joint_params())

    assert model.pca_calls == 0
    assert torch.allclose(fitter.log_beta_scales.data, torch.full((1, N_JOINTS, 3), 0.75))


def test_entangled_mode_applies_values_directly(separate_pca_config):
    """``entangled_with_betas`` values are per-joint regardless of the PCA flag."""
    model = FakeModel(scale_trans_mode="entangled_with_betas")
    fitter = fake_fitter()

    apply_shape_space_params(fitter, model, per_joint_params())

    assert model.pca_calls == 0
    assert torch.allclose(fitter.log_beta_scales.data, torch.full((1, N_JOINTS, 3), 0.75))


def test_ignore_mode_leaves_shape_space_untouched(separate_pca_config):
    model = FakeModel(scale_trans_mode="ignore")
    fitter = fake_fitter()

    apply_shape_space_params(fitter, model, per_joint_params())

    assert model.pca_calls == 0
    assert torch.count_nonzero(fitter.log_beta_scales.data) == 0
    assert torch.count_nonzero(fitter.betas_trans.data) == 0


def test_disable_flags_are_independent(separate_pca_config):
    """``--disable_scaling`` and ``--disable_translation`` must gate separately."""
    model = FakeModel(scale_trans_mode="separate")

    fitter = fake_fitter()
    apply_shape_space_params(fitter, model, pca_params(), disable_scaling=True)
    assert torch.count_nonzero(fitter.log_beta_scales.data) == 0
    assert torch.count_nonzero(fitter.betas_trans.data) > 0

    fitter = fake_fitter()
    apply_shape_space_params(fitter, model, pca_params(), disable_translation=True)
    assert torch.count_nonzero(fitter.log_beta_scales.data) > 0
    assert torch.count_nonzero(fitter.betas_trans.data) == 0


def test_missing_shape_space_keys_are_tolerated(separate_pca_config):
    """A checkpoint without the scale/trans heads must not crash the render."""
    model = FakeModel(scale_trans_mode="separate")
    fitter = fake_fitter()

    apply_shape_space_params(fitter, model, {"betas": torch.zeros(1, N_BETAS)})

    assert torch.count_nonzero(fitter.log_beta_scales.data) == 0


# --------------------------------------------------------------------------- #
# Camera intrinsics / extrinsics
# --------------------------------------------------------------------------- #
def _predicted_camera():
    return {
        "fov": torch.tensor([[42.0]]),
        "cam_rot": torch.eye(3).unsqueeze(0) * 2.0,
        "cam_trans": torch.tensor([[1.0, 2.0, 3.0]]),
    }


def test_camera_centric_uses_fixed_identity_and_calibrated_fov():
    """camera_centric: identity camera + the dataset's own FOV, never the predicted one."""
    model = FakeModel(fixed_camera=True)
    params = _predicted_camera()
    y_data = {"cam_fov": 63.5, "cam_aspect": 1.25}

    R, T, fov, aspect = resolve_render_camera(model, params, y_data, device="cpu")

    assert torch.allclose(R, torch.eye(3).unsqueeze(0))
    assert torch.allclose(T, torch.zeros(1, 3))
    assert pytest.approx(float(fov.reshape(-1)[0]), abs=1e-6) == 63.5
    assert pytest.approx(aspect, abs=1e-6) == 1.25
    # The predicted camera must NOT leak into the render.
    assert not torch.allclose(R, params["cam_rot"])


def test_camera_centric_falls_back_to_predicted_fov_without_calibration():
    """A raw image has no calibration; predict_from_batch's injected FOV is used."""
    model = FakeModel(fixed_camera=True)
    params = _predicted_camera()

    R, T, fov, aspect = resolve_render_camera(model, params, y_data=None, device="cpu")

    assert torch.allclose(R, torch.eye(3).unsqueeze(0))
    assert pytest.approx(float(fov.reshape(-1)[0]), abs=1e-6) == 42.0
    assert aspect is None


def test_model_centric_uses_predicted_camera():
    model = FakeModel(fixed_camera=False)
    params = _predicted_camera()
    y_data = {"cam_fov": 63.5, "cam_aspect": 1.25}

    R, T, fov, aspect = resolve_render_camera(model, params, y_data, device="cpu")

    assert torch.allclose(R, params["cam_rot"])
    assert torch.allclose(T, params["cam_trans"])
    assert pytest.approx(float(fov.reshape(-1)[0]), abs=1e-6) == 42.0
    # Aspect still comes from the intrinsics: it describes the sensor, not the pose.
    assert pytest.approx(aspect, abs=1e-6) == 1.25


def test_camera_fov_shape_is_renderer_ready():
    """``set_camera_parameters`` wants a 1-D FOV; both conventions must supply one."""
    for fixed in (True, False):
        model = FakeModel(fixed_camera=fixed)
        _, _, fov, _ = resolve_render_camera(model, _predicted_camera(), {"cam_fov": 30.0}, device="cpu")
        assert fov.dim() == 1 and fov.numel() == 1


# --------------------------------------------------------------------------- #
# Mesh scaling
# --------------------------------------------------------------------------- #
def test_resolve_mesh_scale_requires_the_flag_and_the_key():
    params = {"mesh_scale": torch.tensor([[3.0]])}
    assert resolve_mesh_scale(FakeModel(allow_mesh_scaling=False), params) is None
    assert resolve_mesh_scale(FakeModel(allow_mesh_scaling=True), {}) is None
    got = resolve_mesh_scale(FakeModel(allow_mesh_scaling=True), params)
    assert got is not None and pytest.approx(float(got.reshape(-1)[0])) == 3.0


def _mesh():
    joints = torch.tensor([[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]])  # root at (1,1,1)
    verts = torch.tensor([[[3.0, 1.0, 1.0]]])
    trans = torch.tensor([[10.0, 0.0, 0.0]])
    return verts, joints, trans


def test_place_mesh_ue_scaling_takes_precedence():
    verts, joints, trans = _mesh()
    v, j = place_mesh(True, verts, joints, trans, mesh_scale=torch.tensor([[3.0]]))
    # (3,1,1) - root(1,1,1) = (2,0,0) -> *10 -> (20,0,0) + trans -> (30,0,0)
    assert torch.allclose(v[0, 0], torch.tensor([30.0, 0.0, 0.0]))
    assert torch.allclose(j[0, 0], torch.tensor([10.0, 0.0, 0.0]))


def test_place_mesh_applies_predicted_mesh_scale():
    verts, joints, trans = _mesh()
    v, _ = place_mesh(False, verts, joints, trans, mesh_scale=torch.tensor([[3.0]]))
    # (2,0,0) * 3 + (10,0,0) = (16,0,0)
    assert torch.allclose(v[0, 0], torch.tensor([16.0, 0.0, 0.0]))


def test_place_mesh_without_scaling_is_translation_only():
    verts, joints, trans = _mesh()
    v, j = place_mesh(False, verts, joints, trans, mesh_scale=None)
    assert torch.allclose(v[0, 0], torch.tensor([13.0, 1.0, 1.0]))
    assert torch.allclose(j[0, 0], torch.tensor([11.0, 1.0, 1.0]))


# --------------------------------------------------------------------------- #
# Checkpoint convention resolution
# --------------------------------------------------------------------------- #
def test_model_centric_is_the_default_for_legacy_checkpoints():
    conv = resolve_frame_convention({}, {})
    assert conv["frame_convention"] == "model_centric"
    assert conv["fixed_camera"] is False
    assert conv["use_ue_scaling"] is True  # legacy replicAnt convention
    assert conv["allow_mesh_scaling"] is False


def test_camera_centric_implies_fixed_camera_and_no_ue_scaling():
    conv = resolve_frame_convention({"frame_convention": "camera_centric"}, {})
    assert conv["camera_centric"] is True
    assert conv["fixed_camera"] is True
    assert conv["use_ue_scaling"] is False


def test_mesh_scale_head_is_detected_for_older_checkpoints():
    """Dropping the head makes the mesh render ~35x too large - detect it."""
    conv = resolve_frame_convention({}, {"mesh_scale_head.0.weight": None})
    assert conv["allow_mesh_scaling"] is True


def test_explicit_flags_win_over_inference():
    conv = resolve_frame_convention(
        {"frame_convention": "camera_centric", "fixed_camera": False, "use_ue_scaling": True},
        {},
    )
    assert conv["fixed_camera"] is False
    assert conv["use_ue_scaling"] is True


# --------------------------------------------------------------------------- #
# Dataset dispatch
# --------------------------------------------------------------------------- #
def _write_metadata_h5(path, **attrs):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata")
        for k, v in attrs.items():
            meta.attrs[k] = v
    return str(path)


def test_multiview_dataset_is_opened_in_single_view_mode(tmp_path):
    """A single-view checkpoint must never receive a per-view image list."""
    path = _write_metadata_h5(tmp_path / "mv.h5", is_multiview=True, dataset_type="sleap")
    assert hdf5_is_multiview(path) is True

    kwargs = resolve_singleview_dataset_kwargs(path, {"camera_centric": True})
    assert kwargs == {"return_single_view": True, "camera_centric": True, "expand_all_views": True}


def test_multiview_dataset_follows_the_checkpoint_frame_convention(tmp_path):
    """model_centric checkpoints keep the shared world frame (camera_centric=False)."""
    path = _write_metadata_h5(tmp_path / "mv.h5", is_multiview=True, dataset_type="sleap")
    kwargs = resolve_singleview_dataset_kwargs(path, {"camera_centric": False})
    assert kwargs["camera_centric"] is False
    assert kwargs["return_single_view"] is True


def test_true_single_view_dataset_needs_no_extra_kwargs(tmp_path):
    path = _write_metadata_h5(tmp_path / "sv.h5", is_multiview=False, dataset_type="sleap")
    assert hdf5_is_multiview(path) is False
    assert resolve_singleview_dataset_kwargs(path, {"camera_centric": True}) == {}


def test_unreadable_dataset_is_treated_as_single_view(tmp_path):
    bogus = tmp_path / "not_an_hdf5.h5"
    bogus.write_text("definitely not HDF5")
    assert hdf5_is_multiview(str(bogus)) is False


# --------------------------------------------------------------------------- #
# Smoothing + sub-clips (shared with the multi-view entrypoint)
# --------------------------------------------------------------------------- #
def test_smoother_returns_the_moving_average():
    smoother = PredictionSmoother(window_size=3)
    outs = [smoother({"trans": torch.tensor([[float(i)]])}) for i in range(5)]
    assert pytest.approx(float(outs[0]["trans"])) == 0.0  # first frame passes through
    assert pytest.approx(float(outs[1]["trans"])) == 0.5  # mean(0, 1)
    assert pytest.approx(float(outs[3]["trans"])) == 2.0  # mean(1, 2, 3) - window of 3
    assert pytest.approx(float(outs[4]["trans"])) == 3.0  # mean(2, 3, 4)


def test_smoother_is_a_no_op_when_disabled():
    smoother = PredictionSmoother(window_size=0)
    params = {"trans": torch.tensor([[7.0]])}
    assert smoother(params) is params


def test_smoother_passes_metadata_through():
    smoother = PredictionSmoother(window_size=2)
    smoother({"trans": torch.tensor([[0.0]]), "num_views": 3})
    out = smoother({"trans": torch.tensor([[2.0]]), "num_views": 4})
    assert out["num_views"] == 4
    assert pytest.approx(float(out["trans"])) == 1.0


def test_subclip_ranges_single_clip_and_max_frames():
    assert compute_subclip_ranges(100, None, 1) == [(0, 100)]
    assert compute_subclip_ranges(100, 30, 1) == [(0, 30)]
    assert compute_subclip_ranges(10, 30, 1) == [(0, 10)]


def test_subclip_ranges_evenly_spaced():
    assert compute_subclip_ranges(100, 10, 4) == [(0, 10), (25, 35), (50, 60), (75, 85)]


def test_subclip_ranges_fall_back_when_they_do_not_fit():
    assert compute_subclip_ranges(100, None, 4) == [(0, 100)]  # needs --max_frames
    assert compute_subclip_ranges(20, 10, 4) == [(0, 20)]  # only 5 frames per slot


def test_subclip_ranges_are_within_bounds_and_ordered():
    ranges = compute_subclip_ranges(1000, 50, 7)
    assert len(ranges) == 7
    for start, end in ranges:
        assert 0 <= start < end <= 1000
    assert ranges == sorted(ranges)
    starts = np.array([s for s, _ in ranges])
    assert np.all(np.diff(starts) > 0), "subclip starts must be strictly increasing"


# --------------------------------------------------------------------------- #
# Distributed plumbing (shared with the multi-view entrypoint)
#
# These exercise the pure parts — index assignment, temp-storage round trips and
# launch-mode detection — without standing up a process group, which needs real
# GPUs and a rendezvous.
# --------------------------------------------------------------------------- #
def test_rank_indices_partition_the_range_exactly_once():
    """Every frame is rendered by exactly one rank — no gaps, no duplicates."""
    world_size = 4
    dataset_size = 103
    assigned = [compute_rank_indices(dataset_size, r, world_size) for r in range(world_size)]

    flat = sorted(i for chunk in assigned for i in chunk)
    assert flat == list(range(dataset_size))
    assert sum(len(chunk) for chunk in assigned) == dataset_size


def test_rank_indices_are_striped_not_chunked():
    """Striping keeps each rank's slice spread over the whole clip."""
    assert compute_rank_indices(10, 0, 3) == [0, 3, 6, 9]
    assert compute_rank_indices(10, 1, 3) == [1, 4, 7]
    assert compute_rank_indices(10, 2, 3) == [2, 5, 8]


def test_rank_indices_respect_the_subclip_window():
    world_size = 2
    assigned = [compute_rank_indices(100, r, world_size, start_idx=20, end_idx=30) for r in range(world_size)]
    flat = sorted(i for chunk in assigned for i in chunk)
    assert flat == list(range(20, 30))


def test_rank_indices_clamp_to_the_dataset():
    assert compute_rank_indices(5, 0, 1, start_idx=-3, end_idx=99) == [0, 1, 2, 3, 4]


def test_rank_indices_single_process_is_contiguous():
    assert compute_rank_indices(6, 0, 1) == [0, 1, 2, 3, 4, 5]


def test_more_ranks_than_frames_leaves_some_ranks_idle():
    """An empty slice must be a valid outcome, not a crash."""
    assigned = [compute_rank_indices(2, r, 4) for r in range(4)]
    assert assigned[0] == [0] and assigned[1] == [1]
    assert assigned[2] == [] and assigned[3] == []


def test_gather_predictions_single_process_just_sorts():
    raw = [(5, {"a": 1}), (1, {"a": 2}), (3, {"a": 3})]
    got = gather_predictions(raw, rank=0, world_size=1, temp_dir=Path("/nonexistent"))
    assert [idx for idx, _ in got] == [1, 3, 5]


def test_frame_streams_round_trip_restores_clip_order(tmp_path):
    """The temp-storage round trip must undo the striping applied at dispatch."""
    world_size = 3
    dataset_size = 11
    temp_dir = tmp_path / "frames"

    # Each rank writes the frames for its striped slice, tagged by global index.
    for rank in range(world_size):
        indices = compute_rank_indices(dataset_size, rank, world_size)
        frames = [np.full((4, 4, 3), idx, dtype=np.uint8) for idx in indices]
        write_frame_streams_to_temp({"sv": (frames, indices)}, temp_dir, rank)

    merged = merge_frame_streams_from_temp(temp_dir, world_size, ["sv"])
    assert [idx for idx, _ in merged["sv"]] == list(range(dataset_size))
    for _, path in merged["sv"]:
        assert os.path.exists(path)


def test_frame_streams_keep_multiple_streams_separate(tmp_path):
    """Multi-view writes one grid stream plus one stream per rendered camera."""
    temp_dir = tmp_path / "frames"
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    streams = {
        "mv": ([frame, frame], [0, 1]),
        "sv_view0": ([frame], [0]),
        "sv_view3": ([frame], [1]),
    }
    write_frame_streams_to_temp(streams, temp_dir, rank=0)

    merged = merge_frame_streams_from_temp(temp_dir, 1, ["mv", "sv_view0", "sv_view3"])
    assert len(merged["mv"]) == 2
    assert [idx for idx, _ in merged["sv_view0"]] == [0]
    assert [idx for idx, _ in merged["sv_view3"]] == [1]


def test_merge_tolerates_a_rank_that_wrote_nothing(tmp_path):
    """A rank with an empty slice writes no manifest; the merge must not fail."""
    temp_dir = tmp_path / "frames"
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    write_frame_streams_to_temp({"sv": ([frame], [0])}, temp_dir, rank=0)

    merged = merge_frame_streams_from_temp(temp_dir, world_size=3, stream_names=["sv"])
    assert [idx for idx, _ in merged["sv"]] == [0]


def test_dist_timeout_default_is_long_enough_for_rank0_phases(monkeypatch):
    """The rank-0-only gather/merge phases can outlast a short NCCL watchdog."""
    monkeypatch.delenv("SMILIFY_DIST_TIMEOUT_S", raising=False)
    assert resolve_dist_timeout() == 14400
    assert resolve_dist_timeout(60) == 60


def test_dist_timeout_reads_the_env_var(monkeypatch):
    monkeypatch.setenv("SMILIFY_DIST_TIMEOUT_S", "900")
    assert resolve_dist_timeout() == 900


def test_dist_timeout_falls_back_on_a_bad_env_var(monkeypatch):
    monkeypatch.setenv("SMILIFY_DIST_TIMEOUT_S", "not-a-number")
    assert resolve_dist_timeout() == 14400


def test_launch_mode_detection(monkeypatch):
    for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(var, raising=False)
    assert is_torchrun_launched() is False
    # mp.spawn mode: the spawn arguments are authoritative, local rank == rank.
    assert resolve_launch(2, 4) == (2, 4, 2)

    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")
    assert is_torchrun_launched() is True
    # torchrun mode: the environment wins, and the GPU comes from LOCAL_RANK.
    assert resolve_launch(0, 1) == (3, 8, 1)


def test_ipv4_patch_is_idempotent_and_preserves_results():
    import socket

    force_ipv4_getaddrinfo()
    patched = socket.getaddrinfo
    force_ipv4_getaddrinfo()
    assert socket.getaddrinfo is patched, "the patch must not stack on itself"

    results = socket.getaddrinfo("127.0.0.1", 80, type=socket.SOCK_STREAM)
    assert results, "IPv4 lookups must still resolve"
    assert all(r[0] == socket.AF_INET for r in results)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
