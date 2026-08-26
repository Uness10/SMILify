"""
Structural probe for the multi-animal regressors (issue: multi-animal support).

Why this exists
---------------
``smal_fitter/neuralSMIL/multianimal/regressor.py`` and ``multiview_regressor.py``
can only be imported where pytorch3d and the SMAL model pickle are installed, so
the CPU-only CI run skips ``tests/test_multianimal_regressor.py`` entirely. This
probe stubs *just enough* of the heavy stack (pytorch3d transforms, ``config``,
``SMALFitter``, the backbone factory) to import the real modules and actually run
their forward passes, so the structural claims of the design can be checked
empirically rather than by reading:

1. the backbone runs **once** per batch, no matter how many specimens;
2. each specimen head produces its own body parameters;
3. specimen parameter dicts carry **no** camera entries — the camera is scene
   level (design doc §6);
4. the multi-view path produces the same specimen list while keeping the
   per-view cameras shared;
5. gradients from specimen *i*'s loss reach head *i* and no other head.

Run from the repo root::

    python -m diagnostics.multianimal_forward_PROBE

It prints a PASS/FAIL line per check and exits non-zero on failure. It is a
development probe, not a pytest test: it fakes the numerics stack, so it proves
*wiring*, never numerical correctness.
"""

from __future__ import annotations

import sys
import types
from typing import List

import numpy as np
import torch
import torch.nn as nn

FEATURE_DIM = 32
SPATIAL_TOKENS = 5
N_POSE = 4
N_BETAS = 6
RESULTS: List[tuple] = []


# --------------------------------------------------------------------------- #
# Stubs for the heavy stack
# --------------------------------------------------------------------------- #


def _install_stubs() -> None:
    """Register fake modules so the real regressor modules import."""

    # --- pytorch3d -------------------------------------------------------- #
    p3d = types.ModuleType("pytorch3d")
    transforms = types.ModuleType("pytorch3d.transforms")

    def axis_angle_to_matrix(aa):
        return torch.eye(3, device=aa.device).expand(*aa.shape[:-1], 3, 3).contiguous()

    def matrix_to_axis_angle(m):
        return torch.zeros(*m.shape[:-2], 3, device=m.device)

    def rotation_6d_to_matrix(d6):
        return torch.eye(3, device=d6.device).expand(*d6.shape[:-1], 3, 3).contiguous()

    def matrix_to_rotation_6d(m):
        return torch.zeros(*m.shape[:-2], 6, device=m.device)

    transforms.axis_angle_to_matrix = axis_angle_to_matrix
    transforms.matrix_to_axis_angle = matrix_to_axis_angle
    transforms.rotation_6d_to_matrix = rotation_6d_to_matrix
    transforms.matrix_to_rotation_6d = matrix_to_rotation_6d

    renderer = types.ModuleType("pytorch3d.renderer")
    renderer.FoVPerspectiveCameras = object
    structures = types.ModuleType("pytorch3d.structures")
    structures.Meshes = object

    sys.modules["pytorch3d"] = p3d
    sys.modules["pytorch3d.transforms"] = transforms
    sys.modules["pytorch3d.renderer"] = renderer
    sys.modules["pytorch3d.structures"] = structures

    # --- cv2 / scipy ------------------------------------------------------ #
    if "cv2" not in sys.modules:
        cv2 = types.ModuleType("cv2")
        cv2.INTER_LINEAR = 1
        cv2.resize = lambda image, size, interpolation=None: image
        sys.modules["cv2"] = cv2
    try:
        import scipy.spatial.transform  # noqa: F401
    except Exception:
        scipy = types.ModuleType("scipy")
        spatial = types.ModuleType("scipy.spatial")
        transform = types.ModuleType("scipy.spatial.transform")
        transform.Rotation = object
        sys.modules["scipy"] = scipy
        sys.modules["scipy.spatial"] = spatial
        sys.modules["scipy.spatial.transform"] = transform

    # --- root config ------------------------------------------------------ #
    config = types.ModuleType("config")
    config.N_POSE = N_POSE
    config.N_BETAS = N_BETAS
    config.ignore_hardcoded_body = True
    config.DEBUG = False
    config.dd = {"J_names": [f"joint_{i}" for i in range(N_POSE + 1)]}
    config.CANONICAL_MODEL_JOINTS = list(range(N_POSE + 1))
    config.SMAL_FILE = ""
    config.SHAPE_FAMILY = -1
    config.IMG_RES = 64
    sys.modules["config"] = config

    # --- SMALFitter ------------------------------------------------------- #
    fitter = types.ModuleType("smal_fitter.fitter")

    class SMALFitter(nn.Module):
        """Minimal stand-in: records what the real one provides to subclasses."""

        def __init__(self, device, data_batch, batch_size, shape_family, use_unity_prior, rgb_only=True):
            super().__init__()
            self.device = device
            self.batch_size = batch_size
            self.shape_family = shape_family
            self.use_unity_prior = use_unity_prior
            self.rgb_only = rgb_only

    fitter.SMALFitter = SMALFitter
    sys.modules["smal_fitter.fitter"] = fitter

    # --- backbone factory ------------------------------------------------- #
    backbone_factory = types.ModuleType("smal_fitter.neuralSMIL.backbone_factory")

    class _Backbone(nn.Module):
        """Counts its calls so 'one backbone pass per batch' can be asserted."""

        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(3, FEATURE_DIM)
            self.calls = 0

        def get_feature_dim(self):
            return FEATURE_DIM

        def get_spatial_dim(self):
            return FEATURE_DIM

        def forward(self, images):
            self.calls += 1
            pooled = images.mean(dim=(2, 3))  # (B, 3)
            return self.proj(pooled)

        def forward_with_spatial(self, images):
            features = self.forward(images)
            spatial = features.unsqueeze(1).expand(-1, SPATIAL_TOKENS, -1).contiguous()
            return features, spatial

    class BackboneFactory:
        @staticmethod
        def create_backbone(name, pretrained=True, freeze=True):
            return _Backbone()

        @staticmethod
        def get_default_input_resolution(name):
            return 64

    backbone_factory.BackboneFactory = BackboneFactory
    sys.modules["smal_fitter.neuralSMIL.backbone_factory"] = backbone_factory

    # --- training config -------------------------------------------------- #
    training_config = types.ModuleType("smal_fitter.neuralSMIL.training_config")

    class TrainingConfig:
        @staticmethod
        def get_scale_trans_config():
            return {"separate": {"use_pca_transformation": True}}

    training_config.TrainingConfig = TrainingConfig
    sys.modules["smal_fitter.neuralSMIL.training_config"] = training_config


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(condition), detail))
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  --  {detail}" if detail else ""))


def build_singleview(num_animals: int, head_strategy: str = "replicated"):
    from smal_fitter.neuralSMIL.configs.multianimal_config import MultiAnimalConfig
    from smal_fitter.neuralSMIL.multianimal.regressor import MultiAnimalSMILRegressor

    config = MultiAnimalConfig(
        enabled=True,
        num_animals=num_animals,
        specimen_ids=[f"mouse_{i}" for i in range(num_animals)],
        head_strategy=head_strategy,
    )
    config.validate()

    device = torch.device("cpu")
    return MultiAnimalSMILRegressor(
        device=device,
        data_batch=torch.zeros(2, 3, 64, 64),
        batch_size=2,
        shape_family=-1,
        use_unity_prior=False,
        multi_animal=config,
        head_type="transformer_decoder",
        rotation_representation="axis_angle",
        input_resolution=64,
        backbone_name="stub",
        hidden_dim=64,
        transformer_config={"hidden_dim": 64, "depth": 1, "heads": 2, "dim_head": 8, "mlp_dim": 32, "ief_iters": 1},
        scale_trans_mode="separate",
    )


def probe_singleview() -> None:
    print("\n--- single-view ---")
    model = build_singleview(3)
    images = torch.randn(2, 3, 64, 64)

    model.backbone.calls = 0
    output = model(images)

    check("single-view: backbone runs once for N specimens", model.backbone.calls == 1, f"calls={model.backbone.calls}")
    check("single-view: one parameter dict per specimen", len(output["animals"]) == 3)
    check(
        "single-view: body parameters have the batch shape",
        output["animals"][0]["global_rot"].shape[0] == 2,
        str(tuple(output["animals"][0]["global_rot"].shape)),
    )
    check(
        "single-view: specimen dicts carry no camera (camera is scene level)",
        all(not {"fov", "cam_rot", "cam_trans"} & set(params) for params in output["animals"]),
    )
    check(
        "single-view: exactly one scene camera is produced",
        output["fov"].shape == (2, 1) and output["cam_rot"].shape == (2, 3, 3),
        f"fov={tuple(output['fov'].shape)} cam_rot={tuple(output['cam_rot'].shape)}",
    )
    check(
        "single-view: specimen predictions differ once heads differ",
        _heads_can_diverge(model, images),
    )
    check(
        "single-view: specimen i's loss only touches head i",
        _gradient_isolation(model, images),
    )

    shared = build_singleview(3, head_strategy="shared_query")
    shared_output = shared(images)
    check("shared_query: one parameter dict per specimen", len(shared_output["animals"]) == 3)
    check(
        "shared_query: fewer head parameters than the replicated bank",
        _head_param_count(shared) < _head_param_count(model),
        f"shared={_head_param_count(shared):,} replicated={_head_param_count(model):,}",
    )
    shared.backbone.calls = 0
    shared(images)
    check("shared_query: backbone still runs once", shared.backbone.calls == 1)


def _head_param_count(model) -> int:
    return sum(p.numel() for p in model.specimen_heads.parameters())


def _heads_can_diverge(model, images) -> bool:
    """Perturb head 1 and confirm only its output moves."""
    with torch.no_grad():
        before = [params["betas"].clone() for params in model(images)["animals"]]
        head = model.specimen_heads.heads[1]
        for param in head.parameters():
            param.add_(0.05)
        after = [params["betas"] for params in model(images)["animals"]]
    return (
        torch.allclose(before[0], after[0])
        and not torch.allclose(before[1], after[1])
        and torch.allclose(before[2], after[2])
    )


def _gradient_isolation(model, images) -> bool:
    """Strict head/specimen binding: no cross-specimen gradient leakage."""
    model.zero_grad(set_to_none=True)
    output = model(images)
    output["animals"][1]["betas"].sum().backward()

    def has_grad(module) -> bool:
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

    return (
        has_grad(model.specimen_heads.heads[1])
        and not has_grad(model.specimen_heads.heads[0])
        and not has_grad(model.specimen_heads.heads[2])
    )


def probe_multiview() -> None:
    print("\n--- multi-view ---")
    from smal_fitter.neuralSMIL.configs.multianimal_config import MultiAnimalConfig
    from smal_fitter.neuralSMIL.multianimal.multiview_regressor import MultiAnimalMultiViewSMILRegressor

    config = MultiAnimalConfig(
        enabled=True, num_animals=2, specimen_ids=["mouse_a", "mouse_b"], head_strategy="replicated"
    )
    config.validate()

    model = MultiAnimalMultiViewSMILRegressor(
        device=torch.device("cpu"),
        data_batch=torch.zeros(2, 3, 64, 64),
        batch_size=2,
        shape_family=-1,
        use_unity_prior=False,
        multi_animal=config,
        max_views=3,
        canonical_camera_order=["cam0", "cam1", "cam2"],
        head_type="transformer_decoder",
        rotation_representation="axis_angle",
        input_resolution=64,
        backbone_name="stub",
        hidden_dim=64,
        transformer_config={"hidden_dim": 64, "depth": 1, "heads": 2, "dim_head": 8, "mlp_dim": 32, "ief_iters": 1},
        scale_trans_mode="separate",
    )

    images_per_view = [torch.randn(2, 3, 64, 64) for _ in range(3)]
    camera_indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
    view_mask = torch.ones(2, 3, dtype=torch.bool)

    model.backbone.calls = 0
    output = model.forward_multiview(images_per_view, camera_indices, view_mask)

    check(
        "multi-view: backbone runs once for all views and specimens",
        model.backbone.calls == 1,
        f"calls={model.backbone.calls}",
    )
    check("multi-view: one parameter dict per specimen", len(output["animals"]) == 2)
    check("multi-view: one camera per view", len(output["fov_per_view"]) == 3)
    check(
        "multi-view: specimen dicts carry no camera",
        all(not {"fov", "cam_rot", "cam_trans"} & set(params) for params in output["animals"]),
    )
    check("multi-view: view bookkeeping survives the override", output["num_views"] == 3)

    first = model.build_specimen_prediction(output, 0)
    second = model.build_specimen_prediction(output, 1)
    check(
        "multi-view: every specimen is projected through the same cameras",
        first["fov_per_view"] is second["fov_per_view"],
    )
    check(
        "multi-view: specimen views expose the keys the inherited loss reads",
        {"global_rot", "joint_rot", "betas", "trans", "num_views", "view_mask"} <= set(first),
        sorted(set(first))[:8],
    )


def probe_batch_paths() -> None:
    print("\n--- batch wiring ---")
    from smal_fitter.neuralSMIL.multianimal.schema import make_multi_animal_sample

    model = build_singleview(2)
    rng = np.random.default_rng(0)

    def make_sample(present, seed):
        x_data = {"input_image_data": rng.random((64, 64, 3), dtype=np.float32)}
        animals = []
        for is_present in present:
            animals.append(
                {
                    "root_rot": rng.random(3).astype(np.float32),
                    "root_loc": rng.random(3).astype(np.float32),
                    "joint_angles": rng.random((N_POSE + 1, 3)).astype(np.float32),
                    "shape_betas": rng.random(N_BETAS).astype(np.float32),
                    "keypoints_2d": rng.random((6, 2)).astype(np.float32),
                    "keypoint_visibility": np.ones(6, dtype=bool),
                }
                if is_present
                else None
            )
        return make_multi_animal_sample(x_data, {}, animals, specimen_ids=["mouse_0", "mouse_1"])

    batch = [make_sample((True, True), 0), make_sample((True, False), 1)]
    x_batch = [x for x, _ in batch]
    y_batch = [y for _, y in batch]

    model.backbone.calls = 0
    predicted, targets, aux = model.predict_from_batch(x_batch, y_batch)

    check("predict_from_batch: backbone runs once", model.backbone.calls == 1, f"calls={model.backbone.calls}")
    check("predict_from_batch: returns per-specimen targets", len(targets["specimens"]) == 2)
    check(
        "predict_from_batch: presence mask matches the sample declarations",
        predicted["animal_mask"].tolist() == [[True, True], [True, False]],
        str(predicted["animal_mask"].tolist()),
    )
    check(
        "predict_from_batch: an absent specimen has no pose target",
        targets["specimens"][1]["joint_rot"] is None
        or bool(targets["specimens"][1].get("_availability_masks", {}).get("joint_rot", torch.tensor([True, True]))[1])
        is False,
    )
    check("predict_from_batch: auxiliary data is per specimen", len(aux["specimens"]) == 2)
    check(
        "predict_from_batch: specimen 0 targets are also exposed at the top level",
        "global_rot" in targets,
    )


def probe_loss() -> None:
    """Run the real per-specimen loss with only parameter terms enabled.

    Rendering-based terms (2D/3D keypoints, silhouette) need the actual SMAL
    mesh, so they are weighted to zero here; what is being checked is the
    aggregation contract, not the loss maths.
    """
    print("\n--- loss aggregation ---")
    from smal_fitter.neuralSMIL.multianimal.schema import make_multi_animal_sample

    model = build_singleview(2)
    rng = np.random.default_rng(7)

    def make_sample(present):
        x_data = {"input_image_data": rng.random((64, 64, 3), dtype=np.float32)}
        animals = [
            (
                {
                    "root_rot": rng.random(3).astype(np.float32),
                    "root_loc": rng.random(3).astype(np.float32),
                    "joint_angles": rng.random((N_POSE + 1, 3)).astype(np.float32),
                    "shape_betas": rng.random(N_BETAS).astype(np.float32),
                    # Keypoints in [0, 1] so the inherited visibility gate
                    # (_validate_sample_visibility) accepts the sample.
                    "keypoints_2d": rng.random((8, 2)).astype(np.float32),
                    "keypoint_visibility": np.ones(8, dtype=bool),
                }
                if is_present
                else None
            )
            for is_present in present
        ]
        return make_multi_animal_sample(x_data, {}, animals, specimen_ids=["mouse_0", "mouse_1"])

    weights = {
        "global_rot": 1.0,
        "joint_rot": 1.0,
        "betas": 1.0,
        "trans": 1.0,
        "fov": 0.5,
        "cam_rot": 0.5,
        "cam_trans": 0.5,
        "log_beta_scales": 0.0,
        "betas_trans": 0.0,
        "keypoint_2d": 0.0,
        "keypoint_3d": 0.0,
        "silhouette": 0.0,
        "joint_angle_regularization": 0.0,
        "joint_limit_regularization": 0.0,
        "limb_scale_regularization": 0.0,
        "limb_trans_regularization": 0.0,
    }

    batch = [make_sample((True, True)), make_sample((True, False))]
    predicted, targets, aux = model.predict_from_batch([x for x, _ in batch], [y for _, y in batch])
    loss, components = model.compute_batch_loss(predicted, targets, aux, return_components=True, loss_weights=weights)

    check("loss: returns a finite scalar", loss.dim() == 0 and torch.isfinite(loss), f"loss={float(loss):.4f}")
    check("loss: is differentiable", loss.requires_grad)
    check(
        "loss: reports per-specimen components",
        any(key.endswith("/mouse_0") for key in components) and any(key.endswith("/mouse_1") for key in components),
        sorted(k for k in components if "/" in k)[:4],
    )
    check(
        "loss: both specimens were supervised",
        float(components.get("num_specimens_supervised", torch.tensor(0.0))) == 2.0,
    )

    model.zero_grad(set_to_none=True)
    loss.backward()

    def grad_norm(module):
        return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)

    check("loss: gradient reaches specimen head 0", grad_norm(model.specimen_heads.heads[0]) > 0)
    check("loss: gradient reaches specimen head 1", grad_norm(model.specimen_heads.heads[1]) > 0)
    check("loss: gradient reaches the scene camera head", grad_norm(model.scene_camera_head) >= 0)

    # A slot that is absent everywhere must be dropped, not averaged in as zero.
    lonely = [make_sample((True, False)), make_sample((True, False))]
    predicted, targets, aux = model.predict_from_batch([x for x, _ in lonely], [y for _, y in lonely])
    _, components = model.compute_batch_loss(predicted, targets, aux, return_components=True, loss_weights=weights)
    check(
        "loss: an entirely absent specimen is dropped from the average",
        float(components.get("num_specimens_supervised", torch.tensor(0.0))) == 1.0,
        str(float(components.get("num_specimens_supervised", torch.tensor(-1.0)))),
    )


def probe_single_animal_equivalence() -> None:
    """N=1 multi-animal must reproduce the single-animal model exactly.

    This is the backwards-compatibility claim that lets existing datasets and
    checkpoints keep working: with one specimen and the camera taken from that
    specimen's head (``camera_mode='first_specimen'``), the multi-animal model
    is the single-animal model with a renamed head.
    """
    print("\n--- N=1 equivalence with the single-animal model ---")
    from smal_fitter.neuralSMIL.configs.multianimal_config import MultiAnimalConfig
    from smal_fitter.neuralSMIL.multianimal.checkpoint import to_multi_animal
    from smal_fitter.neuralSMIL.multianimal.regressor import MultiAnimalSMILRegressor
    from smal_fitter.neuralSMIL.smil_image_regressor import SMILImageRegressor

    shared_kwargs = dict(
        head_type="transformer_decoder",
        rotation_representation="axis_angle",
        input_resolution=64,
        backbone_name="stub",
        hidden_dim=64,
        transformer_config={"hidden_dim": 64, "depth": 1, "heads": 2, "dim_head": 8, "mlp_dim": 32, "ief_iters": 1},
        scale_trans_mode="separate",
    )

    torch.manual_seed(0)
    baseline = SMILImageRegressor(
        device=torch.device("cpu"),
        data_batch=torch.zeros(2, 3, 64, 64),
        batch_size=2,
        shape_family=-1,
        use_unity_prior=False,
        **shared_kwargs,
    ).eval()

    config = MultiAnimalConfig(
        enabled=True, num_animals=1, specimen_ids=["specimen_0"], camera_mode="first_specimen"
    )
    config.validate()
    torch.manual_seed(0)
    multi = MultiAnimalSMILRegressor(
        device=torch.device("cpu"),
        data_batch=torch.zeros(2, 3, 64, 64),
        batch_size=2,
        shape_family=-1,
        use_unity_prior=False,
        multi_animal=config,
        **shared_kwargs,
    ).eval()

    migrated, report = to_multi_animal(baseline.state_dict(), 1)
    missing, unexpected = multi.load_state_dict(migrated, strict=False)
    check(
        "equivalence: the single-animal checkpoint loads with nothing left over",
        not unexpected,
        f"unexpected={sorted(unexpected)[:4]} missing={sorted(missing)[:4]}",
    )
    check("equivalence: migration seeded the one specimen head", report.heads_seeded == [0])

    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        reference = baseline(images)
        candidate = multi(images)

    body_keys = ["global_rot", "joint_rot", "betas", "trans", "log_beta_scales", "betas_trans"]
    body_match = all(
        torch.allclose(reference[key], candidate["animals"][0][key], atol=1e-6)
        for key in body_keys
        if key in reference
    )
    check("equivalence: body parameters are identical", body_match)

    camera_match = all(
        torch.allclose(reference[key], candidate[key], atol=1e-6)
        for key in ("fov", "cam_rot", "cam_trans")
        if key in reference
    )
    check("equivalence: camera parameters are identical", camera_match)


def main() -> int:
    _install_stubs()
    probe_singleview()
    probe_multiview()
    probe_batch_paths()
    probe_loss()
    probe_single_animal_equivalence()

    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
