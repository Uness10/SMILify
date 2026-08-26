"""
Wiring tests for the multi-animal regressors.

Unlike the rest of the multi-animal suite, these need the heavyweight stack
(pytorch3d, the SMAL model pickle referenced by ``config.py``), so the module
skips itself when that stack is unavailable — matching how the other
model-touching tests in this repo behave.

The tests deliberately avoid constructing a full regressor (that would download a
ViT and load a mesh): they exercise the *wiring* methods — batch normalisation,
the presence mask, and the per-specimen prediction views — on a bare instance.
Those are the methods where a mistake silently mispairs a specimen with another
animal's camera or targets, which no loss curve would reveal.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# The schema is dependency-free, so it imports even where the heavy stack is not
# installed; the regressors below are what may be unavailable.
from smal_fitter.neuralSMIL.multianimal.schema import (
    ANIMALS_KEY,
    MultiAnimalSchemaError,
    make_multi_animal_sample,
)

try:  # pragma: no cover - environment dependent
    from smal_fitter.neuralSMIL.configs.multianimal_config import MultiAnimalConfig
    from smal_fitter.neuralSMIL.multianimal.multiview_regressor import MultiAnimalMultiViewSMILRegressor
    from smal_fitter.neuralSMIL.multianimal.regressor import MultiAnimalSMILRegressor
except Exception as exc:  # noqa: BLE001 - any import failure means "stack unavailable"
    pytest.skip(f"multi-animal regressors unavailable in this environment: {exc}", allow_module_level=True)


def bare(cls, num_animals=2, **overrides):
    """A regressor instance with only the multi-animal wiring populated.

    ``object.__new__`` skips the parent constructors (backbone download, SMAL
    mesh load, pytorch3d renderer) so the wiring can be tested in milliseconds.
    """
    model = object.__new__(cls)
    config = MultiAnimalConfig(
        enabled=True,
        num_animals=num_animals,
        specimen_ids=[f"mouse_{i}" for i in range(num_animals)],
        **overrides,
    )
    config.validate()
    model.multi_animal = config
    model.num_animals = num_animals
    model.specimen_ids = list(config.specimen_ids)
    model.device = torch.device("cpu")
    return model


def sample(present=(True, True), seed=0, visibility=None):
    rng = np.random.default_rng(seed)
    x_data = {"input_image_data": rng.random((4, 4, 3), dtype=np.float32)}
    scene = {"cam_fov_per_view": np.array([[8.0]], dtype=np.float32)}
    animals = []
    for index, is_present in enumerate(present):
        if not is_present:
            animals.append(None)
            continue
        vis = np.ones(6, dtype=bool) if visibility is None else np.asarray(visibility[index], dtype=bool)
        animals.append(
            {
                "root_rot": rng.random(3).astype(np.float32),
                "keypoints_2d": rng.random((6, 2)).astype(np.float32),
                "keypoint_visibility": vis,
            }
        )
    return make_multi_animal_sample(
        x_data, scene, animals, specimen_ids=[f"mouse_{i}" for i in range(len(present))]
    )


class TestSingleViewWiring:
    def test_legacy_samples_are_promoted(self):
        model = bare(MultiAnimalSMILRegressor, num_animals=1)
        x_batch = [{"input_image_data": np.zeros((4, 4, 3), dtype=np.float32)}]
        y_batch = [{"root_rot": np.zeros(3, dtype=np.float32)}]

        x_norm, y_norm = model._normalize_batch(x_batch, y_batch)

        assert len(x_norm) == 1
        assert ANIMALS_KEY in y_norm[0]

    def test_image_less_samples_are_dropped_once_for_every_specimen(self):
        # Dropping per specimen instead of up front would misalign the shared
        # backbone pass with the per-specimen targets.
        model = bare(MultiAnimalSMILRegressor, num_animals=2)
        good_x, good_y = sample()
        bad_x, bad_y = sample(seed=1)
        bad_x["input_image_data"] = None

        x_norm, y_norm = model._normalize_batch([bad_x, good_x], [bad_y, good_y])

        assert len(x_norm) == len(y_norm) == 1
        assert x_norm[0] is good_x

    def test_shuffled_specimen_order_is_rejected(self):
        model = bare(MultiAnimalSMILRegressor, num_animals=2)
        first_x, first_y = sample()
        second_x, second_y = sample(seed=1)
        second_x["specimen_ids"] = ["mouse_1", "mouse_0"]

        with pytest.raises(MultiAnimalSchemaError, match="not stable"):
            model._normalize_batch([first_x, second_x], [first_y, second_y])

    def test_identity_check_can_be_disabled(self):
        model = bare(MultiAnimalSMILRegressor, num_animals=2, require_stable_identity=False)
        first_x, first_y = sample()
        second_x, second_y = sample(seed=1)
        second_x["specimen_ids"] = ["mouse_1", "mouse_0"]

        x_norm, _ = model._normalize_batch([first_x, second_x], [first_y, second_y])
        assert len(x_norm) == 2

    def test_animal_mask_reflects_presence(self):
        model = bare(MultiAnimalSMILRegressor, num_animals=2)
        batch = [sample(present=(True, True), seed=0), sample(present=(True, False), seed=1)]
        x_norm = [x for x, _ in batch]
        y_norm = [y for _, y in batch]

        mask = model._build_animal_mask(x_norm, y_norm, specimen_aux=[{}, {}])
        assert mask.tolist() == [[True, True], [True, False]]

    def test_visibility_floor_marks_an_occluded_specimen_absent(self):
        model = bare(MultiAnimalSMILRegressor, num_animals=2, min_visible_keypoints_per_specimen=4)
        x_data, y_data = sample(present=(True, True), visibility=[[1] * 6, [1, 1, 0, 0, 0, 0]])
        # The single-view mask reads visibility off the auxiliary data the
        # inherited batch assembly produced.
        aux = [
            {"keypoint_data": [{"keypoint_visibility": np.ones(6, dtype=bool)}]},
            {"keypoint_data": [{"keypoint_visibility": np.array([1, 1, 0, 0, 0, 0], dtype=bool)}]},
        ]
        mask = model._build_animal_mask([x_data], [y_data], specimen_aux=aux)
        assert mask.tolist() == [[True, False]]

    def test_specimen_prediction_pairs_body_params_with_the_shared_camera(self):
        model = bare(MultiAnimalSMILRegressor, num_animals=2)
        predicted = {
            "animals": [
                {"global_rot": torch.zeros(2, 3), "betas": torch.full((2, 5), 1.0)},
                {"global_rot": torch.ones(2, 3), "betas": torch.full((2, 5), 2.0)},
            ],
            "fov": torch.full((2, 1), 8.0),
            "cam_rot": torch.eye(3).expand(2, 3, 3),
            "cam_trans": torch.zeros(2, 3),
        }

        first = model.build_specimen_prediction(predicted, 0)
        second = model.build_specimen_prediction(predicted, 1)

        assert torch.equal(first["betas"], torch.full((2, 5), 1.0))
        assert torch.equal(second["betas"], torch.full((2, 5), 2.0))
        # Same camera object for both: it is a property of the view, not the animal.
        assert first["fov"] is second["fov"]
        assert first["cam_rot"] is second["cam_rot"]

    def test_specimen_prediction_does_not_mutate_the_source(self):
        model = bare(MultiAnimalSMILRegressor, num_animals=1)
        predicted = {"animals": [{"global_rot": torch.zeros(1, 3)}], "fov": torch.zeros(1, 1)}
        model.build_specimen_prediction(predicted, 0)
        assert "fov" not in predicted["animals"][0]


class TestMultiViewWiring:
    def test_specimen_prediction_shares_the_per_view_cameras(self):
        model = bare(MultiAnimalMultiViewSMILRegressor, num_animals=2)
        predicted = {
            ANIMALS_KEY: [
                {"global_rot": torch.zeros(2, 3)},
                {"global_rot": torch.ones(2, 3)},
            ],
            "fov_per_view": [torch.zeros(2, 1), torch.zeros(2, 1)],
            "cam_rot_per_view": [torch.eye(3).expand(2, 3, 3)] * 2,
            "cam_trans_per_view": [torch.zeros(2, 3)] * 2,
            "num_views": 2,
            "view_mask": torch.ones(2, 2, dtype=torch.bool),
            "camera_indices": torch.zeros(2, 2, dtype=torch.long),
        }

        first = model.build_specimen_prediction(predicted, 0)
        second = model.build_specimen_prediction(predicted, 1)

        assert torch.equal(first["global_rot"], torch.zeros(2, 3))
        assert torch.equal(second["global_rot"], torch.ones(2, 3))
        for key in ("fov_per_view", "cam_rot_per_view", "cam_trans_per_view", "num_views", "view_mask"):
            assert first[key] is second[key]

    def test_multiview_mask_uses_keypoints_from_the_target_dicts(self):
        model = bare(MultiAnimalMultiViewSMILRegressor, num_animals=2, min_visible_keypoints_per_specimen=4)
        x_data, y_data = sample(present=(True, True), visibility=[[1] * 6, [1, 0, 0, 0, 0, 0]])
        mask = model._build_animal_mask([x_data], [y_data])
        assert mask.tolist() == [[True, False]]

    def test_camera_mode_is_forced_to_scene_head(self):
        # The multi-view model has per-view camera heads; 'first_specimen' would
        # be meaningless there, so the constructor pins it.
        model = bare(MultiAnimalMultiViewSMILRegressor, num_animals=2)
        assert model.multi_animal.camera_mode == "scene_head"
