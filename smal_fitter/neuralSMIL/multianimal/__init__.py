"""
Multi-animal SMIL reconstruction.

Implements ``docs/design/multianimal.md``: keep the shared ViT backbone and all
existing SMILify machinery, add ``N - 1`` copies of the parameter head, bind each
head permanently to one known specimen, predict the camera at scene/view level,
run SMAL once per specimen, and apply the existing losses per specimen.

Layering (each module depends only on the ones above it)::

    schema.py            data contract for the animal axis (numpy only)
    batching.py          (B, N) <-> (B*N) tensor plumbing (torch only)
    parameter_layout.py  flat head-output layout + parsing
    heads.py             one parameter head, as a real nn.Module
    specimen_heads.py    the N-head bank: replicated | shared_query
    losses.py            per-specimen loss aggregation
    collate.py           rectangular animal axis at batch time
    datasets.py          N=1 promotion, per-track grouping
    hdf5_schema.py       on-disk animal axis
    checkpoint.py        single-animal <-> multi-animal weight migration
    regressor.py         single-view model  (needs torch + pytorch3d + config)
    multiview_regressor.py  multi-view model (needs torch + pytorch3d + config)

The two regressor modules are the only ones that pull in pytorch3d and the SMAL
model, so everything above them can be imported and unit-tested with plain
PyTorch.  They are therefore imported lazily by :func:`__getattr__` rather than
at package import time.

Typical use::

    from smal_fitter.neuralSMIL.configs.multianimal_config import MultiAnimalConfig
    from smal_fitter.neuralSMIL.multianimal import (
        MultiAnimalSMILRegressor, MultiAnimalDatasetAdapter, make_multianimal_collate_fn,
    )

    cfg = MultiAnimalConfig(enabled=True, num_animals=3,
                            specimen_ids=["mouse_a", "mouse_b", "mouse_c"])
    dataset = MultiAnimalDatasetAdapter(existing_dataset, num_animals=cfg.num_animals,
                                        specimen_ids=cfg.specimen_ids)
    loader = DataLoader(dataset, batch_size=8,
                        collate_fn=make_multianimal_collate_fn(cfg.num_animals, cfg.specimen_ids))
    model = MultiAnimalSMILRegressor(device, batch, 8, shape_family, False, multi_animal=cfg)
"""

from __future__ import annotations

from typing import Any

from .batching import (
    aggregate_specimen_losses,
    animal_mask_to_tensor,
    expand_scene_to_animals,
    flatten_animal_axis,
    masked_mean,
    merge_loss_components,
    select_specimen,
    stack_specimen_params,
    unflatten_animal_axis,
)
from .checkpoint import MigrationReport, load_into_model, to_multi_animal, to_single_animal
from .collate import make_multianimal_collate_fn, multianimal_collate_fn, pad_sample_to_num_animals
from .datasets import (
    GroupedSpecimenDataset,
    MultiAnimalDatasetAdapter,
    sleap_frame_key,
    sleap_track_specimen_key,
)
from .heads import MLPParameterHead, make_head_factory
from .hdf5_schema import DatasetLayout, declare_layout, detect_layout as detect_hdf5_layout
from .losses import MultiAnimalLossAggregator, apply_visibility_floor, weights_for_specimen
from .parameter_layout import ParameterLayout, parse_flat_parameter_vector
from .schema import (
    ANIMAL_MASK_KEY,
    ANIMALS_KEY,
    NUM_ANIMALS_KEY,
    SCHEMA_VERSION,
    SPECIMEN_IDS_KEY,
    MultiAnimalSchemaError,
    animal_mask_of,
    is_multi_animal,
    make_multi_animal_sample,
    num_animals_of,
    specimen_ids_of,
    specimen_target_view,
    split_batch_by_specimen,
    validate_sample,
    wrap_single_animal,
)
from .specimen_heads import (
    ReplicatedSpecimenHeads,
    SharedQuerySpecimenHeads,
    SpecimenHeads,
    build_specimen_heads,
)

#: Names resolved lazily because importing them pulls in pytorch3d, the SMAL
#: model pickle and the legacy root ``config`` module.
_LAZY = {
    "MultiAnimalSMILRegressor": ".regressor",
    "create_multianimal_regressor": ".regressor",
    "MultiAnimalMultiViewSMILRegressor": ".multiview_regressor",
    "create_multianimal_multiview_regressor": ".multiview_regressor",
}


def __getattr__(name: str) -> Any:
    """Import the heavyweight regressors on first access (PEP 562)."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


__all__ = [
    # Schema / data contract
    "ANIMALS_KEY",
    "ANIMAL_MASK_KEY",
    "NUM_ANIMALS_KEY",
    "SPECIMEN_IDS_KEY",
    "SCHEMA_VERSION",
    "MultiAnimalSchemaError",
    "is_multi_animal",
    "num_animals_of",
    "animal_mask_of",
    "specimen_ids_of",
    "wrap_single_animal",
    "make_multi_animal_sample",
    "specimen_target_view",
    "split_batch_by_specimen",
    "validate_sample",
    # Batching
    "flatten_animal_axis",
    "unflatten_animal_axis",
    "expand_scene_to_animals",
    "stack_specimen_params",
    "select_specimen",
    "animal_mask_to_tensor",
    "masked_mean",
    "aggregate_specimen_losses",
    "merge_loss_components",
    # Heads
    "ParameterLayout",
    "parse_flat_parameter_vector",
    "MLPParameterHead",
    "make_head_factory",
    "SpecimenHeads",
    "ReplicatedSpecimenHeads",
    "SharedQuerySpecimenHeads",
    "build_specimen_heads",
    # Loss
    "MultiAnimalLossAggregator",
    "weights_for_specimen",
    "apply_visibility_floor",
    # Data
    "multianimal_collate_fn",
    "make_multianimal_collate_fn",
    "pad_sample_to_num_animals",
    "MultiAnimalDatasetAdapter",
    "GroupedSpecimenDataset",
    "sleap_frame_key",
    "sleap_track_specimen_key",
    "DatasetLayout",
    "declare_layout",
    "detect_hdf5_layout",
    # Checkpoints
    "MigrationReport",
    "to_multi_animal",
    "to_single_animal",
    "load_into_model",
    # Models (lazy)
    "MultiAnimalSMILRegressor",
    "create_multianimal_regressor",
    "MultiAnimalMultiViewSMILRegressor",
    "create_multianimal_multiview_regressor",
]
