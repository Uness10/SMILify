"""
Entry points for wiring multi-animal support into the existing trainers.

The single- and multi-view training scripts each construct a regressor, a
dataset and a collate function.  Rather than scattering ``if multi_animal:``
branches through those (very large) scripts, they call the three helpers here,
which return the single-animal object unchanged when the feature is off.  The
resulting diff in each trainer is a handful of lines, and "multi-animal is
disabled by default and then costs nothing" is enforced in one place.

Typical use in a training script::

    multi_animal = resolve_multi_animal_config(config)

    model = build_regressor(
        multi_animal,
        single_animal_factory=create_multiview_regressor,
        multi_animal_factory=create_multianimal_multiview_regressor,
        device=device, batch_size=..., ...,
    )
    dataset = wrap_dataset(dataset, multi_animal)
    collate_fn = resolve_collate_fn(multiview_collate_fn, multi_animal)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from smal_fitter.neuralSMIL.configs.multianimal_config import MultiAnimalConfig

from .collate import compose_multianimal_collate
from .datasets import MultiAnimalDatasetAdapter


def resolve_multi_animal_config(config: Any) -> MultiAnimalConfig:
    """Extract the multi-animal settings from any of the config shapes in use.

    Accepts the dataclass config objects (``BaseTrainingConfig`` and
    subclasses), the flat legacy dicts the training ``main()`` functions
    consume, and a bare :class:`MultiAnimalConfig`.  Returns a *disabled*
    config when the section is absent, so callers never have to null-check.

    Raises:
        MultiAnimalConfigError: if the section is present but inconsistent —
            better at startup than at the first batch.
    """
    if isinstance(config, MultiAnimalConfig):
        resolved = config
    elif isinstance(config, dict):
        resolved = MultiAnimalConfig.from_dict(config.get("multi_animal"))
    else:
        resolved = getattr(config, "multi_animal", None) or MultiAnimalConfig()

    resolved.validate()
    return resolved


def build_regressor(
    multi_animal: MultiAnimalConfig,
    single_animal_factory: Callable[..., Any],
    multi_animal_factory: Callable[..., Any],
    **kwargs: Any,
) -> Any:
    """Construct the regressor the configuration asks for.

    When multi-animal is disabled the single-animal factory is called with the
    unchanged keyword arguments, so an existing run is bit-for-bit unaffected.
    When it is enabled the multi-animal factory receives the same arguments plus
    ``multi_animal=...``.
    """
    if not multi_animal.is_active:
        return single_animal_factory(**kwargs)
    return multi_animal_factory(multi_animal=multi_animal, **kwargs)


def wrap_dataset(dataset: Any, multi_animal: MultiAnimalConfig, validate: bool = True) -> Any:
    """Present ``dataset`` with the configured number of specimen slots.

    A dataset that already emits multi-animal samples is padded/truncated to the
    configured ``N``; a legacy single-animal dataset is promoted to ``N = 1``
    slots with the remaining slots marked absent.  Returns the dataset unchanged
    when multi-animal is disabled.
    """
    if not multi_animal.is_active:
        return dataset
    return MultiAnimalDatasetAdapter(
        dataset,
        num_animals=multi_animal.num_animals,
        specimen_ids=multi_animal.specimen_ids,
        validate=validate,
    )


def resolve_collate_fn(single_animal_collate: Callable, multi_animal: MultiAnimalConfig) -> Callable:
    """Return the collate function to hand to the ``DataLoader``.

    The multi-animal collate keeps the animal axis rectangular at the run's
    configured ``N``, which is what allows head ``i`` to be indexed by specimen
    ``i`` in every batch.
    """
    if not multi_animal.is_active:
        return single_animal_collate
    # Wrap rather than replace: the training scripts' collate functions also
    # unpack the batch, carry `available_labels` and log dataset composition.
    return compose_multianimal_collate(
        single_animal_collate, multi_animal.num_animals, multi_animal.specimen_ids
    )


def load_checkpoint_state(
    model: Any,
    state_dict: Dict[str, Any],
    multi_animal: MultiAnimalConfig,
    strict: bool = False,
) -> Optional[str]:
    """Load a checkpoint into ``model``, migrating the head layout if needed.

    Returns a one-line summary of what the migration did (for the training log),
    or ``None`` when no migration was required.
    """
    if not multi_animal.is_active or not multi_animal.init_heads_from_single_animal:
        model.load_state_dict(state_dict, strict=strict)
        return None

    from .checkpoint import load_into_model

    report = load_into_model(
        model,
        state_dict,
        num_animals=multi_animal.num_animals,
        head_strategy=multi_animal.head_strategy,
        strict=strict,
    )
    return report.summary()


def describe(multi_animal: MultiAnimalConfig) -> str:
    """One-line status string for the training banner."""
    if not multi_animal.is_active:
        return "Multi-animal: disabled (single specimen per sample)"
    return (
        f"Multi-animal: {multi_animal.num_animals} specimens "
        f"{list(multi_animal.specimen_ids)}, heads={multi_animal.head_strategy}, "
        f"camera={multi_animal.camera_mode}, loss_reduction={multi_animal.loss_reduction}"
    )
