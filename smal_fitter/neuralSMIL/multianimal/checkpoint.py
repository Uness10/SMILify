"""
Checkpoint migration between single-animal and multi-animal models.

Design doc §3: *"the new heads can be initialized by copying the pretrained
single-animal head weights"*.  That is not just a nice-to-have — a multi-animal
run that starts from scratch throws away every hour already spent on the
single-animal model, so being able to resume from an existing checkpoint is what
makes the prototype practical.

Three transitions are supported, all of them key-level rewrites of a
``state_dict`` (no model instantiation, no CUDA, no side effects):

single-animal  ->  multi-animal
    The one head's weights are copied into every specimen head.  The backbone
    and every other module keep their keys, so they load exactly as before.

multi-animal   ->  multi-animal with a different N
    Existing specimen heads keep their slot; new slots are seeded from a chosen
    donor slot (head 0 by default).  Dropping slots is allowed but reported.

multi-animal   ->  single-animal
    Specimen 0's head is written back to the single-animal key names, so a
    multi-animal checkpoint can still be evaluated with the base model.

Every function returns a :class:`MigrationReport` rather than printing, so
callers decide how to log and tests can assert on the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

import torch

#: Key prefix of the inherited transformer decoder head in a single-animal model.
SINGLE_TRANSFORMER_PREFIX = "transformer_head."

#: Top-level module names of the inline MLP head in a single-animal model.
SINGLE_MLP_MODULES = ("fc1", "ln1", "fc2", "ln2", "fc3", "ln3", "regressor")

#: Key prefix of the replicated head bank in a multi-animal model.
MULTI_REPLICATED_PREFIX = "specimen_heads.heads."

#: Key prefix of the shared head in a ``shared_query`` multi-animal model.
MULTI_SHARED_PREFIX = "specimen_heads.head."


@dataclass
class MigrationReport:
    """What a migration did, for logging and for tests to assert on."""

    source_layout: str
    target_layout: str
    num_animals: int
    heads_seeded: List[int] = field(default_factory=list)
    heads_kept: List[int] = field(default_factory=list)
    heads_dropped: List[int] = field(default_factory=list)
    keys_rewritten: int = 0
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-line human-readable summary."""
        parts = [f"{self.source_layout} -> {self.target_layout} (N={self.num_animals})"]
        if self.heads_kept:
            parts.append(f"kept heads {self.heads_kept}")
        if self.heads_seeded:
            parts.append(f"seeded heads {self.heads_seeded} from the pretrained head")
        if self.heads_dropped:
            parts.append(f"dropped heads {self.heads_dropped}")
        parts.append(f"{self.keys_rewritten} keys rewritten")
        return "; ".join(parts)


def detect_layout(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Classify a checkpoint's head layout.

    Returns:
        ``"multi_replicated"``, ``"multi_shared"``, ``"single_transformer"``,
        ``"single_mlp"`` or ``"unknown"``.
    """
    keys = list(state_dict.keys())
    if any(key.startswith(MULTI_REPLICATED_PREFIX) for key in keys):
        return "multi_replicated"
    if any(key.startswith(MULTI_SHARED_PREFIX) for key in keys):
        return "multi_shared"
    if any(key.startswith(SINGLE_TRANSFORMER_PREFIX) for key in keys):
        return "single_transformer"
    if any(key.split(".")[0] in SINGLE_MLP_MODULES for key in keys):
        return "single_mlp"
    return "unknown"


def count_specimen_heads(state_dict: Mapping[str, torch.Tensor]) -> int:
    """Number of specimen heads present in a ``multi_replicated`` checkpoint."""
    indices = set()
    for key in state_dict:
        if not key.startswith(MULTI_REPLICATED_PREFIX):
            continue
        remainder = key[len(MULTI_REPLICATED_PREFIX) :]
        head_index, _, _ = remainder.partition(".")
        if head_index.isdigit():
            indices.add(int(head_index))
    return (max(indices) + 1) if indices else 0


def to_multi_animal(
    state_dict: Mapping[str, torch.Tensor],
    num_animals: int,
    head_strategy: str = "replicated",
    donor_head: int = 0,
) -> tuple[Dict[str, torch.Tensor], MigrationReport]:
    """Rewrite a checkpoint so it loads into a multi-animal model.

    Args:
        state_dict: Source checkpoint (single- or multi-animal).
        num_animals: Slot count of the target model.
        head_strategy: ``"replicated"`` or ``"shared_query"``; must match the
            target model's configuration.
        donor_head: Which existing slot seeds new slots.

    Returns:
        ``(new_state_dict, report)``.  Keys that do not belong to a head are
        carried through untouched, so the backbone, the camera heads and the
        cross-view fusion all load normally.

    Raises:
        ValueError: on an unrecognised source layout or an invalid ``num_animals``.
    """
    if num_animals < 1:
        raise ValueError(f"num_animals must be >= 1, got {num_animals}")

    layout = detect_layout(state_dict)
    if layout == "unknown":
        raise ValueError(
            "cannot migrate this checkpoint: no recognisable parameter head found "
            f"(looked for '{SINGLE_TRANSFORMER_PREFIX}*', {SINGLE_MLP_MODULES}, "
            f"'{MULTI_REPLICATED_PREFIX}*', '{MULTI_SHARED_PREFIX}*')."
        )

    head_state, passthrough = _split_head_state(state_dict, layout)
    report = MigrationReport(
        source_layout=layout,
        target_layout=f"multi_{head_strategy}",
        num_animals=num_animals,
    )

    if head_strategy == "shared_query":
        donor = _donor_head_state(head_state, layout, donor_head, report)
        new_state = dict(passthrough)
        for key, value in donor.items():
            new_state[f"{MULTI_SHARED_PREFIX}{key}"] = value
        report.heads_seeded = [0]
        report.keys_rewritten = len(donor)
        return new_state, report

    if head_strategy != "replicated":
        raise ValueError(f"unknown head_strategy '{head_strategy}' (expected 'replicated' or 'shared_query')")

    new_state = dict(passthrough)
    rewritten = 0

    existing = head_state if layout == "multi_replicated" else {}
    donor = _donor_head_state(head_state, layout, donor_head, report)

    for index in range(num_animals):
        source = existing.get(index)
        if source is not None:
            report.heads_kept.append(index)
        else:
            source = donor
            report.heads_seeded.append(index)
        for key, value in source.items():
            new_state[f"{MULTI_REPLICATED_PREFIX}{index}.{key}"] = value.clone()
            rewritten += 1

    if layout == "multi_replicated":
        report.heads_dropped = [index for index in sorted(existing) if index >= num_animals]
        if report.heads_dropped:
            report.warnings.append(
                f"checkpoint had {len(existing)} specimen heads but the target model has "
                f"{num_animals}; heads {report.heads_dropped} were dropped."
            )

    report.keys_rewritten = rewritten
    return new_state, report


def to_single_animal(
    state_dict: Mapping[str, torch.Tensor],
    specimen_index: int = 0,
    head_type: str = "transformer_decoder",
) -> tuple[Dict[str, torch.Tensor], MigrationReport]:
    """Rewrite a multi-animal checkpoint so it loads into the base model.

    Useful for evaluating one specimen with the single-animal tooling, or for
    exporting a multi-animal run back to the single-animal inference scripts.
    """
    layout = detect_layout(state_dict)
    if layout not in ("multi_replicated", "multi_shared"):
        raise ValueError(f"expected a multi-animal checkpoint, found layout '{layout}'")

    head_state, passthrough = _split_head_state(state_dict, layout)
    report = MigrationReport(
        source_layout=layout,
        target_layout="single_transformer" if head_type == "transformer_decoder" else "single_mlp",
        num_animals=1,
    )

    donor = head_state[specimen_index] if layout == "multi_replicated" else head_state
    if donor is None:
        raise ValueError(f"specimen index {specimen_index} is not present in the checkpoint")

    new_state = dict(passthrough)
    prefix = SINGLE_TRANSFORMER_PREFIX if head_type == "transformer_decoder" else ""
    for key, value in donor.items():
        new_state[f"{prefix}{key}"] = value.clone()

    report.heads_kept = [specimen_index]
    report.keys_rewritten = len(donor)
    return new_state, report


def load_into_model(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    num_animals: Optional[int] = None,
    head_strategy: Optional[str] = None,
    strict: bool = False,
) -> MigrationReport:
    """Migrate ``state_dict`` if needed and load it into ``model``.

    The model's own configuration decides the target layout, so a training
    script can pass any checkpoint — single-animal, or multi-animal with a
    different ``N`` — and get the right thing without branching.

    Args:
        model: A constructed multi-animal regressor.
        state_dict: Checkpoint contents.
        num_animals: Override for the target slot count; defaults to the model's.
        head_strategy: Override for the target strategy; defaults to the model's.
        strict: Passed to ``load_state_dict``.  Left False by default because a
            single-animal checkpoint legitimately lacks the scene camera head.

    Returns:
        The :class:`MigrationReport`, whose ``warnings`` list names any module
        the checkpoint could not populate.
    """
    target_n = num_animals if num_animals is not None else getattr(model, "num_animals", 1)
    target_strategy = head_strategy or getattr(getattr(model, "specimen_heads", None), "strategy", "replicated")

    migrated, report = to_multi_animal(state_dict, target_n, head_strategy=target_strategy)
    missing, unexpected = model.load_state_dict(migrated, strict=strict)

    if missing:
        report.warnings.append(f"randomly initialised (absent from checkpoint): {sorted(missing)[:12]}")
    if unexpected:
        report.warnings.append(f"ignored (not present in this model): {sorted(unexpected)[:12]}")
    return report


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _split_head_state(state_dict: Mapping[str, torch.Tensor], layout: str):
    """Separate head parameters from everything else.

    Returns ``(head_state, passthrough)`` where ``head_state`` is a flat dict of
    head-local keys for the single/shared layouts, and a ``{index: dict}`` map
    for ``multi_replicated``.
    """
    passthrough: Dict[str, torch.Tensor] = {}

    if layout == "single_transformer":
        head: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith(SINGLE_TRANSFORMER_PREFIX):
                head[key[len(SINGLE_TRANSFORMER_PREFIX) :]] = value
            else:
                passthrough[key] = value
        return head, passthrough

    if layout == "single_mlp":
        head = {}
        for key, value in state_dict.items():
            if key.split(".")[0] in SINGLE_MLP_MODULES:
                head[key] = value
            else:
                passthrough[key] = value
        return head, passthrough

    if layout == "multi_shared":
        head = {}
        for key, value in state_dict.items():
            if key.startswith(MULTI_SHARED_PREFIX):
                head[key[len(MULTI_SHARED_PREFIX) :]] = value
            elif key.startswith("specimen_heads."):
                # specimen/context embeddings: rebuilt for the target N
                continue
            else:
                passthrough[key] = value
        return head, passthrough

    # multi_replicated
    heads: Dict[int, Dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        if key.startswith(MULTI_REPLICATED_PREFIX):
            remainder = key[len(MULTI_REPLICATED_PREFIX) :]
            head_index, _, tail = remainder.partition(".")
            if head_index.isdigit():
                heads.setdefault(int(head_index), {})[tail] = value
                continue
        if key.startswith("specimen_heads."):
            continue
        passthrough[key] = value
    return heads, passthrough


def _donor_head_state(head_state, layout: str, donor_head: int, report: MigrationReport):
    """Pick the head whose weights seed new specimen slots."""
    if layout == "multi_replicated":
        if donor_head in head_state:
            return head_state[donor_head]
        if not head_state:
            raise ValueError("checkpoint declares a replicated head bank but contains no head parameters")
        fallback = min(head_state)
        report.warnings.append(f"donor head {donor_head} not in checkpoint; seeding from head {fallback} instead")
        return head_state[fallback]
    return head_state


def summarize_specimen_heads(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, object]:
    """Describe a checkpoint's head layout without loading it into a model."""
    layout = detect_layout(state_dict)
    return {
        "layout": layout,
        "num_specimen_heads": count_specimen_heads(state_dict) if layout == "multi_replicated" else (
            1 if layout != "unknown" else 0
        ),
    }


def filter_optimizer_state(
    optimizer_state: MutableMapping[str, object],
    keep: bool = False,
) -> Optional[MutableMapping[str, object]]:
    """Decide whether an optimiser state can be reused after a head migration.

    Adding specimen heads changes the parameter list, so Adam moment buffers no
    longer line up with the parameters they belong to.  Reusing them silently
    would apply one head's momentum to another's weights, so the default is to
    drop the state and let the optimiser warm up again.

    Args:
        optimizer_state: The checkpoint's optimiser state.
        keep: Force reuse (only safe when ``N`` did not change).

    Returns:
        The state when ``keep``, otherwise ``None``.
    """
    return optimizer_state if keep else None


def specimen_head_keys(num_animals: int, head_keys: Sequence[str]) -> List[str]:
    """Expand head-local key names into the full multi-animal key names."""
    return [f"{MULTI_REPLICATED_PREFIX}{i}.{key}" for i in range(num_animals) for key in head_keys]
