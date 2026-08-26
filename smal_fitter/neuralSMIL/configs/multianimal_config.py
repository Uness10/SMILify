"""
Multi-animal configuration.

Lives in the ``configs`` package (rather than next to the model code) so that
every training knob stays in the project's one dataclass-based config system and
is JSON-serialisable through :func:`load_config` like everything else.

Example JSON fragment::

    {
      "mode": "multiview",
      "multi_animal": {
        "enabled": true,
        "num_animals": 3,
        "specimen_ids": ["mouse_a", "mouse_b", "mouse_c"],
        "head_strategy": "replicated",
        "loss_reduction": "mean"
      }
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

#: Realisations of the "N parameter heads" idea.  See
#: :mod:`smal_fitter.neuralSMIL.multianimal.specimen_heads`.
HEAD_STRATEGIES = ("replicated", "shared_query")

#: Where the *single-view* camera prediction comes from.  Multi-view always uses
#: the existing per-canonical-view camera heads, which are already scene level.
CAMERA_MODES = ("scene_head", "first_specimen")

#: How per-specimen losses are combined into the sample loss.
LOSS_REDUCTIONS = ("mean", "sum", "weighted_mean")


class MultiAnimalConfigError(ValueError):
    """Raised for an internally inconsistent multi-animal configuration."""


@dataclass
class MultiAnimalConfig:
    """Settings for reconstructing ``N`` known specimens per sample.

    Attributes:
        enabled: Master switch.  When False the model, datasets and trainers
            behave exactly as before — the multi-animal code is not reached at
            all, so existing runs are bit-for-bit unaffected.
        num_animals: Number of specimen slots ``N``.  Fixed for a run: head *i*
            is permanently bound to specimen *i*.
        specimen_ids: Stable identity labels, one per slot, used for per-specimen
            metrics and to assert that the dataset's ordering never shuffles.
            Defaults to ``specimen_0 … specimen_{N-1}``.
        head_strategy: ``"replicated"`` (N independent copies of the existing
            head — the design-document default) or ``"shared_query"`` (one head
            conditioned on a learned per-specimen embedding).
        tie_head_init: Start every replicated head from the same initialisation,
            matching "new heads copied from the existing head".
        init_heads_from_single_animal: When resuming from a single-animal
            checkpoint, copy its head weights into every specimen head.
        camera_mode: Single-view camera source.  ``"scene_head"`` adds a
            dedicated scene-level camera head (design doc §6: the camera belongs
            to the view, not to an animal).  ``"first_specimen"`` reads the
            camera out of specimen head 0, which reproduces the legacy
            single-animal model exactly and is the safe choice for ``N == 1``.
        scene_camera_hidden_dim: Hidden width of the scene camera head.
        loss_reduction: How per-specimen losses are aggregated (design doc §9).
        per_specimen_lr_scale: Optional per-slot learning-rate multipliers, for
            when one specimen is systematically harder than the others.
        require_stable_identity: Assert every sample in a batch lists the same
            specimen ids in the same order.  Strict head/specimen correspondence
            is what lets the loss skip Hungarian matching, so violating it is a
            silent correctness bug — checked by default.
        min_visible_keypoints_per_specimen: A specimen with fewer visible
            keypoints than this in a sample is treated as absent for that
            sample.  Design doc §10: keypoint supervision must respect
            per-specimen visibility once animals overlap.
        drop_absent_specimen_loss: Skip loss for specimen slots that are absent
            everywhere in a batch, instead of contributing a zero term.
    """

    enabled: bool = False
    num_animals: int = 1
    specimen_ids: List[str] = field(default_factory=list)

    head_strategy: str = "replicated"
    tie_head_init: bool = True
    init_heads_from_single_animal: bool = True

    camera_mode: str = "scene_head"
    scene_camera_hidden_dim: int = 256

    loss_reduction: str = "mean"
    per_specimen_lr_scale: Optional[List[float]] = None

    require_stable_identity: bool = True
    min_visible_keypoints_per_specimen: int = 0
    drop_absent_specimen_loss: bool = True

    def __post_init__(self):
        if not self.specimen_ids:
            self.specimen_ids = [f"specimen_{i}" for i in range(max(1, int(self.num_animals)))]

    def normalize(self) -> None:
        """Fill in derived defaults after a merge.

        ``__post_init__`` runs before the JSON/CLI merge, so a config that sets
        ``num_animals`` but not ``specimen_ids`` would otherwise be left with the
        single default id.  Auto-generated ids (``specimen_<i>``) are regenerated
        for the new count; explicitly authored ids are left alone so a genuine
        mismatch is still reported by :meth:`validate`.
        """
        n = max(1, int(self.num_animals))
        if len(self.specimen_ids) == n:
            return
        auto_generated = all(
            str(name) == f"specimen_{i}" for i, name in enumerate(self.specimen_ids)
        )
        if not self.specimen_ids or auto_generated:
            self.specimen_ids = [f"specimen_{i}" for i in range(n)]

    def validate(self) -> None:
        """Raise :class:`MultiAnimalConfigError` on an inconsistent configuration."""
        self.normalize()
        if not self.enabled:
            return

        if self.num_animals < 1:
            raise MultiAnimalConfigError(f"num_animals must be >= 1, got {self.num_animals}")
        if self.head_strategy not in HEAD_STRATEGIES:
            raise MultiAnimalConfigError(
                f"head_strategy '{self.head_strategy}' is not one of {HEAD_STRATEGIES}"
            )
        if self.camera_mode not in CAMERA_MODES:
            raise MultiAnimalConfigError(f"camera_mode '{self.camera_mode}' is not one of {CAMERA_MODES}")
        if self.loss_reduction not in LOSS_REDUCTIONS:
            raise MultiAnimalConfigError(f"loss_reduction '{self.loss_reduction}' is not one of {LOSS_REDUCTIONS}")
        if len(self.specimen_ids) != self.num_animals:
            raise MultiAnimalConfigError(
                f"specimen_ids has {len(self.specimen_ids)} entries but num_animals is {self.num_animals}"
            )
        if len(set(self.specimen_ids)) != len(self.specimen_ids):
            raise MultiAnimalConfigError(f"specimen_ids must be unique, got {self.specimen_ids}")
        if self.per_specimen_lr_scale is not None:
            if len(self.per_specimen_lr_scale) != self.num_animals:
                raise MultiAnimalConfigError(
                    f"per_specimen_lr_scale has {len(self.per_specimen_lr_scale)} entries, "
                    f"expected num_animals={self.num_animals}"
                )
            if any(scale <= 0 for scale in self.per_specimen_lr_scale):
                raise MultiAnimalConfigError("per_specimen_lr_scale entries must be > 0")
        if self.num_animals > 1 and self.camera_mode == "first_specimen":
            # Not fatal, but it means the camera gradient only ever reaches one
            # head, which contradicts §6 (camera is scene level).
            raise MultiAnimalConfigError(
                "camera_mode='first_specimen' is only meaningful for num_animals=1; "
                "with several specimens the camera must be predicted scene-level "
                "(camera_mode='scene_head')."
            )
        if self.min_visible_keypoints_per_specimen < 0:
            raise MultiAnimalConfigError("min_visible_keypoints_per_specimen must be >= 0")

    @property
    def is_active(self) -> bool:
        """True when the multi-animal code path should actually be taken."""
        return bool(self.enabled)

    def to_dict(self) -> dict:
        """Flat dict for the legacy dict-based training entry points."""
        return {
            "enabled": self.enabled,
            "num_animals": self.num_animals,
            "specimen_ids": list(self.specimen_ids),
            "head_strategy": self.head_strategy,
            "tie_head_init": self.tie_head_init,
            "init_heads_from_single_animal": self.init_heads_from_single_animal,
            "camera_mode": self.camera_mode,
            "scene_camera_hidden_dim": self.scene_camera_hidden_dim,
            "loss_reduction": self.loss_reduction,
            "per_specimen_lr_scale": (
                list(self.per_specimen_lr_scale) if self.per_specimen_lr_scale is not None else None
            ),
            "require_stable_identity": self.require_stable_identity,
            "min_visible_keypoints_per_specimen": self.min_visible_keypoints_per_specimen,
            "drop_absent_specimen_loss": self.drop_absent_specimen_loss,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "MultiAnimalConfig":
        """Build from a (possibly partial) dict, ignoring unknown keys."""
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}  # noqa: F821 - dataclass attribute
        return cls(**{k: v for k, v in data.items() if k in known})
