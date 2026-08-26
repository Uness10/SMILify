"""
Multi-view multi-animal SMIL regressor (design doc "Prototype 2").

::

    V images
      |
      +--> shared ViT --> cross-view fusion --+--> V camera heads --> V cameras
                                              |
                                              +--> N specimen heads --> theta_1..theta_N
                                                        |
                                                        +--> SMAL --> N meshes
                                                                |
                                              the SAME N animals are projected
                                              into EVERY camera --> per-specimen loss

The multi-view model already treats the camera as a per-view (scene-level)
quantity and already fuses ``V`` views into one body prediction, so extending it
to ``N`` specimens needs only one structural change: the single body head
becomes the ``N``-head bank.  Concretely this class overrides

* :meth:`_predict_body_params` — decode ``N`` specimens from the fused features.
  Because the inherited ``forward_multiview`` merges whatever this returns into
  its output dict, the rest of the forward pass (backbone chunking, view
  embeddings, cross-view attention, per-view camera heads) is reused untouched.
* :meth:`predict_from_multiview_batch` — normalise samples and attach the
  ``(B, N)`` presence mask.
* :meth:`compute_multiview_batch_loss` — evaluate the inherited multi-view loss
  once per specimen and aggregate.

Nothing about triangulation, view masking or 2D reprojection needs to know that
several animals exist: each specimen is scored through the same shared cameras.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from smal_fitter.neuralSMIL.configs.multianimal_config import MultiAnimalConfig
from smal_fitter.neuralSMIL.multiview_smil_regressor import MultiViewSMILImageRegressor

from .batching import animal_mask_to_tensor
from .heads import make_head_factory
from .losses import MultiAnimalLossAggregator, apply_visibility_floor, presence_counts_from_mask
from .parameter_layout import strip_camera
from .schema import (
    ANIMALS_KEY,
    assert_stable_identity,
    animal_mask_of,
    is_multi_animal,
    num_animals_of,
    specimen_ids_of,
    specimen_target_view,
    validate_sample,
    wrap_single_animal,
)
from .specimen_heads import build_specimen_heads

#: Per-view camera predictions produced by the inherited camera heads.  They are
#: scene level and are shared verbatim by every specimen.
PER_VIEW_CAMERA_KEYS = ("fov_per_view", "cam_rot_per_view", "cam_trans_per_view")

#: Scene-level bookkeeping that the inherited multi-view loss reads off the
#: prediction dict.
SCENE_CONTEXT_KEYS = ("num_views", "view_mask", "camera_indices")


class MultiAnimalMultiViewSMILRegressor(MultiViewSMILImageRegressor):
    """Reconstructs ``N`` known specimens from ``V`` synchronised views.

    Args:
        device: Torch device.
        data_batch: Placeholder batch for ``SMALFitter`` initialisation.
        batch_size: Batch size.
        shape_family: SMIL shape family.
        use_unity_prior: Whether to use the Unity shape prior.
        multi_animal: Multi-animal settings; ``enabled`` must be True.
        **kwargs: Forwarded to :class:`MultiViewSMILImageRegressor`.
    """

    def __init__(
        self,
        device,
        data_batch,
        batch_size,
        shape_family,
        use_unity_prior,
        multi_animal: Optional[MultiAnimalConfig] = None,
        **kwargs,
    ):
        super().__init__(device, data_batch, batch_size, shape_family, use_unity_prior, **kwargs)

        self.multi_animal = multi_animal or MultiAnimalConfig(enabled=True, num_animals=1)
        if not self.multi_animal.enabled:
            raise ValueError(
                "MultiAnimalMultiViewSMILRegressor requires multi_animal.enabled=True; "
                "use MultiViewSMILImageRegressor for single-animal runs."
            )
        # The multi-view camera is per-view by construction, so the single-view
        # 'first_specimen' camera shortcut is not applicable here.
        self.multi_animal.camera_mode = "scene_head"
        self.multi_animal.validate()

        self.num_animals = self.multi_animal.num_animals
        self.specimen_ids = list(self.multi_animal.specimen_ids)

        self._build_specimen_heads()

        self.loss_aggregator = MultiAnimalLossAggregator(
            num_animals=self.num_animals,
            specimen_ids=self.specimen_ids,
            reduction=self.multi_animal.loss_reduction,
            drop_absent=self.multi_animal.drop_absent_specimen_loss,
        )

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def _build_specimen_heads(self) -> None:
        """Swap the inherited single body head for the ``N``-head bank."""
        head_factory = make_head_factory(self)

        if self.head_type == "transformer_decoder" and hasattr(self, "transformer_head"):
            del self.transformer_head
        elif self.head_type == "mlp":
            for name in ("fc1", "ln1", "fc2", "ln2", "fc3", "ln3", "regressor", "dropout"):
                if hasattr(self, name):
                    delattr(self, name)

        context_dim = self.backbone.get_spatial_dim() if hasattr(self.backbone, "get_spatial_dim") else self.feature_dim

        self.specimen_heads = build_specimen_heads(
            strategy=self.multi_animal.head_strategy,
            head_factory=head_factory,
            num_animals=self.num_animals,
            feature_dim=self.feature_dim,
            context_dim=context_dim,
            tie_first_head_init=self.multi_animal.tie_head_init,
        ).to(self.device)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def _predict_body_params(
        self,
        features: torch.Tensor,
        batch_size: int,
        view_mask: Optional[torch.Tensor] = None,
        patch_tokens: Optional[torch.Tensor] = None,
        camera_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Decode every specimen from the cross-view-fused features.

        Overrides the single-body version.  The inherited ``forward_multiview``
        splices the returned dict into its output, so returning the specimen list
        here is all that is needed to make the whole multi-view forward pass
        multi-animal.

        Returns:
            ``{"animals": [params_0, ..., params_{N-1}], "num_animals": N,
            "specimen_ids": [...]}`` plus specimen 0's ``iteration_history`` at
            the top level so existing IEF diagnostics keep working.
        """
        if self.head_type == "transformer_decoder":
            global_feats, spatial_feats = self._prepare_decoder_inputs(
                features, view_mask=view_mask, patch_tokens=patch_tokens, camera_indices=camera_indices
            )
        elif self.head_type == "mlp":
            # forward_multiview already pooled and passed features through
            # body_aggregator for the MLP path.
            global_feats, spatial_feats = features, None
        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}")

        specimen_params = self.specimen_heads(global_feats, spatial_feats)

        for params in specimen_params:
            if self.scale_trans_mode == "entangled_with_betas" and "betas" in params:
                log_beta_scales, betas_trans = self._transform_betas_to_joint_values(params["betas"])
                params["log_beta_scales"] = log_beta_scales
                params["betas_trans"] = betas_trans
            # The camera belongs to the view, not to an animal: the per-view
            # camera heads are the only camera source in this model.
            strip_camera(params, inplace=True)

        output: Dict[str, Any] = {
            ANIMALS_KEY: specimen_params,
            "num_animals": self.num_animals,
            "specimen_ids": list(self.specimen_ids),
        }
        if "iteration_history" in specimen_params[0]:
            output["iteration_history"] = specimen_params[0]["iteration_history"]
        return output

    # ------------------------------------------------------------------ #
    # Batch prediction
    # ------------------------------------------------------------------ #

    def predict_from_multiview_batch(
        self,
        x_data_batch: List[Dict[str, Any]],
        y_data_batch: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        """Normalise the batch, run the shared forward pass, attach the mask.

        Images, cameras and view masks are scene level, so the inherited
        implementation does all the heavy lifting unchanged; this wrapper only
        promotes legacy single-animal samples, checks the identity ordering and
        records which specimens are actually present.
        """
        x_norm, y_norm = self._normalize_multiview_batch(x_data_batch, y_data_batch)
        if not x_norm:
            return None, None, None

        predicted_params, _, auxiliary_data = super().predict_from_multiview_batch(x_norm, y_norm)
        if predicted_params is None:
            return None, None, None

        animal_mask = self._build_animal_mask(x_norm, y_norm)
        predicted_params["animal_mask"] = animal_mask

        auxiliary_data = dict(auxiliary_data or {})
        auxiliary_data.update(
            {
                "multi_animal": True,
                "num_animals": self.num_animals,
                "specimen_ids": list(self.specimen_ids),
                "animal_mask": animal_mask,
            }
        )
        return predicted_params, y_norm, auxiliary_data

    def _normalize_multiview_batch(
        self,
        x_data_batch: Sequence[Dict[str, Any]],
        y_data_batch: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Promote legacy samples to ``N = 1`` and assert a stable specimen order."""
        x_out: List[Dict[str, Any]] = []
        y_out: List[Dict[str, Any]] = []
        ids: List[List[str]] = []

        for x_data, y_data in zip(x_data_batch, y_data_batch):
            if not is_multi_animal(x_data, y_data):
                x_data, y_data = wrap_single_animal(x_data, y_data, specimen_id=self.specimen_ids[0])
            validate_sample(x_data, y_data)
            x_out.append(x_data)
            y_out.append(y_data)
            ids.append(specimen_ids_of(x_data, y_data, num_animals_of(x_data, y_data)))

        if self.multi_animal.require_stable_identity:
            assert_stable_identity(ids)

        return x_out, y_out

    def _build_animal_mask(
        self,
        x_norm: Sequence[Dict[str, Any]],
        y_norm: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        """``(B, N)`` presence mask, tightened by the per-specimen visibility floor.

        For multi-view the visibility floor counts keypoints visible in *any*
        view: a mouse hidden in one camera but clearly seen in another is still
        supervised (design doc §10).
        """
        masks = [animal_mask_of(x_data, y_data, self.num_animals) for x_data, y_data in zip(x_norm, y_norm)]
        mask = animal_mask_to_tensor(masks, self.num_animals, device=self.device)

        floor = self.multi_animal.min_visible_keypoints_per_specimen
        if floor <= 0:
            return mask

        counts = torch.zeros_like(mask, dtype=torch.long)
        for row, y_data in enumerate(y_norm):
            animals = y_data.get(ANIMALS_KEY) or []
            for index in range(min(self.num_animals, len(animals))):
                visibility = animals[index].get("keypoint_visibility")
                if visibility is None:
                    continue
                counts[row, index] = int(np.asarray(visibility).astype(bool).sum())
        return apply_visibility_floor(mask, counts, floor)

    # ------------------------------------------------------------------ #
    # Loss
    # ------------------------------------------------------------------ #

    def compute_multiview_batch_loss(
        self,
        predicted_params: Dict[str, Any],
        target_data: List[Dict[str, Any]],
        loss_weights: Optional[Dict[str, float]] = None,
        return_components: bool = False,
    ):
        """Evaluate the inherited multi-view loss once per specimen.

        Every specimen is scored against the *same* per-view cameras, which is
        exactly the "same N 3D animals projected into every camera" the design
        asks for.  Camera supervision is applied on specimen 0 only so it is not
        counted ``N`` times.

        Falls back to the single-animal implementation when handed a plain
        single-animal prediction dict.
        """
        if not isinstance(predicted_params, dict) or ANIMALS_KEY not in predicted_params:
            return super().compute_multiview_batch_loss(
                predicted_params, target_data, loss_weights=loss_weights, return_components=return_components
            )

        animal_mask = predicted_params.get("animal_mask")
        presence = (
            presence_counts_from_mask(animal_mask)
            if animal_mask is not None
            else [len(target_data)] * self.num_animals
        )

        def specimen_loss(index: int, weights: Dict[str, float]):
            specimen_prediction = self.build_specimen_prediction(predicted_params, index)
            specimen_targets = [
                specimen_target_view(
                    y_data,
                    index,
                    self.num_animals,
                    present=bool(animal_mask[row, index]) if animal_mask is not None else True,
                )
                for row, y_data in enumerate(target_data)
            ]
            return super(MultiAnimalMultiViewSMILRegressor, self).compute_multiview_batch_loss(
                specimen_prediction,
                specimen_targets,
                loss_weights=weights or None,
                return_components=True,
            )

        return self.loss_aggregator(
            specimen_loss,
            loss_weights or {},
            presence,
            device=self.device,
            return_components=return_components,
        )

    def build_specimen_prediction(self, predicted_params: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Single-animal view of the multi-animal multi-view prediction.

        Body parameters come from specimen ``index``; the per-view cameras and
        the view bookkeeping are shared verbatim, so the result is precisely what
        the inherited multi-view loss expects.
        """
        specimen = dict(predicted_params[ANIMALS_KEY][index])
        for key in PER_VIEW_CAMERA_KEYS + SCENE_CONTEXT_KEYS:
            if key in predicted_params:
                specimen[key] = predicted_params[key]
        return specimen

    def describe(self) -> str:
        """One-line summary for training logs."""
        return (
            f"{type(self).__name__}(N={self.num_animals}, ids={self.specimen_ids}, "
            f"views={self.num_canonical_cameras}, {self.specimen_heads.describe()})"
        )


def create_multianimal_multiview_regressor(
    device,
    batch_size: int,
    shape_family,
    use_unity_prior: bool,
    multi_animal: MultiAnimalConfig,
    max_views: int = 4,
    canonical_camera_order: Optional[List[str]] = None,
    data_batch=None,
    **kwargs,
) -> MultiAnimalMultiViewSMILRegressor:
    """Factory mirroring ``create_multiview_regressor`` for the multi-animal model."""
    if data_batch is None:
        # Mirror create_multiview_regressor: the placeholder batch must match the
        # backbone's native resolution, since the renderer is sized from it.
        from smal_fitter.neuralSMIL.backbone_factory import BackboneFactory

        backbone_name = kwargs.get("backbone_name", "resnet152")
        resolution = kwargs.get("input_resolution") or BackboneFactory.get_default_input_resolution(backbone_name)
        data_batch = torch.zeros(batch_size, 3, resolution, resolution, dtype=torch.float32, device=device)

    return MultiAnimalMultiViewSMILRegressor(
        device=device,
        data_batch=data_batch,
        batch_size=batch_size,
        shape_family=shape_family,
        use_unity_prior=use_unity_prior,
        multi_animal=multi_animal,
        max_views=max_views,
        canonical_camera_order=canonical_camera_order,
        **kwargs,
    )
