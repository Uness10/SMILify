"""
Single-view multi-animal SMIL regressor (design doc "Prototype 1").

::

    Image
     |
     +-> shared ViT ----+--> scene camera head --> camera
                        |
                        +--> N specimen heads --> theta_1 .. theta_N
                                                    |
                                                    +--> SMAL (batched) --> N meshes
                                                            |
                                                            +--> shared camera --> 2D
                                                                    |
                                                                    +--> per-specimen loss

Everything except the head bank and the camera routing is inherited unchanged
from :class:`~smal_fitter.neuralSMIL.smil_image_regressor.SMILImageRegressor`:
the backbone, the SMAL forward, the renderer, every loss term, the availability
masking, the joint-limit penalty.  The multi-animal behaviour is obtained by
overriding exactly three methods -- :meth:`forward`, :meth:`predict_from_batch`
and :meth:`compute_batch_loss` -- which are also the only three the single-view
trainer calls.  Existing training scripts therefore need no changes beyond
constructing this class instead of the base one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from smal_fitter.neuralSMIL.configs.multianimal_config import MultiAnimalConfig
from smal_fitter.neuralSMIL.multiview_smil_regressor import CameraHead
from smal_fitter.neuralSMIL.smil_image_regressor import SMILImageRegressor

from .batching import animal_mask_to_tensor
from .heads import make_head_factory
from .losses import MultiAnimalLossAggregator, apply_visibility_floor, presence_counts_from_mask
from .parameter_layout import camera_only, strip_camera
from .schema import (
    assert_stable_identity,
    animal_mask_of,
    is_multi_animal,
    num_animals_of,
    specimen_ids_of,
    split_batch_by_specimen,
    validate_sample,
    wrap_single_animal,
)
from .specimen_heads import build_specimen_heads


class MultiAnimalSMILRegressor(SMILImageRegressor):
    """Predicts ``N`` known specimens plus one scene camera from a single image.

    Args:
        device: Torch device.
        data_batch: Placeholder batch for ``SMALFitter`` initialisation.
        batch_size: Batch size.
        shape_family: SMIL shape family.
        use_unity_prior: Whether to use the Unity shape prior.
        multi_animal: Multi-animal settings.  ``enabled`` must be True; use the
            plain :class:`SMILImageRegressor` otherwise.
        **kwargs: Forwarded verbatim to :class:`SMILImageRegressor` (backbone,
            head type, rotation representation, scale/trans mode, ...).
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
                "MultiAnimalSMILRegressor requires multi_animal.enabled=True; "
                "use SMILImageRegressor for single-animal runs."
            )
        self.multi_animal.validate()

        self.num_animals = self.multi_animal.num_animals
        self.specimen_ids = list(self.multi_animal.specimen_ids)

        self._build_specimen_heads()
        self._build_scene_camera_head()

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
        """Replace the inherited single head with the ``N``-head bank.

        The parent constructor already built one head; it is released here so
        the model never carries an unused extra head, and the *same* factory the
        parent used builds the ``N`` replacements, which keeps head
        hyper-parameters defined in exactly one place.
        """
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

    def _build_scene_camera_head(self) -> None:
        """Add the scene/view-level camera head (design doc §6).

        The camera describes the *view*, so it must be predicted once per image
        and shared by every specimen -- not duplicated per animal.  The head is
        the same :class:`CameraHead` the multi-view model already uses, so the
        two paths cannot drift apart.
        """
        if self.multi_animal.camera_mode == "scene_head":
            self.scene_camera_head = CameraHead(
                self.feature_dim, hidden_dim=self.multi_animal.scene_camera_hidden_dim
            ).to(self.device)
        else:
            self.scene_camera_head = None

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, images: torch.Tensor) -> Dict[str, Any]:
        """Run the shared backbone once and decode every specimen.

        Args:
            images: ``(B, 3, H, W)``.

        Returns:
            Dict with:
                ``animals``: list of ``N`` per-specimen parameter dicts, each
                    holding ``global_rot``, ``joint_rot``, ``betas``, ``trans``
                    and (mode-dependent) ``log_beta_scales`` / ``betas_trans``.
                    Camera entries are *not* present -- the camera is scene level.
                ``fov`` / ``cam_rot`` / ``cam_trans``: the shared scene camera.
                ``num_animals``, ``specimen_ids``: identity bookkeeping.
                ``iteration_history``: specimen 0's IEF history, kept at the top
                    level so existing IEF diagnostics keep working.
        """
        global_features, spatial_features = self._extract_features(images)

        specimen_params = self.specimen_heads(global_features, spatial_features)

        camera = self._select_camera(specimen_params, global_features)

        for params in specimen_params:
            self._postprocess_specimen_params(params)
            strip_camera(params, inplace=True)

        output: Dict[str, Any] = {
            "animals": specimen_params,
            "num_animals": self.num_animals,
            "specimen_ids": list(self.specimen_ids),
        }
        output.update(camera)
        if "iteration_history" in specimen_params[0]:
            output["iteration_history"] = specimen_params[0]["iteration_history"]
        return output

    def _extract_features(self, images: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Pooled and (when available) spatial backbone features."""
        if self.head_type == "transformer_decoder" and hasattr(self.backbone, "forward_with_spatial"):
            return self.backbone.forward_with_spatial(images)
        return self.backbone(images), None

    def _select_camera(
        self,
        specimen_params: Sequence[Dict[str, torch.Tensor]],
        global_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Produce the single scene-level camera for this image."""
        if self.scene_camera_head is not None:
            return self.scene_camera_head(global_features)
        # camera_mode == "first_specimen": reproduce the legacy single-animal
        # model exactly (only valid for N == 1, enforced by the config).
        return camera_only(specimen_params[0])

    def _postprocess_specimen_params(self, params: Dict[str, torch.Tensor]) -> None:
        """Apply the scale/trans mode transform the base model applies per sample."""
        if self.scale_trans_mode == "entangled_with_betas" and "betas" in params:
            log_beta_scales, betas_trans = self._transform_betas_to_joint_values(params["betas"])
            params["log_beta_scales"] = log_beta_scales
            params["betas_trans"] = betas_trans

    # ------------------------------------------------------------------ #
    # Batch prediction
    # ------------------------------------------------------------------ #

    def predict_from_batch(self, x_data_batch, y_data_batch):
        """Multi-animal replacement for the single-animal batch step.

        The backbone and the image preprocessing run **once** for the batch; the
        per-specimen target assembly reuses the inherited
        :meth:`SMILImageRegressor.assemble_batch_inputs` verbatim, so every
        availability-masking rule is shared with the single-animal path.

        Returns:
            ``(predicted_params, target_bundle, auxiliary_data)`` with the same
            three-tuple contract the trainers expect.  ``target_bundle`` carries
            one entry per specimen plus, for backwards compatibility, specimen
            0's targets at the top level.
        """
        x_norm, y_norm = self._normalize_batch(x_data_batch, y_data_batch)
        if not x_norm:
            return None, None, None

        specimen_targets: List[Dict[str, Any]] = []
        specimen_aux: List[Dict[str, Any]] = []
        batch_images = None

        for index in range(self.num_animals):
            x_slice, y_slice, _ = split_batch_by_specimen(x_norm, y_norm, index, self.num_animals)
            images, targets, aux = self.assemble_batch_inputs(x_slice, y_slice)
            if images is None:
                return None, None, None
            if batch_images is None:
                batch_images = images  # images are scene level: identical for every specimen
            specimen_targets.append(targets)
            specimen_aux.append(aux)

        image_tensor = self.preprocess_image(batch_images).to(self.device)
        predicted_params = self.forward(image_tensor)

        if self.fixed_camera:
            self._apply_fixed_scene_camera(predicted_params, specimen_targets[0])

        for targets, aux in zip(specimen_targets, specimen_aux):
            self.merge_available_label_masks(targets, aux)

        animal_mask = self._build_animal_mask(x_norm, y_norm, specimen_aux)
        predicted_params["animal_mask"] = animal_mask

        target_bundle: Dict[str, Any] = {
            "_multi_animal": True,
            "num_animals": self.num_animals,
            "specimen_ids": list(self.specimen_ids),
            "specimens": specimen_targets,
            "animal_mask": animal_mask,
        }
        # Backwards compatibility: expose specimen 0's targets at the top level so
        # single-animal visualisation/metric helpers keep working unchanged.
        target_bundle.update({k: v for k, v in specimen_targets[0].items() if k not in target_bundle})

        auxiliary_data: Dict[str, Any] = {
            "multi_animal": True,
            "num_animals": self.num_animals,
            "specimen_ids": list(self.specimen_ids),
            "specimens": specimen_aux,
            "animal_mask": animal_mask,
        }
        auxiliary_data.update({k: v for k, v in specimen_aux[0].items() if k not in auxiliary_data})

        return predicted_params, target_bundle, auxiliary_data

    def _normalize_batch(
        self,
        x_data_batch: Sequence[Dict[str, Any]],
        y_data_batch: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Promote legacy samples, drop image-less ones and check identity order.

        Dropping the image-less samples *here* (rather than letting each
        per-specimen call drop them independently) is what guarantees every
        specimen sees the same batch rows in the same order, which the shared
        backbone pass depends on.
        """
        x_out: List[Dict[str, Any]] = []
        y_out: List[Dict[str, Any]] = []
        ids: List[List[str]] = []

        for x_data, y_data in zip(x_data_batch, y_data_batch):
            if x_data.get("input_image_data") is None:
                continue
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
        specimen_aux: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        """``(B, N)`` presence mask, tightened by the per-specimen visibility floor."""
        masks = [animal_mask_of(x_data, y_data, self.num_animals) for x_data, y_data in zip(x_norm, y_norm)]
        mask = animal_mask_to_tensor(masks, self.num_animals, device=self.device)

        floor = self.multi_animal.min_visible_keypoints_per_specimen
        if floor <= 0:
            return mask

        counts = torch.zeros_like(mask, dtype=torch.long)
        for index, aux in enumerate(specimen_aux):
            for row, keypoint_data in enumerate(aux.get("keypoint_data", [])):
                if keypoint_data is None:
                    continue
                visibility = keypoint_data.get("keypoint_visibility")
                if visibility is None:
                    continue
                counts[row, index] = int(np.asarray(visibility).astype(bool).sum())
        return apply_visibility_floor(mask, counts, floor)

    def _apply_fixed_scene_camera(
        self,
        predicted_params: Dict[str, Any],
        specimen_zero_targets: Dict[str, Any],
    ) -> None:
        """Camera-centric mode for the scene camera.

        Identical to :meth:`SMILImageRegressor.apply_fixed_camera` but reads the
        batch size off a specimen's body parameters, since the multi-animal
        prediction dict has no top-level ``global_rot``.
        """
        batch_size = predicted_params["animals"][0]["global_rot"].shape[0]
        predicted_params["cam_rot"] = (
            torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, 3, 3).contiguous()
        )
        predicted_params["cam_trans"] = torch.zeros(batch_size, 3, device=self.device)
        gt_fov = specimen_zero_targets.get("fov") if specimen_zero_targets is not None else None
        if gt_fov is not None:
            predicted_params["fov"] = gt_fov.to(self.device).reshape(batch_size, 1)

    # ------------------------------------------------------------------ #
    # Loss
    # ------------------------------------------------------------------ #

    def compute_batch_loss(
        self,
        predicted_params: Dict[str, Any],
        target_params_batch: Dict[str, Any],
        auxiliary_data: Optional[Dict[str, Any]] = None,
        return_components: bool = False,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        """Sum the existing single-animal loss over the specimens.

        Each specimen is scored with the *unmodified* inherited loss (2D/3D
        keypoints, silhouette, pose/shape/translation, priors, joint limits),
        seeing that specimen's body parameters together with the shared scene
        camera.  Camera terms are supervised on specimen 0 only, so the camera
        is not penalised ``N`` times (design doc §6/§9).

        Falls back to the single-animal implementation when handed a plain
        single-animal prediction dict, which keeps mixed call sites safe.
        """
        if not isinstance(predicted_params, dict) or "animals" not in predicted_params:
            return super().compute_batch_loss(
                predicted_params,
                target_params_batch,
                auxiliary_data,
                return_components=return_components,
                loss_weights=loss_weights,
            )

        specimen_targets = target_params_batch["specimens"]
        specimen_aux = (auxiliary_data or {}).get("specimens", [None] * self.num_animals)
        animal_mask = predicted_params.get("animal_mask")
        if animal_mask is None:
            animal_mask = target_params_batch.get("animal_mask")
        presence = (
            presence_counts_from_mask(animal_mask)
            if animal_mask is not None
            else [len(specimen_targets[0].get("global_rot", []))] * self.num_animals
        )

        base_weights = loss_weights if loss_weights is not None else {}

        def specimen_loss(index: int, weights: Dict[str, float]):
            specimen_prediction = self.build_specimen_prediction(predicted_params, index)
            return super(MultiAnimalSMILRegressor, self).compute_batch_loss(
                specimen_prediction,
                specimen_targets[index],
                specimen_aux[index],
                return_components=True,
                loss_weights=weights or None,
            )

        return self.loss_aggregator(
            specimen_loss,
            base_weights,
            presence,
            device=self.device,
            return_components=return_components,
        )

    def build_specimen_prediction(self, predicted_params: Dict[str, Any], index: int) -> Dict[str, torch.Tensor]:
        """Assemble the single-animal prediction dict for one specimen.

        Body parameters come from head ``index``; the camera is the shared
        scene camera.  The result is exactly the shape the inherited loss and
        renderer already expect, which is why neither needs to change.
        """
        specimen = dict(predicted_params["animals"][index])
        for key in ("fov", "cam_rot", "cam_trans"):
            if key in predicted_params:
                specimen[key] = predicted_params[key]
        return specimen

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    def predict_from_image(self, image_data: np.ndarray) -> Dict[str, Any]:
        """Predict every specimen from one raw image (inference helper)."""
        self.eval()
        with torch.no_grad():
            image_tensor = self.preprocess_image(image_data).to(self.device)
            return self.forward(image_tensor)

    def get_trainable_parameters(self):
        """Trainable parameters, including the specimen heads and scene camera."""
        return [p for p in self.parameters() if p.requires_grad]

    def describe(self) -> str:
        """One-line summary for training logs."""
        camera = "scene_head" if self.scene_camera_head is not None else "first_specimen"
        return (
            f"{type(self).__name__}(N={self.num_animals}, ids={self.specimen_ids}, "
            f"camera={camera}, {self.specimen_heads.describe()})"
        )


def create_multianimal_regressor(
    device,
    batch_size: int,
    shape_family,
    use_unity_prior: bool,
    multi_animal: MultiAnimalConfig,
    data_batch=None,
    **kwargs,
) -> MultiAnimalSMILRegressor:
    """Factory mirroring ``create_multiview_regressor`` for the single-view model.

    Builds the placeholder batch ``SMALFitter`` needs when the caller has none,
    so training scripts can construct the model before touching the dataset.
    """
    if data_batch is None:
        # Mirror create_multiview_regressor / SMILImageRegressor: the placeholder batch must match the
        # backbone's native resolution, since the renderer is sized from it.
        from smal_fitter.neuralSMIL.backbone_factory import BackboneFactory

        backbone_name = kwargs.get("backbone_name", "resnet152")
        resolution = kwargs.get("input_resolution") or BackboneFactory.get_default_input_resolution(backbone_name)
        data_batch = torch.zeros(batch_size, 3, resolution, resolution, dtype=torch.float32, device=device)

    return MultiAnimalSMILRegressor(
        device=device,
        data_batch=data_batch,
        batch_size=batch_size,
        shape_family=shape_family,
        use_unity_prior=use_unity_prior,
        multi_animal=multi_animal,
        **kwargs,
    )
