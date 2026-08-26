"""
Parameter heads that satisfy a single, explicit protocol.

The multi-animal stack needs to hold ``N`` interchangeable parameter heads, so
every head must present the same call signature::

    head(global_features, spatial_features) -> Dict[str, Tensor]

``SMILTransformerDecoderHead`` already satisfies this exactly, so it is used
unwrapped.  The MLP head, however, currently lives *inline* on
``SMILImageRegressor`` as loose ``fc1 … regressor`` attributes and cannot be
replicated or swapped.  :class:`MLPParameterHead` is the same network extracted
into a real ``nn.Module`` so it can be.

Nothing here imports ``config`` or pytorch3d at module scope, which keeps the
head abstraction unit-testable on a machine with plain PyTorch.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parameter_layout import ParameterLayout, parse_flat_parameter_vector


class MLPParameterHead(nn.Module):
    """The single-animal MLP regression head, extracted as a standalone module.

    Layer sizes, normalisation, dropout rate and initialisation are kept
    identical to ``SMILImageRegressor._create_mlp_regression_head`` /
    ``_initialize_mlp_parameters`` so that a checkpoint trained with the inline
    head can be loaded into this module without any numerical change — see
    :meth:`load_inline_head_state`.

    Args:
        feature_dim: Width of the backbone's pooled feature vector.
        hidden_dim: Width of the first hidden layer (halved, then quartered).
        layout: Output-vector layout; also fixes the head's output width.
        dropout: Dropout probability (0.3 in the original head).
    """

    #: Parameter names of the inline head, in the order they appear on the
    #: regressor.  Used to translate legacy checkpoints.
    INLINE_SUBMODULES = ("fc1", "ln1", "fc2", "ln2", "fc3", "ln3", "regressor")

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        layout: ParameterLayout,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.layout = layout

        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.ln3 = nn.LayerNorm(hidden_dim // 4)
        self.regressor = nn.Linear(hidden_dim // 4, layout.total_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """He init for the ReLU stack, Xavier for the linear output layer."""
        for linear in (self.fc1, self.fc2, self.fc3):
            nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
            nn.init.constant_(linear.bias, 0)
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.constant_(self.regressor.bias, 0)

    def forward(
        self,
        global_features: torch.Tensor,
        spatial_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Regress SMIL parameters from a pooled feature vector.

        ``spatial_features`` is accepted (and ignored) so the MLP head is
        interchangeable with the transformer decoder head.
        """
        del spatial_features  # MLP head has no cross-attention

        x = F.relu(self.ln1(self.fc1(global_features)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout(x)
        output = self.regressor(x)
        return parse_flat_parameter_vector(output, self.layout, include_camera=True)

    def load_inline_head_state(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True) -> None:
        """Load weights saved from the *inline* head of ``SMILImageRegressor``.

        A single-animal checkpoint stores the head as top-level regressor keys
        (``fc1.weight``, ``ln1.bias``, …, ``regressor.weight``).  This lifts
        exactly those keys into this module, ignoring backbone and SMAL entries.

        Raises:
            KeyError: if ``strict`` and any expected parameter is missing.
        """
        own: Dict[str, torch.Tensor] = {}
        for name in self.INLINE_SUBMODULES:
            for suffix in ("weight", "bias"):
                key = f"{name}.{suffix}"
                if key in state_dict:
                    own[key] = state_dict[key]
                elif strict:
                    raise KeyError(f"inline MLP head state is missing '{key}'")
        missing, unexpected = self.load_state_dict(own, strict=False)
        if strict and missing:
            raise KeyError(f"MLP head parameters not provided by the checkpoint: {sorted(missing)}")
        del unexpected  # by construction we only pass keys this module owns


def make_head_factory(regressor: Any):
    """Return a zero-argument callable that builds one fresh parameter head.

    The factory delegates to the regressor's *own* construction routines, so
    head hyper-parameters (transformer depth/heads/IEF iterations, hidden width,
    rotation representation, scale/trans mode, mesh scaling) stay defined in
    exactly one place and a multi-animal head can never drift from the
    single-animal one it is meant to replicate.

    Args:
        regressor: A constructed ``SMILImageRegressor`` (or subclass).

    Returns:
        ``Callable[[], nn.Module]`` producing an independently initialised head.
    """
    head_type = getattr(regressor, "head_type", "transformer_decoder")

    if head_type == "transformer_decoder":

        def _build_transformer_head() -> nn.Module:
            # `_create_transformer_decoder_head` assigns to
            # `self.transformer_head`; capture and detach it so the regressor
            # never keeps a stray extra head registered.
            previous = getattr(regressor, "transformer_head", None)
            regressor._create_transformer_decoder_head()
            head = regressor.transformer_head
            if previous is not None:
                regressor.transformer_head = previous
            else:
                delattr(regressor, "transformer_head")
            return head

        return _build_transformer_head

    if head_type == "mlp":

        def _build_mlp_head() -> nn.Module:
            layout = ParameterLayout.from_regressor(regressor)
            return MLPParameterHead(
                feature_dim=regressor.feature_dim,
                hidden_dim=regressor.hidden_dim,
                layout=layout,
            ).to(regressor.device)

        return _build_mlp_head

    raise ValueError(f"Unsupported head_type '{head_type}' (expected 'mlp' or 'transformer_decoder')")
