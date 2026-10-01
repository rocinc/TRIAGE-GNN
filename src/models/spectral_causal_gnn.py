"""SpectralCausalGNN — full model assembling encoder, disentangler, and classifiers.

Symbol mapping (paper → code):
    X       → x              raw node features,                [N, F]
    h       → h              shared GPR-GNN output,            [N, hidden_dim]
    Z_s     → z_stable       causal/invariant channels,        [N, d_stable]
    Z_e     → z_env          environment/spurious channels,    [N, d_env]
    W_cls   → classifier     stable-head classifier (Linear)
    W_env   → env_classifier environment-head classifier (Linear)

Two operating modes (controlled by use_disentangle):

    use_disentangle=False  (E0/E1 ablation):
        h = SpectralEncoder(x, edge_index)   # [N, hidden_dim]
        logits = classifier(h)               # [N, num_classes]
        z_stable = h, z_env = None

    use_disentangle=True  (E2+ with C2):
        h = SpectralEncoder(x, edge_index)   # [N, hidden_dim]
        z_stable, z_env = CausalDisentangler(h)
        logits_stable = classifier(z_stable)
        logits_env    = env_classifier(z_env)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.causal_disentangler import CausalDisentangler
from src.models.spectral_encoder import SpectralEncoder


@dataclass
class ModelOutput:
    """Container for all tensors produced by a forward pass.

    Attributes:
        logits_stable: [N, num_classes]  primary classification logits.
        logits_env:    [N, num_classes]  env-head logits (None if not disentangling).
        z_stable:      [N, d_stable]     stable representation.
        z_env:         [N, d_env]        environment representation (None if not disentangling).
        h:             [N, hidden_dim]   shared encoder output (for prototype computation).
    """

    logits_stable: torch.Tensor
    logits_env: torch.Tensor | None
    z_stable: torch.Tensor
    z_env: torch.Tensor | None
    h: torch.Tensor


class SpectralCausalGNN(nn.Module):
    """GPR-based encoder with optional causal disentanglement.

    Args:
        in_channels:      Input feature dimension F.
        hidden_dim:       GPR encoder output dimension.
        num_classes:      Number of output classes C.
        num_hops:         GPR propagation steps K.
        alpha:            GPR weight initialization parameter.
        dropout:          Dropout probability.
        use_disentangle:  If True, use two-head projection (C2 mode).
                          If False, classify directly from encoder output (E0/E1 mode).
        d_stable:         Stable head output dimension (only used when use_disentangle=True).
        d_env:            Environment head output dimension (only used when use_disentangle=True).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_classes: int,
        num_hops: int = 50,
        alpha: float = 0.1,
        dropout: float = 0.5,
        use_disentangle: bool = False,
        d_stable: int = 128,
        d_env: int = 128,
    ) -> None:
        super().__init__()
        assert in_channels > 0, f"in_channels must be > 0, got {in_channels}"
        assert hidden_dim > 0, f"hidden_dim must be > 0, got {hidden_dim}"
        assert num_classes >= 2, f"num_classes must be >= 2, got {num_classes}"
        assert d_stable > 0, f"d_stable must be > 0, got {d_stable}"
        assert d_env > 0, f"d_env must be > 0, got {d_env}"

        self.use_disentangle = use_disentangle

        self.encoder = SpectralEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_hops=num_hops,
            alpha=alpha,
            dropout=dropout,
            use_gpr=True,
        )

        if use_disentangle:
            self.disentangler = CausalDisentangler(
                hidden_dim=hidden_dim,
                d_stable=d_stable,
                d_env=d_env,
                dropout=dropout,
            )
            self.classifier = nn.Linear(d_stable, num_classes)
            self.env_classifier = nn.Linear(d_env, num_classes)
        else:
            self.disentangler = None
            self.classifier = nn.Linear(hidden_dim, num_classes)
            self.env_classifier = None

    def forward(
        self,
        x: torch.Tensor,           # [N, in_channels]
        edge_index: torch.Tensor,  # [2, E]
    ) -> ModelOutput:
        """Forward pass.

        Args:
            x:          Node feature matrix.
            edge_index: Graph connectivity in COO format.

        Returns:
            ModelOutput with logits_stable, logits_env, z_stable, z_env, h.
        """
        h = self.encoder(x, edge_index)  # [N, hidden_dim]

        if self.use_disentangle:
            z_stable, z_env = self.disentangler(h)  # [N, d_stable], [N, d_env]
            logits_stable = self.classifier(z_stable)      # [N, num_classes]
            logits_env = self.env_classifier(z_env)        # [N, num_classes]
        else:
            z_stable = h                                   # [N, hidden_dim]
            z_env = None
            logits_stable = self.classifier(z_stable)      # [N, num_classes]
            logits_env = None

        return ModelOutput(
            logits_stable=logits_stable,
            logits_env=logits_env,
            z_stable=z_stable,
            z_env=z_env,
            h=h,
        )
