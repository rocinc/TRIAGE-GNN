"""Spectral encoder: linear feature projection followed by GPR propagation.

Symbol mapping (paper → code):
    X       → x        raw node features,              [N, F]
    H^(0)   → h        linearly projected features,    [N, d]
    Z       → z        GPR multi-hop aggregation,      [N, d]
    W       → encoder  linear projection weight matrix

Architecture (paper Sec. 3.1, Depth Prescription):
    x  →  Linear(F, d)  →  ReLU  →  GPR(K hops)  →  z

K is set analytically via Proposition 2 (ERM-Optimal Filter Order)
instead of grid search; see Sec. 3.2 of the paper and the proof in Appendix B.1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gpr_conv import GPRConv


class SpectralEncoder(nn.Module):
    """Linear projection + GPR multi-hop propagation.

    Args:
        in_channels: Input feature dimension F.
        hidden_dim:  Hidden / output dimension d.
        num_hops:    GPR propagation steps K.
        alpha:       GPR weight initialization parameter.
        dropout:     Dropout probability applied after the linear projection.
                     Set to 0.0 to disable.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_hops: int,
        alpha: float = 0.1,
        dropout: float = 0.5,
        use_gpr: bool = True,
    ) -> None:
        super().__init__()
        assert in_channels > 0, f"in_channels must be > 0, got {in_channels}"
        assert hidden_dim > 0, f"hidden_dim must be > 0, got {hidden_dim}"
        assert 0.0 <= dropout < 1.0, f"dropout must be in [0, 1), got {dropout}"

        self.use_gpr = use_gpr
        self.encoder = nn.Linear(in_channels, hidden_dim)
        self.gpr_conv = GPRConv(num_hops=num_hops, alpha=alpha) if use_gpr else None
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,           # [N, in_channels]
        edge_index: torch.Tensor,  # [2, E]
    ) -> torch.Tensor:             # [N, hidden_dim]
        """Project features and aggregate over multiple hops.

        Args:
            x:          Node feature matrix.
            edge_index: Graph connectivity in COO format.

        Returns:
            z: Multi-hop aggregated node representations.
        """
        h = F.relu(self.encoder(x))                          # [N, hidden_dim]
        h = F.dropout(h, p=self.dropout, training=self.training)
        if self.use_gpr:
            z = self.gpr_conv(h, edge_index)                 # [N, hidden_dim]
        else:
            # No GPR: use the linearly projected features directly.
            z = h                                            # [N, hidden_dim]
        return z
