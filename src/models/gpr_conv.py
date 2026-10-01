"""GPR-GNN propagation layer.

Symbol mapping (paper → code):
    γ_k     → gamma        learnable propagation weights,  [K+1]
    H^(k)   → h_k          k-th hop node representation,  [N, d]
    Ã_sym   → (computed)   symmetrically normalized adjacency with self-loops
    Z       → z            output multi-hop aggregation,   [N, d]

Reference: Chien et al., "Adaptive Universal Generalized PageRank GNN",
           ICLR 2021.  Eq.(8): Z = Σ_{k=0}^{K} γ_k · H^{(k)}
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GPRConv(MessagePassing):
    """Generalized PageRank convolution layer.

    Computes Z = Σ_{k=0}^{K} γ_k · H^{(k)}, where H^{(k)} = Ã_sym · H^{(k-1)}
    and γ_k are learnable scalar weights initialized to α(1-α)^k.

    Args:
        num_hops: Number of propagation steps K.  Output aggregates hops 0..K.
        alpha:    Initialization parameter.  γ_k = α · (1-α)^k at init.
    """

    def __init__(self, num_hops: int, alpha: float = 0.1) -> None:
        super().__init__(aggr="add")
        assert num_hops >= 1, f"num_hops must be >= 1, got {num_hops}"
        assert 0.0 < alpha < 1.0, f"alpha must be in (0, 1), got {alpha}"

        # γ_k = α(1-α)^k, shape [K+1]; index 0 is the 0-hop (identity) weight.
        gamma_init = alpha * (1.0 - alpha) ** torch.arange(num_hops + 1)
        self.gamma = nn.Parameter(gamma_init.float())  # [K+1], learnable

    def forward(
        self,
        x: torch.Tensor,           # [N, d]
        edge_index: torch.Tensor,  # [2, E]
    ) -> torch.Tensor:             # [N, d]
        """Aggregate multi-hop representations with learnable weights.

        Args:
            x:          Node feature matrix before propagation.
            edge_index: Graph connectivity in COO format.

        Returns:
            z: Multi-hop aggregated node representations, same shape as x.
        """
        num_nodes = x.size(0)
        edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        row, col = edge_index_with_loops
        deg = degree(col, num_nodes=num_nodes, dtype=x.dtype)  # [N]
        deg_inv_sqrt = deg.pow(-0.5)                            # [N]
        # Symmetric normalization: norm_ij = 1 / sqrt(d_i * d_j)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]            # [E+N]

        h_k = x                          # H^(0) = X,  [N, d]
        z = self.gamma[0] * h_k          # [N, d]

        for k in range(1, len(self.gamma)):
            h_k = self.propagate(edge_index_with_loops, x=h_k, norm=norm)  # [N, d]
            z = z + self.gamma[k] * h_k

        return z  # [N, d]

    def message(
        self,
        x_j: torch.Tensor,   # [E, d]  source node features
        norm: torch.Tensor,  # [E]     edge normalization coefficients
    ) -> torch.Tensor:       # [E, d]
        return norm.unsqueeze(-1) * x_j
