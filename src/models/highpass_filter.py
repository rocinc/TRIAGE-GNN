"""Degree-normalized high-pass filter of the non-amplifying branch (Eq. hp).

Symbol mapping (paper -> code):
    X             -> x         node features,               [N, F]
    D^{-1} A X    -> x_mean    degree-normalized neighbor mean, [N, F]
    Z_HP          -> z         X - D^{-1} A X,              [N, F]
"""

import torch
from torch_geometric.utils import degree


def highpass_filter(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    assert not (src == dst).any()
    deg = degree(dst, x.size(0), dtype=x.dtype)  # [N]
    assert (deg > 0).all()
    x_mean = torch.zeros_like(x).index_add_(0, dst, x[src]) / deg.unsqueeze(1)  # [N, F]
    return x - x_mean  # [N, F]
