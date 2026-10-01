"""Causal disentanglement module: two independent projection heads.

Symbol mapping (paper → code):
    h       → h              shared GPR-GNN output,          [N, hidden_dim]
    Z_s     → z_stable       causal/invariant channels,      [N, d_stable]
    Z_e     → z_env          environment/spurious channels,  [N, d_env]
    W_s     → proj_stable    stable projection head (Linear)
    W_e     → proj_env       environment projection head (Linear)
    q       → gce_q          GCE truncation parameter

Design rationale:
    The previous soft-mask design (z_stable = z * M, z_env = z * (1-M)) failed
    because both heads share the same encoder output z, so any loss produces
    uniform gradients on M — there is no differential signal to drive
    differentiation.

    Two independent linear heads (proj_stable, proj_env) solve this:
    - Task loss (CE on z_stable) only updates proj_stable.
    - GCE loss on z_env makes proj_env actively learn spurious correlations.
    - WCE loss on z_stable (weighted by 1 - env confidence) forces proj_stable
      to learn what proj_env cannot — the causally invariant signal.
    - HSIC penalty discourages z_stable and z_env from sharing information.

    This is the DisC / CIGA-style disentanglement approach.

Paper reference: Sec. 3.3 (Non-Amplifying Filter) and Appendix prop:enc_shift;
GCE/WCE/HSIC losses correspond to L_env, L_wce, L_hsic in the paper's loss
decomposition (Sec. 3.3, Eq. for the full objective).

Reference:
    Chen et al., "Discovering Invariant Rationales for Graph Neural Networks",
    ICLR 2022 (CIGA).
    Yao et al., "Improving Out-of-Distribution Robustness via Selective
    Augmentation", ICML 2022 (DisC).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDisentangler(nn.Module):
    """Two independent linear projection heads for causal disentanglement.

    Args:
        hidden_dim: Input dimension (GPR encoder output).
        d_stable:   Output dimension for the stable (causal) head.
        d_env:      Output dimension for the environment (spurious) head.
        dropout:    Dropout applied to both heads.
    """

    def __init__(
        self,
        hidden_dim: int,
        d_stable: int,
        d_env: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        assert hidden_dim > 0, f"hidden_dim must be > 0, got {hidden_dim}"
        assert d_stable > 0, f"d_stable must be > 0, got {d_stable}"
        assert d_env > 0, f"d_env must be > 0, got {d_env}"
        assert 0.0 <= dropout < 1.0, f"dropout must be in [0, 1), got {dropout}"

        self.proj_stable = nn.Linear(hidden_dim, d_stable)
        self.proj_env = nn.Linear(hidden_dim, d_env)
        self.dropout = dropout

    def forward(
        self,
        h: torch.Tensor,  # [N, hidden_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project shared encoder output into stable and environment channels.

        Args:
            h: Shared GPR encoder output.

        Returns:
            z_stable: [N, d_stable]  causal/invariant representation.
            z_env:    [N, d_env]     environment/spurious representation.
        """
        h = F.dropout(h, p=self.dropout, training=self.training)
        z_stable = self.proj_stable(h)  # [N, d_stable]
        z_env = self.proj_env(h)        # [N, d_env]
        return z_stable, z_env


# ---------------------------------------------------------------------------
# GCE loss — makes proj_env learn spurious correlations
# ---------------------------------------------------------------------------

def gce_loss(
    logits: torch.Tensor,  # [N, C]
    labels: torch.Tensor,  # [N]
    q: float = 0.7,
) -> torch.Tensor:
    """Generalized Cross-Entropy loss (GCE).

    GCE(f, y) = (1 - f_y^q) / q

    With q < 1, GCE down-weights hard examples and up-weights easy ones,
    causing the model to fit spurious correlations first.  Applied to
    proj_env to make it actively learn shortcut features.

    Reference:
        Zhang et al., "Generalized Cross Entropy Loss for Training Deep
        Neural Networks with Noisy Labels", NeurIPS 2018.

    Args:
        logits: Raw logits from the environment head.
        labels: Integer class labels.
        q:      Truncation parameter in (0, 1].  Smaller q → stronger
                bias toward easy examples.

    Returns:
        Scalar GCE loss.
    """
    assert 0.0 < q <= 1.0, f"q must be in (0, 1], got {q}"
    probs = F.softmax(logits, dim=1)                    # [N, C]
    p_y = probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # [N]
    # Clamp to avoid numerical issues with pow.
    p_y = p_y.clamp(min=1e-7)
    loss = (1.0 - p_y.pow(q)) / q                      # [N]
    return loss.mean()


# ---------------------------------------------------------------------------
# WCE loss — forces proj_stable to learn what proj_env cannot
# ---------------------------------------------------------------------------

def wce_loss(
    logits_stable: torch.Tensor,  # [N, C]
    logits_env: torch.Tensor,     # [N, C]
    labels: torch.Tensor,         # [N]
) -> torch.Tensor:
    """Weighted Cross-Entropy loss (WCE) for the stable head.

    Weight for node i = 1 - p_env(y_i | x_i).

    Nodes that proj_env already classifies correctly (high p_env) get low
    weight — they are likely shortcut-driven.  Nodes that proj_env fails on
    get high weight — they require causal features.  This forces proj_stable
    to focus on causally invariant signal.

    Args:
        logits_stable: Raw logits from the stable head.
        logits_env:    Raw logits from the environment head (detached).
        labels:        Integer class labels.

    Returns:
        Scalar weighted cross-entropy loss.
    """
    with torch.no_grad():
        probs_env = F.softmax(logits_env, dim=1)                    # [N, C]
        p_env_y = probs_env.gather(1, labels.unsqueeze(1)).squeeze(1)  # [N]
        # Weight = 1 - env confidence; clamp to avoid zero weights.
        weights = (1.0 - p_env_y).clamp(min=1e-3)                  # [N]

    log_probs = F.log_softmax(logits_stable, dim=1)                 # [N, C]
    nll = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)      # [N]
    return (weights * nll).mean()


# ---------------------------------------------------------------------------
# HSIC penalty — discourages z_stable and z_env from sharing information
# ---------------------------------------------------------------------------

def hsic_loss(
    z_stable: torch.Tensor,  # [N, d_stable]
    z_env: torch.Tensor,     # [N, d_env]
) -> torch.Tensor:
    """Hilbert-Schmidt Independence Criterion (HSIC) penalty.

    Encourages statistical independence between z_stable and z_env by
    minimizing their empirical HSIC with RBF kernels.

    HSIC(X, Y) = (1/(n-1)^2) * tr(K_X H K_Y H)
    where H = I - (1/n) 1 1^T is the centering matrix.

    Args:
        z_stable: Stable channel representations.
        z_env:    Environment channel representations.

    Returns:
        Scalar HSIC value (>= 0).  Minimize to encourage independence.
    """
    n = z_stable.size(0)
    assert n > 1, f"HSIC requires at least 2 samples, got {n}"

    # RBF kernel with median heuristic bandwidth.
    def rbf_kernel(z: torch.Tensor) -> torch.Tensor:
        dist_sq = torch.cdist(z, z).pow(2)  # [N, N]
        # Median heuristic: bandwidth = median of pairwise distances.
        bandwidth = dist_sq.median().clamp(min=1e-8)
        return torch.exp(-dist_sq / bandwidth)  # [N, N]

    K_s = rbf_kernel(z_stable)  # [N, N]
    K_e = rbf_kernel(z_env)     # [N, N]

    # Centering matrix H = I - (1/n) 11^T.
    H = torch.eye(n, device=z_stable.device) - (1.0 / n)  # [N, N]

    # HSIC = (1/(n-1)^2) * tr(K_s H K_e H)
    hsic = torch.trace(K_s @ H @ K_e @ H) / ((n - 1) ** 2)
    return hsic
