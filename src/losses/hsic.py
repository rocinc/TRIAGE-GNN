"""Hilbert-Schmidt Independence Criterion (HSIC) loss.

Symbol mapping (paper → code):
    Z_s  → z_stable   causal/stable representation,      [N, d_s]
    Z_e  → z_env      environment/spurious representation,[N, d_e]
    HSIC → (return)   scalar independence penalty

Design:
    HSIC(Z_s, Z_e) ≈ 0 forces the stable and environment channels to
    carry statistically independent information.  We use the empirical
    biased estimator with RBF kernels and median bandwidth heuristic:

        HSIC(X, Y) = 1/n² · tr(K_X · H · K_Y · H)

    where H = I - 1/n · 11^T is the centering matrix.

    The median heuristic sets σ = median(pairwise distances) / sqrt(2·log(n+1))
    which is robust to scale changes across training.

Reference:
    Gretton et al., "A Kernel Statistical Test of Independence", NeurIPS 2007.
    Li et al., "Deep Domain Generalization via Conditional Invariant Adversarial
    Networks", ECCV 2018 (median heuristic usage).
"""

from __future__ import annotations

import torch


def _rbf_kernel(X: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """Compute RBF kernel matrix K[i,j] = exp(-||x_i - x_j||^2 / (2σ²)).

    Args:
        X:     Feature matrix [N, D].
        sigma: Bandwidth.  If None, use the median heuristic.

    Returns:
        K: Kernel matrix [N, N].
    """
    dist_sq = torch.cdist(X, X).pow(2)  # [N, N]
    if sigma is None:
        # Median heuristic (exclude diagonal zeros).
        n = X.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=X.device)
        median_dist = dist_sq[mask].median().clamp(min=1e-8)
        sigma_sq = median_dist / (2.0 * (n + 1) ** 0.5)
    else:
        sigma_sq = torch.tensor(sigma ** 2, device=X.device)
    return torch.exp(-dist_sq / (2.0 * sigma_sq))


def hsic_loss(
    z_stable: torch.Tensor,
    z_env: torch.Tensor,
    sigma_s: float | None = None,
    sigma_e: float | None = None,
    max_samples: int = 512,
) -> torch.Tensor:
    """Empirical biased HSIC between z_stable and z_env.

    Measures statistical dependence; minimising this loss encourages
    the two representations to be statistically independent.

    Args:
        z_stable:    Stable/causal channel representations.
        z_env:       Environment/spurious channel representations.
        sigma_s:     RBF bandwidth for z_stable (None → median heuristic).
        sigma_e:     RBF bandwidth for z_env (None → median heuristic).
        max_samples: Subsample to this many nodes if N > max_samples
                     (HSIC is O(N²), subsampling controls cost).

    Returns:
        Scalar HSIC estimate (≥ 0).  Zero iff z_stable ⊥ z_env.
    """
    n = z_stable.size(0)

    # Subsample for computational efficiency.
    if n > max_samples:
        idx = torch.randperm(n, device=z_stable.device)[:max_samples]
        z_stable = z_stable[idx]
        z_env    = z_env[idx]
        n = max_samples

    # Centering matrix H = I - 1/n · 11^T.
    H = torch.eye(n, device=z_stable.device) - (1.0 / n)

    # Kernel matrices.
    Ks = _rbf_kernel(z_stable, sigma_s)  # [N, N]
    Ke = _rbf_kernel(z_env,    sigma_e)  # [N, N]

    # HSIC = 1/n² · tr(Ks · H · Ke · H)
    #       = 1/n² · tr((H · Ks · H) · Ke)   (cyclic property of trace)
    HKsH = H @ Ks @ H  # [N, N]
    hsic = (HKsH * Ke).sum() / (n ** 2)

    return hsic
