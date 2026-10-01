"""Gradient variance consistency loss (VREx / Fishr) for C3.

Symbol mapping (paper → code):
    L_e     → loss_per_domain[e]   per-domain CE loss scalar
    L_var   → (return)             variance of per-domain losses (VREx)
    ∇_θ l   → grads[e]             per-domain gradient vector (Fishr)

Two variants are provided:

    vrex_loss:
        Minimizes the variance of per-domain CE losses.
        L_var = Var_e [ L_e ]  =  E[(L_e - mean_L)^2]
        Simple and effective; does not require gradient computation.

        Reference:
            Krueger et al., "Out-of-Distribution Generalization via Risk
            Extrapolation (REx)", ICML 2021.

    fishr_loss:
        Minimizes the variance of per-domain gradient covariance matrices.
        More principled but more expensive (requires retain_graph=True).

        Reference:
            Rame et al., "FISHR: Invariant Gradient Variances for
            Out-of-Distribution Generalization", ICML 2022.

Note on graph-domain applicability:
    NeurIPS 2024 work showed VREx/IRM can structurally fail on graphs when
    the graph Laplacian couples node representations across domains.
    We include both variants for empirical evaluation; the working-principles
    rule requires diagnosing before committing to either.
"""

import torch
import torch.nn.functional as F


def vrex_loss(
    loss_per_domain: list[torch.Tensor],
) -> torch.Tensor:
    """VREx: variance of per-domain CE losses.

    Eq. (VREx): L_var = (1/|E|) * Σ_e (L_e - mean_L)^2

    Args:
        loss_per_domain: List of scalar CE loss tensors, one per source domain.
                         Each must be a differentiable scalar (not detached).

    Returns:
        Scalar variance loss.

    Raises:
        ValueError: If fewer than 2 domains are provided.
    """
    if len(loss_per_domain) < 2:
        raise ValueError(
            f"vrex_loss requires >= 2 domains, got {len(loss_per_domain)}"
        )
    losses = torch.stack(loss_per_domain)  # [num_domains]
    mean_loss = losses.mean()
    return ((losses - mean_loss) ** 2).mean()


def fishr_loss(
    classifier_params: list[torch.Tensor],
    loss_per_domain: list[torch.Tensor],
) -> torch.Tensor:
    """Fishr: variance of per-domain gradient covariance matrices.

    For each domain e, compute the gradient of L_e w.r.t. classifier
    parameters θ.  Then minimize the variance of these gradient vectors
    across domains.

    L_fishr = Σ_{e1,e2} || Var_{e1}[∇_θ l] - Var_{e2}[∇_θ l] ||^2

    This requires loss_per_domain[e] to be computed with retain_graph=True
    for all but the last domain.

    Args:
        classifier_params: List of classifier parameter tensors (from
                           model.classifier.parameters()).
        loss_per_domain:   List of scalar CE loss tensors, one per domain.
                           Must be computed with create_graph=False but
                           retain_graph=True for all but the last.

    Returns:
        Scalar Fishr loss.

    Raises:
        ValueError: If fewer than 2 domains are provided.
    """
    if len(loss_per_domain) < 2:
        raise ValueError(
            f"fishr_loss requires >= 2 domains, got {len(loss_per_domain)}"
        )

    num_domains = len(loss_per_domain)
    grad_vecs: list[torch.Tensor] = []

    for i, loss_e in enumerate(loss_per_domain):
        retain = (i < num_domains - 1)
        grads = torch.autograd.grad(
            loss_e,
            classifier_params,
            retain_graph=retain,
            create_graph=False,
            allow_unused=True,
        )
        # Flatten and concatenate all parameter gradients into one vector.
        grad_vec = torch.cat(
            [g.flatten() for g in grads if g is not None]
        )  # [P]
        grad_vecs.append(grad_vec)

    grad_mat = torch.stack(grad_vecs)  # [num_domains, P]
    # Variance across domains for each parameter dimension.
    grad_var = grad_mat.var(dim=0)     # [P]
    # Fishr loss = mean squared deviation of per-domain gradient variances.
    # Simplified: minimize the variance of the gradient variance vector.
    return grad_var.mean()
