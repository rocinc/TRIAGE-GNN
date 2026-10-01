"""Domain adversarial loss with Gradient Reversal Layer (GRL).

Symbol mapping (paper → code):
    Z_e     → z_env          environment channel representations,  [N, d]
    d       → domain_labels  integer domain IDs,                   [N]
    L_adv   → (return)       domain adversarial loss scalar

Core idea (DANN-style):
    A domain classifier is trained to predict domain ID from z_env.
    The Gradient Reversal Layer (GRL) negates the gradient during backprop,
    so the encoder is forced to make z_env domain-specific (the domain
    classifier can predict domain correctly), while z_stable remains
    domain-invariant (no GRL on z_stable path).

Why this drives mask differentiation:
    - Task loss gradient on mask_weights: pushes label-causal channels → mask=1
      (include in z_stable, which feeds the task classifier)
    - GRL + domain loss gradient on mask_weights: pushes domain-specific channels
      → mask=0 (include in z_env, which feeds the domain classifier)
    - These two gradients act in OPPOSITE DIRECTIONS on different channels,
      creating the differential signal needed for mask differentiation.

    This is the key property missing from L_align/L_sep (which had near-zero
    gradients due to small cross-domain mean differences) and L_inv (which
    had uniform gradients because it only acts through z_stable).

GRL schedule (DANN paper recommendation):
    alpha = 2 / (1 + exp(-10 * p)) - 1
    where p = current_epoch / total_epochs (training progress in [0, 1])
    This gradually increases the adversarial pressure from 0 to 1.

Reference:
    Ganin et al., "Domain-Adversarial Training of Neural Networks",
    JMLR 2016.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GradReverseFn(torch.autograd.Function):
    """Custom autograd function that negates gradients during backprop."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Negate gradient; return None for alpha (not a tensor).
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal with scale alpha.

    Forward pass: identity.
    Backward pass: multiply gradient by -alpha.

    Args:
        x:     Input tensor.
        alpha: Reversal scale (>= 0).  alpha=0 means no reversal (identity).

    Returns:
        Tensor identical to x in the forward pass.
    """
    assert alpha >= 0.0, f"alpha must be >= 0, got {alpha}"
    return _GradReverseFn.apply(x, alpha)


def dann_alpha(epoch: int, total_epochs: int) -> float:
    """DANN-style GRL schedule: alpha increases from 0 to 1 during training.

    alpha = 2 / (1 + exp(-10 * p)) - 1,  p = epoch / total_epochs

    Args:
        epoch:        Current epoch (0-indexed).
        total_epochs: Total number of training epochs.

    Returns:
        alpha in [0, 1).
    """
    assert total_epochs > 0, f"total_epochs must be > 0, got {total_epochs}"
    p = epoch / total_epochs
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


# ---------------------------------------------------------------------------
# Domain Classifier
# ---------------------------------------------------------------------------

class DomainClassifier(nn.Module):
    """Small MLP that predicts domain ID from z_env.

    Args:
        in_dim:      Input dimension (= hidden_dim of z_env).
        num_domains: Number of source domains.
        hidden_dim:  Hidden dimension of the MLP.
    """

    def __init__(self, in_dim: int, num_domains: int, hidden_dim: int = 64) -> None:
        super().__init__()
        assert in_dim > 0 and num_domains >= 2 and hidden_dim > 0
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, z_env: torch.Tensor) -> torch.Tensor:
        """Predict domain logits from z_env (without GRL — caller applies GRL).

        Args:
            z_env: [N, in_dim] environment channel representations.

        Returns:
            [N, num_domains] domain logits.
        """
        return self.mlp(z_env)  # [N, num_domains]


# ---------------------------------------------------------------------------
# Domain adversarial loss
# ---------------------------------------------------------------------------

def domain_adversarial_loss(
    domain_clf: DomainClassifier,
    domain_z_env: dict[int, torch.Tensor],  # domain_id → [N_e, d]
    alpha: float = 1.0,
) -> torch.Tensor:
    """Domain adversarial loss with GRL on z_env.

    For each domain, apply GRL to z_env, then predict domain ID.
    The GRL ensures that the encoder gradient is negated, forcing z_env
    to become domain-specific (the domain classifier can predict correctly).

    Args:
        domain_clf:   DomainClassifier instance.
        domain_z_env: Per-domain environment channel tensors.
        alpha:        GRL reversal scale.  Use dann_alpha() for scheduling.

    Returns:
        Scalar adversarial loss (cross-entropy on domain prediction).

    Raises:
        ValueError: If fewer than 2 domains are provided.
    """
    domain_ids = sorted(domain_z_env.keys())
    if len(domain_ids) < 2:
        raise ValueError(
            f"domain_adversarial_loss requires >= 2 domains, got {len(domain_ids)}"
        )

    all_z_env: list[torch.Tensor] = []
    all_domain_labels: list[torch.Tensor] = []

    for eid in domain_ids:
        z_env = domain_z_env[eid]  # [N_e, d]
        # Apply GRL: forward is identity, backward negates gradient.
        z_env_reversed = grad_reverse(z_env, alpha=alpha)  # [N_e, d]
        all_z_env.append(z_env_reversed)
        # Domain label = eid for all nodes in this domain.
        labels = torch.full(
            (z_env.shape[0],), eid, dtype=torch.long, device=z_env.device
        )
        all_domain_labels.append(labels)

    z_env_cat = torch.cat(all_z_env, dim=0)          # [N_total, d]
    domain_labels_cat = torch.cat(all_domain_labels)  # [N_total]

    domain_logits = domain_clf(z_env_cat)  # [N_total, num_domains]
    return F.cross_entropy(domain_logits, domain_labels_cat)
