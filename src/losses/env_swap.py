"""Environment channel swap loss for C3.

Symbol mapping (paper → code):
    Z_s     → z_stable       stable channel representations,      [N, d_s]
    Z_e     → z_env          environment channel representations,  [N, d_e]
    y       → labels         integer class labels,                 [N]
    L_swap  → (return)       cross-entropy after z_env swap

Design:
    For each source domain e, replace its z_env with z_env sampled from
    another source domain e' ≠ e.  The stable head must still classify
    correctly after the swap.

    L_swap = E_{e,e'≠e} [ CE(classifier(z_stable_e + z_env_e'), y_e) ]

    This encourages z_stable to carry all label-predictive information
    independently of the environment channel.

    z_env is detached before swapping to prevent gradients from flowing
    back through the environment path into the shared encoder — otherwise
    z_stable gets contaminated via z_env's gradient.

    The swap is done by randomly permuting domain indices, so each domain
    receives z_env from a different domain.
"""

import torch
import torch.nn.functional as F


def env_swap_loss(
    classifier: torch.nn.Module,
    domain_z_stable: dict[int, torch.Tensor],  # domain_id → [N_e, d_s]
    domain_z_env: dict[int, torch.Tensor],     # domain_id → [N_e, d_e]
    domain_labels: dict[int, torch.Tensor],    # domain_id → [N_e]
) -> torch.Tensor:
    """Cross-entropy loss after swapping z_env across domains.

    Args:
        classifier:       The stable-head linear classifier.
        domain_z_stable:  Per-domain stable representations.
        domain_z_env:     Per-domain environment representations.
        domain_labels:    Per-domain integer class labels.

    Returns:
        Scalar swap loss (mean CE over all domain pairs).

    Raises:
        ValueError: If fewer than 2 domains are provided.
    """
    domain_ids = sorted(domain_z_stable.keys())
    if len(domain_ids) < 2:
        raise ValueError(
            f"env_swap_loss requires >= 2 domains, got {len(domain_ids)}"
        )

    # Build a cyclic permutation of domain IDs so each domain gets z_env
    # from a different domain.  Cyclic shift avoids any domain receiving
    # its own z_env.
    shifted_ids = domain_ids[1:] + domain_ids[:1]  # cyclic shift by 1

    total_loss = torch.tensor(0.0, device=next(iter(domain_z_stable.values())).device)
    num_pairs = 0

    for eid, eid_prime in zip(domain_ids, shifted_ids):
        z_s = domain_z_stable[eid]                          # [N_e, d_s]
        # Detach z_env to prevent gradients from flowing back through the
        # environment path into the shared encoder.
        z_e_prime = domain_z_env[eid_prime].detach()        # [N_e', d_e]
        y = domain_labels[eid]                              # [N_e]

        # Sizes may differ across domains; use the minimum to avoid index errors.
        n = min(z_s.size(0), z_e_prime.size(0))
        z_s = z_s[:n]
        z_e_prime = z_e_prime[:n]
        y = y[:n]

        # Concatenate stable and swapped environment channels.
        z_swapped = torch.cat([z_s, z_e_prime], dim=1)     # [n, d_s + d_e]
        logits = classifier(z_swapped)                      # [n, num_classes]
        total_loss = total_loss + F.cross_entropy(logits, y)
        num_pairs += 1

    return total_loss / num_pairs
