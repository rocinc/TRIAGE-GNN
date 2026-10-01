"""Supervised contrastive loss for cross-domain class alignment.

Symbol mapping (paper → code):
    Z_s     → z_stable     stable channel representations,  [N, d]
    y       → labels       integer class labels,            [N]
    τ       → temperature  InfoNCE temperature scalar
    L_con   → (return)     supervised contrastive loss

Design:
    For each anchor node i, positive samples are all other nodes with the
    same class label (regardless of domain).  Negative samples are all
    nodes with different class labels.

    L_con = -1/|P(i)| * Σ_{p∈P(i)} log [
        exp(z_i · z_p / τ) /
        Σ_{a≠i} exp(z_i · z_a / τ)
    ]

    This directly encourages same-class nodes from different domains to
    cluster together in the representation space, addressing label shift.

    For large graphs (N >> 1000) the full [N, N] similarity matrix causes
    fp16/fp32 overflow and OOM.  We subsample max_nodes_per_class nodes
    per class before computing the loss, which is standard practice in
    supervised contrastive learning on graphs.

Reference:
    Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
"""

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    z: torch.Tensor,      # [N, d]
    labels: torch.Tensor, # [N]
    temperature: float = 0.5,
    max_nodes_per_class: int = 256,
) -> torch.Tensor:
    """Supervised contrastive loss over a batch of node representations.

    To avoid overflow and OOM on large graphs, at most max_nodes_per_class
    nodes are sampled per class before computing the [N', N'] similarity
    matrix (N' <= num_classes * max_nodes_per_class).

    Args:
        z:                    Node representations (L2-normalized internally).
        labels:               Integer class labels.
        temperature:          InfoNCE temperature τ.
        max_nodes_per_class:  Maximum nodes sampled per class.

    Returns:
        Scalar contrastive loss.

    Raises:
        ValueError: If all nodes belong to the same class (no negatives).
    """
    assert temperature > 0.0, f"temperature must be > 0, got {temperature}"
    n_orig = z.size(0)
    assert n_orig >= 2, f"Need at least 2 nodes, got {n_orig}"

    # Subsample: keep at most max_nodes_per_class per class.
    unique_classes = labels.unique()
    selected_idx: list[torch.Tensor] = []
    for cls in unique_classes:
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        if idx.size(0) > max_nodes_per_class:
            perm = torch.randperm(idx.size(0), device=z.device)[:max_nodes_per_class]
            idx = idx[perm]
        selected_idx.append(idx)
    sel = torch.cat(selected_idx)  # [N']

    z = z[sel]           # [N', d]
    labels = labels[sel] # [N']
    n = z.size(0)

    # L2-normalize so dot product equals cosine similarity.
    z_norm = F.normalize(z, dim=1)  # [N', d]

    # Pairwise cosine similarity matrix, scaled by temperature.
    sim = z_norm @ z_norm.T / temperature  # [N', N']

    # Numerical stability: subtract row-max before exp (equivalent to
    # standard logsumexp trick; does not change the gradient).
    sim = sim - sim.detach().max(dim=1, keepdim=True).values

    # Mask out self-similarity on the diagonal.
    eye_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim_no_self = sim.masked_fill(eye_mask, float("-inf"))

    # Positive mask: same class, different node.
    labels_col = labels.unsqueeze(1)  # [N', 1]
    labels_row = labels.unsqueeze(0)  # [1, N']
    pos_mask = (labels_col == labels_row) & ~eye_mask  # [N', N']

    num_positives = pos_mask.sum(dim=1)  # [N']
    valid = num_positives > 0            # [N']
    if not valid.any():
        raise ValueError(
            "supervised_contrastive_loss: no node has a positive pair. "
            "All nodes belong to singleton classes in this batch."
        )

    # log-sum-exp denominator (all non-self pairs).
    log_denom = torch.logsumexp(sim_no_self, dim=1)  # [N']

    # Log-prob of each pair: sim[i,p] - log_denom[i].
    log_prob = sim_no_self - log_denom.unsqueeze(1)   # [N', N']
    # Replace -inf (self-diagonal and subsampling padding) with 0 before
    # multiplying by pos_mask to avoid 0 * (-inf) = nan.
    log_prob = log_prob.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    # Mean log-prob over positives per anchor, then mean over valid anchors.
    pos_log_prob = (pos_mask.float() * log_prob).sum(dim=1)  # [N']
    per_anchor_loss = -pos_log_prob[valid] / num_positives[valid].float()

    return per_anchor_loss.mean()

