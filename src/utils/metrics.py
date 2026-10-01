import torch
from sklearn.metrics import f1_score


def micro_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute Micro-F1 for multi-class node classification.

    Args:
        logits: [N, C] raw logits (unnormalized).
        labels: [N]   integer class indices.

    Returns:
        Micro-F1 score as a Python float.
    """
    preds = logits.argmax(dim=1).cpu().numpy()
    y = labels.cpu().numpy()
    return float(f1_score(y, preds, average="micro"))


def macro_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute Macro-F1 for multi-class node classification.

    Args:
        logits: [N, C] raw logits (unnormalized).
        labels: [N]   integer class indices.

    Returns:
        Macro-F1 score as a Python float.
    """
    preds = logits.argmax(dim=1).cpu().numpy()
    y = labels.cpu().numpy()
    return float(f1_score(y, preds, average="macro", zero_division=0))


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy for multi-class node classification.

    Args:
        logits: [N, C] raw logits (unnormalized).
        labels: [N]   integer class indices.

    Returns:
        Accuracy as a Python float in [0, 1].
    """
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())
