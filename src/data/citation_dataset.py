"""Citation network dataset loader for multi-domain generalization.

Loads ACMv9, Citationv1, and DBLPv7 from pre-processed PyG .pt files
downloaded from the ADAlign Google Drive package.

Expected directory layout on the server:
    data/citation/raw_download/
        ACMv9/processed/data.pt
        Citationv1/processed/data.pt
        DBLPv7/processed/data.pt

Each data.pt contains a tuple (Data, None) where Data has:
    x:          [N, 6775]  node feature matrix (sparse TF-IDF, stored dense)
    edge_index: [2, E]     undirected edges
    y:          [N]        integer class labels in {0, 1, 2, 3, 4}
    train_mask: [N]        boolean training mask
    val_mask:   [N]        boolean validation mask
    test_mask:  [N]        boolean test mask
"""

from pathlib import Path

import torch
from torch_geometric.data import Data

VALID_NAMES = ("ACMv9", "Citationv1", "DBLPv7")

# Statistics for quick sanity checks (nodes, features, edges, classes).
_EXPECTED = {
    "ACMv9":      (9360, 6775, 31112, 5),
    "Citationv1": (8935, 6775, 30196, 5),
    "DBLPv7":     (5484, 6775, 16234, 5),
}


def load_citation_graph(name: str, data_root: str, device: torch.device) -> Data:
    """Load a single citation graph by name.

    Args:
        name:      One of 'ACMv9', 'Citationv1', 'DBLPv7'.
        data_root: Absolute path to the citation data root, i.e. the
                   directory that contains ACMv9/, Citationv1/, DBLPv7/.
        device:    Target device for all tensors.

    Returns:
        PyG Data object with x, edge_index, y, train_mask, val_mask,
        test_mask, and a .name attribute set to `name`.

    Raises:
        ValueError: If `name` is not one of the valid dataset names.
        FileNotFoundError: If the expected .pt file does not exist.
        RuntimeError: If the loaded graph statistics deviate from expected.
    """
    if name not in VALID_NAMES:
        raise ValueError(f"Unknown citation graph '{name}'. Valid: {VALID_NAMES}")

    pt_path = Path(data_root) / name / "processed" / "data.pt"
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {pt_path}. "
            "Run the download script first."
        )

    payload = torch.load(pt_path, weights_only=False)
    # The ADAlign package stores (Data, None) tuples.
    assert isinstance(payload, tuple) and len(payload) == 2, (
        f"Unexpected payload format in {pt_path}: {type(payload)}"
    )
    data: Data = payload[0]

    _validate_graph(name, data, pt_path)

    data.name = name
    return data.to(device)


def _validate_graph(name: str, data: Data, pt_path: Path) -> None:
    """Assert that the loaded graph matches expected statistics."""
    exp_nodes, exp_feats, exp_edges, exp_classes = _EXPECTED[name]

    assert data.x.shape == (exp_nodes, exp_feats), (
        f"{name}: expected x shape ({exp_nodes}, {exp_feats}), "
        f"got {tuple(data.x.shape)} in {pt_path}"
    )
    assert data.edge_index.shape == (2, exp_edges), (
        f"{name}: expected edge_index shape (2, {exp_edges}), "
        f"got {tuple(data.edge_index.shape)} in {pt_path}"
    )
    assert data.y.shape == (exp_nodes,), (
        f"{name}: expected y shape ({exp_nodes},), "
        f"got {tuple(data.y.shape)} in {pt_path}"
    )
    num_classes = int(data.y.max().item()) + 1
    assert num_classes == exp_classes, (
        f"{name}: expected {exp_classes} classes, found {num_classes} in {pt_path}"
    )
