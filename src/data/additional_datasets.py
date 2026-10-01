"""Data loaders for EDG* additional benchmark datasets.

Supported datasets:
    - Twitch-explicit (6 language domains, binary classification, ROC-AUC)
    - Facebook-100 (8 university campus graphs, binary classification, Accuracy)
    - Cora-EERM (10 artificial domain shifts, multi-class, Accuracy)
    - Amazon-Photo-EERM (10 artificial domain shifts, multi-class, Accuracy)

Symbol mapping:
    x           → node feature matrix,        [N, F]
    edge_index  → COO edge list,              [2, E]
    y           → integer class labels,       [N]
    train_mask  → boolean training mask,      [N]
    val_mask    → boolean validation mask,    [N]
    test_mask   → boolean test mask,          [N]

Split strategy:
    Twitch / FB-100: domain-level (entire graph is one domain).
        We use 60/20/20 random split per graph, as in EDG*.
    Cora / Photo EERM: domain-level, same 60/20/20 split.

Note on Cora/Photo EERM format:
    The pkl files contain (x, y) only — no edge_index.
    The graph structure comes from the standard PyG Planetoid/Amazon
    datasets (edge_index is shared across all domains).
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


# ---------------------------------------------------------------------------
# Twitch-explicit
# ---------------------------------------------------------------------------

TWITCH_LANGS = ["DE", "EN", "ES", "FR", "PT", "RU", "TW"]
TWITCH_FEAT_DIM = 3170


def _make_split_masks(n: int, train_frac: float = 0.6, val_frac: float = 0.2,
                      seed: int = 42, device: torch.device = torch.device("cpu")) -> tuple:
    """Random 60/20/20 train/val/test split masks."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    train_mask = torch.zeros(n, dtype=torch.bool, device=device)
    val_mask = torch.zeros(n, dtype=torch.bool, device=device)
    test_mask = torch.zeros(n, dtype=torch.bool, device=device)
    train_mask[torch.tensor(train_idx, device=device)] = True
    val_mask[torch.tensor(val_idx, device=device)] = True
    test_mask[torch.tensor(test_idx, device=device)] = True
    return train_mask, val_mask, test_mask


def load_twitch_graph(
    lang: str,
    data_root: str,
    device: torch.device,
    split_seed: int = 42,
) -> Data:
    """Load one Twitch language domain as a PyG Data object.

    Args:
        lang:       Language code, one of DE/EN/ES/FR/PT/RU.
        data_root:  Path to the eerm/twitch/ directory.
        device:     Target device.
        split_seed: Seed for 60/20/20 random split.

    Returns:
        PyG Data with x [N, 3170], edge_index [2, E], y [N] (0/1),
        train_mask, val_mask, test_mask.
    """
    assert lang in TWITCH_LANGS, f"Unknown Twitch lang: {lang}. Valid: {TWITCH_LANGS}"
    base = Path(data_root) / lang

    # Labels: binary (mature=1 / not mature=0).
    # Load target first to get the definitive node ID set.
    target_df = pd.read_csv(base / f"musae_{lang}_target.csv")
    target_df = target_df.sort_values("new_id").reset_index(drop=True)
    # n_nodes is determined by the maximum node ID in the target file + 1.
    n_nodes = int(target_df["new_id"].max()) + 1
    y = torch.zeros(n_nodes, dtype=torch.long, device=device)
    for _, row in target_df.iterrows():
        y[int(row["new_id"])] = int(row["mature"])

    # Node features: multi-hot over game IDs.
    with open(base / f"musae_{lang}_features.json") as f:
        feat_dict = json.load(f)
    x = torch.zeros(n_nodes, TWITCH_FEAT_DIM, dtype=torch.float32, device=device)
    for node_str, game_ids in feat_dict.items():
        node_id = int(node_str)
        if node_id >= n_nodes:
            continue  # skip nodes not present in target file
        for gid in game_ids:
            if gid < TWITCH_FEAT_DIM:
                x[node_id, gid] = 1.0

    # Edges (undirected).
    edges_df = pd.read_csv(base / f"musae_{lang}_edges.csv")
    src = torch.tensor(edges_df["from"].values, dtype=torch.long, device=device)
    dst = torch.tensor(edges_df["to"].values, dtype=torch.long, device=device)
    edge_index = torch.stack([src, dst], dim=0)  # [2, E]
    edge_index = to_undirected(edge_index, num_nodes=n_nodes)  # [2, 2E]

    train_mask, val_mask, test_mask = _make_split_masks(n_nodes, seed=split_seed, device=device)

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


# ---------------------------------------------------------------------------
# Facebook-100
# ---------------------------------------------------------------------------

# EDG* uses a specific subset; we use all 8 available .mat files.
# Facebook-100 nodes have features encoded in the .mat files.
FB100_VALID_NAMES = [
    "Amherst41", "Brown11", "Caltech36", "Cornell5",
    "Johns Hopkins55", "Penn94", "Texas80", "Yale4",
]


def load_fb100_graph(
    campus: str,
    data_root: str,
    device: torch.device,
    split_seed: int = 42,
    feat_dim: int = 128,
) -> Data:
    """Load one Facebook-100 campus graph.

    Label: col 1 of local_info (student status).
        1 → 1 (student), 2 → 0 (non-student/alumni).
        Nodes with col 1 == 0 (unknown status) are excluded from masks.

    Features: sparse one-hot of cols 0,2,3,4,5,6, then TruncatedSVD to
        feat_dim.  The raw one-hot can be ~65k-dimensional (OOM on GPU).
        SVD reduces it to feat_dim dense dimensions, as is standard for
        large categorical features.

    Args:
        campus:    Campus name, e.g. 'Penn94', 'Texas80'.
        data_root: Path to the facebook100/ directory.
        device:    Target device.
        feat_dim:  Output feature dimension after SVD.
    """
    if campus not in FB100_VALID_NAMES:
        raise ValueError(f"Unknown campus '{campus}'. Valid: {FB100_VALID_NAMES}")

    mat_path = Path(data_root) / f"{campus}.mat"
    mat = scipy.io.loadmat(str(mat_path))
    A = mat["A"]
    local_info = mat["local_info"].astype(np.int64)  # [N, 7]

    n_nodes = local_info.shape[0]

    # Label: col 1.  student=1 → label 1, non-student=2 → label 0.
    # col 1 == 0 means unknown; those nodes are masked out of splits.
    status = local_info[:, 1]
    y = np.where(status == 1, 1, 0).astype(np.int64)  # non-student=0, student=1
    labeled = (status > 0)  # boolean mask of labeled nodes

    # Features: build sparse one-hot from cols 0,2,3,4,5,6, then TruncatedSVD.
    # Cache the SVD result to avoid re-computing on every run.
    cache_path = mat_path.parent / f"{campus}_svd{feat_dim}.npy"
    if cache_path.exists():
        feat_dense = np.load(str(cache_path))
    else:
        feature_cols = [0, 2, 3, 4, 5, 6]
        col_offset = 0
        row_indices = []
        col_indices = []
        for col in feature_cols:
            col_vals = local_info[:, col]
            n_cats = int(col_vals.max()) + 1
            rows = np.arange(n_nodes)
            cols_shifted = col_vals + col_offset
            row_indices.append(rows)
            col_indices.append(cols_shifted)
            col_offset += n_cats

        all_rows = np.concatenate(row_indices)
        all_cols = np.concatenate(col_indices)
        total_feat_dim = col_offset
        feat_sparse = scipy.sparse.csr_matrix(
            (np.ones(len(all_rows), dtype=np.float32), (all_rows, all_cols)),
            shape=(n_nodes, total_feat_dim),
        )
        actual_dim = min(feat_dim, total_feat_dim - 1, n_nodes - 1)
        svd = TruncatedSVD(n_components=actual_dim, random_state=split_seed)
        feat_dense = svd.fit_transform(feat_sparse).astype(np.float32)  # [N, feat_dim]
        np.save(str(cache_path), feat_dense)

    x = torch.tensor(feat_dense, dtype=torch.float32, device=device)  # [N, feat_dim]
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    # Edges from sparse adjacency matrix.
    A_coo = A.tocoo()
    src = torch.tensor(A_coo.row.astype(np.int64), device=device)
    dst = torch.tensor(A_coo.col.astype(np.int64), device=device)
    edge_index = torch.stack([src, dst], dim=0)
    edge_index = to_undirected(edge_index, num_nodes=n_nodes)

    # Split only labeled nodes 60/20/20.
    labeled_idx = np.where(labeled)[0]
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(len(labeled_idx))
    n_tr = int(len(labeled_idx) * 0.6)
    n_va = int(len(labeled_idx) * 0.2)
    tr_idx = labeled_idx[perm[:n_tr]]
    va_idx = labeled_idx[perm[n_tr:n_tr + n_va]]
    te_idx = labeled_idx[perm[n_tr + n_va:]]

    train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    val_mask   = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    test_mask  = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    train_mask[torch.tensor(tr_idx, device=device)] = True
    val_mask[torch.tensor(va_idx, device=device)]   = True
    test_mask[torch.tensor(te_idx, device=device)]  = True

    return Data(
        x=x,
        edge_index=edge_index,
        y=y_t,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


# ---------------------------------------------------------------------------
# Cora-EERM  (requires PyG Planetoid graph + pkl features)
# ---------------------------------------------------------------------------

def load_cora_eerm_graphs(
    data_root_eerm: str,
    data_root_pyg: str,
    device: torch.device,
    split_seed: int = 42,
    num_domains: int = 10,
) -> list[Data]:
    """Load all Cora EERM domain graphs.

    Args:
        data_root_eerm: Path containing gen/ folder with 0-gat.pkl ... 9-gat.pkl.
        data_root_pyg:  Path to PyG Planetoid Cora processed/data.pt.
        device:         Target device.
        num_domains:    Number of domains to load (max 10).

    Returns:
        List of num_domains Data objects, each with perturbed features but
        sharing the same edge_index from the original Cora graph.
    """
    # Load the original Cora graph for edge_index.
    pyg_data_path = Path(data_root_pyg) / "Cora" / "processed" / "data.pt"
    assert pyg_data_path.exists(), (
        f"PyG Cora processed data not found at {pyg_data_path}. "
        "Run: from torch_geometric.datasets import Planetoid; "
        "Planetoid(root=data_root_pyg, name='Cora')"
    )
    cora_base = torch.load(str(pyg_data_path), map_location="cpu", weights_only=False)
    # PyG 2.x stores processed data as (data_dict, slices, cls) tuple.
    if isinstance(cora_base, tuple):
        cora_base = cora_base[0]  # data_dict with 'x', 'edge_index', 'y', masks
    if isinstance(cora_base, dict):
        edge_index = cora_base['edge_index'].to(device)
        n_nodes = int(cora_base['x'].shape[0])
    else:
        edge_index = cora_base.edge_index.to(device)
        n_nodes = int(cora_base.x.size(0))

    train_mask, val_mask, test_mask = _make_split_masks(n_nodes, seed=split_seed, device=device)

    graphs = []
    gen_dir = Path(data_root_eerm) / "gen"
    for i in range(num_domains):
        pkl_path = gen_dir / f"{i}-gat.pkl"
        assert pkl_path.exists(), f"Cora EERM pkl not found: {pkl_path}"
        with open(pkl_path, "rb") as f:
            x_dom, y_dom = pickle.load(f)
        x_dom = x_dom.float().to(device)   # [N, F]
        y_dom = y_dom.long().to(device)    # [N]
        assert x_dom.size(0) == n_nodes, (
            f"Cora domain {i}: x has {x_dom.size(0)} nodes, expected {n_nodes}"
        )
        graphs.append(Data(
            x=x_dom,
            edge_index=edge_index,
            y=y_dom,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            domain_id=i,
        ))

    return graphs


# ---------------------------------------------------------------------------
# Amazon-Photo-EERM
# ---------------------------------------------------------------------------

def load_photo_eerm_graphs(
    data_root_eerm: str,
    data_root_pyg: str,
    device: torch.device,
    split_seed: int = 42,
    num_domains: int = 10,
) -> list[Data]:
    """Load all Amazon-Photo EERM domain graphs.

    Args:
        data_root_eerm: Path containing gen/ folder with 0-gat.pkl ... 9-gat.pkl.
        data_root_pyg:  Path to PyG Amazon Photo processed directory.
        device:         Target device.

    Returns:
        List of num_domains Data objects sharing the Amazon-Photo edge_index.
    """
    pyg_data_path = Path(data_root_pyg) / "Photo" / "processed" / "data.pt"
    assert pyg_data_path.exists(), (
        f"PyG Amazon Photo processed data not found at {pyg_data_path}. "
        "Run: from torch_geometric.datasets import Amazon; "
        "Amazon(root=data_root_pyg, name='Photo')"
    )
    photo_base = torch.load(str(pyg_data_path), map_location="cpu", weights_only=False)
    # PyG 2.x stores processed data as (data_dict, slices, cls) tuple.
    if isinstance(photo_base, tuple):
        photo_base = photo_base[0]
    if isinstance(photo_base, dict):
        edge_index = photo_base['edge_index'].to(device)
        n_nodes = int(photo_base['x'].shape[0])
    else:
        edge_index = photo_base.edge_index.to(device)
        n_nodes = int(photo_base.x.size(0))

    train_mask, val_mask, test_mask = _make_split_masks(n_nodes, seed=split_seed, device=device)

    graphs = []
    gen_dir = Path(data_root_eerm) / "gen"
    for i in range(num_domains):
        pkl_path = gen_dir / f"{i}-gat.pkl"
        assert pkl_path.exists(), f"Photo EERM pkl not found: {pkl_path}"
        with open(pkl_path, "rb") as f:
            x_dom, y_dom = pickle.load(f)
        x_dom = x_dom.float().to(device)
        y_dom = y_dom.long().to(device)
        assert x_dom.size(0) == n_nodes, (
            f"Photo domain {i}: x has {x_dom.size(0)} nodes, expected {n_nodes}"
        )
        graphs.append(Data(
            x=x_dom,
            edge_index=edge_index,
            y=y_dom,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            domain_id=i,
        ))

    return graphs
