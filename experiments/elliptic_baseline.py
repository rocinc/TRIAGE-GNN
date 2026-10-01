"""Elliptic temporal OOD evaluation with GPR ERM.

Split mirrors EERM (GraphOOD-EERM/temp_elliptic):
  train: timesteps 6-10 (0-indexed: 5-9)
  val:   timesteps 11-15 (0-indexed: 10-14)
  test T1: timesteps 16-26 (0-indexed: 15-25)
  test T2: timesteps 27-37 (0-indexed: 26-36)
  test T3: timesteps 38-48 (0-indexed: 37-47)

EDG* splits the EERM test range into 3 equal thirds; this matches that convention.
Metric: macro F1 (binary illicit vs licit), matching EDG* Table 4.
Labels: class=1 -> illicit=1, class=2 -> licit=0, unknown -> masked.
"""
import argparse
import csv
import copy
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gpr_conv import GPRConv
from src.utils.seed import set_seed

DEFAULT_DATA_ROOT = PROJECT_ROOT / 'data/elliptic/raw/elliptic_bitcoin_dataset'

# ── Data loading ──────────────────────────────────────────────────────────────

def load_elliptic_timesteps(data_dir):
    """Parse raw CSVs into per-timestep (edge_index, x, y, mask) dicts. 0-indexed ts."""
    data_dir = Path(data_dir)
    feat_rows = {}  # old_id -> (ts_0idx, feat_list)
    with open(data_dir / 'elliptic_txs_features.csv') as f:
        for row in csv.reader(f):
            old_id = int(row[0])
            ts = int(row[1]) - 1  # 1-based -> 0-based
            feats = list(map(float, row[2:]))
            feat_rows[old_id] = (ts, feats)

    label_map = {}  # old_id -> label (0/1) or -1 for unknown
    with open(data_dir / 'elliptic_txs_classes.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            old_id = int(row['txId'])
            cls = row['class'].strip()
            label_map[old_id] = -1 if cls == 'unknown' else (1 if cls == '1' else 0)

    # group nodes by timestep
    ts_nodes = {}  # ts -> list of old_id in insertion order
    ts_old2new = {}  # (ts, old_id) -> new local idx
    for old_id, (ts, _) in feat_rows.items():
        ts_nodes.setdefault(ts, []).append(old_id)
    for ts, nodes in ts_nodes.items():
        for new_idx, old_id in enumerate(nodes):
            ts_old2new[(ts, old_id)] = new_idx

    # build per-timestep edge lists
    ts_edges = {ts: [] for ts in ts_nodes}
    with open(data_dir / 'elliptic_txs_edgelist.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = int(row['txId1']); v = int(row['txId2'])
            if u not in feat_rows:
                continue
            ts = feat_rows[u][0]
            if v not in feat_rows or feat_rows[v][0] != ts:
                continue  # skip cross-timestep edges (rare)
            nu = ts_old2new[(ts, u)]; nv = ts_old2new[(ts, v)]
            ts_edges[ts].append((nu, nv))

    # assemble PyG-style dicts
    result = {}
    for ts, nodes in ts_nodes.items():
        n = len(nodes)
        x = torch.tensor([feat_rows[nid][1] for nid in nodes], dtype=torch.float)
        y = torch.tensor([label_map.get(nid, -1) for nid in nodes], dtype=torch.long)
        mask = y >= 0
        if ts_edges[ts]:
            ei = torch.tensor(ts_edges[ts], dtype=torch.long).t().contiguous()
            ei = to_undirected(ei, num_nodes=n)
        else:
            ei = torch.zeros(2, 0, dtype=torch.long)
        result[ts] = {'x': x, 'edge_index': ei, 'y': y, 'mask': mask}
    return result

# ── Model ─────────────────────────────────────────────────────────────────────

class SimpleGPR(torch.nn.Module):
    def __init__(self, in_ch, hidden, K, alpha, dropout=0.5):
        super().__init__()
        self.enc = torch.nn.Linear(in_ch, hidden)
        self.gpr = GPRConv(num_hops=K, alpha=alpha)
        self.cls = torch.nn.Linear(hidden, 1)  # binary
        self.dp = dropout

    def forward(self, x, ei):
        h = F.dropout(F.relu(self.enc(x)), self.dp, self.training)
        h = F.dropout(self.gpr(h, ei), self.dp, self.training)
        return self.cls(h).squeeze(-1)  # [N] logits

# ── Metric ────────────────────────────────────────────────────────────────────

def macro_f1_binary(logits, labels):
    preds = (logits > 0).long()
    tp0 = ((preds==0) & (labels==0)).sum().item()
    fp0 = ((preds==0) & (labels==1)).sum().item()
    fn0 = ((preds==1) & (labels==0)).sum().item()
    tp1 = ((preds==1) & (labels==1)).sum().item()
    fp1 = ((preds==1) & (labels==0)).sum().item()
    fn1 = ((preds==0) & (labels==1)).sum().item()
    p0 = tp0/(tp0+fp0) if tp0+fp0 > 0 else 0
    r0 = tp0/(tp0+fn0) if tp0+fn0 > 0 else 0
    f0 = 2*p0*r0/(p0+r0) if p0+r0 > 0 else 0
    p1 = tp1/(tp1+fp1) if tp1+fp1 > 0 else 0
    r1 = tp1/(tp1+fn1) if tp1+fn1 > 0 else 0
    f1 = 2*p1*r1/(p1+r1) if p1+r1 > 0 else 0
    return (f0 + f1) / 2

# ── Training ──────────────────────────────────────────────────────────────────

def train_and_eval(tr_ts, val_ts, te_ts_list, all_data, K, alpha, hidden,
                   lr, wd, dropout, epochs, seed, device):
    set_seed(seed)
    in_ch = all_data[tr_ts[0]]['x'].shape[1]
    model = SimpleGPR(in_ch, hidden, K, alpha, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    tr_graphs = [{k: v.to(device) for k, v in all_data[t].items()} for t in tr_ts]
    val_g = {k: v.to(device) for k, v in all_data[val_ts[0]].items()}

    best_val, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        for g in tr_graphs:
            opt.zero_grad()
            logits = model(g['x'], g['edge_index'])
            m = g['mask']
            F.binary_cross_entropy_with_logits(logits[m], g['y'][m].float()).backward()
            opt.step()
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(val_g['x'], val_g['edge_index'])
                m = val_g['mask']
                vf = macro_f1_binary(logits[m], val_g['y'][m])
                if vf > best_val:
                    best_val = vf
                    best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    te_scores = []
    with torch.no_grad():
        for te_ts in te_ts_list:
            scores = []
            for t in te_ts:
                g = {k: v.to(device) for k, v in all_data[t].items()}
                logits = model(g['x'], g['edge_index'])
                m = g['mask']
                if m.sum() == 0:
                    continue
                scores.append(macro_f1_binary(logits[m], g['y'][m]))
            te_scores.append(float(np.mean(scores)) if scores else 0.0)
    return te_scores

# ── Split definition (EERM base, EDG* T1/T2/T3) ───────────────────────────────
# EERM: train=t6-10(0idx:5-9), val=t11-15(0idx:10-14), test=t16-48(0idx:15-47)
# EDG* splits test into 3 equal thirds: T1=15-25, T2=26-36, T3=37-47
TR_TS  = list(range(5, 10))   # 0-indexed 5..9  = 1-indexed 6..10
VAL_TS = list(range(10, 15))  # 0-indexed 10..14 = 1-indexed 11..15
TE_T1  = list(range(15, 26))  # 0-indexed 15..25 = 1-indexed 16..26
TE_T2  = list(range(26, 37))  # 0-indexed 26..36 = 1-indexed 27..37
TE_T3  = list(range(37, 48))  # 0-indexed 37..47 = 1-indexed 38..48

EDG_STAR = {'T1': 0.707, 'T2': 0.689, 'T3': 0.682, 'Avg': 0.697}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=[0,1,2,3,4])
    p.add_argument('--K', type=int, default=50)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--data_root', default=str(DEFAULT_DATA_ROOT))
    args = p.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Elliptic data...", flush=True)
    all_data = load_elliptic_timesteps(args.data_root)
    print(f"Loaded {len(all_data)} timesteps. train={len(TR_TS)}, val={len(VAL_TS)}, "
          f"T1={len(TE_T1)}, T2={len(TE_T2)}, T3={len(TE_T3)}", flush=True)

    t1s, t2s, t3s = [], [], []
    for seed in args.seeds:
        scores = train_and_eval(
            TR_TS, VAL_TS, [TE_T1, TE_T2, TE_T3], all_data,
            K=args.K, alpha=args.alpha, hidden=args.hidden,
            lr=1e-3, wd=5e-4, dropout=0.5, epochs=args.epochs,
            seed=seed, device=device)
        t1s.append(scores[0]); t2s.append(scores[1]); t3s.append(scores[2])
        print(f"  seed={seed}: T1={scores[0]:.4f}  T2={scores[1]:.4f}  T3={scores[2]:.4f}", flush=True)

    m1,m2,m3 = np.mean(t1s),np.mean(t2s),np.mean(t3s)
    avg = np.mean([m1,m2,m3])
    print(f"\nSummary GPR K={args.K}:", flush=True)
    print(f"  T1: {m1:.4f}±{np.std(t1s):.4f}  vs EDG* {EDG_STAR['T1']:.3f}  Δ={m1-EDG_STAR['T1']:+.4f}", flush=True)
    print(f"  T2: {m2:.4f}±{np.std(t2s):.4f}  vs EDG* {EDG_STAR['T2']:.3f}  Δ={m2-EDG_STAR['T2']:+.4f}", flush=True)
    print(f"  T3: {m3:.4f}±{np.std(t3s):.4f}  vs EDG* {EDG_STAR['T3']:.3f}  Δ={m3-EDG_STAR['T3']:+.4f}", flush=True)
    print(f"  Avg: {avg:.4f}  vs EDG* {EDG_STAR['Avg']:.3f}  Δ={avg-EDG_STAR['Avg']:+.4f}", flush=True)
    print(f"EXIT:0", flush=True)

if __name__ == '__main__':
    main()
