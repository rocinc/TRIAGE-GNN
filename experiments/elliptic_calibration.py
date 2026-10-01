"""Elliptic with test-time prior calibration.

After training (BCE, same as baseline), at inference time:
1. Run model on test graph, get soft probabilities p_i = sigmoid(logit_i)
2. Estimate P(illicit|test) = mean(p_i)  [unsupervised, no test labels needed]
3. Compute Bayes-optimal threshold shift:
   threshold = log(p_test / (1-p_test)) - log(p_train / (1-p_train))
   where p_train is the illicit rate in training data
4. Apply shifted threshold for final prediction

This is the same logic as TTA-EncStd in Citations:
post-hoc calibration, zero retraining, theoretically grounded.
"""
import argparse
import copy
import csv
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from sklearn.metrics import f1_score

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gpr_conv import GPRConv
from src.utils.seed import set_seed

DEFAULT_DATA_ROOT = PROJECT_ROOT / 'data/elliptic/raw/elliptic_bitcoin_dataset'

def load_elliptic_timesteps(data_dir):
    data_dir = Path(data_dir)
    feat_rows = {}
    with open(data_dir / 'elliptic_txs_features.csv') as f:
        for row in csv.reader(f):
            old_id = int(row[0]); ts = int(row[1]) - 1
            feat_rows[old_id] = (ts, list(map(float, row[2:])))
    label_map = {}
    with open(data_dir / 'elliptic_txs_classes.csv') as f:
        for row in csv.DictReader(f):
            oid = int(row['txId']); c = row['class'].strip()
            label_map[oid] = -1 if c == 'unknown' else (1 if c == '1' else 0)
    ts_nodes = {}; ts_old2new = {}
    for old_id, (ts, _) in feat_rows.items():
        ts_nodes.setdefault(ts, []).append(old_id)
    for ts, nodes in ts_nodes.items():
        for ni, oid in enumerate(nodes): ts_old2new[(ts, oid)] = ni
    ts_edges = {ts: [] for ts in ts_nodes}
    with open(data_dir / 'elliptic_txs_edgelist.csv') as f:
        for row in csv.DictReader(f):
            u = int(row['txId1']); v = int(row['txId2'])
            if u not in feat_rows: continue
            ts = feat_rows[u][0]
            if v not in feat_rows or feat_rows[v][0] != ts: continue
            ts_edges[ts].append((ts_old2new[(ts,u)], ts_old2new[(ts,v)]))
    result = {}
    for ts, nodes in ts_nodes.items():
        n = len(nodes)
        x = torch.tensor([feat_rows[nid][1] for nid in nodes], dtype=torch.float)
        y = torch.tensor([label_map.get(nid,-1) for nid in nodes], dtype=torch.long)
        mask = y >= 0
        ei = (torch.tensor(ts_edges[ts], dtype=torch.long).t().contiguous()
              if ts_edges[ts] else torch.zeros(2,0,dtype=torch.long))
        if ei.numel() > 0: ei = to_undirected(ei, num_nodes=n)
        result[ts] = {'x': x, 'edge_index': ei, 'y': y, 'mask': mask}
    return result


class SimpleGPR(torch.nn.Module):
    def __init__(self, in_ch, hidden, K, alpha, dropout=0.5):
        super().__init__()
        self.enc = torch.nn.Linear(in_ch, hidden)
        self.gpr = GPRConv(num_hops=K, alpha=alpha)
        self.cls = torch.nn.Linear(hidden, 1)
        self.dp = dropout
    def forward(self, x, ei):
        h = F.dropout(F.relu(self.enc(x)), self.dp, self.training)
        return self.cls(F.dropout(self.gpr(h, ei), self.dp, self.training)).squeeze(-1)


TR_TS  = list(range(5, 10))
VAL_TS = list(range(10, 15))
TE_T1  = list(range(15, 26))
TE_T2  = list(range(26, 37))
TE_T3  = list(range(37, 48))

# illicit rate in training timesteps (computed from data)
# t6-t10 (0idx 5-9): approx 10-15% illicit on average
TRAIN_ILLICIT_RATE = 0.12  # estimated from training split


def prior_shift_threshold(logits_np, train_prior, eps=1e-6):
    """Estimate test prior from soft probs, compute Bayes threshold shift."""
    probs = 1.0 / (1.0 + np.exp(-logits_np))  # sigmoid
    test_prior_est = float(np.clip(probs.mean(), eps, 1 - eps))
    # threshold shift: move decision boundary to account for prior change
    # log-odds shift: log(p_test/(1-p_test)) - log(p_train/(1-p_train))
    shift = (np.log(test_prior_est) - np.log(1 - test_prior_est)
             - np.log(train_prior) + np.log(1 - train_prior))
    return shift, test_prior_est


def eval_with_threshold(logits_np, labels_np, threshold):
    preds = (logits_np > threshold).astype(int)
    return f1_score(labels_np, preds, average='macro', zero_division=0)


def run(all_data, K, alpha, hidden, lr, wd, dropout, epochs, seed, device):
    set_seed(seed)
    in_ch = all_data[TR_TS[0]]['x'].shape[1]
    model = SimpleGPR(in_ch, hidden, K, alpha, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    tr_gs = [{k: v.to(device) for k,v in all_data[t].items()} for t in TR_TS]
    val_gs = [{k: v.to(device) for k,v in all_data[t].items()} for t in VAL_TS]

    best_val, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        for g in tr_gs:
            opt.zero_grad()
            logits = model(g['x'], g['edge_index'])
            m = g['mask']
            F.binary_cross_entropy_with_logits(logits[m], g['y'][m].float()).backward()
            opt.step()
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                vf1s = []
                for g in val_gs:
                    m = g['mask']
                    if m.sum() == 0: continue
                    logits = model(g['x'], g['edge_index'])
                    preds = (logits[m] > 0).long()
                    vf1s.append(f1_score(g['y'][m].cpu().numpy(), preds.cpu().numpy(),
                                         average='macro', zero_division=0))
                vf = float(np.mean(vf1s)) if vf1s else 0.0
                if vf > best_val: best_val = vf; best_state = copy.deepcopy(model.state_dict())

    # compute actual train illicit rate from training data
    train_labels = []
    for g in tr_gs:
        m = g['mask']
        train_labels.extend(g['y'][m].cpu().numpy().tolist())
    actual_train_prior = sum(train_labels) / len(train_labels) if train_labels else TRAIN_ILLICIT_RATE

    model.load_state_dict(best_state)
    model.eval()

    results = []
    with torch.no_grad():
        for te_ts in [TE_T1, TE_T2, TE_T3]:
            logits_list, labels_list = [], []
            for t in te_ts:
                g = {k: v.to(device) for k,v in all_data[t].items()}
                m = g['mask']
                if m.sum() == 0: continue
                logits = model(g['x'], g['edge_index'])
                logits_list.append(logits[m].cpu().numpy())
                labels_list.append(g['y'][m].cpu().numpy())

            all_logits = np.concatenate(logits_list)
            all_labels = np.concatenate(labels_list)

            shift, test_prior_est = prior_shift_threshold(all_logits, actual_train_prior)
            f1_default  = eval_with_threshold(all_logits, all_labels, 0.0)
            f1_calib    = eval_with_threshold(all_logits, all_labels, shift)  # prior calibration
            results.append({
                'f1_default':  f1_default,
                'f1_calib':    f1_calib,
                'shift':       shift,
                'test_prior':  test_prior_est,
                'train_prior': actual_train_prior,
            })
    return results


OUR_BCE = {'T1': 0.8314, 'T2': 0.6896, 'T3': 0.4540}
EDG_BCE = {'T1': 0.707,  'T2': 0.689,  'T3': 0.682, 'Avg': 0.697}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=[0,1,2,3,4])
    p.add_argument('--K', type=int, default=50)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--data_root', default=str(DEFAULT_DATA_ROOT))
    args = p.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Loading Elliptic...", flush=True)
    all_data = load_elliptic_timesteps(args.data_root)

    t1_def, t2_def, t3_def = [], [], []
    t1_cal, t2_cal, t3_cal = [], [], []

    for seed in args.seeds:
        res = run(all_data, K=args.K, alpha=0.1, hidden=128,
                  lr=1e-3, wd=5e-4, dropout=0.5, epochs=args.epochs,
                  seed=seed, device=device)
        t1_def.append(res[0]['f1_default']); t1_cal.append(res[0]['f1_calib'])
        t2_def.append(res[1]['f1_default']); t2_cal.append(res[1]['f1_calib'])
        t3_def.append(res[2]['f1_default']); t3_cal.append(res[2]['f1_calib'])
        print(f"  seed={seed}: "
              f"T1 def={res[0]['f1_default']:.4f} cal={res[0]['f1_calib']:.4f}(shift={res[0]['shift']:+.3f},p_test={res[0]['test_prior']:.3f}) | "
              f"T2 def={res[1]['f1_default']:.4f} cal={res[1]['f1_calib']:.4f}(shift={res[1]['shift']:+.3f},p_test={res[1]['test_prior']:.3f}) | "
              f"T3 def={res[2]['f1_default']:.4f} cal={res[2]['f1_calib']:.4f}(shift={res[2]['shift']:+.3f},p_test={res[2]['test_prior']:.3f})", flush=True)

    print(f"\n=== Summary (default | prior-calib) ===", flush=True)
    for name, defs, cals, bce_ours, edg in [
        ('T1', t1_def, t1_cal, OUR_BCE['T1'], EDG_BCE['T1']),
        ('T2', t2_def, t2_cal, OUR_BCE['T2'], EDG_BCE['T2']),
        ('T3', t3_def, t3_cal, OUR_BCE['T3'], EDG_BCE['T3']),
    ]:
        print(f"  {name}: default={np.mean(defs):.4f}±{np.std(defs):.4f}  "
              f"prior-calib={np.mean(cals):.4f}±{np.std(cals):.4f}  "
              f"vs BCE_ours={bce_ours:.4f}  vs EDG*={edg:.3f}  "
              f"Δ_vs_edg={np.mean(cals)-edg:+.4f}", flush=True)
    avg_cal = np.mean([np.mean(t1_cal), np.mean(t2_cal), np.mean(t3_cal)])
    avg_def = np.mean([np.mean(t1_def), np.mean(t2_def), np.mean(t3_def)])
    print(f"  Avg: default={avg_def:.4f}  prior-calib={avg_cal:.4f}  EDG*={EDG_BCE['Avg']:.3f}  Δ={avg_cal-EDG_BCE['Avg']:+.4f}", flush=True)
    print("EXIT:0", flush=True)


if __name__ == '__main__':
    main()
