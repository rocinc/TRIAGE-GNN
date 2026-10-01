"""WebKB: GPR-GNN K=50 vs MLP (no propagation).

TRIAGE diagnostic prediction:
  |rho_C| < 1 for all 3 campuses → propagation destroys signal → GPR K=50 <= MLP.

Campuses: Cornell, Texas, Wisconsin
Protocol: 10 fixed splits x 5 seeds, average Accuracy + Macro F1
Device:   CPU or GPU (auto-detected)
"""

import sys, json, time, copy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.datasets import WebKB
from src.models.gpr_conv import GPRConv
from src.utils.seed import set_seed

DEVICE    = torch.device('cpu')
DATA_ROOT = '/tmp/webkb'
SEEDS     = [0, 1, 2, 3, 4]
K_GPR     = 50
ALPHA     = 0.1
HIDDEN    = 64
LR        = 1e-2
WD        = 5e-4
DROPOUT   = 0.5
EPOCHS    = 200


class GPRNet(torch.nn.Module):
    def __init__(self, in_ch, hidden, n_cls, K, alpha, dropout, use_gpr):
        super().__init__()
        self.enc = torch.nn.Linear(in_ch, hidden)
        self.use_gpr = use_gpr
        if use_gpr:
            self.gpr = GPRConv(num_hops=K, alpha=alpha)
        self.cls = torch.nn.Linear(hidden, n_cls)
        self.drop = dropout

    def forward(self, x, edge_index):
        h = F.dropout(F.relu(self.enc(x)), self.drop, self.training)
        z = self.gpr(h, edge_index) if self.use_gpr else h
        return self.cls(F.dropout(z, self.drop, self.training))


def run_one(data, split_idx, use_gpr, seed):
    set_seed(seed)
    n_cls = int(data.y.max().item()) + 1
    model = GPRNet(data.num_node_features, HIDDEN, n_cls,
                   K_GPR, ALPHA, DROPOUT, use_gpr).to(DEVICE)
    g = data.to(DEVICE)
    tr = g.train_mask[:, split_idx]
    va = g.val_mask[:, split_idx]
    te = g.test_mask[:, split_idx]

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best_val, best_state = 0.0, None

    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        logits = model(g.x, g.edge_index)
        F.cross_entropy(logits[tr], g.y[tr]).backward()
        opt.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                lv = model(g.x, g.edge_index)[va]
                acc_v = accuracy_score(g.y[va].cpu(), lv.argmax(-1).cpu())
                if acc_v > best_val:
                    best_val = acc_v
                    best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state or model.state_dict())
    model.eval()
    with torch.no_grad():
        lt = model(g.x, g.edge_index)[te]
        y_pred = lt.argmax(-1).cpu().numpy()
        y_true = g.y[te].cpu().numpy()
    return (accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average='macro', zero_division=0))


def eval_campus(name):
    t0 = time.time()
    dataset = WebKB(DATA_ROOT, name=name)
    data = dataset[0]
    n_splits = data.train_mask.shape[1]
    print(f"\n{'='*60}")
    print(f"  {name}  |  nodes={data.num_nodes}  feats={data.num_node_features}"
          f"  classes={int(data.y.max())+1}  splits={n_splits}")
    print(f"{'='*60}")

    gpr_accs, gpr_macros, mlp_accs, mlp_macros = [], [], [], []

    for sp in range(n_splits):
        for sd in SEEDS:
            a, m = run_one(data, sp, use_gpr=True, seed=sd)
            gpr_accs.append(a); gpr_macros.append(m)
            a, m = run_one(data, sp, use_gpr=False, seed=sd)
            mlp_accs.append(a); mlp_macros.append(m)
        print(f"  split {sp+1:2d}/{n_splits} done "
              f"(elapsed {time.time()-t0:.0f}s)", flush=True)

    ga, gm = np.mean(gpr_accs)*100, np.mean(gpr_macros)*100
    ma, mm = np.mean(mlp_accs)*100, np.mean(mlp_macros)*100
    delta = ga - ma
    label = ('CONFIRMED: GPR<=MLP  |rho_C|<1 prediction holds'
             if delta <= 0 else 'WARNING: GPR>MLP  prediction NOT met')

    print(f"\n  GPR K={K_GPR}:  Acc={ga:.2f}+/-{np.std(gpr_accs)*100:.2f}"
          f"  MacroF1={gm:.2f}+/-{np.std(gpr_macros)*100:.2f}")
    print(f"  MLP:       Acc={ma:.2f}+/-{np.std(mlp_accs)*100:.2f}"
          f"  MacroF1={mm:.2f}+/-{np.std(mlp_macros)*100:.2f}")
    print(f"  Delta(GPR-MLP) Acc = {delta:+.2f}pp  [{label}]")

    return dict(campus=name,
                gpr_acc=ga, gpr_acc_std=np.std(gpr_accs)*100,
                gpr_macro=gm, gpr_macro_std=np.std(gpr_macros)*100,
                mlp_acc=ma, mlp_acc_std=np.std(mlp_accs)*100,
                mlp_macro=mm, mlp_macro_std=np.std(mlp_macros)*100,
                delta_acc=delta, confirmed=(delta <= 0))


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"Config: K={K_GPR} alpha={ALPHA} hidden={HIDDEN} "
          f"epochs={EPOCHS} seeds={SEEDS}")
    print("TRIAGE prediction: |rho_C|<1 for all campuses => GPR K=50 <= MLP")

    results = [eval_campus(c) for c in ['Cornell', 'Texas', 'Wisconsin']]

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Campus':<12}{'GPR Acc':>10}{'MLP Acc':>10}{'Delta':>8}{'Confirmed':>12}")
    print("-"*52)
    for r in results:
        print(f"{r['campus']:<12}{r['gpr_acc']:>9.2f}%"
              f"{r['mlp_acc']:>9.2f}%"
              f"{r['delta_acc']:>+7.2f}pp"
              f"{'YES' if r['confirmed'] else 'NO':>12}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'webkb_gpr_vs_mlp_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")
    print("EXIT:0")
