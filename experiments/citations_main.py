"""Citations AUCLoss + per-graph encoder output standardization (TTA-EncStd).

After enc(x), standardize output by per-graph (mean, std) before GPR.
This corrects encoder feature distribution shift caused by homophily difference
between source (h~0.76-0.83) and target (Citationv1 h=0.86) domains.
Produces the Full TRIAGE Citations result (Table 1).

Paper reference: Sec. 3.4 (Structural-Shift Protocol) and Appendix prop:enc_shift
for the TTA-EncStd derivation; Sec. 4 Table 1 for the reported numbers.
"""

import sys
import argparse
import torch
import torch.nn.functional as F
import copy
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.citation_dataset import load_citation_graph, VALID_NAMES
from src.models.gpr_conv import GPRConv
from src.utils.seed import set_seed
from src.utils.metrics import micro_f1, macro_f1


class SimpleGPREncStd(torch.nn.Module):
    """GPR with per-graph enc output standardization before propagation."""

    def __init__(self, in_ch, hidden, n_cls, K, alpha, dropout=0.5):
        super().__init__()
        self.enc = torch.nn.Linear(in_ch, hidden)
        self.gpr = GPRConv(num_hops=K, alpha=alpha)
        self.cls = torch.nn.Linear(hidden, n_cls)
        self.dp = dropout

    def forward(self, x, ei):
        h = F.relu(self.enc(x))                           # [N, hidden]
        # Per-graph standardize: correct homophily-induced distribution shift
        h = (h - h.mean(0)) / h.std(0).clamp(min=1e-8)   # [N, hidden]
        h = F.dropout(h, self.dp, self.training)
        return self.cls(F.dropout(self.gpr(h, ei), self.dp, self.training))


def auc_loss(logits, labels):
    n_cls = logits.size(1)
    probs = F.softmax(logits, dim=1)
    total = torch.tensor(0.0, device=logits.device)
    count = 0
    for c in range(n_cls):
        pos = probs[(labels == c), c]
        neg = probs[(labels != c), c]
        if len(pos) == 0 or len(neg) == 0:
            continue
        total = total + (-F.logsigmoid(pos.unsqueeze(1) - neg.unsqueeze(0))).mean()
        count += 1
    return total / max(count, 1)


def run_single(src_graphs, tgt_graph, K=50, alpha=0.1, hidden=128,
               lr=1e-3, wd=5e-4, dropout=0.5, epochs=300, seed=42,
               device=torch.device('cpu')):
    set_seed(seed)
    src_gs = [g.to(device) for g in src_graphs]
    tgt_g = tgt_graph.to(device)

    n_cls = int(max(g.y.max() for g in src_gs + [tgt_g]).item()) + 1
    model = SimpleGPREncStd(src_gs[0].x.shape[1], hidden, n_cls, K, alpha, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val_macro, best_state, best_epoch = 0.0, None, -1
    for epoch in range(epochs):
        model.train()
        for g in src_gs:
            opt.zero_grad()
            auc_loss(model(g.x, g.edge_index)[g.train_mask], g.y[g.train_mask]).backward()
            opt.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_macro = macro_f1(model(tgt_g.x, tgt_g.edge_index)[tgt_g.val_mask],
                                     tgt_g.y[tgt_g.val_mask])
                if val_macro > best_val_macro:
                    best_val_macro = val_macro
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch + 1

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(tgt_g.x, tgt_g.edge_index)
        return {
            'test_micro': micro_f1(logits[tgt_g.test_mask], tgt_g.y[tgt_g.test_mask]),
            'test_macro': macro_f1(logits[tgt_g.test_mask], tgt_g.y[tgt_g.test_mask]),
            'best_val_macro': best_val_macro,
            'best_epoch': best_epoch,
        }


# Baselines for comparison
AUCLOSS_E500 = {'Citationv1': (0.7975, 0.7754), 'ACMv9': (0.7615, 0.7682), 'DBLPv7': (0.7825, 0.7761)}
EDG_STAR =     {'Citationv1': (0.7914, 0.7766), 'ACMv9': (0.7162, 0.7211), 'DBLPv7': (0.7254, 0.6981)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', required=True, choices=VALID_NAMES)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--K', type=int, default=50)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--epochs', type=int, default=300)
    args = p.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    all_graphs = {n: load_citation_graph(n, 'data/citation/raw_download', 'cpu') for n in VALID_NAMES}
    src_names = [n for n in VALID_NAMES if n != args.target]
    src_gs = [all_graphs[n] for n in src_names]
    tgt_g = all_graphs[args.target]

    print(f"Citations AUCLoss+EncStd: {'|'.join(src_names)} -> {args.target}", flush=True)

    micros, macros = [], []
    for seed in args.seeds:
        r = run_single(src_gs, tgt_g, K=args.K, alpha=args.alpha,
                       hidden=args.hidden, epochs=args.epochs, seed=seed, device=device)
        micros.append(r['test_micro'])
        macros.append(r['test_macro'])
        print(f"  seed={seed}: micro={r['test_micro']:.4f}  macro={r['test_macro']:.4f}"
              f"  (best_val={r['best_val_macro']:.4f} @ ep {r['best_epoch']})", flush=True)

    mm, mma = np.mean(micros), np.mean(macros)
    print(f"\nSummary -> {args.target} AUCLoss+EncStd:", flush=True)
    print(f"  Micro={mm:.4f}+-{np.std(micros):.4f}  Macro={mma:.4f}+-{np.std(macros):.4f}", flush=True)

    if args.target in AUCLOSS_E500:
        am, ama = AUCLOSS_E500[args.target]
        print(f"  vs AUCLoss+e500: ΔMicro={mm-am:+.4f}  ΔMacro={mma-ama:+.4f}", flush=True)
    if args.target in EDG_STAR:
        em, ema = EDG_STAR[args.target]
        print(f"  vs EDG*:         ΔMicro={mm-em:+.4f}  ΔMacro={mma-ema:+.4f}", flush=True)


if __name__ == '__main__':
    main()
