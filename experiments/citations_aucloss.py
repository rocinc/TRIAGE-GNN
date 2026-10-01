"""Citations evaluation with AUCLoss (pairwise sigmoid ranking loss).

Replaces CE with AUCLoss to align training objective with ranking-based metrics.
Val criterion: Macro F1 (the metric where we lag EDG* most).

Usage:
    python eval_citations_aucloss.py --target ACMv9
    python eval_citations_aucloss.py --target Citationv1
    python eval_citations_aucloss.py --target DBLPv7
"""

import sys
import argparse
import torch
import torch.nn.functional as F
import copy
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.citation_dataset import load_citation_graph, VALID_NAMES
from src.models.gpr_conv import GPRConv
from src.utils.seed import set_seed
from src.utils.metrics import micro_f1, macro_f1


class SimpleGPR(torch.nn.Module):
    def __init__(self, in_ch, hidden, n_cls, K, alpha, dropout=0.5):
        super().__init__()
        self.enc = torch.nn.Linear(in_ch, hidden)
        self.gpr = GPRConv(num_hops=K, alpha=alpha)
        self.cls = torch.nn.Linear(hidden, n_cls)
        self.dp = dropout

    def forward(self, x, ei):
        h = F.dropout(F.relu(self.enc(x)), self.dp, self.training)
        return self.cls(F.dropout(self.gpr(h, ei), self.dp, self.training))


def auc_loss(logits, labels):
    """Pairwise sigmoid ranking loss for AUC maximization.

    For each (pos, neg) pair: loss += -log(sigmoid(score_pos - score_neg)).
    Uses class 1 score for binary case; for multi-class uses per-class one-vs-rest.
    Since Citations is 5-class, we use softmax probs and a macro one-vs-rest AUC loss.
    """
    n_cls = logits.size(1)
    probs = F.softmax(logits, dim=1)  # [N, C]
    total_loss = torch.tensor(0.0, device=logits.device)
    count = 0
    for c in range(n_cls):
        pos_mask = (labels == c)
        neg_mask = ~pos_mask
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue
        pos_scores = probs[pos_mask, c]    # [P]
        neg_scores = probs[neg_mask, c]    # [Q]
        # pairwise: [P, Q] -> -log(sigmoid(pos - neg))
        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)  # [P, Q]
        total_loss = total_loss + (-F.logsigmoid(diff)).mean()
        count += 1
    return total_loss / max(count, 1)


def run_single(src_graphs, tgt_graph, K=50, alpha=0.1, hidden=128,
               lr=1e-3, wd=5e-4, dropout=0.5, epochs=300, seed=42,
               device=torch.device('cpu')):
    set_seed(seed)
    src_gs = [g.to(device) for g in src_graphs]
    tgt_g = tgt_graph.to(device)

    n_cls = int(max(g.y.max() for g in src_gs + [tgt_g]).item()) + 1
    model = SimpleGPR(src_gs[0].x.shape[1], hidden, n_cls, K, alpha, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val_macro = 0.0
    best_state = None
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        for g in src_gs:
            opt.zero_grad()
            logits = model(g.x, g.edge_index)
            loss = auc_loss(logits[g.train_mask], g.y[g.train_mask])
            loss.backward()
            opt.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(tgt_g.x, tgt_g.edge_index)
                val_macro = macro_f1(val_logits[tgt_g.val_mask], tgt_g.y[tgt_g.val_mask])
                if val_macro > best_val_macro:
                    best_val_macro = val_macro
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch + 1

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(tgt_g.x, tgt_g.edge_index)
        t_micro = micro_f1(test_logits[tgt_g.test_mask], tgt_g.y[tgt_g.test_mask])
        t_macro = macro_f1(test_logits[tgt_g.test_mask], tgt_g.y[tgt_g.test_mask])

    return {'test_micro': t_micro, 'test_macro': t_macro,
            'best_val_macro': best_val_macro, 'best_epoch': best_epoch}


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
    all_graphs = {name: load_citation_graph(name, 'data/citation/raw_download', 'cpu')
                  for name in VALID_NAMES}
    src_names = [n for n in VALID_NAMES if n != args.target]
    src_gs = [all_graphs[n] for n in src_names]
    tgt_g = all_graphs[args.target]

    print(f"Citations AUCLoss: {'|'.join(src_names)} -> {args.target}", flush=True)
    print(f"K={args.K}  alpha={args.alpha}  hidden={args.hidden}  epochs={args.epochs}", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    print(f"Val criterion: Macro F1", flush=True)

    # CE baseline for comparison (same seeds, same config, standard CE)
    print(f"\nCE baseline (from prior run):", flush=True)
    ce_refs = {'ACMv9': (0.7321, 0.7247), 'Citationv1': (0.7841, 0.7327), 'DBLPv7': (0.7468, 0.7014)}
    if args.target in ce_refs:
        m, ma = ce_refs[args.target]
        print(f"  CE Micro={m:.4f}  Macro={ma:.4f}", flush=True)

    micros, macros = [], []
    for seed in args.seeds:
        r = run_single(src_gs, tgt_g, K=args.K, alpha=args.alpha,
                       hidden=args.hidden, epochs=args.epochs, seed=seed, device=device)
        micros.append(r['test_micro'])
        macros.append(r['test_macro'])
        print(f"  seed={seed}: micro={r['test_micro']:.4f}  macro={r['test_macro']:.4f}"
              f"  (best_val_macro={r['best_val_macro']:.4f} @ epoch {r['best_epoch']})", flush=True)

    print(f"\nSummary -> {args.target} AUCLoss:", flush=True)
    print(f"  Micro F1: {np.mean(micros):.4f} +- {np.std(micros):.4f}", flush=True)
    print(f"  Macro F1: {np.mean(macros):.4f} +- {np.std(macros):.4f}", flush=True)

    if args.target in ce_refs:
        dm = np.mean(micros) - ce_refs[args.target][0]
        dma = np.mean(macros) - ce_refs[args.target][1]
        print(f"  Delta vs CE: Micro {dm:+.4f}  Macro {dma:+.4f}", flush=True)


if __name__ == '__main__':
    main()
