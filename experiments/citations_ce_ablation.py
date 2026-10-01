"""Citations CE baseline with val criterion = Macro F1 (control experiment).

Purpose: isolate whether AUCLoss gain comes from the loss function itself
vs the val criterion change (CE used val=Micro, AUCLoss used val=Macro).

If CE+val=Macro ≈ CE+val=Micro: AUCLoss gain is real (from the loss)
If CE+val=Macro ≈ AUCLoss+val=Macro: gain comes from val criterion, not loss
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


def run_single(src_graphs, tgt_graph, K=50, alpha=0.1, hidden=128,
               lr=1e-3, wd=5e-4, dropout=0.5, epochs=300, seed=42,
               val_criterion='macro', device=torch.device('cpu')):
    set_seed(seed)
    src_gs = [g.to(device) for g in src_graphs]
    tgt_g = tgt_graph.to(device)

    n_cls = int(max(g.y.max() for g in src_gs + [tgt_g]).item()) + 1
    model = SimpleGPR(src_gs[0].x.shape[1], hidden, n_cls, K, alpha, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for g in src_gs:
            opt.zero_grad()
            F.cross_entropy(model(g.x, g.edge_index)[g.train_mask], g.y[g.train_mask]).backward()
            opt.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(tgt_g.x, tgt_g.edge_index)
                if val_criterion == 'macro':
                    val_score = macro_f1(val_logits[tgt_g.val_mask], tgt_g.y[tgt_g.val_mask])
                else:
                    val_score = micro_f1(val_logits[tgt_g.val_mask], tgt_g.y[tgt_g.val_mask])
                if val_score > best_val:
                    best_val = val_score
                    best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(tgt_g.x, tgt_g.edge_index)
        return {
            'test_micro': micro_f1(test_logits[tgt_g.test_mask], tgt_g.y[tgt_g.test_mask]),
            'test_macro': macro_f1(test_logits[tgt_g.test_mask], tgt_g.y[tgt_g.test_mask]),
        }


# CE baseline with val=Micro (from prior run)
CE_MICRO_BASELINE = {
    'ACMv9':      (0.7321, 0.7247),
    'Citationv1': (0.7841, 0.7327),
    'DBLPv7':     (0.7468, 0.7014),
}
# AUCLoss with val=Macro (from prior run)
AUC_MACRO = {
    'ACMv9':      (0.7615, 0.7682),
    'Citationv1': (0.7906, 0.7734),
    'DBLPv7':     (0.7825, 0.7761),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', required=True, choices=VALID_NAMES)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--K', type=int, default=50)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--val_criterion', default='macro', choices=['micro', 'macro'])
    args = p.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    all_graphs = {n: load_citation_graph(n, 'data/citation/raw_download', 'cpu') for n in VALID_NAMES}
    src_names = [n for n in VALID_NAMES if n != args.target]
    src_gs = [all_graphs[n] for n in src_names]
    tgt_g = all_graphs[args.target]

    print(f"Citations CE+val={args.val_criterion}: {'|'.join(src_names)} -> {args.target}", flush=True)

    micros, macros = [], []
    for seed in args.seeds:
        r = run_single(src_gs, tgt_g, K=args.K, alpha=args.alpha,
                       hidden=args.hidden, epochs=args.epochs, seed=seed,
                       val_criterion=args.val_criterion, device=device)
        micros.append(r['test_micro'])
        macros.append(r['test_macro'])
        print(f"  seed={seed}: micro={r['test_micro']:.4f}  macro={r['test_macro']:.4f}", flush=True)

    mm, mma = np.mean(micros), np.mean(macros)
    print(f"\nSummary -> {args.target} CE+val={args.val_criterion}:", flush=True)
    print(f"  Micro={mm:.4f}+-{np.std(micros):.4f}  Macro={mma:.4f}+-{np.std(macros):.4f}", flush=True)

    if args.target in CE_MICRO_BASELINE:
        cm, cma = CE_MICRO_BASELINE[args.target]
        print(f"  vs CE+val=Micro:  ΔMicro={mm-cm:+.4f}  ΔMacro={mma-cma:+.4f}", flush=True)
    if args.target in AUC_MACRO:
        am, ama = AUC_MACRO[args.target]
        print(f"  vs AUCLoss+val=Macro: ΔMicro={mm-am:+.4f}  ΔMacro={mma-ama:+.4f}", flush=True)


if __name__ == '__main__':
    main()
