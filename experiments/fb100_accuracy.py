"""Facebook-100 evaluation of TRIAGE in the non-amplifying branch (Algorithms 2 and 3).

Every campus has |rho_s| <= 1, so TRIAGE applies the degree-normalized high-pass
filter to each graph and trains a linear classifier with cross-entropy.
8-campus leave-one-out, accuracy metric.

Symbol mapping (paper -> code):
    Z_HP^{(s)}    -> z                 high-pass representation of graph s, [N, F]
    W             -> proj, cls         linear classifier W = W_proj W_cls, rank <= H
    L_CE          -> F.cross_entropy
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.additional_datasets import load_fb100_graph, FB100_VALID_NAMES
from src.models.highpass_filter import highpass_filter
from src.utils.seed import set_seed


class HighPassLinear(torch.nn.Module):
    def __init__(self, in_ch: int, hidden: int, n_cls: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(in_ch, hidden)
        self.cls = torch.nn.Linear(hidden, n_cls)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.cls(self.proj(z))  # [N, n_cls]


def accuracy(model: HighPassLinear, z: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    with torch.no_grad():
        pred = model(z[mask]).argmax(1)
    return (pred == y[mask]).float().mean().item()


def run_single(src_graphs, tgt_graph, hidden, lr, wd, epochs, seed, device):
    set_seed(seed)
    src_gs = [g.to(device) for g in src_graphs]
    tgt_g = tgt_graph.to(device)
    src_zs = [highpass_filter(g.x, g.edge_index) for g in src_gs]
    tgt_z = highpass_filter(tgt_g.x, tgt_g.edge_index)

    model = HighPassLinear(tgt_z.size(1), hidden, 2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val_acc = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        for g, z in zip(src_gs, src_zs):
            opt.zero_grad()
            F.cross_entropy(model(z[g.train_mask]), g.y[g.train_mask]).backward()
            opt.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            val_acc = accuracy(model, tgt_z, tgt_g.y, tgt_g.val_mask)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return {'test_acc': accuracy(model, tgt_z, tgt_g.y, tgt_g.test_mask), 'best_val_acc': best_val_acc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', default=None, choices=FB100_VALID_NAMES)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--data_root', default='data/facebook100')
    args = p.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Config: {vars(args)}  torch={torch.__version__}  cuda={torch.version.cuda}", flush=True)

    all_graphs = {name: load_fb100_graph(name, args.data_root, 'cpu') for name in FB100_VALID_NAMES}
    targets = [args.target] if args.target else FB100_VALID_NAMES

    all_accs = {}
    for tgt_name in targets:
        src_names = [n for n in FB100_VALID_NAMES if n != tgt_name]
        print(f"\n{'|'.join(n[:4] for n in src_names)} -> {tgt_name}", flush=True)
        accs = []
        for seed in args.seeds:
            r = run_single([all_graphs[n] for n in src_names], all_graphs[tgt_name],
                           args.hidden, args.lr, args.wd, args.epochs, seed, device)
            accs.append(r['test_acc'])
            print(f"  seed={seed}: acc={r['test_acc']:.4f}  best_val={r['best_val_acc']:.4f}", flush=True)
        all_accs[tgt_name] = (float(np.mean(accs)), float(np.std(accs)))
        print(f"  Summary -> {tgt_name}: {all_accs[tgt_name][0]:.4f} +- {all_accs[tgt_name][1]:.4f}", flush=True)

    if len(targets) > 1:
        avg = np.mean([m for m, _ in all_accs.values()])
        print(f"\nFB-100 (8-campus LOO) avg: {avg:.4f}", flush=True)
        for name, (m, s) in all_accs.items():
            print(f"  {name}: {m:.4f} +- {s:.4f}", flush=True)


if __name__ == '__main__':
    main()
