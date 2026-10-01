"""Per-domain K ablation for Twitch: K=10 ERM / K=50 ERM / K=50 AUCLoss.

Isolates depth vs AUCLoss contribution per domain.
Expected pattern (theoretically consistent with Corollary 3):
  - DE (super-KS target): K=10->K=50 ERM shows gain (depth needed)
  - sub-KS targets (EN/ES/FR/PT/RU): K=10->K=50 ERM flat; AUCLoss drives gain

Hyperparameters of the per-target depth ablation (lighter than the
main-table configuration hidden=128, lr=1e-3, epochs=300):
  hidden=64, lr=0.01, wd=5e-4, epochs=200, alpha=0.1

Usage:
    python experiments/twitch_depth_ablation.py --K 10  --loss ce  --out results_K10_ce.json
    python experiments/twitch_depth_ablation.py --K 50  --loss ce  --out results_K50_ce.json
    python experiments/twitch_depth_ablation.py --K 50  --loss auc --out results_K50_auc.json
"""

import sys, argparse, json, copy
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.additional_datasets import load_twitch_graph
from src.models.gpr_conv import GPRConv
from src.utils.seed import set_seed

sys.stdout.reconfigure(line_buffering=True)

LANGS = ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']
DATA_ROOT = 'data/eerm/twitch'


class SimpleGPR(torch.nn.Module):
    def __init__(self, in_ch, hidden, n_cls, K, alpha, dropout=0.5):
        super().__init__()
        self.enc = torch.nn.Linear(in_ch, hidden)
        self.gpr = GPRConv(num_hops=K, alpha=alpha)
        self.cls = torch.nn.Linear(hidden, n_cls)
        self.dp  = dropout

    def forward(self, x, ei):
        h = F.dropout(F.relu(self.enc(x)), self.dp, self.training)
        return self.cls(F.dropout(self.gpr(h, ei), self.dp, self.training))


def auc_loss(logits, labels):
    probs = F.softmax(logits, dim=1)[:, 1]
    pos = (labels == 1)
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return F.cross_entropy(logits, labels)
    diff = probs[pos].unsqueeze(1) - probs[neg].unsqueeze(0)
    return (-F.logsigmoid(diff)).mean()


def run_one(src_graphs, tgt_graph, K, loss_fn, hidden, lr, wd, epochs,
            alpha, dropout, seed, device):
    set_seed(seed)
    src_gs = [g.to(device) for g in src_graphs]
    tgt_g  = tgt_graph.to(device)

    in_ch = src_gs[0].x.shape[1]
    model = SimpleGPR(in_ch, hidden, 2, K, alpha, dropout).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        for g in src_gs:
            opt.zero_grad()
            out = model(g.x, g.edge_index)[g.train_mask]
            lbl = g.y[g.train_mask]
            loss = auc_loss(out, lbl) if loss_fn == 'auc' else F.cross_entropy(out, lbl)
            loss.backward()
            opt.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(tgt_g.x, tgt_g.edge_index)
                probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                y_val  = tgt_g.y[tgt_g.val_mask].cpu().numpy()
                try:
                    val_auc = roc_auc_score(y_val, probs[tgt_g.val_mask.cpu()])
                    if val_auc > best_val:
                        best_val   = val_auc
                        best_state = copy.deepcopy(model.state_dict())
                except ValueError:
                    pass

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(tgt_g.x, tgt_g.edge_index)
        probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_test = tgt_g.y[tgt_g.test_mask].cpu().numpy()
        test_auc = roc_auc_score(y_test, probs[tgt_g.test_mask.cpu()])
    return test_auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--K',       type=int,   default=50)
    p.add_argument('--loss',    default='ce', choices=['ce','auc'])
    p.add_argument('--hidden',  type=int,   default=64)
    p.add_argument('--lr',      type=float, default=0.01)
    p.add_argument('--wd',      type=float, default=5e-4)
    p.add_argument('--epochs',  type=int,   default=200)
    p.add_argument('--alpha',   type=float, default=0.1)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--seeds',   nargs='+',  type=int, default=[0,1,2,3,4])
    p.add_argument('--target',  default=None, help='run single target only')
    p.add_argument('--out',     default=None)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  K={args.K}  loss={args.loss}  hidden={args.hidden}')

    all_graphs = {l: load_twitch_graph(l, DATA_ROOT, 'cpu') for l in LANGS}
    targets = [args.target] if args.target else LANGS
    results = {}

    for tgt in targets:
        src_langs = [l for l in LANGS if l != tgt]
        src_gs = [all_graphs[l] for l in src_langs]
        tgt_g  = all_graphs[tgt]
        print(f'\n[{"|".join(src_langs)} -> {tgt}]')
        aucs = []
        for seed in args.seeds:
            auc = run_one(src_gs, tgt_g, args.K, args.loss,
                          args.hidden, args.lr, args.wd, args.epochs,
                          args.alpha, args.dropout, seed, device)
            aucs.append(auc)
            print(f'  seed={seed}  auc={auc:.4f}')
        mean, std = float(np.mean(aucs)), float(np.std(aucs))
        print(f'  => {tgt}: {mean*100:.2f} ± {std*100:.2f}')
        results[tgt] = {'mean': mean, 'std': std, 'aucs': aucs}

    avg = float(np.mean([v['mean'] for v in results.values()]))
    print(f'\nAVERAGE: {avg*100:.2f}%')
    results['_avg'] = avg

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'K': args.K, 'loss': args.loss,
                       'hidden': args.hidden, 'lr': args.lr,
                       'epochs': args.epochs, 'results': results}, f, indent=2)
        print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
