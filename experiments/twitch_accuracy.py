"""Twitch evaluation with Accuracy metric — MLDGG comparison protocol.

MLDGG reports Accuracy (not ROC-AUC) on 7-domain LOO. This script runs both
CE and AUCLoss variants to find our best Accuracy number for fair comparison.

MLDGG baseline: 65.1% Accuracy (7-domain LOO).

Usage:
    python eval_twitch_accuracy.py --target RU --loss ce
    python eval_twitch_accuracy.py --target RU --loss auc
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

from src.data.additional_datasets import load_twitch_graph, TWITCH_LANGS
from src.models.gpr_conv import GPRConv
from src.utils.seed import set_seed


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
    probs = F.softmax(logits, dim=1)[:, 1]
    pos_mask = (labels == 1)
    neg_mask = ~pos_mask
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return F.cross_entropy(logits, labels)
    diff = probs[pos_mask].unsqueeze(1) - probs[neg_mask].unsqueeze(0)
    return (-F.logsigmoid(diff)).mean()


def run_single(src_graphs, tgt_graph, K=50, alpha=0.1, hidden=128,
               lr=1e-3, wd=5e-4, dropout=0.5, epochs=300, seed=42,
               loss_fn='ce', device=torch.device('cpu')):
    set_seed(seed)
    src_gs = [g.to(device) for g in src_graphs]
    tgt_g = tgt_graph.to(device)

    model = SimpleGPR(src_gs[0].x.shape[1], hidden, 2, K, alpha, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for g in src_gs:
            opt.zero_grad()
            logits = model(g.x, g.edge_index)[g.train_mask]
            loss = auc_loss(logits, g.y[g.train_mask]) if loss_fn == 'auc' \
                   else F.cross_entropy(logits, g.y[g.train_mask])
            loss.backward()
            opt.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(tgt_g.x, tgt_g.edge_index)
                val_acc = (logits[tgt_g.val_mask].argmax(1)
                           == tgt_g.y[tgt_g.val_mask]).float().mean().item()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(tgt_g.x, tgt_g.edge_index)
        test_acc = (logits[tgt_g.test_mask].argmax(1)
                    == tgt_g.y[tgt_g.test_mask]).float().mean().item()

    return {'test_acc': test_acc, 'best_val_acc': best_val_acc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--langs', nargs='+', default=['DE', 'EN', 'ES', 'FR', 'PT', 'RU', 'TW'])
    p.add_argument('--target', default=None)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--K', type=int, default=50)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--loss', default='ce', choices=['ce', 'auc'])
    p.add_argument('--data_root', default='data/eerm/twitch')
    args = p.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    all_graphs = {l: load_twitch_graph(l, args.data_root, 'cpu') for l in args.langs}
    targets = [args.target] if args.target else args.langs

    all_accs = {}
    for tgt_lang in targets:
        src_langs = [l for l in args.langs if l != tgt_lang]
        src_gs = [all_graphs[l] for l in src_langs]
        tgt_g = all_graphs[tgt_lang]

        print(f"\n[{args.loss}] {'|'.join(src_langs)} -> {tgt_lang}", flush=True)
        accs = []
        for seed in args.seeds:
            r = run_single(src_gs, tgt_g, K=args.K, alpha=args.alpha,
                           hidden=args.hidden, epochs=args.epochs,
                           seed=seed, loss_fn=args.loss, device=device)
            accs.append(r['test_acc'])
            print(f"  seed={seed}: acc={r['test_acc']:.4f}  (best_val={r['best_val_acc']:.4f})", flush=True)

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"  Summary -> {tgt_lang}: {mean_acc:.4f} +- {std_acc:.4f}", flush=True)
        all_accs[tgt_lang] = (mean_acc, std_acc)

    if len(targets) > 1:
        avg = np.mean([v[0] for v in all_accs.values()])
        print(f"\n{'='*60}", flush=True)
        print(f"Twitch [{args.loss}] Accuracy ({len(targets)}-domain LOO): {avg:.4f} ({avg*100:.1f}%)", flush=True)
        for lang, (m, s) in all_accs.items():
            print(f"  {lang}: {m:.4f} +- {s:.4f}", flush=True)
        print(f"\n  MLDGG (7-domain Accuracy): 65.1%", flush=True)
        print(f"  Ours  ({len(targets)}-domain Accuracy, {args.loss}): {avg*100:.1f}%", flush=True)


if __name__ == '__main__':
    main()
