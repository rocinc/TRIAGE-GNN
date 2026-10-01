"""Per-domain linear probe of K-hop propagated features (source-only depth diagnostic).

For each domain the features are propagated with the same operator GPR-GNN uses,
H^(k) = A_sym^k X with A_sym the symmetrically normalized adjacency with self-loops,
and at every hop a linear classifier is fitted on the domain's training nodes and
scored on its test nodes. The balanced error of every class pair is converted to
an effective SNR through the Gaussian error relation eps = Q(sqrt(SNR)).

Symbol mapping (paper -> code):
    A_sym           -> adj_norm      D^{-1/2}(A + I)D^{-1/2},        [n, n] sparse
    H^(k)           -> h             k-hop propagated features,      [n, F]
    eps_K^{(a,b)}   -> pair_err      balanced test error of class pair (a, b) at hop k
    SNR_K           -> snr           (Q^{-1}(eps))^2 of the hardest pair (Prop. c_class, Delta_min)
    SNR_0           -> snr at k = 0
    A_s(tau)        -> computed downstream as log(SNR_tau / SNR_0)

A pair error of exactly zero means the pair is separated beyond the resolution of
the test set; it is stored as 0 and the SNR of that hop is left unresolved (null).
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.citation_dataset import load_citation_graph, VALID_NAMES
from src.data.additional_datasets import load_twitch_graph, load_fb100_graph, FB100_VALID_NAMES
from src.utils.seed import set_seed

sys.stdout.reconfigure(line_buffering=True)

TWITCH_LANGS = ["DE", "EN", "ES", "FR", "PT", "RU"]
# Near-separable pairs under the weakest ridge put the optimum far from the origin,
# and damped Newton needs many short steps before the quadratic phase; the budget is
# a guard against divergence only, and the iteration count is recorded per fit.
NEWTON_MAX_ITER = 20000
NEWTON_GRAD_TOL = 1e-10
NEWTON_DEC_TOL = 1e-14  # loss suboptimality certified at stop, near float64 resolution of loss ~ 0.7


def sym_norm_adjacency(edge_index, num_nodes, device):
    loops = torch.arange(num_nodes, device=device)
    ei = torch.cat([edge_index.to(device), torch.stack([loops, loops])], dim=1)  # [2, E + n]
    ones = torch.ones(ei.size(1), dtype=torch.float64, device=device)
    deg = torch.zeros(num_nodes, dtype=torch.float64, device=device).index_add_(0, ei[1], ones)
    w = deg[ei[0]].rsqrt() * deg[ei[1]].rsqrt()  # [E + n]
    return torch.sparse_coo_tensor(ei, w, (num_nodes, num_nodes)).coalesce()  # [n, n]


def fit_binary_newton(f_tr, y_tr, l2):
    """Class-balanced ridge logistic regression solved by damped Newton in float64.

    Objective on coordinates f (see row_space_basis):
        L(beta, b) = sum_i c_i * softplus(-s_i (f_i beta + b)) + l2 * ||beta||^2,
    s_i in {-1, +1}, c_i the balanced class weight normalized to sum one. The Hessian
    is formed exactly and factored by Cholesky, so convergence is quadratic and
    independent of how collinear the propagated features are. The iteration stops when
    the max-abs gradient reaches NEWTON_GRAD_TOL or when the Newton decrement certifies
    a loss suboptimality below NEWTON_DEC_TOL; the criterion that fired is recorded.
    """
    z = torch.cat([f_tr, torch.ones(f_tr.size(0), 1, dtype=f_tr.dtype, device=f_tr.device)], 1)  # [n, r+1]
    counts = torch.bincount(y_tr, minlength=2).to(f_tr.dtype)
    assert (counts > 0).all()
    c = (counts.sum() / (2 * counts))[y_tr]  # [n]
    c = c / c.sum()
    s = 2.0 * y_tr.to(f_tr.dtype) - 1.0  # [n]
    pen = torch.full((z.size(1),), 2.0 * l2, dtype=f_tr.dtype, device=f_tr.device)
    pen[-1] = 0.0  # intercept is unpenalized

    def loss_grad(theta):
        m = s * (z @ theta)  # [n]
        loss = (c * torch.nn.functional.softplus(-m)).sum() + 0.5 * (pen * theta * theta).sum()
        grad = z.T @ (-c * s * torch.sigmoid(-m)) + pen * theta  # [r+1]
        return loss, grad, m

    theta = torch.zeros(z.size(1), dtype=f_tr.dtype, device=f_tr.device)  # [r+1]
    loss, grad, m = loss_grad(theta)
    n_iter = 0
    stop = None
    while True:
        if grad.abs().max() <= NEWTON_GRAD_TOL:
            stop = "gradient"
            break
        assert n_iter < NEWTON_MAX_ITER
        p = torch.sigmoid(m)
        hess = z.T @ (z * (c * p * (1.0 - p)).unsqueeze(1)) + torch.diag(pen)  # [r+1, r+1]
        # With l2 > 0 the Hessian is positive definite except on the intercept, whose
        # curvature is strictly positive whenever both classes are present.
        step = torch.cholesky_solve(-grad.unsqueeze(1), torch.linalg.cholesky(hess)).squeeze(1)
        # Newton decrement: lambda^2 / 2 estimates the remaining suboptimality in loss.
        # With large feature magnitudes the float64 gradient has a roundoff floor above
        # NEWTON_GRAD_TOL, so the decrement is the criterion that certifies the optimum.
        decrement = float(-(grad @ step))
        if 0.5 * decrement <= NEWTON_DEC_TOL:
            stop = "decrement"
            break
        t = 1.0
        while True:  # Armijo backtracking keeps the damped Newton step globally convergent
            new_loss, new_grad, new_m = loss_grad(theta + t * step)
            if new_loss <= loss + 1e-4 * t * (grad @ step):
                break
            t *= 0.5
            assert t > 1e-12
        theta, loss, grad, m = theta + t * step, new_loss, new_grad, new_m
        n_iter += 1
    beta, b = theta[:-1], theta[-1]
    diag = {"n_iter": n_iter, "grad_norm": float(grad.abs().max()), "stop": stop, "converged": True}
    return (lambda f: torch.stack([-(f @ beta + b), f @ beta + b], 1)), diag


def row_space_basis(x_tr):
    """Exact coordinates for ridge regression on standardized features.

    With a ridge penalty on w, the minimizer lies in the row space of the
    standardized training matrix z_tr (the component of w orthogonal to it changes
    no training margin and only adds penalty). With z_tr = U S V^T and V_r the right
    singular vectors of nonzero singular values, w = V_r beta and ||w|| = ||beta||,
    so fitting on f = z V_r is the same problem in r <= min(n, F) coordinates.
    """
    mu, sd = x_tr.mean(0), x_tr.std(0).clamp(min=1e-6)  # guard zero-variance columns only
    z_tr = (x_tr - mu) / sd  # [n, F]
    _, sv, vh = torch.linalg.svd(z_tr, full_matrices=False)  # sv [q], vh [q, F]
    keep = sv > sv[0] * max(z_tr.shape) * torch.finfo(z_tr.dtype).eps
    v_r = vh[keep].T  # [F, r]
    return (lambda x: ((x - mu) / sd) @ v_r), int(keep.sum())


def balanced_error(pred, y):
    return float(np.mean([(pred[y == c] != c).float().mean().item() for c in (0, 1)]))


def tuned_error(h, y, tr, va, te, l2_grid):
    # Ridge strength is chosen on the domain's validation nodes, so every hop gets
    # its best linear probe and the curve is free of a fixed-regularization artifact.
    to_coords, rank = row_space_basis(h[tr])
    f_tr, f_va, f_te = to_coords(h[tr]), to_coords(h[va]), to_coords(h[te])  # [n_*, r]
    best = None
    grid_stops = []  # stopping criterion of every fit in the grid, selected or not
    for l2 in l2_grid:
        probe, diag = fit_binary_newton(f_tr, y[tr], l2)
        grid_stops.append(diag["stop"])
        val_err = balanced_error(probe(f_va).argmax(1), y[va])
        if best is None or val_err < best[0]:
            best = (val_err, l2, probe, diag)
    best[3]["rank"] = rank
    best[3]["grid_stops"] = grid_stops
    return balanced_error(best[2](f_te).argmax(1), y[te]), best[1], best[3]


def probe_domain(g, num_classes, hops, l2_grid, device):
    tr, va, te = g.train_mask.to(device), g.val_mask.to(device), g.test_mask.to(device)
    y = g.y.to(device)
    adj_norm = sym_norm_adjacency(g.edge_index, g.num_nodes, device)
    pairs = list(itertools.combinations(range(num_classes), 2))
    # float64 throughout: deep hops are nearly collinear, and the row-space basis and
    # Cholesky factor both need the extra precision to stay exact.
    h = g.x.to(device, torch.float64)  # [n, F]
    out = {"hops": hops, "pair_err": [], "pair_l2": [], "pair_iter": [], "pair_grad": [],
           "pair_converged": [], "pair_rank": [], "pair_stop": [], "snr": []}
    for k in range(max(hops) + 1):
        if k > 0:
            h = torch.sparse.mm(adj_norm, h)  # [n, F]
        if k not in hops:
            continue
        errs, l2s, iters, grads, conv, ranks, stops = [], [], [], [], [], [], []
        for a, b in pairs:
            in_pair = (y == a) | (y == b)
            y_bin = (y == b).long()  # [n]
            err, l2, diag = tuned_error(h, y_bin, tr & in_pair, va & in_pair, te & in_pair, l2_grid)
            errs.append(err)
            l2s.append(l2)
            iters.append(diag["n_iter"])
            grads.append(diag["grad_norm"])
            conv.append(diag["converged"])
            ranks.append(diag["rank"])
            stops.append(diag["grid_stops"])
        worst = max(errs)
        snr = float(norm.isf(worst) ** 2) if 0.0 < worst < 0.5 else None
        out["pair_err"].append(errs)
        out["pair_l2"].append(l2s)
        out["pair_iter"].append(iters)
        out["pair_grad"].append(grads)
        out["pair_converged"].append(conv)
        out["pair_rank"].append(ranks)
        out["pair_stop"].append(stops)
        out["snr"].append(snr)
        print(f"    k={k:2d} worst_pair_err={worst:.4f} "
              f"snr={'unresolved' if snr is None else f'{snr:.4f}'} "
              f"max_iter_used={max(iters)} max_grad={max(grads):.2e} "
              f"rank=[{min(ranks)},{max(ranks)}] "
              f"decrement_stops={sum(s.count('decrement') for s in stops)}/{sum(len(s) for s in stops)} "
              f"converged={all(conv)}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--citation_root", default="data/citation/raw_download")
    p.add_argument("--twitch_root", default="data/eerm/twitch")
    p.add_argument("--fb100_root", default="data/facebook100")
    p.add_argument("--hops", nargs="+", type=int, default=list(range(51)))
    p.add_argument("--l2_grid", nargs="+", type=float, default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--benchmarks", nargs="+", default=["Citations", "Twitch", "Facebook-100"])
    p.add_argument("--domains", nargs="+", default=None,
                   help="restrict to these domain names, so one process can own one domain")
    # The probe issues thousands of tiny L-BFGS evaluations per hop; on a GPU shared
    # with training jobs, time slicing dominates, so CPU with many threads is the default.
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    print(f"Config: {vars(args)}  torch={torch.__version__} cuda={torch.version.cuda}")

    loaders = {
        "Citations": (VALID_NAMES, lambda n: load_citation_graph(n, args.citation_root, "cpu"), 5),
        "Twitch": (TWITCH_LANGS, lambda n: load_twitch_graph(n, args.twitch_root, "cpu"), 2),
        "Facebook-100": (FB100_VALID_NAMES, lambda n: load_fb100_graph(n, args.fb100_root, "cpu"), 2),
    }
    results = {"config": vars(args)}
    for bench in args.benchmarks:
        names, load, num_classes = loaders[bench]
        if args.domains is not None:
            assert all(d in names for d in args.domains)
            names = [n for n in names if n in args.domains]
        results[bench] = {}
        for name in names:
            print(f"[{bench}] {name}")
            results[bench][name] = probe_domain(load(name), num_classes, args.hops, args.l2_grid, device)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
