"""Source-statistic diagnostic of TRIAGE (Algorithm 1, lines 1-7) for every benchmark and split.

Per domain the script measures the graph statistics that enter the depth rule, and
per leave-one-domain-out split it aggregates them over the source domains only.
Everything is written to one JSON file; plotting scripts read that file and never
retype numbers.

Symbol mapping (paper -> code):
    h_s             -> h                edge homophily over edges with both endpoints labeled
    dbar_s          -> dbar             mean degree 2|E|/n of the propagation graph
    E[d^2]/E[d] - 1 -> branching        mean excess degree (branching ratio of the local tree)
    lambda_1(B)     -> nb_lambda1       leading eigenvalue of the non-backtracking operator B
    lambda_2(B)     -> nb_lambda2       largest nontrivial real eigenvalue of B
    rho_s           -> rho_mean         |(C h_s - 1)/(C - 1)| sqrt(dbar_s)          (Eq. amplification)
                       rho_branch       |(C h_s - 1)/(C - 1)| sqrt(branching_s)     (degree-heterogeneous form)
                       rho_nb_label     |(C h_s - 1)/(C - 1)| sqrt(lambda_1(B_s))
                       rho_nb_free      lambda_2(B_s) / sqrt(lambda_1(B_s))          (label-free)
    sigma_h         -> sigma_h          population std of source h_s               (Eq. sigma_h)
    S_+             -> s_plus           source domains with rho_s > 1
    rho_min^+       -> rho_min_plus     min over S_+ of rho_s
    n_eff           -> n_eff            labeled training nodes summed over the sources
    pi_+            -> pi_minority      smallest class fraction among labeled nodes

The non-backtracking spectrum is obtained from the Ihara-Bass companion matrix
[[A, I - D], [I, 0]], whose nontrivial eigenvalues coincide with those of B.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.citation_dataset import load_citation_graph, VALID_NAMES
from src.data.additional_datasets import load_twitch_graph, load_fb100_graph, FB100_VALID_NAMES

sys.stdout.reconfigure(line_buffering=True)

TWITCH_LANGS = ["DE", "EN", "ES", "FR", "PT", "RU"]
RHO_KEYS = ["rho_mean", "rho_branch", "rho_nb_label", "rho_nb_free"]
NB_NUM_EIGS = 12


def symmetric_adjacency(edge_index, num_nodes):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    adj = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes, num_nodes)).tocsr()
    adj.data[:] = 1.0  # collapse duplicate entries
    adj.setdiag(0.0)
    adj.eliminate_zeros()
    assert (adj != adj.T).nnz == 0, "adjacency must be symmetric"
    return adj  # [n, n]


def non_backtracking_real_eigs(adj):
    n = adj.shape[0]
    deg = np.asarray(adj.sum(axis=1)).ravel()  # [n]
    eye = sp.identity(n, format="csr")
    companion = sp.bmat([[adj, eye - sp.diags(deg)], [eye, None]], format="csr")  # [2n, 2n]
    vals = spla.eigs(companion, k=NB_NUM_EIGS, which="LR", return_eigenvectors=False,
                     tol=1e-8, maxiter=100000)
    real = np.sort(vals[np.abs(vals.imag) < 1e-6].real)[::-1]
    assert len(real) >= 2
    return real


def domain_stats(g, num_classes, labeled):
    adj = symmetric_adjacency(g.edge_index, g.num_nodes)
    deg = np.asarray(adj.sum(axis=1)).ravel()  # [n]
    y = g.y.numpy()
    rows, cols = adj.nonzero()
    both = labeled[rows] & labeled[cols]
    assert both.sum() > 0
    h = float((y[rows[both]] == y[cols[both]]).mean())
    dbar = float(deg.mean())
    branching = float((deg ** 2).mean() / deg.mean() - 1.0)
    nb_real = non_backtracking_real_eigs(adj)
    nb_lambda1, nb_lambda2 = float(nb_real[0]), float(nb_real[1])
    signal = abs((num_classes * h - 1.0) / (num_classes - 1.0))
    counts = np.bincount(y[labeled], minlength=num_classes)
    return {
        "num_nodes": int(g.num_nodes),
        "num_undirected_edges": int(adj.nnz // 2),
        "num_labeled": int(labeled.sum()),
        "num_train": int(g.train_mask.sum()),
        "num_classes": num_classes,
        "class_counts": counts.tolist(),
        "pi_minority": float(counts.min() / counts.sum()),
        "h": h,
        "dbar": dbar,
        "branching": branching,
        "nb_real_eigs": nb_real.tolist(),
        "nb_lambda1": nb_lambda1,
        "nb_lambda2": nb_lambda2,
        "rho_mean": signal * math.sqrt(dbar),
        "rho_branch": signal * math.sqrt(branching),
        "rho_nb_label": signal * math.sqrt(nb_lambda1),
        "rho_nb_free": nb_lambda2 / math.sqrt(nb_lambda1),
    }


def split_stats(domains, target):
    sources = [name for name in domains if name != target]
    hs = np.array([domains[s]["h"] for s in sources])
    out = {
        "target": target,
        "sources": sources,
        "num_sources": len(sources),
        "vandermonde_bound": len(sources) - 1,
        "h_bar": float(hs.mean()),
        "sigma_h": float(hs.std()),
        "n_eff": int(sum(domains[s]["num_train"] for s in sources)),
    }
    for key in RHO_KEYS:
        s_plus = [s for s in sources if domains[s][key] > 1.0]
        out[key] = {
            "s_plus": s_plus,
            "rho_min_plus": min(domains[s][key] for s in s_plus) if s_plus else None,
            "target_rho": domains[target][key],
        }
    return out


def benchmark(domains):
    names = list(domains)
    hs = np.array([domains[n]["h"] for n in names])
    return {
        "domains": domains,
        "collection": {"h_bar": float(hs.mean()), "sigma_h": float(hs.std()),
                       "dbar_mean": float(np.mean([domains[n]["dbar"] for n in names]))},
        "splits": {t: split_stats(domains, t) for t in names},
    }


def report(name, bench):
    print(f"\n[{name}] collection h_bar={bench['collection']['h_bar']:.4f} "
          f"sigma_h={bench['collection']['sigma_h']:.4f} dbar_mean={bench['collection']['dbar_mean']:.2f}")
    for dom, s in bench["domains"].items():
        print(f"  {dom:16s} n={s['num_nodes']:6d} E={s['num_undirected_edges']:7d} h={s['h']:.4f} "
              f"dbar={s['dbar']:.2f} branch={s['branching']:.2f} nb1={s['nb_lambda1']:.2f} "
              f"nb2={s['nb_lambda2']:.2f} pi_min={s['pi_minority']:.3f} "
              + " ".join(f"{k}={s[k]:.4f}" for k in RHO_KEYS))
    for tgt, s in bench["splits"].items():
        line = f"  split->{tgt:16s} S={s['num_sources']} sigma_h={s['sigma_h']:.4f} n_eff={s['n_eff']}"
        for key in RHO_KEYS:
            r = s[key]
            rmin = "none" if r["rho_min_plus"] is None else f"{r['rho_min_plus']:.4f}"
            line += f" | {key}: S+={len(r['s_plus'])} min={rmin} tgt={r['target_rho']:.4f}"
        print(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--citation_root", default="data/citation/raw_download")
    p.add_argument("--twitch_root", default="data/eerm/twitch")
    p.add_argument("--fb100_root", default="data/facebook100")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    print(f"Config: {vars(args)}  torch={torch.__version__}")

    results = {}
    cit = {}
    for n in VALID_NAMES:
        g = load_citation_graph(n, args.citation_root, "cpu")
        cit[n] = domain_stats(g, 5, np.ones(g.num_nodes, dtype=bool))
    results["Citations"] = benchmark(cit)
    report("Citations", results["Citations"])

    twi = {}
    for lang in TWITCH_LANGS:
        g = load_twitch_graph(lang, args.twitch_root, "cpu")
        twi[lang] = domain_stats(g, 2, np.ones(g.num_nodes, dtype=bool))
    results["Twitch"] = benchmark(twi)
    report("Twitch", results["Twitch"])

    fb = {}
    for campus in FB100_VALID_NAMES:
        g = load_fb100_graph(campus, args.fb100_root, "cpu")
        labeled = (g.train_mask | g.val_mask | g.test_mask).numpy()
        fb[campus] = domain_stats(g, 2, labeled)
    results["Facebook-100"] = benchmark(fb)
    report("Facebook-100", results["Facebook-100"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
