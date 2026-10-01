"""
Synthetic CSBM test of Proposition 2 (ERM-Optimal Filter Order): K* vs homo_std sweep.

Symbol mapping (paper -> code):
    h_i       -> homo_vals[i]   per-domain homophily,              [N_domains]
    rho_i     -> 2*h_i - 1      spectral signal amplitude per hop
    p_i, q_i  -> p, q           intra/inter-class edge probabilities in CSBM
    A_norm    -> A               sym-normalized adjacency,          [n, n] sparse
    z_K       -> feat            APPNP-smoothed features at hop K,  [n, feat_dim]
    K*        -> kstar           K achieving 95% of peak DG accuracy

Experiment design:
    - Generate CSBM(n=1000, d_avg=10, mu=0.3) graphs for N=5 domains
    - Domain homophily: h_i = h_bar + homo_std * z_i, z evenly spaced
    - For each K in K_VALS: smooth features via APPNP(K, alpha=0.1),
      train logistic regression on 4 domains, evaluate on held-out domain
    - K* = smallest K reaching 95% of best leave-one-out accuracy
    - Repeat for n_trials random seeds, report mean ± std
"""
import sys
import json
import time
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)

N_GRAPH      = 1000     # nodes per domain graph
N_DOMAINS    = 5
H_BAR        = 0.6
D_AVG        = 10       # average degree
MU           = 0.3      # inter-class mean shift (small so graph signal matters)
FEAT_DIM     = 16
ALPHA        = 0.1      # APPNP teleport probability
HOMO_STD_VALS = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]
K_VALS        = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
N_TRIALS      = 10
import os as _os
_HERE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
JSON_PATH     = _os.path.join(_HERE, "logs", "b1_kstar_results.json")
PNG_PATH      = _os.path.join(_HERE, "logs", "b1_kstar_curve.png")
PDF_PATH      = _os.path.join(_HERE, "logs", "b1_kstar_curve.pdf")


def gen_csbm(n: int, h: float, d: float, mu: float, seed: int):
    """
    Generate CSBM graph with n nodes, homophily h, avg degree d, signal strength mu.

    Returns:
        x:     [n, FEAT_DIM] node features
        A:     [n, n] sym-normalized adjacency (sparse)
        y:     [n] binary labels
    """
    rng = np.random.default_rng(seed)
    p = h * d / (n - 1)
    q = (1.0 - h) * d / (n - 1)
    y = rng.integers(0, 2, n)
    x = rng.normal(0.0, 1.0, (n, FEAT_DIM))
    x[y == 1] += mu

    # Sparse SBM adjacency
    same = y[:, None] == y[None, :]
    np.fill_diagonal(same, False)
    probs = np.where(same, p, q)
    upper = np.triu(rng.random((n, n)) < probs, 1)
    r, c = np.where(upper)
    rows = np.concatenate([r, c])
    cols = np.concatenate([c, r])
    A_raw = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

    # Sym-normalize: D^{-1/2} A D^{-1/2}
    deg = np.array(A_raw.sum(1)).ravel()
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1)))
    A_norm = d_inv_sqrt @ A_raw @ d_inv_sqrt
    return x, A_norm, y


def appnp_smooth(x: np.ndarray, A, K: int, alpha: float = ALPHA) -> np.ndarray:
    """
    APPNP propagation: z^(k) = (1-alpha)*A*z^(k-1) + alpha*x, z^(0) = x.

    Args:
        x:     [n, feat_dim] initial features
        A:     sym-normalized adjacency (sparse)
        K:     number of propagation steps
        alpha: teleport probability

    Returns:
        z: [n, feat_dim] smoothed features after K steps
    """
    z = x.copy()
    for _ in range(K):
        z = (1.0 - alpha) * (A @ z) + alpha * x
    return z


def leave_one_out_acc(homo_vals: np.ndarray, K: int, seed_offset: int) -> float:
    """
    Train on N-1 domains, evaluate on held-out domain, average over all N splits.

    Args:
        homo_vals:   [N_DOMAINS] per-domain homophily
        K:           APPNP filter order
        seed_offset: base seed for reproducibility

    Returns:
        average leave-one-out accuracy
    """
    domain_data = []
    for i, h in enumerate(homo_vals):
        x, A, y = gen_csbm(N_GRAPH, h, D_AVG, MU, seed_offset + i)
        feat = appnp_smooth(x, A, K)
        domain_data.append((feat, y))

    accs = []
    for test_i in range(N_DOMAINS):
        train_x = np.vstack([domain_data[i][0] for i in range(N_DOMAINS) if i != test_i])
        train_y = np.concatenate([domain_data[i][1] for i in range(N_DOMAINS) if i != test_i])
        test_x, test_y = domain_data[test_i]
        clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
        clf.fit(train_x, train_y)
        accs.append(clf.score(test_x, test_y))
    return float(np.mean(accs))


def find_kstar(k_vals: list, accs: np.ndarray, threshold: float = 0.95) -> int:
    """Smallest K reaching threshold * max accuracy."""
    max_acc = accs.max()
    target = threshold * max_acc
    for k, a in zip(k_vals, accs):
        if a >= target:
            return k
    return k_vals[-1]


def make_homo_vals(h_bar: float, homo_std: float, n: int,
                   rng: np.random.Generator) -> np.ndarray:
    """n evenly-spaced domain homophily values centered at h_bar + small jitter."""
    z = np.linspace(-1.0, 1.0, n)
    z = z / (np.std(z) + 1e-12) * homo_std
    base = np.clip(h_bar + z, 0.01, 0.99)
    jitter = rng.normal(0.0, homo_std * 0.05, size=n)
    return np.clip(base + jitter, 0.01, 0.99)


def run_sweep() -> dict:
    results = {}
    print(
        f"K* sweep | {len(HOMO_STD_VALS)} homo_std x {N_TRIALS} trials | "
        f"h_bar={H_BAR}  N_domains={N_DOMAINS}  n={N_GRAPH}  mu={MU}  d_avg={D_AVG}",
        flush=True,
    )
    print(f"K_vals={K_VALS}", flush=True)
    print("-" * 60, flush=True)

    for homo_std in HOMO_STD_VALS:
        kstar_list = []
        t0 = time.time()
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(trial * 1000)
            homo_vals = make_homo_vals(H_BAR, homo_std, N_DOMAINS, rng)
            seed_offset = trial * 100
            accs = np.array([
                leave_one_out_acc(homo_vals, K, seed_offset) for K in K_VALS
            ])
            ks = find_kstar(K_VALS, accs)
            kstar_list.append(ks)

        kstar_mean = float(np.mean(kstar_list))
        kstar_std  = float(np.std(kstar_list))
        elapsed = time.time() - t0
        results[homo_std] = {
            "kstar_mean": kstar_mean,
            "kstar_std":  kstar_std,
            "kstar_list": kstar_list,
        }
        print(
            f"homo_std={homo_std:.3f} | K*={kstar_mean:.1f} +/- {kstar_std:.1f} | "
            f"{kstar_list} | {elapsed:.1f}s",
            flush=True,
        )

    return results


def fit_and_plot(results: dict) -> None:
    homo_std_arr = np.array(sorted(results.keys()), dtype=float)
    kstar_means  = np.array([results[h]["kstar_mean"] for h in homo_std_arr])
    kstar_stds   = np.array([results[h]["kstar_std"]  for h in homo_std_arr])

    # Fit power law K* ~ C * homo_std^beta
    def power_model(x, C, beta):
        return C * x ** beta

    try:
        popt, _ = curve_fit(power_model, homo_std_arr, kstar_means,
                            p0=[5.0, 0.5], maxfev=5000)
        C_fit, beta_fit = float(popt[0]), float(popt[1])
        print(f"Fit: K* = {C_fit:.4f} * homo_std^{beta_fit:.4f}", flush=True)
        fit_label = f"K* = {C_fit:.2f} * homo_std^{{{beta_fit:.2f}}}"
    except RuntimeError:
        C_fit = beta_fit = None
        print("power_model curve_fit failed", flush=True)
        fit_label = None

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(homo_std_arr, kstar_means, yerr=kstar_stds,
                fmt="o-", color="steelblue", capsize=4, label="Numerical K*")
    if C_fit is not None:
        x_fine = np.linspace(homo_std_arr.min(), homo_std_arr.max(), 300)
        ax.plot(x_fine, power_model(x_fine, C_fit, beta_fit),
                "--", color="tomato", label=fit_label)

    ax.set_xlabel("homo_std", fontsize=13)
    ax.set_ylabel("K*", fontsize=13)
    ax.set_title(
        "Optimal Filter Order K* vs Homophily Variance\n(CSBM, n=1000, N=5, APPNP+LR)",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=300)
    fig.savefig(PDF_PATH)
    print(f"Saved: {PNG_PATH}  {PDF_PATH}", flush=True)


def main() -> None:
    results = run_sweep()

    with open(JSON_PATH, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"Results: {JSON_PATH}", flush=True)

    fit_and_plot(results)

    print("=" * 60, flush=True)
    print(f"{'homo_std':>10}  {'K* mean':>8}  {'K* std':>8}", flush=True)
    for h in sorted(results.keys()):
        r = results[h]
        print(f"{h:>10.3f}  {r['kstar_mean']:>8.1f}  {r['kstar_std']:>8.2f}",
              flush=True)
    print("=" * 60, flush=True)
    print("EXIT:0", flush=True)


if __name__ == "__main__":
    main()
