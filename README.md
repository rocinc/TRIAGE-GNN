# TRIAGE: Three-Regime Identification for Adaptive Graph Estimation

Standard GPR-GNN leaves the propagation depth K as a hyperparameter because no general selection rule existed. TRIAGE closes that gap. Given source-domain homophily statistics, Proposition 2 of the paper returns the ERM-optimal K in closed form, which removes one axis of tuning entirely. Experiments follow the leave-one-domain-out protocol on three benchmarks (Citations, Twitch, and Facebook-100).

---

## Requirements

Python 3.10. Tested on 8× NVIDIA RTX 3090 (24 GB each), CUDA 12.4.

PyTorch and PyG require CUDA-matched wheels. Install in this order:

```bash
conda create -n triage python=3.10
conda activate triage

# Step 1: PyTorch (adjust cu124 if your CUDA version differs)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Step 2: PyG scatter/sparse/cluster extensions (must match the torch version above)
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 \
    -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

# Step 3: torch-geometric and remaining packages
pip install torch-geometric==2.7.0
pip install numpy==2.2.6 scikit-learn==1.7.2 scipy==1.15.3 pandas==2.2.3 ogb==1.3.6
```

`requirements.txt` lists the same versions without CUDA tags and can be used for CPU-only installs or as a reference freeze.

---

## Datasets

**Citations: ACMv9 / Citationv1 / DBLPv7**

Downloaded automatically by PyG on first run. PyG writes to `data/citation/raw_download/` (created if absent).

**Twitch: DE / EN / ES / FR / PT / RU / TW**

Raw MUSAE files from [benedekrozemberczki/MUSAE](https://github.com/benedekrozemberczki/MUSAE). Each language needs three files in its own subdirectory:

```
data/eerm/twitch/
    DE/
        musae_DE_target.csv
        musae_DE_features.json
        musae_DE_edges.csv
    EN/  ES/  FR/  PT/  RU/  TW/   ← same structure
```

**Facebook-100: Penn94, Johns Hopkins55, Cornell5, Yale4, Brown11, Caltech36, Amherst41, Texas80**

`.mat` files from [CUAI/Non-Homophily-Benchmarks](https://github.com/CUAI/Non-Homophily-Benchmarks). Place under `data/facebook100/`.

**Elliptic Bitcoin**

CSV files from [Kaggle Elliptic](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set). Place under `data/elliptic/raw/elliptic_bitcoin_dataset/`.

---

## Usage

**Core training (Citations, leave-one-domain-out):**

```bash
python main.py --target DBLPv7 --exp_name run_dblp
python main.py --target Citationv1 --exp_name run_cit
python main.py --target ACMv9 --exp_name run_acm
```

**Experiment scripts** (in `experiments/`):

| Script | Produces |
|---|---|
| `citations_main.py --target ACMv9` | Table 1 TRIAGE (AUCLoss + TTA-EncStd) |
| `citations_aucloss.py --target ACMv9` | Table 4 ablation: K=50 + AUCLoss row |
| `citations_ce_ablation.py --target ACMv9` | Table 4 ablation: K=50 + domain-avg CE row |
| `twitch_accuracy.py` | Supplementary per-domain Accuracy table |
| `twitch_depth_ablation.py --K 50 --loss auc` | Supplementary per-domain depth ablation figures |
| `fb100_accuracy.py` | Table 2 FB-100 Accuracy (8-campus LOO, degree-normalized high-pass filter) |
| `webkb_diagnostic.py` | Supplementary WebKB GPR vs MLP table |
| `elliptic_baseline.py` | Supplementary Elliptic T1/T2/T3 GPR baseline |
| `elliptic_calibration.py` | Supplementary Elliptic + prior calibration |
| `csbm_kstar_sweep.py` | Supplementary K* vs homophily variance figures |

Logs and figures write to `logs/`. Create it first: `mkdir -p logs`.

---
