"""Entry point for TRIAGE experiments.

Usage:
    # E0: ERM GCN baseline (no GPR, no disentangle)
    python main.py --target DBLPv7 --exp_name e0_cd_a --no_gpr --no_disentangle

    # E1: GPR K=50 only
    python main.py --target DBLPv7 --exp_name e1_cd_a

    # E2: GPR + disentangle (GCE+WCE+HSIC)
    python main.py --target DBLPv7 --exp_name e2_cd_a --use_disentangle --no_con --no_var --no_swap

    # E5: full method
    python main.py --target DBLPv7 --exp_name e5_cd_a --use_disentangle

    # Dry run (2 epochs, no checkpoint saved)
    python main.py --target DBLPv7 --exp_name dryrun --epochs 2 --dry_run

All source graphs are inferred as the complement of --target within
{ACMv9, Citationv1, DBLPv7}.

Results are logged to logs/<exp_name>.log and printed to stdout.
"""

import argparse
import sys
from pathlib import Path

import torch

from src.data.citation_dataset import VALID_NAMES, load_citation_graph
from src.models.spectral_causal_gnn import SpectralCausalGNN
from src.trainer.mdg_trainer import MDGTrainer
from src.utils.logger import get_logger
from src.utils.seed import set_seed


DATA_ROOT = "data/citation/raw_download"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TRIAGE experiments")

    # --- Experiment identity ---
    p.add_argument("--target", required=True, choices=VALID_NAMES,
                   help="Leave-one-out target domain.")
    p.add_argument("--exp_name", required=True,
                   help="Experiment name used for log file naming.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true",
                   help="Run 2 epochs and exit without saving checkpoints.")

    # --- Model architecture ---
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_hops", type=int, default=50,
                   help="GPR propagation steps K.")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="GPR weight initialization parameter.")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--no_gpr", action="store_true",
                   help="Disable GPR propagation (use plain linear encoder). E0 mode.")
    p.add_argument("--use_disentangle", action="store_true",
                   help="Enable two-head causal disentanglement (C2 mode).")
    p.add_argument("--d_stable", type=int, default=128,
                   help="Stable head output dimension (only used with --use_disentangle).")
    p.add_argument("--d_env", type=int, default=128,
                   help="Environment head output dimension (only used with --use_disentangle).")

    # --- Training schedule ---
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--warmup_sup", type=int, default=50,
                   help="Epoch at which Phase 2 losses (contrastive, disentangle) activate.")
    p.add_argument("--warmup_con", type=int, default=150,
                   help="Epoch at which Phase 3 losses (VREx, swap) activate.")

    # --- Loss weights ---
    p.add_argument("--lambda_env", type=float, default=1.0,
                   help="Weight for GCE loss on env head.")
    p.add_argument("--lambda_wce", type=float, default=1.0,
                   help="Weight for WCE loss on stable head.")
    p.add_argument("--lambda_con", type=float, default=0.1,
                   help="Weight for supervised contrastive loss.")
    p.add_argument("--lambda_hsic", type=float, default=0.1,
                   help="Weight for HSIC independence penalty.")
    p.add_argument("--lambda_var", type=float, default=0.1,
                   help="Weight for VREx variance loss.")
    p.add_argument("--lambda_swap", type=float, default=0.1,
                   help="Weight for environment swap loss.")

    # --- GCE parameter ---
    p.add_argument("--gce_q", type=float, default=0.7,
                   help="GCE truncation parameter q in (0, 1].")

    # --- Ablation flags ---
    p.add_argument("--no_con", action="store_true",
                   help="Disable supervised contrastive loss.")
    p.add_argument("--no_var", action="store_true",
                   help="Disable VREx variance loss.")
    p.add_argument("--no_swap", action="store_true",
                   help="Disable environment swap loss.")

    args = p.parse_args()

    if args.dry_run:
        args.epochs = 2
        # Collapse warmup phases so all loss terms activate immediately.
        args.warmup_sup = 0
        args.warmup_con = 0

    # Validate warmup ordering.
    assert args.warmup_sup <= args.warmup_con <= args.epochs, (
        f"warmup_sup ({args.warmup_sup}) <= warmup_con ({args.warmup_con}) "
        f"<= epochs ({args.epochs}) must hold"
    )

    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    log_dir = "logs" if not args.dry_run else None
    logger = get_logger(args.exp_name, log_dir=log_dir)

    logger.info(f"Config: {vars(args)}")
    logger.info(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Infer source domains.
    src_names = [n for n in VALID_NAMES if n != args.target]
    logger.info(f"Source domains: {src_names}  |  Target: {args.target}")

    # Load graphs.
    src_graphs = [
        load_citation_graph(name, DATA_ROOT, device) for name in src_names
    ]
    tgt_graph = load_citation_graph(args.target, DATA_ROOT, device)

    in_channels = src_graphs[0].x.size(1)
    num_classes = int(src_graphs[0].y.max().item()) + 1

    # Build model.
    model = SpectralCausalGNN(
        in_channels=in_channels,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_hops=args.num_hops,
        alpha=args.alpha,
        dropout=args.dropout,
        use_disentangle=args.use_disentangle,
        d_stable=args.d_stable,
        d_env=args.d_env,
    ).to(device)

    # Disable GPR if requested (E0 mode: plain linear encoder).
    if args.no_gpr:
        model.encoder.gpr_conv = None
        model.encoder.use_gpr = False

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    trainer = MDGTrainer(model=model, optimizer=optimizer, args=args, logger=logger)
    result = trainer.train(src_graphs, tgt_graph)

    # Final test evaluation.
    test_micro, test_macro = trainer.test(tgt_graph)

    logger.info(
        f"Final results | "
        f"best_micro_f1={result.best_micro_f1:.4f} | "
        f"best_macro_f1={result.best_macro_f1:.4f} | "
        f"best_epoch={result.best_epoch} | "
        f"test_micro={test_micro:.4f} | "
        f"test_macro={test_macro:.4f}"
    )


if __name__ == "__main__":
    main()
