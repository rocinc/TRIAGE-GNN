"""Multi-domain generalization trainer.

Training schedule (three phases):

    Phase 1 (epochs 0 .. warmup_sup-1):
        L_sup only (CE on z_stable / h).
        Lets the encoder and classifier converge before adding regularization.

    Phase 2 (epochs warmup_sup .. warmup_con-1):
        L_sup + L_env (GCE on z_env, only when use_disentangle=True)
                     + L_wce (WCE on z_stable, only when use_disentangle=True)
                     + L_con (supervised contrastive on z_stable)
                     + L_hsic (HSIC penalty, only when use_disentangle=True)

    Phase 3 (epochs warmup_con .. total_epochs-1):
        All Phase 2 losses + L_var (VREx) + L_swap (env swap).

All loss weights are controlled by args.lambda_* flags.
All components can be individually disabled via args.no_* flags for ablation.

Symbol mapping (paper → code):
    L_sup   → loss_sup     supervised CE on stable head
    L_env   → loss_env     GCE on environment head (drives spurious learning)
    L_wce   → loss_wce     weighted CE on stable head (forces causal learning)
    L_con   → loss_con     supervised contrastive loss
    L_hsic  → loss_hsic    HSIC independence penalty
    L_var   → loss_var     VREx gradient variance loss
    L_swap  → loss_swap    environment channel swap loss
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from src.losses.contrastive import supervised_contrastive_loss
from src.losses.env_swap import env_swap_loss
from src.losses.grad_variance import vrex_loss
from src.models.causal_disentangler import gce_loss, hsic_loss, wce_loss
from src.models.spectral_causal_gnn import SpectralCausalGNN
from src.utils.metrics import macro_f1, micro_f1


@dataclass
class TrainResult:
    """Summary of a completed training run.

    Attributes:
        best_micro_f1: Best validation Micro-F1 across all epochs.
        best_macro_f1: Macro-F1 at the epoch with best Micro-F1.
        best_epoch:    Epoch index (0-based) where best_micro_f1 was achieved.
    """

    best_micro_f1: float
    best_macro_f1: float
    best_epoch: int


class MDGTrainer:
    """Trainer for multi-domain generalization on citation graphs.

    Args:
        model:     SpectralCausalGNN instance.
        optimizer: Optimizer bound to model parameters.
        args:      Parsed argument namespace.  Required fields:
                       epochs, warmup_sup, warmup_con,
                       lambda_env, lambda_wce, lambda_con, lambda_hsic,
                       lambda_var, lambda_swap,
                       no_disentangle, no_con, no_var, no_swap,
                       gce_q.
        logger:    Python logger instance.
    """

    def __init__(
        self,
        model: SpectralCausalGNN,
        optimizer: torch.optim.Optimizer,
        args: Any,
        logger: logging.Logger,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.logger = logger

        # Swap classifier: accepts [d_stable + d_env] → num_classes.
        # Only created when disentanglement is active.
        if model.use_disentangle:
            d_stable = model.classifier.in_features
            d_env = model.env_classifier.in_features
            num_classes = model.classifier.out_features
            self.swap_classifier = nn.Linear(d_stable + d_env, num_classes)
            device = next(model.parameters()).device
            self.swap_classifier = self.swap_classifier.to(device)
            # Include swap_classifier parameters in optimizer.
            optimizer.add_param_group({"params": self.swap_classifier.parameters()})
        else:
            self.swap_classifier = None

    def train(
        self,
        src_graphs: list[Data],
        tgt_graph: Data,
    ) -> TrainResult:
        """Run the full training loop and return the best validation result.

        Args:
            src_graphs: List of source domain PyG Data objects.
            tgt_graph:  Target domain PyG Data object (used for validation/test).

        Returns:
            TrainResult with best_micro_f1, best_macro_f1, best_epoch.
        """
        best_micro = 0.0
        best_macro = 0.0
        best_epoch = 0
        best_state: dict | None = None  # saved weights at best val epoch

        for epoch in range(self.args.epochs):
            train_loss = self._train_epoch(src_graphs, epoch)
            val_micro, val_macro = self._evaluate(tgt_graph)

            if val_micro > best_micro:
                best_micro = val_micro
                best_macro = val_macro
                best_epoch = epoch
                import copy
                best_state = copy.deepcopy(self.model.state_dict())

            if epoch % 10 == 0 or epoch == self.args.epochs - 1:
                self.logger.info(
                    f"Epoch {epoch:3d}/{self.args.epochs} | "
                    f"loss={train_loss:.4f} | "
                    f"val_micro={val_micro:.4f} | val_macro={val_macro:.4f} | "
                    f"best={best_micro:.4f}@{best_epoch}"
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return TrainResult(
            best_micro_f1=best_micro,
            best_macro_f1=best_macro,
            best_epoch=best_epoch,
        )

    def _train_epoch(
        self,
        src_graphs: list[Data],
        epoch: int,
    ) -> float:
        """Run one training epoch over all source graphs.

        Returns:
            Total loss value as a Python float.
        """
        self.model.train()
        if self.swap_classifier is not None:
            self.swap_classifier.train()
        self.optimizer.zero_grad()

        all_outputs = []
        for graph in src_graphs:
            out = self.model(graph.x, graph.edge_index)
            all_outputs.append((graph, out))

        # --- Phase 1: supervised CE on stable head ---
        loss_sup = torch.tensor(0.0, device=src_graphs[0].x.device)
        loss_per_domain: list[torch.Tensor] = []

        for graph, out in all_outputs:
            mask = graph.train_mask
            logits = out.logits_stable[mask]  # [N_train, C]
            labels = graph.y[mask]            # [N_train]
            l_e = F.cross_entropy(logits, labels)
            loss_sup = loss_sup + l_e
            loss_per_domain.append(l_e)

        loss_sup = loss_sup / len(src_graphs)
        total_loss = loss_sup

        # --- Phase 2: disentanglement + contrastive ---
        if epoch >= self.args.warmup_sup:

            # GCE on env head + WCE on stable head (C2).
            if self.model.use_disentangle:
                loss_env = torch.tensor(0.0, device=src_graphs[0].x.device)
                loss_wce = torch.tensor(0.0, device=src_graphs[0].x.device)
                loss_hsic = torch.tensor(0.0, device=src_graphs[0].x.device)

                for graph, out in all_outputs:
                    mask = graph.train_mask
                    labels = graph.y[mask]
                    logits_env = out.logits_env[mask]
                    logits_stable = out.logits_stable[mask]
                    z_s = out.z_stable[mask]
                    z_e = out.z_env[mask]

                    loss_env = loss_env + gce_loss(logits_env, labels, q=self.args.gce_q)
                    loss_wce = loss_wce + wce_loss(logits_stable, logits_env, labels)
                    loss_hsic = loss_hsic + hsic_loss(z_s, z_e)

                loss_env = loss_env / len(src_graphs)
                loss_wce = loss_wce / len(src_graphs)
                loss_hsic = loss_hsic / len(src_graphs)

                total_loss = (
                    total_loss
                    + self.args.lambda_env * loss_env
                    + self.args.lambda_wce * loss_wce
                    + self.args.lambda_hsic * loss_hsic
                )

            # Supervised contrastive loss on z_stable (C2 / cross-domain alignment).
            if not self.args.no_con:
                all_z_stable = torch.cat(
                    [out.z_stable[g.train_mask] for g, out in all_outputs], dim=0
                )  # [N_total, d_stable]
                all_labels = torch.cat(
                    [g.y[g.train_mask] for g, _ in all_outputs], dim=0
                )  # [N_total]
                loss_con = supervised_contrastive_loss(
                    all_z_stable, all_labels, temperature=0.5
                )
                total_loss = total_loss + self.args.lambda_con * loss_con

        # --- Phase 3: VREx + env swap ---
        if epoch >= self.args.warmup_con:

            # VREx variance loss (C3).
            if not self.args.no_var:
                loss_var = vrex_loss(loss_per_domain)
                total_loss = total_loss + self.args.lambda_var * loss_var

            # Environment channel swap loss (C3).
            if not self.args.no_swap and self.model.use_disentangle:
                domain_z_stable = {
                    i: out.z_stable[g.train_mask]
                    for i, (g, out) in enumerate(all_outputs)
                }
                domain_z_env = {
                    i: out.z_env[g.train_mask]
                    for i, (g, out) in enumerate(all_outputs)
                }
                domain_labels = {
                    i: g.y[g.train_mask]
                    for i, (g, _) in enumerate(all_outputs)
                }
                loss_swap = env_swap_loss(
                    self.swap_classifier,
                    domain_z_stable,
                    domain_z_env,
                    domain_labels,
                )
                total_loss = total_loss + self.args.lambda_swap * loss_swap

        total_loss.backward()
        # Gradient clipping prevents NaN from HSIC or large contrastive gradients.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(total_loss.item())

    @torch.no_grad()
    def _evaluate(self, graph: Data) -> tuple[float, float]:
        """Evaluate on the target graph's validation set.

        Returns:
            (micro_f1, macro_f1) on the validation mask.
        """
        self.model.eval()
        out = self.model(graph.x, graph.edge_index)
        mask = graph.val_mask
        logits = out.logits_stable[mask]
        labels = graph.y[mask]
        return micro_f1(logits, labels), macro_f1(logits, labels)

    @torch.no_grad()
    def test(self, graph: Data) -> tuple[float, float]:
        """Evaluate on the target graph's test set.

        Returns:
            (micro_f1, macro_f1) on the test mask.
        """
        self.model.eval()
        out = self.model(graph.x, graph.edge_index)
        mask = graph.test_mask
        logits = out.logits_stable[mask]
        labels = graph.y[mask]
        return micro_f1(logits, labels), macro_f1(logits, labels)
