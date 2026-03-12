"""InfoNCE / CPC-style contrastive loss for self-supervised audio pretraining.

Simplified CPC objective (Oord et al. 2018, Baevski et al. 2020):
  - Sample K anchor frames from the encoder output
  - For each anchor, predict the next N frames (positives)
  - Use all other in-batch/in-sequence frames as negatives
  - Minimize InfoNCE (log-softmax) over (positives, negatives)

This is the loss used in Stage 3 (Milestone 3) of the project.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for latent frame prediction.

    Predicts `pred_steps` future latents from a context vector using
    a set of in-batch negatives.

    Args:
        d_model:     Encoder dimension.
        pred_steps:  Number of future steps to predict (positive frames).
        n_negatives: Number of negative samples drawn per anchor.
        temperature: Logit scaling temperature.
    """

    def __init__(
        self,
        d_model: int = 256,
        pred_steps: int = 12,
        n_negatives: int = 100,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.pred_steps = pred_steps
        self.n_negatives = n_negatives
        self.temperature = temperature

        # One linear predictor per future step
        self.predictors = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(pred_steps)
        ])

    def forward(self, context: Tensor, targets: Tensor, lengths: Tensor | None = None) -> Tensor:
        """
        Args:
            context: (B, T, d_model) — encoded context (e.g. output of Conformer)
            targets: (B, T, d_model) — quantized or projected target representations
            lengths: (B,) — actual sequence lengths (to avoid using padding as negatives)
        Returns:
            Scalar InfoNCE loss averaged over steps and batch.
        """
        B, T, D = context.shape
        total_loss = context.new_zeros(1)
        count = 0

        for step, predictor in enumerate(self.predictors, start=1):
            if T - step <= 0:
                break

            # Anchor context: all positions except last `step`
            ctx = context[:, :-step, :]    # (B, T-step, D)
            # Positive: true future latent at t+step
            pos = targets[:, step:, :]     # (B, T-step, D)

            # Predicted query from anchor
            query = predictor(ctx)         # (B, T-step, D)
            query = F.normalize(query, dim=-1)
            pos = F.normalize(pos, dim=-1)

            # Positive similarity
            pos_sim = (query * pos).sum(dim=-1, keepdim=True) / self.temperature  # (B, T-step, 1)

            # Negative sampling: draw random frames from the batch
            # Shape: (B, T-step, n_negatives, D)
            T2 = T - step
            neg_idx = torch.randint(0, B * T2, (B, T2, self.n_negatives), device=ctx.device)
            flat_targets = targets[:, step:, :].reshape(B * T2, D)  # (B*T2, D)
            flat_targets = F.normalize(flat_targets, dim=-1)
            negs = flat_targets[neg_idx.view(-1)].view(B, T2, self.n_negatives, D)  # (B, T2, K, D)

            neg_sim = torch.einsum("btd,btkd->btk", query, negs) / self.temperature  # (B, T2, K)

            # InfoNCE: cross-entropy with positive as class 0
            logits = torch.cat([pos_sim, neg_sim], dim=-1)  # (B, T2, 1+K)
            labels = torch.zeros(B, T2, dtype=torch.long, device=ctx.device)
            loss_step = F.cross_entropy(logits.view(B * T2, -1), labels.view(-1))
            total_loss = total_loss + loss_step
            count += 1

        return total_loss / max(count, 1)
