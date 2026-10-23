"""Multi-task loss module: Pinball loss for quantiles + BCE for adverse events."""
import torch
import torch.nn as nn
from typing import List
from gluco_twin.config import cfg

class PinballLoss(nn.Module):
    """Pinball (quantile) loss for P10, P50, P90."""
    def __init__(self, quantiles: List[float] = None):
        super().__init__()
        self.quantiles = quantiles or cfg.quantiles

    def forward(self, pred_quantiles: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # pred_quantiles: (B, H, 3)
        # targets: (B, H)
        losses = []
        targets_expanded = targets.unsqueeze(-1)  # (B, H, 1)
        diff = targets_expanded - pred_quantiles  # (B, H, 3)

        for i, q in enumerate(self.quantiles):
            error = diff[..., i]
            loss_q = torch.maximum(q * error, (q - 1.0) * error)
            losses.append(loss_q.mean())

        return sum(losses) / len(losses)


class MultiTaskTwinLoss(nn.Module):
    """Combines Pinball quantile loss on future trajectory with BCE on spike event."""
    def __init__(self, pos_weight: float = 10.0, event_weight: float = 1.0):
        super().__init__()
        self.pinball = PinballLoss()
        self.event_weight = event_weight
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(
        self,
        pred_quantiles: torch.Tensor,
        event_logit: torch.Tensor,
        target_path: torch.Tensor,
        target_event: torch.Tensor,
    ) -> torch.Tensor:
        loss_quant = self.pinball(pred_quantiles, target_path)
        
        bce = nn.functional.binary_cross_entropy_with_logits(
            event_logit.squeeze(-1),
            target_event,
            pos_weight=self.pos_weight,
        )
        
        return loss_quant + self.event_weight * bce, loss_quant, bce
