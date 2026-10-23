"""Training loop for FusionTwinNet."""
import time
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from gluco_twin.config import cfg, PROCESSED_DATA_DIR, CURRENT_MODEL_DIR
from gluco_twin.data.dataset import TwinDataset
from gluco_twin.models.fusion_net import FusionTwinNet
from gluco_twin.models.loss import MultiTaskTwinLoss

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_fusion_twin_net(epochs: int = 15, batch_size: int = 256, lr: float = 1e-3):
    """Train FusionTwinNet on train.npz and validate on val.npz."""
    device = get_device()
    print(f"=== Training FusionTwinNet on {device} ===")

    train_ds = TwinDataset(str(PROCESSED_DATA_DIR / "train.npz"), is_train=True, augment=True)
    val_ds = TwinDataset(str(PROCESSED_DATA_DIR / "val.npz"), is_train=False, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Class weighting
    pos_count = train_ds.events.sum().item()
    neg_count = len(train_ds) - pos_count
    pos_weight = neg_count / max(1.0, pos_count)
    print(f"Dataset sizes: Train={len(train_ds)} (pos_weight={pos_weight:.2f}), Val={len(val_ds)}")

    model = FusionTwinNet().to(device)
    criterion = MultiTaskTwinLoss(pos_weight=pos_weight, event_weight=1.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_score = -float("inf")
    best_metrics = {}
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_q_loss, total_bce = 0.0, 0.0, 0.0

        for batch in train_loader:
            dyn_x = batch["history"].to(device)
            cov_x = batch["covariates"].to(device)
            stat_x = batch["static"].to(device)
            targets = batch["targets"].to(device)
            events = batch["event"].to(device)

            optimizer.zero_grad()
            out = model(dyn_x, cov_x, stat_x)
            loss, q_loss, bce = criterion(out["quantiles"], out["event_logit"], targets, events)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_q_loss += q_loss.item()
            total_bce += bce.item()

        scheduler.step()
        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches
        avg_q_loss = total_q_loss / n_batches
        avg_bce = total_bce / n_batches

        # Validation
        model.eval()
        val_preds_p50 = []
        val_targets = []
        val_event_logits = []
        val_event_trues = []
        val_already_high = []

        with torch.no_grad():
            for batch in val_loader:
                dyn_x = batch["history"].to(device)
                cov_x = batch["covariates"].to(device)
                stat_x = batch["static"].to(device)
                targets = batch["targets"].to(device)
                events = batch["event"].to(device)

                out = model(dyn_x, cov_x, stat_x)
                p50 = out["quantiles"][:, :, 1]  # (B, 8)
                val_preds_p50.append(p50.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
                val_event_logits.append(out["event_logit"].squeeze(-1).cpu().numpy())
                val_event_trues.append(events.cpu().numpy())
                val_already_high.append(batch["already_high"].numpy())

        val_preds_p50 = np.concatenate(val_preds_p50, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        val_event_logits = np.concatenate(val_event_logits, axis=0)
        val_event_trues = np.concatenate(val_event_trues, axis=0)
        val_already_high = np.concatenate(val_already_high, axis=0)

        # Un-scaled RMSE (mg/dL)
        val_rmse_30 = np.sqrt(np.mean(((val_preds_p50[:, 1] - val_targets[:, 1]) * cfg.glucose_std) ** 2))
        val_rmse_60 = np.sqrt(np.mean(((val_preds_p50[:, 3] - val_targets[:, 3]) * cfg.glucose_std) ** 2))
        val_rmse_120 = np.sqrt(np.mean(((val_preds_p50[:, 7] - val_targets[:, 7]) * cfg.glucose_std) ** 2))

        # Event metrics on !already_high
        val_probs = 1.0 / (1.0 + np.exp(-val_event_logits))
        eval_mask = ~val_already_high
        val_auroc = roc_auc_score(val_event_trues[eval_mask], val_probs[eval_mask])
        val_auprc = average_precision_score(val_event_trues[eval_mask], val_probs[eval_mask])

        val_score = val_auprc - 0.01 * val_rmse_60
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:2d}/{epochs:2d} [{elapsed:.1f}s] - "
            f"Train Loss: {avg_loss:.4f} (Q: {avg_q_loss:.4f}, BCE: {avg_bce:.4f}) | "
            f"Val RMSE 30m: {val_rmse_30:.1f}, 60m: {val_rmse_60:.1f}, 120m: {val_rmse_120:.1f} | "
            f"AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            best_metrics = {
                "epoch": epoch,
                "val_auroc": float(val_auroc),
                "val_auprc": float(val_auprc),
                "val_rmse_30": float(val_rmse_30),
                "val_rmse_60": float(val_rmse_60),
                "val_rmse_120": float(val_rmse_120),
            }
            # Save checkpoint
            torch.save(model.state_dict(), CURRENT_MODEL_DIR / "model.pt")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (patience reached).")
                break

    print(f"Training completed. Best model saved to {CURRENT_MODEL_DIR / 'model.pt'}")
    return best_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train FusionTwinNet")
    parser.add_argument("--epochs", type=int, default=cfg.max_epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.learning_rate)
    args = parser.parse_args()

    train_fusion_twin_net(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
