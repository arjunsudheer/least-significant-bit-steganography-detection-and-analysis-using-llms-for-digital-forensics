import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    balanced_accuracy_score,
    average_precision_score,
)

from binary_classifier.model import SteganalysisNet
from binary_classifier.data_loader import (
    get_train_val_samples,
    get_test_samples,
    BinaryStegoDataset,
)


# CONFIG
ARGS = {
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 10,
    "batch_size": 8,
    "accum_steps": 16,
    "grad_clip": 1.0,
    "focal_gamma": 2.0,
    "n_folds": 5,
    "num_workers": 4,
    "dataset": "dataset",
    "out": "artifacts/results",
    "random_state": 42,
}


class FocalLoss(nn.Module):
    """
    Binary Focal Loss:  FL(p_t) = -(1 - p_t)^γ · log(p_t)

    Operates on raw logits (numerically stable via BCE-with-logits).
    No alpha / pos_weight term is used here because WeightedRandomSampler
    already ensures balanced mini-batches.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        return ((1.0 - p_t).pow(self.gamma) * bce).mean()


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0.0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0.0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0.0)),
        "roc_auc": float(auc(fpr, tpr)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
    }


def save_plots(y_true, y_prob, output_dir: str, prefix: str = "model") -> dict:
    os.makedirs(output_dir, exist_ok=True)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Clean", "Stego"],
        yticklabels=["Clean", "Stego"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{prefix} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_cm.png"), dpi=150)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc_val:.4f}")
    plt.plot([0, 1], [0, 1], "navy", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} — ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_roc.png"), dpi=150)
    plt.close()

    # Precision-Recall curve
    prec_pts, rec_pts, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(rec_pts, prec_pts, lw=2, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{prefix} — Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_pr.png"), dpi=150)
    plt.close()

    return compute_metrics(y_true.tolist(), y_prob.tolist())


def make_weighted_sampler(labels: list) -> WeightedRandomSampler:
    """
    Returns a WeightedRandomSampler that draws ≈50/50 clean/stego per
    mini-batch by assigning each sample a weight of 1/class_count.
    replacement=True allows the minority class to be drawn more than once.
    """
    labels_arr = np.array(labels)
    class_counts = np.bincount(labels_arr)
    weights = (1.0 / class_counts)[labels_arr]
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(weights),
        replacement=True,
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    accum_steps,
    grad_clip,
    fold,
    epoch,
    total_epochs,
) -> float:
    """
    One training epoch with gradient accumulation + AMP + gradient clipping.
    """
    model.train()
    optimizer.zero_grad()

    running_loss = 0.0
    n_batches = len(loader)
    pbar = tqdm(
        loader, desc=f"Fold {fold} | Ep {epoch:02d}/{total_epochs}", leave=False
    )

    for i, (images, labels, _) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        with autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, labels) / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
            scaler.unscale_(optimizer)  # must precede clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps
        pbar.set_postfix(loss=f"{running_loss / (i + 1):.4f}")

    return running_loss / n_batches


@torch.no_grad()
def run_inference(model, loader, criterion, device):
    """
    Full-dataset inference (validation or test).

    Returns:
        avg_loss  float
        y_true    list[int]    ground-truth labels
        y_prob    list[float]  sigmoid probabilities
    """
    model.eval()
    total_loss = 0.0
    y_true, y_prob = [], []

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        with autocast(device_type=device.type):
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()

        y_prob.extend(torch.sigmoid(outputs).squeeze(1).cpu().tolist())
        y_true.extend(labels.squeeze(1).cpu().tolist())

    return total_loss / len(loader), y_true, y_prob


def main():
    os.makedirs(ARGS["out"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {props.name}  |  VRAM: {props.total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True

    effective_batch = ARGS["batch_size"] * ARGS["accum_steps"]
    print(
        f'Batch  : {ARGS["batch_size"]} × {ARGS["accum_steps"]} accum = {effective_batch} effective'
    )

    # Report parameter count (SRM buffers are frozen — not counted)
    _tmp = SteganalysisNet()
    n_params = sum(p.numel() for p in _tmp.parameters() if p.requires_grad)
    print(f"Params : {n_params:,} learnable")
    del _tmp

    # Dataset
    train_val_samples = get_train_val_samples(ARGS["dataset"])
    test_samples = get_test_samples(ARGS["dataset"])

    def _counts(s):
        return sum(l == 0 for _, l in s), sum(l == 1 for _, l in s)

    n0tv, n1tv = _counts(train_val_samples)
    n0ts, n1ts = _counts(test_samples)
    print(f"\nTrain+Val : {len(train_val_samples):,}  (clean={n0tv:,}, stego={n1tv:,})")
    print(f"Test      : {len(test_samples):,}  (clean={n0ts:,}, stego={n1ts:,})")

    X = np.array(train_val_samples, dtype=object)
    y = np.array([s[1] for s in train_val_samples])

    skf = StratifiedKFold(
        n_splits=ARGS["n_folds"], shuffle=True, random_state=ARGS["random_state"]
    )

    # Criterion is shared across folds (stateless)
    criterion = FocalLoss(gamma=ARGS["focal_gamma"])

    global_best_val_loss = float("inf")
    best_model_path = os.path.join(ARGS["out"], "best_model.pt")
    fold_metrics_list = []

    # Stratified K-Fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f'\n{"=" * 60}')
        print(f'  FOLD {fold}/{ARGS["n_folds"]}')
        print(f'{"=" * 60}')

        train_samples = X[train_idx].tolist()
        val_samples = X[val_idx].tolist()
        train_labels = [s[1] for s in train_samples]
        n0f, n1f = train_labels.count(0), train_labels.count(1)
        print(f"  clean={n0f:,} | stego={n1f:,}")

        # WeightedRandomSampler: ≈50/50 per mini-batch
        sampler = make_weighted_sampler(train_labels)

        train_loader = DataLoader(
            BinaryStegoDataset(train_samples, augment=True),
            batch_size=ARGS["batch_size"],
            sampler=sampler,
            num_workers=ARGS["num_workers"],
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            BinaryStegoDataset(val_samples, augment=False),
            batch_size=ARGS["batch_size"],
            shuffle=False,
            num_workers=ARGS["num_workers"],
            pin_memory=(device.type == "cuda"),
        )

        model = SteganalysisNet().to(device)
        optimizer = AdamW(
            model.parameters(), lr=ARGS["lr"], weight_decay=ARGS["weight_decay"]
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=ARGS["epochs"], eta_min=1e-6)
        scaler = GradScaler()

        fold_best_val_loss = float("inf")
        fold_best_val_metrics = None

        for epoch in range(1, ARGS["epochs"] + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                scaler,
                device,
                ARGS["accum_steps"],
                ARGS["grad_clip"],
                fold,
                epoch,
                ARGS["epochs"],
            )
            val_loss, y_true_val, y_prob_val = run_inference(
                model, val_loader, criterion, device
            )
            scheduler.step()

            val_metrics = compute_metrics(y_true_val, y_prob_val)
            lr_now = optimizer.param_groups[0]["lr"]

            print(
                f'  Ep {epoch:02d}/{ARGS["epochs"]} | '
                f"train={train_loss:.4f} | "
                f"val={val_loss:.4f} | "
                f'acc={val_metrics["accuracy"]:.4f} | '
                f'bal={val_metrics["balanced_acc"]:.4f} | '
                f'f1={val_metrics["f1"]:.4f} | '
                f'auc={val_metrics["roc_auc"]:.4f} | '
                f"lr={lr_now:.2e}"
            )

            if val_loss < fold_best_val_loss:
                fold_best_val_loss = val_loss
                fold_best_val_metrics = val_metrics

            if val_loss < global_best_val_loss:
                global_best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"  --> NEW GLOBAL BEST  (val_loss={val_loss:.4f})")

        fold_metrics_list.append(
            {
                "fold": fold,
                "best_val_loss": fold_best_val_loss,
                **{f"val_{k}": v for k, v in fold_best_val_metrics.items()},
            }
        )
        print(f"\n  Fold {fold} best → {fold_best_val_metrics}")

    print(f'\n{"=" * 60}')
    print("  CROSS-VALIDATION SUMMARY")
    print(f'{"=" * 60}')
    cv_acc = [m["val_accuracy"] for m in fold_metrics_list]
    cv_bal = [m["val_balanced_acc"] for m in fold_metrics_list]
    cv_f1 = [m["val_f1"] for m in fold_metrics_list]
    cv_auc = [m["val_roc_auc"] for m in fold_metrics_list]
    print(f"  Accuracy (balanced) : {np.mean(cv_bal):.4f} ± {np.std(cv_bal):.4f}")
    print(f"  Accuracy            : {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}")
    print(f"  F1-score            : {np.mean(cv_f1):.4f}  ± {np.std(cv_f1):.4f}")
    print(f"  ROC-AUC             : {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}")

    with open(os.path.join(ARGS["out"], "fold_metrics.json"), "w") as fh:
        json.dump(fold_metrics_list, fh, indent=4)

    # Test evaluation
    print(f'\n{"=" * 60}')
    print("  FINAL TEST SET EVALUATION  (best global checkpoint)")
    print(f'{"=" * 60}')

    test_loader = DataLoader(
        BinaryStegoDataset(test_samples, augment=False),
        batch_size=ARGS["batch_size"],
        shuffle=False,
        num_workers=ARGS["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    best_model = SteganalysisNet().to(device)
    best_model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)
    )

    test_loss, y_true_test, y_prob_test = run_inference(
        best_model, test_loader, criterion, device
    )
    test_metrics = save_plots(y_true_test, y_prob_test, ARGS["out"], prefix="test")
    test_metrics["loss"] = test_loss

    print(f"\n  Test results:")
    for k, v in test_metrics.items():
        print(f"    {k:<18}: {v:.4f}")

    # Save summary
    summary = {
        "config": ARGS,
        "cv_summary": {
            "mean_balanced_acc": float(np.mean(cv_bal)),
            "std_balanced_acc": float(np.std(cv_bal)),
            "mean_accuracy": float(np.mean(cv_acc)),
            "std_accuracy": float(np.std(cv_acc)),
            "mean_f1": float(np.mean(cv_f1)),
            "std_f1": float(np.std(cv_f1)),
            "mean_roc_auc": float(np.mean(cv_auc)),
            "std_roc_auc": float(np.std(cv_auc)),
        },
        "test_metrics": test_metrics,
        "fold_metrics": fold_metrics_list,
    }
    with open(os.path.join(ARGS["out"], "final_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=4)

    print(f'\nAll outputs saved to: {ARGS["out"]}')
    print("Training complete.")


if __name__ == "__main__":
    main()
