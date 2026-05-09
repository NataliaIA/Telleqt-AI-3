from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover - older sklearn fallback
    StratifiedGroupKFold = None
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import MultiViewDataset, find_train_samples, parse_views_arg, VIEW_DESCRIPTIONS
from .metrics import (
    choose_threshold_by_f1,
    choose_threshold_by_target_recall,
    compute_binary_metrics,
    save_confusion_matrix_png,
    save_pr_curve_png,
    save_threshold_report_csv,
)
from .model import MultiViewEfficientNet
from .transforms import build_transforms
from .utils import get_device, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-view defect classifier with cross-validation")
    parser.add_argument("--train-root", required=True, help="Path to unpacked train directory")
    parser.add_argument("--out-dir", default="runs/effnet_b0", help="Directory for checkpoints and metrics")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--max-views", type=int, default=None, help="Default: number of selected --views")
    parser.add_argument("--views", default="all", help="View preset or comma-separated ids. Examples: all, front, back, barlight, toplight, 01,02")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda, mps, cpu. Default: auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--cv",
        choices=["group", "stratified"],
        default="group",
        help=(
            "group = split by source train subfolder / collection session; "
            "stratified = random StratifiedKFold by samples. For this dataset group is more honest."
        ),
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=["f1", "target_recall"],
        default="f1",
        help="How to choose threshold from out-of-fold predictions.",
    )
    parser.add_argument("--target-recall", type=float, default=0.95, help="Used only with --threshold-strategy target_recall")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_logits: list[float] = []
    all_labels: list[int] = []
    all_ids: list[str] = []
    for views, labels, sample_ids in loader:
        views = views.to(device, non_blocking=True)
        logits = model(views)
        all_logits.extend(logits.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())
        all_ids.extend(list(sample_ids))
    probs = 1.0 / (1.0 + np.exp(-np.asarray(all_logits)))
    return probs, np.asarray(all_labels, dtype=int), all_ids


def train_one_fold(
    fold: int,
    train_samples,
    val_samples,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_ds = MultiViewDataset(
        train_samples,
        transform=build_transforms(args.image_size, train=True),
        max_views=args.max_views,
        return_label=True,
        view_ids=args.view_ids,
    )
    val_ds = MultiViewDataset(
        val_samples,
        transform=build_transforms(args.image_size, train=False),
        max_views=args.max_views,
        return_label=True,
        view_ids=args.view_ids,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = MultiViewEfficientNet(pretrained=not args.no_pretrained).to(device)

    labels = np.array([s.label for s in train_samples], dtype=np.float32)
    pos_count = max(float(labels.sum()), 1.0)
    neg_count = max(float(len(labels) - labels.sum()), 1.0)
    pos_weight = torch.tensor([neg_count / pos_count], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_state = None
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"fold {fold} epoch {epoch}/{args.epochs}", leave=False)
        for views, labels_tensor, _ in pbar:
            views = views.to(device, non_blocking=True)
            labels_tensor = labels_tensor.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(views)
            loss = criterion(logits, labels_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * views.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        val_probs, val_labels, _ = evaluate(model, val_loader, device)
        eps = 1e-7
        val_loss = -np.mean(
            val_labels * np.log(np.clip(val_probs, eps, 1 - eps))
            + (1 - val_labels) * np.log(np.clip(1 - val_probs, eps, 1 - eps))
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"fold={fold} epoch={epoch} "
            f"train_loss={running_loss / max(len(train_ds), 1):.4f} val_loss={val_loss:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = out_dir / f"fold_{fold}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "image_size": args.image_size,
            "max_views": args.max_views,
            "view_ids": args.view_ids,
            "view_descriptions": {k: VIEW_DESCRIPTIONS[k] for k in args.view_ids},
            "fold": fold,
        },
        checkpoint_path,
    )
    print(f"saved {checkpoint_path}")

    return evaluate(model, val_loader, device)


def build_splits(samples, y: np.ndarray, groups: np.ndarray, args: argparse.Namespace):
    if args.cv == "group":
        unique_groups = sorted(set(groups.tolist()))
        n_splits = min(args.folds, len(unique_groups))
        if n_splits < 2:
            raise ValueError("Need at least two source groups for group CV")
        if n_splits != args.folds:
            print(f"Requested {args.folds} folds, but only {len(unique_groups)} source groups exist. Using {n_splits} folds.")
        if StratifiedGroupKFold is not None:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
            return list(splitter.split(np.zeros(len(samples)), y, groups=groups)), n_splits, "StratifiedGroupKFold"
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(np.zeros(len(samples)), y, groups=groups)), n_splits, "GroupKFold"

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    return list(splitter.split(np.zeros(len(samples)), y)), args.folds, "StratifiedKFold"


def make_dataset_summary(samples, y: np.ndarray, groups: np.ndarray) -> dict:
    by_group = defaultdict(lambda: {"total": 0, "good_0": 0, "bad_1": 0})
    for s in samples:
        g = s.source_group or "unknown"
        by_group[g]["total"] += 1
        if s.label == 0:
            by_group[g]["good_0"] += 1
        elif s.label == 1:
            by_group[g]["bad_1"] += 1
    return {
        "total_samples": int(len(samples)),
        "label_counts": {"good_0": int((y == 0).sum()), "bad_1": int((y == 1).sum())},
        "image_count_distribution": dict(Counter(len(s.image_paths) for s in samples)),
        "source_groups": dict(sorted(by_group.items())),
    }


def main() -> None:
    args = parse_args()
    args.view_ids = parse_views_arg(args.views)
    args.max_views = int(args.max_views or len(args.view_ids))
    seed_everything(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = find_train_samples(args.train_root)
    y = np.asarray([s.label for s in samples], dtype=int)
    groups = np.asarray([s.source_group for s in samples])

    summary = make_dataset_summary(samples, y, groups)
    summary["selected_views"] = args.view_ids
    summary["selected_view_descriptions"] = {k: VIEW_DESCRIPTIONS[k] for k in args.view_ids}
    summary["max_views"] = args.max_views
    save_json(summary, out_dir / "dataset_summary.json")

    print(f"Found {len(samples)} train samples: good={(y == 0).sum()}, bad={(y == 1).sum()}")
    print(f"Source groups: {sorted(set(groups.tolist()))}")
    print(f"Using device: {device}")
    print(f"Selected views: {args.view_ids} ({[VIEW_DESCRIPTIONS[v] for v in args.view_ids]})")

    splits, n_splits, splitter_name = build_splits(samples, y, groups, args)
    print(f"CV splitter: {splitter_name}, folds={n_splits}")

    fold_rows = []
    for fold, (_train_idx, val_idx) in enumerate(splits):
        labels = y[val_idx]
        val_groups = sorted(set(groups[val_idx].tolist()))
        fold_rows.append(
            {
                "fold": fold,
                "val_size": int(len(val_idx)),
                "val_good_0": int((labels == 0).sum()),
                "val_bad_1": int((labels == 1).sum()),
                "val_source_groups": " | ".join(val_groups),
            }
        )
    pd.DataFrame(fold_rows).to_csv(out_dir / "folds.csv", index=False)
    print(pd.DataFrame(fold_rows).to_string(index=False))

    oof_records = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        val_probs, val_labels, val_ids = train_one_fold(fold, train_samples, val_samples, args, device, out_dir)
        for sample_id, label, prob in zip(val_ids, val_labels.tolist(), val_probs.tolist()):
            oof_records.append({"sample_id": sample_id, "label": int(label), "prob_bad": float(prob), "fold": fold})

    oof_df = pd.DataFrame(oof_records)
    oof_path = out_dir / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)

    y_true = oof_df["label"].to_numpy(dtype=int)
    y_prob = oof_df["prob_bad"].to_numpy(dtype=float)
    if args.threshold_strategy == "target_recall":
        threshold = choose_threshold_by_target_recall(y_true, y_prob, target_recall=args.target_recall)
    else:
        threshold = choose_threshold_by_f1(y_true, y_prob)
    (out_dir / "threshold.txt").write_text(f"{threshold:.8f}\n", encoding="utf-8")

    metrics = compute_binary_metrics(y_true, y_prob, threshold)
    metrics["cv_splitter"] = splitter_name
    metrics["folds"] = int(n_splits)
    metrics["threshold_strategy"] = args.threshold_strategy
    metrics["selected_views"] = args.view_ids
    metrics["selected_view_descriptions"] = {k: VIEW_DESCRIPTIONS[k] for k in args.view_ids}
    if args.threshold_strategy == "target_recall":
        metrics["target_recall"] = float(args.target_recall)
    save_json(metrics, out_dir / "metrics.json")
    save_confusion_matrix_png(y_true, y_prob, threshold, out_dir / "confusion_matrix.png")
    save_pr_curve_png(y_true, y_prob, out_dir / "pr_curve.png")
    threshold_rows = save_threshold_report_csv(y_true, y_prob, out_dir / "threshold_report.csv")
    save_json({"operating_points": threshold_rows}, out_dir / "threshold_report.json")

    print("\nOOF metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"\nSaved metrics and plots to {out_dir}")


if __name__ == "__main__":
    main()
