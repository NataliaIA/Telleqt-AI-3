from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    recall_score,
)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def choose_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds has length len(precision)-1. Ignore last PR point without threshold.
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    if len(f1) == 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1))])


def choose_threshold_by_target_recall(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.95) -> float:
    """Pick the highest-precision threshold among points with recall >= target_recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision = precision[:-1]
    recall = recall[:-1]
    if len(thresholds) == 0:
        return 0.5
    ok = np.where(recall >= target_recall)[0]
    if len(ok) == 0:
        # If the requested recall is unreachable, fall back to F1.
        return choose_threshold_by_f1(y_true, y_prob)
    best_local = ok[int(np.argmax(precision[ok]))]
    return float(thresholds[best_local])


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    fpr = fp / max(fp + tn, 1)
    precision, pr_recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(pr_recall, precision)
    return {
        "threshold": float(threshold),
        "confusion_matrix_labels": ["good_0", "bad_1"],
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "recall_bad": float(recall),
        "false_positive_rate": float(fpr),
        "pr_auc": float(pr_auc),
    }


def save_confusion_matrix_png(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, path: str | Path) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    im = ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["pred good", "pred bad"])
    ax.set_yticks([0, 1], labels=["true good", "true bad"])
    ax.set_title("Out-of-fold confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_pr_curve_png(y_true: np.ndarray, y_prob: np.ndarray, path: str | Path) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Out-of-fold PR curve")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return float(pr_auc)


def build_threshold_report(y_true: np.ndarray, y_prob: np.ndarray, target_recalls: tuple[float, ...] = (0.90, 0.95, 0.98)) -> list[dict]:
    """Return comparable operating points for README / production discussion."""
    rows: list[dict] = []

    candidates: list[tuple[str, float]] = [("fixed_0.50", 0.5), ("best_f1", choose_threshold_by_f1(y_true, y_prob))]
    for target in target_recalls:
        candidates.append((f"target_recall_{target:.2f}", choose_threshold_by_target_recall(y_true, y_prob, target)))

    seen: set[tuple[str, float]] = set()
    for name, threshold in candidates:
        key = (name, round(float(threshold), 8))
        if key in seen:
            continue
        seen.add(key)
        m = compute_binary_metrics(y_true, y_prob, float(threshold))
        rows.append({
            "mode": name,
            "threshold": m["threshold"],
            "tn": m["tn"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tp": m["tp"],
            "recall_bad": m["recall_bad"],
            "false_positive_rate": m["false_positive_rate"],
            "pr_auc": m["pr_auc"],
        })
    return rows


def save_threshold_report_csv(y_true: np.ndarray, y_prob: np.ndarray, path: str | Path) -> list[dict]:
    import pandas as pd

    rows = build_threshold_report(y_true, y_prob)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return rows
