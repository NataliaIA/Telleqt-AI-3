from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import MultiViewDataset, find_test_samples, natural_key, parse_views_arg, VIEW_DESCRIPTIONS
from .model import MultiViewEfficientNet
from .transforms import build_transforms
from .utils import get_device, load_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict test labels and create submission CSV")
    parser.add_argument("--test-root", required=True, help="Path to unpacked test directory")
    parser.add_argument("--model-dir", required=True, help="Directory with fold_*.pt checkpoints")
    parser.add_argument("--out-csv", default="submission.csv")
    parser.add_argument("--image-size", type=int, default=None, help="Override image size. Default: from checkpoint")
    parser.add_argument("--max-views", type=int, default=None, help="Override max views. Default: from checkpoint")
    parser.add_argument("--views", default=None, help="Override views. Default: from checkpoint. Examples: all, front, 01,02")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None, help="Default: threshold.txt or 0.5")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


@torch.no_grad()
def predict_checkpoint(checkpoint_path: Path, loader: DataLoader, device: torch.device) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MultiViewEfficientNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    probs: list[float] = []
    for views, _sample_ids in tqdm(loader, desc=f"predict {checkpoint_path.name}", leave=False):
        views = views.to(device, non_blocking=True)
        logits = model(views)
        batch_probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        probs.extend(batch_probs)
    return np.asarray(probs, dtype=float)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    checkpoints = sorted(model_dir.glob("fold_*.pt"), key=lambda p: natural_key(p.name))
    if not checkpoints:
        raise FileNotFoundError(f"No fold_*.pt checkpoints found in {model_dir}")

    first_checkpoint = torch.load(checkpoints[0], map_location="cpu")
    image_size = args.image_size or int(first_checkpoint.get("image_size", 384))
    checkpoint_views = first_checkpoint.get("view_ids", ["01", "02", "03", "04"])
    view_ids = parse_views_arg(args.views) if args.views is not None else list(checkpoint_views)
    max_views = args.max_views or int(first_checkpoint.get("max_views", len(view_ids)))
    threshold = args.threshold if args.threshold is not None else load_threshold(model_dir, default=0.5)

    device = get_device(args.device)
    samples = find_test_samples(args.test_root)
    print(f"Selected views: {view_ids} ({[VIEW_DESCRIPTIONS[v] for v in view_ids]})")
    print(f"Threshold: {threshold:.6f}")
    ds = MultiViewDataset(
        samples,
        transform=build_transforms(image_size, train=False),
        max_views=max_views,
        return_label=False,
        view_ids=view_ids,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    all_probs = []
    for checkpoint_path in checkpoints:
        all_probs.append(predict_checkpoint(checkpoint_path, loader, device))
    prob_bad = np.mean(np.stack(all_probs, axis=0), axis=0)
    prediction = (prob_bad >= threshold).astype(int)

    df = pd.DataFrame(
        {
            "sample_id": [s.sample_id for s in samples],
            "prediction": prediction,
        }
    )
    # Ensure stable natural ordering in the final CSV.
    df["_sort"] = df["sample_id"].map(natural_key)
    df = df.sort_values("_sort").drop(columns=["_sort"])

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved submission to {out_csv}")
    print(df.head())


if __name__ == "__main__":
    main()
