from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from .data import find_train_samples, filter_image_paths_by_views, natural_key, parse_views_arg, VIEW_DESCRIPTIONS
from .utils import load_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save false positives / false negatives for visual error analysis")
    parser.add_argument("--train-root", required=True, help="Path to unpacked train directory")
    parser.add_argument("--run-dir", required=True, help="Directory with oof_predictions.csv and threshold.txt")
    parser.add_argument("--out-dir", default=None, help="Default: <run-dir>/errors")
    parser.add_argument("--threshold", type=float, default=None, help="Default: run threshold.txt or 0.5")
    parser.add_argument("--views", default="all", help="Which views to save in grids/copies")
    parser.add_argument("--max-per-type", type=int, default=50, help="Max FP and max FN grids to save")
    parser.add_argument("--copy-originals", action="store_true", help="Also copy original images into per-sample folders")
    return parser.parse_args()


def make_grid(image_paths: list[Path], out_path: Path, title: str, thumb_size: int = 320) -> None:
    images = []
    for path in image_paths:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((thumb_size, thumb_size))
            canvas = Image.new("RGB", (thumb_size, thumb_size + 34), "white")
            x = (thumb_size - img.width) // 2
            y = 0
            canvas.paste(img, (x, y))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, thumb_size + 8), path.name[:44], fill="black")
            images.append(canvas)

    if not images:
        return

    w = thumb_size * len(images)
    h = thumb_size + 74
    grid = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((8, 8), title, fill="black")
    for i, img in enumerate(images):
        grid.paste(img, (i * thumb_size, 40))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, quality=92)


def safe_name(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("\\", "__")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "errors"
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_path = run_dir / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"Not found: {oof_path}")

    threshold = args.threshold if args.threshold is not None else load_threshold(run_dir, default=0.5)
    view_ids = parse_views_arg(args.views)

    samples = find_train_samples(args.train_root)
    sample_map = {s.sample_id: s for s in samples}

    df = pd.read_csv(oof_path)
    df["prediction"] = (df["prob_bad"] >= threshold).astype(int)
    df["error_type"] = "correct"
    df.loc[(df["label"] == 0) & (df["prediction"] == 1), "error_type"] = "false_positive"
    df.loc[(df["label"] == 1) & (df["prediction"] == 0), "error_type"] = "false_negative"

    # Sort important errors first: confident false positives and confident false negatives.
    fp = df[df["error_type"] == "false_positive"].copy()
    fp["severity"] = fp["prob_bad"]
    fp = fp.sort_values("severity", ascending=False).head(args.max_per_type)

    fn = df[df["error_type"] == "false_negative"].copy()
    fn["severity"] = 1.0 - fn["prob_bad"]
    fn = fn.sort_values("severity", ascending=False).head(args.max_per_type)

    for error_name, part in [("false_positives", fp), ("false_negatives", fn)]:
        error_dir = out_dir / error_name
        error_dir.mkdir(parents=True, exist_ok=True)
        for _, row in part.iterrows():
            sample_id = str(row["sample_id"])
            sample = sample_map.get(sample_id)
            if sample is None:
                continue
            paths = filter_image_paths_by_views(sample.image_paths, view_ids)
            name = safe_name(sample_id)
            title = (
                f"{error_name} | label={int(row['label'])} pred={int(row['prediction'])} "
                f"prob_bad={float(row['prob_bad']):.4f} fold={int(row['fold'])}"
            )
            make_grid(paths, error_dir / f"{name}.jpg", title=title)

            if args.copy_originals:
                sample_out = error_dir / name
                sample_out.mkdir(parents=True, exist_ok=True)
                for p in paths:
                    shutil.copy2(p, sample_out / p.name)

    report_path = out_dir / "error_report.csv"
    df.to_csv(report_path, index=False)

    summary = {
        "threshold": threshold,
        "selected_views": view_ids,
        "selected_view_descriptions": {v: VIEW_DESCRIPTIONS[v] for v in view_ids},
        "false_positives": int((df["error_type"] == "false_positive").sum()),
        "false_negatives": int((df["error_type"] == "false_negative").sum()),
        "saved_false_positive_grids": int(len(fp)),
        "saved_false_negative_grids": int(len(fn)),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "error_summary.csv", index=False)

    print("Error analysis saved to", out_dir)
    print(summary)


if __name__ == "__main__":
    main()
