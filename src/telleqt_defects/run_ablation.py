from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_ABLATIONS = [
    ("01_front_barlight", "01"),
    ("02_front_toplight", "02"),
    ("03_back_barlight", "03"),
    ("04_back_toplight", "04"),
    ("front_01_02", "front"),
    ("back_03_04", "back"),
    ("barlight_01_03", "barlight"),
    ("toplight_02_04", "toplight"),
    ("all_01_02_03_04", "all"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run view ablation experiments")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--out-dir", default="runs/ablation")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--cv", choices=["group", "stratified"], default="group")
    parser.add_argument("--threshold-strategy", choices=["f1", "target_recall"], default="f1")
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--only", default=None, help="Comma-separated ablation names to run, e.g. front_01_02,all_01_02_03_04")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = DEFAULT_ABLATIONS
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        selected = [(name, views) for name, views in DEFAULT_ABLATIONS if name in wanted]
        unknown = wanted - {name for name, _ in DEFAULT_ABLATIONS}
        if unknown:
            raise ValueError(f"Unknown ablation names: {sorted(unknown)}")

    rows: list[dict] = []
    for name, views in selected:
        run_dir = out_dir / name
        metrics_path = run_dir / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            print(f"[skip] {name}: {metrics_path} already exists")
        else:
            cmd = [
                sys.executable,
                "-m",
                "telleqt_defects.train_cv",
                "--train-root",
                args.train_root,
                "--out-dir",
                str(run_dir),
                "--views",
                views,
                "--epochs",
                str(args.epochs),
                "--folds",
                str(args.folds),
                "--batch-size",
                str(args.batch_size),
                "--image-size",
                str(args.image_size),
                "--cv",
                args.cv,
                "--threshold-strategy",
                args.threshold_strategy,
                "--target-recall",
                str(args.target_recall),
                "--num-workers",
                str(args.num_workers),
            ]
            if args.device:
                cmd.extend(["--device", args.device])
            if args.no_pretrained:
                cmd.append("--no-pretrained")
            print("\n=== Running", name, "views=", views, "===")
            subprocess.run(cmd, check=True)

        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "experiment": name,
                    "views": views,
                    "threshold": metrics.get("threshold"),
                    "recall_bad": metrics.get("recall_bad"),
                    "false_positive_rate": metrics.get("false_positive_rate"),
                    "pr_auc": metrics.get("pr_auc"),
                    "tn": metrics.get("tn"),
                    "fp": metrics.get("fp"),
                    "fn": metrics.get("fn"),
                    "tp": metrics.get("tp"),
                    "run_dir": str(run_dir),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["pr_auc", "recall_bad"], ascending=False)
    summary_path = out_dir / "ablation_summary.csv"
    df.to_csv(summary_path, index=False)
    print("\nSaved ablation summary:", summary_path)
    if not df.empty:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
