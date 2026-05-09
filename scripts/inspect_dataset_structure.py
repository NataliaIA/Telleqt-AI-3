from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from collections import Counter

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def read_paths(source: Path):
    items = []
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source, "r") as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = info.filename.replace("\\", "/")
                if "__MACOSX" in name:
                    continue
                items.append({"path": name, "size_bytes": info.file_size})
    elif source.is_dir():
        for p in source.rglob("*"):
            if p.is_file():
                rel = p.relative_to(source).as_posix()
                if "__MACOSX" in rel:
                    continue
                items.append({"path": rel, "size_bytes": p.stat().st_size})
    else:
        raise ValueError(f"Source not found or unsupported: {source}")
    return items


def inspect_train(source: Path):
    files = read_paths(source)
    samples = {}
    ignored_images = []
    ignored_other = 0

    for item in files:
        path = item["path"]
        parts = Path(path).parts
        if not is_image(path):
            ignored_other += 1
            continue

        label = None
        label_index = None
        for cls in ["good", "bad"]:
            if cls in parts:
                label = cls
                label_index = parts.index(cls)
                break

        if label is None or label_index is None or label_index + 1 >= len(parts):
            ignored_images.append(path)
            continue

        sample_id = parts[label_index + 1]
        group_path = "/".join(parts[:label_index])
        key = f"{group_path}/{label}/{sample_id}"
        samples.setdefault(
            key,
            {
                "sample_key": key,
                "sample_id": sample_id,
                "label_name": label,
                "label": 0 if label == "good" else 1,
                "group_path": group_path,
                "images": [],
                "total_size_bytes": 0,
            },
        )
        samples[key]["images"].append(path)
        samples[key]["total_size_bytes"] += item["size_bytes"]

    sample_list = list(samples.values())
    label_counts = Counter(s["label_name"] for s in sample_list)
    image_count_distribution = Counter(len(s["images"]) for s in sample_list)
    groups = Counter(s["group_path"] for s in sample_list)

    examples = {"good": [], "bad": []}
    for s in sample_list:
        cls = s["label_name"]
        if len(examples[cls]) < 5:
            examples[cls].append(
                {"sample_key": s["sample_key"], "image_count": len(s["images"]), "images": [Path(x).name for x in s["images"]]}
            )

    report = {
        "source": str(source),
        "total_files": len(files),
        "total_image_files_used": sum(len(s["images"]) for s in sample_list),
        "total_samples": len(sample_list),
        "label_counts": dict(label_counts),
        "image_count_distribution_per_sample": dict(sorted(image_count_distribution.items())),
        "groups_or_subfolders": dict(groups),
        "ignored_non_image_files": ignored_other,
        "ignored_images_without_good_bad_structure_count": len(ignored_images),
        "ignored_images_without_good_bad_structure_examples": ignored_images[:20],
        "examples": examples,
    }
    return report, sample_list


def inspect_test(source: Path):
    files = read_paths(source)
    samples = {}
    ignored_other = 0
    ignored_images = []

    for item in files:
        path = item["path"]
        if not is_image(path):
            ignored_other += 1
            continue
        parts = Path(path).parts
        if len(parts) < 2:
            ignored_images.append(path)
            continue
        sample_id = parts[-2]
        samples.setdefault(sample_id, {"sample_id": sample_id, "images": [], "total_size_bytes": 0})
        samples[sample_id]["images"].append(path)
        samples[sample_id]["total_size_bytes"] += item["size_bytes"]

    sample_list = list(samples.values())
    image_count_distribution = Counter(len(s["images"]) for s in sample_list)
    examples = [
        {"sample_id": s["sample_id"], "image_count": len(s["images"]), "images": [Path(x).name for x in s["images"]]}
        for s in sample_list[:10]
    ]
    report = {
        "source": str(source),
        "total_files": len(files),
        "total_image_files_used": sum(len(s["images"]) for s in sample_list),
        "total_samples": len(sample_list),
        "image_count_distribution_per_sample": dict(sorted(image_count_distribution.items())),
        "ignored_non_image_files": ignored_other,
        "ignored_images_without_sample_folder_count": len(ignored_images),
        "ignored_images_without_sample_folder_examples": ignored_images[:20],
        "examples": examples,
    }
    return report, sample_list


def write_train_csv(path: Path, samples):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_key", "sample_id", "label_name", "label", "group_path", "image_count", "images", "total_size_bytes"])
        for s in samples:
            writer.writerow(
                [s["sample_key"], s["sample_id"], s["label_name"], s["label"], s["group_path"], len(s["images"]), " | ".join(s["images"]), s["total_size_bytes"]]
            )


def write_test_csv(path: Path, samples):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "image_count", "images", "total_size_bytes"])
        for s in samples:
            writer.writerow([s["sample_id"], len(s["images"]), " | ".join(s["images"]), s["total_size_bytes"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train.zip or unpacked train directory")
    parser.add_argument("--test", required=True, help="Path to test.zip or unpacked test directory")
    parser.add_argument("--out-dir", default="dataset_report")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_report, train_samples = inspect_train(Path(args.train))
    test_report, test_samples = inspect_test(Path(args.test))

    with (out_dir / "dataset_structure_report.json").open("w", encoding="utf-8") as f:
        json.dump({"train": train_report, "test": test_report}, f, ensure_ascii=False, indent=2)
    write_train_csv(out_dir / "train_samples.csv", train_samples)
    write_test_csv(out_dir / "test_samples.csv", test_samples)

    print(f"Saved report to {out_dir}")
    print(json.dumps({"train": train_report, "test": test_report}, ensure_ascii=False, indent=2)[:5000])


if __name__ == "__main__":
    main()
