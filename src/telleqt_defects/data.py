from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence
import re

from PIL import Image
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# In this task every sample has four views. Train filenames contain detailed
# camera/light names, while test filenames are anonymized as 01.jpg..04.jpg.
# This mapping keeps the same semantic order in train and test:
# 01 = front + bar light, 02 = front + top light,
# 03 = back + bar light,  04 = back + top light.
VIEW_PREFIX_TO_ORDER = {
    "01": 0,
    "02": 1,
    "03": 2,
    "04": 3,
}

VIEW_PRESETS = {
    "all": ["01", "02", "03", "04"],
    "front": ["01", "02"],
    "back": ["03", "04"],
    "barlight": ["01", "03"],
    "toplight": ["02", "04"],
    "front_barlight": ["01"],
    "front_toplight": ["02"],
    "back_barlight": ["03"],
    "back_toplight": ["04"],
}

VIEW_DESCRIPTIONS = {
    "01": "front_barlight",
    "02": "front_toplight",
    "03": "back_barlight",
    "04": "back_toplight",
}


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_paths: list[Path]
    label: int | None = None
    source_group: str | None = None


def natural_key(text: str) -> tuple:
    """Sort strings with numeric ids naturally: 2 before 10."""
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text))


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def image_view_prefix(path: Path) -> str | None:
    """Return 01/02/03/04 from train or test filename, if present."""
    match = re.match(r"^(\d{2})", path.name.lower())
    if match and match.group(1) in VIEW_PREFIX_TO_ORDER:
        return match.group(1)
    return None


def parse_views_arg(views: str | Sequence[str] | None) -> list[str]:
    """Parse --views.

    Accepted values:
      all, front, back, barlight, toplight,
      front_barlight, front_toplight, back_barlight, back_toplight,
      or comma-separated ids such as 01,02,03,04.
    """
    if views is None:
        return list(VIEW_PRESETS["all"])
    if isinstance(views, str):
        value = views.strip().lower()
        if value in VIEW_PRESETS:
            return list(VIEW_PRESETS[value])
        raw = [v.strip() for v in value.split(",") if v.strip()]
    else:
        raw = [str(v).strip().lower() for v in views if str(v).strip()]

    normalized: list[str] = []
    reverse_desc = {v: k for k, v in VIEW_DESCRIPTIONS.items()}
    for item in raw:
        if item in VIEW_PRESETS:
            normalized.extend(VIEW_PRESETS[item])
        elif item in VIEW_PREFIX_TO_ORDER:
            normalized.append(item)
        elif item in reverse_desc:
            normalized.append(reverse_desc[item])
        else:
            raise ValueError(f"Unknown view '{item}'. Use one of {sorted(VIEW_PRESETS)} or ids 01,02,03,04.")

    # Deduplicate and sort by semantic order.
    unique = sorted(set(normalized), key=lambda x: VIEW_PREFIX_TO_ORDER[x])
    if not unique:
        raise ValueError("At least one view must be selected")
    return unique


def view_sort_key(path: Path) -> tuple:
    """Stable view order for both train and test image names."""
    prefix = image_view_prefix(path)
    if prefix is not None:
        return (VIEW_PREFIX_TO_ORDER[prefix], natural_key(path.name.lower()))
    return (99, natural_key(path.name.lower()))


def list_image_paths(sample_dir: Path) -> list[Path]:
    paths = [p for p in sample_dir.iterdir() if _is_image(p)]
    return sorted(paths, key=view_sort_key)


def filter_image_paths_by_views(paths: Sequence[Path], view_ids: Sequence[str] | None) -> list[Path]:
    if not view_ids:
        return list(paths)
    allowed = set(view_ids)
    filtered = [p for p in paths if image_view_prefix(p) in allowed]
    return sorted(filtered, key=view_sort_key)


def find_train_samples(train_root: str | Path) -> list[Sample]:
    """Find samples in any nested directories containing good/bad folders.

    Expected real dataset structure from the inspection report:

        train/<collection_group>/good/<sample_folder>/01_...jpg ... 04_...jpg
        train/<collection_group>/bad/<sample_folder>/01_...jpg ... 04_...jpg

    The parent of the good/bad directory is stored as source_group. It is useful
    for GroupKFold / leave-one-collection-out validation, because the four train
    subfolders were collected at different times.
    """
    root = Path(train_root)
    if not root.exists():
        raise FileNotFoundError(f"Train root does not exist: {root}")

    samples: list[Sample] = []
    for class_dir in root.rglob("*"):
        if not class_dir.is_dir():
            continue
        name = class_dir.name.lower()
        if name not in {"good", "bad"}:
            continue

        label = 0 if name == "good" else 1
        source_group = class_dir.parent.name
        for sample_dir in sorted([p for p in class_dir.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name)):
            image_paths = list_image_paths(sample_dir)
            if not image_paths:
                continue
            # Prefix group and class to avoid collisions when different train parts contain same folder names.
            sample_id = f"{source_group}/{name}/{sample_dir.name}"
            samples.append(
                Sample(
                    sample_id=sample_id,
                    image_paths=image_paths,
                    label=label,
                    source_group=source_group,
                )
            )

    if not samples:
        raise RuntimeError(
            f"No train samples found under {root}. Expected nested good/bad folders with sample directories."
        )
    return samples


def find_test_samples(test_root: str | Path) -> list[Sample]:
    """Find anonymized test samples.

    Expected structure:

        test/<sample_id>/01.jpg ... 04.jpg

    sample_id in submission is exactly the folder name.
    """
    root = Path(test_root)
    if not root.exists():
        raise FileNotFoundError(f"Test root does not exist: {root}")

    samples: list[Sample] = []
    for sample_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name)):
        image_paths = list_image_paths(sample_dir)
        if not image_paths:
            continue
        samples.append(Sample(sample_id=sample_dir.name, image_paths=image_paths, label=None, source_group=None))

    if not samples:
        raise RuntimeError(f"No test samples found under {root}")
    return samples


class MultiViewDataset(Dataset):
    def __init__(
        self,
        samples: Iterable[Sample],
        transform: Callable | None = None,
        max_views: int | None = None,
        return_label: bool = True,
        view_ids: Sequence[str] | None = None,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.view_ids = list(view_ids) if view_ids is not None else None
        self.max_views = int(max_views) if max_views is not None else len(self.view_ids or VIEW_PRESETS["all"])
        self.return_label = return_label

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        paths = filter_image_paths_by_views(sample.image_paths, self.view_ids)
        paths = list(paths[: self.max_views])
        if len(paths) == 0:
            raise RuntimeError(f"Sample has no selected images: {sample.sample_id}, view_ids={self.view_ids}")

        # Pad by repeating the last selected image if there are fewer than max_views.
        while len(paths) < self.max_views:
            paths.append(paths[-1])

        images = []
        for path in paths:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                images.append(img)

        views = torch.stack(images, dim=0)  # [V, C, H, W]
        if self.return_label:
            if sample.label is None:
                raise ValueError("return_label=True but sample.label is None")
            return views, torch.tensor(sample.label, dtype=torch.float32), sample.sample_id
        return views, sample.sample_id
