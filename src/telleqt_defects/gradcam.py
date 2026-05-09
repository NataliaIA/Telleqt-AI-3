from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .data import (
    MultiViewDataset,
    Sample,
    filter_image_paths_by_views,
    find_test_samples,
    find_train_samples,
    natural_key,
    parse_views_arg,
    VIEW_DESCRIPTIONS,
)
from .model import MultiViewEfficientNet
from .transforms import build_transforms
from .utils import get_device, load_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanations for multi-view defect classifier")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-root", default=None)
    source.add_argument("--test-root", default=None)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--checkpoint", default=None, help="Specific checkpoint. Default: first fold_*.pt")
    parser.add_argument("--sample-id", default=None, help="Exact sample_id. For train it is group/good|bad/folder; for test it is folder name")
    parser.add_argument(
        "--from-oof",
        choices=["confident_bad", "false_positive", "false_negative"],
        default=None,
        help="Pick samples from oof_predictions.csv. Only for --train-root.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out-dir", default=None, help="Default: <model-dir>/gradcam")
    parser.add_argument("--views", default=None, help="Override views. Default: from checkpoint")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def safe_name(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("\\", "__")


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[MultiViewEfficientNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MultiViewEfficientNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def overlay_cam(original: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    original = original.convert("RGB")
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(original.size, resample=Image.BILINEAR)
    heat = plt.get_cmap("jet")(np.asarray(cam_img) / 255.0)[:, :, :3]
    heat_img = Image.fromarray(np.uint8(heat * 255)).convert("RGB")
    return Image.blend(original, heat_img, alpha=alpha)


def make_grid(items: list[tuple[str, Image.Image]], out_path: Path, title: str, thumb_size: int = 320) -> None:
    cells = []
    for label, image in items:
        image.thumbnail((thumb_size, thumb_size))
        canvas = Image.new("RGB", (thumb_size, thumb_size + 34), "white")
        x = (thumb_size - image.width) // 2
        canvas.paste(image, (x, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, thumb_size + 8), label[:44], fill="black")
        cells.append(canvas)

    w = thumb_size * max(1, len(cells))
    h = thumb_size + 76
    grid = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((8, 8), title, fill="black")
    for i, cell in enumerate(cells):
        grid.paste(cell, (i * thumb_size, 42))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, quality=92)


def gradcam_for_sample(
    model: MultiViewEfficientNet,
    sample: Sample,
    checkpoint: dict,
    device: torch.device,
    out_path: Path,
    view_ids: list[str],
) -> None:
    image_size = int(checkpoint.get("image_size", 384))
    max_views = int(checkpoint.get("max_views", len(view_ids)))
    transform = build_transforms(image_size, train=False)

    ds = MultiViewDataset([sample], transform=transform, return_label=False, max_views=max_views, view_ids=view_ids)
    views, sample_id = ds[0]
    views = views.unsqueeze(0).to(device)

    activations = []
    gradients = []

    def fwd_hook(_module, _inp, out):
        activations.append(out.detach())

    def bwd_hook(_module, _grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    target_layer = model.backbone.features[-1]
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)
    logit = model(views)[0]
    prob = torch.sigmoid(logit).item()
    # Positive class Grad-CAM: what increases probability of defect.
    logit.backward()

    h1.remove()
    h2.remove()

    if not activations or not gradients:
        raise RuntimeError("Could not collect activations/gradients for Grad-CAM")

    acts = activations[-1]  # [V, C, H, W]
    grads = gradients[-1]  # [V, C, H, W]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cams = (weights * acts).sum(dim=1)
    cams = F.relu(cams)
    cams = cams.detach().cpu().numpy()

    selected_paths = filter_image_paths_by_views(sample.image_paths, view_ids)[:max_views]
    while len(selected_paths) < max_views and selected_paths:
        selected_paths.append(selected_paths[-1])

    items = []
    for i, path in enumerate(selected_paths):
        cam = cams[i]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        with Image.open(path) as img:
            original = img.convert("RGB")
        overlay = overlay_cam(original, cam)
        prefix = path.name[:2]
        desc = VIEW_DESCRIPTIONS.get(prefix, prefix)
        items.append((f"{prefix} {desc}", overlay))

    make_grid(items, out_path, title=f"Grad-CAM | prob_bad={prob:.4f} | {sample_id}")


def choose_samples_from_oof(args: argparse.Namespace, samples: list[Sample], threshold: float) -> list[str]:
    run_dir = Path(args.model_dir)
    oof_path = run_dir / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"Not found: {oof_path}")
    df = pd.read_csv(oof_path)
    df["prediction"] = (df["prob_bad"] >= threshold).astype(int)
    if args.from_oof == "confident_bad":
        part = df.sort_values("prob_bad", ascending=False).head(args.top_k)
    elif args.from_oof == "false_positive":
        part = df[(df["label"] == 0) & (df["prediction"] == 1)].sort_values("prob_bad", ascending=False).head(args.top_k)
    elif args.from_oof == "false_negative":
        part = df[(df["label"] == 1) & (df["prediction"] == 0)].sort_values("prob_bad", ascending=True).head(args.top_k)
    else:
        raise ValueError("--from-oof was not provided")
    return [str(x) for x in part["sample_id"].tolist()]


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir) if args.out_dir else model_dir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else sorted(model_dir.glob("fold_*.pt"), key=lambda p: natural_key(p.name))[0]
    model, checkpoint = load_model(checkpoint_path, device)
    view_ids = parse_views_arg(args.views) if args.views is not None else list(checkpoint.get("view_ids", ["01", "02", "03", "04"]))

    if args.train_root:
        samples = find_train_samples(args.train_root)
    else:
        samples = find_test_samples(args.test_root)
    sample_map = {s.sample_id: s for s in samples}

    if args.from_oof:
        if not args.train_root:
            raise ValueError("--from-oof is available only with --train-root")
        threshold = load_threshold(model_dir, default=0.5)
        sample_ids = choose_samples_from_oof(args, samples, threshold)
    elif args.sample_id:
        sample_ids = [args.sample_id]
    else:
        raise ValueError("Provide --sample-id or --from-oof")

    for sample_id in sample_ids:
        sample = sample_map.get(sample_id)
        if sample is None:
            available_hint = list(sample_map.keys())[:5]
            raise KeyError(f"Sample not found: {sample_id}. Examples: {available_hint}")
        out_path = out_dir / f"{safe_name(sample_id)}.jpg"
        gradcam_for_sample(model, sample, checkpoint, device, out_path, view_ids)
        print("Saved", out_path)


if __name__ == "__main__":
    main()
