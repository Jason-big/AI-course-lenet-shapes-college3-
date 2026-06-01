"""Generate a self-constructed synthetic grayscale shape dataset.
Dataset: Shape32-5, five classes: circle, square, triangle, cross, star.
The script creates train/val/test folders and a metadata CSV.
"""
from __future__ import annotations
import csv
import math
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASSES = ["circle", "square", "triangle", "cross", "star"]
SPLITS = {"train": 350, "val": 80, "test": 80}
IMAGE_SIZE = 32


def _star_points(cx: float, cy: float, outer: float, inner: float, n: int = 5) -> List[Tuple[float, float]]:
    pts = []
    start = -math.pi / 2
    for i in range(n * 2):
        r = outer if i % 2 == 0 else inner
        a = start + i * math.pi / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _draw_shape(draw: ImageDraw.ImageDraw, label: str, cx: int, cy: int, scale: float, width: int, fill: int) -> None:
    s = 10 * scale
    if label == "circle":
        box = [cx - s, cy - s, cx + s, cy + s]
        draw.ellipse(box, outline=fill, width=width)
        if random.random() < 0.10:
            inner = [cx - s * 0.55, cy - s * 0.55, cx + s * 0.55, cy + s * 0.55]
            draw.ellipse(inner, outline=fill, width=max(1, width - 1))
    elif label == "square":
        box = [cx - s, cy - s, cx + s, cy + s]
        draw.rectangle(box, outline=fill, width=width)
        if random.random() < 0.05:
            draw.line([cx - s, cy - s, cx + s, cy + s], fill=fill, width=max(1, width - 1))
    elif label == "triangle":
        pts = [(cx, cy - 1.15 * s), (cx - 1.08 * s, cy + 0.9 * s), (cx + 1.08 * s, cy + 0.9 * s)]
        draw.line(pts + [pts[0]], fill=fill, width=width, joint="curve")
    elif label == "cross":
        length = 1.25 * s
        draw.line([cx - length, cy, cx + length, cy], fill=fill, width=width + 1)
        draw.line([cx, cy - length, cx, cy + length], fill=fill, width=width + 1)
        if random.random() < 0.2:
            draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=fill)
    elif label == "star":
        pts = _star_points(cx, cy, 1.25 * s, 0.55 * s)
        draw.line(pts + [pts[0]], fill=fill, width=width, joint="curve")
    else:
        raise ValueError(label)


def make_image(label: str, split: str) -> Image.Image:
    # Test set is deliberately a little harder to better reflect generalization.
    harder = split == "test"
    bg = random.randint(226, 255)
    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=bg)
    # Draw on a larger canvas to reduce aliasing during rotation/cropping.
    big = Image.new("L", (64, 64), color=bg)
    draw = ImageDraw.Draw(big)
    cx = 32 + random.randint(-4, 4)
    cy = 32 + random.randint(-4, 4)
    scale = random.uniform(0.82, 1.14 if harder else 1.12)
    width = random.randint(2, 4)
    fill = random.randint(0, 55)
    _draw_shape(draw, label, cx, cy, scale, width, fill)

    # Extra random strokes make the task less trivial.
    if random.random() < (0.10 if harder else 0.06):
        for _ in range(random.randint(1, 2)):
            x1, y1 = random.randint(8, 56), random.randint(8, 56)
            x2, y2 = x1 + random.randint(-8, 8), y1 + random.randint(-8, 8)
            draw.line([x1, y1, x2, y2], fill=random.randint(70, 150), width=1)

    # Rotate and downsample.
    angle_range = 28 if harder else 25
    big = big.rotate(random.uniform(-angle_range, angle_range), resample=Image.BILINEAR, fillcolor=bg)
    img = big.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.LANCZOS)

    if random.random() < (0.35 if harder else 0.2):
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.7)))

    arr = np.asarray(img).astype(np.float32)
    noise_sigma = random.uniform(4.0, 13.5 if harder else 11.0)
    arr += np.random.normal(0, noise_sigma, arr.shape)
    if harder and random.random() < 0.03:
        # Mild occlusion, useful for error case analysis.
        ox, oy = random.randint(4, 24), random.randint(4, 24)
        arr[oy:oy + random.randint(2, 5), ox:ox + random.randint(3, 8)] = random.randint(205, 255)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def generate(root: Path, seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    out_dir = root / "shapes32"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    rows = []
    for split, n_per_class in SPLITS.items():
        for class_id, cls in enumerate(CLASSES):
            folder = out_dir / split / cls
            folder.mkdir(parents=True, exist_ok=True)
            for idx in range(n_per_class):
                img = make_image(cls, split)
                name = f"{cls}_{idx:04d}.png"
                path = folder / name
                img.save(path)
                rows.append({
                    "filepath": str(path.relative_to(root)).replace("\\", "/"),
                    "split": split,
                    "label": cls,
                    "class_id": class_id,
                    "source": "generated_by_generate_dataset.py",
                })
    with open(root / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "split", "label", "class_id", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} images at {out_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    generate(project_root / "data")
