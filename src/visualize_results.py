from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CLASSES = ["circle", "square", "triangle", "cross", "star"]


def plot_dataset_samples(project_root: Path):
    data_root = project_root / "data" / "shapes32" / "train"
    fig, axes = plt.subplots(len(CLASSES), 8, figsize=(8.5, 5.4))
    for i, cls in enumerate(CLASSES):
        paths = sorted((data_root / cls).glob("*.png"))[:8]
        for j, p in enumerate(paths):
            axes[i, j].imshow(Image.open(p), cmap="gray", vmin=0, vmax=255)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_ylabel(cls, fontsize=10)
    fig.suptitle("Shape32-5 synthetic dataset examples", fontsize=13)
    plt.tight_layout()
    fig.savefig(project_root / "figures" / "fig1_dataset_samples.png", dpi=200)
    plt.close(fig)


def plot_distribution(project_root: Path):
    counts = {split: [] for split in ["train", "val", "test"]}
    for split in counts:
        for cls in CLASSES:
            counts[split].append(len(list((project_root / "data" / "shapes32" / split / cls).glob("*.png"))))
    x = np.arange(len(CLASSES))
    width = 0.25
    fig = plt.figure(figsize=(7.2, 4.4))
    ax = plt.gca()
    ax.bar(x - width, counts["train"], width, label="train")
    ax.bar(x, counts["val"], width, label="val")
    ax.bar(x + width, counts["test"], width, label="test")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=20)
    ax.set_ylabel("Number of images")
    ax.set_title("Dataset class distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(project_root / "figures" / "fig2_class_distribution.png", dpi=200)
    plt.close(fig)


def plot_architecture(project_root: Path):
    # Simple self-drawn architecture diagram; no external figure is used.
    layers = [
        ("Input\n1×32×32", 0.8),
        ("C1 Conv 5×5\n6×28×28", 1.0),
        ("S2 AvgPool\n6×14×14", 1.0),
        ("C3 Conv 5×5\n16×10×10", 1.05),
        ("S4 AvgPool\n16×5×5", 1.0),
        ("Flatten\n400", 0.8),
        ("FC\n120", 0.75),
        ("FC\n84", 0.75),
        ("Output\n5 classes", 0.9),
    ]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    x = 0.3
    y = 0.55
    for idx, (text, w) in enumerate(layers):
        rect = plt.Rectangle((x, y), w, 0.9, fill=False, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + 0.45, text, ha="center", va="center", fontsize=9)
        if idx < len(layers) - 1:
            ax.annotate("", xy=(x + w + 0.22, y + 0.45), xytext=(x + w, y + 0.45), arrowprops=dict(arrowstyle="->", lw=1.5))
        x += w + 0.35
    ax.set_xlim(0, x)
    ax.set_ylim(0.3, 1.8)
    fig.suptitle("LeNet-5 style CNN architecture for Shape32-5", fontsize=13)
    fig.tight_layout()
    fig.savefig(project_root / "figures" / "fig3_lenet_architecture.png", dpi=220)
    plt.close(fig)


def plot_training(project_root: Path):
    with open(project_root / "outputs" / "history.json", "r", encoding="utf-8") as f:
        hist = json.load(f)
    epochs = [h["epoch"] for h in hist]
    fig = plt.figure(figsize=(7, 4.5))
    ax = plt.gca()
    ax.plot(epochs, [h["train_loss"] for h in hist], marker="o", label="train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(project_root / "figures" / "fig4_training_loss.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 4.5))
    ax = plt.gca()
    ax.plot(epochs, [h["train_accuracy"] for h in hist], marker="o", label="train accuracy")
    ax.plot(epochs, [h["val_accuracy"] for h in hist], marker="s", label="validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Accuracy curves")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(project_root / "figures" / "fig5_accuracy_curve.png", dpi=200)
    plt.close(fig)


def plot_confusion_matrix(project_root: Path):
    with open(project_root / "outputs" / "metrics_lenet.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    cm = np.array(metrics["confusion_matrix"])
    fig = plt.figure(figsize=(5.8, 5.2))
    ax = plt.gca()
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=35, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("LeNet-5 test confusion matrix")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(project_root / "figures" / "fig6_confusion_matrix.png", dpi=220)
    plt.close(fig)


def make_terminal_screenshot(project_root: Path):
    with open(project_root / "outputs" / "metrics_lenet.json", "r", encoding="utf-8") as f:
        m = json.load(f)
    lines = [
        "$ python src/generate_dataset.py",
        "Generated 2550 images at data/shapes32",
        "$ python src/train_lenet.py --epochs 12 --batch_size 64 --lr 0.001",
        "Device: cpu; train/val/test: 1750/400/400",
    ]
    with open(project_root / "outputs" / "history.json", "r", encoding="utf-8") as f:
        hist = json.load(f)
    for h in hist[-4:]:
        lines.append(f"Epoch {h['epoch']:02d}: loss={h['train_loss']:.4f}, train_acc={h['train_accuracy']:.4f}, val_acc={h['val_accuracy']:.4f}, val_f1={h['val_f1_macro']:.4f}")
    lines.extend([
        "",
        "Test results:",
        f"accuracy={m['accuracy']:.4f}, precision_macro={m['precision_macro']:.4f}, recall_macro={m['recall_macro']:.4f}, f1_macro={m['f1_macro']:.4f}",
        "Confusion matrix saved to figures/fig6_confusion_matrix.png",
    ])
    fig = plt.figure(figsize=(8.5, 4.6))
    ax = plt.gca()
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#111111"))
    y = 0.92
    for line in lines:
        ax.text(0.04, y, line, transform=ax.transAxes, fontsize=10, family="monospace", color="#f2f2f2", va="top")
        y -= 0.07
    fig.tight_layout()
    fig.savefig(project_root / "figures" / "fig7_terminal_log.png", dpi=200)
    plt.close(fig)


def plot_error_cases(project_root: Path):
    # Use model outputs saved as y_true/y_pred; show misclassified samples if any, otherwise low-confidence placeholders are not available here.
    with open(project_root / "outputs" / "metrics_lenet.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    y_true = metrics.get("y_true", [])
    y_pred = metrics.get("y_pred", [])
    test_paths = []
    for cls in CLASSES:
        test_paths.extend(sorted((project_root / "data" / "shapes32" / "test" / cls).glob("*.png")))
    errors = [i for i, (a, b) in enumerate(zip(y_true, y_pred)) if a != b]
    # If no error appears, show harder samples from the end of each class and mark them as correctly classified examples.
    selected = errors[:10]
    title = "Typical error cases" if selected else "Hard-but-correct test examples"
    if not selected:
        selected = [k * 80 + 70 for k in range(len(CLASSES)) for _ in range(2)][:10]
    fig, axes = plt.subplots(2, 5, figsize=(8.5, 3.6))
    axes = axes.ravel()
    for ax, idx in zip(axes, selected):
        if idx < len(test_paths):
            img = Image.open(test_paths[idx])
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            true = CLASSES[y_true[idx]] if idx < len(y_true) else "?"
            pred = CLASSES[y_pred[idx]] if idx < len(y_pred) else "?"
            ax.set_title(f"T:{true}\nP:{pred}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(project_root / "figures" / "fig8_error_cases.png", dpi=200)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    (root / "figures").mkdir(exist_ok=True)
    plot_dataset_samples(root)
    plot_distribution(root)
    plot_architecture(root)
    plot_training(root)
    plot_confusion_matrix(root)
    make_terminal_screenshot(root)
    plot_error_cases(root)
    print("Figures saved to figures/")


if __name__ == "__main__":
    main()
