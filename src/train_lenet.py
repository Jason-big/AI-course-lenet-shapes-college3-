from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import LeNet5

CLASSES = ["circle", "square", "triangle", "cross", "star"]


class ShapesDataset(Dataset):
    def __init__(self, root: Path, split: str):
        xs, ys = [], []
        for class_id, cls in enumerate(CLASSES):
            for p in sorted((root / split / cls).glob("*.png")):
                img = Image.open(p).convert("L")
                x = np.asarray(img, dtype=np.float32) / 255.0
                x = (x - 0.5) / 0.5
                xs.append(x[None, :, :])
                ys.append(class_id)
        self.x = torch.tensor(np.stack(xs), dtype=torch.float32)
        self.y = torch.tensor(ys, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []
    probs = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(y.numpy().tolist())
            probs.extend(torch.softmax(logits, dim=1).cpu().numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    return {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/shapes32")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(2)
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / args.data_dir
    out_dir = project_root / "outputs"
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = ShapesDataset(data_root, "train")
    val_ds = ShapesDataset(data_root, "val")
    test_ds = ShapesDataset(data_root, "test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = LeNet5(num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    history = []
    best_val = -1.0
    best_state = None
    print(f"Device: {device}; train/val/test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
        }
        history.append(row)
        print(f"Epoch {epoch:02d}: loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_metrics['accuracy']:.4f}, val_f1={val_metrics['f1_macro']:.4f}")
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)

    torch.save(model.state_dict(), out_dir / "lenet5_shapes32_best.pt")
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(out_dir / "metrics_lenet.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in test_metrics.items() if k not in ["probs"]}, f, ensure_ascii=False, indent=2)
    print("\nTest classification report:")
    print(test_metrics["classification_report"])
    print("Confusion matrix:")
    print(np.array(test_metrics["confusion_matrix"]))


if __name__ == "__main__":
    main()
