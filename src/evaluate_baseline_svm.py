from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSES = ["circle", "square", "triangle", "cross", "star"]


def load_split(root: Path, split: str):
    xs, ys = [], []
    for class_id, cls in enumerate(CLASSES):
        for p in sorted((root / split / cls).glob("*.png")):
            img = Image.open(p).convert("L")
            xs.append(np.asarray(img, dtype=np.float32).reshape(-1) / 255.0)
            ys.append(class_id)
    return np.vstack(xs), np.array(ys)


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "shapes32"
    out_dir = project_root / "outputs"
    out_dir.mkdir(exist_ok=True)
    x_train, y_train = load_split(data_root, "train")
    x_test, y_test = load_split(data_root, "test")
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale"))
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    acc = accuracy_score(y_test, pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=list(range(len(CLASSES))))
    result = {
        "accuracy": float(acc),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, pred, target_names=CLASSES, digits=4, zero_division=0),
    }
    with open(out_dir / "metrics_svm.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("SVM baseline classification report:")
    print(result["classification_report"])
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
