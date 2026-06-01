#!/usr/bin/env bash
set -e
python src/generate_dataset.py
python src/train_lenet.py --epochs 30 --batch_size 64 --lr 0.001
python src/evaluate_baseline_svm.py
python src/visualize_results.py
