from __future__ import annotations
import torch
from torch import nn


class LeNet5(nn.Module):
    """LeNet-5 style CNN adapted to 32x32 grayscale images and 5 classes."""
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),  # 32 -> 28
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),                 # 28 -> 14
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),   # 14 -> 10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),                 # 10 -> 5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
