from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Compact CNN tuned for CIFAR-like image classification tasks."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.2, hidden_channels: tuple[int, ...] | list[int] = (32, 64)):
        super().__init__()
        channels = list(hidden_channels)
        self.features = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[1], channels[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[1] * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    @staticmethod
    def from_config(config: dict) -> "SmallCNN":
        model_config = config.get("model", {})
        return SmallCNN(
            num_classes=int(model_config.get("num_classes", 10)),
            dropout=float(model_config.get("dropout", 0.2)),
            hidden_channels=tuple(model_config.get("hidden_channels", [32, 64])),
        )
