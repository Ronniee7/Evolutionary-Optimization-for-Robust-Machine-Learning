from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn.functional as F

CorruptionType = Literal["gaussian_noise", "gaussian_blur", "brightness", "contrast"]


def _ensure_tensor(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image, dtype=torch.float32)
    if image.dtype not in (torch.float16, torch.float32, torch.float64):
        image = image.float()
    return image.clamp(0.0, 1.0)


def gaussian_noise(image: torch.Tensor, severity: int = 1, rng: np.random.Generator | None = None) -> torch.Tensor:
    image = _ensure_tensor(image)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    sigma = float(np.linspace(0.04, 0.20, 5)[max(0, min(severity - 1, 4))])
    if rng is None:
        noise = torch.randn_like(image) * sigma
    else:
        noise = torch.tensor(rng.normal(0.0, sigma, size=image.shape), dtype=image.dtype, device=image.device)
    return (image + noise).clamp(0.0, 1.0).squeeze(0) if image.shape[0] == 1 else (image + noise).clamp(0.0, 1.0)


def gaussian_blur(image: torch.Tensor, severity: int = 1) -> torch.Tensor:
    image = _ensure_tensor(image)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    kernel_size = 3 + 2 * (severity - 1)
    if kernel_size > image.shape[-1]:
        kernel_size = min(kernel_size, max(3, image.shape[-1] - 1))
    pad = kernel_size // 2
    sigma = float(np.linspace(0.5, 2.0, 5)[max(0, min(severity - 1, 4))])
    xs = torch.linspace(-(kernel_size // 2), kernel_size // 2, steps=kernel_size, device=image.device, dtype=image.dtype)
    ys = xs.unsqueeze(1)
    kernel = torch.exp(-(xs.pow(2) + ys.pow(2)) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    kernel = kernel.expand(image.shape[1], 1, kernel_size, kernel_size)
    blurred = F.conv2d(image, kernel, padding=pad, groups=image.shape[1])
    return blurred.clamp(0.0, 1.0).squeeze(0) if image.shape[0] == 1 else blurred.clamp(0.0, 1.0)


def brightness(image: torch.Tensor, severity: int = 1) -> torch.Tensor:
    image = _ensure_tensor(image)
    delta = float(np.linspace(0.10, 0.60, 5)[max(0, min(severity - 1, 4))])
    return (image + delta).clamp(0.0, 1.0)


def contrast(image: torch.Tensor, severity: int = 1) -> torch.Tensor:
    image = _ensure_tensor(image)
    factor = float(np.linspace(1.0, 2.0, 5)[max(0, min(severity - 1, 4))])
    mean = image.mean(dim=(1, 2), keepdim=True)
    adjusted = (image - mean) * factor + mean
    return adjusted.clamp(0.0, 1.0)


def apply_corruption(image: torch.Tensor, corruption_type: str, severity: int = 1, rng: np.random.Generator | None = None) -> torch.Tensor:
    corruption_type = corruption_type.lower()
    if corruption_type == "gaussian_noise":
        return gaussian_noise(image, severity=severity, rng=rng)
    if corruption_type == "gaussian_blur":
        return gaussian_blur(image, severity=severity)
    if corruption_type == "brightness":
        return brightness(image, severity=severity)
    if corruption_type == "contrast":
        return contrast(image, severity=severity)
    raise ValueError(f"Unsupported corruption type: {corruption_type}")


def corruption_suite(image: torch.Tensor, severities: Iterable[int] | None = None, corruption_types: Iterable[str] | None = None):
    severities = list(severities) if severities is not None else [1, 2, 3, 4, 5]
    corruption_types = list(corruption_types) if corruption_types is not None else ["gaussian_noise", "gaussian_blur", "brightness", "contrast"]
    corrupted = {}
    for corruption_type in corruption_types:
        for severity in severities:
            patched = apply_corruption(image, corruption_type, severity=severity)
            corrupted[(corruption_type, severity)] = patched
    return corrupted
