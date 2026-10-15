import numpy as np
import torch

from src.corruptions import apply_corruption, brightness, contrast, gaussian_blur, gaussian_noise


def test_corruption_outputs_are_valid():
    image = torch.rand(3, 32, 32)
    for corruption in [gaussian_noise, gaussian_blur, brightness, contrast]:
        output = corruption(image, severity=3)
        assert torch.isfinite(output).all()
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    for name in ["gaussian_noise", "gaussian_blur", "brightness", "contrast"]:
        output = apply_corruption(image, name, severity=3, rng=np.random.default_rng(0))
        assert torch.isfinite(output).all()
        assert output.min() >= 0.0
        assert output.max() <= 1.0
