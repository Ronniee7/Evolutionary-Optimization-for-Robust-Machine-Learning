import torch

from src.model import SmallCNN


def test_small_cnn_output_shape():
    model = SmallCNN(num_classes=10, dropout=0.1)
    inputs = torch.randn(4, 3, 32, 32)
    outputs = model(inputs)
    assert outputs.shape == (4, 10)
