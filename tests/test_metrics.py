import torch

from src.metrics import accuracy, clean_corrupted_gap


def test_accuracy_and_gap():
    logits = torch.tensor([[2.0, 0.5], [0.1, 2.0], [0.1, 1.0]])
    targets = torch.tensor([0, 1, 0])
    assert abs(accuracy(logits, targets) - 2 / 3) < 1e-6
    assert abs(clean_corrupted_gap(0.9, 0.75) - 0.15) < 1e-8
