from __future__ import annotations

from typing import Any, Dict


def evaluate_fitness(clean_accuracy: float, corruption_accuracy: float, clean_weight: float = 0.5, corruption_weight: float = 0.5) -> float:
    total_weight = clean_weight + corruption_weight
    if total_weight <= 0:
        raise ValueError("The combined fitness weights must be positive.")
    return float((clean_weight / total_weight) * clean_accuracy + (corruption_weight / total_weight) * corruption_accuracy)


def fitness_from_report(report: Dict[str, Any], clean_weight: float = 0.5, corruption_weight: float = 0.5) -> float:
    clean_accuracy = float(report.get("clean_accuracy", 0.0))
    corruption_accuracy = float(report.get("corruption_accuracy", 0.0))
    return evaluate_fitness(clean_accuracy, corruption_accuracy, clean_weight, corruption_weight)
