from src.genetic_optimizer import GeneticOptimizer


def test_genetic_optimizer_returns_best_candidate():
    bounds = {"learning_rate": (1e-5, 1e-2), "dropout": (0.0, 0.5)}
    optimizer = GeneticOptimizer(bounds, seed=7)
    result = optimizer.optimize(lambda chrom: chrom["learning_rate"] + chrom["dropout"], population_size=6, generations=3)
    assert "best_candidate" in result
    assert "history" in result
    assert len(result["history"]) == 3
    assert set(result["best_candidate"]).issubset({"learning_rate", "dropout"})
