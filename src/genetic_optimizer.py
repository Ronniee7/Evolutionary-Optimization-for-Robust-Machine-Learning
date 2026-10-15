from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Tuple


class GeneticOptimizer:
    """A compact genetic algorithm for hyperparameter search."""

    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]], mutation_rate: float = 0.2, crossover_rate: float = 0.8, seed: int = 42):
        self.parameter_bounds = parameter_bounds
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = random.Random(seed)

    def _initial_population(self, population_size: int) -> List[Dict[str, float]]:
        population = []
        for _ in range(population_size):
            chromosome = {}
            for name, (low, high) in self.parameter_bounds.items():
                if isinstance(low, int) and isinstance(high, int):
                    chromosome[name] = self.rng.randint(low, high)
                else:
                    chromosome[name] = self.rng.uniform(low, high)
            population.append(chromosome)
        return population

    def _mutate(self, chromosome: Dict[str, float]) -> Dict[str, float]:
        mutated = copy.deepcopy(chromosome)
        for name, (low, high) in self.parameter_bounds.items():
            if self.rng.random() < self.mutation_rate:
                if isinstance(low, int) and isinstance(high, int):
                    mutated[name] = self.rng.randint(low, high)
                else:
                    mutated[name] = self.rng.uniform(low, high)
        return mutated

    def _crossover(self, parent_a: Dict[str, float], parent_b: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        if self.rng.random() > self.crossover_rate:
            return copy.deepcopy(parent_a), copy.deepcopy(parent_b)
        child_a = {}
        child_b = {}
        for key in self.parameter_bounds:
            if self.rng.random() < 0.5:
                child_a[key] = parent_a[key]
                child_b[key] = parent_b[key]
            else:
                child_a[key] = parent_b[key]
                child_b[key] = parent_a[key]
        return child_a, child_b

    def optimize(self, objective, population_size: int = 8, generations: int = 4) -> Dict[str, Any]:
        population = self._initial_population(population_size)
        history = []
        for generation in range(generations):
            scored = []
            for chromosome in population:
                fitness = objective(chromosome)
                scored.append((fitness, chromosome))
            scored.sort(key=lambda item: item[0], reverse=True)
            best = scored[0][1]
            history.append({"generation": generation, "fitness": scored[0][0], "best": copy.deepcopy(best)})
            elites = [chromosome for _, chromosome in scored[: max(1, population_size // 4)]]
            next_population = list(elites)
            while len(next_population) < population_size:
                parent_a = self.rng.choice(population)
                parent_b = self.rng.choice(population)
                child_a, child_b = self._crossover(parent_a, parent_b)
                next_population.append(self._mutate(child_a))
                if len(next_population) < population_size:
                    next_population.append(self._mutate(child_b))
            population = next_population[:population_size]
        final_score = max(objective(chromosome) for chromosome in population)
        best_candidate = max(population, key=objective)
        return {"best_candidate": best_candidate, "final_score": final_score, "history": history }


def run_genetic_search(parameter_bounds: Dict[str, Tuple[float, float]], objective, population_size: int = 8, generations: int = 4, mutation_rate: float = 0.2, crossover_rate: float = 0.8, seed: int = 42) -> Dict[str, Any]:
    optimizer = GeneticOptimizer(parameter_bounds, mutation_rate=mutation_rate, crossover_rate=crossover_rate, seed=seed)
    return optimizer.optimize(objective, population_size=population_size, generations=generations)
