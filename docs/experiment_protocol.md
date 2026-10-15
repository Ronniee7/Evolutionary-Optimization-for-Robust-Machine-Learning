# Experiment protocol

## Setup

- Dataset: CIFAR-10.
- Split: train/validation/test with validation reserved for model selection.
- Device: automatic CPU/CUDA detection.
- Reproducibility: configured seeds and deterministic library settings.

## Baselines

1. Train the default CNN with the baseline configuration.
2. Evaluate on clean and corrupted validation data.
3. Run random search with a modest number of trials.
4. Run the genetic optimizer with a small population and limited generations.
5. Compare the models on the final test set.

## Reporting

The final report should include:

- clean accuracy,
- mean corruption accuracy,
- per-corruption performance,
- corruption severity plots,
- training time and search time,
- summary tables from real experiment logs.

## Decision rules

The conclusion should be drawn from actual results. If the genetic algorithm is not better than the baseline or random search, the study should report that outcome and explain the likely reasons.
