# Methodology

## Objective

This project studies whether evolutionary hyperparameter optimization can improve robustness to common image corruptions without degrading clean accuracy. The study keeps a fixed model architecture and evaluates candidates on a validation set that excludes the final test split.

## Model family

The baseline model is a compact CNN trained on CIFAR-10. The architecture remains fixed across experimental comparisons so that performance differences reflect hyperparameter search strategy rather than model design changes.

## Corruptions

The pipeline includes Gaussian noise, Gaussian blur, brightness shifts, and contrast changes. Severity is controlled with a five-level scale, with each transformation normalized to remain within valid image ranges.

## Search methods

- Default training: standard configuration.
- Random search: sampling hyperparameters from a bounded search space.
- Genetic algorithm: a lightweight evolutionary search over learning rate, weight decay, dropout, batch size, and augmentation strength.

## Fitness definition

The core fitness value is an equal-weight combination of clean validation accuracy and mean corruption accuracy:

`fitness = 0.5 * clean_accuracy + 0.5 * mean_corruption_accuracy`

The weights are configurable and can be adjusted to emphasize robustness or accuracy.

## Robustness evaluation

For each model, we measure clean test accuracy, mean corrupted accuracy, corruption-specific accuracy, and clean-to-corrupted gap. This avoids over-claiming improvements when a model only excels on a restricted subset of conditions.

## Limitations

The project uses a small CNN and a limited optimization budget. This keeps the experiments computationally feasible but means the conclusions are most relevant to this benchmark and setup rather than all vision tasks.
