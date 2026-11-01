## Codes/Unsupervised-Learning/README.md

# Unsupervised Learning Methods

This directory contains code implementations of the unsupervised learning algorithms presented in *Machine Learning Methods*. It is still under construction, and more implementations will be added.

Each algorithm is a standalone script with a command-line interface, and the shared modules at the end are imported by all of them.

## Clustering Methods (K-Means)

- `k_means_clustering.py` - K-Means clustering alternating nearest-centroid assignment and centroid update, re-seeding empty clusters with the worst-fitted sample, and recording the sum of squared errors together with the centroid trajectory.

## EM Algorithm

- `gaussian_mixture_em_algorithm.py` - Gaussian mixture model trained by the EM algorithm, with responsibilities and log-likelihood evaluated in log space via logsumexp and convergence tested on the per-sample log-likelihood change.

## Markov Chain Monte Carlo Methods (Metropolis-Hastings Algorithm and Gibbs Sampling)

- `gaussian_mixture_metropolis_hastings.py` - Gaussian mixture model sampled by Metropolis-Hastings, using Gaussian random walks on the means and the mixing logits and an asymmetric Inverse-Wishart proposal on the covariances, with proposal scales adapted during burn-in and frozen afterwards.
- `gaussian_mixture_gibbs_sampling.py` - Gaussian mixture model sampled by Gibbs sampling with conjugate Dirichlet and Normal-Inverse-Wishart priors, alternating assignment and parameter draws and keeping the post-burn-in sample with the highest log-posterior.

## Shared Modules

- `data_creators.py` - Dataset creation: a small hand-specified 2D set, an unbalanced 2D three-component Gaussian mixture, and the sklearn Iris dataset, all split into training and test sets and standardized.
- `plot_functions.py` - Shared plotting: convergence curves with an optional secondary axis, ground-truth versus learned clustering scatter plots with Hungarian label matching, and PCA projections of the parameter trajectory coloured by the diagnostic value.

## Requirements

Python 3.8+ with `numpy`, `scipy`, `scikit-learn` and `matplotlib`.

```bash
pip install numpy scipy scikit-learn matplotlib
```

The scripts import their siblings by module name, so run them from inside this directory.

## Quick start

```bash
python k_means_clustering.py
python gaussian_mixture_em_algorithm.py
python gaussian_mixture_gibbs_sampling.py
python gaussian_mixture_metropolis_hastings.py
```

Each script loads a dataset, trains while printing progress, saves three figures to `./plot`, and reports the Adjusted Rand Index (`sklearn.metrics.adjusted_rand_score`) on both the training and the test set.

Use `-h` to see all options with their defaults.

## Datasets

Chosen with `--data`. All ship ground-truth labels and a train/test split.

| Name | Size | Clusters | Notes |
| --- | --- | --- | --- |
| `artificial` | 24 / 16, 2D | 2 | Hand-specified, widely separated, very few points |
| `gaussians` | 1000 / 100, 2D | 3 | Unbalanced weights `(0.4, 0.5, 0.1)`, non-spherical covariances; the default |
| `iris` | 120 / 30, 4D | 3 | Real data, stratified split, two classes overlap |

`iris` is 4-dimensional, so the clustering scatter plots are skipped for it, while the convergence and trajectory plots still apply.

## Options

| Option | Scripts | Default | Meaning |
| --- | --- | --- | --- |
| `--data` | all | `gaussians` | `artificial` \| `gaussians` \| `iris` |
| `--num-clusters` / `--num-components` | K-Means / GMMs | true number in the data | Number of clusters |
| `--num-iterations` | all | 100 / 1000 / 5000 / 20000 | Iteration budget (K-Means / EM / Gibbs / MH) |
| `--convergence-threshold` | K-Means, EM | `1e-8` / `1e-4` | Stop on the centroid shift (K-Means) or the per-sample log-likelihood change (EM) |
| `--burn-in` | Gibbs, MH | 1000 / 10000 | Iterations discarded before recording; must be `< --num-iterations` |
| `--proposal-step` | MH | auto | Initial random-walk step on the means; auto-scaled from the data when omitted |

## Output

All figures are saved to `./plot` (created automatically, re-runs overwrite) and also displayed: `k-means-*.png`, `gmm-em-*.png`, `gmm-gs-*.png`, `gmm-mh-*.png`, each with a `-curve`, `-traj` and `-clusters` variant.

- **Convergence curve** - diagnostics against the iteration index; for the samplers the x axis starts at `burn_in`, and the acceptance rate (MH) or the log-posterior (Gibbs) is drawn alongside the log-likelihood.
- **Trajectory plot** - parameter snapshots projected to 2D by PCA and coloured by the diagnostic value, with the start, end and best points highlighted; the retained fraction of variance is reported on the axes.
- **Clustering plot** - ground truth on top, learned result below, sharing one colour scale; predicted labels are permuted to best match the true ones, which changes colours only, never the partition.

On a headless machine, set a non-interactive backend so that `plt.show()` does not block:

```bash
MPLBACKEND=Agg python k_means_clustering.py
```
