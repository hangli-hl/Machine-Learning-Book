# Unsupervised Learning Methods

This directory contains code implementations of the unsupervised learning algorithms presented in *Machine Learning Methods*. It is still under construction, and more implementations will be added.

Each algorithm is implemented from scratch with NumPy and SciPy and exposed as a standalone script with a command-line interface. Datasets are obtained through the shared `Data` package at the repository root (`Data/data_loaders.py`); scikit-learn is used here only to compute the Adjusted Rand Index and to run the PCA behind the trajectory plots.

## Contents

### Clustering Methods (K-Means)

- `k_means_clustering.py` - K-Means clustering alternating nearest-centroid assignment and centroid update, re-seeding empty clusters with the worst-fitted sample, and recording the sum of squared errors together with the centroid trajectory.

### EM Algorithm

- `gaussian_mixture_em_algorithm.py` - Gaussian mixture model trained by the EM algorithm, with responsibilities and log-likelihood evaluated in log space via logsumexp and convergence tested on the per-sample log-likelihood change.

### Markov Chain Monte Carlo Methods (Metropolis-Hastings Algorithm and Gibbs Sampling)

- `gaussian_mixture_metropolis_hastings.py` - Gaussian mixture model sampled by Metropolis-Hastings, using Gaussian random walks on the means and the mixing logits and an asymmetric Inverse-Wishart proposal on the covariances, with proposal scales adapted during burn-in and frozen afterwards.
- `gaussian_mixture_gibbs_sampling.py` - Gaussian mixture model sampled by Gibbs sampling with conjugate Dirichlet and Normal-Inverse-Wishart priors, alternating assignment and parameter draws and keeping the post-burn-in sample with the highest log-posterior.

### Shared Modules

- `plot_functions.py` - Shared plotting: convergence curves with an optional secondary axis, ground-truth versus learned clustering scatter plots with Hungarian label matching, and PCA projections of the parameter trajectory coloured by the diagnostic value.

`plot_functions.py` is a library imported by the algorithm scripts; all the others are entry points. The four algorithm scripts do not import each other.

Datasets are not created in this directory: they come from `Data/data_loaders.py` (see [Datasets](#datasets)).

---

## Requirements

Python 3.11+ with `numpy`, `scipy`, `scikit-learn` and `matplotlib`.

```bash
pip install numpy scipy scikit-learn matplotlib
```

### Install the repository in editable mode (once per environment)

All scripts import their data from the `Data` package at the repository root, which lies outside this directory. Install the repository once per environment, from the repository root (where `pyproject.toml` is located):

```bash
python -m pip install -e .
```

What this does:

- `-e` (editable) registers the repository in the current Python environment as a link to the source tree instead of copying files. Therefore `from Data.data_loaders import ...` works from any working directory.
- Edits to the source files take effect immediately, with no reinstall.
- It is needed only once per environment (virtualenv, conda env, ...); repeat it when you create a new one.
- Without it, the scripts fail with `ModuleNotFoundError: No module named 'Data'`.

### Running the scripts

Run a script directly, either from the repository root or from inside this directory:

```bash
# from the repository root
python Unsupervised_Learning/k_means_clustering.py

# or from inside this directory
cd Unsupervised_Learning
python k_means_clustering.py
```

The sibling module `plot_functions` is imported by module name, which works because Python puts the script's own directory on the import path. Figures are saved to `./plot` relative to the directory from which the command is launched.

---

## Quick start

```bash
python k_means_clustering.py
python gaussian_mixture_em_algorithm.py
python gaussian_mixture_metropolis_hastings.py
python gaussian_mixture_gibbs_sampling.py
```

Each script loads a dataset, trains while printing progress, saves its figures to `./plot`, and reports the Adjusted Rand Index (ARI) on both the training and the test set.

Use `-h` to see all options with their defaults:

```bash
python gaussian_mixture_metropolis_hastings.py -h
```

The Metropolis-Hastings sampler is the slowest one, because it evaluates the full mixture likelihood in every one of its 20000 default iterations; pass a smaller `--num-iterations` and `--burn-in` for a quick smoke test:

```bash
python gaussian_mixture_metropolis_hastings.py --num-iterations 2000 --burn-in 1000
```

---

## Datasets

Chosen with `--data`. Data are loaded through `load_clustering_data` in `Data/data_loaders.py`. Both datasets ship ground-truth labels and a fixed train/test split, and all features are standardized with the training mean and standard deviation. The labels are never used for training; they serve only for evaluation and for the ground-truth panel of the clustering plot.

| Name | Size | Clusters | Notes |
| --- | --- | --- | --- |
| `gaussians` | 1000 / 100, 2D | 3 | Unbalanced weights `(0.4, 0.5, 0.1)`, non-spherical covariances, stratified split; the default |
| `iris` | 120 / 30, 4D | 3 | Real data, stratified split, two classes overlap |

Model visualizations require 2D features, so the clustering plot is skipped for `iris`, while the training-curve and trajectory plots still apply.

---

## Options

Dataset and model size:

| Option | Scripts | Default | Meaning |
| --- | --- | --- | --- |
| `--data` | all | `gaussians` | `gaussians` \| `iris` |
| `--num-clusters` | K-Means | true number in the data | Number of clusters `k` |
| `--num-components` | the three GMM scripts | true number in the data | Number of Gaussian components `K` |

Iterative methods:

| Option | Scripts | Default | Meaning |
| --- | --- | --- | --- |
| `--num-iterations` | all | `100` (K-Means) / `1000` (EM) / `20000` (MH) / `5000` (Gibbs) | Iteration budget; for the samplers, the total number of iterations including burn-in |
| `--convergence-threshold` | K-Means, EM | `1e-6` / `1e-4` | Stop when the squared Frobenius norm of the centroid update (K-Means) or the per-sample log-likelihood change (EM) drops to this value or below |

Sampling methods:

| Option | Scripts | Default | Meaning |
| --- | --- | --- | --- |
| `--burn-in` | MH, Gibbs | `10000` / `1000` | Iterations discarded before recording; must be `< --num-iterations` |
| `--proposal-step` | MH | auto | Initial step of the Gaussian random walk on the means; auto-scaled from the data and the sample size when omitted, then tuned during burn-in towards an acceptance rate of 0.234 and frozen afterwards |

The prior hyperparameters of the Gibbs sampler and the remaining tuning constants of the MH sampler are module-level constants in the scripts, not command-line options.

Examples:

```bash
python k_means_clustering.py --data gaussians --num-clusters 4
python gaussian_mixture_em_algorithm.py --data iris --convergence-threshold 1e-6
python gaussian_mixture_metropolis_hastings.py --data gaussians --num-iterations 5000 --burn-in 2000 --proposal-step 0.05
python gaussian_mixture_gibbs_sampling.py --data iris --num-iterations 2000 --burn-in 500
```

---

## Output

All figures are saved to `./plot` (created automatically, re-runs overwrite previous files) and also displayed. File names are derived from the model name, for example `./plot/gmm_em_training_curve.png` and `./plot/k_means_trajectory.png`. The prefixes are `k_means`, `gmm_em`, `gmm_mh` and `gmm_gs`.

| Figure | Produced by | Content |
| --- | --- | --- |
| `*_training_curve.png` | every script | The diagnostic against the iteration index: sum of squared errors (K-Means), log-likelihood (EM), log-likelihood with the acceptance rate on a secondary axis (MH), or log-posterior (Gibbs). The x axis is linear and, for the samplers, starts at `burn_in` |
| `*_trajectory.png` | every script | The parameter trajectory projected to 2D by PCA and coloured by the diagnostic value |
| `*_model.png` | every script, when the data is 2D | Ground truth on top, learned clustering below |

Details of the three figures:

* **Training curve** - for MH, the acceptance rate is the windowed rate over the last 200 iterations, and the counters restart at the burn-in boundary, when the proposal scales are frozen.
* **Trajectory plot** - flattened parameters (centroids for K-Means; coefficients, means and covariances for the GMMs) are standardized column-wise and projected by PCA. The start, end and best points are highlighted, and the retained fraction of variance is reported on the axes. "Best" means the minimum SSE for K-Means, the maximum log-likelihood for EM and MH, and the maximum log-posterior for Gibbs. For the samplers, every 10th snapshot is kept, the best one is always kept, and repeated snapshots (rejected proposals) are dropped. The plot is skipped with a message when fewer than three distinct snapshots remain, for example when K-Means converges within a couple of iterations.
* **Clustering plot** - the training samples with the true and the learned centers (red crosses), sharing one colour scale. Predicted labels are permuted by Hungarian matching to best fit the true ones, which changes colours only, never the partition.

On a headless machine, set a non-interactive backend so that `plt.show()` does not block:

```bash
MPLBACKEND=Agg python k_means_clustering.py
```

---

## Evaluation

Each script prints the Adjusted Rand Index between the ground-truth labels and the learned cluster assignments on both the training and the test set, computed with `sklearn.metrics.adjusted_rand_score`:

* **ARI** - lies in `[-1, 1]`; `1` means identical partitions and values near `0` mean chance-level agreement. It is invariant to any permutation of the cluster ids, so no label matching is needed for the numbers.

The test samples are assigned by `predict` under the final parameters, which are chosen as follows:

| Script | Final parameters | Additionally printed |
| --- | --- | --- |
| K-Means | Centroids of the last iteration | Final SSE |
| EM | Parameters of the last iteration | Final log-likelihood |
| Metropolis-Hastings | Post-burn-in sample with the highest log-likelihood | Post-burn-in acceptance rate, best log-likelihood |
| Gibbs sampling | Post-burn-in sample with the highest log-posterior | Best log-posterior (up to a constant) |

All four methods start from `k` distinct random training samples as initial centroids or means, so the result is a local optimum that depends on the initialization. To make repeated runs reproducible, every script fixes `MODEL_SEED` after the data is loaded (`32` for K-Means, EM and MH, `64` for Gibbs sampling); the train/test split is fixed by a random seed inside the `Data` package, and `plot_functions.py` uses `PLOT_SEED = 64` for the PCA.