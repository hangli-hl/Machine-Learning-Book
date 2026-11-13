# Data

This directory is the single data layer of the repository. Every algorithm in
`Supervised_Learning/`, `Unsupervised_Learning/` and `Deep_Learning/` gets its
data from here, so that all methods are evaluated on identical, reproducible
datasets.

---

## Contents

| File | Role |
| --- | --- |
| `data_loaders.py` | **The only entry point.** Loads or generates a dataset, splits it, standardizes it and encodes the labels. |
| `data_creators.py` | Generates the synthetic datasets defined by this repository. Returns raw data only. |
| `MNIST/` | Created automatically on the first MNIST call (downloaded by torchvision). Not part of the source code. |

---

## Requirements

Python 3.11+. `torch` and `torchvision` are imported inside the functions that use them, so they are needed only when you call those loaders.

| Loaders | Packages |
| --- | --- |
| `load_binary_classification_data`, `load_multiclass_classification_data`, `load_regression_data`, `load_clustering_data` | `numpy`, `scikit-learn` |
| `load_gaussian_mixture_1d_data`, `gaussian_mixture_1d_pdf`, `gaussian_mixture_1d_score` | `numpy`, `torch` |
| `load_mnist_classification_data`, `load_mnist_generation_data` | `numpy`, `torch`, `torchvision` |

The repository must be installed once per environment with `python -m pip install -e .` (see [Getting Started](../README.md#getting-started) in the root README).

---

## Usage rule

Algorithm code imports from `data_loaders` and **never** from `data_creators`:

```python
from Data.data_loaders import load_binary_classification_data

train_x, train_y, test_x, test_y = load_binary_classification_data("moons")
```

Without the editable install, the import fails with `ModuleNotFoundError: No module named 'Data'`.

---

## Loader reference

| Function | Arguments | Returns |
| --- | --- | --- |
| `load_binary_classification_data` | `dataset`, `signed_label=False` | `(train_x, train_y, test_x, test_y)` |
| `load_multiclass_classification_data` | `dataset` | `(train_x, train_y, test_x, test_y)` |
| `load_regression_data` | `dataset` | `(train_x, train_y, test_x, test_y)` |
| `load_clustering_data` | `dataset` | `(train_x, train_y, test_x, test_y)` |
| `load_gaussian_mixture_1d_data` | `num_samples`, `device="cpu"` | tensor of shape `(num_samples, 1)` |
| `gaussian_mixture_1d_pdf` | `x` (NumPy array) | NumPy array, same shape as `x` |
| `gaussian_mixture_1d_score` | `x` (tensor, `(num_points, 1)`) | tensor, `(num_points, 1)` |
| `load_mnist_classification_data` | `batch_size` | `(training_loader, test_loader)` |
| `load_mnist_generation_data` | `batch_size`, `tanh_range=False` | `training_loader` |

Passing an unknown dataset name raises a `ValueError` that lists the available
names.

For the table-style loaders (binary, multiclass, regression, clustering),
`train_x` and `test_x` are NumPy arrays of shape `(num_instances, num_features)`,
and `train_y` and `test_y` are NumPy arrays of shape `(num_instances,)`.

---

## Conventions

### Processing pipeline

Every table-style dataset goes through the same steps:

1. **Generate or load** the full dataset (raw).
2. **Split** into train and test with explicit sizes `(num_train, num_test)`.
   The two sizes must add up to the total number of samples, otherwise a
   `ValueError` is raised, so no sample is silently dropped. The split is
   shuffled with `random_state=32`.
3. **Standardize** the features with the **training** mean and standard deviation
   (z-score, population standard deviation). A feature with zero standard
   deviation is divided by 1 instead. Test data is transformed with the training
   statistics, so nothing leaks from the test set.
4. **Encode the labels** (binary classification only, see below).

### Labels

| Task | Label values |
| --- | --- |
| Binary classification | `{0, 1}` by default, `{-1, 1}` with `signed_label=True`. `1` is the positive class. |
| Multiclass classification | `{0, 1, ..., k-1}` |
| Clustering | `{0, 1, ..., k-1}`. Labels are for evaluation only and must not be used for fitting. |
| Regression | Real-valued targets, standardized with the training mean and standard deviation. |

### Stratification

The split is stratified by label for binary classification, multiclass
classification and clustering, so the class proportions in train and test match
the full dataset. The regression split is not stratified.

### Reproducibility

A single constant, `DATA_SEED = 32`, controls the generation of synthetic
data, the scikit-learn `make_*` datasets and the train/test split. Data creators
use local random generators (`np.random.RandomState` or `torch.Generator`) and
never touch the global random state. Calling a table-style loader or
`load_gaussian_mixture_1d_data` twice on the same device returns identical data.
The MNIST batch order is the exception (see [MNIST](#mnist)).

---

## Datasets

The same name can refer to different data in different tasks. **Always check the
task.**

### Binary classification

Total of 220 samples for the 2D datasets. Features are standardized.

| Name | Features | Split (train / test) | Description |
| --- | --- | --- | --- |
| `gaussians` | 2 | 200 / 20 | Two Gaussians, 110 samples each. Positive: mean `(2, 5)`, covariance `[[1, -1], [-1, 3]]`. Negative: mean `(3, -2)`, covariance `[[1, 0.5], [0.5, 2]]`. |
| `jigsaw` | 2 | 200 / 20 | XOR-like pattern from four Gaussians (55 samples each, shared covariance `[[1.5, 0.3], [0.3, 1.5]]`). Positive means: `(2.5, 2.5)`, `(-2.5, -2.5)`. Negative means: `(-2.5, 2.5)`, `(2.5, -2.5)`. Not linearly separable. |
| `circle` | 2 | 200 / 20 | Concentric rings, 110 samples each. Positive: radius uniform in `[0, 2]`. Negative: radius uniform in `[4, 6]`. Angles are uniform. Not linearly separable. |
| `moons` | 2 | 200 / 20 | scikit-learn `make_moons`, 220 samples, `noise=0.2`. |
| `iris` | 4 | 120 / 30 | Iris, **class 2 (virginica) versus the rest**. Positive class has 50 of the 150 samples. |

### Multiclass classification

| Name | Features | Classes | Split (train / test) | Description |
| --- | --- | --- | --- | --- |
| `classification` | 20 | 4 | 800 / 200 | scikit-learn `make_classification`: 1000 samples, 15 informative and 5 redundant features, 1 cluster per class. |
| `blobs` | 2 | 3 | 240 / 60 | scikit-learn `make_blobs`: 300 samples, 3 centers, `cluster_std=1.0`. |
| `iris` | 4 | 3 | 120 / 30 | Iris with its three original classes. |

### Regression

| Name | Features | Split (train / test) | Description |
| --- | --- | --- | --- |
| `plane` | 2 | 200 / 20 | Noisy plane `y = 5*x1 - 3*x2 + 2 + eps`, with `x ~ U(-2, 2)^2` and `eps ~ N(0, 0.2^2)`, 220 samples. |
| `diabetes` | 10 | 353 / 89 | scikit-learn Diabetes (442 samples). |

Both features **and targets** are standardized with training statistics. Learned
coefficients are therefore expressed in standardized units, and they differ from
the raw coefficients `5`, `-3`, `2` of `plane`.

### Clustering

| Name | Features | Clusters | Split (train / test) | Description |
| --- | --- | --- | --- | --- |
| `gaussians` | 2 | 3 | 1000 / 100 | Three Gaussians with unbalanced mixing weights `0.4 / 0.5 / 0.1` (440 / 550 / 110 of 1100 samples). Means: `(3, 1)`, `(0, -2)`, `(-1, 2)`. Covariances: `[[1, 0], [0, 1]]`, `[[2, -1.5], [-1.5, 2]]`, `[[3, 1], [1, 2]]`. |
| `iris` | 4 | 3 | 120 / 30 | Iris with its three original classes. |

Note that `gaussians` here is a **different dataset** from the binary
`gaussians`, and that `iris` appears in all of binary classification, multiclass
classification and clustering with different labels.

### 1D Gaussian mixture (generative models)

A two-component mixture: 30% from `N(-5, 1)` and 70% from `N(5, 1)`.

- `load_gaussian_mixture_1d_data(num_samples, device="cpu")` draws samples using
  `DATA_SEED`. There is no train/test split and no standardization.
- `gaussian_mixture_1d_pdf(x)` is the analytic density (NumPy in, NumPy out),
  intended for plotting.
- `gaussian_mixture_1d_score(x)` is the analytic score, i.e. the gradient of the
  log-density (tensor of shape `(num_points, 1)` in and out). It serves as the
  ground truth for score-based models.

Sampling uses a `torch.Generator` on the requested device, so the same seed does
not give identical samples on CPU and on GPU.

### MNIST

Handwritten digits, 60,000 training and 10,000 test images of size `1 x 28 x 28`.
The files are downloaded to `Data/MNIST/` on the first call if they are missing.
The loaders return `torch.utils.data.DataLoader` objects that yield
`(images, labels)` batches.

| Function | Pixel range | Splits | Shuffling |
| --- | --- | --- | --- |
| `load_mnist_classification_data(batch_size)` | `[0, 1]` | train and test loaders | train: yes, test: no |
| `load_mnist_generation_data(batch_size, tanh_range=False)` | `[0, 1]` (VAE) or `[-1, 1]` with `tanh_range=True` (GAN, DDPM, SMLD, FM) | train loader only | yes |

MNIST is not standardized with dataset statistics. Pixels are only scaled to
`[0, 1]`, or mapped to `[-1, 1]` with `Normalize(0.5, 0.5)`. The batch order is
drawn from PyTorch's global random state, so set `torch.manual_seed(...)` in your
script if you need reproducible batches.