
# Supervised Learning Methods

This directory contains code implementations of the supervised learning algorithms presented in *Machine Learning Methods*. It is still under construction, and more implementations will be added.

Each algorithm is implemented from scratch with NumPy and SciPy and exposed as a standalone script with a command-line interface. Datasets are obtained through the shared `Data` package at the repository root (`Data/data_loaders.py`), which uses scikit-learn to load and split them.

## Contents

### Linear Regression

- `linear_regression.py` - Linear regression trained either by gradient descent or by the closed-form normal equation with the Moore-Penrose pseudoinverse, evaluated with mean squared error.

### Perceptron

- `perceptron.py` - Perceptron for binary classification, trained by online stochastic updates on misclassified instances, with an optional convergence threshold on the training error rate.

### K-Nearest Neighbors

- `k_nearest_neighbor.py` - k-nearest neighbors classifier built on a balanced KD-tree, searching with a bounded max-heap and branch-and-bound pruning, then predicting by majority vote.

### Naive Bayes

- `gaussian_naive_bayes.py` - Gaussian naive Bayes with maximum likelihood estimates of class means, variances and priors, predicting by maximum a posteriori in the log domain.

### Decision Trees

- `classification_tree.py` - CART classification tree grown greedily by minimizing the weighted Gini index, with leaves predicting the majority class.
- `regression_tree.py` - CART regression tree grown greedily by minimizing the weighted mean squared error, with leaves predicting the node mean.

### Logistic Regression

- `binomial_logistic_regression.py` - Binomial logistic regression minimizing the logistic loss, trained by gradient descent with explicit forward and backward passes or by BFGS with analytical gradients.
- `multinomial_logistic_regression.py` - Multinomial logistic regression with a softmax output, minimizing the cross-entropy loss by gradient descent or BFGS.

### Support Vector Machines

- `linear_svm.py` - Linear SVM trained by stochastic gradient descent on the L2-regularized hinge loss, visualized with its decision boundary and margins.
- `kernel_svm.py` - Kernel SVM solving the dual quadratic program with SLSQP, supporting linear, RBF and polynomial kernels, and identifying the support vectors.

### Boosting Methods (AdaBoost and GBDT)

- `adaboost.py` - AdaBoost with classification trees as base learners, reweighting instances each round by bootstrap resampling and combining learners by their coefficients.
- `gradient_boosting_binary.py` - Gradient boosting binary classifier fitting regression trees to the pseudo-residuals of the logistic loss, with shrinkage and a sigmoid output.
- `gradient_boosting_multiclass.py` - Gradient boosting multiclass classifier maintaining one tree ensemble per class and fitting the pseudo-residuals of the cross-entropy loss, with a softmax output.
- `gradient_boosting_regressor.py` - Gradient boosting regressor fitting regression trees to the residuals of the squared error loss, with shrinkage and an initial prediction set to the mean.

### Shared Modules

- `utility_functions.py` - Shared utilities: evaluation measures (accuracy, MSE), activation functions (sigmoid, softmax, hinge), loss functions (logistic, cross-entropy) and one-hot encoding.
- `plot_functions.py` - Shared plotting: training curves and 2D model visualizations for binary classification (predicted regions, decision boundaries, margins, support vectors, Gaussian densities), multiclass classification and regression surfaces.

The files under Shared Modules are libraries imported by the algorithm scripts; all the others are entry points. `adaboost.py` additionally imports `ClassificationTree`, and the three gradient boosting scripts import `RegressionTree`, as their base learners.

Datasets are not created in this directory: they come from `Data/data_loaders.py` (see [Datasets](#datasets)).

---

## Requirements

Python 3.11+ with `numpy`, `scipy`, `scikit-learn` and `matplotlib`.

```bash
pip install numpy scipy scikit-learn matplotlib
```

PyTorch is not required: the `Data` package imports it only inside the MNIST and 1D Gaussian mixture loaders, which these scripts do not use.

### Install the repository (once per environment)

All scripts import their data from the `Data` package at the repository root. From the repository root (where `pyproject.toml` is located), run:

```bash
python -m pip install -e .
```

Without it, the scripts fail with `ModuleNotFoundError: No module named 'Data'`. See [Getting Started](../README.md#getting-started) in the root README for details.

### Running the scripts

Run a script directly, either from the repository root or from inside this directory:

```bash
# from the repository root
python Supervised_Learning/kernel_svm.py

# or from inside this directory
cd Supervised_Learning
python kernel_svm.py
```

The sibling modules (`utility_functions`, `plot_functions`, ...) are imported by module name, which works because Python puts the script's own directory on the import path. Figures are saved to `./plot` relative to the directory from which the command is launched.

---

## Quick start

```bash
python linear_regression.py
python perceptron.py
python k_nearest_neighbor.py
python gaussian_naive_bayes.py
python classification_tree.py
python regression_tree.py
python binomial_logistic_regression.py
python multinomial_logistic_regression.py
python linear_svm.py
python kernel_svm.py
python adaboost.py
python gradient_boosting_binary.py
python gradient_boosting_multiclass.py
python gradient_boosting_regressor.py
```

Each script loads a dataset, trains while printing progress, saves its figures to `./plot`, and reports accuracy (classification) or mean squared error (regression) on both the training and the test set.

Use `-h` to see all options with their defaults:

```bash
python kernel_svm.py -h
```

---

## Datasets

Chosen with `--data`. Data are loaded through `Data/data_loaders.py`; the available dataset names, sizes and conventions of each task are listed in [Data/README.md](../Data/README.md). All datasets ship ground-truth labels or values and a fixed train/test split, and all features are standardized with the training mean and standard deviation.

Default dataset of each script:

| Task | Loader | Scripts | Default |
| --- | --- | --- | --- |
| Binary classification | `load_binary_classification_data` | `adaboost.py`, `gradient_boosting_binary.py`, `kernel_svm.py` | `jigsaw` |
| Binary classification | `load_binary_classification_data` | all other binary classifiers | `gaussians` |
| Multiclass classification | `load_multiclass_classification_data` | the multiclass scripts | `blobs` |
| Regression | `load_regression_data` | the regression scripts | `plane` |

`jigsaw` is not linearly separable, which is why the scripts that can model non-linear boundaries default to it.

Label conventions differ by algorithm: `perceptron.py`, `linear_svm.py`, `kernel_svm.py` and `adaboost.py` request signed labels in `{-1, 1}` (`signed_label=True` in `load_binary_classification_data`), while the remaining classifiers use `{0, 1}` or `{0, ..., k-1}`. The scripts set this flag themselves, so no manual conversion is needed.

Model visualizations require 2D features, so they are skipped for `iris`, `classification` and `diabetes`. The nearest-neighbor, naive Bayes and classification tree plots additionally require exactly two classes.

---

## Options

Iterative methods:

| Option | Scripts | Default | Meaning |
| --- | --- | --- | --- |
| `--num-iterations` | linear regression, logistic regressions | `10000` (GD) / `100` (BFGS) | Iteration budget |
| `--num-epochs` | perceptron, linear SVM | `100` | Number of passes over the training data |
| `--learning-rate` | linear regression, perceptron, logistic regressions, linear SVM | `0.1` (linear regression, perceptron), `0.01` (logistic regressions, linear SVM) | Step size for gradient descent |
| `--convergence-threshold` | linear regression, perceptron, logistic regressions, linear SVM, AdaBoost, gradient boosting | `None` | Stop when the loss change drops below this value (the training error rate for the perceptron and AdaBoost); run the full budget when omitted |
| `--algorithm` | linear regression, logistic regressions | `gd` | `gd` \| `ne` for linear regression, `gd` \| `bfgs` for the logistic regressions |

Instance-based, generative and kernel methods:

| Option | Script | Default | Meaning |
| --- | --- | --- | --- |
| `--num-neighbors` | k-nearest neighbors | `1` | Number of neighbors `k` in the majority vote |
| `--kernel` | kernel SVM | `rbf` | `linear` \| `rbf` \| `poly` |
| `--C` | kernel SVM | `1.0` | Regularization parameter; larger means less regularization |
| `--gamma` | kernel SVM | `auto` | Kernel coefficient for RBF and polynomial kernels; `auto` means `1 / num_features` |
| `--degree`, `--coef0` | kernel SVM | `3`, `1.0` | Polynomial kernel degree and independent term |

`gaussian_naive_bayes.py` takes only `--data`, since its maximum likelihood estimates are closed-form.

Tree and ensemble methods:

| Option | Scripts | Default | Meaning |
| --- | --- | --- | --- |
| `--max-tree-depth` | decision trees, AdaBoost and gradient boosting | `5` (decision trees), `2` (ensembles) | Maximum depth of a tree; `1` gives a stump |
| `--min-sample-split` | decision trees, AdaBoost and gradient boosting | `5` | Minimum instances required to split a node |
| `--num-learners` | AdaBoost, gradient boosting | `100` (`50` for the multiclass classifier) | Base learners in the ensemble, per class for the multiclass classifier |
| `--shrinkage` | gradient boosting | `0.1` | Shrinkage coefficient in `(0, 1]` applied to every learner |

Examples:

```bash
python linear_regression.py --data diabetes --algorithm ne
python perceptron.py --data gaussians --num-epochs 20 --convergence-threshold 0.01
python k_nearest_neighbor.py --data moons --num-neighbors 5
python binomial_logistic_regression.py --data circle --algorithm bfgs
python kernel_svm.py --data jigsaw --kernel poly --degree 2 --C 10
python adaboost.py --data moons --num-learners 200 --max-tree-depth 1
python gradient_boosting_regressor.py --data plane --shrinkage 0.05 --num-learners 300
```

---

## Output

All figures are saved to `./plot` (created automatically, re-runs overwrite previous files) and also displayed. File names are derived from the model name, for example `./plot/linear_svm_training_curve.png` and `./plot/linear_svm_model.png`.

| Figure | Produced by | Content |
| --- | --- | --- |
| `*_training_curve.png` | linear regression (GD only), perceptron, logistic regressions, linear SVM, AdaBoost, the three gradient boosting scripts | The diagnostic against iterations, epochs or boosting rounds: MSE, training error rate, logistic loss, cross-entropy loss or regularized hinge loss. The x axis is logarithmic except for the ensembles |
| `*_model.png` | every script, when the data is 2D | The fitted model over the padded data range, together with the training instances |

Three visualization styles are shared by all models:

- **Binary classification** - predicted regions as filled contours, drawn either as two label regions or as four probability bins with the `p = 0.5` boundary, plus positive and negative instances. Individual models overlay their own details: the separating hyperplane (perceptron), hyperplane and margins (linear SVM), the `f(x) = 0` and `f(x) = ±1` contours with circled support vectors (kernel SVM), and the learned per-class Gaussian densities (naive Bayes).
- **Multiclass classification** - predicted class regions as a filled mesh sharing one discrete colour scale with the instances.
- **Regression** - the prediction surface as filled contours with contour lines, optionally labelled with their values, and the instances grayscaled by their true values.

On a headless machine, set a non-interactive backend so that `plt.show()` does not block:

```bash
# Linux / macOS
MPLBACKEND=Agg python classification_tree.py

# Windows (PowerShell)
$env:MPLBACKEND = "Agg"; python classification_tree.py
```

---

## Evaluation

Each script prints its evaluation measure on both the training and the test set, computed in `utility_functions`:

- **Classification** - accuracy from `calculate_accuracy`, the fraction of correctly predicted labels
- **Regression** - mean squared error from `calculate_mse`

The scripts with stochastic training (`perceptron.py`, `linear_svm.py`, `adaboost.py`) fix `MODEL_SEED = 64` after the data is loaded. The train/test splits are fixed by `DATA_SEED = 32` inside the `Data` package, so repeated runs are reproducible. The remaining models are deterministic given the data.