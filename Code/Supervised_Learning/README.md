
# Supervised Learning Methods

This directory contains code implementations of the supervised learning algorithms presented in *Machine Learning Methods*. It is still under construction, and more implementations will be added.

Each algorithm is implemented from scratch with NumPy and SciPy and exposed as a standalone script with a command-line interface; scikit-learn is used only to load datasets and to split them.

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

- `data_creators.py` - Dataset creation: small manually defined 2D sets, synthetic Gaussian, jigsaw, circle, moons, blobs and plane data, and the sklearn Iris, classification and Diabetes datasets, all split and standardized.
- `utility_functions.py` - Shared utilities: evaluation measures (accuracy, MSE), activation functions (sigmoid, softmax, hinge), loss functions (logistic, cross-entropy) and one-hot encoding.
- `plot_functions.py` - Shared plotting: training curves and 2D model visualizations for binary classification (predicted regions, decision boundaries, margins, support vectors, Gaussian densities), multiclass classification and regression surfaces.

The last three files are libraries imported by the algorithm scripts; all the others are entry points. `adaboost.py` additionally imports `ClassificationTree`, and the three gradient boosting scripts import `RegressionTree`, as their base learners.

---

## Requirements

Python 3.11+ with `numpy`, `scipy`, `scikit-learn` and `matplotlib`.

```bash
pip install numpy scipy scikit-learn matplotlib
```

The scripts import their siblings by module name, so run them from inside this directory:

```bash
cd Code/Supervised_Learning
```

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

Chosen with `--data`. All ship ground-truth labels or values and a train/test split, and all features are standardized with the training mean and standard deviation.

The default dataset is `gaussians` for the binary classifiers, except `adaboost.py`, `gradient_boosting_binary.py` and `kernel_svm.py`, which default to `jigsaw`.

Binary classification:

| Name | Size | Notes |
| --- | --- | --- |
| `artificial` | 14 / 10, 2D | Hand-specified, linearly separable, very few points |
| `gaussians` | 200 / 20, 2D | Two Gaussians with different covariances; the default for all binary classifiers except `adaboost.py`, `gradient_boosting_binary.py` and `kernel_svm.py` |
| `jigsaw` | 200 / 20, 2D | XOR-like, four Gaussian blobs in two classes, not linearly separable; the default for `adaboost.py`, `gradient_boosting_binary.py` and `kernel_svm.py` |
| `circle` | 200 / 20, 2D | Inner disc against an outer ring, not linearly separable |
| `moons` | 200 / 20, 2D | Two interleaving half moons with noise |
| `iris` | 120 / 30, 4D | Real data, class 2 against classes 0 and 1 |

Multiclass classification:

| Name | Size | Classes | Notes |
| --- | --- | --- | --- |
| `classification` | 800 / 200, 20D | 4 | Synthetic, 15 informative and 5 redundant features |
| `blobs` | 240 / 60, 2D | 3 | Three isotropic blobs; the default for the multiclass scripts |
| `iris` | 120 / 30, 4D | 3 | Real data, stratified split, two classes overlap |

Regression:

| Name | Size | Notes |
| --- | --- | --- |
| `artificial` | 14 / 10, 2D | Hand-specified, values standardized |
| `plane` | 200 / 20, 2D | `y = 5*x1 - 3*x2 + 2` with Gaussian noise; the default for the regression scripts |
| `diabetes` | 353 / 89, 10D | Real data, values standardized |

Label conventions differ by algorithm: `perceptron.py`, `linear_svm.py`, `kernel_svm.py` and `adaboost.py` request signed labels in `{-1, 1}`, while the remaining classifiers use `{0, 1}` or `{0, ..., k-1}`. The flag is handled inside `data_creators.select_binary_classification_data`, so no manual conversion is needed.

Model visualizations require 2D features, so they are skipped for `iris`, `classification` and `diabetes`. The nearest-neighbor, naive Bayes and classification tree plots additionally require exactly two classes.

---

## Options

Iterative methods:

| Option | Scripts | Default | Meaning |
| --- | --- | --- | --- |
| `--num-iterations` | linear regression, logistic regressions | `10000` (GD) / `100` (BFGS) | Iteration budget |
| `--num-epochs` | perceptron, linear SVM | `100` | Number of passes over the training data |
| `--learning-rate` | linear regression, perceptron, logistic regressions, linear SVM | `0.1` / `0.1` / `0.01` / `0.01` | Step size for gradient descent |
| `--convergence-threshold` | the same scripts | `None` | Stop when the loss change (or the training error rate) drops below this; run the full budget when omitted |
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
| `--max-tree-depth` | trees / ensembles | `5` / `2` | Maximum depth of a tree; `1` gives a stump |
| `--min-sample-split` | trees and ensembles | `5` | Minimum instances required to split a node |
| `--num-learners` | AdaBoost, gradient boosting | `100` (`50` for the multiclass classifier) | Base learners in the ensemble, per class for the multiclass classifier |
| `--shrinkage` | gradient boosting | `0.1` | Shrinkage coefficient in `(0, 1]` applied to every learner |
| `--convergence-threshold` | AdaBoost, gradient boosting | `None` | Stop when the training error rate (AdaBoost) or the loss change (gradient boosting) drops below this |

Examples:

```bash
python linear_regression.py --data diabetes --algorithm ne
python perceptron.py --data artificial --num-epochs 20 --convergence-threshold 0.01
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

* **Binary classification** - predicted regions as filled contours, drawn either as two label regions or as four probability bins with the `p = 0.5` boundary, plus positive and negative instances. Individual models overlay their own details: the separating hyperplane (perceptron), hyperplane and margins (linear SVM), the `f(x) = 0` and `f(x) = ±1` contours with circled support vectors (kernel SVM), and the learned per-class Gaussian densities (naive Bayes).
* **Multiclass classification** - predicted class regions as a filled mesh sharing one discrete colour scale with the instances.
* **Regression** - the prediction surface as filled contours with contour lines, optionally labelled with their values, and the instances grayscaled by their true values.

On a headless machine, set a non-interactive backend so that `plt.show()` does not block:

```bash
MPLBACKEND=Agg python classification_tree.py
```

---

## Evaluation

Each script prints its evaluation measure on both the training and the test set, computed in `utility_functions`:

* **Classification** - accuracy from `calculate_accuracy`, the fraction of correctly predicted labels
* **Regression** - mean squared error from `calculate_mse`

The scripts with stochastic training (`perceptron.py`, `linear_svm.py`, `adaboost.py`) fix `MODEL_SEED = 36` after the data is created, and `data_creators.py` uses `DATA_SEED = 28`, so repeated runs are reproducible. The remaining models are deterministic given the data.