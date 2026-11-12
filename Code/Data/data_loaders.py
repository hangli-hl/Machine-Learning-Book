##############################################################################
#                  UNIFIED DATA LOADERS (THE ONLY ENTRY POINT)
##############################################################################
#
# All algorithms (Supervised_Learning / Unsupervised_Learning / Deep_Learning)
# get their data through this module:
#
#   from Data.data_loaders import load_binary_classification_data
#
# Responsibilities:
#   * call data_creators.py for repository-defined synthetic datasets
#   * load standard datasets (Iris, Diabetes, MNIST, sklearn make_*)
#   * split train/test, standardize with TRAINING statistics, encode labels
#
# Section order mirrors data_creators.py (binary, regression, clustering,
# 1D Gaussian mixture); sections without a creator (multiclass, MNIST) follow.
#
# Data conventions:
#   instances / samples : shape (num_instances, num_features)
#   binary labels       : {0, 1} or {-1, 1} (signed_label=True); 1 = positive
#   multiclass / cluster labels : {0, 1, ..., k-1}
#   regression values   : float, standardized

from pathlib import Path

import numpy as np
from sklearn.datasets import (
    load_diabetes,
    load_iris,
    make_blobs,
    make_classification,
    make_moons,
)
from sklearn.model_selection import train_test_split

from . import data_creators

__all__ = [
    "load_binary_classification_data",
    "load_regression_data",
    "load_clustering_data",
    "load_gaussian_mixture_1d_data",
    "gaussian_mixture_1d_pdf",
    "gaussian_mixture_1d_score",
    "load_multiclass_classification_data",
    "load_mnist_classification_data",
    "load_mnist_generation_data",
]

# Random seed for supervised learning and deep learning datasets
DATA_SEED = 28
# Random seed for clustering datasets (kept at its original value)
CLUSTERING_DATA_SEED = 16
# Directory for downloaded standard datasets (torchvision creates MNIST/ below it)
DATA_DIR = Path(__file__).resolve().parent


##############################################################################
#                           HELPER FUNCTIONS
##############################################################################


def _check_dataset(dataset, available):
    """Raise ValueError if dataset is not in available."""
    if dataset not in available:
        names = ", ".join(f"'{name}'" for name in available)
        raise ValueError(f"Unknown dataset: '{dataset}'. Available datasets: {names}")


def _standardize_instances(training_instances, test_instances):
    """
    Z-score features with the TRAINING mean/std (zero std is replaced by 1).

    Returns:
        tuple: (standardized training instances, standardized test instances)
    """
    mean = training_instances.mean(axis=0)
    std = training_instances.std(axis=0)
    std[std == 0.0] = 1.0
    return (training_instances - mean) / std, (test_instances - mean) / std


def _standardize_values(training_values, test_values):
    """
    Z-score regression targets with the TRAINING mean/std.

    Returns:
        tuple: (standardized training values, standardized test values)
    """
    mean = training_values.mean()
    std = training_values.std()
    std = std if std != 0.0 else 1.0
    return (training_values - mean) / std, (test_values - mean) / std


def _encode_labels(labels, signed_label):
    """Map labels in {0, 1} to {-1, 1} if signed_label, otherwise keep {0, 1}."""
    return np.where(labels == 1, 1, -1) if signed_label else labels


def _split(instances, targets, test_size, seed, stratify):
    """
    Shuffled train/test split.

    Returns:
        tuple: (training_instances, training_targets, test_instances, test_targets)
    """
    train_x, test_x, train_y, test_y = train_test_split(
        instances,
        targets,
        test_size=test_size,
        random_state=seed,
        stratify=targets if stratify else None,
    )
    return train_x, train_y, test_x, test_y


##############################################################################
#                           BINARY CLASSIFICATION
##############################################################################
# Each source returns raw (instances, labels in {0, 1}); the registry also
# stores the test size (int = number of test samples, float = fraction).
# All binary datasets use a stratified split.


def _binary_gaussians():
    return data_creators.create_gaussians_binary_data(DATA_SEED)


def _binary_jigsaw():
    return data_creators.create_jigsaw_binary_data(DATA_SEED)


def _binary_circle():
    return data_creators.create_circle_binary_data(DATA_SEED)


def _binary_moons():
    return make_moons(n_samples=220, noise=0.2, random_state=DATA_SEED)


def _binary_iris():
    # class 2 (virginica) is positive, classes 0 & 1 are negative
    iris = load_iris()
    return iris.data, (iris.target == 2).astype(int)


_BINARY_SOURCES = {
    "gaussians": (_binary_gaussians, 20),  # synthetic, 200 train / 20 test
    "jigsaw": (_binary_jigsaw, 20),  # synthetic XOR-like, 200 / 20
    "circle": (_binary_circle, 20),  # synthetic rings, 200 / 20
    "moons": (_binary_moons, 20),  # sklearn two moons, 200 / 20
    "iris": (_binary_iris, 0.2),  # Iris (4D), 120 / 30
}


def load_binary_classification_data(dataset, signed_label=False):
    """
    Load a binary classification dataset.

    Args:
        dataset: one of
            - 'gaussians': two 2D Gaussians (200 train, 20 test)
            - 'jigsaw': 2D XOR-like pattern (200 train, 20 test)
            - 'circle': 2D concentric rings (200 train, 20 test)
            - 'moons': 2D two moons (200 train, 20 test)
            - 'iris': Iris, class 2 vs. rest, 4D (120 train, 30 test)
        signed_label: True for labels {-1, 1}, False for {0, 1}

    Returns:
        tuple: (training_instances, training_labels, test_instances, test_labels),
        instances standardized with training statistics
    """
    _check_dataset(dataset, _BINARY_SOURCES)

    source, test_size = _BINARY_SOURCES[dataset]
    instances, labels = source()
    train_x, train_y, test_x, test_y = _split(
        instances, labels, test_size, DATA_SEED, stratify=True
    )

    train_x, test_x = _standardize_instances(train_x, test_x)
    return (
        train_x,
        _encode_labels(train_y, signed_label),
        test_x,
        _encode_labels(test_y, signed_label),
    )


##############################################################################
#                               REGRESSION
##############################################################################


def _regression_plane():
    return data_creators.create_plane_regression_data(DATA_SEED)


def _regression_diabetes():
    diabetes = load_diabetes()
    return diabetes.data, diabetes.target


_REGRESSION_SOURCES = {
    "plane": (_regression_plane, 20),  # synthetic plane (200 / 20)
    "diabetes": (_regression_diabetes, 0.2),  # Diabetes, 10D (353 / 89)
}


def load_regression_data(dataset):
    """
    Load a regression dataset.

    Args:
        dataset: one of
            - 'plane': y = 5*x1 - 3*x2 + 2 + noise (200 train, 20 test)
            - 'diabetes': Diabetes, 10D (353 train, 89 test)

    Returns:
        tuple: (training_instances, training_values, test_instances, test_values),
        instances and values standardized with training statistics
    """
    _check_dataset(dataset, _REGRESSION_SOURCES)

    source, test_size = _REGRESSION_SOURCES[dataset]
    instances, values = source()
    train_x, train_y, test_x, test_y = _split(
        instances, values, test_size, DATA_SEED, stratify=False
    )

    train_x, test_x = _standardize_instances(train_x, test_x)
    train_y, test_y = _standardize_values(train_y, test_y)
    return train_x, train_y, test_x, test_y


##############################################################################
#                                CLUSTERING
##############################################################################


def _clustering_gaussians():
    return data_creators.create_gaussians_clustering_data(CLUSTERING_DATA_SEED)


def _clustering_iris():
    iris = load_iris()
    return iris.data, iris.target


_CLUSTERING_SOURCES = {
    "gaussians": (_clustering_gaussians, 100),  # 3 clusters (1000 / 100)
    "iris": (_clustering_iris, 0.2),  # 3 clusters (120 / 30)
}


def load_clustering_data(dataset):
    """
    Load a clustering dataset (labels are only for evaluation).

    Args:
        dataset: one of
            - 'gaussians': 2D Gaussian mixture, 3 clusters (1000 train, 100 test)
            - 'iris': Iris, 4D, 3 clusters (120 train, 30 test)

    Returns:
        tuple: (training_samples, training_labels, test_samples, test_labels, num_clusters),
        samples standardized with training statistics
    """
    _check_dataset(dataset, _CLUSTERING_SOURCES)

    source, test_size = _CLUSTERING_SOURCES[dataset]
    samples, labels = source()
    train_x, train_y, test_x, test_y = _split(
        samples, labels, test_size, CLUSTERING_DATA_SEED, stratify=True
    )

    num_clusters = len(np.unique(np.concatenate([train_y, test_y])))
    train_x, test_x = _standardize_instances(train_x, test_x)
    return train_x, train_y, test_x, test_y, num_clusters


##############################################################################
#                  1D GAUSSIAN MIXTURE FOR GENERATIVE MODELS
##############################################################################


def load_gaussian_mixture_1d_data(num_samples, device="cpu"):
    """
    Sample the 1D Gaussian mixture (30% at -5, 70% at +5).

    Returns:
        data: tensor of shape (num_samples, 1)
    """
    return data_creators.create_gaussian_mixture_1d_data(num_samples, DATA_SEED, device)


def gaussian_mixture_1d_pdf(x):
    """Analytic density of the 1D mixture (numpy in, numpy out; for plotting)."""
    return data_creators.gaussian_mixture_1d_pdf(x)


def gaussian_mixture_1d_score(x):
    """Analytic score of the 1D mixture, x: tensor (num_points, 1)."""
    return data_creators.gaussian_mixture_1d_score(x)


##############################################################################
#                         MULTICLASS CLASSIFICATION
##############################################################################
# No creator: all multiclass datasets are standard datasets (sklearn, Iris).


def _multiclass_classification():
    return make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=DATA_SEED,
    )


def _multiclass_blobs():
    return make_blobs(
        n_samples=300,
        n_features=2,
        centers=3,
        cluster_std=1.0,
        random_state=DATA_SEED,
    )


def _multiclass_iris():
    iris = load_iris()
    return iris.data, iris.target


_MULTICLASS_SOURCES = {
    "classification": _multiclass_classification,  # 20D, 4 classes (800 / 200)
    "blobs": _multiclass_blobs,  # 2D, 3 classes (240 / 60)
    "iris": _multiclass_iris,  # 4D, 3 classes (120 / 30)
}


def load_multiclass_classification_data(dataset):
    """
    Load a multiclass classification dataset.

    Args:
        dataset: one of
            - 'classification': sklearn make_classification, 20D, 4 classes (800 train, 200 test)
            - 'blobs': sklearn make_blobs, 2D, 3 classes (240 train, 60 test)
            - 'iris': Iris, 4D, 3 classes (120 train, 30 test)

    Returns:
        tuple: (training_instances, training_labels, test_instances, test_labels),
        labels in {0, ..., k-1}, instances standardized with training statistics
    """
    _check_dataset(dataset, _MULTICLASS_SOURCES)

    instances, labels = _MULTICLASS_SOURCES[dataset]()
    train_x, train_y, test_x, test_y = _split(
        instances, labels, 0.2, DATA_SEED, stratify=True
    )

    train_x, test_x = _standardize_instances(train_x, test_x)
    return train_x, train_y, test_x, test_y


##############################################################################
#                                  MNIST
##############################################################################
# No creator: MNIST comes from torchvision. torchvision is imported inside
# the functions, so it is only needed when MNIST is actually used.
# (torch itself is already imported through data_creators.)


def _mnist_dataset(train, tanh_range=False):
    """MNIST torchvision dataset, downloaded to DATA_DIR if missing."""
    from torchvision import datasets, transforms

    steps = [transforms.ToTensor()]
    if tanh_range:
        steps.append(transforms.Normalize((0.5,), (0.5,)))  # pixels -> [-1, 1]
    return datasets.MNIST(
        root=str(DATA_DIR),
        train=train,
        transform=transforms.Compose(steps),
        download=True,
    )


def load_mnist_classification_data(batch_size):
    """
    Load MNIST for image classification (pixels in [0, 1]).

    Args:
        batch_size: positive integer

    Returns:
        tuple: (training_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    training_loader = DataLoader(
        _mnist_dataset(train=True), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        _mnist_dataset(train=False), batch_size=batch_size, shuffle=False
    )
    return training_loader, test_loader


def load_mnist_generation_data(batch_size, tanh_range=False):
    """
    Load MNIST training set for image generation.

    Args:
        batch_size: positive integer
        tanh_range: True to scale pixels to [-1, 1] (GAN, DDPM, SMLD, FM),
            False to keep [0, 1] (VAE)

    Returns:
        training_loader: DataLoader
    """
    from torch.utils.data import DataLoader

    return DataLoader(
        _mnist_dataset(train=True, tanh_range=tanh_range),
        batch_size=batch_size,
        shuffle=True,
    )
