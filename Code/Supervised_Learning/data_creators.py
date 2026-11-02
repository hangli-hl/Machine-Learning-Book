##############################################################################
#                  DATASET CREATION FOR SUPERVISED LEARNING
##############################################################################


# Data conventions:
#   training_instances, test_instances, shape (num_instances, num_features)
#   training_labels, test_labels: for classification, shape (num_instances,)
#       binary: {0, 1} or {-1, 1}, multiclass: {0, 1, ..., num_classes-1}
#   training_values, test_values: for regression, shape (num_instances,), float

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    load_iris,
    load_diabetes,
    make_moons,
    make_classification,
    make_blobs,
)

# Random seed for reproducibility
DATA_SEED = 28


##############################################################################
#                           HELPER FUNCTIONS
##############################################################################


def _generate_polar_points(radius_min, radius_max, num_samples):
    """
    Generate 2D points with radius sampled uniformly (ring) using polar coordinates.

    Args:
        radius_min: inner radius of the annulus, a non-negative float
        radius_max: outer radius of the annulus, a float with radius_max >= radius_min
        num_samples: number of points to generate, a positive integer

    Returns:
        instances: generated 2D points, shape (num_samples, 2)
    """
    theta = np.random.uniform(0, 2 * np.pi, num_samples)
    radius = np.random.uniform(radius_min, radius_max, num_samples)
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))


def _generate_gaussian_points(means, cov, num_samples_per_distribution):
    """
    Generate points from several Gaussian distributions sharing one covariance.

    Args:
        means: cluster centres, a sequence of num_distributions vectors of
            length num_features, e.g. [[2.5, 2.5], [-2.5, -2.5]]
        cov: covariance matrix shared by all distributions, shape (num_features, num_features)
        num_samples_per_distribution: number of points drawn per mean, a positive integer

    Returns:
        instances: generated points,
            shape (num_distributions * num_samples_per_distribution, num_features)
    """
    return np.vstack(
        [
            np.random.multivariate_normal(mean, cov, num_samples_per_distribution)
            for mean in means
        ]
    )


def _create_binary_labels(n_pos, n_neg, signed_label):
    """
    Create a label array with all positive labels before the negatives.

    Args:
        n_pos: number of positive labels, a non-negative integer
        n_neg: number of negative labels, a non-negative integer
        signed_label: if True, use signed labels {1, -1};
            otherwise use binary labels {1, 0}

    Returns:
        labels: class labels of shape (n_pos + n_neg, ),
            values in {1, -1} if signed_label is True, otherwise {1, 0}
    """
    pos_label, neg_label = _get_labels(signed_label)
    return np.array([pos_label] * n_pos + [neg_label] * n_neg)


def _get_labels(signed_label):
    """
    Get positive and negative label values based on signed_label flag.

    Args:
        signed_label: if True, use signed labels {1, -1};
            otherwise use binary labels {1, 0}

    Returns:
        tuple: (positive_label, negative_label) where
            - positive_label: the value used for positive samples, always 1
            - negative_label: the value used for negative samples,
              -1 if signed_label is True, 0 otherwise
    """
    return (1, -1) if signed_label else (1, 0)


def _standardize_instances(training_instances, test_instances):
    """
    Standardize features to zero mean and unit variance (z-score scaling).

    Args:
        training_instances: training instances, shape (num_training_instances, num_features)
        test_instances: test instances, shape (num_test_instances, num_features)

    Returns:
        tuple: (training_instances, test_instances) where
            - training_instances: standardized training instances,
                shape (num_training_instances, num_features)
            - test_instances: test instances standardized with the training mean and std,
                shape (num_test_instances, num_features)
    """
    mean = training_instances.mean(axis=0)
    std = training_instances.std(axis=0)
    std[std == 0.0] = 1.0
    return (training_instances - mean) / std, (test_instances - mean) / std


def _standardize_values(training_values, test_values):
    """
    Standardize values of training and test sets to zero mean and unit variance.

    Args:
        training_values: training values, shape (num_instances,)
        test_values: test values, shape (num_instances,)

    Returns:
        tuple: (training_values, test_values) where
            - training_values: standardized training values, shape (num_instances,)
            - test_values: test values standardized with the training mean and std,
                shape (num_instances,)
    """
    mean = training_values.mean()
    std = training_values.std()
    std = std if std != 0.0 else 1.0

    return (training_values - mean) / std, (test_values - mean) / std


##############################################################################
#                           BINARY DATASET SELECTION
##############################################################################


def select_binary_classification_data(dataset, signed_label=False):
    """
    Select training and test datasets for binary classification.

    Args:
        dataset: dataset name as a string, one of:
            - 'artificial': small manually-defined 2D dataset (14 train, 10 test)
            - 'gaussians': synthetic 2D Gaussian dataset (200 train, 20 test)
            - 'jigsaw': synthetic 2D xor pattern dataset (200 train, 20 test)
            - 'iris': Iris dataset with 4D features (120 train, 30 test)
            - 'circle': synthetic 2D circular pattern dataset (200 train, 20 test)
            - 'moons': synthetic 2D two moons dataset (200 train, 20 test)
        signed_label: whether to use signed labels {-1, 1} or unsigned labels {0, 1}

    Returns:
        tuple: (training_instances, training_labels, test_instances, test_labels) where
            - training_instances: shape (num_training_instances, num_features)
            - training_labels: shape (num_training_instances,), labels in {-1, 1} or {0, 1}
            - test_instances: shape (num_test_instances, num_features)
            - test_labels: shape (num_test_instances,), labels in {-1, 1} or {0, 1}
    """
    # Dataset generator registry
    generators = {
        "artificial": create_artificial_binary_data,
        "gaussians": create_gaussians_binary_data,
        "jigsaw": create_jigsaw_binary_data,
        "iris": create_iris_binary_data,
        "circle": create_circle_binary_data,
        "moons": create_moons_binary_data,
    }

    # Validate dataset name
    if dataset not in generators:
        available = ", ".join(f"'{name}'" for name in generators.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset}'. " f"Available datasets: {available}"
        )

    # Dispatch to appropriate generator
    return generators[dataset](signed_label)


##############################################################################
#                         ARTIFICIAL CLASSIFICATION DATASET
##############################################################################


def create_artificial_binary_data(signed_label=False):
    """Define small 2D dataset for binary classification (14 train, 10 test)."""
    # Reset global random number generator so that data is independent of caller state
    np.random.seed(DATA_SEED)

    # Define training data
    training_instances = np.array(
        [
            [5, 9],
            [3, 12],
            [-1, 12],
            [2, 10],
            [1, 12],
            [2, -3],
            [4, 4.5],  # positive
            [1, 1],
            [1.5, -3],
            [3, -2],
            [4, -5],
            [3.5, 8],
            [-1, 4],
            [-1, -1],  # negative
        ]
    )
    training_labels = _create_binary_labels(7, 7, signed_label)

    # Define test data
    test_instances = np.array(
        [
            [4, 10],
            [2, 13],
            [0, 11],
            [5, 9.5],
            [4, 5],  # positive
            [0, 0],
            [2.5, 1],
            [1, -5],
            [-1, 2],
            [3.5, 6.5],  # negative
        ]
    )
    test_labels = _create_binary_labels(5, 5, signed_label)

    # Standardize training and test instances
    training_instances, test_instances = _standardize_instances(
        training_instances, test_instances
    )

    return training_instances, training_labels, test_instances, test_labels


##############################################################################
#                           TWO GAUSSIANS DATASET
##############################################################################


def create_gaussians_binary_data(signed_label=False):
    """Create 2D Gaussian dataset for binary classification (200 train, 20 test)."""

    # Reset global random number generator so that data is independent of caller state
    np.random.seed(DATA_SEED)

    # Define means and co-variances of two Gaussian distributions
    pos_mean, pos_cov = [2, 5], [[1, -1], [-1, 3]]
    neg_mean, neg_cov = [3, -2], [[1, 0.5], [0.5, 2]]

    # Generate data
    pos_instances = np.random.multivariate_normal(pos_mean, pos_cov, 110)
    neg_instances = np.random.multivariate_normal(neg_mean, neg_cov, 110)
    instances = np.vstack((pos_instances, neg_instances))
    labels = _create_binary_labels(110, 110, signed_label)

    # Split data to training and test data (shuffled and class-balanced)
    training_instances, test_instances, training_labels, test_labels = train_test_split(
        instances,
        labels,
        test_size=20,
        random_state=DATA_SEED,
        stratify=labels,
    )

    # Standardize training and test instances
    training_instances, test_instances = _standardize_instances(
        training_instances, test_instances
    )

    return training_instances, training_labels, test_instances, test_labels


##############################################################################
#                              JIGSAW DATASET
##############################################################################


def create_jigsaw_binary_data(signed_label=False):
    """Create 2D jigsaw (XOR-like pattern) dataset (200 train, 20 test)."""

    # Reset global random number generator so that data is independent of caller state
    np.random.seed(DATA_SEED)

    # Define Gaussian distributions approximately in four quadrants
    pos_means = [[2.5, 2.5], [-2.5, -2.5]]  # quadrants 1 & 3
    neg_means = [[-2.5, 2.5], [1.5, -1.5]]  # quadrants 2 & 4
    cov = [[1.5, 0.3], [0.3, 1.5]]

    # Generate data
    pos_instances = _generate_gaussian_points(pos_means, cov, 55)
    neg_instances = _generate_gaussian_points(neg_means, cov, 55)
    instances = np.vstack((pos_instances, neg_instances))
    labels = _create_binary_labels(110, 110, signed_label)

    # Split data to training and test data (shuffled and class-balanced)
    training_instances, test_instances, training_labels, test_labels = train_test_split(
        instances,
        labels,
        test_size=20,
        random_state=DATA_SEED,
        stratify=labels,
    )

    # Standardize training and test instances
    training_instances, test_instances = _standardize_instances(
        training_instances, test_instances
    )

    return training_instances, training_labels, test_instances, test_labels


##############################################################################
#                           SKLEARN IRIS BINARY DATASET
##############################################################################


def create_iris_binary_data(signed_label=False):
    """Create Iris binary classification dataset (120 train, 30 test, 4 features)."""

    # Load data from sklearn
    iris = load_iris()

    # Get binary label format
    pos_label, neg_label = _get_labels(signed_label)

    # Convert class labels to binary: class 2 vs classes 0&1
    labels = np.where(iris.target == 2, pos_label, neg_label)

    # Split data to training and test data
    training_instances, test_instances, training_labels, test_labels = train_test_split(
        iris.data, labels, test_size=0.2, random_state=DATA_SEED, stratify=labels
    )

    # Standardize training and test instances
    training_instances, test_instances = _standardize_instances(
        training_instances, test_instances
    )

    return training_instances, training_labels, test_instances, test_labels


##############################################################################
#                              CIRCLE DATASET
##############################################################################


def create_circle_binary_data(signed_label=False):
    """Create 2D circle dataset for binary classification (200 train, 20 test)."""

    # Reset global random number generator so that data is independent of caller state
    np.random.seed(DATA_SEED)

    # Generate data
    inner_instances = _generate_polar_points(0, 2, 110)
    outer_instances = _generate_polar_points(4, 6, 110)
    instances = np.vstack((inner_instances, outer_instances))
    labels = _create_binary_labels(110, 110, signed_label)

    # Split data to training and test data (shuffled and class-balanced)
    training_instances, test_instances, training_labels, test_labels = train_test_split(
        instances,
        labels,
        test_size=20,
        random_state=DATA_SEED,
        stratify=labels,
    )

    # Standardize training and test instances
    training_instances, test_instances = _standardize_instances(
        training_instances, test_instances
    )

    return training_instances, training_labels, test_instances, test_labels


##############################################################################
#                              MOONS DATASET
##############################################################################


def create_moons_binary_data(signed_label=False):
    """Create moons dataset for binary classification (200 train, 20 test)."""

    # Reset global random number generator so that data is independent of caller state
    np.random.seed(DATA_SEED)

    # Load data from sklearn
    instances, labels_ori = make_moons(n_samples=220, noise=0.2, random_state=DATA_SEED)

    # Get positive and negative label formats based on signed_label flag
    pos_label, neg_label = _get_labels(signed_label)

    # Convert original 0/1 labels to desired label format
    labels = np.where(labels_ori == 0, neg_label, pos_label)

    # Split data to training and test data
    training_instances, test_instances, training_labels, test_labels = train_test_split(
        instances,
        labels,
        test_size=20,
        random_state=DATA_SEED,
        stratify=labels,
    )

    # Standardize training and test instances
    training_instances, test_instances = _standardize_instances(
        training_instances, test_instances
    )

    return training_instances, training_labels, test_instances, test_labels


##############################################################################
#                          MULTICLASS DATASET SELECTION
##############################################################################


def select_multiclass_classification_data(dataset):
    """
    Create training and test datasets for multiclass classification.

    Available datasets:
        'classification': 20 features, 4 classes, 800 train, 200 test
        'blobs': 2 features, 3 classes, 240 train, 60 test
        'iris': 4 features, 3 classes, 120 train, 30 test

    Args:
        dataset: dataset name

    Returns:
        training_instances: shape (num_train, num_features)
        training_labels: shape (num_train,)
        test_instances: shape (num_test, num_features)
        test_labels: shape (num_test,)
    """
    generators = {
        "classification": create_classification_multiclass_data,
        "blobs": create_blobs_multiclass_data,
        "iris": create_iris_multiclass_data,
    }

    if dataset not in generators:
        available = ", ".join(f"'{name}'" for name in generators.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset}'. Available datasets: {available}"
        )

    return generators[dataset]()


##############################################################################
#                         SKLEARN CLASSIFICATION DATASET
##############################################################################


def create_classification_multiclass_data():
    """Create synthetic high-D multiclass dataset (800 train, 200 test, 20 features, 4 classes)."""

    # Load data from sklearn
    instances, labels = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=DATA_SEED,
    )

    # Split data to training and test data
    training_instances, test_instances, training_labels, test_labels = train_test_split(
        instances, labels, test_size=0.2, random_state=DATA_SEED, stratify=labels
    )

    # Standardize training and test instances
    training_instances, test_instances = _standardize_instances(
        training_instances, test_instances
    )

    return training_instances, training_labels, test_instances, test_labels


##############################################################################
#                            SKLEARN BLOBS DATASET
##############################################################################


def create_blobs_multiclass_data():
    """Create synthetic 2D blob dataset (240 train, 60 test, 2D, 3 classes)."""

    # Load data from sklearn
    instances, labels = make_blobs(
        n_samples=300,
        n_features=2,
        centers=3,
        cluster_std=1.0,
        random_state=DATA_SEED,
    )

    # Split data into training and test data
    training_instances, test_instances, training_labels, test_labels = train_test_split(
        instances,
        labels,
        test_size=0.2,
        random_state=DATA_SEED,
        stratify=labels,
    )

    # Standardize training and test instances
    training_instances, test_instances = _standardize_instances(
        training_instances, test_instances
    )

    return training_instances, training_labels, test_instances, test_labels


##############################################################################
#                        SKLEARN IRIS MULTICLASS DATASET
##############################################################################


def create_iris_multiclass_data():
    """Create Iris multiclass dataset (120 train, 30 test, 4 features, 3 classes)."""

    # Load data from sklearn
    iris = load_iris()

    # Split data into training and test data
    training_instances, test_instances, training_labels, test_labels = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=DATA_SEED,
        stratify=iris.target,
    )

    # Standardize training and test instances
    training_instances, test_instances = _standardize_instances(
        training_instances, test_instances
    )

    return training_instances, training_labels, test_instances, test_labels


##############################################################################
#                         REGRESSION DATASET SELECTION
##############################################################################


def select_regression_data(dataset):
    """
    Create training and test datasets for regression.

    Args:
        dataset: dataset name as a string, one of:
            - 'artificial': small manually-defined 2D dataset (14 train, 10 test)
            - 'plane': synthetic 2D dataset (200 train, 20 test)
            - 'diabetes': Diabetes dataset, 10 features (353 train, 89 test)

    Returns:
        tuple: (training_instances, training_values, test_instances, test_values) where
        - training_instances: shape (num_training_instances, num_features)
        - training_values: shape (num_training_instances,)
        - test_instances: shape (num_test_instances, num_features)
        - test_values: shape (num_test_instances,)
    """
    # Dataset generator registry
    generators = {
        "artificial": create_artificial_regression_data,
        "plane": create_plane_regression_data,
        "diabetes": create_diabetes_regression_data,
    }

    # Validate dataset name
    if dataset not in generators:
        available = ", ".join(f"'{name}'" for name in generators.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset}'. " f"Available datasets: {available}"
        )

    # Dispatch to appropriate generator
    return generators[dataset]()


##############################################################################
#                         ARTIFICIAL REGRESSION DATASET
##############################################################################


def create_artificial_regression_data():
    """Define small 2D dataset for regression (14 train, 10 test)."""
    # Reset global random number generator so that data is independent of caller state
    np.random.seed(DATA_SEED)

    # Define training data
    training_instances = np.array(
        [
            [5, 9],
            [3, 12],
            [-1, 12],
            [2, 10],
            [1, 12],
            [2, -3],
            [4, 4.5],
            [1, 1],
            [1.5, -3],
            [3, -2],
            [4, -5],
            [3.5, 8],
            [-1, 4],
            [-1, -1],
        ]
    )
    training_values = np.array([1.5, 1.5, 0, 1, 1, 0, 0, -1, -1, -1, -1, 1, -1, -1.5])

    # Define test data
    test_instances = np.array(
        [
            [4, 11],
            [0, 12],
            [2, 9],
            [5, 6],
            [3, 5],
            [1, 3],
            [-1, 8],
            [0, -2],
            [2.5, -4],
            [-1, 0],
        ]
    )
    test_values = np.array([1.5, 0.5, 0.5, 1.0, 0.5, -0.5, -0.5, -1.0, -1.0, -1.5])

    # Standardize training and test instances
    training_instances, test_instances = _standardize_instances(
        training_instances, test_instances
    )

    # Standardize training and test values
    training_values, test_values = _standardize_values(training_values, test_values)

    return training_instances, training_values, test_instances, test_values


##############################################################################
#                               PLANE DATASET
##############################################################################


def create_plane_regression_data():
    """Create 2D linear regression with noise: y = 5·x1 - 3·x2 + 2 + ε (200 train, 20 test).
    Features and values are then standardized.
    """

    # Reset global random number generator so that data is independent of caller state
    np.random.seed(DATA_SEED)

    # Define plane
    weights, bias = np.array([5, -3]), 2

    # Generate data
    instances = np.random.uniform(low=-2, high=2, size=(220, 2))
    values = instances @ weights + bias + np.random.normal(0, 0.2, 220)

    # Split data to training and test data (shuffled)
    training_instances, test_instances, training_values, test_values = train_test_split(
        instances,
        values,
        test_size=20,
        random_state=DATA_SEED,
    )

    # Standardize training and test instances
    training_instances, test_instances = _standardize_instances(
        training_instances, test_instances
    )

    # Standardize training and test values
    training_values, test_values = _standardize_values(training_values, test_values)

    return training_instances, training_values, test_instances, test_values


##############################################################################
#                               DIABETES DATASET
##############################################################################


def create_diabetes_regression_data():
    """Create Diabetes regression dataset (353 train, 89 test, 10 features)."""

    # Load data from sklearn
    diabetes = load_diabetes()

    # Split data into training and test data (shuffled)
    training_instances, test_instances, training_values, test_values = train_test_split(
        diabetes.data, diabetes.target, test_size=0.2, random_state=DATA_SEED
    )

    # Standardize training and test instances
    training_instances, test_instances = _standardize_instances(
        training_instances, test_instances
    )

    # Standardize training and test values
    training_values, test_values = _standardize_values(training_values, test_values)

    return training_instances, training_values, test_instances, test_values
