##############################################################################
#                    DATASET CREATION FOR UNSUPERVISED LEARNING
##############################################################################

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Random seed for reproducibility
DATA_SEED = 16


##############################################################################
#                           HELPER FUNCTIONS
##############################################################################


def _standardize_samples(training_samples, test_samples):
    """
    Standardize features to zero mean and unit variance.

    Args:
        training_samples: training samples, shape (num_training_samples, num_features)
        test_samples: test samples, shape (num_test_samples, num_features)

    Returns:
        tuple: (training_samples, test_samples) where
            - training_samples: standardized training samples,
                shape (num_training_samples, num_features)
            - test_samples: test samples standardized with the training mean and std,
                shape (num_test_samples, num_features)
    """
    mean = training_samples.mean(axis=0)
    std = training_samples.std(axis=0)
    std[std == 0.0] = 1.0
    return (training_samples - mean) / std, (test_samples - mean) / std


def _generate_gaussian_mixture_points(means, covs, num_samples_per_distribution):
    """
    Generate points from several Gaussian distributions, each with its own covariance.

    Args:
        means: cluster centers, a sequence of num_distributions vectors of length num_features
        covs: covariance matrices, one per distribution,
            shape (num_distributions, num_features, num_features)
        num_samples_per_distribution: number of points drawn per distribution,
            a sequence of num_distributions positive integers

    Returns:
        samples: generated points, shape (sum(num_samples_per_distribution), num_features)
    """
    return np.vstack(
        [
            np.random.multivariate_normal(mean, cov, num_samples)
            for mean, cov, num_samples in zip(means, covs, num_samples_per_distribution)
        ]
    )


##############################################################################
#                          CLUSTERING DATASET SELECTION
##############################################################################


def select_clustering_data(dataset):
    """
    Select training and test datasets for clustering.

    Args:
        dataset: dataset name as a string, one of:
            - 'artificial': small manually-defined 2D dataset, 2 clusters (24 train, 16 test)
            - 'gaussians': synthetic 2D Gaussian mixture, 3 clusters (1000 train, 100 test)
            - 'iris': Iris dataset, 4D features, 3 clusters (120 train, 30 test)

    Returns:
        tuple: (training_samples, training_labels, test_samples, test_labels, num_clusters) where
        - training_samples: samples of training data, shape (num_train_samples, num_features)
        - training_labels: cluster labels for training data, shape (num_train_samples,)
        - test_samples: samples of test data, shape (num_test_samples, num_features)
        - test_labels: cluster labels for test data, shape (num_test_samples,)
        - num_clusters: number of clusters

    Raises:
        ValueError: if dataset name not recognized
    """
    generators = {
        "artificial": create_artificial_clustering_data,
        "gaussians": create_gaussians_clustering_data,
        "iris": create_iris_clustering_data,
    }

    if dataset not in generators:
        available = ", ".join(f"'{name}'" for name in generators.keys())
        raise ValueError(f"Unknown dataset: '{dataset}'. Available: {available}")

    return generators[dataset]()


##############################################################################
#                         ARTIFICIAL CLUSTERING DATASET
##############################################################################


def create_artificial_clustering_data():
    """
    Define a small, hand-specified 2D dataset with two clearly separated clusters.

    Returns:
        tuple: (training_samples, training_labels, test_samples, test_labels, num_clusters) where
        - training_samples: samples of training data, shape (24, 2)
        - training_labels: cluster labels for training data, shape (24,)
        - test_samples: samples of test data, shape (16, 2)
        - test_labels: cluster labels for test data, shape (16,)
        - num_clusters: 2
    """
    num_clusters = 2

    # Cluster 0: 12 points
    cluster_0_train = np.array(
        [
            [2.0, 2.5],
            [3.0, 2.0],
            [4.0, 2.5],
            [2.5, 3.5],
            [3.5, 3.0],
            [4.5, 3.5],
            [2.0, 4.0],
            [3.0, 4.5],
            [4.0, 4.0],
            [2.5, 2.0],
            [3.5, 4.0],
            [4.5, 2.5],
        ]
    )

    # Cluster 1: exact mirror image of cluster 0 through the origin.
    cluster_1_train = -cluster_0_train

    training_samples = np.vstack([cluster_0_train, cluster_1_train])
    training_labels = np.array([0] * 12 + [1] * 12)

    # Test set: 8 points per cluster, distinct from the training points but well
    # inside the same regions
    cluster_0_test = np.array(
        [
            [2.2, 3.2],
            [3.8, 2.8],
            [3.2, 4.2],
            [4.2, 3.8],
            [2.8, 2.2],
            [4.0, 3.0],
            [3.0, 3.0],
            [2.5, 4.5],
        ]
    )
    cluster_1_test = -cluster_0_test

    test_samples = np.vstack([cluster_0_test, cluster_1_test])
    test_labels = np.array([0] * 8 + [1] * 8)

    # Standardize training and test samples
    training_samples, test_samples = _standardize_samples(
        training_samples, test_samples
    )
    return (
        training_samples,
        training_labels,
        test_samples,
        test_labels,
        num_clusters,
    )


##############################################################################
#                           THREE GAUSSIANS DATASET
##############################################################################


def create_gaussians_clustering_data():
    """
    Create 2D Gaussian dataset for clustering (1000 train, 100 test, 3 clusters).

    Sample data of three clusters from three Gaussian distributions with different means
    and covariances. The mixing coefficients are unbalanced (0.4, 0.5, 0.1).

    Returns:
        tuple: (training_samples, training_labels, test_samples, test_labels, num_clusters)
            - training_samples: samples of training data, shape (1000, 2)
            - training_labels: cluster labels for training data, shape (1000,)
            - test_samples: samples of test data, shape (100, 2)
            - test_labels: cluster labels for test data, shape (100,)
            - num_clusters: 3
    """
    # Reset global random number generator
    np.random.seed(DATA_SEED)

    num_clusters = 3

    # Define means and co-variances of three Gaussian distributions
    means = np.array([[3, 1], [0, -2], [-1, 2]])
    covs = np.array([[[1, 0], [0, 1]], [[2, -1.5], [-1.5, 2]], [[3, 1], [1, 2]]])

    # Unbalanced mixing coefficients (0.4, 0.5, 0.1) -> samples per cluster
    num_samples_per_cluster = [440, 550, 110]

    # Generate data, one Gaussian cluster at a time
    samples = _generate_gaussian_mixture_points(means, covs, num_samples_per_cluster)
    labels = np.repeat(np.arange(num_clusters), num_samples_per_cluster)

    # Split data to training and test data (shuffled and cluster-balanced)
    training_samples, test_samples, training_labels, test_labels = train_test_split(
        samples,
        labels,
        test_size=100,
        random_state=DATA_SEED,
        stratify=labels,
    )

    # Standardize training and test samples
    training_samples, test_samples = _standardize_samples(
        training_samples, test_samples
    )

    return (
        training_samples,
        training_labels,
        test_samples,
        test_labels,
        num_clusters,
    )


##############################################################################
#                                 IRIS DATASET
##############################################################################


def create_iris_clustering_data():
    """
    Create Iris dataset for clustering.

    Load the Iris dataset from sklearn and create train/test split.

    Returns:
        tuple: (training_samples, training_labels, test_samples, test_labels, num_clusters)
            - 120 training samples, 4 features
            - 30 test samples, 4 features
            - 3 clusters (labels: 0, 1, 2)
            - num_clusters: 3
    """
    # Load Iris dataset
    iris = load_iris()
    samples, labels = iris.data, iris.target
    num_clusters = len(np.unique(labels))

    # Split into training and test sets with stratification (80/20 split)
    training_samples, test_samples, training_labels, test_labels = train_test_split(
        samples, labels, test_size=0.2, random_state=DATA_SEED, stratify=labels
    )

    # Standardize training and test samples
    training_samples, test_samples = _standardize_samples(
        training_samples, test_samples
    )

    return (
        training_samples,
        training_labels,
        test_samples,
        test_labels,
        num_clusters,
    )
