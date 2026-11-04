##############################################################################
#                  DATASET GENERATION FOR DEEP LEARNING
##############################################################################


import numpy as np

from sklearn.datasets import make_moons
from sklearn.datasets import make_classification, make_blobs, load_iris
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Directory of training data
DATA_DIR = "./data"
# Random seed for reproducibility, used for data
DATA_SEED = 28


##############################################################################
#                             HELPER FUNCTIONS
##############################################################################


def _generate_polar_points(radius_min, radius_max, num_samples):
    """
    Generate 2D points uniformly in an annulus (ring) using polar coordinates.

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
        cov: covariance matrix shared by all distributions,
            shape (num_features, num_features)
        num_samples_per_distribution: number of points drawn per mean,
            a positive integer

    Returns:
        instances: generated points, shape
            (num_distributions * num_samples_per_distribution, num_features)
    """
    return np.vstack(
        [
            np.random.multivariate_normal(mean, cov, num_samples_per_distribution)
            for mean in means
        ]
    )


def _create_binary_labels(num_positive, num_negative):
    """
    Create a binary label array with all positive labels before the negatives.

    Args:
        num_positive: number of positive labels (1), a non-negative integer
        num_negative: number of negative labels (0), a non-negative integer

    Returns:
        labels: class labels of shape (num_positive + num_negative, ), labels in {0, 1},
            with the first num_positive entries equal to 1
    """
    return np.array([1] * num_positive + [0] * num_negative)


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
            - test_instances: test instances standardized with the training
              mean and std, shape (num_test_instances, num_features)
    """
    mean = training_instances.mean(axis=0)
    std = training_instances.std(axis=0)
    std[std == 0.0] = 1.0
    return (training_instances - mean) / std, (test_instances - mean) / std


##############################################################################
#                  DATASET SELECTION FOR BINARY CLASSIFICATION
##############################################################################


def select_ffn_binary_data(dataset):
    """
    Select training and test datasets for binary classification.

    Args:
        dataset: dataset name as a string, one of:
        - 'gaussians': synthetic 2D Gaussian dataset (200 train, 20 test)
        - 'jigsaw': synthetic 2D XOR-like pattern dataset (200 train, 20 test)
        - 'circle': synthetic 2D circular pattern dataset (200 train, 20 test)
        - 'moons': synthetic 2D two moons dataset (200 train, 20 test)

    Returns:
        tuple: (training_instances, training_labels, test_instances, test_labels) where
            - training_instances: instances of training data, shape (num_training_instances, num_features)
            - training_labels: class labels for training data, shape (num_training_instances, ), labels in {0, 1}
            - test_instances: instances of test data, shape (num_test_instances, num_features)
            - test_labels: class labels for test data, shape (num_test_instances, ), labels in {0, 1}
    """
    # Dataset generator registry
    generators = {
        "gaussians": create_gaussians_binary_data,
        "jigsaw": create_jigsaw_binary_data,
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
    return generators[dataset]()


##############################################################################
#                           TWO GAUSSIANS DATASET
##############################################################################


def create_gaussians_binary_data():
    """Create 2D Gaussian dataset for binary classification (200 train, 20 test)."""

    # Set random seed for this data creation
    np.random.seed(DATA_SEED)

    # Define means and co-variances of two Gaussian distributions
    pos_mean, pos_cov = [2, 5], [[1, -1], [-1, 3]]
    neg_mean, neg_cov = [3, -2], [[1, 0.5], [0.5, 2]]

    # Generate data
    pos_instances = np.random.multivariate_normal(pos_mean, pos_cov, 110)
    neg_instances = np.random.multivariate_normal(neg_mean, neg_cov, 110)
    instances = np.vstack((pos_instances, neg_instances))
    labels = _create_binary_labels(110, 110)

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


def create_jigsaw_binary_data():
    """Create 2D jigsaw (XOR-like pattern) dataset (200 train, 20 test)."""

    # Set random seed for this data creation
    np.random.seed(DATA_SEED)

    # Define Gaussian distributions approximately in four quadrants
    pos_means = [[2.5, 2.5], [-2.5, -2.5]]  # quadrants 1 & 3
    neg_means = [[-2.5, 2.5], [0.5, -0.5]]  # quadrants 2 & 4
    cov = [[1.5, 0.3], [0.3, 1.5]]

    # Generate data
    pos_instances = _generate_gaussian_points(pos_means, cov, 55)
    neg_instances = _generate_gaussian_points(neg_means, cov, 55)
    instances = np.vstack((pos_instances, neg_instances))
    labels = _create_binary_labels(110, 110)

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
#                              CIRCLE DATASET
##############################################################################


def create_circle_binary_data():
    """Create 2D circle dataset for binary classification (200 train, 20 test)."""

    # Set random seed for this data creation
    np.random.seed(DATA_SEED)

    # Generate data
    inner_instances = _generate_polar_points(0, 2, 110)
    outer_instances = _generate_polar_points(4, 6, 110)
    instances = np.vstack((inner_instances, outer_instances))
    labels = _create_binary_labels(110, 110)

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
#                             SKLEARN MOONS DATASET
##############################################################################


def create_moons_binary_data():
    """Create moons dataset for binary classification (200 train, 20 test)."""

    # Load data from sklearn
    instances, labels = make_moons(n_samples=220, noise=0.2, random_state=DATA_SEED)

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
#                DATASET SELECTION FOR MULTICLASS CLASSIFICATION
##############################################################################


def select_ffn_multiclass_data(dataset):
    """
    Select training and test datasets for multiclass classification.

    Args:
        dataset: dataset name as a string, one of:
        - 'classification': make_classification synthetic dataset, 20D features, 4 classes (800 train, 200 test)
        - 'blobs': make_blobs synthetic dataset, 2D features, 3 classes (240 train, 60 test)
        - 'iris': Iris dataset, 4D features, 3 classes (120 train, 30 test)

    Returns:
        tuple: (training_instances, training_labels, test_instances, test_labels) where
            - training_instances: instances of training data, shape (num_training_instances, num_features)
            - training_labels: class labels for training data, shape (num_training_instances, ), labels in {0, 1, ..., k-1}
            - test_instances: instances of test data, shape (num_test_instances, num_features)
            - test_labels: class labels for test data, shape (num_test_instances, ), labels in {0, 1, ..., k-1}
    """
    # Dataset generator registry
    generators = {
        "classification": create_classification_multiclass_data,
        "blobs": create_blobs_multiclass_data,
        "iris": create_iris_multiclass_data,
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
#                             SKLEARN IRIS DATASET
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
#                              MNIST DATASET
##############################################################################


def load_mnist_classification_data(batch_size):
    """
    Load and process the MNIST dataset for image classification.

    Args:
        batch_size: number of samples per batch, a positive integer

    Returns:
        training_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    transform = transforms.Compose([transforms.ToTensor()])

    training_dataset = datasets.MNIST(
        root=DATA_DIR, train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root=DATA_DIR, train=False, transform=transform, download=True
    )

    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return training_loader, test_loader


def load_mnist_generation_data(batch_size, tanh_range=False):
    """
    Load and preprocess the MNIST dataset for image generation by VAE and GAN.

    Args:
        batch_size: number of samples per batch, a positive integer
        tanh_range: whether to scale pixels to [-1, 1] or keep [0, 1], a boolean,
            set to True for GAN and False for VAE

    Returns:
        training_loader: DataLoader for training data
    """
    if tanh_range:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    dataset = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return training_loader


##############################################################################
#                      1D GAUSSIAN MIXTURE FOR GENERATION
##############################################################################


def _gaussian_mixture_1d(device="cpu"):
    """
    Build the target two-component 1D Gaussian mixture (30% at -5, 70% at +5).

    Args:
        device: device to place the distribution's parameters on

    Returns:
        distribution: a torch MixtureSameFamily distribution
    """
    return torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(torch.tensor([0.3, 0.7], device=device)),
        torch.distributions.Normal(
            torch.tensor([-5.0, 5.0], device=device),
            torch.tensor([1.0, 1.0], device=device),
        ),
    )


def create_gaussian_mixture_1d_data(num_samples, device="cpu"):
    """
    Create 1D Gaussian mixture data with two components (30% at -5, 70% at +5).

    Args:
        num_samples: number of samples to generate, a positive integer
        device: device to place tensors on (CPU or GPU)

    Returns:
        data: 1D Gaussian mixture samples of shape (num_samples, 1)
    """
    torch.manual_seed(DATA_SEED)
    data = _gaussian_mixture_1d(device=device).sample((num_samples, 1))

    return data


def gaussian_mixture_1d_pdf(x):
    """
    Analytic density of the target 1D Gaussian mixture (numpy, for plotting).

    Args:
        x: evaluation points, a numpy array of any shape

    Returns:
        pdf: density values, same shape as x
    """
    points = torch.as_tensor(x, dtype=torch.float32)
    pdf = _gaussian_mixture_1d(device=points.device).log_prob(points).exp()

    return pdf.detach().cpu().numpy()


def gaussian_mixture_1d_score(x):
    """
    Analytic score (gradient of log-density) of the 1D Gaussian mixture.

    Args:
        x: evaluation points, a torch tensor of shape (num_points, 1)

    Returns:
        score: analytic score of shape (num_points, 1)
    """
    points = x.detach().reshape(-1).requires_grad_(True)
    log_prob = _gaussian_mixture_1d(device=points.device).log_prob(points).sum()
    score = torch.autograd.grad(log_prob, points)[0]

    return score.reshape(-1, 1)
