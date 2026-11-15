##############################################################################
#             SYNTHETIC DATASET CREATORS (NEW DATASETS DEFINED HERE)
##############################################################################
#
# Rules:
#   * Only datasets that are defined by this repository live here.
#     Standard datasets (Iris, Diabetes, MNIST, sklearn make_*) are handled
#     in data_loaders.py.
#   * Creators only GENERATE data: they return RAW data (no train/test split,
#     no standardization) and do no file IO. Those are done by data_loaders.py.
#   * Algorithm code must NOT import this module; use data_loaders.py.
#   * Randomness uses a local np.random.RandomState(seed) (NumPy) or
#     torch.Generator (torch); the global RNG states are never touched.
#
# Label convention: binary labels are {0, 1} with positives (1) listed first.

# torch is imported inside the 1D Gaussian mixture functions only,
# so the NumPy-based creators work without PyTorch installed.
import numpy as np


##############################################################################
#                           HELPER FUNCTIONS
##############################################################################


def _generate_gaussian_points(means, covs, num_samples_per_distribution, rng):
    """
    Generate points from several Gaussian distributions, one block per distribution.

    Args:
        means: sequence of num_distributions mean vectors of length num_features
        covs: sequence of num_distributions covariance matrices,
            each of shape (num_features, num_features)
        num_samples_per_distribution: sequence of num_distributions positive integers
        rng: a np.random.RandomState

    Returns:
        samples: shape (sum(num_samples_per_distribution), num_features)
    """
    return np.vstack(
        [
            rng.multivariate_normal(mean, cov, num_samples)
            for mean, cov, num_samples in zip(means, covs, num_samples_per_distribution)
        ]
    )


def _generate_polar_points(radius_min, radius_max, num_samples, rng):
    """
    Generate 2D points with radius sampled uniformly in [radius_min, radius_max]
    and angle uniformly in [0, 2*pi).

    Args:
        rng: a np.random.RandomState

    Returns:
        points: shape (num_samples, 2)
    """
    theta = rng.uniform(0, 2 * np.pi, num_samples)
    radius = rng.uniform(radius_min, radius_max, num_samples)
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))


def _binary_labels(num_positive, num_negative):
    """Labels in {0, 1}; the first num_positive entries are 1."""
    return np.array([1] * num_positive + [0] * num_negative)


##############################################################################
#                       BINARY CLASSIFICATION DATASETS
##############################################################################


def create_gaussians_binary_data(seed):
    """
    Two 2D Gaussians with different covariances (110 positive + 110 negative).

    Returns:
        tuple: (instances, labels), shapes (220, 2) and (220,), labels in {0, 1}
    """
    rng = np.random.RandomState(seed)

    means = [[2, 2], [2, -1]]  # positive, negative
    covs = [[[1, -1], [-1, 3]], [[1, 0.5], [0.5, 2]]]

    instances = _generate_gaussian_points(means, covs, [110, 110], rng)
    return instances, _binary_labels(110, 110)


def create_jigsaw_binary_data(seed):
    """
    XOR-like pattern: four Gaussians, positives in quadrants 1 & 3,
    negatives in quadrants 2 & 4 (110 positive + 110 negative).

    Returns:
        tuple: (instances, labels), shapes (220, 2) and (220,), labels in {0, 1}
    """
    rng = np.random.RandomState(seed)

    # Jigsaw (XOR-like) Gaussian centres
    jigsaw_pos_means = [[2.5, 2.5], [-2.5, -2.5]]  # quadrants 1 & 3
    jigsaw_neg_means = [[-2.5, 2.5], [2.5, -2.5]]  # quadrants 2 & 4

    cov = [[1.5, 0.3], [0.3, 1.5]]
    pos_instances = _generate_gaussian_points(
        jigsaw_pos_means,
        [cov] * len(jigsaw_pos_means),
        [55] * len(jigsaw_pos_means),
        rng,
    )
    neg_instances = _generate_gaussian_points(
        jigsaw_neg_means,
        [cov] * len(jigsaw_neg_means),
        [55] * len(jigsaw_neg_means),
        rng,
    )
    instances = np.vstack((pos_instances, neg_instances))
    return instances, _binary_labels(110, 110)


def create_circle_binary_data(seed):
    """
    Concentric rings: positives inside radius 2, negatives in radius [4, 6]
    (110 positive + 110 negative).

    Returns:
        tuple: (instances, labels), shapes (220, 2) and (220,), labels in {0, 1}
    """
    rng = np.random.RandomState(seed)

    inner_instances = _generate_polar_points(0, 2, 110, rng)
    outer_instances = _generate_polar_points(4, 6, 110, rng)
    instances = np.vstack((inner_instances, outer_instances))
    return instances, _binary_labels(110, 110)


##############################################################################
#                           REGRESSION DATASETS
##############################################################################


def create_plane_regression_data(seed):
    """
    Noisy plane: y = 5*x1 - 3*x2 + 2 + eps, x ~ U(-2, 2)^2, eps ~ N(0, 0.2^2).

    Returns:
        tuple: (instances, values), shapes (220, 2) and (220,)
    """
    rng = np.random.RandomState(seed)

    weights, bias = np.array([5, -3]), 2
    instances = rng.uniform(low=-2, high=2, size=(220, 2))
    values = instances @ weights + bias + rng.normal(0, 0.2, 220)
    return instances, values


##############################################################################
#                           CLUSTERING DATASETS
##############################################################################


def create_gaussians_clustering_data(seed):
    """
    Three 2D Gaussians with different means/covariances and unbalanced mixing
    coefficients (0.4, 0.5, 0.1) -> (440, 550, 110) samples.

    Returns:
        tuple: (samples, labels), shapes (1100, 2) and (1100,), labels in {0, 1, 2}
    """
    rng = np.random.RandomState(seed)

    means = np.array([[3, 1], [0, -2], [-1, 2]])
    covs = np.array([[[1, 0], [0, 1]], [[2, -1.5], [-1.5, 2]], [[3, 1], [1, 2]]])
    num_samples_per_cluster = [440, 550, 110]

    samples = _generate_gaussian_points(means, covs, num_samples_per_cluster, rng)
    labels = np.repeat(np.arange(len(means)), num_samples_per_cluster)
    return samples, labels


##############################################################################
#                  1D GAUSSIAN MIXTURE (FOR GENERATIVE MODELS)
##############################################################################


def _gaussian_mixture_1d_params(device="cpu"):
    """
    Parameters of the 1D mixture (30% at -5, 70% at +5, std 1).

    Returns:
        tuple: (weights, means, stds), float32 tensors of shape (2,) on device
    """
    import torch

    def as_tensor(values):
        return torch.tensor(values, dtype=torch.float32, device=device)

    return as_tensor([0.3, 0.7]), as_tensor([-5.0, 5.0]), as_tensor([1.0, 1.0])


def create_gaussian_mixture_1d_data(num_samples, seed, device="cpu"):
    """
    Sample from the 1D Gaussian mixture using a local torch.Generator
    (the global torch RNG is not touched).

    Returns:
        data: float32 tensor of shape (num_samples, 1) on device
    """
    import torch

    generator = torch.Generator(device=device).manual_seed(seed)
    weights, means, stds = _gaussian_mixture_1d_params(device)

    components = torch.multinomial(
        weights, num_samples, replacement=True, generator=generator
    )
    noise = torch.randn(
        num_samples, dtype=torch.float32, device=device, generator=generator
    )
    data = means[components] + stds[components] * noise
    return data.reshape(num_samples, 1)


def gaussian_mixture_1d_distribution(device="cpu"):
    """
    Target two-component 1D Gaussian mixture (30% at -5, 70% at +5, std 1).
    Used for the analytic density and score; sampling is done separately in
    create_gaussian_mixture_1d_data.

    Returns:
        distribution: a torch.distributions.MixtureSameFamily (float32)
    """
    import torch

    weights, means, stds = _gaussian_mixture_1d_params(device)
    return torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(weights),
        torch.distributions.Normal(means, stds),
    )


def gaussian_mixture_1d_pdf(x):
    """
    Analytic density of the 1D mixture (for plotting).

    Args:
        x: evaluation points, numpy array (or tensor) of any shape;
            converted to float32 internally

    Returns:
        pdf: float32 numpy array with the same shape as x
    """
    import torch

    points = torch.as_tensor(x, dtype=torch.float32)
    pdf = gaussian_mixture_1d_distribution(device=points.device).log_prob(points).exp()
    return pdf.detach().cpu().numpy()


def gaussian_mixture_1d_score(x):
    """
    Analytic score (gradient of the log-density) of the 1D mixture.

    Args:
        x: torch tensor of shape (num_points, 1)

    Returns:
        score: float32 torch tensor of shape (num_points, 1), no autograd graph
    """
    import torch

    points = x.detach().to(torch.float32).reshape(-1).requires_grad_(True)
    log_prob = (
        gaussian_mixture_1d_distribution(device=points.device).log_prob(points).sum()
    )
    score = torch.autograd.grad(log_prob, points)[0]
    return score.reshape(-1, 1)