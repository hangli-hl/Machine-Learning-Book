##############################################################################
#                  GAUSSIAN MIXTURE MODEL WITH EM ALGORITHM
##############################################################################


import argparse

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.metrics import adjusted_rand_score

import data_creators as dc
import plot_functions as pf

# Small positive number to avoid division by zero or log(0)
EPS = 1e-6


##############################################################################
#                          GAUSSIAN MIXTURE EM CLASS
##############################################################################


class GaussianMixtureEM:
    """
    Gaussian Mixture Model learned with the EM algorithm.

    The model is the density
        p(x) = Σ_{j=1}^K π_j * N(x | μ_j, Σ_j)
    where π_j is the mixing coefficient of component j, N(x | μ_j, Σ_j) is its
    Gaussian density, and K is the number of components. A sample x is assigned to
    the component with the highest posterior probability, which is the component
    with the highest π_j * N(x | μ_j, Σ_j). Ties are broken in favor of the smallest
    component index.

    The learning algorithm alternates two steps:
        - E-step: compute the responsibilities r_ij = p(z_i = j | x_i) under the
          current parameters
        - M-step: re-estimate coefficients, means and covariances from the
          responsibilities

    Every iteration keeps the log-likelihood
        log L = Σ_i log p(x_i)
    from decreasing (up to the numerical safeguards below), so training converges to
    a local optimum that depends on the initialization.

    All densities are evaluated in log-space and combined with logsumexp.

    Attributes:
        num_components: number of Gaussian components, a positive integer
        num_iterations: maximum number of EM iterations, a positive integer
        convergence_threshold: training stops when the change of the log-likelihood
            per sample drops to this value or below, a positive float
        num_features: number of features, a positive integer
        coefficients: mixing coefficients, shape (num_components,)
        means: component means, shape (num_components, num_features)
        covariances: component covariances,
            shape (num_components, num_features, num_features)
        responsibilities: posterior probabilities of the training samples under the
            final parameters, shape (num_training_samples, num_components)
        assignments: component assignments of the training samples under the final
            parameters, shape (num_training_samples,)
        log_likelihood_history: log-likelihood per recorded snapshot, list of floats
        parameter_history: flattened parameters (means, covariances, coefficients)
            per recorded snapshot, list of 1D arrays. Entry t holds the parameters
            that produced log_likelihood_history[t], so position and color cannot
            drift apart in the trajectory plot.
    """

    def __init__(self, num_components, num_iterations=1000, convergence_threshold=1e-4):
        """
        Initialize a Gaussian mixture model.

        Args:
            num_components: number of Gaussian components, a positive integer
            num_iterations: maximum number of EM iterations, a positive integer
            convergence_threshold: threshold for the per-sample log-likelihood
                change, a positive float
        """
        self.num_components = int(num_components)
        self.num_iterations = int(num_iterations)
        self.convergence_threshold = float(convergence_threshold)

        self.num_features = None
        self.coefficients = None
        self.means = None
        self.covariances = None
        self.responsibilities = None
        self.assignments = None
        self.log_likelihood_history = []
        self.parameter_history = []

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _check_samples(self, samples):
        """
        Check that samples form a 2D array and that there are enough of them.

        Args:
            samples: training samples, shape (num_samples, num_features)

        Returns:
            samples: the samples as a float array, shape (num_samples, num_features)

        Raises:
            ValueError: if samples is not 2D or if num_components is not in
                [1, num_samples]
        """
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2:
            raise ValueError(f"samples must be 2D, got shape {samples.shape}")

        num_samples = samples.shape[0]
        if not 1 <= self.num_components <= num_samples:
            raise ValueError(
                f"num_components must be in [1, {num_samples}], "
                f"got {self.num_components}"
            )
        return samples

    def _init_parameters(self, samples):
        """
        Initialize the means with num_components distinct random samples, the
        covariances with identity matrices and the coefficients uniformly,
        and reset the responsibilities, the assignments and the training histories.

        Args:
            samples: training samples, shape (num_samples, num_features)
        """
        num_samples, num_features = samples.shape
        initial_indices = np.random.choice(
            num_samples, self.num_components, replace=False
        )
        self.num_features = num_features
        self.means = samples[initial_indices]
        self.covariances = np.tile(np.eye(num_features), (self.num_components, 1, 1))
        self.coefficients = np.full(self.num_components, 1 / self.num_components)
        self.responsibilities = None
        self.assignments = None
        self.log_likelihood_history = []
        self.parameter_history = []

    def _record_parameters(self, log_likelihood):
        """
        Append the current parameters and their log-likelihood to the two histories.

        Both histories are appended together so that they stay the same length and
        entry t of one always describes entry t of the other.

        Args:
            log_likelihood: log-likelihood under the current parameters, a float
        """
        # concatenate() returns a copy, so later updates cannot modify the history
        self.parameter_history.append(
            np.concatenate(
                [
                    self.means.flatten(),
                    self.covariances.flatten(),
                    self.coefficients.flatten(),
                ]
            )
        )
        self.log_likelihood_history.append(log_likelihood)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, samples):
        """
        Train the Gaussian mixture model with the EM algorithm.

        The initial parameters and their log-likelihood are recorded first. In each
        iteration, the parameters are updated from the responsibilities (M-step),
        then the responsibilities and the log-likelihood are recomputed under the new
        parameters (E-step) and recorded together with them.

        Training stops when it reaches the specified number of iterations or when
        the change of the log-likelihood per sample drops to the convergence
        threshold or below. The last record always belongs to the final parameters,
        so len(log_likelihood_history) = number of iterations performed + 1. The
        assignments of the training samples are computed under the final parameters.

        Args:
            samples: training samples, shape (num_samples, num_features)
        """
        samples = self._check_samples(samples)
        num_samples = samples.shape[0]
        self._init_parameters(samples)

        # E-step on the initial parameters, then record them
        responsibilities, log_likelihood = self._e_step(samples)
        self._record_parameters(log_likelihood)

        for _ in range(self.num_iterations):
            # M-step
            self._m_step(samples, responsibilities)

            # E-step under the new parameters, then record them with their
            # log-likelihood
            responsibilities, new_log_likelihood = self._e_step(samples)
            self._record_parameters(new_log_likelihood)

            log_likelihood_change = abs(new_log_likelihood - log_likelihood)
            log_likelihood = new_log_likelihood

            # Stop if the log-likelihood no longer changes
            if log_likelihood_change / num_samples <= self.convergence_threshold:
                break

        self.responsibilities = responsibilities
        self.assignments = self.predict(samples)

    def _e_step(self, samples):
        """
        E-step: compute responsibilities and the log-likelihood of the current
        parameters.

        The responsibilities are defined as
            r_ij = p(z_i = j | x_i)
                 = π_j * N(x_i | μ_j, Σ_j) / Σ_{k=1}^K π_k * N(x_i | μ_k, Σ_k)
        where z_i is the latent variable indicating the component of sample x_i.

        The log-likelihood is defined as
            log L = Σ_{i=1}^N log( Σ_{j=1}^K π_j * N(x_i | μ_j, Σ_j) )

        Args:
            samples: input samples, shape (num_samples, num_features)

        Returns:
            responsibilities: posterior probabilities,
                shape (num_samples, num_components)
            log_likelihood: log-likelihood under the current parameters, a float
        """
        log_weighted = self.forward(samples)
        log_total = logsumexp(log_weighted, axis=1, keepdims=True)

        responsibilities = np.exp(log_weighted - log_total)
        log_likelihood = float(np.sum(log_total))

        return responsibilities, log_likelihood

    def _m_step(self, samples, responsibilities):
        """
        M-step: update coefficients, means and covariances from the responsibilities.

        Update formulas:
            π_j = (1/N) * Σ_{i=1}^N r_ij
            μ_j = (Σ_{i=1}^N r_ij * x_i) / (Σ_{i=1}^N r_ij)
            Σ_j = (Σ_{i=1}^N r_ij * (x_i - μ_j)(x_i - μ_j)^T) / (Σ_{i=1}^N r_ij)

        A component that lost all responsibility keeps its previous means and
        covariances. Its coefficient is still updated to reflect that it is no
        longer used.

        Args:
            samples: input samples, shape (num_samples, num_features)
            responsibilities: posterior probabilities,
                shape (num_samples, num_components)
        """
        total_responsibilities = np.sum(responsibilities, axis=0)
        self.coefficients = total_responsibilities / samples.shape[0]

        for j in range(self.num_components):
            if total_responsibilities[j] <= EPS:
                continue

            self.means[j] = (responsibilities[:, j] @ samples) / (
                total_responsibilities[j]
            )

            deviations = samples - self.means[j]
            weighted_deviations = responsibilities[:, j : j + 1] * deviations
            self.covariances[j] = (weighted_deviations.T @ deviations) / (
                total_responsibilities[j]
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, samples):
        """
        Predict component assignments by the highest posterior probability.

        Ties are broken in favor of the smallest component index.

        Args:
            samples: input samples, shape (num_samples, num_features)

        Returns:
            assignments: predicted component assignments in {0, ..., num_components-1},
                shape (num_samples,)
        """
        # The posterior is the weighted density divided by a per-sample constant,
        # so both have the same argmax
        return np.argmax(self.forward(samples), axis=1)

    def forward(self, samples):
        """
        Forward pass: compute the log weighted densities of samples under all
        components.

            log( π_j * N(x_i | μ_j, Σ_j) ) = log(π_j) + log(N(x_i | μ_j, Σ_j))

        Args:
            samples: input samples, shape (num_samples, num_features)

        Returns:
            log_weighted: log_weighted[i, j] = log( π_j * N(x_i | μ_j, Σ_j) ),
                shape (num_samples, num_components)
        """
        samples = np.asarray(samples, dtype=float)
        log_weighted = np.zeros((samples.shape[0], self.num_components))

        for j in range(self.num_components):
            log_density = multivariate_normal.logpdf(
                samples,
                mean=self.means[j],
                cov=self.covariances[j],
                allow_singular=True,
            )
            log_weighted[:, j] = np.log(max(self.coefficients[j], EPS)) + log_density

        return log_weighted


##############################################################################
#                              MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Gaussian Mixture Model (GMM) with the EM algorithm "
        "on a selected dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="gaussians",
        choices=["artificial", "gaussians", "iris"],
        help="Dataset to use: artificial (manual 2D), "
        "gaussians (synthetic 2D Gaussian mixture) or iris (Iris 4D)",
    )
    parser.add_argument(
        "--num-components",
        type=int,
        default=None,
        help="Number of Gaussian components (if not specified, uses the true number "
        "from the dataset)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1000,
        help="Maximum number of EM iterations",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=1e-4,
        help="Stop once the change of the log-likelihood per sample drops "
        "to this value or below",
    )
    args = parser.parse_args()
    print("Train a Gaussian Mixture Model (GMM) with the EM algorithm")

    # Create dataset
    (
        training_samples,
        training_labels,
        test_samples,
        test_labels,
        num_components_in_data,
    ) = dc.select_clustering_data(dataset=args.data)
    num_features = training_samples.shape[1]

    # Use specified num_components or fall back to true number
    num_components = (
        args.num_components
        if args.num_components is not None
        else num_components_in_data
    )

    print(f"Using dataset: {args.data}")
    print(f"Training set: {training_samples.shape[0]} samples, {num_features} features")
    print(f"Test set: {test_samples.shape[0]} samples, {num_features} features")
    print(f"Number of components in data: {num_components_in_data}")
    print(f"Number of components for GMM: {num_components}")

    # Random seed for reproducibility, after data creation
    MODEL_SEED = 32
    np.random.seed(MODEL_SEED)

    # Initialize the Gaussian mixture model
    model = GaussianMixtureEM(
        num_components, args.num_iterations, args.convergence_threshold
    )

    # Train the model and log the log-likelihood
    print(
        f"Training with EM algorithm, max iterations = {model.num_iterations}, "
        f"convergence threshold = {model.convergence_threshold:.2e}"
    )
    model.train(training_samples)
    print(
        f"Training completed after {len(model.log_likelihood_history) - 1} iterations"
    )
    print(f"Final log-likelihood: {model.log_likelihood_history[-1]:.4f}")

    # Plot the curve of the log-likelihood
    pf.plot_convergence_curve(
        series={"log-likelihood": model.log_likelihood_history},
        x_label="iteration",
        y_label="log-likelihood",
        method_name="GMM with EM",
        file_name="gmm-em-curve.png",
    )

    # Visualize the parameter trajectory in 2D using PCA, colored by the
    # log-likelihood
    pf.visualize_parameter_space_em(model, file_name="gmm-em-traj.png")

    # Evaluate the clustering against the true labels. ARI is invariant to any
    # permutation of the component ids, so no label matching is needed here.
    training_ari = adjusted_rand_score(training_labels, model.assignments)
    print(f"Training data ARI [-1, 1]: {training_ari:.4f}")

    test_ari = adjusted_rand_score(test_labels, model.predict(test_samples))
    print(f"Test data ARI [-1, 1]: {test_ari:.4f}")

    # Visualize clustering results (only for 2D data)
    if num_features == 2:
        pf.visualize_clustering_result(
            training_samples,
            model,
            model.assignments,
            true_labels=training_labels,
            method_name="GMM with EM",
            file_name="gmm-em-clusters.png",
        )
