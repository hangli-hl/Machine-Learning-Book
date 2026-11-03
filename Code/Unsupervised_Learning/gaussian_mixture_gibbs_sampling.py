##############################################################################
#                GAUSSIAN MIXTURE MODEL WITH GIBBS SAMPLING
##############################################################################


import argparse

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import dirichlet, invwishart, multivariate_normal, wishart
from sklearn.metrics import adjusted_rand_score

import data_creators as dc
import plot_functions as pf

# Lower bound for the mixing coefficients inside logarithms
EPS = 1e-12

# Number of iterations between two progress messages during training
LOG_INTERVAL = 500


##############################################################################
#                         GAUSSIAN MIXTURE GIBBS CLASS
##############################################################################


class GaussianMixtureGibbs:
    """
    Gaussian Mixture Model learned with Gibbs sampling and conjugate Dirichlet and
    Normal-Inverse-Wishart priors.

    The model is the density
        p(x) = Σ_{j=1}^K π_j * N(x | μ_j, Σ_j)
    where π_j is the mixing coefficient of component j, N(x | μ_j, Σ_j) is its
    Gaussian density, and K is the number of components. A sample x is assigned to
    the component with the highest posterior probability, which is the component
    with the highest π_j * N(x | μ_j, Σ_j). Ties are broken in favor of the smallest
    component index.

    The priors are
        π ~ Dirichlet(α)
        Σ_j ~ Inverse-Wishart(ν_0, Ψ_0)
        μ_j | Σ_j ~ N(μ_0, Σ_j / κ_0)

    Each iteration of the Gibbs sampler performs two steps:
        - sample the component assignment z_i of every training sample from its
          posterior probabilities under the current parameters
        - sample π, μ_j and Σ_j from their conditional posteriors given the
          assignments

    After each iteration, the log-likelihood
        log L = Σ_i log p(x_i)
    and the log-prior of the current parameters are computed. The iterations before
    burn_in are discarded. Among the remaining ones, the parameters with the highest
    log-posterior (log-likelihood plus log-prior, up to the constant log-evidence)
    are kept as the final estimate.

    All densities are evaluated in log-space and combined with logsumexp.

    Attributes:
        num_components: number of Gaussian components, a positive integer
        num_features: number of features, a positive integer
        coefficients: mixing coefficients, shape (num_components,)
        means: component means, shape (num_components, num_features)
        covariances: component covariances,
            shape (num_components, num_features, num_features)
        assignments: component assignments of the training samples under the final
            parameters, shape (num_training_samples,). During training it holds the
            sampled assignments of the current iteration.
        best_log_posterior: highest log-posterior (up to a constant) found after
            burn-in, a float
        num_iterations: total number of Gibbs sampling iterations, a positive integer
        burn_in: number of initial iterations to discard, an integer
            with 0 <= burn_in < num_iterations
        log_posterior_history: log-posterior (up to a constant) per recorded
            parameter vector, list of floats
        parameter_history: flattened parameters (coefficients, means, covariances)
            per recorded snapshot, list of 1D arrays. Entry t holds the parameters
            that produced log_posterior_history[t]
        alpha, mu_0, kappa_0, psi_0, nu_0: prior hyperparameters, set in train()
            once the number of features is known
    """

    def __init__(self, num_components, num_iterations=5000, burn_in=1000):
        """
        Initialize a Gaussian mixture model.

        Args:
            num_components: number of Gaussian components, a positive integer
            num_iterations: total number of Gibbs sampling iterations, a positive
                integer
            burn_in: number of initial iterations to discard,
                0 <= burn_in < num_iterations

        Raises:
            ValueError: if burn_in is not in [0, num_iterations)
        """
        self.num_components = int(num_components)
        self.num_iterations = int(num_iterations)
        self.burn_in = int(burn_in)

        if not 0 <= self.burn_in < self.num_iterations:
            raise ValueError(
                f"burn_in must satisfy 0 <= burn_in < num_iterations, got "
                f"burn_in={self.burn_in}, num_iterations={self.num_iterations}"
            )

        self.num_features = None
        self.coefficients = None
        self.means = None
        self.covariances = None
        self.assignments = None
        self.best_log_posterior = None
        self.log_posterior_history = []
        self.parameter_history = []

        # Prior hyperparameters, set in train() once the data shape is known
        self.alpha = None
        self.mu_0 = None
        self.kappa_0 = None
        self.psi_0 = None
        self.nu_0 = None

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
        covariances with identity matrices and the coefficients uniformly, set the
        priors, and reset the assignments and the training histories.

        Args:
            samples: training samples, shape (num_samples, num_features)
        """
        num_samples, num_features = samples.shape
        initial_indices = np.random.choice(
            num_samples, self.num_components, replace=False
        )
        self.num_features = num_features
        self.coefficients = np.full(self.num_components, 1.0 / self.num_components)
        self.means = samples[initial_indices]
        self.covariances = np.tile(np.eye(num_features), (self.num_components, 1, 1))
        self.assignments = None
        self.best_log_posterior = None
        self.log_posterior_history = []
        self.parameter_history = []

        # Initialize the prior hyperparameters once the number of features is known
        self.alpha = np.ones(self.num_components)
        self.mu_0 = np.zeros(self.num_features)
        self.kappa_0 = 1.0
        self.psi_0 = np.eye(self.num_features)
        self.nu_0 = self.num_features + 2

    def _record_parameters(self, log_posterior):
        """
        Append the current parameters and their scores to the two histories.

        The histories are appended together so that they stay the same length and
        entry t of one always describes entry t of the others.

        Args:
            log_posterior: log-posterior (up to a constant) under the current
                parameters, a float
        """
        # concatenate() returns a copy, so later updates cannot modify the history
        self.parameter_history.append(
            np.concatenate(
                [
                    self.coefficients.flatten(),
                    self.means.flatten(),
                    self.covariances.flatten(),
                ]
            )
        )
        self.log_posterior_history.append(log_posterior)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, samples):
        """
        Train the Gaussian mixture model with Gibbs sampling.

        In each iteration, the component assignments are sampled under the current
        parameters, then the parameters are sampled given the assignments, and finally
        the log-posterior of the new parameters is computed. From iteration burn_in on,
        the parameters and their scores are recorded together. After the last iteration,
        the recorded parameters with the highest log-posterior become the final parameters.

        Args:
            samples: training samples, shape (num_samples, num_features)

        Raises:
            ValueError: if samples is invalid (see _check_samples)
            FloatingPointError: if the log-posterior is not finite
        """
        samples = self._check_samples(samples)
        self._init_parameters(samples)

        # Initialize the best parameters
        best_log_posterior = -np.inf
        best_parameters = None

        for iteration in range(self.num_iterations):
            # Sample component assignments, then parameters given the assignments
            self._sample_assignments(samples)
            self._sample_parameters(samples)

            # Score the new parameters with the log-posterior (up to a constant)
            (
                log_prior,
                log_likelihood,
                log_posterior,
            ) = self._compute_log_posterior(samples)

            if not np.isfinite(log_posterior):
                raise FloatingPointError(
                    f"Non-finite log-posterior ({log_posterior}) at iteration "
                    f"{iteration}: log-prior={log_prior}, log-likelihood={log_likelihood}"
                )

            if iteration >= self.burn_in:
                self._record_parameters(log_posterior)

                # Keep track of the best parameters based on log-posterior
                if log_posterior > best_log_posterior:
                    best_log_posterior = log_posterior
                    best_parameters = (
                        self.means.copy(),
                        self.covariances.copy(),
                        self.coefficients.copy(),
                    )

            if iteration % LOG_INTERVAL == 0:
                print(
                    f"Iteration {iteration}/{self.num_iterations}, "
                    f"log-posterior: {log_posterior:.3f}"
                )

        # Keep the parameters with the highest log-posterior after burn-in
        self.means, self.covariances, self.coefficients = best_parameters
        self.best_log_posterior = best_log_posterior
        self.assignments = self.predict(samples)

    def _sample_assignments(self, samples):
        """
        Sample one categorical assignment per sample via the inverse-CDF method.

        Args:
            samples: training samples, shape (num_samples, num_features)
        """
        log_weighted_densities = self._compute_log_weighted_densities(samples)
        probabilities = np.exp(
            log_weighted_densities
            - logsumexp(log_weighted_densities, axis=1, keepdims=True)
        )
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)

        cumulative_probabilities = np.cumsum(probabilities, axis=1)
        # Rounding errors must not leave the last value below a uniform draw,
        # otherwise argmax would silently return component 0
        cumulative_probabilities[:, -1] = 1.0
        uniform_samples = np.random.rand(samples.shape[0], 1)
        self.assignments = np.argmax(uniform_samples < cumulative_probabilities, axis=1)

    def _sample_parameters(self, samples):
        """
        Sample covariances, means and mixing coefficients given the current
        assignments.

        For each component, the posterior hyperparameters of the Normal-Inverse-
        Wishart prior are
            κ_n = κ_0 + N_j
            ν_n = ν_0 + N_j
            μ_n = (κ_0 * μ_0 + N_j * x̄_j) / κ_n
            Ψ_n = Ψ_0 + S_j + (κ_0 * N_j / κ_n) * (x̄_j - μ_0)(x̄_j - μ_0)^T
        where N_j is the number of samples assigned to the component, x̄_j their
        mean and S_j their scatter matrix. Then Σ_j ~ Inverse-Wishart(ν_n, Ψ_n) and
        μ_j ~ N(μ_n, Σ_j / κ_n). A component without samples is sampled from the
        prior. The mixing coefficients follow Dirichlet(α + N_1, ..., α + N_K).

        Args:
            samples: training samples, shape (num_samples, num_features)
        """
        component_sizes = np.zeros(self.num_components, dtype=int)

        for j in range(self.num_components):
            component_samples = samples[self.assignments == j]
            component_sizes[j] = len(component_samples)

            if component_sizes[j] > 0:
                kappa_n = self.kappa_0 + component_sizes[j]
                nu_n = self.nu_0 + component_sizes[j]
                component_mean = np.mean(component_samples, axis=0)
                mu_n = (
                    self.kappa_0 * self.mu_0 + component_sizes[j] * component_mean
                ) / kappa_n
                deviations = component_samples - component_mean
                mean_difference = component_mean - self.mu_0
                psi_n = (
                    self.psi_0
                    + deviations.T @ deviations
                    + (self.kappa_0 * component_sizes[j] / kappa_n)
                    * np.outer(mean_difference, mean_difference)
                )
            else:
                # Empty component: fall back to the prior
                kappa_n, nu_n, mu_n, psi_n = (
                    self.kappa_0,
                    self.nu_0,
                    self.mu_0,
                    self.psi_0,
                )

            # W ~ Wishart(ν, Ψ^-1)  <=>  W^-1 ~ Inverse-Wishart(ν, Ψ)
            # atleast_2d keeps the matrix shape when there is a single feature
            precision = np.atleast_2d(wishart.rvs(df=nu_n, scale=np.linalg.inv(psi_n)))
            self.covariances[j] = np.linalg.inv(precision)
            self.means[j] = multivariate_normal.rvs(
                mean=mu_n, cov=self.covariances[j] / kappa_n
            )

        self.coefficients = dirichlet.rvs(self.alpha + component_sizes).flatten()

    def _compute_log_posterior(self, samples):
        """
        Compute the log-prior, log-likelihood, and the log-posterior of the
        current parameters. The log-posterior is only known up to the constant
        log-evidence:
            log p(θ | X) = log p(θ) + log p(X | θ) - log p(X)

        The log-likelihood is defined as
            log L = Σ_{i=1}^N log( Σ_{j=1}^K π_j * N(x_i | μ_j, Σ_j) )

        Args:
            samples: input samples, shape (num_samples, num_features)

        Returns:
            log_prior: log-prior of the current parameters, a float
            log_likelihood: log-likelihood under the current parameters, a float
            log_posterior: log_likelihood + log_prior, a float
        """
        log_prior = self._compute_log_prior()
        log_likelihood = float(
            np.sum(logsumexp(self._compute_log_weighted_densities(samples), axis=1))
        )
        return log_prior, log_likelihood, log_likelihood + log_prior

    def _compute_log_prior(self):
        """
        Compute the log-prior of the current parameters under the Dirichlet and
        Normal-Inverse-Wishart priors.

            log p(π) + Σ_{j=1}^K [ log p(Σ_j) + log p(μ_j | Σ_j) ]

        Returns:
            log_prior: total log-prior density, a float
        """
        # Dirichlet term written out: dirichlet.logpdf rejects inputs slightly off
        # the simplex, and clipping avoids 0 * log(0)
        log_prior = float(
            gammaln(np.sum(self.alpha))
            - np.sum(gammaln(self.alpha))
            + np.sum((self.alpha - 1.0) * np.log(np.maximum(self.coefficients, EPS)))
        )

        for j in range(self.num_components):
            log_prior += float(
                invwishart.logpdf(self.covariances[j], df=self.nu_0, scale=self.psi_0)
            )
            # μ_j is a priori conditional on Σ_j, with covariance Σ_j / κ_0
            log_prior += float(
                multivariate_normal.logpdf(
                    self.means[j],
                    mean=self.mu_0,
                    cov=self.covariances[j] / self.kappa_0,
                )
            )

        return log_prior

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
        return np.argmax(self._compute_log_weighted_densities(samples), axis=1)

    def _compute_log_weighted_densities(self, samples):
        """
        Compute the log weighted densities of samples under all components.

            log( π_j * N(x_i | μ_j, Σ_j) ) = log(π_j) + log(N(x_i | μ_j, Σ_j))

        Args:
            samples: input samples, shape (num_samples, num_features)

        Returns:
            log_weighted_densities: Entry[i, j] is log( π_j * N(x_i | μ_j, Σ_j) ),
                shape (num_samples, num_components)
        """
        log_weighted_densities = np.zeros((samples.shape[0], self.num_components))

        with np.errstate(divide="ignore"):
            log_coefficients = np.log(self.coefficients)
        for j in range(self.num_components):
            log_density = multivariate_normal.logpdf(
                samples,
                mean=self.means[j],
                cov=self.covariances[j],
            )
            log_weighted_densities[:, j] = log_coefficients[j] + log_density

        return log_weighted_densities


##############################################################################
#                              MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Gaussian Mixture Model (GMM) with Gibbs sampling "
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
        default=5000,
        help="Total number of Gibbs sampling iterations",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=1000,
        help="Number of initial iterations to discard (burn-in period), "
        "must be smaller than the number of iterations",
    )
    args = parser.parse_args()
    print("Train a Gaussian Mixture Model (GMM) with Gibbs sampling")

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
    MODEL_SEED = 64
    np.random.seed(MODEL_SEED)

    # Initialize the Gaussian mixture model
    model = GaussianMixtureGibbs(num_components, args.num_iterations, args.burn_in)

    # Train the model with Gibbs sampling
    print(
        f"Training with Gibbs sampling, number of iterations = "
        f"{model.num_iterations}, burn-in = {model.burn_in}"
    )
    model.train(training_samples)
    print(
        f"Training completed after {model.num_iterations} iterations, "
        f"{len(model.parameter_history)} parameter vectors collected"
    )
    print(f"Best log-posterior (up to a constant): {model.best_log_posterior:.4f}")

    # Plot the curve of the log-posterior
    pf.plot_convergence_curve(
        series={
            "log-posterior": model.log_posterior_history,
        },
        x_label="iteration",
        y_label="log-posterior",
        method_name="GMM with GS",
        file_name="gmm-gs-curve.png",
        start_iteration=model.burn_in,
    )

    # Visualize the parameter trajectory in 2D using PCA
    pf.visualize_parameter_space_gs(model, file_name="gmm-gs-traj.png")

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
            method_name="GMM with GS",
            file_name="gmm-gs-clusters.png",
        )
