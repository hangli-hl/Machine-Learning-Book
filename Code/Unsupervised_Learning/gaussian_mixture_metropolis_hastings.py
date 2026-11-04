##############################################################################
#           GAUSSIAN MIXTURE MODEL WITH METROPOLIS-HASTINGS SAMPLING
##############################################################################


import argparse
from collections import deque

import numpy as np
from scipy.special import logsumexp
from scipy.stats import invwishart, multivariate_normal
from sklearn.metrics import adjusted_rand_score

import data_creators as dc
import plot_functions as pf

# Default number of iterations for the Metropolis-Hastings sampler
DEFAULT_NUM_ITERATIONS = 20000
# Default number of initial iterations to discard (burn-in)
DEFAULT_BURN_IN = 10000
# Small positive number to avoid division by zero or log(0)
EPS = 1e-6
# Number of iterations between two progress messages during training
LOG_INTERVAL = 1000
# Optimal scaling factor of the random-walk proposals (Roberts et al., 1997)
PROPOSAL_SCALE = 2.38
# Number of recent iterations used for the windowed acceptance rate.
# ACCEPT_WINDOW >= ADAPT_INTERVAL
ACCEPT_WINDOW = 200
# Number of iterations between two adjustments of the proposal scales (burn-in only)
ADAPT_INTERVAL = 50
# Acceptance rate the adaptation aims for
TARGET_ACCEPT_RATE = 0.234
# Upper bound of the relative deviation of the covariance proposal.
# ν at least D + 3 + 2 / MAX_COV_REL_SD²
MAX_COV_REL_SD = 0.9


##############################################################################
#                  GAUSSIAN MIXTURE METROPOLIS-HASTINGS CLASS
##############################################################################


class GaussianMixtureMetropolis:
    """
    Gaussian Mixture Model learned with Metropolis-Hastings sampling.

    The model is the density
        p(x) = Σ_{j=1}^K π_j * N(x | μ_j, Σ_j)
    where π_j is the mixing coefficient of component j, N(x | μ_j, Σ_j) is its
    Gaussian density, and K is the number of components. A sample x is assigned to
    the component with the highest posterior probability, which is the component
    with the highest π_j * N(x | μ_j, Σ_j). Ties are broken in favor of the smallest
    component index.

    The mixing coefficients are parameterized by K - 1 free logits η, with the last
    logit fixed to 0:
        π = softmax(η_1, ..., η_{K-1}, 0)
    No prior is used, which is equivalent to flat improper priors, so the acceptance
    ratio contains only the likelihood and the proposal ratio. Without a prior, the
    likelihood is unbounded when a covariance collapses onto a data point, so the
    final estimate is selected by likelihood, not by posterior.

    Each iteration of the Metropolis-Hastings sampler performs two steps:
        - propose new parameters θ* from the current parameters θ:
            μ_j* = μ_j + mean_step * ε, with ε ~ N(0, I)
            η* = η + logit_step * ε, with ε ~ N(0, I)
            Σ_j* ~ Inverse-Wishart(ν, (ν - D - 1) * Σ_j), with D the number of
            features and E[Σ_j*] = Σ_j
        - accept θ* with probability min(1, a), otherwise keep θ, where
            log a = log L(θ*) - log L(θ) + log q(θ | θ*) - log q(θ* | θ)
          The proposals of the means and the logits are symmetric, so only the
          Inverse-Wishart proposal contributes to the proposal ratio.

    The proposal scales are initialized from the data, tuned every ADAPT_INTERVAL
    iterations during burn-in towards TARGET_ACCEPT_RATE, and frozen at the burn-in
    boundary, so the retained samples come from a fixed-kernel chain. The iterations
    before burn_in are discarded. Among the remaining ones, the parameters with the
    highest log-likelihood are kept as the final estimate. For simplicity, the
    posterior mean of the parameters is not used.

    All densities are evaluated in log-space and combined with logsumexp.

    Attributes:
        num_components: number of Gaussian components, a positive integer
        num_features: number of features, a positive integer
        coefficients: mixing coefficients, shape (num_components,)
        means: component means, shape (num_components, num_features)
        covariances: component covariances,
            shape (num_components, num_features, num_features)
        logits: free logits of the final parameters, shape (num_components - 1,)
        assignments: component assignments of the training samples under the final
            parameters, shape (num_training_samples,)
        best_log_likelihood: highest log-likelihood found after burn-in, a float
        acceptance_rate: fraction of accepted proposals after burn-in, a float
        num_iterations: total number of Metropolis-Hastings iterations, a positive
            integer
        burn_in: number of initial iterations to discard,
            an integer with 0 <= burn_in < num_iterations
        proposal_step: initial step size of the random walk on the means. If
            None, it is set from the data as PROPOSAL_SCALE * standard_deviation
            / sqrt(effective_num_samples * num_features)
        log_likelihood_history: log-likelihood per recorded parameter vector,
            list of floats
        acceptance_rate_history: windowed acceptance rate per recorded parameter
            vector, list of floats
        parameter_history: flattened parameters (coefficients, means, covariances)
            per recorded snapshot, list of 1D arrays. Entry t holds the
            parameters that produced log_likelihood_history[t] and
            acceptance_rate_history[t]
        mean_step, logit_step, covariance_relative_deviation: proposal scales of the
            chain, tuned during burn-in and frozen afterwards
    """

    def __init__(
        self,
        num_components,
        num_iterations=DEFAULT_NUM_ITERATIONS,
        burn_in=DEFAULT_BURN_IN,
        proposal_step=None,
    ):
        """
        Initialize a Gaussian mixture model.

        Args:
            num_components: number of Gaussian components, a positive integer
            num_iterations: total number of Metropolis-Hastings iterations, a
                positive integer
            burn_in: number of initial iterations to discard,
                0 <= burn_in < num_iterations
            proposal_step: initial step size of the random walk on the means. If
                None, it is set from the data as
                2.38 * standard_deviation / sqrt(effective_num_samples * num_features)

        Raises:
            ValueError: if burn_in is not in [0, num_iterations)
        """
        self.num_components = int(num_components)
        self.num_iterations = int(num_iterations)
        self.burn_in = int(burn_in)
        self.proposal_step = proposal_step

        if not 0 <= self.burn_in < self.num_iterations:
            raise ValueError(
                f"burn_in must satisfy 0 <= burn_in < num_iterations, got "
                f"burn_in={self.burn_in}, num_iterations={self.num_iterations}"
            )

        self.num_features = None
        self.coefficients = None
        self.means = None
        self.covariances = None
        self.logits = None
        self.assignments = None
        self.best_log_likelihood = None
        self.acceptance_rate = None
        self.log_likelihood_history = []
        self.acceptance_rate_history = []
        self.parameter_history = []

        # Proposal scales, set in train() once the data statistics are known
        self.mean_step = None
        self.logit_step = None
        self.covariance_relative_deviation = None

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
        covariances with identity matrices, the coefficients uniformly and the
        logits with zeros, and reset the assignments and the training histories.

        Args:
            samples: training samples, shape (num_samples, num_features)
        """
        num_samples, num_features = samples.shape
        initial_indices = np.random.choice(
            num_samples, self.num_components, replace=False
        )
        self.num_features = num_features
        self.coefficients = np.full(self.num_components, 1 / self.num_components)
        self.means = samples[initial_indices]
        self.covariances = np.tile(np.eye(num_features), (self.num_components, 1, 1))
        self.logits = np.zeros(self.num_components - 1)
        self.assignments = None
        self.best_log_likelihood = None
        self.acceptance_rate = None
        self.log_likelihood_history = []
        self.acceptance_rate_history = []
        self.parameter_history = []

    def _record_parameters(self, log_likelihood, acceptance_rate):
        """
        Append the current parameters and their scores to the three histories.

        All histories are appended together so that they stay the same length and
        entry t of one always describes entry t of the others.

        Args:
            log_likelihood: log-likelihood under the current parameters, a float
            acceptance_rate: windowed acceptance rate of the chain, a float
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
        self.log_likelihood_history.append(log_likelihood)
        self.acceptance_rate_history.append(acceptance_rate)

    def _init_proposal_scales(self, samples):
        """
        Initialize the proposal scales from the statistics of the data, using the
        existing theoretical results on optimal MCMC proposals.

        Args:
            samples: training samples, shape (num_samples, num_features)
        """
        num_samples, num_features = samples.shape
        standard_deviation = float(np.sqrt(max(np.mean(np.var(samples, axis=0)), EPS)))
        effective_num_samples = max(num_samples / self.num_components, 2.0)
        num_covariance_params = num_features * (num_features + 1) / 2.0

        if self.proposal_step is not None:
            self.mean_step = float(self.proposal_step)
        else:
            self.mean_step = float(
                PROPOSAL_SCALE
                * standard_deviation
                / np.sqrt(effective_num_samples * num_features)
            )
        self.logit_step = float(PROPOSAL_SCALE / np.sqrt(effective_num_samples))
        self.covariance_relative_deviation = min(
            float(
                PROPOSAL_SCALE
                * np.sqrt(2.0 / effective_num_samples)
                / np.sqrt(num_covariance_params)
            ),
            MAX_COV_REL_SD,
        )

    def _adapt_proposal_scales(self, recent_accepts):
        """
        Rescale the three proposal scales by exp(acceptance rate - TARGET_ACCEPT_RATE),
        where the acceptance rate is measured over the last ADAPT_INTERVAL iterations.

        Args:
            recent_accepts: recent acceptance decisions (1.0 or 0.0), a deque with at
                least ADAPT_INTERVAL entries
        """
        recent_acceptance_rate = float(np.mean(list(recent_accepts)[-ADAPT_INTERVAL:]))
        factor = float(np.exp(recent_acceptance_rate - TARGET_ACCEPT_RATE))

        self.mean_step *= factor
        self.logit_step *= factor
        # Keep the covariance_relative_deviation below MAX_COV_REL_SD
        self.covariance_relative_deviation = min(
            self.covariance_relative_deviation * factor, MAX_COV_REL_SD
        )

    def _print_proposal_scales(self, title):
        """
        Print the current proposal scales.

        Args:
            title: text printed in front of the scales, a string
        """
        print(
            f"{title}: "
            f"mean_step={self.mean_step:.4g}, "
            f"logit_step={self.logit_step:.4g}, "
            f"covariance_relative_deviation={self.covariance_relative_deviation:.4g}"
        )

    def _propose_parameters(self):
        """
        Propose new parameters from the current ones.

        Returns:
            proposed_coefficients: shape (num_components,)
            proposed_means: shape (num_components, num_features)
            proposed_covariances: shape (num_components, num_features, num_features)
            proposed_logits: shape (num_components - 1,)
            log_proposal_ratio: log q(current | proposed) - log q(proposed | current)
        """
        num_features = self.num_features

        # Derive the degrees of freedom ν from the relative deviation of the
        # covariance proposal: sqrt(2 / (ν - D - 3)) = covariance_relative_deviation
        nu = num_features + 3.0 + 2.0 / self.covariance_relative_deviation**2

        # 1. Means: symmetric random walk, ratio = 0
        proposed_means = self.means + self.mean_step * np.random.randn(
            self.num_components, num_features
        )

        # 2. Coefficients: random walk on the logits with the last logit set to 0
        if self.num_components > 1:
            proposed_logits = self.logits + self.logit_step * np.random.randn(
                self.num_components - 1
            )
            full_logits = np.append(proposed_logits, 0.0)
            proposed_coefficients = np.exp(full_logits - logsumexp(full_logits))
        else:
            proposed_logits = self.logits
            proposed_coefficients = np.array([1.0])

        # 3. Covariances: asymmetric Inverse-Wishart proposal
        log_proposal_ratio = 0.0
        proposed_covariances = np.zeros_like(self.covariances)
        for j in range(self.num_components):
            current_covariance = self.covariances[j]

            # Forward scale chosen so that E[Σ*] = Σ_current
            scale_forward = (nu - num_features - 1) * current_covariance
            proposed_covariance = invwishart.rvs(df=nu, scale=scale_forward)
            # atleast_2d keeps the matrix shape when there is a single feature
            proposed_covariance = np.atleast_2d(proposed_covariance)
            proposed_covariance = (proposed_covariance + proposed_covariance.T) / 2.0
            proposed_covariances[j] = proposed_covariance

            scale_reverse = (nu - num_features - 1) * proposed_covariance

            log_proposal_ratio += invwishart.logpdf(
                current_covariance, df=nu, scale=scale_reverse
            ) - invwishart.logpdf(proposed_covariance, df=nu, scale=scale_forward)

        return (
            proposed_coefficients,
            proposed_means,
            proposed_covariances,
            proposed_logits,
            log_proposal_ratio,
        )

    def _compute_log_likelihood(self, samples, coefficients, means, covariances):
        """
        Compute the total log-likelihood in log-space for the given parameters.

            log L = Σ_{i=1}^N log( Σ_{j=1}^K π_j * N(x_i | μ_j, Σ_j) )

        Args:
            samples: input samples, shape (num_samples, num_features)
            coefficients: shape (num_components,)
            means: shape (num_components, num_features)
            covariances: shape (num_components, num_features, num_features)

        Returns:
            log_likelihood: total log-likelihood, a float
        """
        log_weighted_densities = self.compute_log_weighted_densities(
            samples, coefficients, means, covariances
        )
        return float(np.sum(logsumexp(log_weighted_densities, axis=1)))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, samples):
        """
        Train the Gaussian mixture model with Metropolis-Hastings sampling.

        In each iteration, new parameters are proposed from the current ones and
        accepted or rejected. During burn-in, the proposal scales are tuned every
        ADAPT_INTERVAL iterations and frozen at the burn-in boundary. From iteration
        burn_in on, the parameters and their scores are recorded together. After the
        last iteration, the recorded parameters with the highest log-likelihood
        become the final parameters.

        Args:
            samples: training samples, shape (num_samples, num_features)
        """
        samples = self._check_samples(samples)
        self._init_parameters(samples)
        self._init_proposal_scales(samples)
        self._print_proposal_scales("Initial proposal scales")

        num_accepts = 0
        recent_accepts = deque(maxlen=ACCEPT_WINDOW)

        # The first recorded point always becomes the best one
        best_log_likelihood = -np.inf
        best_parameters = None

        current_log_likelihood = self._compute_log_likelihood(
            samples, self.coefficients, self.means, self.covariances
        )

        for iteration in range(self.num_iterations):
            # Restart the counters so the reported rates describe the frozen chain
            if iteration == self.burn_in and self.burn_in > 0:
                self._print_proposal_scales("Proposal scales frozen at burn-in")
                num_accepts = 0
                recent_accepts.clear()

            # Propose new parameters, then accept or reject them
            (
                proposed_coefficients,
                proposed_means,
                proposed_covariances,
                proposed_logits,
                log_proposal_ratio,
            ) = self._propose_parameters()

            proposed_log_likelihood = self._compute_log_likelihood(
                samples, proposed_coefficients, proposed_means, proposed_covariances
            )

            # log a = log L(θ*) - log L(θ) + log q(θ | θ*) - log q(θ* | θ)
            log_acceptance_ratio = (
                proposed_log_likelihood - current_log_likelihood + log_proposal_ratio
            )

            accepted = np.log(np.random.rand()) < log_acceptance_ratio
            if accepted:
                self.coefficients = proposed_coefficients
                self.means = proposed_means
                self.covariances = proposed_covariances
                self.logits = proposed_logits
                current_log_likelihood = proposed_log_likelihood
                num_accepts += 1
            recent_accepts.append(float(accepted))
            recent_acceptance_rate = float(np.mean(recent_accepts))

            # Tune the proposal scales during burn-in
            if iteration < self.burn_in and (iteration + 1) % ADAPT_INTERVAL == 0:
                self._adapt_proposal_scales(recent_accepts)

            if iteration >= self.burn_in:
                self._record_parameters(current_log_likelihood, recent_acceptance_rate)

                # Keep track of the best parameters based on log-likelihood
                if (
                    best_parameters is None
                    or current_log_likelihood > best_log_likelihood
                ):
                    best_log_likelihood = current_log_likelihood
                    best_parameters = (
                        self.coefficients.copy(),
                        self.means.copy(),
                        self.covariances.copy(),
                        self.logits.copy(),
                    )

            if iteration % LOG_INTERVAL == 0:
                print(
                    f"Iteration {iteration}/{self.num_iterations}, "
                    f"acceptance rate: {recent_acceptance_rate:.3f}, "
                    f"log-likelihood: {current_log_likelihood:.3f}"
                )

        # Keep the parameters with the highest log-likelihood after burn-in
        self.coefficients = best_parameters[0]
        self.means = best_parameters[1]
        self.covariances = best_parameters[2]
        self.logits = best_parameters[3]
        self.best_log_likelihood = best_log_likelihood
        self.acceptance_rate = num_accepts / (self.num_iterations - self.burn_in)
        self.assignments = self.predict(samples)

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
        samples = np.asarray(samples, dtype=float)
        # The posterior is the weighted density divided by a per-sample constant,
        # so both have the same argmax
        log_weighted_densities = self.compute_log_weighted_densities(
            samples, self.coefficients, self.means, self.covariances
        )
        return np.argmax(log_weighted_densities, axis=1)

    def compute_log_weighted_densities(self, samples, coefficients, means, covariances):
        """
        Compute the log weighted densities of samples under all components.

            log( π_j * N(x_i | μ_j, Σ_j) ) = log(π_j) + log(N(x_i | μ_j, Σ_j))

        Args:
            samples: input samples, shape (num_samples, num_features)
            coefficients: shape (num_components,)
            means: shape (num_components, num_features)
            covariances: shape (num_components, num_features, num_features)

        Returns:
            log_weighted_densities: Entry[i, j] is log( π_j * N(x_i | μ_j, Σ_j) ),
                shape (num_samples, num_components)
        """
        log_weighted_densities = np.zeros((samples.shape[0], self.num_components))

        # log(0) = -inf is valid here: the component then has zero weight
        with np.errstate(divide="ignore"):
            log_coefficients = np.log(coefficients)
        for j in range(self.num_components):
            log_density = multivariate_normal.logpdf(
                samples,
                mean=means[j],
                cov=covariances[j],
            )
            log_weighted_densities[:, j] = log_coefficients[j] + log_density

        return log_weighted_densities


##############################################################################
#                              MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Gaussian Mixture Model (GMM) with Metropolis-Hastings "
        "sampling on a selected dataset",
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
        default=DEFAULT_NUM_ITERATIONS,
        help="Total number of Metropolis-Hastings sampling iterations",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=DEFAULT_BURN_IN,
        help="Number of initial iterations to discard (burn-in period), "
        "must be smaller than the number of iterations",
    )
    parser.add_argument(
        "--proposal-step",
        type=float,
        default=None,
        help="Initial step size of the Gaussian random walk on the means "
        "(default: scaled from the data and the sample size)",
    )
    args = parser.parse_args()
    print("Train a Gaussian Mixture Model (GMM) with Metropolis-Hastings sampling")

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
    print(
        "Initial step size of the random walk on the means: "
        + (
            "automatic (scaled from the data)"
            if args.proposal_step is None
            else f"{args.proposal_step}"
        )
    )

    # Random seed for reproducibility, after data creation
    MODEL_SEED = 32
    np.random.seed(MODEL_SEED)

    # Initialize the Gaussian mixture model
    model = GaussianMixtureMetropolis(
        num_components, args.num_iterations, args.burn_in, args.proposal_step
    )

    # Train the model with Metropolis-Hastings sampling
    print(
        f"Training with Metropolis-Hastings sampling, number of iterations = "
        f"{model.num_iterations}, burn-in = {model.burn_in}"
    )
    model.train(training_samples)
    print(
        f"Training completed after {model.num_iterations} iterations, "
        f"{len(model.parameter_history)} parameter vectors collected"
    )
    print(f"Post-burn-in acceptance rate: {model.acceptance_rate:.4f}")
    print(f"Best log-likelihood: {model.best_log_likelihood:.4f}")

    # Plot the curves of the log-likelihood and the acceptance rate
    pf.plot_convergence_curve(
        series={"log-likelihood": model.log_likelihood_history},
        twin_series={"acceptance rate": model.acceptance_rate_history},
        x_label="iteration",
        y_label="log-likelihood",
        twin_y_label="acceptance rate",
        method_name="GMM with MH",
        file_name="gmm-mh-curve.png",
        start_iteration=model.burn_in,
    )

    # Visualize the parameter trajectory in 2D using PCA
    pf.visualize_parameter_space_mh(model, file_name="gmm-mh-traj.png")

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
            method_name="GMM with MH",
            file_name="gmm-mh-clusters.png",
        )
