##############################################################################
#                             K-MEANS CLUSTERING
##############################################################################


import argparse

import numpy as np
from sklearn.metrics import adjusted_rand_score

import data_creators as dc
import plot_functions as pf

##############################################################################
#                           K-MEANS CLUSTERING CLASS
##############################################################################


class KMeansClustering:
    """
    K-Means clustering.

    The model assigns an input sample x to its nearest centroid:
        c(x) = argmin_j ||x - μ_j||
    where μ_j is the centroid of cluster j. Ties are broken in favor of the smallest
    cluster index.

    The learning algorithm alternates two steps:
        - assignment step: assign every sample to its nearest cluster
        - update step: move every centroid to the mean of its samples

    Both steps minimize the SSE (sum of squared errors, also called inertia)
        L = Σ_i ||x_i - μ_{c(i)}||^2
    over one block of variables while the other is fixed, so L never increases and a
    fixed point is reached after finitely many iterations. That point is only a local
    optimum and depends on the initialization.

    Attributes:
        num_clusters: number of clusters, a positive integer
        num_iterations: maximum number of iterations, a positive integer
        convergence_threshold: training stops when the Frobenius norm of the
            centroid update drops to this value or below, a positive float
        num_features: number of features, a positive integer
        centroids: shape (num_clusters, num_features)
        assignments: cluster assignment of the training samples under the final
            centroids, shape (num_training_samples,)
        sse_history: SSE per recorded snapshot, list of floats
        parameter_history: flattened centroids per recorded snapshot, list of arrays of
            shape (num_clusters * num_features,). Entry t holds the centroids that
            produced sse_history[t], so position and color cannot drift apart in the
            trajectory plot.
    """

    def __init__(self, num_clusters, num_iterations=100, convergence_threshold=1e-6):
        """
        Initialize a K-Means clustering model.

        Args:
            num_clusters: number of clusters, a positive integer
            num_iterations: maximum number of iterations, a positive integer
            convergence_threshold: a positive float
        """
        self.num_clusters = int(num_clusters)
        self.num_iterations = int(num_iterations)
        self.convergence_threshold = float(convergence_threshold)
        self.num_features = None
        self.centroids = None
        self.assignments = None
        self.sse_history = []
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
            ValueError: if samples is not 2D or if num_clusters is not in
                [1, num_samples]
        """
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2:
            raise ValueError(f"samples must be 2D, got shape {samples.shape}")

        num_samples = samples.shape[0]
        if not 1 <= self.num_clusters <= num_samples:
            raise ValueError(
                f"num_clusters must be in [1, {num_samples}], got {self.num_clusters}"
            )
        return samples

    def _init_parameters(self, samples):
        """
        Initialize the centroids with num_clusters distinct random samples
        and reset the training histories.

        Args:
            samples: training samples, shape (num_samples, num_features)
        """
        num_samples, num_features = samples.shape
        initial_indices = np.random.choice(
            num_samples, self.num_clusters, replace=False
        )
        self.num_features = num_features
        self.centroids = samples[initial_indices]
        self.assignments = None
        self.sse_history = []
        self.parameter_history = []

    def _compute_sse(self, samples, assignments):
        """
        Compute the K-Means objective (SSE, sum of squared errors) under the
        current centroids.

        Args:
            samples: input samples, shape (num_samples, num_features)
            assignments: cluster assignments, shape (num_samples,)

        Returns:
            sse: sum of squared errors, a float
        """
        return float(np.sum((samples - self.centroids[assignments]) ** 2))

    def _record_parameters(self, samples, assignments):
        """
        Append the current centroids and their SSE to the two histories.

        Both histories are appended together so that they stay the same length and
        entry t of one always describes entry t of the other.

        Args:
            samples: input samples, shape (num_samples, num_features)
            assignments: cluster assignments under the current centroids,
                shape (num_samples,)
        """
        # flatten() returns a copy, so later centroid updates cannot modify the history
        self.parameter_history.append(self.centroids.flatten())
        self.sse_history.append(self._compute_sse(samples, assignments))

    def _rank_samples_by_errors(self, samples, assignments):
        """
        Rank the samples by their squared distance to their own centroid,
        from the furthest to the closest.

        Args:
            samples: input samples, shape (num_samples, num_features)
            assignments: cluster assignments, shape (num_samples,)

        Returns:
            order: sample indices sorted by descending squared error,
                shape (num_samples,)
        """
        squared_errors = np.sum((samples - self.centroids[assignments]) ** 2, axis=1)
        return np.argsort(
            -squared_errors, kind="stable"
        )  # stable to preserve original order

    @staticmethod
    def _compute_centroid_shift(old_centroids, new_centroids):
        """
        Compute the Frobenius norm of the centroid update.

        Args:
            old_centroids: centroids before the update,
                shape (num_clusters, num_features)
            new_centroids: centroids after the update,
                shape (num_clusters, num_features)

        Returns:
            shift: Frobenius norm of the difference, a float
        """
        return float(np.linalg.norm(new_centroids - old_centroids))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, samples):
        """
        Train the K-Means model, i.e. conduct clustering.

        In each iteration, the samples are first assigned to their nearest centroids
        (assignment step), then every centroid is moved to the mean of its samples
        (update step). The centroids and the SSE they produce are recorded before
        each update.

        Training stops when it reaches the specified number of iterations or when
        the Frobenius norm of the centroid update drops to the convergence threshold
        or below. The final centroids and their SSE are recorded at the end, so
        len(sse_history) = number of iterations performed + 1.

        Args:
            samples: training samples, shape (num_samples, num_features)
        """
        samples = self._check_samples(samples)
        self._init_parameters(samples)

        for _ in range(self.num_iterations):
            # Assignment step, then record the centroids together with their SSE
            assignments = self.predict(samples)
            self._record_parameters(samples, assignments)

            # Update step
            new_centroids = self._update_centroids(samples, assignments)
            centroid_shift = self._compute_centroid_shift(self.centroids, new_centroids)
            self.centroids = new_centroids

            # Stop if the centroids no longer move
            if centroid_shift <= self.convergence_threshold:
                break

        # Record the final centroids and the SSE they produce
        self.assignments = self.predict(samples)
        self._record_parameters(samples, self.assignments)

    def _update_centroids(self, samples, assignments):
        """
        Update step: recompute every centroid as the mean of its assigned samples.

        An empty cluster has no mean and is re-seeded with the sample that is
        furthest away from its own centroid.

        Args:
            samples: input samples, shape (num_samples, num_features)
            assignments: cluster assignments, shape (num_samples,)

        Returns:
            new_centroids: shape (num_clusters, num_features)
        """
        new_centroids = np.zeros((self.num_clusters, self.num_features))

        reseed_order = self._rank_samples_by_errors(samples, assignments)
        num_reseeded = 0

        for j in range(self.num_clusters):
            cluster_samples = samples[assignments == j]

            if len(cluster_samples) > 0:
                new_centroids[j] = np.mean(cluster_samples, axis=0)
            else:
                new_centroids[j] = samples[reseed_order[num_reseeded]]
                num_reseeded += 1

        return new_centroids

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, samples):
        """
        Predict cluster assignments by nearest-centroid assignment.

        Ties are broken in favor of the smallest cluster index.

        Args:
            samples: input samples, shape (num_samples, num_features)

        Returns:
            assignments: predicted cluster assignments in {0, ..., num_clusters-1},
                shape (num_samples,)
        """
        # Assign each sample to the centroid with the smallest distance
        return np.argmin(self.forward(samples), axis=1)

    def forward(self, samples):
        """
        Forward pass: compute the distances of samples to all centroids.

        Args:
            samples: input samples, shape (num_samples, num_features)

        Returns:
            distances: distances[i, j] = ||x_i - mu_j||,
                shape (num_samples, num_clusters)
        """
        samples = np.asarray(samples, dtype=float)

        # Broadcast samples against centroids to get all pairwise differences:
        # (num_samples, 1, num_features) - (1, num_clusters, num_features)
        #   -> (num_samples, num_clusters, num_features)
        differences = samples[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]

        return np.linalg.norm(differences, axis=2)


##############################################################################
#                              MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a K-Means clustering model on a selected dataset",
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
        "--num-clusters",
        type=int,
        default=None,
        help="Number of clusters (if not specified, uses the true number from the "
        "dataset)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Maximum number of K-Means iterations",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=1e-6,
        help="Stop once the Frobenius norm of the centroid update drops "
        "to this value or below",
    )
    args = parser.parse_args()
    print("Train a K-Means clustering model")

    # Create dataset
    (
        training_samples,
        training_labels,
        test_samples,
        test_labels,
        num_clusters_in_data,
    ) = dc.select_clustering_data(dataset=args.data)
    num_features = training_samples.shape[1]

    # Use specified num_clusters or fall back to true number
    num_clusters = (
        args.num_clusters if args.num_clusters is not None else num_clusters_in_data
    )

    print(f"Using dataset: {args.data}")
    print(f"Training set: {training_samples.shape[0]} samples, {num_features} features")
    print(f"Test set: {test_samples.shape[0]} samples, {num_features} features")
    print(f"Number of clusters in data: {num_clusters_in_data}")
    print(f"Number of clusters for K-Means: {num_clusters}")

    # Random seed for reproducibility, after data creation
    MODEL_SEED = 32
    np.random.seed(MODEL_SEED)

    # Initialize the K-Means model
    model = KMeansClustering(
        num_clusters, args.num_iterations, args.convergence_threshold
    )

    # Train the model and log SSE
    print(
        f"Training with K-Means, maximum number of iterations = {model.num_iterations}, "
        f"convergence threshold = {model.convergence_threshold:.3e}"
    )
    model.train(training_samples)
    print(f"Training completed after {len(model.sse_history) - 1} iterations")
    print(f"Final SSE: {model.sse_history[-1]:.3f}")

    # Plot the curve of sum of squared errors
    pf.plot_convergence_curve(
        series={"sum of squared errors": model.sse_history},
        x_label="iteration",
        y_label="sum of squared errors",
        method_name="K-Means",
        file_name="k-means-curve.png",
    )

    # Visualize the centroid trajectory in parameter space, colored by the SSE
    pf.visualize_parameter_space_kmeans(model, file_name="k-means-traj.png")

    # Evaluate the clustering against the true labels. ARI is invariant to any
    # permutation of the cluster ids, so no label matching is needed here.
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
            method_name="K-Means",
            file_name="k-means-clusters.png",
        )
