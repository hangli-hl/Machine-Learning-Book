##############################################################################
#                                 ADABOOST
##############################################################################


import argparse

import numpy as np

import data_creators as dc
import utility_functions as uf
import plot_functions as pf

from classification_tree import ClassificationTree

# Default number of base learners (classification trees) in the ensemble
DEFAULT_NUM_LEARNERS = 100
# Default maximum depth of each base classification tree
DEFAULT_MAX_TREE_DEPTH = 3
# Default minimum number of instances required to split a node of a base tree
DEFAULT_MIN_SAMPLE_SPLIT = 5
# Default convergence threshold: stop training when the ensemble's
# training error rate is below this value
DEFAULT_CONVERGENCE_THRESHOLD = 1e-6


##############################################################################
#                              ADABOOST CLASS
##############################################################################


class AdaBoost:
    """
    AdaBoost (Adaptive Boosting) for binary classification.

    The learning algorithm combines multiple base learners (classification trees) into an
    ensemble by iteratively reweighting training instances and adding new base learners.

    At each round a bootstrap sample is drawn from the training set according to the current
    weight distribution. The weighted error of the learner is evaluated on the full training set
    with the current weight distribution.

    Attributes:
        ensemble: list of trained base learners (classification trees)
        num_learners: number of base learners requested for the ensemble, a positive integer
        coefficients: list of learner coefficients α_j, parallel to ensemble
        max_tree_depth: maximum depth of each base classification tree, a positive integer
        min_sample_split: minimum instances required to split a node, a positive integer
        convergence_threshold: if not None, stop training when the training error rate of
            the ensemble is below this value (a float in [0, 1])
        training_error_log: list of training error rates of the ensemble, one per round;
            training_error_log[j] is the error rate after the (j+1)-th learner has been added,
            len = len(self.ensemble)
    """

    def __init__(
        self,
        num_learners=DEFAULT_NUM_LEARNERS,
        max_tree_depth=DEFAULT_MAX_TREE_DEPTH,
        min_sample_split=DEFAULT_MIN_SAMPLE_SPLIT,
        convergence_threshold=DEFAULT_CONVERGENCE_THRESHOLD,
    ):
        """
        Initialize an AdaBoost model.
        """
        self.ensemble = None
        self.coefficients = None
        self.num_learners = num_learners
        self.max_tree_depth = max_tree_depth
        self.min_sample_split = min_sample_split
        self.convergence_threshold = convergence_threshold
        self.training_error_log = None

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    @staticmethod
    def _check_labels(labels):
        """
        Check that labels are in {-1, 1} and that both classes are present.

        Args:
            labels: training labels of shape (num_instances,)
        """
        if set(np.unique(labels)) != {-1, 1}:
            raise ValueError("Labels must be in {-1, 1} and contain both classes")

    @staticmethod
    def _to_tree_labels(labels):
        """
        Convert labels from the AdaBoost encoding {-1, 1} to the tree encoding {0, 1}.

        Args:
            labels: labels in {-1, 1}, shape (num_instances,)

        Returns:
            labels: labels in {0, 1}, shape (num_instances,)
        """
        return ((labels + 1) // 2).astype(int)

    @staticmethod
    def _from_tree_labels(labels):
        """
        Convert labels from the tree encoding {0, 1} back to the AdaBoost encoding {-1, 1}.

        Args:
            labels: labels in {0, 1}, shape (num_instances,)

        Returns:
            labels: labels in {-1, 1}, shape (num_instances,)
        """
        return (2 * labels - 1).astype(int)

    @staticmethod
    def _compute_error_rate(labels, predicted_labels):
        """
        Compute the error rate, i.e. the fraction of misclassified instances:
            error rate = 1 - accuracy

        Args:
            labels: true labels in {-1, 1}, shape (num_instances,)
            predicted_labels: predicted labels in {-1, 1}, shape (num_instances,)

        Returns:
            error_rate: fraction of misclassified instances, a float in [0, 1]
        """
        error_rate = 1.0 - uf.evaluate_accuracy(labels, predicted_labels)
        return error_rate

    def _train_learner(self, instances, tree_labels, weights):
        """
        Train a new base learner (classification tree) on a bootstrap sample drawn
        according to the current weight distribution.

        Args:
            instances: training instances, shape (num_instances, num_features)
            tree_labels: training labels in the tree encoding {0, 1}, shape (num_instances,)
            weights: current sample weights (a probability distribution),
                shape (num_instances,)

        Returns:
            learner: a trained classification tree
        """
        num_instances = instances.shape[0]

        # Draw bootstrap samples from the current weight distribution
        indices = np.random.choice(
            num_instances, size=num_instances, replace=True, p=weights
        )

        # Train a classification tree on the resampled data
        learner = ClassificationTree(
            max_tree_depth=self.max_tree_depth,
            min_sample_split=self.min_sample_split,
        )
        learner.train(instances[indices], tree_labels[indices])
        return learner

    def _predict_learner(self, learner, instances):
        """
        Make predictions with a base learner in the AdaBoost label encoding.

        Args:
            learner: a trained base learner (classification tree)
            instances: input instances, shape (num_instances, num_features)

        Returns:
            labels: predicted class labels in {-1, 1}, shape (num_instances,), dtype=int
        """
        # Predict with the base learner of classification tree
        predicted_tree_labels = learner.predict(instances)

        # Convert back labels from {0, 1} to {-1, 1}
        return self._from_tree_labels(predicted_tree_labels)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, instances, labels):
        """
        Train the AdaBoost model using iterative reweighting.

        The algorithm:
        1. Initialize uniform weights for all instances
        2. In each round j:
            a. Draw a bootstrap sample according to the current weight distribution and
               train a base learner (classification tree) g_j on it
            b. Calculate the weighted error ε_j of g_j on the training set
            c. Calculate the learner coefficient: α_j = 0.5 · log((1 - ε_j) / ε_j)
                (a small DIV_EPS is added in the code to avoid division by zero;
                α_j < 0 if ε_j > 0.5)
            d. Add g_j and α_j to the ensemble
            e. Update sample weights: w_i := w_i · exp(-α_j · y_i · g_j(x_i))
            f. Normalize sample weights
            g. Compute and store the training error rate of the ensemble
            h. Stop if the training error rate is below convergence_threshold

        Since the base learner (ClassificationTree) expects labels in {0, 1} while AdaBoost
        works in {-1, 1}, labels are converted at the boundary: {0, 1} is used only inside
        the base learners, and every prediction is mapped back to {-1, 1}.

        Args:
            instances: training instances, shape (num_instances, num_features)
            labels: training labels of shape (num_instances,), labels in {-1, 1}
        """
        self._check_labels(labels)
        num_instances = instances.shape[0]

        # Initialize the AdaBoost model for learning
        self.ensemble = []
        self.coefficients = []

        # Initialize the training error log: one training error rate per round
        self.training_error_log = []

        # Initialize weights of instances
        weights = np.full(num_instances, 1.0 / num_instances)

        # Convert labels from {-1, 1} to {0, 1} for the base learners
        tree_labels = self._to_tree_labels(labels)

        for _ in range(self.num_learners):
            # Train a classification tree on a bootstrap sample drawn from the weights
            learner = self._train_learner(instances, tree_labels, weights)

            # Make predictions of this learner on the original training set
            predicted_labels = self._predict_learner(learner, instances)

            # Calculate the weighted error of this learner ε_j
            error = weights[labels != predicted_labels].sum()

            # Coefficient α_j: DIV_EPS avoids division by zero / log(0)
            # α_j < 0 when ε_j > 0.5: the learner's vote is flipped
            coefficient = 0.5 * np.log(
                (1.0 - error + uf.DIV_EPS) / (error + uf.DIV_EPS)
            )

            # Update sample weights: w_i := w_i · exp(-α_j · y_i · g_j(x_i)), then normalize
            weights *= np.exp(-coefficient * labels * predicted_labels)
            weights /= weights.sum()

            # Add learner and its coefficient to the ensemble
            self.ensemble.append(learner)
            self.coefficients.append(coefficient)

            # Evaluate the ensemble and log the training error rate
            training_error = self._compute_error_rate(labels, self.predict(instances))
            self.training_error_log.append(training_error)

            # Check convergence: stop if the training error rate is below the threshold
            if (
                self.convergence_threshold is not None
                and training_error < self.convergence_threshold
            ):
                break

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, instances):
        """
        Make predictions on instances using the AdaBoost model.

        Args:
            instances: input instances, shape (num_instances, num_features)

        Returns:
            labels: predicted class labels in {-1, 1}, shape (num_instances,), dtype=int
        """
        # Compute scores of instances using the model
        scores = self.forward(instances)

        # Assign labels: label = 1 if score >= 0, label = -1 otherwise
        return np.where(scores >= 0, 1, -1)

    def forward(self, instances):
        """
        Forward pass: compute the scores of instances using the AdaBoost model.

        The score is the weighted sum of the base learner predictions:
            f(x) = Σ α_j · g_j(x)
        where α_j is the coefficient and g_j is the prediction of the j-th learner.

        Args:
            instances: input instances, shape (num_instances, num_features)

        Returns:
            scores: weighted sum of predictions from all learners, shape (num_instances,)
        """
        scores = np.zeros(instances.shape[0], dtype=float)

        # Accumulate the weighted predictions of all learners
        for learner, coefficient in zip(self.ensemble, self.coefficients):
            scores += coefficient * self._predict_learner(learner, instances)

        return scores


##############################################################################
#                              MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train an AdaBoost model with classification trees on a selected dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="jigsaw",
        choices=["artificial", "gaussians", "iris", "jigsaw", "circle", "moons"],
        help="Dataset to use: artificial (manual 2D), gaussians (synthetic 2D, two gaussians), "
        "iris (Iris 4D), jigsaw (synthetic 2D, xor patterns), circle (synthetic 2D, "
        "circular patterns) or moons (synthetic 2D, two half moons)",
    )
    parser.add_argument(
        "--num-learners",
        type=int,
        default=DEFAULT_NUM_LEARNERS,
        help="Number of base learners (classification trees) in the ensemble",
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=DEFAULT_MAX_TREE_DEPTH,
        help="Maximum depth of each base classification tree (1 = classification stump)",
    )
    parser.add_argument(
        "--min-sample-split",
        type=int,
        default=DEFAULT_MIN_SAMPLE_SPLIT,
        help="Minimum number of instances required to split a node of a base tree",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=None,
        help="Stop training when the ensemble's training error rate is below this value",
    )
    args = parser.parse_args()
    print("Train an AdaBoost model for binary classification")

    # Select dataset, labels in {-1, +1}
    training_instances, training_labels, test_instances, test_labels = (
        dc.select_binary_classification_data(dataset=args.data, signed_label=True)
    )
    num_features = training_instances.shape[1]
    num_classes = 2
    print(f"Using dataset: {args.data}")
    print(
        f"Training set: {training_instances.shape[0]} instances, "
        f"{num_features} features, {num_classes} classes"
    )
    print(
        f"Test set: {test_instances.shape[0]} instances, "
        f"{num_features} features, {num_classes} classes"
    )

    # Random seed for reproducibility, after data creation
    MODEL_SEED = 36
    np.random.seed(MODEL_SEED)

    # Initialize AdaBoost model
    model = AdaBoost(
        args.num_learners,
        args.max_tree_depth,
        args.min_sample_split,
        args.convergence_threshold,
    )

    # Train the model and log training error rate
    print(
        f"Training with {model.num_learners} classification trees, "
        f"maximum tree depth = {model.max_tree_depth}, "
        f"minimum split size = {model.min_sample_split}, "
        f"convergence threshold = {model.convergence_threshold}"
    )
    model.train(training_instances, training_labels)
    print(f"Training completed after {len(model.training_error_log)} rounds")

    # Plot training error rate curve
    pf.plot_training_curve(
        model.training_error_log, "AdaBoost", "Training Error Rate", log_x=False
    )

    # Evaluate on training set
    predicted_training_labels = model.predict(training_instances)
    training_accuracy = uf.evaluate_accuracy(
        training_labels, predicted_training_labels
    )
    print(f"Training accuracy: {training_accuracy:.4f}")

    # Evaluate on test set
    predicted_test_labels = model.predict(test_instances)
    test_accuracy = uf.evaluate_accuracy(test_labels, predicted_test_labels)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Visualize the adaboost model (only for 2D data)
    if num_features == 2:
        pf.visualize_adaboost(training_instances, training_labels, model)
