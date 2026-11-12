##############################################################################
#                      GRADIENT BOOSTING BINARY CLASSIFIER
##############################################################################


import argparse
import numpy as np

from Data import data_loaders as dl
from regression_tree import RegressionTree
import utility_functions as uf
import plot_functions as pf

# Default number of base learners (regression trees) in the ensemble
DEFAULT_NUM_LEARNERS = 100
# Default shrinkage coefficient of each learner (0 < shrinkage <= 1)
DEFAULT_SHRINKAGE = 0.1
# Default maximum depth of the regression tree
DEFAULT_MAX_TREE_DEPTH = 2
# Default minimum number of instances required to split a node
DEFAULT_MIN_SAMPLE_SPLIT = 5
# Default convergence threshold for training loss change between consecutive rounds
DEFAULT_CONVERGENCE_THRESHOLD = 1e-6


##############################################################################
#                   GRADIENT BOOSTING BINARY CLASSIFIER CLASS
##############################################################################


class GradientBoostingBinary:
    """
    Gradient Boosting Binary Classifier for binary classification.

    Gradient Boosting combines multiple base learners (regression trees) into an ensemble of
    learners by iteratively training new learners on the negative gradient (pseudo-residuals)
    of the logistic loss (binary cross-entropy loss).

    The ensemble produces a logit for the positive class, computed as the sum of the
    initial prediction and all learner predictions, scaled by the shrinkage:
            f(x) = f_0 + sum_{j=1}^{M} η · h_j(x)
    where f(x) is the logit, f_0 is the initial prediction (log-odds of the positive class prior),
    h_j is the j-th trained learner (a regression tree), η is the shrinkage,
    and M is the number of learners.

    The probability of the positive class is computed with the sigmoid function:
            P(y=1|x) = σ(f(x)) = 1 / (1 + exp(-f(x)))

    Note that the Newton leaf-value refinement is omitted;
    leaf values are set to the mean of the negative gradients in each leaf.

    Attributes:
        ensemble: ensemble of base learners (regression trees)
        num_learners: requested number of base learners in the ensemble, a positive integer
        shrinkage: shrinkage coefficient for each learner, a float in (0, 1]
        max_tree_depth: maximum depth of each base regression tree, a positive integer
        min_sample_split: minimum instances required to split a node, a positive integer
        convergence_threshold: if specified, training stops when the absolute
            change in loss between consecutive rounds is below this value
        initial_prediction: initial prediction (log-odds of the positive class), a float
        loss_log: list of training logistic losses of the ensemble;
            loss_log[0] is the loss of the constant initial model;
            loss_log[j] is the loss after the j-th learner has been added;
            len = len(self.ensemble) + 1
    """

    def __init__(
        self,
        num_learners=DEFAULT_NUM_LEARNERS,
        shrinkage=DEFAULT_SHRINKAGE,
        max_tree_depth=DEFAULT_MAX_TREE_DEPTH,
        min_sample_split=DEFAULT_MIN_SAMPLE_SPLIT,
        convergence_threshold=DEFAULT_CONVERGENCE_THRESHOLD,
    ):
        """
        Initialize a Gradient Boosting Binary Classifier for binary classification.
        """
        self.ensemble = None
        self.num_learners = num_learners
        self.shrinkage = shrinkage
        self.max_tree_depth = max_tree_depth
        self.min_sample_split = min_sample_split
        self.convergence_threshold = convergence_threshold
        self.initial_prediction = None
        self.loss_log = None

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    @staticmethod
    def _check_labels(labels):
        """
        Check that labels are in {0, 1} and that both classes are present.

        Args:
            labels: training labels, shape (num_instances,)
        """
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Labels must be exactly {0, 1}")

    @staticmethod
    def _compute_residuals(labels, logits):
        """
        Compute the pseudo-residuals, i.e. the negative gradient of the logistic loss
        with respect to the logits:
            r_i = y_i - σ(f(x_i))

        Args:
            labels: true class labels, shape (num_instances,), values in {0, 1}
            logits: current ensemble logits, shape (num_instances,)

        Returns:
            residuals: pseudo-residuals, shape (num_instances,)
        """
        return labels - uf.sigmoid(logits)

    @staticmethod
    def _compute_loss(labels, logits):
        """
        Compute the logistic loss of the current ensemble logits.

        Args:
            labels: true class labels, shape (num_instances,), values in {0, 1}
            logits: current ensemble logits, shape (num_instances,)

        Returns:
            loss: logistic loss, a float
        """
        return uf.logistic_loss(labels, uf.sigmoid(logits))

    def _train_learner(self, instances, residuals):
        """
        Train a new base learner (regression tree) on the pseudo-residuals.

        Args:
            instances: training instances, shape (num_instances, num_features)
            residuals: targets of the new learner, shape (num_instances,)

        Returns:
            learner: a trained regression tree
        """
        learner = RegressionTree(
            max_tree_depth=self.max_tree_depth,
            min_sample_split=self.min_sample_split,
        )
        learner.train(instances, residuals)
        return learner

    def _predict_learner(self, learner, instances):
        """
        Make predictions with a base learner, scaled by the shrinkage.

        Args:
            learner: a trained base learner (regression tree)
            instances: input instances, shape (num_instances, num_features)

        Returns:
            values: shrunken predictions η · h_j(x), shape (num_instances,)
        """
        return self.shrinkage * learner.predict(instances)

    def _compute_logits(self, instances):
        """
        Compute the logits of the positive class of all instances.

        The logit is the sum of the initial prediction and all learner predictions,
        scaled by the shrinkage:
            f(x) = f_0 + sum_{j=1}^{M} η · h_j(x)
        where f_0 is the initial prediction (log-odds of the positive class prior),
        h_j is the j-th trained learner (a regression tree), η is the shrinkage,
        and M is the number of learners in the ensemble.

        Args:
            instances: input instances, shape (num_instances, num_features)

        Returns:
            logits: predicted logits f(x), shape (num_instances,)
        """
        logits = np.full(instances.shape[0], self.initial_prediction, dtype=float)

        # Accumulate the shrunken predictions of all learners
        for learner in self.ensemble:
            logits += self._predict_learner(learner, instances)

        return logits

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, instances, labels):
        """
        Train the Gradient Boosting Binary Classifier by iteratively learning
        the pseudo-residuals.

        The learning algorithm with shrinkage:
        1. Initialize logits with the log-odds of the class prior: f_0 = log(p / (1 - p))
            where p is the fraction of positive instances in the training set
        2. For each round j:
            a. Calculate pseudo-residuals (negative gradients): r_i = y_i - σ(f_{j-1}(x_i))
            b. Train a base learner (regression tree) h_j on the pseudo-residuals
            c. Add h_j to the ensemble
            d. Update logits: f_j(x) = f_{j-1}(x) + η · h_j(x)
            e. Compute and store the training logistic loss of the ensemble
            f. Stop if the absolute change in loss is below convergence_threshold

        Args:
            instances: training instances, shape (num_instances, num_features)
            labels: binary class labels, shape (num_instances,), values in {0, 1}
        """
        self._check_labels(labels)
        num_instances = instances.shape[0]

        # Initialize the Gradient Boosting Binary Classifier for learning
        self.ensemble = []

        # Make initial prediction: log-odds of the positive class
        # (both classes are present, so 0 < positive_prior < 1 and the log-odds is finite)
        positive_prior = np.mean(labels)
        self.initial_prediction = np.log(positive_prior / (1 - positive_prior))
        logits = np.full(num_instances, self.initial_prediction, dtype=float)

        # Initialize loss log: one for the initial model and one per round
        self.loss_log = [self._compute_loss(labels, logits)]

        for _ in range(self.num_learners):
            # Train a regression tree on the pseudo-residuals (negative gradient of logistic loss)
            residuals = self._compute_residuals(labels, logits)
            learner = self._train_learner(instances, residuals)

            # Add learner to ensemble and update logits with shrinkage
            self.ensemble.append(learner)
            logits += self._predict_learner(learner, instances)

            # Evaluate the ensemble and log the training loss
            self.loss_log.append(self._compute_loss(labels, logits))

            # Check convergence: stop if loss change is below threshold
            if self.convergence_threshold is not None:
                loss_change = abs(self.loss_log[-2] - self.loss_log[-1])
                if loss_change < self.convergence_threshold:
                    break

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, instances):
        """
        Predict class labels using the Gradient Boosting Binary Classifier.

        Return 1 if the logit of the positive class is >= 0, else 0.
        This is equivalent to thresholding the probability σ(f(x)) at 0.5,
        but avoids the sigmoid and its rounding errors near the decision boundary.

        Args:
            instances: input instances, shape (num_instances, num_features)

        Returns:
            labels: predicted class labels in {0, 1}, shape (num_instances,), dtype=int
        """
        # Compute logits of the positive class
        logits = self._compute_logits(instances)

        # Apply decision rule: predict 1 if logit >= 0, else 0
        return np.where(logits >= 0, 1, 0)

    def forward(self, instances):
        """
        Forward pass: compute the probabilities of the positive class of all instances.

        Args:
            instances: input instances, shape (num_instances, num_features)

        Returns:
            probs: predicted probabilities, shape (num_instances,), values in [0, 1]
        """
        # Compute logits and apply sigmoid to get probabilities
        logits = self._compute_logits(instances)
        probs = uf.sigmoid(logits)

        return probs


##############################################################################
#                              MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Gradient Boosting Binary Classifier on a selected dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="jigsaw",
        choices=["gaussians", "iris", "jigsaw", "circle", "moons"],
        help="Dataset to use: gaussians (synthetic 2D, two gaussians), "
        "iris (Iris 4D), jigsaw (synthetic 2D, xor patterns), circle (synthetic 2D, "
        "circular patterns) or moons (synthetic 2D, two half moons)",
    )
    parser.add_argument(
        "--num-learners",
        type=int,
        default=DEFAULT_NUM_LEARNERS,
        help="Number of base learners (regression trees) in the ensemble",
    )
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=DEFAULT_SHRINKAGE,
        help="Shrinkage coefficient of each learner (0 < shrinkage <= 1)",
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=DEFAULT_MAX_TREE_DEPTH,
        help="Maximum depth of each base regression tree (1 = regression stump)",
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
        default=DEFAULT_CONVERGENCE_THRESHOLD,
        help="Stop training when the absolute change in training loss "
        "between consecutive rounds is below this value",
    )
    args = parser.parse_args()
    print("Train a Gradient Boosting Binary Classifier for binary classification")

    # Select dataset, labels in {0, 1}
    training_instances, training_labels, test_instances, test_labels = (
        dl.load_binary_classification_data(dataset=args.data, signed_label=False)
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

    # Initialize the Gradient Boosting Binary Classifier
    model = GradientBoostingBinary(
        args.num_learners,
        args.shrinkage,
        args.max_tree_depth,
        args.min_sample_split,
        args.convergence_threshold,
    )

    # Train the model and log training loss
    print(
        f"Training with {model.num_learners} regression trees, "
        f"shrinkage {model.shrinkage:.3f}, "
        f"maximum tree depth = {model.max_tree_depth}, "
        f"minimum split size = {model.min_sample_split}, "
        f"convergence threshold = {model.convergence_threshold}"
    )
    model.train(training_instances, training_labels)
    print(f"Training completed after {len(model.loss_log) - 1} rounds")

    # Plot training loss curve in terms of logistic loss
    pf.plot_training_curve(
        model.loss_log,
        "Gradient Boosting Binary Classifier",
        "Training Logistic Loss",
        log_x=False,
    )

    # Evaluate on training set
    predicted_training_labels = model.predict(training_instances)
    training_accuracy = uf.evaluate_accuracy(training_labels, predicted_training_labels)
    print(f"Training accuracy: {training_accuracy:.4f}")

    # Evaluate on test set
    predicted_test_labels = model.predict(test_instances)
    test_accuracy = uf.evaluate_accuracy(test_labels, predicted_test_labels)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Visualize the gradient boosting binary classifier (only for 2D data)
    if num_features == 2:
        pf.visualize_gradient_boosting_binary(
            training_instances, training_labels, model
        )
