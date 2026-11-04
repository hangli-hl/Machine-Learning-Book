##############################################################################
#                     GRADIENT BOOSTING MULTICLASS CLASSIFIER
##############################################################################


import argparse

import numpy as np

import data_creators as dc
import utility_functions as uf
import plot_functions as pf

from regression_tree import RegressionTree

# Default number of base learners (regression trees) in the ensemble
DEFAULT_NUM_LEARNERS = 100
# Default shrinkage coefficient of each learner (0 < shrinkage <= 1)
DEFAULT_SHRINKAGE = 0.1
# Default maximum depth of the regression tree
DEFAULT_MAX_TREE_DEPTH = 3
# Default minimum number of instances required to split a node
DEFAULT_MIN_SAMPLE_SPLIT = 5
# Default convergence threshold for training loss change between consecutive rounds
DEFAULT_CONVERGENCE_THRESHOLD = 1e-6


##############################################################################
#                  GRADIENT BOOSTING MULTICLASS CLASSIFIER CLASS
##############################################################################


class GradientBoostingMulticlass:
    """
    Gradient Boosting Multiclass Classifier for multi-class classification.

    For K classes, the model maintains K ensembles of regression trees
    (one ensemble per class). Each ensemble produces a logit for its class:
            f_k(x) = f_{k,0} + sum_{j=1}^{M} η · h_{k,j}(x)
    where f_k(x) is the logit for class k, f_{k,0} is the initial prediction for class k
    (log of the class prior), h_{k,j} is the j-th trained learner (a regression tree)
    of class k, η is the shrinkage, and M is the number of learners per class.

    The probabilities of the classes are computed with the softmax function:
            P(y=k|x) = exp(f_k(x)) / Σ_k' exp(f_k'(x))

    For each class, the learning algorithm combines multiple base learners (regression
    trees) into an ensemble by iteratively training new learners on the negative gradient
    (pseudo-residuals) of the cross-entropy loss with respect to the logits.

    Note that the Newton leaf-value refinement is omitted;
    leaf values are set to the mean of the negative gradients in each leaf.

    Attributes:
        ensembles: list of K ensembles (one per class), each ensemble is a list of
            base learners (regression trees)
        num_learners: requested number of base learners per class, a positive integer;
            the actual number may be smaller if training stops early by convergence
        num_classes: number of classes K, inferred from the labels during training
        shrinkage: shrinkage coefficient for each learner, a float in (0, 1]
        max_tree_depth: maximum depth of each base regression tree, a positive integer
        min_sample_split: minimum instances required to split a node, a positive integer
        convergence_threshold: if specified, training stops when the absolute
            change in loss between consecutive rounds is below this value
        initial_predictions: initial predictions (log of the class priors),
            shape (num_classes,)
        loss_log: list of training cross-entropy losses of the ensemble;
            loss_log[0] is the loss of the constant initial model;
            loss_log[j] is the loss after the j-th round, i.e. after one new learner
            has been added to every class;
            len = len(self.ensembles[0]) + 1
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
        Initialize a Gradient Boosting Multiclass Classifier for multi-class classification.
        """
        self.ensembles = None
        self.num_classes = None
        self.num_learners = num_learners
        self.shrinkage = shrinkage
        self.max_tree_depth = max_tree_depth
        self.min_sample_split = min_sample_split
        self.convergence_threshold = convergence_threshold
        self.initial_predictions = None
        self.loss_log = None

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    @staticmethod
    def _check_labels(labels):
        """
        Check that labels are in {0, 1, ..., K-1}, that every class is present,
        and that there are at least two classes.

        Args:
            labels: training labels, shape (num_instances,)
        """
        classes = np.unique(labels)
        if len(classes) < 2 or not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError("Labels must be exactly {0, 1, ..., K-1} with K >= 2")

    @staticmethod
    def _compute_residuals(labels, logits):
        """
        Compute the pseudo-residuals of all classes, i.e. the negative gradient of the
        cross-entropy loss with respect to the logits:
            r_ik = y_ik - p_ik,  p_ik = softmax(f(x_i))_k

        Args:
            labels: true class labels, shape (num_instances,), values in [0, num_classes-1]
            logits: current ensemble logits, shape (num_instances, num_classes)

        Returns:
            residuals: pseudo-residuals, shape (num_instances, num_classes)
        """
        num_classes = logits.shape[1]
        return uf.one_hot_encode(labels, num_classes) - uf.softmax(logits)

    @staticmethod
    def _compute_loss(labels, logits):
        """
        Compute the cross-entropy loss of the current ensemble logits.

        Args:
            labels: true class labels, shape (num_instances,), values in [0, num_classes-1]
            logits: current ensemble logits, shape (num_instances, num_classes)

        Returns:
            loss: cross-entropy loss, a float
        """
        return uf.cross_entropy_loss(labels, uf.softmax(logits))

    def _train_learner(self, instances, residuals):
        """
        Train a new base learner (regression tree) on the pseudo-residuals of one class.

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
            values: shrunken predictions η · h_{k,j}(x), shape (num_instances,)
        """
        return self.shrinkage * learner.predict(instances)

    def _compute_logits(self, instances):
        """
        Compute the logits of all classes of all instances.
        Logits are unnormalized class scores before softmax.

        For each class k, the logit is the sum of the initial prediction and all
        learner predictions of class k, scaled by the shrinkage:
            f_k(x) = f_{k,0} + sum_{j=1}^{M} η · h_{k,j}(x)

        Args:
            instances: input instances, shape (num_instances, num_features)

        Returns:
            logits: predicted class logits, shape (num_instances, num_classes)
        """
        num_instances = instances.shape[0]

        # Initialize logits with the initial predictions
        logits = np.tile(self.initial_predictions, (num_instances, 1))

        # Accumulate the shrunken predictions of all learners of each class
        for class_index in range(self.num_classes):
            for learner in self.ensembles[class_index]:
                logits[:, class_index] += self._predict_learner(learner, instances)

        return logits

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, instances, labels):
        """
        Train the Gradient Boosting Multiclass Classifier by iteratively learning
        the pseudo-residuals.

        The learning algorithm with shrinkage:
        1. Initialize logits with the log of the class priors: f_{k,0} = log(p_k)
            where p_k is the fraction of instances of class k in the training set
        2. For each round j:
            a. Calculate pseudo-residuals of all classes from the current logits:
                r_ik = y_ik - softmax(f(x_i))_k
            b. For each class k:
                - Train a base learner h_{k,j} on the pseudo-residuals r_ik
                - Add h_{k,j} to the ensemble of class k
                - Update logits: f_k(x) = f_k(x) + η · h_{k,j}(x)
            c. Compute and store the training cross-entropy loss of the ensemble
            d. Stop if the absolute change in loss is below convergence_threshold

        Args:
            instances: training instances, shape (num_instances, num_features)
            labels: class labels, shape (num_instances,), values in [0, num_classes-1]
        """
        self._check_labels(labels)
        num_instances = instances.shape[0]

        # Initialize the Gradient Boosting Multiclass Classifier for learning
        self.num_classes = len(np.unique(labels))
        self.ensembles = [[] for _ in range(self.num_classes)]

        # Make initial prediction: log of the class priors
        # (every class is present, so all priors are positive and the logs are finite)
        class_priors = np.mean(uf.one_hot_encode(labels, self.num_classes), axis=0)
        self.initial_predictions = np.log(class_priors)
        logits = np.tile(self.initial_predictions, (num_instances, 1))

        # Initialize loss log: one for the initial model and one per round
        self.loss_log = [self._compute_loss(labels, logits)]

        for _ in range(self.num_learners):
            # Pseudo-residuals of all classes, computed from the logits at the start of the
            # round, so that all classes in the same round use the same probabilities
            residuals = self._compute_residuals(labels, logits)

            for class_index in range(self.num_classes):
                # Train a regression tree on the pseudo-residuals of this class
                learner = self._train_learner(instances, residuals[:, class_index])

                # Add learner to the ensemble of this class and update its logits
                self.ensembles[class_index].append(learner)
                logits[:, class_index] += self._predict_learner(learner, instances)

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
        Predict class labels using the Gradient Boosting Multiclass Classifier.

        Return the class with the highest logit. This is equivalent to the class with the
        highest softmax probability, but avoids the softmax and its rounding errors
        (e.g. ties caused by saturated probabilities).

        Args:
            instances: input instances, shape (num_instances, num_features)

        Returns:
            labels: predicted class labels in [0, num_classes-1], shape (num_instances,), dtype=int
        """
        # Compute logits of all classes
        logits = self._compute_logits(instances)

        # Find the class with the highest logit for each instance
        return np.argmax(logits, axis=1)

    def forward(self, instances):
        """
        Forward pass: compute the probabilities of all classes of all instances.

        Args:
            instances: input instances, shape (num_instances, num_features)

        Returns:
            probs: predicted class probabilities, shape (num_instances, num_classes),
                values in [0, 1], each row sums to 1
        """
        # Compute logits and apply softmax to get probabilities
        logits = self._compute_logits(instances)
        return uf.softmax(logits)


##############################################################################
#                              MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Gradient Boosting Multiclass Classifier on a selected dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="blobs",
        choices=["classification", "blobs", "iris"],
        help="Dataset to use: classification (Classification 20D), blobs (Blobs 2D) "
        "or iris (Iris 4D)",
    )
    parser.add_argument(
        "--num-learners",
        type=int,
        default=DEFAULT_NUM_LEARNERS,
        help="Number of base learners (regression trees) per class",
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
    print(
        "Train a Gradient Boosting Multiclass Classifier for multi-class classification"
    )

    # Select dataset, labels in {0, 1, ..., num_classes-1}
    training_instances, training_labels, test_instances, test_labels = (
        dc.select_multiclass_classification_data(dataset=args.data)
    )
    num_features = training_instances.shape[1]
    num_classes = len(np.unique(training_labels))
    print(f"Using dataset: {args.data}")
    print(
        f"Training set: {training_instances.shape[0]} instances, "
        f"{num_features} features, {num_classes} classes"
    )
    print(
        f"Test set: {test_instances.shape[0]} instances, "
        f"{num_features} features, {num_classes} classes"
    )

    # Initialize the Gradient Boosting Multiclass Classifier
    model = GradientBoostingMulticlass(
        args.num_learners,
        args.shrinkage,
        args.max_tree_depth,
        args.min_sample_split,
        args.convergence_threshold,
    )

    # Train the model and log training loss
    print(
        f"Training with {model.num_learners} regression trees per class, "
        f"shrinkage {model.shrinkage:.3f}, "
        f"maximum tree depth = {model.max_tree_depth}, "
        f"minimum split size = {model.min_sample_split}, "
        f"convergence threshold = {model.convergence_threshold}"
    )
    model.train(training_instances, training_labels)
    print(f"Training completed after {len(model.loss_log) - 1} rounds")

    # Plot training loss curve in terms of cross-entropy loss
    pf.plot_training_curve(
        model.loss_log,
        "Gradient Boosting Multiclass Classifier",
        "Training Cross-Entropy Loss",
        log_x=False,
    )

    # Evaluate on training set
    predicted_training_labels = model.predict(training_instances)
    training_accuracy = uf.calculate_accuracy(
        training_labels, predicted_training_labels
    )
    print(f"Training accuracy: {training_accuracy:.4f}")

    # Evaluate on test set
    predicted_test_labels = model.predict(test_instances)
    test_accuracy = uf.calculate_accuracy(test_labels, predicted_test_labels)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Visualize the gradient boosting multiclass classifier (only for 2D data)
    if num_features == 2:
        pf.visualize_gradient_boosting_multiclass(
            training_instances, training_labels, model
        )
