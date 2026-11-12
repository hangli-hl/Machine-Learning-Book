##############################################################################
#                        GRADIENT BOOSTING REGRESSOR
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
DEFAULT_MAX_TREE_DEPTH = 3
# Default minimum number of instances required to split a node
DEFAULT_MIN_SAMPLE_SPLIT = 5
# Default convergence threshold for training loss change between consecutive rounds
DEFAULT_CONVERGENCE_THRESHOLD = 1e-6


##############################################################################
#                    GRADIENT BOOSTING REGRESSOR CLASS
##############################################################################


class GradientBoostingRegressor:
    """
    Gradient Boosting Regressor for regression.

    Gradient Boosting combines multiple base learners (regression trees) into an ensemble of
    learners by iteratively training new learners on the negative gradient (residuals) of the
    loss function.

    The prediction is computed as the sum of the initial prediction and all
    learner predictions, scaled by the shrinkage:
            f(x) = f_0 + sum_{j=1}^{M} η · h_j(x)
    where f_0 is the initial prediction, h_j is the j-th trained learner
    (a regression tree), η is the shrinkage, and M is the number of learners.

    Attributes:
        ensemble: ensemble of base learners (regression trees)
        num_learners: requested number of base learners in the ensemble, a positive integer
        shrinkage: shrinkage coefficient for each learner, a float in (0, 1]
        max_tree_depth: maximum depth of each base regression tree, a positive integer
        min_sample_split: minimum instances required to split a node, a positive integer
        convergence_threshold: if specified, training stops when the absolute
            change in loss between consecutive rounds is below this value
        initial_prediction: initial prediction (mean of training values), a float
        loss_log: list of training mean squared errors of the ensemble;
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
        Initialize a Gradient Boosting Regressor for regression.
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
    def _compute_residuals(values, predicted_values):
        """
        Compute the residuals, i.e. the negative gradient of the squared-error loss
        (1/2)(y - f)^2 with respect to the prediction:
            r_i = y_i - f(x_i)

        Args:
            values: true values, shape (num_instances,)
            predicted_values: current ensemble predictions, shape (num_instances,)

        Returns:
            residuals: residuals, shape (num_instances,)
        """
        return values - predicted_values

    @staticmethod
    def _compute_loss(values, predicted_values):
        """
        Compute the training loss, i.e. the mean squared error (MSE):
            L = (1/N) * sum_i (y_i - f(x_i))^2

        The MSE is used to monitor training and to check convergence.

        Args:
            values: true values, shape (num_instances,)
            predicted_values: current ensemble predictions, shape (num_instances,)

        Returns:
            loss: mean squared error, a float
        """
        return uf.evaluate_mse(values, predicted_values)

    def _train_learner(self, instances, residuals):
        """
        Train a new base learner (regression tree) on the residuals.

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

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, instances, values):
        """
        Train the Gradient Boosting Regressor by iteratively learning the residuals.

        The learning algorithm with shrinkage:
        1. Initialize prediction with mean of training values: f_0 = mean(y)
        2. For each round j:
            a. Calculate residuals (negative gradients): r_i = y_i - f_{j-1}(x_i)
            b. Train a base learner (regression tree) h_j on the residuals
            c. Add h_j to the ensemble
            d. Update predictions: f_j(x) = f_{j-1}(x) + η · h_j(x)
            e. Compute and store the training MSE loss of the ensemble
            f. Stop if the absolute change in loss is below convergence_threshold

        Args:
            instances: training instances, shape (num_instances, num_features)
            values: training values, shape (num_instances,)
        """
        num_instances = instances.shape[0]

        # Initialize the Gradient Boosting Regressor for learning
        self.ensemble = []

        # Make initial prediction
        self.initial_prediction = np.mean(values)
        predicted_values = np.full(num_instances, self.initial_prediction, dtype=float)

        # Initialize loss log: one for the initial model and one per round
        self.loss_log = [self._compute_loss(values, predicted_values)]

        for _ in range(self.num_learners):
            # Train a regression tree on the residuals (negative gradient of the squared-error loss)
            residuals = self._compute_residuals(values, predicted_values)
            learner = self._train_learner(instances, residuals)

            # Add learner to ensemble and update predictions with shrinkage
            self.ensemble.append(learner)
            predicted_values += self._predict_learner(learner, instances)

            # Evaluate the ensemble and log the training loss
            self.loss_log.append(self._compute_loss(values, predicted_values))

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
        Make predictions using the Gradient Boosting Regressor.

        Args:
            instances: input instances, shape (num_instances, num_features)

        Returns:
            values: predicted values of instances, shape (num_instances,)
        """
        values = np.full(instances.shape[0], self.initial_prediction, dtype=float)

        # Accumulate the shrunken predictions of all learners
        for learner in self.ensemble:
            values += self._predict_learner(learner, instances)

        return values


##############################################################################
#                              MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Gradient Boosting Regressor on a selected dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="plane",
        choices=["plane", "diabetes"],
        help="Dataset to use: plane (synthetic 2D) " "or diabetes (Diabetes 10D)",
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
    print("Train a Gradient Boosting Regressor for regression")

    # Select dataset
    training_instances, training_values, test_instances, test_values = (
        dl.load_regression_data(dataset=args.data)
    )
    num_features = training_instances.shape[1]
    print(f"Using dataset: {args.data}")
    print(
        f"Training set: {training_instances.shape[0]} instances, "
        f"{num_features} features"
    )
    print(f"Test set: {test_instances.shape[0]} instances, {num_features} features")

    # Initialize the Gradient Boosting Regressor
    model = GradientBoostingRegressor(
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
    model.train(training_instances, training_values)
    print(f"Training completed after {len(model.loss_log) - 1} rounds")

    # Plot training loss curve in terms of mean squared error
    pf.plot_training_curve(
        model.loss_log,
        "Gradient Boosting Regressor",
        "Training Mean Squared Error",
        log_x=False,
    )

    # Evaluate on training set
    predicted_training_values = model.predict(training_instances)
    training_mse = uf.evaluate_mse(training_values, predicted_training_values)
    print(f"Training MSE: {training_mse:.4f}")

    # Evaluate on test set
    predicted_test_values = model.predict(test_instances)
    test_mse = uf.evaluate_mse(test_values, predicted_test_values)
    print(f"Test MSE: {test_mse:.4f}")

    # Visualize the gradient boosting regressor (only for 2D data)
    if num_features == 2:
        pf.visualize_gradient_boosting_regressor(
            training_instances, training_values, model
        )
