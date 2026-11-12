##############################################################################
#                             LINEAR REGRESSION
##############################################################################


import argparse
import numpy as np

from Data import data_loaders as dl
import utility_functions as uf
import plot_functions as pf

# Default number of iterations for gradient descent
DEFAULT_NUM_ITERATIONS = 10000
# Default learning rate for gradient descent
DEFAULT_LEARNING_RATE = 0.1
# Default convergence threshold for gradient descent
DEFAULT_CONVERGENCE_THRESHOLD = 1e-6


##############################################################################
#                           LINEAR REGRESSION CLASS
##############################################################################


class LinearRegression:
    """
    Linear Regression for continuous value prediction.

    The model predicts values using: y = f(x) = w · x + b, where w is the weight vector,
    x is the input instance, b is the bias, and y is the value.

    The loss function is mean squared error (MSE).
        L = (1/N) Σ (y_i - ŷ_i)²
    where N is sample size, y_i is the true value for instance i,
    and ŷ_i is the predicted value for instance i.

    The learning algorithm trains the model using two methods:
    1. Gradient Descent (GD): iterative optimization
    2. Normal Equation (NE): closed-form solution

    Attributes:
        weights: feature weights, shape (num_features,)
        bias: bias term, a float
        num_iterations: number of iterations for gradient descent, a positive integer
        learning_rate: step size for weight updates, a positive float
        convergence_threshold: if specified, training stops when the absolute
            change in MSE between consecutive iterations is below this value;
            if None, training runs for all num_iterations iterations
        loss_log: list of training losses (mean squared errors);
            loss_log[k] is the loss after k+1 iterations (parameter updates);
            loss_log[-1] is the loss of the final trained model;
            len = number of iterations actually performed
    """

    def __init__(
        self,
        num_iterations=DEFAULT_NUM_ITERATIONS,
        learning_rate=DEFAULT_LEARNING_RATE,
        convergence_threshold=DEFAULT_CONVERGENCE_THRESHOLD,
    ):
        """
        Initialize a Linear Regression model.
        """
        self.weights = None
        self.bias = None
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.loss_log = None

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _init_parameters(self, num_features):
        """
        Initialize weights and bias to zero.

        Args:
            num_features: number of features, a positive integer
        """
        self.weights = np.zeros(num_features, dtype=float)
        self.bias = 0.0

    def _compute_loss(self, instances, values):
        """
        Compute the mean squared error (MSE) on a dataset.

            L = (1/N) Σ (y_i - ŷ_i)²

        Args:
            instances: instances of shape (num_instances, num_features)
            values: true values of shape (num_instances,)

        Returns:
            loss: mean squared error, a float
        """
        return uf.evaluate_mse(values, self.predict(instances))

    @staticmethod
    def _compute_gradients(instances, values, predicted_values):
        """
        Compute the gradients of the MSE loss on the full training set.

            ∂L/∂w = - 2/N Σ (y_i - ŷ_i) · x_i
            ∂L/∂b = - 2/N Σ (y_i - ŷ_i)

        Args:
            instances: training instances of shape (num_instances, num_features)
            values: true values of shape (num_instances,)
            predicted_values: predicted values of shape (num_instances,)

        Returns:
            tuple: (grad_weights, grad_bias) where
                - grad_weights: shape (num_features,)
                - grad_bias: a float
        """
        errors = values - predicted_values
        grad_weights = -2 * (errors @ instances) / len(values)
        grad_bias = -2 * np.sum(errors) / len(values)
        return grad_weights, grad_bias

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_gd(self, instances, values):
        """
        Train the Linear Regression model using gradient descent (GD).

        The MSE on the full training set is logged after each update.

        Training stops early if the absolute change in MSE between consecutive
        iterations falls below convergence_threshold (when specified).

        Args:
            instances: training instances of shape (num_instances, num_features)
            values: training values of shape (num_instances,)
        """
        self._init_parameters(instances.shape[1])

        # Initialize the loss log
        self.loss_log = []

        for _ in range(self.num_iterations):
            # One gradient descent update
            self.backward(instances, values)

            # Log the loss after this update
            self.loss_log.append(self._compute_loss(instances, values))

            # Stop if the loss change is below the threshold
            if self.convergence_threshold is not None and len(self.loss_log) >= 2:
                loss_change = abs(self.loss_log[-2] - self.loss_log[-1])
                if loss_change < self.convergence_threshold:
                    break

    def backward(self, instances, values):
        """
        Backward pass: compute gradients on the full training set and update parameters.

            w := w - η ∂L/∂w
            b := b - η ∂L/∂b

        Args:
            instances: training instances of shape (num_instances, num_features)
            values: training values of shape (num_instances,)
        """
        predicted_values = self.predict(instances)
        grad_weights, grad_bias = self._compute_gradients(
            instances, values, predicted_values
        )
        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias

    def train_ne(self, instances, values):
        """
        Train the Linear Regression model using Normal Equation.

        Solves the least squares problem by computing:
                [w; b] = (X^T X)^(-1) X^T y = X^+ y
        where X is augmented with a column of ones for the bias term and X^+ is the
        Moore-Penrose pseudoinverse. The pseudoinverse is used instead of an explicit
        inverse because X^T X is singular for linearly dependent features and
        ill-conditioned for nearly dependent ones. Note that 1/N is not included
        in the normal equation, which does not affect the solution.

        Args:
            instances: training instances of shape (num_instances, num_features)
            values: training values of shape (num_instances,)
        """
        num_instances = instances.shape[0]

        # Expand instances by adding one dimension of ones for bias term
        instances_with_bias = np.column_stack((instances, np.ones(num_instances)))

        # Compute the least-square solution using the pseudoinverse
        pseudoinverse = np.linalg.pinv(instances_with_bias)

        # Solve for optimal weights (including bias as last element)
        weights_with_bias = pseudoinverse @ values

        # Extract weights and bias
        self.weights = weights_with_bias[:-1]
        self.bias = weights_with_bias[-1]

        # Set loss log
        self.loss_log = [self._compute_loss(instances, values)]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, instances):
        """
        Make predictions on multiple instances.

        Args:
            instances: multiple input instances of shape (num_instances, num_features)

        Returns:
            values: predicted values of shape (num_instances,)
        """
        # Calculate model values: f(x) = w·x + b
        values = instances @ self.weights + self.bias
        return values


##############################################################################
#                              MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Linear Regression model on a selected dataset",
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
        "--algorithm",
        type=str,
        default="gd",
        choices=["gd", "ne"],
        help="Training algorithm: GD (gradient descent) or NE (normal equation)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=DEFAULT_NUM_ITERATIONS,
        help="Number of training iterations (GD only)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for gradient descent (GD only)",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=DEFAULT_CONVERGENCE_THRESHOLD,
        help="Threshold for convergence in terms of change in MSE between consecutive iterations (GD only)",
    )
    args = parser.parse_args()
    print("Train a Linear Regression model for regression")

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

    # Initialize linear regression model
    model = LinearRegression(
        args.num_iterations, args.learning_rate, args.convergence_threshold
    )

    # Train model using selected algorithm
    if args.algorithm == "gd":
        print(
            f"Training with GD: number of iterations = {model.num_iterations}, "
            f"learning rate = {model.learning_rate}, "
            f"convergence threshold = {model.convergence_threshold}"
        )
        model.train_gd(training_instances, training_values)
        print(f"Training completed after {len(model.loss_log)} iterations")
        # Plot training loss curve of linear regression
        pf.plot_training_curve(
            model.loss_log, "Linear Regression", "Mean Squared Error"
        )
    elif args.algorithm == "ne":
        print("Training with NE")
        model.train_ne(training_instances, training_values)
        print("Training completed")

    # Evaluate on training set
    predicted_values = model.predict(training_instances)
    training_mse = uf.evaluate_mse(training_values, predicted_values)
    print(f"Training MSE: {training_mse:.4f}")

    # Evaluate on test set
    predicted_values = model.predict(test_instances)
    test_mse = uf.evaluate_mse(test_values, predicted_values)
    print(f"Test MSE: {test_mse:.4f}")

    # Visualize the linear regression model (only for 2D data)
    if num_features == 2:
        pf.visualize_linear_regression(training_instances, training_values, model)
