##############################################################################
#                          BINOMIAL LOGISTIC REGRESSION
##############################################################################


import argparse

import numpy as np
from scipy.optimize import minimize

import data_creators as dc
import utility_functions as uf
import plot_functions as pf

##############################################################################
#                       BINOMIAL LOGISTIC REGRESSION CLASS
##############################################################################


class BinomialLogisticRegression:
    """
    Logistic Regression for binary classification.

    The model predicts probabilities using: p = σ(w · x + b), where w is
    the weight vector, x is the input instance (feature vector), b is the bias,
    p is the predicted probability of the positive class, and σ() is the sigmoid function.

    The model outputs probabilities and uses a 0.5 threshold for classification.
    The class label is predicted as 1 if p >= 0.5, else 0.
    (Equivalently, predict 1 if w · x + b >= 0; the decision boundary is w · x + b = 0.)

    The model can be trained with two methods:
        1. Gradient Descent (GD): first-order optimization
        2. BFGS: quasi-Newton method (uses gradients only, approximates the
           inverse Hessian)

    Attributes:
        weights: feature weights, shape (num_features,)
        bias: bias term, a float
        num_iterations: maximum number of iterations, a positive integer
            (GD: number of parameter updates; BFGS: scipy's maxiter)
        learning_rate: step size for parameter updates, a positive float
            (GD only, ignored by BFGS)
        convergence_threshold: if specified, GD stops when the absolute change
            in logistic loss between two consecutive iterations is below this
            value, a float (GD only, ignored by BFGS)
        loss_log: list of training logistic losses;
            loss_log[k] is the loss after k+1 iterations (parameter updates);
            loss_log[-1] is the loss of the final trained model;
            len = number of iterations actually performed
    """

    def __init__(self, num_iterations, learning_rate, convergence_threshold=None):
        """
        Initialize a Binomial Logistic Regression model.
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

    @staticmethod
    def _check_labels(labels):
        """
        Check that labels are in {0, 1} and that both classes are present.

        Args:
            labels: training labels of shape (num_instances,)
        """
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Labels must be in {0, 1} and contain both classes")

    def _init_parameters(self, num_features):
        """
        Initialize weights and bias to zero.

        Args:
            num_features: number of features, a positive integer
        """
        self.weights = np.zeros(num_features, dtype=float)
        self.bias = 0.0

    @staticmethod
    def _compute_logits(instances, weights, bias):
        """
        Compute logits z = w · x + b with the given parameters.

        Args:
            instances: multiple input instances of shape (num_instances, num_features)
            weights: weight vector of shape (num_features,)
            bias: bias term, a float

        Returns:
            logits: raw scores, shape (num_instances,)
        """
        return instances @ weights + bias

    @staticmethod
    def _compute_probs(instances, weights, bias):
        """
        Compute probabilities P(y=1|x) = σ(w · x + b) with the given parameters.

        Args:
            instances: instances of shape (num_instances, num_features)
            weights: weight vector of shape (num_features,)
            bias: bias term, a float

        Returns:
            probs: predicted probabilities of shape (num_instances,)
        """
        scores = instances @ weights + bias
        return uf.sigmoid(scores)

    @staticmethod
    def _compute_gradients(instances, labels, probs):
        """
        Compute the gradients of the mean logistic loss.

        For sigmoid with logistic loss, the gradient simplifies to:
            ∂L/∂w = - (1/N) Σ (y_i - p_i) · x_i
            ∂L/∂b = - (1/N) Σ (y_i - p_i)

        Args:
            instances: instances of shape (num_instances, num_features)
            labels: true labels of shape (num_instances,), labels in {0, 1}
            probs: predicted probabilities of shape (num_instances,)

        Returns:
            tuple: (grad_weights, grad_bias) where
                - grad_weights: shape (num_features,)
                - grad_bias: a float
        """
        errors = labels - probs
        grad_weights = -(errors @ instances) / len(labels)
        grad_bias = -np.mean(errors)
        return grad_weights, grad_bias

    def _compute_loss_and_gradients(self, parameters, instances, labels):
        """
        Compute logistic loss and its gradients for a packed parameter vector.

        This is the objective function for scipy's minimize (with jac=True),
        which expects (loss, gradient) to be returned together.

        Args:
            parameters: packed parameter vector [weights..., bias],
                shape (num_features + 1,)
            instances: training instances of shape (num_instances, num_features)
            labels: training labels of shape (num_instances,), labels in {0, 1}

        Returns:
            tuple: (loss, gradient) where
                - loss: average logistic loss, a float
                - gradient: shape (num_features + 1,), same packing as parameters
        """
        # Unpack parameters: last element is the bias
        weights = parameters[:-1]
        bias = parameters[-1]

        # Forward pass, then loss and gradient
        probs = self._compute_probs(instances, weights, bias)
        loss = uf.logistic_loss(labels, probs)
        grad_weights, grad_bias = self._compute_gradients(instances, labels, probs)

        # Pack gradient in the same layout as the parameters
        gradient = np.append(grad_weights, grad_bias)

        return loss, gradient
    
    # ------------------------------------------------------------------
    # Gradient descent
    # ------------------------------------------------------------------

    def train_gd(self, instances, labels):
        """
        Train the model by minimizing logistic loss with gradient descent (GD).

        The logistic loss (binary cross-entropy loss) is:
            L = - (1/N) Σ_i [y_i · log(p_i) + (1-y_i) · log(1-p_i)]
        where N is sample size, y_i is the true label for instance i,
        and p_i is the predicted probability for instance i.

        Each iteration performs:
            1. a backward pass: compute gradients and update parameters
            2. a forward pass: compute predictions with the updated parameters,
               and log the new loss

        Training stops after num_iterations iterations, or earlier if the
        absolute change in loss caused by the latest update is below
        convergence_threshold.

        Args:
            instances: training instances of shape (num_instances, num_features)
            labels: training labels of shape (num_instances,), labels in {0, 1}
        """
        self._check_labels(labels)
        self._init_parameters(instances.shape[1])

        # Forward pass with the initial model
        probs = self.forward(instances)
        self.loss_log = []

        for _ in range(self.num_iterations):
            # Backward pass: compute gradients and update parameters
            self.backward(instances, labels, probs)

            # Forward pass with updated parameters; log the new loss
            probs = self.forward(instances)
            self.loss_log.append(uf.logistic_loss(labels, probs))

            # Stop if this update barely changed the loss
            if self.convergence_threshold is not None and len(self.loss_log) >= 2:
                loss_change = abs(self.loss_log[-2] - self.loss_log[-1])
                if loss_change < self.convergence_threshold:
                    break

    def backward(self, instances, labels, probs):
        """
        Backward pass: compute gradients and update parameters (one GD step).

            w := w - η ∂L/∂w
            b := b - η ∂L/∂b

        Args:
            instances: training instances of shape (num_instances, num_features)
            labels: training labels of shape (num_instances,), labels in {0, 1}
            probs: predicted probabilities from the forward pass,
                computed with the current parameters, shape (num_instances,)
        """
        grad_weights, grad_bias = self._compute_gradients(instances, labels, probs)
        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias

    # ------------------------------------------------------------------
    # BFGS
    # ------------------------------------------------------------------

    def train_bfgs(self, instances, labels):
        """
        Train the model by minimizing logistic loss with BFGS (scipy).

        BFGS is a quasi-Newton method: it uses only gradients, and builds an
        approximation of the inverse Hessian from successive gradient changes
        to choose the search direction. It does not use learning_rate or
        convergence_threshold; num_iterations is passed as scipy's maxiter.

        The weights and bias are packed into one 1D vector [w_1, ..., w_d, b],
        because scipy optimizes a single flat vector. The objective function
        returns both the loss and its analytical gradient (jac=True).

        If BFGS stops before its own convergence test is satisfied (e.g. it
        reaches maxiter, or the data is linearly separable so the optimal
        weights go to infinity), a warning message is printed.

        Args:
            instances: training instances of shape (num_instances, num_features)
            labels: training labels of shape (num_instances,), labels in {0, 1}
        """
        self._check_labels(labels)
        self._init_parameters(instances.shape[1])

        # Pack parameters into a 1D vector: [weights..., bias]
        initial_params = np.append(self.weights, self.bias)

        # Initialize the loss log
        self.loss_log = []

        def log_loss(params):
            """Called by scipy once per iteration; record the current loss."""
            loss, _ = self._compute_loss_and_gradients(params, instances, labels)
            self.loss_log.append(loss)

        # Run BFGS optimization
        result = minimize(
            fun=self._compute_loss_and_gradients,
            x0=initial_params,
            args=(instances, labels),
            method="BFGS",
            jac=True,
            callback=log_loss,
            options={"maxiter": self.num_iterations},
        )

        # Unpack optimized parameters: last element is the bias
        self.weights = result.x[:-1]
        self.bias = float(result.x[-1])

        # Report if BFGS did not meet its own convergence criteria
        if not result.success:
            print(f"Warning: BFGS did not converge - {result.message}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, instances):
        """
        Predict class labels.

        Args:
            instances: multiple input instances of shape (num_instances, num_features)

        Returns:
            labels: predicted class labels in {0, 1}, shape (num_instances,), dtype=int
        """
        logits = self._compute_logits(instances, self.weights, self.bias)

        # Apply decision rule: predict 1 if logit >= 0, else 0
        return np.where(logits >= 0, 1, 0)

    def forward(self, instances):
        """
        Forward pass: compute predicted probabilities with current parameters.

        Args:
            instances: multiple input instances of shape (num_instances, num_features)

        Returns:
            probs: predicted probabilities of positive class, shape (num_instances,)
        """
        return self._compute_probs(instances, self.weights, self.bias)


##############################################################################
#                              MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Binomial Logistic Regression model on a selected dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="gaussians",
        choices=["artificial", "gaussians", "iris", "jigsaw", "circle", "moons"],
        help="Dataset to use: artificial (manual 2D), gaussians (synthetic 2D, two gaussians), "
        "iris (Iris 4D), jigsaw (synthetic 2D, xor patterns), circle (synthetic 2D, "
        "circular patterns) or moons (synthetic 2D, two half moons)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="gd",
        choices=["gd", "bfgs"],
        help="Optimization algorithm: GD (gradient descent) or BFGS (quasi-Newton)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=None,
        help="Number of training iterations (default: 10000 for GD, 100 for BFGS)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate only for gradient descent",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=None,
        help="Threshold for convergence in terms of change in loss between consecutive iterations (only applies to GD)",
    )
    args = parser.parse_args()
    print("Train a Binomial Logistic Regression model for binary classification")

    # Select dataset, labels in {0, 1}
    training_instances, training_labels, test_instances, test_labels = (
        dc.select_binary_classification_data(dataset=args.data, signed_label=False)
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

    # Set num_iterations depending on algorithm
    num_iterations = (
        args.num_iterations
        if args.num_iterations is not None
        else {"gd": 10000, "bfgs": 100}[args.algorithm]
    )

    # Initialize the binomial logistic regression model
    model = BinomialLogisticRegression(
        num_iterations, args.learning_rate, args.convergence_threshold
    )

    # Train model with selected optimization algorithm
    if args.algorithm == "gd":
        print(
            f"Training with GD: number of iterations = {model.num_iterations}, "
            f"learning rate = {model.learning_rate}, "
            f"convergence threshold = {model.convergence_threshold}"
        )
        model.train_gd(training_instances, training_labels)
    elif args.algorithm == "bfgs":
        print(f"Training with BFGS, number of iterations = {model.num_iterations}")
        model.train_bfgs(training_instances, training_labels)
    print(f"Training completed after {len(model.loss_log)} iterations")

    # Plot training loss curve of binomial logistic regression
    pf.plot_training_curve(
        model.loss_log, "Binomial Logistic Regression", "Logistic Loss"
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

    # Visualize the binomial logistic regression model (only for 2D data)
    if num_features == 2:
        pf.visualize_binomial_logistic_regression(
            training_instances, training_labels, model
        )
