##############################################################################
#                      MULTINOMIAL LOGISTIC REGRESSION
##############################################################################


import argparse

import numpy as np
from scipy.optimize import minimize

import data_creators as dc
import utility_functions as uf
import plot_functions as pf

# Default number of iterations for gradient descent
DEFAULT_NUM_ITERATIONS_GD = 10000
# Default number of iterations for BFGS
DEFAULT_NUM_ITERATIONS_BFGS = 100
# Default learning rate for gradient descent
DEFAULT_LEARNING_RATE = 0.01
# Default convergence threshold for gradient descent
DEFAULT_CONVERGENCE_THRESHOLD = 1e-6


##############################################################################
#                  MULTINOMIAL LOGISTIC REGRESSION CLASS
##############################################################################


class MultinomialLogisticRegression:
    """
    Multinomial Logistic Regression for multi-class classification.

    Model:
        For each class k, compute score: z_k = w_k · x + b_k
        Convert scores to probabilities: P(y=k|x) = softmax(z_k)
        Predict class with highest probability

    The learning algorithm trains the model using two methods:
        1. Gradient Descent (GD): first-order optimization
        2. BFGS: quasi-Newton method (uses gradients only, approximates the
           inverse Hessian)

    Attributes:
        num_features: number of input features
        num_classes: number of output classes
        weights: weight matrix, shape (num_features, num_classes)
        biases: bias vector, shape (num_classes,)
        num_iterations: maximum number of iterations, a positive integer
            (GD: number of parameter updates; BFGS: scipy's maxiter)
        learning_rate: step size for parameter updates, a positive float
            (GD only, ignored by BFGS)
        convergence_threshold: if specified, GD stops when the absolute change
            in cross-entropy loss between two consecutive iterations is below
            this value, a float (GD only, ignored by BFGS)
        loss_log: list of training cross-entropy losses;
            loss_log[k] is the loss after k+1 iterations (parameter updates),
            loss_log[-1] is the loss of the final trained model;
            len = number of iterations actually performed
    """

    def __init__(
        self,
        num_iterations=DEFAULT_NUM_ITERATIONS_GD,
        learning_rate=DEFAULT_LEARNING_RATE,
        convergence_threshold=DEFAULT_CONVERGENCE_THRESHOLD,
    ):
        """
        Initialize a Multinomial Logistic Regression model.
        """
        self.num_features = None
        self.num_classes = None
        self.weights = None
        self.biases = None
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
        Check that labels are in {0, 1, ..., num_classes-1} and that all
        classes are present.

        Args:
            labels: training labels of shape (num_instances,)
        """
        classes = np.unique(labels)
        if len(classes) < 2 or not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError("Labels must be exactly {0, 1, ..., K-1} with K >= 2")

    def _init_parameters(self, num_features, num_classes):
        """
        Initialize weights and biases to zero.

        Args:
            num_features: number of features, a positive integer
            num_classes: number of classes, an integer >= 2
        """
        self.num_features = num_features
        self.num_classes = num_classes
        self.weights = np.zeros((num_features, num_classes), dtype=float)
        self.biases = np.zeros(num_classes, dtype=float)

    @staticmethod
    def _compute_logits(instances, weights, biases):
        """
        Compute logits z_k = w_k · x + b_k with the given parameters.

        Args:
            instances: input instances, shape (num_instances, num_features)
            weights: weight matrix, shape (num_features, num_classes)
            biases: bias vector, shape (num_classes,)

        Returns:
            logits: raw scores, shape (num_instances, num_classes)
        """
        return instances @ weights + biases

    @staticmethod
    def _compute_probs(instances, weights, biases):
        """
        Compute probabilities P(y=k|x) = softmax(W^T x + b) with the given parameters.

        Args:
            instances: instances of shape (num_instances, num_features)
            weights: weight matrix of shape (num_features, num_classes)
            biases: bias vector of shape (num_classes,)

        Returns:
            probs: predicted probabilities of shape (num_instances, num_classes)
        """
        logits = instances @ weights + biases
        return uf.softmax(logits)

    @staticmethod
    def _compute_gradients(instances, labels_onehot, probs):
        """
        Compute the gradients of the mean cross-entropy loss.

        For softmax with cross-entropy loss, the gradient simplifies to:
            ∂L/∂W = - (1/N) X^T (y - p)
            ∂L/∂b = - (1/N) Σ_i (y_i - p_i)

        Args:
            instances: instances of shape (num_instances, num_features)
            labels_onehot: one-hot labels of shape (num_instances, num_classes)
            probs: predicted probabilities of shape (num_instances, num_classes)

        Returns:
            tuple: (grad_weights, grad_biases) where
                - grad_weights: shape (num_features, num_classes)
                - grad_biases: shape (num_classes,)
        """
        errors = labels_onehot - probs
        grad_weights = -(instances.T @ errors) / len(instances)
        grad_biases = -np.mean(errors, axis=0)
        return grad_weights, grad_biases

    def _compute_loss_and_gradients(self, parameters, instances, labels, labels_onehot):
        """
        Compute cross-entropy loss and its gradient for a packed parameter vector.

        This is the objective function for scipy's minimize (with jac=True),
        which expects (loss, gradient) to be returned together.

        Args:
            parameters: packed parameter vector [weights..., biases...],
                shape (num_features * num_classes + num_classes,)
            instances: training data, shape (num_instances, num_features)
            labels: class labels, shape (num_instances,)
            labels_onehot: one-hot labels, shape (num_instances, num_classes)

        Returns:
            tuple: (loss, gradient) where
                - loss: average cross-entropy loss, a float
                - gradient: same shape and packing as parameters
        """
        # Unpack parameters: the last num_classes elements are biases
        weight_size = self.num_features * self.num_classes
        weights = parameters[:weight_size].reshape(self.num_features, self.num_classes)
        biases = parameters[weight_size:]

        # Forward pass, then loss and gradient
        probs = self._compute_probs(instances, weights, biases)
        loss = uf.cross_entropy_loss(labels, probs)
        grad_weights, grad_biases = self._compute_gradients(
            instances, labels_onehot, probs
        )

        # Pack gradient in the same layout as the parameters
        gradient = np.concatenate([grad_weights.flatten(), grad_biases])

        return loss, gradient

    # ------------------------------------------------------------------
    # Gradient descent
    # ------------------------------------------------------------------

    def train_gd(self, instances, labels):
        """
        Train the model by minimizing cross-entropy loss with gradient descent (GD).

        The cross-entropy loss is:
            L = -(1/N) Σ_i Σ_k y_ik · log(p_ik)
        where N is sample size, y_ik is the one-hot true label of instance i,
        and p_ik is the predicted probability of class k for instance i.

        Each iteration performs:
            1. a backward pass: compute gradients and update parameters
            2. a forward pass: compute predictions with the updated parameters,
               and log the new loss

        Training stops after num_iterations iterations, or earlier if the
        absolute change in loss caused by the latest update is below
        convergence_threshold.

        Args:
            instances: training data, shape (num_instances, num_features)
            labels: class labels, shape (num_instances,)
        """
        self._check_labels(labels)
        self._init_parameters(instances.shape[1], len(np.unique(labels)))
        labels_onehot = uf.one_hot_encode(labels, self.num_classes)

        # Initialize the loss log
        self.loss_log = []
        probs = self.forward(instances)

        for _ in range(self.num_iterations):
            # Backward pass: compute gradients and update parameters
            self.backward(instances, labels_onehot, probs)

            # Forward pass with updated parameters; log the new loss
            probs = self.forward(instances)
            self.loss_log.append(uf.cross_entropy_loss(labels, probs))

            # Stop if this update barely changed the loss
            if self.convergence_threshold is not None and len(self.loss_log) >= 2:
                loss_change = abs(self.loss_log[-2] - self.loss_log[-1])
                if loss_change < self.convergence_threshold:
                    break

    def backward(self, instances, labels_onehot, probs):
        """
        Backward pass: compute gradients and update parameters (one GD step).

            W := W - η ∂L/∂W
            b := b - η ∂L/∂b

        Args:
            instances: training data, shape (num_instances, num_features)
            labels_onehot: one-hot labels, shape (num_instances, num_classes)
            probs: predicted probabilities from the forward pass,
                computed with the current parameters, shape (num_instances, num_classes)
        """
        grad_weights, grad_biases = self._compute_gradients(
            instances, labels_onehot, probs
        )
        self.weights -= self.learning_rate * grad_weights
        self.biases -= self.learning_rate * grad_biases

    # ------------------------------------------------------------------
    # BFGS
    # ------------------------------------------------------------------

    def train_bfgs(self, instances, labels):
        """
        Train the model by minimizing cross-entropy loss with BFGS (scipy).

        BFGS is a quasi-Newton method: it uses only gradients, and builds an
        approximation of the inverse Hessian from successive gradient changes
        to choose the search direction. It does not use learning_rate or
        convergence_threshold; num_iterations is passed as scipy's maxiter.

        The weights and biases are packed into one 1D vector
        [W.flatten(), b], because scipy optimizes a single flat vector.
        The objective function returns both the loss and its analytical
        gradient (jac=True).

        If BFGS stops before its own convergence test is satisfied (e.g. it
        reaches maxiter, or the data is linearly separable so the optimal
        weights go to infinity), a warning message is printed.

        Args:
            instances: training data, shape (num_instances, num_features)
            labels: class labels, shape (num_instances,)
        """
        self._check_labels(labels)
        self._init_parameters(instances.shape[1], len(np.unique(labels)))
        labels_onehot = uf.one_hot_encode(labels, self.num_classes)

        # Pack parameters into a 1D vector: [weights..., biases...]
        initial_params = np.concatenate([self.weights.flatten(), self.biases])

        # Initialize the loss log
        self.loss_log = []

        def log_loss(params):
            """Called by scipy once per iteration; record the current loss."""
            loss, _ = self._compute_loss_and_gradients(
                params, instances, labels, labels_onehot
            )
            self.loss_log.append(loss)

        # Run BFGS optimization
        result = minimize(
            fun=self._compute_loss_and_gradients,
            x0=initial_params,
            args=(instances, labels, labels_onehot),
            method="BFGS",
            jac=True,
            callback=log_loss,
            options={"maxiter": self.num_iterations},
        )

        # Unpack optimized parameters: the last num_classes elements are biases
        weight_size = self.num_features * self.num_classes
        self.weights = result.x[:weight_size].reshape(
            self.num_features, self.num_classes
        )
        self.biases = result.x[weight_size:]

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
            instances: input data, shape (num_instances, num_features)

        Returns:
            labels: predicted class labels in [0, num_classes-1], shape (num_instances,), dtype=int
        """
        # Compute logits of all classes
        logits = self._compute_logits(instances, self.weights, self.biases)

        # Find the class with the highest logit for each instance
        return np.argmax(logits, axis=1)

    def forward(self, instances):
        """
        Forward pass: compute class probabilities with current parameters.

        Args:
            instances: input data, shape (num_instances, num_features)

        Returns:
            probs: class probabilities, shape (num_instances, num_classes)
        """
        # Compute logits and apply softmax to get probabilities
        logits = self._compute_logits(instances, self.weights, self.biases)
        return uf.softmax(logits)


##############################################################################
#                              MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Multinomial Logistic Regression model on a selected dataset",
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
        "--algorithm",
        type=str,
        default="gd",
        choices=["gd", "bfgs"],
        help="Optimization algorithm: GD (gradient descent) or BFGS (quasi-Newton)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=DEFAULT_NUM_ITERATIONS_GD,
        help="Number of training iterations (default: 10000 for GD, 100 for BFGS)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate only for gradient descent",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=DEFAULT_CONVERGENCE_THRESHOLD,
        help="Threshold for convergence in terms of change in loss between consecutive iterations (only applies to GD)",
    )
    args = parser.parse_args()
    print("Train a Multinomial Logistic Regression model for multiclass classification")

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

    # Set num_iterations depending on algorithm
    num_iterations = (
        args.num_iterations
        if args.num_iterations is not None
        else {"gd": DEFAULT_NUM_ITERATIONS_GD, "bfgs": DEFAULT_NUM_ITERATIONS_BFGS}[
            args.algorithm
        ]
    )

    # Initialize multinomial logistic regression model
    model = MultinomialLogisticRegression(
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

    # Plot training loss curve of multinomial logistic regression
    pf.plot_training_curve(
        model.loss_log,
        "Multinomial Logistic Regression",
        "Cross-Entropy Loss",
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

    # Visualize the multinomial logistic regression model (only for 2D data)
    if num_features == 2:
        pf.visualize_multinomial_logistic_regression(
            training_instances, training_labels, model
        )
