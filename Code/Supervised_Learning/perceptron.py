##############################################################################
#                                PERCEPTRON
##############################################################################


import argparse
import numpy as np

from Data import data_loaders as dl
import utility_functions as uf
import plot_functions as pf

# Default number of epochs for training
DEFAULT_NUM_EPOCHS = 100
# Default learning rate for online updates
DEFAULT_LEARNING_RATE = 0.1
# Default convergence threshold for training error rate
DEFAULT_CONVERGENCE_THRESHOLD = 1e-6


##############################################################################
#                              PERCEPTRON CLASS
##############################################################################


class Perceptron:
    """
    Perceptron for binary classification.

    The model predicts class labels using the model: y = sign( w·x + b ), where w is the weight vector,
    x is the input instance (feature vector), b is the bias, y is the class label in {-1, 1},
    and sign() is the sign function.

    If z >= 0, then sign(z) = +1, otherwise sign(z) = -1.
    By convention, sign(0) = +1, so an instance on the hyperplane (w·x + b = 0) is predicted as +1.

    The learning algorithm employs online updates on the weights and bias to find a hyperplane
    w·x + b = 0 that separates the instances of the two classes by iteratively adjusting the
    weights and bias when there are misclassifications.

    Attributes:
        weights: feature weights, shape (num_features,)
        bias: bias term, a float
        num_epochs: number of epochs, a positive integer
        learning_rate: step size for weight updates, a positive float;
            the learning rate only rescales w and b; with zero initialization
            the predictions are invariant
        convergence_threshold: if specified, training stops when the training
            error rate is below this value
        training_error_log: list of training error rates;
            training_error_log[k] is the error rate after k+1 epochs,
            training_error_log[-1] is the error rate of the final model;
            len = number of epochs actually performed
    """

    def __init__(
        self,
        num_epochs=DEFAULT_NUM_EPOCHS,
        learning_rate=DEFAULT_LEARNING_RATE,
        convergence_threshold=DEFAULT_CONVERGENCE_THRESHOLD,
    ):
        """
        Initialize a Perceptron.
        """
        self.weights = None
        self.bias = None
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
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

    def _init_parameters(self, num_features):
        """
        Initialize weights and bias to zero.

        Args:
            num_features: number of features, a positive integer
        """
        self.weights = np.zeros(num_features, dtype=float)
        self.bias = 0.0

    def _compute_error_rate(self, instances, labels):
        """
        Compute the error rate (fraction of misclassified instances) on a dataset.

        Args:
            instances: instances of shape (num_instances, num_features)
            labels: true labels of shape (num_instances,), labels in {-1, 1}

        Returns:
            error_rate: error rate in [0, 1], a float
        """
        # Error rate = 1 - accuracy
        error_rate = 1.0 - uf.evaluate_accuracy(labels, self.predict(instances))
        return error_rate

    @staticmethod
    def _compute_gradients(instance, label, score):
        """
        Compute the gradients of the perceptron loss for one instance.

        The per-instance loss is L = max(0, -y·f(x)), so the gradients are
            ∂L/∂w = -y·x,  ∂L/∂b = -y    if y·f(x) <= 0 (misclassified or on the boundary)
            ∂L/∂w = 0,     ∂L/∂b = 0     otherwise

        Args:
            instance: one training instance, shape (num_features,)
            label: true label of the instance, in {-1, 1}
            score: score f(x) = w·x + b of the instance under the current parameters

        Returns:
            tuple: (grad_weights, grad_bias) where
                - grad_weights: shape (num_features,)
                - grad_bias: a float
        """
        if label * score <= 0:
            return -label * instance, -float(label)
        return np.zeros_like(instance, dtype=float), 0.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, instances, labels):
        """
        Train the Perceptron using stochastic updates on misclassified points.

        The loss function is heuristically defined as
            L = Σ_M - y_i·f(x_i)
        where M is the set of misclassified instances and f(x) = w·x + b is the model.

        In each epoch, the instances are visited in random order and each one
        triggers a backward pass (one stochastic update). Only misclassified
        instances (or those on the decision boundary) change the parameters.

        The training error rate is logged after each epoch.

        Training stops when it reaches the specified number of epochs or when
        the training error rate converges.

        Args:
            instances: training instances of shape (num_instances, num_features)
            labels: training labels of shape (num_instances,), labels in {-1, 1}
        """
        self._check_labels(labels)
        self._init_parameters(instances.shape[1])

        # Initialize the training error log
        self.training_error_log = []

        for _ in range(self.num_epochs):
            # Stochastic updates in random order
            for index in np.random.permutation(len(labels)):
                self.backward(instances[index], labels[index])

            # Log the training error rate of this epoch
            self.training_error_log.append(self._compute_error_rate(instances, labels))

            # Stop if the error rate is below the threshold
            if (
                self.convergence_threshold is not None
                and self.training_error_log[-1] < self.convergence_threshold
            ):
                break

    def backward(self, instance, label):
        """
        Backward pass: compute gradients on one instance and update parameters.

            w := w - η ∂L/∂w
            b := b - η ∂L/∂b

        Args:
            instance: one training instance, shape (num_features,)
            label: true label of the instance, in {-1, 1}
        """
        score = self.forward(instance)
        grad_weights, grad_bias = self._compute_gradients(instance, label, score)
        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, instances):
        """
        Make predictions on multiple instances.

        Args:
            instances: input instances, shape (num_instances, num_features)

        Returns:
            labels: predicted class labels in {-1, 1}, shape (num_instances,), dtype=int
        """
        # Compute the scores of instances using the model
        scores = self.forward(instances)

        # Assign labels: label = 1 if score >= 0, label = -1 otherwise
        return np.where(scores >= 0, 1, -1)

    def forward(self, instances):
        """
        Forward pass: compute scores of one or multiple instances.

        Args:
            instances: input instances, shape (num_instances, num_features);
                a single instance of shape (num_features,) is also accepted

        Returns:
            scores: scores of instances of shape (num_instances,), values are float;
                a single float if a single instance is given
        """
        # Compute scores of instances using the model: f(x) = w·x + b
        return instances @ self.weights + self.bias


##############################################################################
#                              MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Perceptron on a selected dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="gaussians",
        choices=["gaussians", "iris", "jigsaw", "circle", "moons"],
        help="Dataset to use: gaussians (synthetic 2D, two gaussians), "
        "iris (Iris 4D), jigsaw (synthetic 2D, xor patterns), circle (synthetic 2D, circular patterns) "
        "or moons (synthetic 2D, two half moons)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for online updates",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=DEFAULT_CONVERGENCE_THRESHOLD,
        help="Threshold for convergence in terms of training error in an epoch",
    )
    args = parser.parse_args()
    print("Train a Perceptron model for binary classification")

    # Create dataset, labels in {-1, +1}
    training_instances, training_labels, test_instances, test_labels = (
        dl.load_binary_classification_data(dataset=args.data, signed_label=True)
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

    # Initialize the Perceptron model
    model = Perceptron(args.num_epochs, args.learning_rate, args.convergence_threshold)

    # Train the model and log training error rate
    print(
        f"Training with online updates, number of epochs = {model.num_epochs}, "
        f"learning rate = {model.learning_rate}, "
        f"convergence threshold = {model.convergence_threshold}"
    )
    model.train(training_instances, training_labels)
    print(f"Training completed after {len(model.training_error_log)} epochs")

    # Plot training error rate curve of perceptron
    pf.plot_training_curve(
        model.training_error_log, "Perceptron", "Training Error Rate"
    )

    # Evaluate on training set
    predicted_labels = model.predict(training_instances)
    training_accuracy = uf.evaluate_accuracy(training_labels, predicted_labels)
    print(f"Training accuracy: {training_accuracy:.4f}")

    # Evaluate on test set
    predicted_labels = model.predict(test_instances)
    test_accuracy = uf.evaluate_accuracy(test_labels, predicted_labels)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Visualize the perceptron model (only for 2D data)
    if num_features == 2:
        pf.visualize_perceptron(training_instances, training_labels, model)
