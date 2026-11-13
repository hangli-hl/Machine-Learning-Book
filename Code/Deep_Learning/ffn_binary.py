##############################################################################
#              FEEDFORWARD NETWORK (FFN) FOR BINARY CLASSIFICATION
##############################################################################


import argparse
import numpy as np

from Data import data_loaders as dl
import utility_functions as uf
import plot_functions as pf

# Random seed for model initialization and training shuffling
MODEL_SEED = 64
# Default number of hidden units
DEFAULT_HIDDEN_SIZE = 4
# Default number of training epochs
DEFAULT_NUM_EPOCHS = 10000
# Default learning rate for gradient descent
DEFAULT_LEARNING_RATE = 0.1
# Default mini-batch size (None: full-batch GD)
DEFAULT_BATCH_SIZE = None

##############################################################################
#                         FEEDFORWARD NETWORK CLASS
##############################################################################


class FeedforwardBinary:
    """
    Feedforward Network for binary classification.

    A network with one hidden layer and one output layer using sigmoid activations.
    The network predicts probabilities using: p = σ(W2·σ(W1·x + b1) + b2),
    where W1 and b1 are the first layer weights and bias, W2 and b2 are the
    output layer weights and bias, x is the input instance, p is the predicted
    probability of the positive class, and σ() is the sigmoid function.

    The model outputs a probability and uses a 0.5 threshold for classification.
    The class label is predicted as 1 if p >= 0.5, else 0.

    The learning algorithm is based on gradient descent (GD) or stochastic
    gradient descent (SGD) with backpropagation.

    Attributes:
        hidden_size: number of hidden units, a positive integer
        num_epochs: number of training epochs, a positive integer
        learning_rate: step size for parameter updates, a positive float
        batch_size: number of instances per mini-batch, a positive integer
            or None for full-batch GD
        hidden_weights: hidden layer weights, shape (num_features, hidden_size)
        hidden_biases: hidden layer bias, shape (1, hidden_size)
        output_weights: output layer weights, shape (hidden_size, 1)
        output_biases: output layer bias, shape (1, 1)
        hidden_acts: activations of the hidden layer through forward pass, shape (num_instances, hidden_size)
        output_probs: probabilities of the output layer through forward pass, shape (num_instances, 1)
        loss_log: list of training logistic losses;
            loss_log[k] is the loss after k+1 epochs;
            loss_log[-1] is the loss of the final trained model;
            len = number of epochs performed
    """

    def __init__(
        self,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        num_epochs=DEFAULT_NUM_EPOCHS,
        learning_rate=DEFAULT_LEARNING_RATE,
        batch_size=DEFAULT_BATCH_SIZE,
    ):
        """
        Initialize a Feedforward Network.
        """
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.hidden_weights = None
        self.hidden_biases = None
        self.output_weights = None
        self.output_biases = None

        self.hidden_acts = None
        self.output_probs = None
        self.loss_log = None

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _init_parameters(self, num_features):
        """
        Initialize weights (Xavier's method) and biases (zero).

        Args:
            num_features: number of features, a positive integer
        """
        # First layer
        self.hidden_weights = np.random.randn(num_features, self.hidden_size) * np.sqrt(
            2.0 / (num_features + self.hidden_size)
        )
        self.hidden_biases = np.zeros((1, self.hidden_size))

        # Second layer (single output unit for binary classification)
        self.output_weights = np.random.randn(self.hidden_size, 1) * np.sqrt(
            2.0 / (self.hidden_size + 1)
        )
        self.output_biases = np.zeros((1, 1))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, instances, labels):
        """
        Train the Feedforward Network using gradient descent (GD) or
        stochastic gradient descent (SGD).

        If batch_size is smaller than the number of instances, mini-batch SGD
        is used: the data is shuffled each epoch and processed in mini-batches.
        Otherwise (batch_size is None or too large), full-batch GD is used.

        Each mini-batch step performs:
            1. a forward pass: compute predictions on the mini-batch
            2. a backward pass: compute gradients and update parameters
        After each epoch, the loss on the full training set is logged.

        Args:
            instances: training instances of shape (num_instances, num_features)
            labels: training labels of shape (num_instances,), labels in {0, 1}
        """
        # Set random seed, then initialize parameters
        np.random.seed(MODEL_SEED)
        self._init_parameters(instances.shape[1])

        num_instances = instances.shape[0]
        # Reshape labels to a column vector: (num_instances, 1)
        labels = np.asarray(labels).reshape(-1, 1)

        self.loss_log = []

        # Default to full-batch GD if batch_size is not specified or too large
        batch_size = self.batch_size
        if batch_size is None or batch_size >= num_instances:
            batch_size = num_instances

        use_sgd = batch_size < num_instances

        for epoch in range(self.num_epochs):
            # Shuffle data at the beginning of each epoch for SGD
            if use_sgd:
                indices = np.random.permutation(num_instances)
                instances_shuffled = instances[indices]
                labels_shuffled = labels[indices]
            else:
                instances_shuffled = instances
                labels_shuffled = labels

            # Process mini-batches
            for start in range(0, num_instances, batch_size):
                end = min(start + batch_size, num_instances)
                batch_instances = instances_shuffled[start:end]
                batch_labels = labels_shuffled[start:end]

                # Forward pass: compute predictions on mini-batch
                self.forward(batch_instances)

                # Backward pass: compute gradients and update parameters
                self.backward(batch_instances, batch_labels)

            # Calculate and store average logistic loss on full training set
            self.forward(instances)
            loss = uf.logistic_loss(labels, self.output_probs)
            self.loss_log.append(loss)

            # Print progress
            if (epoch + 1) % 1000 == 0 or (epoch + 1) == self.num_epochs:
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss:.4f}")

    # ------------------------------------------------------------------
    # Forward and backward passes
    # ------------------------------------------------------------------

    def forward(self, instances):
        """
        Forward pass: compute predicted probabilities with current parameters.

        Args:
            instances: multiple input instances of shape (num_instances, num_features)

        Returns:
            probs: predicted probabilities of positive class, shape (num_instances, 1)
        """
        # Compute hidden layer: h = σ(W1·x + b1)
        hidden_scores = instances @ self.hidden_weights + self.hidden_biases
        self.hidden_acts = uf.sigmoid(hidden_scores)

        # Compute output layer: p = σ(W2·h + b2)
        output_scores = self.hidden_acts @ self.output_weights + self.output_biases
        self.output_probs = uf.sigmoid(output_scores)

        return self.output_probs

    def backward(self, instances, labels):
        """
        Backward pass: compute gradients and update parameters (one GD step).

        Must be called after forward() on the same instances.

        Args:
            instances: input instances of shape (num_instances, num_features)
            labels: true labels of shape (num_instances, 1), labels in {0, 1}
        """
        num_instances = instances.shape[0]

        # ============================================================
        # Compute all error signals
        # ============================================================

        # Calculate output layer errors
        output_errors = labels - self.output_probs  # shape: (num_instances, 1)

        # Calculate hidden layer errors
        hidden_errors = uf.sigmoid_derivative(self.hidden_acts) * (
            output_errors @ self.output_weights.T
        )  # shape: (num_instances, hidden_size)

        # ============================================================
        # Update all weights and biases
        # ============================================================

        # Update output layer weights and biases
        self.output_weights += (
            self.learning_rate * (self.hidden_acts.T @ output_errors) / num_instances
        )
        self.output_biases += (
            self.learning_rate
            * np.sum(output_errors, axis=0, keepdims=True)
            / num_instances
        )

        # Update hidden layer weights and biases
        self.hidden_weights += (
            self.learning_rate * (instances.T @ hidden_errors) / num_instances
        )
        self.hidden_biases += (
            self.learning_rate
            * np.sum(hidden_errors, axis=0, keepdims=True)
            / num_instances
        )

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
        probs = self.forward(instances)

        # Apply decision threshold: predict 1 if prob >= 0.5, else 0
        return (probs >= 0.5).astype(int).ravel()


##############################################################################
#                              MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Feedforward Network (FFN) model for binary classification "
        "using Numpy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="jigsaw",
        choices=["gaussians", "jigsaw", "circle", "moons"],
        help="Dataset to use: gaussians (synthetic 2D, two gaussians), jigsaw (synthetic 2D, XOR-like patterns), "
        "circle (synthetic 2D, circular pattern), moons (synthetic 2D, two moons)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help="Number of hidden units in the hidden layer",
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
        help="Learning rate for gradient descent",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Mini-batch size for SGD. If not specified, full-batch GD is used",
    )
    args = parser.parse_args()
    print("Train an FFN for binary classification using Numpy")

    # Load dataset
    training_instances, training_labels, test_instances, test_labels = (
        dl.load_binary_classification_data(dataset=args.data)
    )
    num_training_instances = training_instances.shape[0]
    num_features = training_instances.shape[1]
    num_classes = len(np.unique(training_labels))
    print(f"Using dataset: {args.data}")
    print(
        f"Training set: {num_training_instances} instances, "
        f"{num_features} features, {num_classes} classes"
    )
    print(
        f"Test set: {test_instances.shape[0]} instances, "
        f"{num_features} features, {num_classes} classes"
    )

    # Initialize the feedforward network model
    model = FeedforwardBinary(
        hidden_size=args.hidden_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )

    # Determine optimizer description
    if model.batch_size is None or model.batch_size >= num_training_instances:
        optimizer_desc = "full-batch GD"
    else:
        optimizer_desc = f"mini-batch SGD (batch size = {model.batch_size})"

    # Train the model
    print(
        f"Training with {optimizer_desc}, hidden size = {model.hidden_size}, "
        f"learning rate = {model.learning_rate}, number of epochs = {model.num_epochs}"
    )
    model.train(training_instances, training_labels)
    print(f"Training completed after {len(model.loss_log)} epochs")

    # Plot training loss of ffn binary
    pf.plot_training_curve(
        model.loss_log, "FFN-Binary", x_label="training epochs", y_label="logistic loss"
    )

    # Evaluate on training set
    predicted_training_labels = model.predict(training_instances)
    training_accuracy = uf.evaluate_accuracy(training_labels, predicted_training_labels)
    print(f"Training accuracy: {training_accuracy:.4f}")

    # Evaluate on test set
    predicted_test_labels = model.predict(test_instances)
    test_accuracy = uf.evaluate_accuracy(test_labels, predicted_test_labels)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Visualize the ffn binary model (only for 2D data)
    if num_features == 2:
        pf.visualize_ffn_binary(
            training_instances, training_labels, model, file_name="ffn_binary_model"
        )
