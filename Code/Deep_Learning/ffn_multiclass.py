##############################################################################
#          FEEDFORWARD NETWORK (FFN) FOR MULTICLASS CLASSIFICATION
##############################################################################


import argparse
import numpy as np

from Data import data_loaders as dl
import utility_functions as uf
import plot_functions as pf

# Random seed for model initialization and training shuffling
MODEL_SEED = 64
# Default number of nodes in the first hidden layer
DEFAULT_HIDDEN1_SIZE = 6
# Default number of nodes in the second hidden layer
DEFAULT_HIDDEN2_SIZE = 6
# Default number of training epochs
DEFAULT_NUM_EPOCHS = 10000
# Default learning rate for gradient descent
DEFAULT_LEARNING_RATE = 0.01
# Default mini-batch size (None: full-batch GD)
DEFAULT_BATCH_SIZE = None

##############################################################################
#                           FEEDFORWARD NETWORK CLASS
##############################################################################


class FeedforwardMulticlass:
    """
    Feedforward Network for multiclass classification.

    The network consists of:
    - Input layer: num_features nodes
    - First hidden layer: hidden1_size nodes with ReLU activation
    - Second hidden layer: hidden2_size nodes with ReLU activation
    - Output layer: num_classes nodes with softmax activation

    The learning algorithm minimizes cross-entropy loss using gradient descent
    (GD) or stochastic gradient descent (SGD) with backpropagation.

    Attributes:
        hidden1_size: number of nodes in first hidden layer, a positive integer
        hidden2_size: number of nodes in second hidden layer, a positive integer
        num_epochs: number of training epochs, a positive integer
        learning_rate: step size for parameter updates, a positive float
        batch_size: number of instances per mini-batch, a positive integer
            or None for full-batch GD
        input_size: number of input features
        output_size: number of output classes
        hidden1_weights: weights for first hidden layer, shape (input_size, hidden1_size)
        hidden2_weights: weights for second hidden layer, shape (hidden1_size, hidden2_size)
        output_weights: weights for output layer, shape (hidden2_size, output_size)
        hidden1_biases: biases for first hidden layer, shape (1, hidden1_size)
        hidden2_biases: biases for second hidden layer, shape (1, hidden2_size)
        output_biases: biases for output layer, shape (1, output_size)
        hidden1_acts: activations of the first hidden layer, shape (num_instances, hidden1_size)
        hidden2_acts: activations of the second hidden layer, shape (num_instances, hidden2_size)
        output_probs: probabilities of the output layer, shape (num_instances, output_size)
        loss_log: list of training cross-entropy losses;
            loss_log[k] is the loss after k+1 epochs;
            loss_log[-1] is the loss of the final trained model;
            len = number of epochs performed
    """

    def __init__(
        self,
        hidden1_size=DEFAULT_HIDDEN1_SIZE,
        hidden2_size=DEFAULT_HIDDEN2_SIZE,
        num_epochs=DEFAULT_NUM_EPOCHS,
        learning_rate=DEFAULT_LEARNING_RATE,
        batch_size=DEFAULT_BATCH_SIZE,
    ):
        """
        Initialize a Feedforward Network.
        """
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.input_size = None
        self.output_size = None

        self.hidden1_weights = None
        self.hidden1_biases = None
        self.hidden2_weights = None
        self.hidden2_biases = None
        self.output_weights = None
        self.output_biases = None

        self.hidden1_acts = None
        self.hidden2_acts = None
        self.output_probs = None
        self.loss_log = None

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _init_parameters(self, num_features, num_classes):
        """
        Initialize weights (He's method for hidden layers, Xavier's method
        for output layer) and biases (zero).

        Args:
            num_features: number of features, a positive integer
            num_classes: number of classes, a positive integer
        """
        self.input_size = num_features
        self.output_size = num_classes

        # First layer, He's method
        self.hidden1_weights = np.random.randn(
            self.input_size, self.hidden1_size
        ) * np.sqrt(2.0 / self.input_size)
        self.hidden1_biases = np.zeros((1, self.hidden1_size))

        # Second layer, He's method
        self.hidden2_weights = np.random.randn(
            self.hidden1_size, self.hidden2_size
        ) * np.sqrt(2.0 / self.hidden1_size)
        self.hidden2_biases = np.zeros((1, self.hidden2_size))

        # Output layer, Xavier's method
        self.output_weights = np.random.randn(
            self.hidden2_size, self.output_size
        ) * np.sqrt(2.0 / (self.hidden2_size + self.output_size))
        self.output_biases = np.zeros((1, self.output_size))

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
            instances: training instances of shape (num_instances, input_size)
            labels: class labels of shape (num_instances, ), values in {0, 1, ..., num_classes-1}
        """
        # Set random seed, then initialize parameters
        np.random.seed(MODEL_SEED)
        self._init_parameters(instances.shape[1], len(np.unique(labels)))

        # Convert class labels to one-hot vectors
        labels_onehot = uf.one_hot_encode(labels, self.output_size)

        num_instances = instances.shape[0]

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
                labels_onehot_shuffled = labels_onehot[indices]
            else:
                instances_shuffled = instances
                labels_onehot_shuffled = labels_onehot

            # Process mini-batches
            for start in range(0, num_instances, batch_size):
                end = min(start + batch_size, num_instances)
                batch_instances = instances_shuffled[start:end]
                batch_labels_onehot = labels_onehot_shuffled[start:end]

                # Forward pass: compute predictions on mini-batch
                self.forward(batch_instances)

                # Backward pass: compute gradients and update parameters
                self.backward(batch_instances, batch_labels_onehot)

            # Calculate and store average cross-entropy loss on full training set
            self.forward(instances)
            loss = uf.cross_entropy_loss(labels_onehot, self.output_probs)
            self.loss_log.append(loss)

            # Print progress
            if (epoch + 1) % 1000 == 0 or (epoch + 1) == self.num_epochs:
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss:.4f}")

    # ------------------------------------------------------------------
    # Forward and backward passes
    # ------------------------------------------------------------------

    def forward(self, instances):
        """
        Forward pass: compute class probabilities with current parameters.

        Args:
            instances: input instances of shape (num_instances, input_size)

        Returns:
            output_probs: class probabilities, shape (num_instances, output_size)
        """
        # First hidden layer (ReLU activation)
        self.hidden1_acts = uf.relu(
            instances @ self.hidden1_weights + self.hidden1_biases
        )

        # Second hidden layer (ReLU activation)
        self.hidden2_acts = uf.relu(
            self.hidden1_acts @ self.hidden2_weights + self.hidden2_biases
        )

        # Output layer (softmax activation)
        self.output_probs = uf.softmax(
            self.hidden2_acts @ self.output_weights + self.output_biases
        )

        return self.output_probs

    def backward(self, instances, labels):
        """
        Backward pass: compute gradients and update parameters (one GD step).

        Must be called after forward() on the same instances.

        Args:
            instances: input instances of shape (num_instances, input_size)
            labels: one-hot encoded labels of shape (num_instances, output_size)
        """
        num_instances = instances.shape[0]

        # ============================================================
        # Compute all error signals
        # ============================================================

        # Calculate output layer errors
        output_errors = (
            labels - self.output_probs
        )  # shape: (num_instances, num_classes)

        # Calculate second hidden layer errors
        hidden2_errors = uf.relu_derivative(self.hidden2_acts) * (
            output_errors @ self.output_weights.T
        )

        # Calculate first hidden layer errors
        hidden1_errors = uf.relu_derivative(self.hidden1_acts) * (
            hidden2_errors @ self.hidden2_weights.T
        )

        # ============================================================
        # Update all weights and biases
        # ============================================================

        # Update output layer weights and biases
        self.output_weights += (
            self.learning_rate * (self.hidden2_acts.T @ output_errors) / num_instances
        )
        self.output_biases += (
            self.learning_rate
            * np.sum(output_errors, axis=0, keepdims=True)
            / num_instances
        )

        # Update second hidden layer weights and biases
        self.hidden2_weights += (
            self.learning_rate * (self.hidden1_acts.T @ hidden2_errors) / num_instances
        )
        self.hidden2_biases += (
            self.learning_rate
            * np.sum(hidden2_errors, axis=0, keepdims=True)
            / num_instances
        )

        # Update first hidden layer weights and biases
        self.hidden1_weights += (
            self.learning_rate * (instances.T @ hidden1_errors) / num_instances
        )
        self.hidden1_biases += (
            self.learning_rate
            * np.sum(hidden1_errors, axis=0, keepdims=True)
            / num_instances
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, instances):
        """
        Predict class labels.

        Args:
            instances: input instances of shape (num_instances, input_size)

        Returns:
            probs: predicted class probabilities, shape (num_instances, output_size)
            labels: predicted class labels, shape (num_instances, 1)
        """
        # Compute class probabilities
        probs = self.forward(instances)

        # Find class label with highest probability
        labels = np.argmax(probs, axis=1)[:, np.newaxis]

        return probs, labels


##############################################################################
#                              MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Feedforward Network (FFN) model for multiclass classification "
        "using Numpy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="blobs",
        choices=["classification", "blobs", "iris"],
        help="Dataset to use: classification (Classification 20D), blobs (Blobs 2D), or iris (Iris 4D)",
    )
    parser.add_argument(
        "--hidden1-size",
        type=int,
        default=DEFAULT_HIDDEN1_SIZE,
        help="Number of nodes in first hidden layer",
    )
    parser.add_argument(
        "--hidden2-size",
        type=int,
        default=DEFAULT_HIDDEN2_SIZE,
        help="Number of nodes in second hidden layer",
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
    print("Train an FFN for multiclass classification using Numpy")

    # Load dataset
    training_instances, training_labels, test_instances, test_labels = (
        dl.load_multiclass_classification_data(dataset=args.data)
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
    model = FeedforwardMulticlass(
        hidden1_size=args.hidden1_size,
        hidden2_size=args.hidden2_size,
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
        f"Training with {optimizer_desc}, hidden1 size = {model.hidden1_size}, "
        f"hidden2 size = {model.hidden2_size}, "
        f"learning rate = {model.learning_rate}, number of epochs = {model.num_epochs}"
    )
    model.train(training_instances, training_labels)
    print(f"Training completed after {len(model.loss_log)} epochs")

    # Plot training loss of ffn multiclass
    pf.plot_training_curve(
        model.loss_log,
        "FFN-Multiclass",
        x_label="training epochs",
        y_label="cross-entropy loss",
    )

    # Evaluate on training set
    _, predicted_training_labels = model.predict(training_instances)
    training_accuracy = uf.evaluate_accuracy(
        training_labels, predicted_training_labels.ravel()
    )
    print(f"Training accuracy: {training_accuracy:.4f}")

    # Evaluate on test set
    _, predicted_test_labels = model.predict(test_instances)
    test_accuracy = uf.evaluate_accuracy(test_labels, predicted_test_labels.ravel())
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Visualize the ffn multiclass model (only for 2D data)
    if num_features == 2:
        pf.visualize_ffn_multiclass(
            training_instances, training_labels, model, file_name="ffn_multiclass_model"
        )
