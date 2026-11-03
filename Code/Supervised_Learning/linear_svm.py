##############################################################################
#                                 LINEAR SVM
##############################################################################


import argparse

import numpy as np

import data_creators as dc
import utility_functions as uf
import plot_functions as pf

##############################################################################
#                              LINEAR SVM CLASS
##############################################################################


class LinearSVM:
    """
    Linear SVM (Support Vector Machine) for binary classification.

    The model predicts class labels using: y = sign( w·x + b ), where w is the weight vector,
    x is the input instance (feature vector), b is the bias, y is the predicted class label in {-1, 1},
    and sign() is the sign function.

    If z >= 0, then sign(z) = +1, otherwise sign(z) = -1.
    By convention, sign(0) = +1, so an instance on the hyperplane (w·x + b = 0) is predicted as +1.

    The learning algorithm approximately solves the optimization problem and finds a hyperplane
    w·x + b = 0 using stochastic gradient descent (SGD).

    Attributes:
        weights: feature weights, shape (num_features,)
        bias: bias term, a float
        regularization: regularization coefficient, a positive float
        num_epochs: number of epochs, a positive integer
        learning_rate: step size for weight updates, a positive float
        convergence_threshold: if specified, training stops when the absolute
            change in regularized hinge loss between consecutive epochs is
            below this value
        loss_log: list of regularized hinge losses on the full training set;
            loss_log[k] is the loss after k+1 epochs;
            loss_log[-1] is the loss of the final model;
            len = number of epochs actually performed
    """

    def __init__(
        self, regularization, num_epochs, learning_rate, convergence_threshold=None
    ):
        """
        Initialize a Linear SVM.
        """
        self.weights = None
        self.bias = None
        self.regularization = regularization
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.loss_log = None

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

    def _compute_loss(self, instances, labels):
        """
        Compute the regularized average hinge loss on a dataset.

            L = 1/N Σ max(0, 1 - y_i·f(x_i)) + λ/2||w||²

        Args:
            instances: instances of shape (num_instances, num_features)
            labels: labels of shape (num_instances,), labels in {-1, 1}

        Returns:
            loss: regularized hinge loss, a float
        """
        # Margins y·f(x) of all instances
        margins = labels * self.forward(instances)

        # Average hinge loss plus L2 regularization (bias is not regularized)
        loss = np.mean(uf.hinge(margins)) + 0.5 * self.regularization * (
            self.weights @ self.weights
        )
        return loss

    def _compute_gradients(self, instance, label, score):
        """
        Compute the gradients of the regularized hinge loss for one instance.

        The per-instance loss is L = max(0, 1 - y·f(x)) + λ/2||w||², so the gradients are
            ∂L/∂w = -y·x + λ·w,  ∂L/∂b = -y    if y·f(x) < 1 (margin violation)
            ∂L/∂w = λ·w,         ∂L/∂b = 0     otherwise
        The bias is not regularized.

        Args:
            instance: one training instance, shape (num_features,)
            label: true label of the instance, in {-1, 1}
            score: score f(x) = w·x + b of the instance under the current parameters

        Returns:
            tuple: (grad_weights, grad_bias) where
                - grad_weights: shape (num_features,)
                - grad_bias: a float
        """
        grad_weights = self.regularization * self.weights
        if label * score < 1:
            return grad_weights - label * instance, -float(label)
        return grad_weights, 0.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, instances, labels):
        """
        Train the Linear SVM using stochastic gradient descent (SGD).

        The loss function is the hinge loss with L2 regularization.
            L = 1/N Σ max(0, 1 - y_i·f(x_i)) + λ/2||w||²
        where N is sample size, max(0, 1 - y_i·f(x_i)) is the hinge loss, and
        f(x) = w·x + b is the model.

        In each epoch, the instances are visited in random order and each one
        triggers a backward pass (one stochastic update):
            - If y·f(x) < 1 (margin violation):
                w := w + η y·x - η λ·w
                b := b + η y
            - If y·f(x) >= 1 (no violation):
                w := w - η λ · w
        After each epoch, the regularized hinge loss on the full training set is logged.

        Training stops when it reaches the specified number of epochs or when
        the regularized hinge loss converges.

        Args:
            instances: training instances of shape (num_instances, num_features)
            labels: training labels of shape (num_instances,), labels in {-1, 1}
        """
        self._check_labels(labels)
        self._init_parameters(instances.shape[1])

        # Initialize the loss log
        self.loss_log = []

        for _ in range(self.num_epochs):
            # Stochastic updates in random order
            for index in np.random.permutation(len(labels)):
                self.backward(instances[index], labels[index])

            # Log the regularized hinge loss of this epoch
            self.loss_log.append(self._compute_loss(instances, labels))

            # Stop if the loss change is below the threshold
            if self.convergence_threshold is not None and len(self.loss_log) >= 2:
                loss_change = abs(self.loss_log[-2] - self.loss_log[-1])
                if loss_change < self.convergence_threshold:
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
        # Compute scores of instances using the model
        scores = self.forward(instances)

        # Assign labels: label = 1 if score >= 0, label = -1 otherwise
        return np.where(scores >= 0, 1, -1)

    def forward(self, instances):
        """
        Compute scores of one or multiple instances.

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
        description="Train a Linear SVM on a selected dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="gaussians",
        choices=["artificial", "gaussians", "iris", "jigsaw", "circle", "moons"],
        help="Dataset to use: artificial (manual 2D), gaussians (synthetic 2D, two gaussians), "
        "iris (Iris 4D), jigsaw (synthetic 2D, xor patterns), circle (synthetic 2D, circular patterns) "
        "or moons (synthetic 2D, two half moons)",
    )
    parser.add_argument(
        "--regularization", type=float, default=0.01, help="Regularization coefficient"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for SGD",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=None,
        help="Threshold for convergence in terms of change in loss between consecutive epochs",
    )
    args = parser.parse_args()
    print("Train a Linear SVM for binary classification")

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

    # Initialize the linear Support Vector Machine model
    model = LinearSVM(
        args.regularization,
        args.num_epochs,
        args.learning_rate,
        args.convergence_threshold,
    )

    # Train the model and log training loss
    print(
        f"Training with SGD, number of epochs = {model.num_epochs}, "
        f"regularization coefficient = {model.regularization}, "
        f"learning rate = {model.learning_rate}, "
        f"convergence threshold = {model.convergence_threshold}"
    )
    model.train(training_instances, training_labels)
    print(f"Training completed after {len(model.loss_log)} epochs")

    # Plot training curve of model in terms of regularized hinge loss
    pf.plot_training_curve(model.loss_log, "Linear SVM", "Regularized Hinge Loss")

    # Evaluate on training set
    predicted_labels = model.predict(training_instances)
    training_accuracy = uf.calculate_accuracy(training_labels, predicted_labels)
    print(f"Training accuracy: {training_accuracy:.4f}")

    # Evaluate on test set
    predicted_labels = model.predict(test_instances)
    test_accuracy = uf.calculate_accuracy(test_labels, predicted_labels)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Visualize the linear svm model (only for 2D data)
    if num_features == 2:
        pf.visualize_linear_svm(training_instances, training_labels, model)
