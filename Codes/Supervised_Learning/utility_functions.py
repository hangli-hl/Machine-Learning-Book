##############################################################################
#                    UTILITY FUNCTIONS FOR SUPERVISED LEARNING
##############################################################################


import numpy as np

# Prevents log(0)
LOG_EPS = 1e-12
# Prevents division by zero
DIV_EPS = 1e-8
# Prevents overflow in exp()
EXP_CLIP = 100.0


##############################################################################
#                           EVALUATION MEASURES
##############################################################################


def calculate_accuracy(true_labels, predicted_labels):
    """
    Calculate classification accuracy.

    Accuracy = (1/N) Σ_i I(y_i = ŷ_i)

    Args:
        true_labels: true class labels, shape (num_instances,)
        predicted_labels: predicted class labels, shape (num_instances,)
        true_labels and predicted_labels must have the same shape

    Returns:
        accuracy: accuracy of classification, a float in [0, 1]
    """
    return np.mean(true_labels == predicted_labels)


def calculate_mse(true_values, predicted_values):
    """
    Calculate mean squared error (MSE).

    MSE = (1/N) Σ_i (y_i - ŷ_i)²

    Args:
        true_values: true values, shape (num_instances,)
        predicted_values: predicted values, shape (num_instances,)
        true_values and predicted_values must have the same shape

    Returns:
        mse: mean squared error, a float
    """
    return np.mean((true_values - predicted_values) ** 2)


##############################################################################
#                           ACTIVATION FUNCTIONS
##############################################################################


def sigmoid(z):
    """
    Compute the sigmoid (logistic) function.

    Sigmoid function: σ(z) = 1 / (1 + e^(-z))

    Args:
        z: input value(s), shape (num_instances,) or float

    Returns:
        sigmoid output(s) in [0, 1], same shape as input
    """
    # Clip input to prevent overflow
    z_clipped = np.clip(z, -EXP_CLIP, EXP_CLIP)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def softmax(z):
    """
    Compute the softmax function.

    Convert a vector of real values into a probability distribution:
        softmax(z_i) = exp(z_i) / Σ_j exp(z_j)

    Use the max-shift trick for numerical stability

    Args:
        z: logits of shape (num_instances, num_classes)

    Returns:
        probs: probabilities with same shape as input, rows sum to 1
    """
    # Subtract max from each row for numerical stability
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def hinge(z):
    """
    Compute the hinge function.

    Hinge function: h(z) = max(0, 1 - z)
        - If z >= 1: loss = 0
        - If z < 1: loss = 1 - z

    Args:
        z: margin value(s), shape (num_instances,) or a float

    Returns:
        hinge function value(s), same shape as input
    """
    return np.maximum(0.0, 1.0 - z)


##############################################################################
#                         LOSS FUNCTIONS
##############################################################################


def logistic_loss(labels, probs):
    """
    Calculate average logistic loss (binary cross-entropy loss).

    Logistic loss: L = -(1/N) Σ [y_i·log(p_i) + (1-y_i)·log(1-p_i)]

    Args:
        labels: true labels of shape (num_instances,), labels in {0, 1}
        probs: predicted probabilities of shape (num_instances,), values in (0, 1)

    Returns:
        loss: average logistic loss, a float
    """
    # Clip probabilities to avoid numerical instability from log(0)
    probs_clipped = np.clip(probs, LOG_EPS, 1 - LOG_EPS)

    # Calculate average logistic loss
    return -np.mean(
        labels * np.log(probs_clipped) + (1 - labels) * np.log(1 - probs_clipped)
    )


def cross_entropy_loss(labels, probs):
    """
    Calculate cross-entropy loss using class labels.

    Cross-entropy: L = -(1/N) Σ log(p_i[y_i])

    Args:
        labels: class labels, shape (num_instances,), values are int in [0, num_classes-1]
        probs: predicted probabilities, shape (num_instances, num_classes)

    Returns:
        loss: cross-entropy loss, a float
    """
    # Clip probabilities to avoid numerical instability from log(0)
    probs_clipped = np.clip(probs, LOG_EPS, 1 - LOG_EPS)

    # Calculate average cross-entropy using advanced indexing
    return -np.mean(np.log(probs_clipped[np.arange(len(labels)), labels]))


##############################################################################
#                            ENCODING FUNCTIONS
##############################################################################


def one_hot_encode(labels, num_classes):
    """
    Convert integer labels to one-hot labels.

    Args:
        labels: integer labels, shape (num_instances,)
        num_classes: total number of classes

    Returns:
        one hot vectors: binary matrix, shape (num_instances, num_classes)
    """
    # Make the conversion using fancy indexing
    return np.eye(num_classes)[labels.astype(int)]
