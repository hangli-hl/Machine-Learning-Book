##############################################################################
#                       UTILITY FUNCTIONS FOR DEEP LEARNING
##############################################################################


import numpy as np
import torch

# Lower bound used to clip probabilities and avoid log(0)
EPS = 1e-6
# Constant to prevent overflow in exp()
CLIP = 500


##############################################################################
#                            EVALUATION FUNCTIONS
##############################################################################


def evaluate_accuracy(true_labels, predicted_labels):
    """
    Calculate classification accuracy.

    Accuracy = (1/N) Σ I(y == ŷ)  # where I() is the indicator function

    Args:
        true_labels: true class labels of shape (num_instances, ), labels in {0, 1, ..., num_classes-1}
        predicted_labels: predicted class labels of shape (num_instances, ), labels in {0, 1, ..., num_classes-1}

    Returns:
        accuracy: accuracy of classification, a float in [0, 1]
    """
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    correct_predictions = true_labels == predicted_labels
    accuracy = np.mean(correct_predictions)
    return accuracy


def evaluate_model_accuracy(model, data_loader, device):
    """
    Evaluate the neural network (ffn, cnn, vit) on data.

    Args:
        model: trained neural network model
        data_loader: DataLoader for evaluation
        device: device to run computations on (CPU or GPU)

    Returns:
        accuracy: classification accuracy, a float in [0, 1]
    """
    # Set the model to evaluation mode
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for instances, labels in data_loader:
            # Move tensors to device
            instances = instances.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(instances)
            _, predicted = torch.max(logits, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


def collect_misclassified_instances(model, data_loader, device, num_instances):
    """
    Collect the first misclassified instances from data.

    The instances are moved to the CPU but keep their channel axis, so that
    the display function decides how to lay them out for imshow.

    Args:
        model: trained classification model (FFN, CNN, ViT, ...)
        data_loader: DataLoader for evaluation
        device: device to run computations on (CPU or GPU)
        num_instances: maximum number of misclassified instances to collect

    Returns:
        misclassified_instances: list of misclassified instances, CPU torch
            tensors of shape (image_channels, image_size, image_size)
        true_labels: list of true labels, Python integers
        predicted_labels: list of predicted labels, Python integers
    """
    # Set the model to evaluation mode
    model.eval()

    misclassified_instances = []
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for instances, labels in data_loader:
            # Stop before the forward pass once enough instances are collected
            if len(misclassified_instances) >= num_instances:
                break

            # Move tensors to device
            instances = instances.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(instances)
            _, predicted = torch.max(logits, 1)

            # Find misclassified instances
            misclassified_indices = torch.where(predicted != labels)[0]

            for index in misclassified_indices:
                if len(misclassified_instances) >= num_instances:
                    break

                misclassified_instances.append(instances[index].cpu())
                true_labels.append(labels[index].item())
                predicted_labels.append(predicted[index].item())

    return misclassified_instances, true_labels, predicted_labels


##############################################################################
#                             ACTIVATION FUNCTIONS
##############################################################################


def sigmoid(z):
    """
    Calculate the sigmoid (logistic) function.

    Sigmoid function: σ(z) = 1 / (1 + e^(-z))

    Args:
        z: input value(s), a float or array

    Returns:
        sigmoid output(z) in (0, 1), a float or array
    """
    # Clip input to prevent overflow in exp()
    z = np.clip(z, -CLIP, CLIP)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a):
    """
    Calculate the derivative of the sigmoid function given its output.

    Given a = σ(z), the derivative is: σ'(z) = a * (1 - a)

    Args:
        a: sigmoid activations (outputs of sigmoid function), a float or array

    Returns:
        derivative: dσ/dz evaluated at the points where σ(z) = a, a float or array
    """
    return a * (1 - a)


def softmax(z):
    """
    Compute the softmax function.

    Softmax function: softmax(z_i) = exp(z_i) / Σ_j exp(z_j)

    Subtract max from each row for numerical stability.

    Args:
        z: logits of shape (num_instances, num_classes)

    Returns:
        probs: probabilities of shape (num_instances, num_classes)
    """
    # Subtract max from each row for numerical stability
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def relu(z):
    """
    Compute the ReLU activation function.

    ReLU function: relu(z) = max(0, z)

    Args:
        z: input of any shape

    Returns:
        output: ReLU activation, same shape as z
    """
    return np.maximum(0, z)


def relu_derivative(a):
    """
    Compute the derivative of ReLU function given its output.

    Given a = relu(z), the derivative is: 1 if a > 0, else 0.
    Note: since relu(z) > 0 iff z > 0, this is equivalent to
    checking the sign of the pre-activation.

    Args:
        a: ReLU activations (outputs of ReLU function), a float or array

    Returns:
        derivative: gradient, same shape as a
    """
    return np.where(a > 0, 1, 0).astype(float)


##############################################################################
#                             LOSS FUNCTIONS
##############################################################################


def logistic_loss(labels, probs):
    """
    Calculate average logistic loss (binary cross-entropy loss).

    Logistic loss: L = -(1/N) Σ [y·log(p) + (1-y)·log(1-p)]

    Args:
        labels: true labels of shape (num_instances, ), labels in {0, 1}
        probs: predicted probabilities of shape (num_instances, ), values in (0, 1)

    Returns:
        loss: average logistic loss, a float
    """
    # Clip probabilities to avoid numerical instability from log(0)
    probs_clipped = np.clip(probs, EPS, 1 - EPS)

    # Calculate average logistic loss
    loss = -np.mean(
        labels * np.log(probs_clipped) + (1 - labels) * np.log(1 - probs_clipped)
    )
    return loss


def cross_entropy_loss(labels, probs):
    """
    Calculate average cross-entropy loss.

    Cross-entropy loss:
        L = -1/N Σ_i Σ_k y_ik * log(p_ik)

    Args:
        labels: one-hot encoded labels of shape (num_instances, num_classes)
        probs: predicted probabilities of shape (num_instances, num_classes)

    Returns:
        loss: average cross-entropy loss, a float
    """
    # Clip probabilities to avoid log(0)
    probs_clipped = np.clip(probs, EPS, 1 - EPS)

    # Calculate average cross-entropy loss
    loss = -np.mean(np.sum(labels * np.log(probs_clipped), axis=1))
    return loss


##############################################################################
#                          EXTRACT VAE LATENT REPRESENTATION
##############################################################################


def extract_vae_latent_representations(model, training_loader, device):
    """
    Extract the latent representations (means) for all images in the dataset.

    Args:
        model: trained VAE model
        training_loader: DataLoader for data
        device: device to run computations on (CPU or GPU)

    Returns:
        representations: latent representations (means), numpy array of shape (num_images, latent_size)
        labels: corresponding labels, numpy array of shape (num_images,)
    """
    # Set the model to evaluation mode
    model.eval()

    representations_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in training_loader:
            # Move tensors to device
            images = images.to(device)

            # Encode images to latent distribution parameters
            representations, _ = model.encoder(images)
            representations_list.append(representations.cpu())
            labels_list.append(labels)

    all_representations = torch.cat(representations_list)
    all_labels = torch.cat(labels_list)

    return all_representations.numpy(), all_labels.numpy()


##############################################################################
#                            ENCODING FUNCTIONS
##############################################################################


def one_hot_encode(labels, num_classes):
    """
    Convert class labels to one-hot vectors.

    Example: for 3 classes, label 1 becomes [0, 1, 0]

    Args:
        labels: class labels of shape (num_instances,)
        num_classes: total number of classes, a positive integer

    Returns:
        encoded_labels: one-hot encoded labels, shape (num_instances, num_classes)
    """

    return np.eye(num_classes)[labels]


##############################################################################
#                            SET DEVICE
##############################################################################


def set_device():
    """
    Select the PyTorch device for computation.

    Returns:
        device: torch.device, CUDA, MPS or CPU
    """
    # Set device (GPU if available, otherwise CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device
