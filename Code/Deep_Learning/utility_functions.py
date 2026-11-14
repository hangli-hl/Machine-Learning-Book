##############################################################################
#                       UTILITY FUNCTIONS FOR DEEP LEARNING
##############################################################################


import torch


##############################################################################
#                            EVALUATION FUNCTIONS
##############################################################################


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
