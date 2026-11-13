##############################################################################
#       CONVOLUTIONAL NEURAL NETWORK (CNN) FOR IMAGE CLASSIFICATION
##############################################################################


import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from Data import data_loaders as dl
import utility_functions as uf
import plot_functions as pf

# Name used in plot titles and file names
MODEL_NAME = "CNN"
# Random seed for model initialization and training shuffling
MODEL_SEED = 64
# Number of misclassified images to show
NUM_MISCLASSIFIED = 20
# Default number of training epochs
DEFAULT_NUM_EPOCHS = 10
# Default learning rate for Adam optimizer
DEFAULT_LEARNING_RATE = 0.001
# Default batch size for training and testing
DEFAULT_BATCH_SIZE = 128

##############################################################################
#                    CONVOLUTIONAL NEURAL NETWORK CLASS
##############################################################################


class ConvolutionalNeuralNetwork(nn.Module):
    """
    Convolutional Neural Network for image (MNIST) classification.

    The network consists of:
    - Convolutional Block 1:
        - Conv2d: 1 input channel, 32 output channels, 5x5 kernel, padding=2
        - ReLU activation
        - MaxPool2d: 2x2 kernel, stride=2 (reduces spatial dimensions by half)
    - Convolutional Block 2:
        - Conv2d: 32 input channels, 64 output channels, 5x5 kernel, padding=2
        - ReLU activation
        - MaxPool2d: 2x2 kernel, stride=2 (reduces spatial dimensions by half)
    - Flatten: flattens feature maps to vectors of fc_input_size
    - Fully connected layer 1: 200 nodes with ReLU activation
    - Output layer: output_size nodes

    Input: (batch_size, image_channels, image_size, image_size)
    After Conv Block 1: (batch_size, 32, 14, 14)
    After Conv Block 2: (batch_size, 64, 7, 7)
    After Flatten: (batch_size, 64*7*7)
    After Fully Connected Layer 1: (batch_size, 200)
    Output: (batch_size, output_size)

    Attributes:
        cnn: sequential container of convolutional layers
        ffn: sequential container of fully connected layers
    """

    def __init__(self, image_channels, image_size, output_size):
        """
        Initialize a convolutional neural network model.
        """
        super().__init__()

        # Define convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Define fully connected layers
        conv_output_size = image_size // 4  # 28 // 4 = 7
        fc_input_size = 64 * conv_output_size * conv_output_size
        hidden_size = 200

        self.ffn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, images):
        """
        Forward pass: compute class logits.

        Args:
            images: input images of shape (batch_size, image_channels, image_size, image_size)

        Returns:
            logits: class logits, shape (batch_size, output_size)
        """
        # Apply convolutional layers
        features = self.cnn(images)

        # Apply fully connected layers
        logits = self.ffn(features)

        return logits


##############################################################################
#                            TRAINING FUNCTION
##############################################################################


def train_cnn(model, training_loader, num_epochs, criterion, optimizer, device):
    """
    Train the convolutional neural network using Adam.

    Args:
        model: ConvolutionalNeuralNetwork model
        training_loader: DataLoader for training data
        num_epochs: number of training epochs, a positive integer
        criterion: loss function (nn.CrossEntropyLoss)
        optimizer: optimizer (Adam)
        device: device to run computations on (CPU or GPU)

    Returns:
        loss_log: list of training losses for each batch
    """
    num_batches = len(training_loader)
    loss_log = []

    # Set the model to training mode
    model.train()

    for epoch in range(num_epochs):
        for index, (images, labels) in enumerate(training_loader):
            # Move tensors to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Store training loss for plotting
            loss_log.append(loss.item())

            # Print progress
            if (index + 1) % 100 == 0 or (index + 1) == num_batches:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Batch [{index + 1}/{num_batches}], Loss: {loss.item():.4f}"
                )

    return loss_log


##############################################################################
#                              MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Convolutional Neural Network (CNN) for image (MNIST) classification "
        "using PyTorch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training and testing",
    )
    args = parser.parse_args()
    print("Train a CNN for MNIST classification using PyTorch")

    # Set device for computing, CUDA, MPS, or CPU
    device = uf.set_device()
    print(f"Using device: {device}")

    # Set hyperparameters
    image_channels = 1  # MNIST images are grayscale images
    image_size = 28  # MNIST images are 28x28 pixels
    num_classes = 10  # 10 classes for digits 0-9

    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    # Load dataset
    training_loader, test_loader = dl.load_mnist_classification_data(batch_size)
    print(
        f"Training set: {len(training_loader.dataset)} images, "
        f"{image_channels}x{image_size}x{image_size} images, {num_classes} classes"
    )
    print(
        f"Test set: {len(test_loader.dataset)} images, "
        f"{image_channels}x{image_size}x{image_size} images, {num_classes} classes"
    )

    # Set random seed for reproducibility, after data creation
    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)

    # Initialize the convolutional neural network model
    model = ConvolutionalNeuralNetwork(
        image_channels=image_channels, image_size=image_size, output_size=num_classes
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print(
        f"Training with Adam, number of epochs = {num_epochs}, "
        f"learning rate = {learning_rate}, batch size = {batch_size}"
    )
    loss_log = train_cnn(
        model, training_loader, num_epochs, criterion, optimizer, device
    )

    # Plot training loss curve
    pf.plot_training_curve(
        loss_log,
        MODEL_NAME,
        x_label="training iterations",
        y_label="cross-entropy loss",
    )

    # Evaluate on training set
    training_accuracy = uf.evaluate_model_accuracy(model, training_loader, device)
    print(f"Training accuracy: {training_accuracy:.4f}")

    # Evaluate on test set
    test_accuracy = uf.evaluate_model_accuracy(model, test_loader, device)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Always display 20 misclassified images
    misclassified_images, true_labels, predicted_labels = (
        uf.collect_misclassified_instances(
            model, test_loader, device, num_instances=NUM_MISCLASSIFIED
        )
    )
    print(f"Show the first {len(misclassified_images)} misclassified test instances")
    pf.display_misclassified_images(
        misclassified_images,
        true_labels,
        predicted_labels,
        model_name=MODEL_NAME,
    )
