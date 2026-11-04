##############################################################################
#               VISION TRANSFORMER FOR IMAGE CLASSIFICATION
##############################################################################


import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import data_creators as dc
import utility_functions as uf
import plot_functions as pf

# Random seed for reproducibility
MODEL_SEED = 36
# Number of misclassified images to show
NUM_MISCLASSIFIED = 20
# Default number of training epochs
DEFAULT_NUM_EPOCHS = 10
# Default learning rate for Adam optimizer
DEFAULT_LEARNING_RATE = 0.001
# Default batch size for training and testing
DEFAULT_BATCH_SIZE = 128

##############################################################################
#                            VISION TRANSFORMER CLASS
##############################################################################


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image (MNIST) classification.

    The network consists of:
    - Patch Embedding:
        - Conv2d: 1 input channel, 128 output channels, 4x4 kernel, stride=4
        - Flatten and transpose to sequence of patch embeddings
    - Prepend CLS Token: learnable CLS token prepended to patch sequence
    - Add Positional Embedding: learnable positional embeddings added to all tokens
    - Embedding Dropout: dropout rate 0.1
    - Transformer Encoder: 4 Pre-Norm blocks, each with:
        - Layer Norm + Multi-Head Self-Attention (4 heads)
        - Layer Norm + MLP (hidden dim 256, GELU activation)
        - Dropout rate: 0.1
    - Classification Head: Layer Norm + Linear on CLS token

    Input: (batch_size, image_channels, image_size, image_size)
    After Patch Embedding: (batch_size, num_patches, embed_dim) = (batch_size, 49, 128)
    After Prepend CLS Token: (batch_size, num_patches + 1, embed_dim) = (batch_size, 50, 128)
    After Add Positional Embedding: (batch_size, 50, 128)
    After Transformer Encoder: (batch_size, 50, 128)
    Extract CLS Token: (batch_size, 128)
    Output: (batch_size, output_size = 10)

    Attributes:
        patch_embedding: Conv2d layer to extract and project patches
        cls_token: learnable CLS token parameter
        positional_embedding: learnable positional embedding parameter
        embedding_dropout: dropout after embedding
        transformer_encoder: nn.TransformerEncoder with Pre-Norm blocks
        layer_norm: final layer normalization
        classification_head: linear layer for classification
    """

    def __init__(self, image_channels, image_size, output_size):
        """
        Initialize a Vision Transformer model.

        Args:
            image_channels: number of channels of input images, a positive integer
            image_size: size of input images (height = width), a positive integer
            output_size: number of output classes, a positive integer
        """
        super().__init__()

        # Architecture hyperparameters
        patch_size = 4
        embed_dim = 128
        num_layers = 4
        num_heads = 4
        mlp_dim = 256
        dropout_rate = 0.1

        num_patches = (image_size // patch_size) ** 2

        # Patch embedding via Conv2d (equivalent to linear projection of flattened patches)
        self.patch_embedding = nn.Conv2d(
            in_channels=image_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Learnable CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim) * 0.02
        )
        self.embedding_dropout = nn.Dropout(dropout_rate)

        # Transformer encoder blocks (Pre-Norm architecture)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # no padding; avoids a nested-tensor warning
        )

        # Classification head
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.classification_head = nn.Linear(embed_dim, output_size)

    def forward(self, images):
        """
        Forward pass: compute class logits.

        Args:
            images: input images of shape (batch_size, image_channels, image_size, image_size)

        Returns:
            logits: class logits, shape (batch_size, output_size)
        """
        batch_size = images.shape[0]

        # Extract and embed patches via Conv2d
        # (batch, embed_dim, num_patches_h, num_patches_w) -> (batch, num_patches, embed_dim)
        patch_embeddings = self.patch_embedding(images).flatten(2).transpose(1, 2)

        # Prepend CLS token,
        # cls_token: [1, 1, embed_dim] -> cls_tokens: [batch, 1, embed_dim]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # (batch, 1, embed_dim) concat (batch, N, embed_dim) -> (batch, N + 1, embed_dim)
        x = torch.cat([cls_tokens, patch_embeddings], dim=1)

        # Add positional embeddings
        # (batch, N+1, embed_dim) + (1, N+1, embed_dim) -> (batch, N+1, embed_dim)
        x = self.embedding_dropout(x + self.positional_embedding)

        # Pass through transformer encoder
        # (batch, N+1, embed_dim) -> (batch, N+1, embed_dim)
        x = self.transformer_encoder(x)

        # Classification using CLS token
        # x[:, 0]: (batch, embed_dim) -> logits: (batch, output_size)
        logits = self.classification_head(self.layer_norm(x[:, 0]))

        return logits


##############################################################################
#                            TRAINING FUNCTION
##############################################################################


def train_vit(model, training_loader, num_epochs, criterion, optimizer, device):
    """
    Train the Vision Transformer using Adam.

    Args:
        model: VisionTransformer model
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
        description="Train a Vision Transformer (ViT) "
        "for image (MNIST) classification using PyTorch",
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
    print("Train a ViT for MNIST classification using PyTorch")

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
    training_loader, test_loader = dc.load_mnist_classification_data(batch_size)
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

    # Initialize the Vision Transformer model
    model = VisionTransformer(
        image_channels=image_channels,
        image_size=image_size,
        output_size=num_classes,
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print(
        f"Training with Adam, number of epochs = {num_epochs}, "
        f"learning rate = {learning_rate}, batch size = {batch_size}"
    )
    loss_log = train_vit(
        model, training_loader, num_epochs, criterion, optimizer, device
    )

    # Plot training loss curve
    pf.plot_training_curve(
        loss_log, "ViT", x_label="training iterations", y_label="cross-entropy loss"
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
        model_name="ViT",
    )
