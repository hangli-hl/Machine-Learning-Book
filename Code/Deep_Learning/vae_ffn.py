##############################################################################
#       VARIATIONAL AUTOENCODER (VAE) WITH FEEDFORWARD NETWORK (FFN)
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
MODEL_NAME = "VAE-FFN"
# Random seed for reproducibility
MODEL_SEED = 64
# Default size of latent representation
DEFAULT_LATENT_SIZE = 16
# Default number of training epochs
DEFAULT_NUM_EPOCHS = 50
# Default learning rate for Adam optimizer
DEFAULT_LEARNING_RATE = 0.001
# Default batch size for training
DEFAULT_BATCH_SIZE = 128
# Number of images to generate for evaluation
NUM_GEN_IMAGES = 36


##############################################################################
#                            ENCODER NETWORK
##############################################################################


class Encoder(nn.Module):
    """
    Feedforward Encoder network for VAE (MNIST) image generation.

    The network consists of:
    - Flatten: flattens input images to vectors of input_size
    - Hidden layer 1: 1024 nodes with ReLU activation
    - Hidden layer 2: 512 nodes with ReLU activation
    - Hidden layer 3: 256 nodes with ReLU activation
    - Output layer (mu): latent_size nodes
    - Output layer (log_var): latent_size nodes

    Input: (batch_size, image_channels=1, image_size=28, image_size=28)
    After Flatten: (batch_size, 784)
    After Hidden Layer 1: (batch_size, 1024)
    After Hidden Layer 2: (batch_size, 512)
    After Hidden Layer 3: (batch_size, 256)
    Output mu: (batch_size, latent_size)
    Output log_var: (batch_size, latent_size)

    Attributes:
        ffn: sequential container of feedforward layers
        fc_mu: linear layer for mean output
        fc_log_var: linear layer for log variance output
    """

    def __init__(self, image_channels, image_size, latent_size):
        """
        Initialize an encoder network model.

        Args:
            image_channels: number of channels of input images, a positive integer
            image_size: size of input images (height = width), a positive integer
            latent_size: size of latent representations, a positive integer
        """
        super().__init__()

        input_size = image_channels * image_size * image_size

        # Define feedforward layers
        self.ffn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Define mean and log variance layers
        self.fc_mu = nn.Linear(256, latent_size)
        self.fc_log_var = nn.Linear(256, latent_size)

    def forward(self, images):
        """
        Forward pass: encode images to latent distribution parameters.

        Args:
            images: input images of shape (batch_size, image_channels, image_size, image_size)

        Returns:
            mu: mean of latent distribution, shape (batch_size, latent_size)
            log_var: log variance of latent distribution, shape (batch_size, latent_size)
        """
        # Apply feedforward layers
        features = self.ffn(images)

        # Compute mean and log variance
        mu = self.fc_mu(features)
        log_var = self.fc_log_var(features)

        return mu, log_var


##############################################################################
#                            DECODER NETWORK
##############################################################################


class Decoder(nn.Module):
    """
    Feedforward Decoder network for VAE (MNIST) image generation.

    The network consists of:
    - Hidden layer 1: 256 nodes with ReLU activation
    - Hidden layer 2: 512 nodes with ReLU activation
    - Hidden layer 3: 1024 nodes with ReLU activation
    - Output layer: image_channels * image_size * image_size nodes with Sigmoid activation
    - Unflatten: reshapes output to (image_channels, image_size, image_size)

    Input: (batch_size, latent_size)
    After Hidden Layer 1: (batch_size, 256)
    After Hidden Layer 2: (batch_size, 512)
    After Hidden Layer 3: (batch_size, 1024)
    After Output Layer: (batch_size, 784)
    Output: (batch_size, image_channels=1, image_size=28, image_size=28)

    Attributes:
        ffn: sequential container of feedforward layers
    """

    def __init__(self, latent_size, image_channels, image_size):
        """
        Initialize a decoder network model.

        Args:
            latent_size: size of latent representations, a positive integer
            image_channels: number of channels of output images, a positive integer
            image_size: size of output images (height = width), a positive integer
        """
        super().__init__()

        output_size = image_channels * image_size * image_size

        # Define feedforward layers
        self.ffn = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size),
            nn.Sigmoid(),
            nn.Unflatten(1, (image_channels, image_size, image_size)),
        )

    def forward(self, latent_representations):
        """
        Forward pass: decode latent representations to generated images.

        Args:
            latent_representations: latent vectors of shape (batch_size, latent_size)

        Returns:
            gen_images: generated images, shape (batch_size, image_channels, image_size, image_size)
        """
        gen_images = self.ffn(latent_representations)
        return gen_images


##############################################################################
#                          VARIATIONAL AUTOENCODER
##############################################################################


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for (MNIST) image generation.

    The model consists of:
    - Encoder: maps images to latent distribution parameters (mu, log_var)
    - Reparameterization: samples latent vectors from N(mu, var) using N(0, 1)
    - Decoder: reconstructs images from latent samples

    Attributes:
        encoder: Encoder network
        decoder: Decoder network
    """

    def __init__(self, image_channels, image_size, latent_size):
        """
        Initialize a variational autoencoder (VAE) model.

        Args:
            image_channels: number of channels of images, a positive integer
            image_size: size of input images (height = width), a positive integer
            latent_size: size of latent representations, a positive integer
        """
        super().__init__()

        self.encoder = Encoder(image_channels, image_size, latent_size)
        self.decoder = Decoder(latent_size, image_channels, image_size)

    def forward(self, images):
        """
        Forward pass: encode, sample, and decode.

        Args:
            images: input images of shape (batch_size, image_channels, image_size, image_size)

        Returns:
            gen_images: generated images, same shape as input
            mu: mean of latent distribution, shape (batch_size, latent_size)
            log_var: log variance of latent distribution, shape (batch_size, latent_size)
        """
        # Encode images to latent distribution parameters
        mu, log_var = self.encoder(images)

        # Sample from latent distribution using reparameterization trick
        latent_representations = self.reparameterize(mu, log_var)

        # Decode latent samples to generated images
        gen_images = self.decoder(latent_representations)

        return gen_images, mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: sample from N(mu, var) using N(0, 1).

        Args:
            mu: mean of latent distribution, shape (batch_size, latent_size)
            log_var: log variance of latent distribution, shape (batch_size, latent_size)

        Returns:
            latent_representations: sampled latent vectors, shape (batch_size, latent_size)
        """
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std


##############################################################################
#                             TRAINING FUNCTION
##############################################################################


def train_vae(
    model,
    training_loader,
    fixed_latents,
    num_epochs,
    criterion,
    optimizer,
    device,
):
    """
    Train the variational autoencoder (VAE) model using Adam.

    Args:
        model: VAE model
        training_loader: DataLoader for training data
        fixed_latents: latent vectors used for image generation, shape (NUM_GEN_IMAGES, latent_size)
        num_epochs: number of training epochs, a positive integer
        criterion: loss function (nn.BCELoss)
        optimizer: optimizer (Adam)
        device: device to run computations on (CPU or GPU)

    Returns:
        loss_log: list of per-image training losses for each batch
    """
    num_batches = len(training_loader)
    loss_log = []

    # Set the model to training mode
    model.train()

    for epoch in range(num_epochs):
        for index, (images, _) in enumerate(training_loader):
            # Move tensors to device
            images = images.to(device)

            # Forward pass
            gen_images, mu, log_var = model(images)
            loss = compute_vae_loss(images, gen_images, mu, log_var, criterion)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Store training loss for plotting
            loss_log.append(loss.item() / images.size(0))

            # Print progress
            if (index + 1) % 100 == 0 or (index + 1) == num_batches:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{index + 1}/{num_batches}], "
                    f"Loss: {loss.item() / images.size(0):.4f}"
                )

        # Generate images at the end of each epoch
        generate_samples_vae(model, epoch, fixed_latents, num_epochs)

    return loss_log


def compute_vae_loss(images, gen_images, mu, log_var, criterion):
    """
    Compute the VAE loss: reconstruction loss + KL divergence.

    Args:
        images: original images, shape (batch_size, image_channels, image_size, image_size)
        gen_images: generated images, shape (batch_size, image_channels, image_size, image_size)
        mu: mean of latent distribution, shape (batch_size, latent_size)
        log_var: log variance of latent distribution, shape (batch_size, latent_size)
        criterion: loss function (nn.BCELoss)

    Returns:
        loss: total VAE loss (reconstruction + KL divergence)
    """
    # Reconstruction loss (binary cross-entropy)
    # Input images are normalized to [0, 1]
    reconstruction_loss = criterion(gen_images, images)

    # KL divergence loss
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return reconstruction_loss + kl_divergence


def generate_samples_vae(model, epoch, fixed_latents, num_epochs):
    """
    Evaluate the VAE by decoding a fixed set of latent vectors.

    The same latent vectors are decoded after every epoch, so the saved images
    show how training progresses.

    Args:
        model: trained VAE model
        epoch: current epoch, a non-negative integer
        fixed_latents: latent vectors, shape (num_images, latent_size),
            already on the same device as the model, num_images = NUM_GEN_IMAGES
        num_epochs: total number of epochs, a positive integer
    """
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Images are in the sigmoid range [0, 1]
        gen_images = model.decoder(fixed_latents)

    # Set the model back to training mode
    model.train()

    # Save generated images every 5 epochs; display only for the last epoch
    current_epoch = epoch + 1
    if current_epoch % 5 == 0 or current_epoch == num_epochs:
        pf.plot_generated_images(
            gen_images,
            title=f"{MODEL_NAME}, Epoch {current_epoch}",
            file_name=pf.default_file_name(MODEL_NAME, f"epoch_{current_epoch}"),
            value_range=(0.0, 1.0),
            show=(current_epoch == num_epochs),
        )


##############################################################################
#                       EXTRACT VAE LATENT REPRESENTATION
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
#                                MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Variational Autoencoder (VAE) with FFN "
        "for MNIST image generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--latent-size",
        type=int,
        default=DEFAULT_LATENT_SIZE,
        help="Size of latent representation",
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
        help="Batch size for training",
    )
    args = parser.parse_args()
    print("Train a VAE with FFN for MNIST image generation")

    # Set device for computing, CUDA, MPS, or CPU
    device = uf.set_device()
    print(f"Using device: {device}")

    # Set hyperparameters
    image_channels = 1  # MNIST images are grayscale
    image_size = 28  # MNIST images are 28x28 pixels

    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    latent_size = args.latent_size

    # Load dataset
    training_loader = dl.load_mnist_generation_data(batch_size)
    print("Using MNIST dataset")
    print(
        f"Training set: {len(training_loader.dataset)} samples, "
        f"{image_channels}x{image_size}x{image_size} images"
    )

    # Set random seed for reproducibility after data creation
    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)

    # Initialize a variational autoencoder (VAE) model
    model = VAE(image_channels, image_size, latent_size).to(device)

    # Define criterion and optimizer, the loss is defined as sum instead of mean
    criterion = nn.BCELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Generate latent vectors for monitoring training progress
    noise_generator = torch.Generator().manual_seed(MODEL_SEED)
    fixed_latents = torch.randn(
        NUM_GEN_IMAGES, latent_size, generator=noise_generator
    ).to(device)

    # Train model
    print(
        f"Training with Adam, number of epochs = {num_epochs}, "
        f"learning rate = {learning_rate}, batch size = {batch_size}, "
        f"latent size = {latent_size}"
    )
    loss_log = train_vae(
        model,
        training_loader,
        fixed_latents,
        num_epochs,
        criterion,
        optimizer,
        device,
    )
    print(f"Generated images saved to {pf.PLOT_DIR}")

    # Plot training loss curve -> ./plot/vae_ffn_training_curve.png
    pf.plot_training_curve(loss_log, MODEL_NAME, y_label="VAE loss per image")

    # Extract latent means and labels
    means, labels = uf.extract_vae_latent_representations(
        model, training_loader, device
    )

    # Plot latent space using t-SNE -> ./plot/vae_ffn_latent_space.png
    pf.plot_vae_latent_space(means, labels, model_name=MODEL_NAME)
