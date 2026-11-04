##############################################################################
#    VARIATIONAL AUTOENCODER (VAE) WITH CONVOLUTIONAL NEURAL NETWORK (CNN)
##############################################################################


import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import data_creators as dc
import utility_functions as uf
import plot_functions as pf

# Name used in plot titles and file names
MODEL_NAME = "VAE-CNN"
# Random seed for reproducibility
MODEL_SEED = 36


##############################################################################
#                            ENCODER NETWORK
##############################################################################


class Encoder(nn.Module):
    """
    Convolutional Encoder network for VAE.

    The network maps an input image to a latent distribution parameterized
    by mean (mu) and log variance (log_var).

    The network consists of:
    - Conv2d: 1 -> 64, 4x4 kernel, stride=2, padding=1, with BatchNorm2d and ReLU
    - Conv2d: 64 -> 128, 4x4 kernel, stride=2, padding=1, with BatchNorm2d and ReLU
    - Conv2d: 128 -> 256, 3x3 kernel, stride=2, padding=0, with BatchNorm2d and ReLU
    - Conv2d: 256 -> 512, 3x3 kernel, stride=1, padding=0, with BatchNorm2d and ReLU
    - Flatten
    - Linear (mu): 512 -> latent_size
    - Linear (log_var): 512 -> latent_size

    Input: (batch_size, image_channels=1, image_size=28, image_size=28)
    After Layer 1: (batch_size, 64, 14, 14)
    After Layer 2: (batch_size, 128, 7, 7)
    After Layer 3: (batch_size, 256, 3, 3)
    After Layer 4: (batch_size, 512, 1, 1)
    After Flatten: (batch_size, 512)
    Output mu: (batch_size, latent_size)
    Output log_var: (batch_size, latent_size)

    Attributes:
        cnn: sequential container of convolutional layers
        fc_mu: linear layer for mean output
        fc_log_var: linear layer for log variance output
    """

    def __init__(self, image_channels, image_size, latent_size):
        """
        Initialize the encoder network.

        Args:
            image_channels: number of image channels, a positive integer
            image_size: size of input images (height = width), a positive integer
            latent_size: size of latent representation, a positive integer
        """
        super().__init__()
        assert image_size == 28, "The CNN architecture is designed for 28x28 images"

        # Define convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(
                image_channels, 64, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Define mean and log variance layers
        self.fc_mu = nn.Linear(512, latent_size)
        self.fc_log_var = nn.Linear(512, latent_size)

    def forward(self, images):
        """
        Forward pass: encode images to latent distribution parameters.

        Args:
            images: input images of shape (batch_size, image_channels, image_size, image_size)

        Returns:
            mu: mean of latent distribution, shape (batch_size, latent_size)
            log_var: log variance of latent distribution, shape (batch_size, latent_size)
        """
        # Apply convolutional layers
        features = self.cnn(images)

        # Compute mean and log variance
        mu = self.fc_mu(features)
        log_var = self.fc_log_var(features)

        return mu, log_var


##############################################################################
#                            DECODER NETWORK
##############################################################################


class Decoder(nn.Module):
    """
    Convolutional Decoder network for VAE.

    The network maps a latent representation back to a generated image
    using transposed convolutions.

    The network consists of:
    - Unflatten: (batch_size, latent_size) -> (batch_size, latent_size, 1, 1)
    - ConvTranspose2d: 512 -> 256, 3x3 kernel, stride=1, padding=0,
      with BatchNorm2d and ReLU
    - ConvTranspose2d: 256 -> 128, 3x3 kernel, stride=2, padding=0,
      with BatchNorm2d and ReLU
    - ConvTranspose2d: 128 -> 64, 4x4 kernel, stride=2, padding=1,
      with BatchNorm2d and ReLU
    - ConvTranspose2d: 64 -> 1, 4x4 kernel, stride=2, padding=1,
      with Sigmoid activation

    Input: (batch_size, latent_size)
    After Unflatten: (batch_size, latent_size, 1, 1)
    After Layer 1: (batch_size, 256, 3, 3)
    After Layer 2: (batch_size, 128, 7, 7)
    After Layer 3: (batch_size, 64, 14, 14)
    Output: (batch_size, image_channels = 1, image_size = 28, image_size = 28)

    Attributes:
        cnn: sequential container of transposed convolutional layers
    """

    def __init__(self, latent_size, image_channels, image_size):
        """
        Initialize the decoder network.

        Args:
            latent_size: size of latent representation, a positive integer
            image_channels: number of image channels, a positive integer
            image_size: size of input images (height = width), a positive integer
        """
        super().__init__()
        assert image_size == 28, "The CNN architecture is designed for 28x28 images"

        # Define transposed convolutional layers
        self.cnn = nn.Sequential(
            nn.Unflatten(1, (latent_size, 1, 1)),
            nn.ConvTranspose2d(
                latent_size, 256, kernel_size=3, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=0, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, image_channels, kernel_size=4, stride=2, padding=1, bias=True
            ),
            nn.Sigmoid(),
        )

    def forward(self, latent_representations):
        """
        Forward pass: decode latent representations to generated images.

        Args:
            latent_representations: latent vectors of shape (batch_size, latent_size)

        Returns:
            generated_images: generated images, shape (batch_size, image_channels, image_size, image_size)
        """
        # Apply transposed convolutional layers
        generated_images = self.cnn(latent_representations)

        return generated_images


##############################################################################
#                       VARIATIONAL AUTOENCODER
##############################################################################


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) with Convolutional Neural Network.

    The model consists of a convolutional encoder that maps images to a latent
    distribution and a transposed convolutional decoder that reconstructs
    images from latent samples. The reparameterization trick is used to enable
    backpropagation through the stochastic sampling step.

    Attributes:
        encoder: Encoder network (convolutional)
        decoder: Decoder network (transposed convolutional)
    """

    def __init__(self, image_channels, image_size, latent_size):
        """
        Initialize a variational autoencoder (VAE) model.

        Args:
            image_channels: number of image channels, a positive integer
            image_size: size of input images (height = width), a positive integer
            latent_size: size of latent representation, a positive integer
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
            generated_images: generated images, same shape as input
            mu: mean of latent distribution, shape (batch_size, latent_size)
            log_var: log variance of latent distribution, shape (batch_size, latent_size)
        """
        # Encode images to latent distribution parameters
        mu, log_var = self.encoder(images)

        # Sample from latent distribution using reparameterization trick
        latent_representations = self.reparameterize(mu, log_var)

        # Decode latent samples to generated images
        generated_images = self.decoder(latent_representations)

        return generated_images, mu, log_var

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
        fixed_latents: latent vectors used for image generation, shape (36, latent_size)
        num_epochs: number of training epochs, a positive integer
        criterion: loss function (nn.BCELoss)
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
        for index, (images, _) in enumerate(training_loader):
            # Move tensors to device
            images = images.to(device)

            # Forward pass
            generated_images, mu, log_var = model(images)
            loss = compute_vae_loss(images, generated_images, mu, log_var, criterion)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Store training averaged loss for plotting
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


def compute_vae_loss(images, generated_images, mu, log_var, criterion):
    """
    Compute the VAE loss: reconstruction loss + KL divergence.

    Args:
        images: original images, shape (batch_size, image_channels, image_size, image_size)
        generated_images: generated images, shape (batch_size, image_channels, image_size, image_size)
        mu: mean of latent distribution, shape (batch_size, latent_size)
        log_var: log variance of latent distribution, shape (batch_size, latent_size)
        criterion: loss function (nn.BCELoss)

    Returns:
        loss: total VAE loss (reconstruction + KL divergence)
    """
    # Reconstruction loss: sum of per-pixel binary cross-entropy over the whole
    # batch (criterion uses reduction="sum"), and input images are normalized to [0, 1]
    reconstruction_loss = criterion(generated_images, images)

    # KL divergence loss
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return reconstruction_loss + kl_divergence


def generate_samples_vae(model, epoch, fixed_latents, num_epochs):
    """
    Evaluate the VAE by decoding a fixed set of latent vectors.

    The same latent vectors are decoded after every epoch, so the images saved
    in pf.GEN_DIR show how training progresses.

    Args:
        model: trained VAE model
        epoch: current epoch, a non-negative integer
        fixed_latents: fixed latent vectors, shape (num_images, latent_size),
            already on the same device as the model, num_generated_images = 36
        num_epochs: total number of epochs, a positive integer
    """
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Images are in the sigmoid range [0, 1], the default value_range
        generated_images = model.decoder(fixed_latents)

    # Set the model back to training mode
    model.train()

    # Only the image of the last epoch is displayed
    pf.plot_generated_images(
        generated_images,
        title=f"{MODEL_NAME}, Epoch {epoch + 1}",
        file_name=pf.default_file_name(MODEL_NAME, f"epoch_{epoch + 1}"),
        value_range=(0.0, 1.0),
        show=(epoch + 1 == num_epochs),
    )


##############################################################################
#                                MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Variational Autoencoder (VAE) with CNN for MNIST image generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--latent-size",
        type=int,
        default=16,
        help="Size of latent representation",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training",
    )
    args = parser.parse_args()
    print("Train a VAE with CNN for MNIST image generation")

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
    training_loader = dc.load_mnist_generation_data(batch_size)
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

    # Generate fixed latent vectors for monitoring training progress
    num_generated_images = 36
    noise_generator = torch.Generator().manual_seed(MODEL_SEED)
    fixed_latents = torch.randn(
        num_generated_images, latent_size, generator=noise_generator
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
    print(f"Generated images saved to {pf.GEN_DIR}")

    # Plot training loss curve -> ./plot/vae_cnn_training_curve.png
    pf.plot_training_curve(loss_log, "VAE-CNN", y_label="VAE loss per image")

    # Extract latent means and labels
    means, labels = uf.extract_vae_latent_representations(
        model, training_loader, device
    )

    # Plot latent space using t-SNE -> ./plot/vae_cnn_latent_space.png
    pf.plot_vae_latent_space(means, labels, model_name="VAE-CNN")
