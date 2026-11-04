##############################################################################
# GENERATIVE ADVERSARIAL NETWORK (GAN) WITH CONVOLUTIONAL NEURAL NETWORK (CNN)
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
MODEL_NAME = "GAN-CNN"
# Random seed for reproducibility
MODEL_SEED = 36
# Number of generated images to show
NUM_GENERATED_IMAGES = 36
# Default size of latent noise vectors
DEFAULT_LATENT_SIZE = 64
# Default number of training epochs
DEFAULT_NUM_EPOCHS = 50
# Default learning rate for Adam optimizer
DEFAULT_LEARNING_RATE = 0.0002
# Default batch size for training
DEFAULT_BATCH_SIZE = 128

##############################################################################
#                            GENERATOR NETWORK
##############################################################################


class Generator(nn.Module):
    """
    Convolutional Generator network for GAN.

    The network maps a latent noise vector to a generated image using
    transposed convolutions.

    The network consists of:
    - Unflatten: (batch_size, latent_size) -> (batch_size, latent_size, 1, 1)
    - ConvTranspose2d: latent_size -> 384, 4x4 kernel, stride=1, padding=0,
      with BatchNorm2d and ReLU
    - ConvTranspose2d: 384 -> 192, 4x4 kernel, stride=1, padding=0,
      with BatchNorm2d and ReLU
    - ConvTranspose2d: 192 -> 64, 4x4 kernel, stride=2, padding=1,
      with BatchNorm2d and ReLU
    - ConvTranspose2d: 64 -> 1, 4x4 kernel, stride=2, padding=1,
      with Tanh activation

    Input: (batch_size, latent_size)
    After Unflatten: (batch_size, latent_size, 1, 1)
    After Layer 1: (batch_size, 384, 4, 4)
    After Layer 2: (batch_size, 192, 7, 7)
    After Layer 3: (batch_size, 64, 14, 14)
    Output: (batch_size, image_channels = 1, image_size = 28, image_size = 28)

    Attributes:
        cnn: sequential container of transposed convolutional layers
    """

    def __init__(self, latent_size, image_channels, image_size):
        """
        Initialize a generator network model.

        Args:
            latent_size: size of input latent vectors, a positive integer
            image_channels: number of channels of output images, a positive integer
            image_size: size of input images (height = width), a positive integer
        """
        super().__init__()
        assert image_size == 28, "The CNN architecture is designed for 28x28 images"

        # Define transposed convolutional layers
        self.cnn = nn.Sequential(
            nn.Unflatten(1, (latent_size, 1, 1)),
            nn.ConvTranspose2d(
                latent_size, 384, kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.ConvTranspose2d(
                384, 192, kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, image_channels, kernel_size=4, stride=2, padding=1, bias=True
            ),
            nn.Tanh(),
        )

    def forward(self, noises):
        """
        Forward pass: generate images from latent noises.

        Args:
            noises: latent noise vectors of shape (batch_size, latent_size)

        Returns:
            images: generated images, shape (batch_size, image_channels, image_size, image_size)
        """
        # Apply transposed convolutional layers
        images = self.cnn(noises)
        return images


##############################################################################
#                          DISCRIMINATOR NETWORK
##############################################################################


class Discriminator(nn.Module):
    """
    Convolutional Discriminator network for GAN.

    The network maps an input image to the probability that the image is real
    rather than generated.

    The network consists of:
    - Conv2d: 1 -> 64, 4x4 kernel, stride=2, padding=1, with BatchNorm2d and LeakyReLU
    - Conv2d: 64 -> 256, 4x4 kernel, stride=2, padding=1, with BatchNorm2d and LeakyReLU
    - Conv2d: 256 -> 512, 3x3 kernel, stride=2, padding=0, with BatchNorm2d and LeakyReLU
    - Conv2d: 512 -> 1, 3x3 kernel, stride=1, padding=0
    - Flatten
    - Sigmoid

    Input: (batch_size, image_channels = 1, image_size = 28, image_size = 28)
    After Layer 1: (batch_size, 64, 14, 14)
    After Layer 2: (batch_size, 256, 7, 7)
    After Layer 3: (batch_size, 512, 3, 3)
    After Layer 4: (batch_size, 1, 1, 1)
    After Flatten: (batch_size, 1)
    Output: (batch_size, output_size = 1)

    Attributes:
        cnn: sequential container of convolutional layers
    """

    def __init__(self, image_channels, image_size):
        """
        Initialize a discriminator network model.

        Args:
            image_channels: number of channels of input images, a positive integer
            image_size: size of input images (height = width), a positive integer
        """
        super().__init__()
        assert image_size == 28, "The CNN architecture is designed for 28x28 images"

        # Define convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(
                image_channels, 64, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=0, bias=True),
            nn.Flatten(),
            nn.Sigmoid(),
        )

    def forward(self, images):
        """
        Forward pass: classify images as real or generated.

        Args:
            images: input images of shape (batch_size, image_channels, image_size, image_size)

        Returns:
            probabilities: probabilities of images being real, shape (batch_size, 1)
        """
        # Apply convolutional layers
        probabilities = self.cnn(images)
        return probabilities


##############################################################################
#                     GENERATIVE ADVERSARIAL NETWORK
##############################################################################


class GAN(nn.Module):
    """
    Generative Adversarial Network for (MNIST) image generation.

    The model consists of:
    - Generator: maps latent noise vectors to generated images
    - Discriminator: classifies images as real or generated

    Attributes:
        generator: Generator network
        discriminator: Discriminator network
    """

    def __init__(self, image_channels, image_size, latent_size):
        """
        Initialize a generative adversarial network (GAN) model.

        Args:
            image_channels: number of image channels, a positive integer
            image_size: size of input images (height = width), a positive integer
            latent_size: size of latent representation, a positive integer
        """
        super().__init__()

        self.generator = Generator(latent_size, image_channels, image_size)
        self.discriminator = Discriminator(image_channels, image_size)


##############################################################################
#                            TRAINING FUNCTION
##############################################################################


def train_gan(
    model,
    training_loader,
    fixed_noises,
    num_epochs,
    criterion,
    optimizer_generator,
    optimizer_discriminator,
    device,
):
    """
    Train the generative adversarial network (GAN).

    Args:
        model: GAN model containing generator and discriminator
        training_loader: DataLoader for training data
        fixed_noises: latent vectors used for image generation, shape (36, latent_size)
        num_epochs: number of training epochs, a positive integer
        criterion: loss function (nn.BCELoss)
        optimizer_generator: optimizer for generator (Adam)
        optimizer_discriminator: optimizer for discriminator (Adam)
        device: device to run computations on (CPU or GPU)

    Returns:
        generator_loss_log: list of generator losses for each batch
        discriminator_loss_log: list of discriminator losses for each batch
    """
    generator_loss_log = []
    discriminator_loss_log = []
    num_batches = len(training_loader)
    latent_size = fixed_noises.size(1)

    # Set the model to training mode
    model.train()

    for epoch in range(num_epochs):
        for index, (images, _) in enumerate(training_loader):
            # Move tensors to device
            images = images.to(device)
            real_labels = torch.ones(images.size(0), 1).to(device)
            fake_labels = torch.zeros(images.size(0), 1).to(device)

            # --- Train Discriminator ---
            optimizer_discriminator.zero_grad()

            # Calculate discriminator loss with real images
            real_probabilities = model.discriminator(images)
            real_loss = criterion(real_probabilities, real_labels)

            # Calculate discriminator loss with generated images
            noises = torch.randn(images.size(0), latent_size).to(device)
            generated_images = model.generator(noises)
            gen_probabilities = model.discriminator(generated_images.detach())
            gen_loss = criterion(gen_probabilities, fake_labels)

            # Total discriminator loss
            discriminator_loss = (real_loss + gen_loss) / 2

            # Backward pass and optimization
            discriminator_loss.backward()
            optimizer_discriminator.step()

            # --- Train Generator ---
            optimizer_generator.zero_grad()

            # Calculate generator loss with generated images
            noises = torch.randn(images.size(0), latent_size).to(device)
            generated_images = model.generator(noises)
            gen_probabilities = model.discriminator(generated_images)
            generator_loss = criterion(gen_probabilities, real_labels)

            # Backward pass and optimization
            generator_loss.backward()
            optimizer_generator.step()

            # Store training losses for plotting
            generator_loss_log.append(generator_loss.item())
            discriminator_loss_log.append(discriminator_loss.item())

            # Print progress
            if (index + 1) % 100 == 0 or (index + 1) == num_batches:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{index + 1}/{num_batches}], "
                    f"Discriminator Loss: {discriminator_loss.item():.4f}, "
                    f"Generator Loss: {generator_loss.item():.4f}"
                )

        # Generate images at the end of each epoch
        generate_samples_gan(model, epoch, fixed_noises, num_epochs)

    return generator_loss_log, discriminator_loss_log


def generate_samples_gan(model, epoch, fixed_noises, num_epochs):
    """
    Evaluate the GAN by generating images from a fixed set of noise vectors.

    The same noise vectors are used after every epoch, so the images saved in
    pf.GEN_DIR show how training progresses.

    Args:
        model: trained GAN model
        epoch: current epoch, a non-negative integer
        fixed_noises: fixed latent vectors, shape (num_images, latent_size),
            already on the same device as the model, num_generated_images = 36
        num_epochs: total number of epochs, a positive integer
    """
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Images are in the tanh range [-1, 1], which will be re-scaled
        generated_images = model.generator(fixed_noises)

    # Set the model back to training mode
    model.train()

    # Only the image of the last epoch is displayed
    pf.plot_generated_images(
        generated_images,
        title=f"{MODEL_NAME}, Epoch {epoch + 1}",
        file_name=pf.default_file_name(MODEL_NAME, f"epoch_{epoch + 1}"),
        value_range=(-1.0, 1.0),
        show=(epoch + 1 == num_epochs),
    )


##############################################################################
#                              MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Generative Adversarial Network (GAN) with CNN "
        "for MNIST image generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--latent-size",
        type=int,
        default=DEFAULT_LATENT_SIZE,
        help="Size of latent noise vectors",
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
    print("Train a GAN with CNN for MNIST image generation")

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
    training_loader = dc.load_mnist_generation_data(batch_size, tanh_range=True)
    print("Using MNIST dataset")
    print(
        f"Training set: {len(training_loader.dataset)} samples, "
        f"{image_channels}x{image_size}x{image_size} images"
    )

    # Set random seed for reproducibility, after data creation
    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)

    # Initialize a generative adversarial network (GAN) model
    model = GAN(image_channels, image_size, latent_size).to(device)

    # Define loss function and optimizers
    criterion = nn.BCELoss()
    optimizer_generator = optim.Adam(
        model.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )
    optimizer_discriminator = optim.Adam(
        model.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )

    # Generate fixed latent vectors for monitoring training progress
    noise_generator = torch.Generator().manual_seed(MODEL_SEED)
    fixed_noises = torch.randn(
        NUM_GENERATED_IMAGES, latent_size, generator=noise_generator
    ).to(device)

    # Train model
    print(
        f"Training with Adam, number of epochs = {num_epochs}, "
        f"learning rate = {learning_rate}, batch size = {batch_size}, "
        f"latent size = {latent_size}"
    )
    generator_loss_log, discriminator_loss_log = train_gan(
        model,
        training_loader,
        fixed_noises,
        num_epochs,
        criterion,
        optimizer_generator,
        optimizer_discriminator,
        device,
    )
    print(f"Generated images saved to {pf.GEN_DIR}")

    # Plot training loss curves -> ./plot/gan_cnn_training_curve.png
    pf.plot_training_curve(
        {"Discriminator": discriminator_loss_log, "Generator": generator_loss_log},
        MODEL_NAME,
        y_label="binary cross-entropy loss",
    )
