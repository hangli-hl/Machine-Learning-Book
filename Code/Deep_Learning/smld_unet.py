##############################################################################
#  SCORE MATCHING WITH LANGEVIN DYNAMICS (SMLD) WITH U-SHAPED NETWORK (UNET)
##############################################################################


import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Make the repository root importable so that the Data package can be found
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Data import data_loaders as dl
import utility_functions as uf
import plot_functions as pf
import unet

# Name used in plot titles and file names
MODEL_NAME = "SMLD-UNet"
# Random seed for reproducibility
MODEL_SEED = 28
# A small constant to avoid division by zero
EPS = 1e-12
# Default number of training epochs
DEFAULT_NUM_EPOCHS = 30
# Default learning rate for Adam optimizer
DEFAULT_LEARNING_RATE = 0.0001
# Default batch size for training
DEFAULT_BATCH_SIZE = 128
# Default number of channels at the full resolution of the UNet
DEFAULT_BASE_CHANNELS = 64
# Default number of noise levels in the geometric schedule
DEFAULT_NUM_NOISE_LEVELS = 200
# Default smallest noise level
DEFAULT_SIGMA_MIN = 0.005
# Default largest noise level
DEFAULT_SIGMA_MAX = 15.0
# Default number of Langevin corrector steps per noise level
DEFAULT_NUM_LANGEVIN_STEPS = 1
# Default target signal-to-noise ratio for the Langevin corrector step size
DEFAULT_TARGET_SNR = 0.16
# Number of images to generate for evaluation
NUM_GENERATED_IMAGES = 36


##############################################################################
#                           SCORE NETWORK CLASS
##############################################################################


class ScoreUNet(nn.Module):
    """
    Noise score network for image data.

    Wraps a UNet regression network and divides its output by the noise
    level sigma
        s_theta(x, i) = net(x, i) / sigma_i
    This keeps the target of the network at unit scale across noise levels.

    Attributes:
        image_channels: number of image channels, a positive integer
        image_size: size of input images (height = width), a positive integer
        sigmas: geometric noise schedule, shape (num_noise_levels,), ascending
        unet: UNet producing unscaled predicted fields
    """

    def __init__(self, image_channels, image_size, sigmas, base_channels):
        """
        Initialize the score network.

        Args:
            image_channels: number of image channels, a positive integer
            image_size: size of input images (height = width), a positive integer
            sigmas: geometric noise schedule of shape (num_noise_levels,), ascending
            base_channels: number of channels at the full resolution of the UNet
        """
        super().__init__()
        self.image_channels = image_channels
        self.image_size = image_size
        self.register_buffer("sigmas", sigmas)
        self.unet = unet.UNet(image_channels, image_size, base_channels)

    def forward(self, x, noise_level):
        """
        Compute the score at the given noise level.

        Args:
            x: images of shape (batch_size, image_channels, image_size, image_size)
            noise_level: noise level index of shape (batch_size,)

        Returns:
            score: score estimate, same shape as x
        """
        # Predicted score scaled by the corresponding noise level
        score = self.unet(x, noise_level)
        sigma = self.sigmas[noise_level].reshape(-1, 1, 1, 1)
        return score / sigma


##############################################################################
#             SCORE MATCHING WITH LANGEVIN DYNAMICS UTILITIES
##############################################################################


class ScoreMatchingUNet:
    """
    Utilities for SMLD (score matching with Langevin dynamics), including
    forward and reverse processes.

    Forward process (add noise):
        q_sigma(x_tilde | x) = N(x_tilde; x, sigma_i^2 I)
        x_tilde = x_0 + sigma_i * epsilon

    Reverse process (predictor-corrector sampling, the corrector is Langevin dynamics):
        For each noise level i = L-1 .. 0:
          Predictor (reverse diffusion, sigma_i -> sigma_{i-1}):
            x <- x + (sigma_i^2 - sigma_{i-1}^2) * s_theta(x, sigma_i)
                   + sqrt(sigma_i^2 - sigma_{i-1}^2) * z
          Corrector (M Langevin steps at the new level sigma_{i-1}):
            step = 2 * (target_snr * ||z|| / ||s_theta(x, sigma_{i-1})||) ** 2
            x <- x + step * s_theta(x, sigma_{i-1}) + sqrt(2 * step) * z
        At i = 0 the predictor reduces to the deterministic denoising step
            x <- x + sigma_0^2 * s_theta(x, sigma_0)
        and the corrector is skipped, since sigma_{-1} = 0 is not a valid
        noise level for the score network.

    Attributes:
        num_noise_levels: number of noise levels, a positive integer
        sigmas: geometric noise schedule of shape (num_noise_levels,), ascending
        device: device to run computations on (CPU or GPU)
    """

    def __init__(self, num_noise_levels, sigma_min, sigma_max, device):
        """
        Initialize the score matching utilities.

        Args:
            num_noise_levels: number of noise levels, a positive integer
            sigma_min: smallest noise level, a positive float
            sigma_max: largest noise level, a positive float
            device: device to run computations on (CPU or GPU)
        """
        self.num_noise_levels = num_noise_levels
        self.device = device

        # Geometric noise schedule: sigma_0 < sigma_1 < ... < sigma_{L-1}
        self.sigmas = torch.exp(
            torch.linspace(
                math.log(sigma_min),
                math.log(sigma_max),
                num_noise_levels,
                device=device,
            )
        )

    def add_noise(self, x_0, noise, noise_level):
        """
        Forward process: perturb clean images with Gaussian noise.
            x_tilde = x_0 + sigma_i * epsilon

        Args:
            x_0: clean images of shape (batch_size, image_channels, image_size, image_size)
            noise: Gaussian noise of the same shape as x_0
            noise_level: noise level index, shape (batch_size,)

        Returns:
            x_tilde: perturbed images, same shape as x_0
        """
        sigma = self.sigmas[noise_level].reshape(-1, 1, 1, 1)
        return x_0 + sigma * noise

    def langevin_step(self, model, x, noise_level, target_snr):
        """
        Corrector: one Langevin step with an adaptive step size.
            s = s_theta(x, sigma_i)
            step = 2 * (target_snr * ||z|| / ||s||) ** 2
            x <- x + step * s + sqrt(2 * step) * z,   z ~ N(0, I)

        Args:
            model: trained score network
            x: current states, shape (batch_size, image_channels, image_size, image_size)
            noise_level: noise level index, a non-negative integer
            target_snr: target signal-to-noise ratio for the step size, a positive float

        Returns:
            x_next: updated states, same shape as x
        """
        batch_size = x.size(0)
        t = torch.full((batch_size,), noise_level, dtype=torch.long, device=self.device)
        scores = model(x, t)

        z = torch.randn_like(x)
        score_norm = scores.reshape(batch_size, -1).norm(dim=1).mean()
        noise_norm = z.reshape(batch_size, -1).norm(dim=1).mean()
        step = 2.0 * (target_snr * noise_norm / (score_norm + EPS)) ** 2

        return x + step * scores + torch.sqrt(2.0 * step) * z

    @torch.no_grad()
    def predictor_corrector_sampling(
        self,
        model,
        num_gen_images,
        num_langevin_steps,
        target_snr,
        return_traj=False,
    ):
        """
        Generate images with the predictor-corrector sampler: each predictor step
        is followed by num_langevin_steps corrector steps at the noise level
        the predictor moved to.

        Args:
            model: trained score network
            num_gen_images: number of images to generate, a positive integer
            num_langevin_steps: Langevin corrector steps per noise level, a positive integer
            target_snr: target signal-to-noise ratio for the corrector step size, a positive float
            return_traj: whether to also return the sampling trajectories, a boolean

        Returns:
            x: generated images in [-1, 1],
                shape (num_gen_images, image_channels, image_size, image_size)
            trajs: stacked trajectories (only if return_traj), shape
                (num_noise_levels + 1, num_gen_images, image_channels,
                image_size, image_size); one snapshot per noise level, taken
                after the corrector steps, plus the initial noise at trajs[0].
                The individual corrector steps are not recorded,
                so the snapshot count does not grow with num_langevin_steps.
        """
        model.eval()

        # Start from noise scaled to the largest noise level
        shape = (
            num_gen_images,
            model.image_channels,
            model.image_size,
            model.image_size,
        )
        x = torch.randn(shape, device=self.device) * self.sigmas[-1]
        traj_points = [x.cpu().clone()] if return_traj else None

        for level in reversed(range(self.num_noise_levels)):
            # Predictor: reverse-diffusion step from sigma_level to sigma_{level-1}
            noise_level = torch.full(
                (num_gen_images,), level, device=self.device, dtype=torch.long
            )
            sigma_current = self.sigmas[level]
            sigma_next = (
                self.sigmas[level - 1] if level > 0 else torch.zeros_like(sigma_current)
            )
            variance = sigma_current**2 - sigma_next**2

            x = x + variance * model(x, noise_level)
            if level > 0:
                x = x + torch.sqrt(variance) * torch.randn_like(x)

            # Corrector: Langevin steps at the level the predictor just moved to.
            # Skipped after the level-0 predictor, which is the deterministic
            # denoising step to sigma = 0.
            if level > 0:
                for _ in range(num_langevin_steps):
                    x = self.langevin_step(model, x, level - 1, target_snr)

            if return_traj:
                traj_points.append(x.cpu().clone())

        x = x.clamp(-1.0, 1.0)
        if return_traj:
            # The last snapshot is the clamped output, not the raw state
            traj_points[-1] = x.cpu().clone()
            trajs = torch.stack(traj_points)
            return x, trajs
        return x


##############################################################################
#                             TRAINING FUNCTION
##############################################################################


def train_smld(
    model,
    training_loader,
    num_epochs,
    criterion,
    optimizer,
    num_langevin_steps,
    target_snr,
    score_matching,
    device,
    num_gen_images,
):
    """
    Train the score network with denoising score matching.

    Args:
        model: score network (ScoreUNet)
        training_loader: DataLoader for training data
        num_epochs: number of training epochs, a positive integer
        criterion: loss function (nn.MSELoss)
        optimizer: optimizer (Adam)
        num_langevin_steps: Langevin corrector steps per noise level, a positive integer
        target_snr: target signal-to-noise ratio for the corrector step size, a positive float
        score_matching: score matching utilities with the noise schedule
        device: device to run computations on (CPU or GPU)
        num_gen_images: number of images to generate for evaluation, a positive integer

    Returns:
        loss_log: list of training losses for all batches
    """
    loss_log = []
    num_batches = len(training_loader)

    # Set the model to training mode
    model.train()

    for epoch in range(num_epochs):
        for index, (images, _) in enumerate(training_loader):
            # Move tensors to device (images are normalized to [-1, 1])
            images = images.to(device)

            # Sample noise and noise levels, then run the forward process
            noise = torch.randn_like(images)
            t = torch.randint(
                0, score_matching.num_noise_levels, (images.size(0),), device=device
            )
            x_tilde = score_matching.add_noise(images, noise, t)

            # Calculate the denoising score matching loss
            # scaled_scores = sigma * s_theta = raw network output; target is -noise
            scores = model(x_tilde, t)
            scaled_scores = scores * score_matching.sigmas[t].reshape(-1, 1, 1, 1)
            loss = criterion(scaled_scores, -noise)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Store training loss for plotting
            loss_log.append(loss.item())

            # Print progress
            if (index + 1) % 100 == 0 or (index + 1) == num_batches:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{index + 1}/{num_batches}], "
                    f"Loss: {loss.item():.4f}"
                )

        # Generate images at the end of each epoch
        generate_samples_smld(
            model,
            score_matching,
            num_langevin_steps,
            target_snr,
            num_gen_images,
            epoch,
            num_epochs,
        )

    return loss_log


##############################################################################
#                             GENERATION FUNCTION
##############################################################################


def generate_samples_smld(
    model,
    score_matching,
    num_langevin_steps,
    target_snr,
    num_gen_images,
    epoch,
    num_epochs,
):
    """
    Generate sample images using the learned model, i.e., the score network.

    Args:
        model: trained score network
        score_matching: score matching utilities with the noise schedule
        num_langevin_steps: Langevin corrector steps per noise level, a positive integer
        target_snr: target signal-to-noise ratio for the corrector step size, a positive float
        num_gen_images: number of images to generate, a positive integer
        epoch: current epoch, a non-negative integer
        num_epochs: total number of epochs, a positive integer
    """
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Images are in the tanh range [-1, 1]; plot_generated_images rescales them
        gen_images = score_matching.predictor_corrector_sampling(
            model,
            num_gen_images,
            num_langevin_steps,
            target_snr,
        )

    # Set the model back to training mode
    model.train()

    # Only the images from the last epoch are displayed
    pf.plot_generated_images(
        gen_images,
        title=f"{MODEL_NAME}, Epoch {epoch + 1}",
        file_name=pf.default_file_name(MODEL_NAME, f"epoch_{epoch + 1}"),
        value_range=(-1.0, 1.0),
        show=(epoch + 1 == num_epochs),
    )


##############################################################################
#                               MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Score Matching with Langevin Dynamics (SMLD) model "
        "with UNet for MNIST image generation",
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
        help="Batch size for training",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=DEFAULT_BASE_CHANNELS,
        help="Number of channels at the full resolution of the UNet",
    )
    parser.add_argument(
        "--num-noise-levels",
        type=int,
        default=DEFAULT_NUM_NOISE_LEVELS,
        help="Number of noise levels in the geometric schedule",
    )
    parser.add_argument(
        "--sigma-min",
        type=float,
        default=DEFAULT_SIGMA_MIN,
        help="Smallest noise level",
    )
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=DEFAULT_SIGMA_MAX,
        help="Largest noise level",
    )
    parser.add_argument(
        "--num-langevin-steps",
        type=int,
        default=DEFAULT_NUM_LANGEVIN_STEPS,
        help="Number of Langevin corrector steps per noise level",
    )
    parser.add_argument(
        "--target-snr",
        type=float,
        default=DEFAULT_TARGET_SNR,
        help="Target signal-to-noise ratio for the Langevin corrector step size",
    )
    args = parser.parse_args()
    print("Train an SMLD model with UNet for MNIST image generation")

    # Set device for computing, CUDA, MPS, or CPU
    device = uf.set_device()
    print(f"Using device: {device}")

    # Set hyperparameters
    image_channels = 1  # MNIST images are grayscale
    image_size = 28  # MNIST images are 28x28 pixels

    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    base_channels = args.base_channels
    num_noise_levels = args.num_noise_levels
    sigma_min = args.sigma_min
    sigma_max = args.sigma_max
    num_langevin_steps = args.num_langevin_steps
    target_snr = args.target_snr

    # Load dataset
    training_loader = dc.load_mnist_generation_data(batch_size, tanh_range=True)
    print(
        f"Training set: {len(training_loader.dataset)} samples, "
        f"{image_channels}x{image_size}x{image_size} images"
    )

    # Set random seed for reproducibility before model initialization
    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)

    # Initialize the score matching utilities with the geometric noise schedule
    score_matching = ScoreMatchingUNet(num_noise_levels, sigma_min, sigma_max, device)
    print(
        f"Noise schedule: {num_noise_levels} levels, "
        f"sigma from {sigma_min} to {sigma_max}"
    )

    # Initialize the score network
    model = ScoreUNet(
        image_channels, image_size, score_matching.sigmas, base_channels
    ).to(device)

    # Define loss function, optimizer, and learning rate
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    print(
        f"Training with Adam, number of epochs = {num_epochs}, "
        f"learning rate = {learning_rate}, batch size = {batch_size}, "
        f"noise levels = {num_noise_levels}, base channels = {base_channels}, "
        f"Langevin steps per level = {num_langevin_steps}, target SNR = {target_snr}"
    )
    num_gen_images = 36  # Number of images to generate for evaluation
    loss_log = train_smld(
        model,
        training_loader,
        num_epochs,
        criterion,
        optimizer,
        num_langevin_steps,
        target_snr,
        score_matching,
        device,
        num_gen_images,
    )
    print(f"Generated images saved to {pf.GEN_DIR}")

    # Plot training loss curve
    print("Plotting training loss curve")
    pf.plot_training_curve(
        loss_log, MODEL_NAME, y_label="denoising score matching loss"
    )

    # Show the generated images during predictor-corrector sampling
    print("Plotting generation progress")
    num_progress_rows = 8  # One row per trajectory
    num_progress_cols = 8  # One column per snapshot in time

    _, trajs = score_matching.predictor_corrector_sampling(
        model,
        num_progress_rows,
        num_langevin_steps,
        target_snr,
        return_traj=True,
    )

    pf.plot_generation_progress(
        trajs,
        title=f"{MODEL_NAME}: Generation Progress",
        file_name=pf.default_file_name(MODEL_NAME, "prog"),
        num_rows=num_progress_rows,
        num_cols=num_progress_cols,
        value_range=(-1.0, 1.0),
    )
