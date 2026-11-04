##############################################################################
#  DENOISING DIFFUSION PROBABILISTIC MODEL (DDPM) WITH U-SHAPED NETWORK (UNET)
##############################################################################


import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import data_creators as dc
import utility_functions as uf
import plot_functions as pf
import unet

# Name used in plot titles and file names
MODEL_NAME = "DDPM-UNet"
# Random seed for reproducibility
MODEL_SEED = 28
# Default number of training epochs
DEFAULT_NUM_EPOCHS = 30
# Default learning rate for Adam optimizer
DEFAULT_LEARNING_RATE = 0.0001
# Default batch size for training
DEFAULT_BATCH_SIZE = 128
# Default number of channels at the full resolution of the UNet
DEFAULT_BASE_CHANNELS = 64
# Default number of diffusion timesteps
DEFAULT_NUM_TIMESTEPS = 1000
# Default first value of the linear beta schedule
DEFAULT_BETA_START = 0.0001
# Default last value of the linear beta schedule
DEFAULT_BETA_END = 0.02
# Number of images to generate for evaluation
NUM_GENERATED_IMAGES = 36


##############################################################################
#                        DIFFUSION PROCESS UTILITIES
##############################################################################


class DiffusionUNet:
    """
    Utilities for the DDPM (denoising diffusion probabilistic model)
    forward and reverse processes with UNet.

    Indexing: the formulas use the 1-based steps t = 1, ..., T of the DDPM
    paper, with x_0 the clean data. The code is 0-based: array index k = t - 1,
    so betas[k] is beta_t, and the `t` arguments of q_sample / p_sample are
    these 0-based indices k.

    Forward process (q_sample):
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * z

    Reverse process (p_sample / p_sample_loop):
        x_{t-1} = (x_t - beta_t / sqrt(1 - alpha_bar_t) * noise_theta)
                  / sqrt(alpha_t) + sqrt(beta_t) * z
    No noise is added at the last step (t = 1, code index 0, deterministic):
        x_0 = (x_1 - beta_1 / sqrt(1 - alpha_bar_1) * noise_theta) / sqrt(alpha_1)

    - x_t: state at timestep t
    - x_{t-1}: state at timestep t-1
    - x_0: clean data (state) at timestep 0
    - z: Gaussian noise
    - noise_theta: the output of the trained network with parameters theta

    Attributes:
        num_timesteps: number of diffusion timesteps, a positive integer
        betas: linear variance schedule, shape (num_timesteps,)
        alphas: 1 - betas, shape (num_timesteps,)
        alphas_cumprod: cumulative product of alphas, shape (num_timesteps,)
        sqrt_alphas_cumprod: square root of alphas_cumprod, shape (num_timesteps,)
        sqrt_one_minus_alphas_cumprod: square root of (1 - alphas_cumprod), shape (num_timesteps,)
        device: device to run computations on (CPU or GPU)
    """

    def __init__(self, num_timesteps, beta_start, beta_end, device):
        """
        Initialize the diffusion utilities.

        Args:
            num_timesteps: number of diffusion timesteps, a positive integer
            beta_start: first value of the linear beta schedule, a positive float
            beta_end: last value of the linear beta schedule, a positive float
            device: device to run computations on (CPU or GPU)
        """
        self.num_timesteps = num_timesteps
        self.device = device

        # Linear variance schedule and derived quantities
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, noise, t):
        """
        Forward process: sample x_t from q(x_t | x_0).

        Args:
            x_0: clean images of shape (batch_size, image_channels, image_size, image_size)
            noise: Gaussian noise of the same shape as x_0
            t: 0-based diffusion timestep indices, shape (batch_size,)

        Returns:
            x_t: noisy images at timestep t, same shape as x_0
        """
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(
            -1, 1, 1, 1
        )
        return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

    def p_sample(self, model, x, t):
        """
        Reverse process: sample x_{t-1} from x_t using the learned model.

        Args:
            model: trained noise-prediction network
            x: states at timestep t, shape (batch_size, image_channels,
                image_size, image_size)
            t: current 0-based timestep index, an integer or scalar tensor

        Returns:
            x_prev: states at timestep t-1, shape (batch_size, image_channels,
                image_size, image_size)
        """
        t_scalar = int(t)

        beta_t = self.betas[t_scalar]
        alpha_t = self.alphas[t_scalar]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t_scalar]

        # Predict the noise from the noise-prediction network
        t_batch = torch.full(
            (x.size(0),), t_scalar, dtype=torch.long, device=self.device
        )
        pred_noise = model(x, t_batch)

        # Compute the previous state from the predicted noise
        # x_{t-1} = (x_t - beta_t / sqrt(1 - alpha_bar_t) * pred_noise) / sqrt(alpha_t)
        #           + sqrt(beta_t) * z
        mean = (x - beta_t / sqrt_one_minus_alpha_bar_t * pred_noise) / torch.sqrt(
            alpha_t
        )
        if t_scalar > 0:
            return mean + torch.sqrt(beta_t) * torch.randn_like(x)
        return mean

    @torch.no_grad()
    def p_sample_loop(self, model, num_gen_images, return_traj=False):
        """
        Generate images by denoising from pure noise over all timesteps.

        Args:
            model: trained noise-prediction network
            num_gen_images: number of images to generate, a positive integer
            return_traj: whether to also return the sampling trajectories, a boolean

        Returns:
            x: generated images in [-1, 1],
                shape (num_gen_images, image_channels, image_size, image_size)
            trajs: stacked trajectories (only if return_traj), shape
                (num_timesteps + 1, num_gen_images, image_channels,
                image_size, image_size); trajs[0] is the initial noise and
                trajs[-1] the final clamped images
        """
        model.eval()

        # Start from pure Gaussian noise and denoise step by step
        shape = (
            num_gen_images,
            model.image_channels,
            model.image_size,
            model.image_size,
        )
        x = torch.randn(shape, device=self.device)
        traj_points = [x.cpu().clone()] if return_traj else None

        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(model, x, t)
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


def train_ddpm(
    model,
    training_loader,
    num_epochs,
    criterion,
    optimizer,
    diffusion,
    device,
    num_gen_images,
):
    """
    Train the denoising diffusion probabilistic model.

    Args:
        model: noise-prediction network (UNet)
        training_loader: DataLoader for training data
        num_epochs: number of training epochs, a positive integer
        criterion: loss function (nn.MSELoss)
        optimizer: optimizer (Adam)
        diffusion: diffusion utilities with the noise schedule
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

            # Sample noise and random timesteps, then compute the noisy images
            noise = torch.randn_like(images)
            t = torch.randint(
                0, diffusion.num_timesteps, (images.size(0),), device=device
            )
            x_t = diffusion.q_sample(images, noise, t)

            # Predict the noise and compute the MSE objective
            pred_noise = model(x_t, t)
            loss = criterion(pred_noise, noise)

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
        generate_samples_ddpm(model, diffusion, num_gen_images, epoch, num_epochs)

    return loss_log


##############################################################################
#                             GENERATION FUNCTION
##############################################################################


def generate_samples_ddpm(model, diffusion, num_gen_images, epoch, num_epochs):
    """
    Generate sample images using the learned model, i.e., noise-prediction network.

    Args:
        model: trained noise-prediction network
        diffusion: diffusion utilities with the noise schedule
        num_gen_images: number of images to generate, a positive integer
        epoch: current epoch index, a non-negative integer
        num_epochs: total number of epochs, a positive integer
    """
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Images are in the tanh range [-1, 1], which will be re-scaled
        gen_images = diffusion.p_sample_loop(model, num_gen_images)

    # Set the model back to training mode
    model.train()

    # Only the generated images from the last epoch are displayed
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
        description="Train a Denoising Diffusion Probabilistic Model (DDPM) "
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
        "--num-timesteps",
        type=int,
        default=DEFAULT_NUM_TIMESTEPS,
        help="Number of diffusion timesteps",
    )
    parser.add_argument(
        "--beta-start",
        type=float,
        default=DEFAULT_BETA_START,
        help="First value of the linear beta schedule",
    )
    parser.add_argument(
        "--beta-end",
        type=float,
        default=DEFAULT_BETA_END,
        help="Last value of the linear beta schedule",
    )
    args = parser.parse_args()
    print("Train a DDPM with UNet for MNIST image generation")

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
    num_timesteps = args.num_timesteps
    beta_start = args.beta_start
    beta_end = args.beta_end

    # Load dataset
    training_loader = dc.load_mnist_generation_data(batch_size, tanh_range=True)
    print(
        f"Training set: {len(training_loader.dataset)} samples, "
        f"{image_channels}x{image_size}x{image_size} images"
    )

    # Set random seed for reproducibility before model initialization
    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)

    # Initialize the noise-prediction model and diffusion utilities
    model = unet.UNet(image_channels, image_size, base_channels).to(device)
    diffusion = DiffusionUNet(num_timesteps, beta_start, beta_end, device)

    # Define loss function, optimizer, and learning rate
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    print(
        f"Training with Adam, number of epochs = {num_epochs}, "
        f"learning rate = {learning_rate}, batch size = {batch_size}, "
        f"number of timesteps = {num_timesteps}, base channels = {base_channels}, "
        f"beta start = {beta_start}, beta end = {beta_end}"
    )
    num_gen_images = NUM_GENERATED_IMAGES  # Number of images to generate for evaluation
    loss_log = train_ddpm(
        model,
        training_loader,
        num_epochs,
        criterion,
        optimizer,
        diffusion,
        device,
        num_gen_images,
    )
    print(f"Generated images saved to {pf.GEN_DIR}")

    # Plot training loss curve
    print("Plotting training loss curve")
    pf.plot_training_curve(loss_log, MODEL_NAME, y_label="MSE loss")

    # Show the generated images during the reverse diffusion process
    print("Plotting generation progress")
    num_progress_rows = 8  # One row per trajectory
    num_progress_cols = 8  # One column per snapshot in time

    _, trajs = diffusion.p_sample_loop(model, num_progress_rows, return_traj=True)

    pf.plot_generation_progress(
        trajs,
        title=f"{MODEL_NAME}: Generation Progress",
        file_name=pf.default_file_name(MODEL_NAME, "prog"),
        num_rows=num_progress_rows,
        num_cols=num_progress_cols,
        value_range=(-1.0, 1.0),
    )
