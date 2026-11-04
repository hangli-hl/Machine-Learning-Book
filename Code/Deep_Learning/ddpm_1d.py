##############################################################################
#        DENOISING DIFFUSION PROBABILISTIC MODEL (DDPM) FOR 1D DATA
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
MODEL_NAME = "DDPM-1D"
# Random seed for reproducibility
MODEL_SEED = 28
# Default number of training samples to generate
DEFAULT_NUM_SAMPLES = 2000
# Default number of training iterations
DEFAULT_NUM_ITERATIONS = 2000
# Default learning rate for Adam optimizer
DEFAULT_LEARNING_RATE = 0.001
# Default batch size for training
DEFAULT_BATCH_SIZE = 256
# Default number of hidden units in the UNet1D
DEFAULT_HIDDEN_SIZE = 128
# Default number of diffusion timesteps
DEFAULT_NUM_TIMESTEPS = 1000
# Default first value of the linear beta schedule
DEFAULT_BETA_START = 0.0001
# Default last value of the linear beta schedule
DEFAULT_BETA_END = 0.02
# Default number of generated samples to create
DEFAULT_NUM_GEN_SAMPLES = 3000

##############################################################################
#                        DIFFUSION PROCESS UTILITIES
##############################################################################


class Diffusion1D:
    """
    Utilities for the 1D DDPM (denoising diffusion probabilistic model)
    forward and reverse processes.

    Forward process (q_sample):
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * z

    Reverse process (p_sample / p_sample_loop):
        x_{t-1} = (x_t - beta_t / sqrt(1 - alpha_bar_t) * noise_theta)
                  / sqrt(alpha_t) + sqrt(beta_t) * z
    No noise is added at the last step (t = 0, deterministic):
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
            x_0: clean samples, shape (batch_size, 1)
            noise: Gaussian noise, shape (batch_size, 1)
            t: timesteps, shape (batch_size,)

        Returns:
            x_t: noisy samples at timestep t, shape (batch_size, 1)
        """
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(
            -1, 1
        )
        return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

    def p_sample(self, model, x, t):
        """
        Reverse process: sample x_{t-1} from x_t using the learned model.

        Args:
            model: trained noise-prediction network
            x: states at timestep t, shape (batch_size, 1)
            t: current timestep, an integer or scalar tensor

        Returns:
            x_prev: states at timestep t-1, shape (batch_size, 1)
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
    def p_sample_loop(self, model, num_gen_samples, return_traj=False):
        """
        Generate samples by denoising from pure noise over all timesteps.

        Args:
            model: trained noise-prediction network
            num_gen_samples: number of samples to generate, a positive integer
            return_traj: whether to also return the sampling trajectories, a boolean

        Returns:
            x: generated samples, shape (num_gen_samples, 1)
            trajs: stacked trajectories (only if return_traj),
                shape (num_timesteps + 1, num_gen_samples, 1)
        """
        model.eval()

        # Start from pure Gaussian noise and denoise step by step
        shape = (num_gen_samples, 1)
        x = torch.randn(shape, device=self.device)
        traj_points = [x.cpu().clone()] if return_traj else None

        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(model, x, t)
            if return_traj:
                traj_points.append(x.cpu().clone())

        if return_traj:
            trajs = torch.stack(traj_points)
            return x, trajs
        return x


##############################################################################
#                             TRAINING FUNCTION
##############################################################################


def train_ddpm(
    model,
    data,
    num_iterations,
    batch_size,
    criterion,
    optimizer,
    diffusion,
    device,
):
    """
    Train the denoising diffusion probabilistic model.

    Args:
        model: noise-prediction network (UNet1D)
        data: training data, shape (num_samples, 1)
        num_iterations: number of training iterations, a positive integer,
            one random mini-batch per iteration, no pass over the full dataset
        batch_size: batch size for training, a positive integer
        criterion: loss function (nn.MSELoss)
        optimizer: optimizer (Adam)
        diffusion: diffusion utilities with the noise schedule
        device: device to run computations on (CPU or GPU)

    Returns:
        loss_log: list of training losses for all iterations
    """
    num_samples = data.size(0)
    loss_log = []

    # Set the model to training mode
    model.train()

    for iteration in range(num_iterations):
        # Sample a random mini-batch of clean data every time
        # No iteration over the full dataset
        indices = torch.randint(0, num_samples, (batch_size,))
        x_0 = data[indices].to(device)

        # Sample random timesteps and predict corresponding noisy data
        timesteps = torch.randint(
            0, diffusion.num_timesteps, (batch_size,), device=device
        )
        noise = torch.randn_like(x_0)
        noisy_data = diffusion.q_sample(x_0, noise, timesteps)

        # Predict the noise and compute the MSE objective
        pred_noise = model(noisy_data, timesteps)
        loss = criterion(pred_noise, noise)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store training loss for plotting
        loss_log.append(loss.item())

        # Print progress
        if (iteration + 1) % 200 == 0:
            print(
                f"Iteration [{iteration + 1}/{num_iterations}], Loss: {loss.item():.6f}"
            )

    return loss_log


##############################################################################
#                           GENERATION FUNCTIONS
##############################################################################


@torch.no_grad()
def generate_forward_trajs(data, diffusion, num_trajs):
    """
    Generate forward diffusion trajectories for visualization.

    Args:
        data: clean data, shape (num_samples, 1)
        diffusion: diffusion utilities with the noise schedule
        num_trajs: number of trajectories to record, a positive integer

    Returns:
        trajs: stacked trajectories, shape (num_timesteps + 1, num_trajs, 1)
    """
    device = diffusion.device
    data = data.to(device)
    x = data[:num_trajs]
    traj_points = [x.cpu().clone()]

    for t in range(diffusion.num_timesteps):
        alpha_t = diffusion.alphas[t]
        beta_t = diffusion.betas[t]
        x = torch.sqrt(alpha_t) * x + torch.sqrt(beta_t) * torch.randn_like(x)
        traj_points.append(x.cpu().clone())

    trajs = torch.stack(traj_points)
    return trajs


def generate_reverse_trajs(model, diffusion, num_trajs):
    """
    Generate reverse diffusion trajectories for visualization.

    Args:
        model: trained noise-prediction network
        diffusion: diffusion utilities with the noise schedule
        num_trajs: number of trajectories to record, a positive integer

    Returns:
        trajs: stacked trajectories, shape (num_timesteps + 1, num_trajs, 1)
    """
    _, trajs = diffusion.p_sample_loop(model, num_trajs, return_traj=True)
    return trajs


##############################################################################
#                               MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Denoising Diffusion Probabilistic Model (DDPM) for "
        "1D Gaussian Mixture Model (GMM) data generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of training samples to generate",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=DEFAULT_NUM_ITERATIONS,
        help="Number of training iterations",
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
        "--hidden-size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help="Number of hidden units in the UNet1D",
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
    parser.add_argument(
        "--num-gen-samples",
        type=int,
        default=DEFAULT_NUM_GEN_SAMPLES,
        help="Number of generated samples to create",
    )
    args = parser.parse_args()
    print("Train a DDPM for 1D GMM data generation")

    # Set device for computing, CUDA, MPS, or CPU
    device = uf.set_device()
    print(f"Using device: {device}")

    # Set hyperparameters
    num_samples = args.num_samples
    num_iterations = args.num_iterations
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_timesteps = args.num_timesteps
    beta_start = args.beta_start
    beta_end = args.beta_end
    num_gen_samples = args.num_gen_samples

    # Create dataset
    print("Creating GMM data")
    data = dc.create_gaussian_mixture_1d_data(num_samples, device)
    print(f"Training set: {num_samples} samples")

    # Set random seed for reproducibility before model initialization
    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)

    # Initialize the noise-prediction model and diffusion utilities
    model = unet.UNet1D(hidden_size).to(device)
    diffusion = Diffusion1D(num_timesteps, beta_start, beta_end, device)

    # Define loss function, optimizer, and learning rate
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    print(
        f"Training with Adam, iterations = {num_iterations}, "
        f"learning rate = {learning_rate}, batch size = {batch_size}, "
        f"number of timesteps = {num_timesteps}, hidden size = {hidden_size}, "
        f"beta start = {beta_start}, beta end = {beta_end}"
    )
    loss_log = train_ddpm(
        model,
        data,
        num_iterations,
        batch_size,
        criterion,
        optimizer,
        diffusion,
        device,
    )

    # Plot training loss curve
    print("Plotting training loss curve")
    pf.plot_training_curve(loss_log, MODEL_NAME, y_label="MSE loss")

    # Number of trajectories to compute
    num_trajs = min(25, num_samples)

    # Plot the forward diffusion (noise-adding) process
    print("Plotting forward diffusion process")
    forward_trajs = generate_forward_trajs(data, diffusion, num_trajs)
    pf.plot_1d_trajs(
        forward_trajs,
        title=f"{MODEL_NAME} Forward: From Data to Noise",
        file_name=pf.default_file_name(MODEL_NAME, "traj_forward"),
        color="green",
        xlabel="time step",
        num_trajs=num_trajs,
    )

    # Plot the reverse diffusion (denoising) process
    print("Plotting reverse diffusion process")
    reverse_trajs = generate_reverse_trajs(model, diffusion, num_trajs)
    pf.plot_1d_trajs(
        reverse_trajs,
        title=f"{MODEL_NAME} Reverse: Noise to Data",
        file_name=pf.default_file_name(MODEL_NAME, "traj_reverse"),
        color="blue",
        xlabel="reverse time step",
        num_trajs=num_trajs,
    )

    # Sample from the learned model
    print("Sampling from learned model")
    samples = diffusion.p_sample_loop(model, num_gen_samples)
    samples = samples.detach().cpu()

    # Plot generated distribution and overlap with original data
    print("Plotting generated samples")
    pf.plot_1d_gen_samples(
        data,
        samples,
        title=f"{MODEL_NAME}: Original Data vs Generated Samples",
        file_name=pf.default_file_name(MODEL_NAME, "gen"),
        gmm_pdf=dc.gaussian_mixture_1d_pdf,
    )
