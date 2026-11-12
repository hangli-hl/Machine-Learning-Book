##############################################################################
#          SCORE MATCHING WITH LANGEVIN DYNAMICS (SMLD) FOR 1D DATA
##############################################################################


import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

# Make the repository root importable so that the Data package can be found
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Data import data_loaders as dl
import utility_functions as uf
import plot_functions as pf
import unet

# Name used in plot titles and file names
MODEL_NAME = "SMLD-1D"
# Random seed for reproducibility
MODEL_SEED = 28
# A small constant to avoid division by zero
EPS = 1e-12
# Default number of training samples
DEFAULT_NUM_SAMPLES = 2000
# Default number of training iterations
DEFAULT_NUM_ITERATIONS = 5000
# Default learning rate for Adam optimizer
DEFAULT_LEARNING_RATE = 0.001
# Default batch size for training
DEFAULT_BATCH_SIZE = 256
# Default number of hidden units in the UNet1D
DEFAULT_HIDDEN_SIZE = 128
# Default number of noise levels in the geometric schedule
DEFAULT_NUM_NOISE_LEVELS = 10
# Default smallest noise level
DEFAULT_SIGMA_MIN = 0.01
# Default largest noise level
DEFAULT_SIGMA_MAX = 10.0
# Default number of Langevin corrector steps per noise level
DEFAULT_NUM_LANGEVIN_STEPS = 10
# Default target signal-to-noise ratio for the Langevin corrector step size
DEFAULT_TARGET_SNR = 0.10
# Default number of samples to generate
DEFAULT_NUM_GEN_SAMPLES = 3000

##############################################################################
#                           SCORE NETWORK CLASS
##############################################################################


class ScoreUNet1D(nn.Module):
    """
    Noise score network for 1D data.

    Wraps a UNet1D regression network and divides its output by the noise
    level sigma
        s_theta(x, i) = net(x, i) / sigma_i
    This keeps the target of the network at unit scale across noise levels.

    Attributes:
        sigmas: geometric noise schedule, shape (num_noise_levels,), ascending
        unet: UNet1D producing unscaled predicted fields
    """

    def __init__(self, sigmas, hidden_size):
        """
        Initialize the score network.

        Args:
            sigmas: geometric noise schedule of shape (num_noise_levels,), ascending
            hidden_size: number of hidden units in the underlying UNet1D
        """
        super().__init__()
        self.register_buffer("sigmas", sigmas)
        self.unet = unet.UNet1D(hidden_size)

    def forward(self, x, noise_level):
        """
        Compute the score at the given noise level.

        Args:
            x: state of shape (batch_size, 1)
            noise_level: noise level index of shape (batch_size,)

        Returns:
            score: score estimate of shape (batch_size, 1)
        """
        # Predicted score scaled by the corresponding noise level
        score = self.unet(x, noise_level)
        sigma = self.sigmas[noise_level].reshape(-1, 1)
        return score / sigma


##############################################################################
#             SCORE MATCHING WITH LANGEVIN DYNAMICS UTILITIES
##############################################################################


class ScoreMatching1D:
    """
    Utilities for 1D SMLD (score matching with Langevin dynamics), including
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
        Forward process: perturb clean samples with Gaussian noise.
            x_tilde = x_0 + sigma_i * epsilon

        Args:
            x_0: clean data, shape (batch_size, 1)
            noise: Gaussian noise, shape (batch_size, 1)
            noise_level: noise level index, shape (batch_size,)

        Returns:
            x_tilde: perturbed data, shape (batch_size, 1)
        """
        sigma = self.sigmas[noise_level].reshape(-1, 1)
        return x_0 + sigma * noise

    def langevin_step(self, model, x, noise_level, target_snr):
        """
        Corrector: one Langevin step with an adaptive step size.
            s = s_theta(x, sigma_i)
            step = 2 * (target_snr * ||z|| / ||s||) ** 2
            x <- x + step * s + sqrt(2 * step) * z,   z ~ N(0, I)

        Args:
            model: trained score network
            x: current states, shape (batch_size, 1)
            noise_level: noise level index, a non-negative integer
            target_snr: target signal-to-noise ratio for the step size, a positive float

        Returns:
            x_next: updated states, shape (batch_size, 1)
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
        num_gen_samples,
        num_langevin_steps,
        target_snr,
        return_traj=False,
    ):
        """
        Generate data with the predictor-corrector sampler: each predictor step
        is followed by num_langevin_steps corrector steps at the noise level
        the predictor moved to.

        Args:
            model: trained score network
            num_gen_samples: number of samples to generate, a positive integer
            num_langevin_steps: Langevin corrector steps per noise level, a positive integer
            target_snr: target signal-to-noise ratio for the corrector step size, a positive float
            return_traj: whether to also return the sampling trajectories, a boolean

        Returns:
            x: generated samples, shape (num_gen_samples, 1)
            trajs: stacked trajectories (only if return_traj), shape
                (1 + num_noise_levels + (num_noise_levels - 1) * num_langevin_steps,
                num_gen_samples, 1)
        """
        model.eval()

        # Start from noise scaled to the largest noise level
        shape = (num_gen_samples, 1)
        x = torch.randn(shape, device=self.device) * self.sigmas[-1]
        traj_points = [x.cpu().clone()] if return_traj else None

        for level in reversed(range(self.num_noise_levels)):
            # Predictor: reverse-diffusion step from sigma_level to sigma_{level-1}
            noise_level = torch.full(
                (num_gen_samples,), level, device=self.device, dtype=torch.long
            )
            sigma_current = self.sigmas[level]
            sigma_next = (
                self.sigmas[level - 1] if level > 0 else torch.zeros_like(sigma_current)
            )
            variance = sigma_current**2 - sigma_next**2

            x = x + variance * model(x, noise_level)
            if level > 0:
                x = x + torch.sqrt(variance) * torch.randn_like(x)

            if return_traj:
                traj_points.append(x.cpu().clone())

            # Corrector: Langevin steps at the level the predictor just moved to.
            # Skipped after the level-0 predictor, which is the deterministic
            # denoising step to sigma = 0.
            if level > 0:
                for _ in range(num_langevin_steps):
                    x = self.langevin_step(model, x, level - 1, target_snr)
                    if return_traj:
                        traj_points.append(x.cpu().clone())

        if return_traj:
            trajs = torch.stack(traj_points)
            return x, trajs
        return x


##############################################################################
#                             TRAINING FUNCTION
##############################################################################


def train_smld(
    model,
    data,
    num_iterations,
    batch_size,
    criterion,
    optimizer,
    score_matching,
    device,
):
    """
    Train the score network with denoising score matching.

    Args:
        model: score network (ScoreUNet1D)
        data: training data, shape (num_samples, 1)
        num_iterations: number of training iterations, a positive integer,
            one random mini-batch per iteration, no pass over the full dataset
        batch_size: batch size for training, a positive integer
        criterion: loss function (nn.MSELoss)
        optimizer: optimizer (Adam)
        score_matching: score matching utilities with the noise schedule
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

        # Sample noise and noise levels, then run the forward process
        noise = torch.randn_like(x_0)
        t = torch.randint(
            0, score_matching.num_noise_levels, (x_0.size(0),), device=device
        )
        x_tilde = score_matching.add_noise(x_0, noise, t)

        # Calculate the denoising score matching loss
        # scaled_scores = sigma * s_theta = raw network output; target is -noise
        scores = model(x_tilde, t)
        scaled_scores = scores * score_matching.sigmas[t].reshape(-1, 1)
        loss = criterion(scaled_scores, -noise)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store training loss for plotting
        loss_log.append(loss.item())

        # Print progress
        if (iteration + 1) % 500 == 0:
            print(
                f"Iteration [{iteration + 1}/{num_iterations}], Loss: {loss.item():.4f}"
            )

    return loss_log


##############################################################################
#                           GENERATION FUNCTIONS
##############################################################################


@torch.no_grad()
def generate_forward_trajs(data, score_matching, num_trajs):
    """
    Generate forward trajectories (noise-adding) for visualization.

    Args:
        data: clean data, shape (num_samples, 1)
        score_matching: score matching utilities with the noise schedule
        num_trajs: number of trajectories to record, a positive integer

    Returns:
        trajs: stacked trajectories, shape (num_noise_levels + 1, num_trajs, 1)
    """
    device = score_matching.device
    data = data.to(device)
    x = data[:num_trajs]
    traj_points = [x.cpu().clone()]

    # Add noise incrementally following the ascending noise schedule
    sigma_prev = torch.tensor(0.0, device=device)
    for sigma in score_matching.sigmas:
        increment = torch.sqrt(torch.clamp(sigma**2 - sigma_prev**2, min=0.0))
        x = x + increment * torch.randn_like(x)
        traj_points.append(x.cpu().clone())
        sigma_prev = sigma

    trajs = torch.stack(traj_points)
    return trajs


@torch.no_grad()
def generate_reverse_trajs(
    model, score_matching, num_trajs, num_langevin_steps, target_snr
):
    """
    Generate reverse trajectories (predictor-corrector sampling) for visualization.

    Args:
        model: trained score network
        score_matching: score matching utilities with the noise schedule
        num_trajs: number of trajectories to record, a positive integer
        num_langevin_steps: Langevin corrector steps per noise level, a positive integer
        target_snr: target signal-to-noise ratio for the corrector step size, a positive float

    Returns:
        trajs: stacked trajectories, shape
            (1 + num_noise_levels + (num_noise_levels - 1) * num_langevin_steps,
            num_trajs, 1)
    """
    _, trajs = score_matching.predictor_corrector_sampling(
        model,
        num_trajs,
        num_langevin_steps,
        target_snr,
        return_traj=True,
    )
    return trajs


##############################################################################
#                            SCORE FUNCTION
##############################################################################


@torch.no_grad()
def compute_score_function(model, grid, noise_level):
    """
    Compute the learned score on a grid at a fixed noise level.

    Args:
        model: trained score network
        grid: states at which the score is evaluated, shape (num_points, 1)
        noise_level: noise level index, a non-negative integer

    Returns:
        scores: learned scores, numpy array of shape (num_points,)
    """
    model.eval()

    levels = torch.full(
        (grid.size(0),), noise_level, dtype=torch.long, device=grid.device
    )
    scores = model(grid, levels).cpu().numpy().reshape(-1)

    model.train()

    return scores


##############################################################################
#                               MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Score Matching with Langevin Dynamics (SMLD) model "
        "for 1D Gaussian Mixture Model (GMM) data generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of training samples",
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
    parser.add_argument(
        "--num-gen-samples",
        type=int,
        default=DEFAULT_NUM_GEN_SAMPLES,
        help="Number of samples to generate",
    )
    args = parser.parse_args()
    print("Train an SMLD model for 1D GMM data generation")

    # Set device for computing, CUDA, MPS, or CPU
    device = uf.set_device()
    print(f"Using device: {device}")

    # Set hyperparameters
    num_samples = args.num_samples
    num_iterations = args.num_iterations
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_noise_levels = args.num_noise_levels
    sigma_min = args.sigma_min
    sigma_max = args.sigma_max
    num_langevin_steps = args.num_langevin_steps
    target_snr = args.target_snr
    num_gen_samples = args.num_gen_samples

    # Create dataset
    print("Loading GMM data")
    data = dl.load_gaussian_mixture_1d_data(num_samples, device)
    print(f"Training set: {num_samples} samples")

    # Set random seed for reproducibility before model initialization
    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)

    # Initialize the score matching utilities with the geometric noise schedule
    score_matching = ScoreMatching1D(num_noise_levels, sigma_min, sigma_max, device)
    print(
        f"Noise schedule: {num_noise_levels} levels, "
        f"sigma from {sigma_min} to {sigma_max}"
    )

    # Initialize the score network
    model = ScoreUNet1D(score_matching.sigmas, hidden_size).to(device)

    # Define loss function, optimizer, and learning rate
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    print(
        f"Training with Adam, iterations = {num_iterations}, "
        f"learning rate = {learning_rate}, batch size = {batch_size}, "
        f"noise levels = {num_noise_levels}, hidden size = {hidden_size}"
    )
    loss_log = train_smld(
        model,
        data,
        num_iterations,
        batch_size,
        criterion,
        optimizer,
        score_matching,
        device,
    )

    # Plot training loss curve
    print("Plotting training loss curve")
    pf.plot_training_curve(
        loss_log, MODEL_NAME, y_label="denoising score matching loss"
    )

    # Number of trajectories to compute
    num_trajs = min(25, num_samples)

    # Plot the forward diffusion (noise-adding) process
    print("Plotting forward noise-adding process")
    forward_trajs = generate_forward_trajs(data, score_matching, num_trajs)
    pf.plot_1d_trajs(
        forward_trajs,
        title=f"{MODEL_NAME} Forward: From Data to Noise",
        file_name=pf.default_file_name(MODEL_NAME, "traj_forward"),
        color="green",
        xlabel="noise level (sigma min to sigma max)",
        num_trajs=num_trajs,
    )

    # Plot the reverse diffusion (denoising) process
    print("Plotting reverse sampling process")
    reverse_trajs = generate_reverse_trajs(
        model, score_matching, num_trajs, num_langevin_steps, target_snr
    )
    pf.plot_1d_trajs(
        reverse_trajs,
        title=f"{MODEL_NAME} Reverse: Noise to Data",
        file_name=pf.default_file_name(MODEL_NAME, "traj_reverse"),
        color="blue",
        xlabel="predictor-corrector step (sigma max to sigma min)",
        num_trajs=num_trajs,
    )

    # Learned score vs analytic score
    # The score at the smallest noise level approximates the score of the clean data
    print("Plotting learned score function")
    grid = torch.linspace(-10.0, 10.0, 400, device=device).reshape(-1, 1)
    predicted = compute_score_function(model, grid, noise_level=0)
    true = dc.gaussian_mixture_1d_score(grid).detach().cpu().numpy().ravel()
    pf.plot_1d_score_function(
        grid.detach().cpu().numpy().ravel(),
        true,
        predicted,
        title=f"{MODEL_NAME}: Learned Score vs True Score",
        file_name=pf.default_file_name(MODEL_NAME, "score_function"),
        sigma=score_matching.sigmas[0].item(),
    )

    # Sample with the predictor-corrector sampler
    print("Sampling with predictor-corrector sampler")
    samples = score_matching.predictor_corrector_sampling(
        model,
        num_gen_samples,
        num_langevin_steps,
        target_snr,
    )
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
