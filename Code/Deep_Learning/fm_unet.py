##############################################################################
#                FLOW MATCHING (FM) WITH U-SHAPED NETWORK (UNET)
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
MODEL_NAME = "FM-UNet"
# Random seed for reproducibility
MODEL_SEED = 28
# Supported ODE solvers: first order (euler), second order (midpoint, heun),
# fourth order (rk4)
SOLVERS = ("euler", "midpoint", "heun", "rk4")
# Default number of training epochs
DEFAULT_NUM_EPOCHS = 30
# Default learning rate for Adam optimizer
DEFAULT_LEARNING_RATE = 0.0001
# Default batch size for training
DEFAULT_BATCH_SIZE = 128
# Default number of channels at the full resolution of the UNet
DEFAULT_BASE_CHANNELS = 64
# Default number of ODE integration steps for sampling
DEFAULT_NUM_STEPS = 50
# Default ODE solver used for sampling
DEFAULT_SOLVER = "heun"
# Default minimum noise level of the OT conditional path
DEFAULT_SIGMA_MIN = 0.0
# Number of images to generate for evaluation
NUM_GENERATED_IMAGES = 36


##############################################################################
#                          FLOW MATCHING UTILITIES
##############################################################################


class FlowMatchingUNet:
    """
    Utilities for the flow matching (FM) training and generation
    with a UNet.

    Learning:
        x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
        u_t = x_1 - (1 - sigma_min) * x_0
        velocity field is v_theta(x_t, t)
        loss function is L = E || v_theta(x_t, t) - u_t ||^2

    Generation:
        x_{t+dt} = x_t + dt * v_theta(x_t, t)
        (Euler, or a higher-order Runge-Kutta update)
        Integrates with dt > 0 (t: 0 -> 1, noise -> images).

    - x_0: Gaussian noise (source state) at time t = 0
    - x_1: clean images (target state) at time t = 1
    - x_t: state on the conditional path at time t
    - u_t: velocity target of the conditional path
    - v_theta: the output of the trained network with parameters theta

    Attributes:
        num_steps: number of ODE integration steps, a positive integer
        sigma_min: noise scale left at t = 1, a float in [0, 1). At t = 1,
            x_t = x_1 + sigma_min * x_0.
        solver: ODE solver, one of {"euler", "midpoint", "heun", "rk4"}
        device: device to run computations on (CPU or GPU)
    """

    def __init__(self, num_steps, sigma_min, solver, device):
        """
        Initialize the flow matching utilities.

        Args:
            num_steps: number of ODE integration steps, a positive integer
            sigma_min: minimum noise level of the OT conditional path, a float
            solver: ODE solver, one of {"euler", "midpoint", "heun", "rk4"}
            device: device to run computations on (CPU or GPU)
        """
        if solver not in SOLVERS:
            raise ValueError(
                f"unknown solver {solver!r}, expected one of: {', '.join(SOLVERS)}"
            )

        # Optimal transport conditional path and ODE integration settings
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.solver = solver
        self.device = device

    def compute_location_and_velocity(self, x_1, t, x_0):
        """
        Compute the location and conditional velocity of a point
        on the conditional path.

        Args:
            x_1: clean images of shape (batch_size, image_channels, image_size, image_size)
            t: path times in [0, 1], shape (batch_size,)
            x_0: Gaussian noise endpoint of the same shape as x_1

        Returns:
            x_t: points on the conditional path at time t, same shape as x_1
            u_t: conditional velocity targets d x_t / dt, same shape as x_1
        """
        # x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
        noise_coeff_t = (1.0 - (1.0 - self.sigma_min) * t).reshape(-1, 1, 1, 1)
        data_coeff_t = t.reshape(-1, 1, 1, 1)
        x_t = noise_coeff_t * x_0 + data_coeff_t * x_1

        # u_t = x_1 - (1 - sigma_min) * x_0
        u_t = x_1 - (1.0 - self.sigma_min) * x_0

        return x_t, u_t

    def pred_velocity(self, model, x, t):
        """
        Predict the velocity field of a point x at time t using the learned
        network v_theta(x, t).

        Args:
            model: velocity-prediction network
            x: states at time t, shape (batch_size, image_channels,
                image_size, image_size)
            t: current time, a float or scalar tensor

        Returns:
            pred_velocity: predicted velocity, same shape as x
        """
        t_scalar = float(t)

        # Predict the velocity v_theta(x_t, t)
        t_batch = torch.full((x.size(0),), t_scalar, device=self.device)

        return model(x, t_batch)

    def ode_step(self, model, x, t, dt):
        """
        Advance the learned ODE dx/dt = v_theta(x, t) by one step
        with the selected solver.

        The direction is determined by the sign of dt: dt > 0 moves from noise
        toward data, dt < 0 moves from data toward noise.

        Args:
            model: velocity-prediction network
            x: states at time t, shape (batch_size, image_channels,
                image_size, image_size)
            t: current time, a float
            dt: signed step size, a float

        Returns:
            x_next: approximate states at time t + dt, same shape as x
        """
        # x_{t+dt} = x_t + dt * v_theta(x_t, t), one-stage (Euler) update
        if self.solver == "euler":
            k1 = self.pred_velocity(model, x, t)
            return x + dt * k1

        # Two-stage (midpoint) update
        if self.solver == "midpoint":
            k1 = self.pred_velocity(model, x, t)
            k2 = self.pred_velocity(model, x + 0.5 * dt * k1, t + 0.5 * dt)
            return x + dt * k2

        # Two-stage (Heun) update
        if self.solver == "heun":
            k1 = self.pred_velocity(model, x, t)
            k2 = self.pred_velocity(model, x + dt * k1, t + dt)
            return x + dt * 0.5 * (k1 + k2)

        # Default: fourth-order Runge-Kutta (rk4) update
        k1 = self.pred_velocity(model, x, t)
        k2 = self.pred_velocity(model, x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.pred_velocity(model, x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.pred_velocity(model, x + dt * k3, t + dt)
        return x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    @torch.no_grad()
    def generate_ode(self, model, num_gen_images, return_traj=False):
        """
        Generation: draw x_0 ~ N(0, I) and advance the learned ODE from
        t = 0 to t = 1 (noise -> data) with num_steps steps of size 1/num_steps.

        Args:
            model: trained velocity-prediction network
            num_gen_images: number of images to generate, a positive integer
            return_traj: whether to also return the trajectories, a boolean

        Returns:
            x: generated images in [-1, 1],
                shape (num_gen_images, image_channels, image_size, image_size)
            trajs: states at all steps (only if return_traj), shape
                (num_steps + 1, num_gen_images, image_channels,
                image_size, image_size); trajs[0] is the initial noise and
                trajs[-1] the final clamped images
        """
        model.eval()

        num_steps = self.num_steps
        dt = 1.0 / num_steps

        # Start from pure Gaussian noise and integrate step by step
        shape = (
            num_gen_images,
            model.image_channels,
            model.image_size,
            model.image_size,
        )
        x = torch.randn(shape, device=self.device)
        traj_points = [x.cpu().clone()] if return_traj else None

        for i in range(num_steps):
            t = i * dt
            x = self.ode_step(model, x, t, dt)
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


def train_fm(
    model,
    training_loader,
    num_epochs,
    criterion,
    optimizer,
    flow_matching,
    device,
    num_gen_images,
):
    """
    Train the flow matching model.

    Args:
        model: velocity-prediction network (UNet)
        training_loader: DataLoader for training data
        num_epochs: number of training epochs, a positive integer
        criterion: loss function (nn.MSELoss)
        optimizer: optimizer (Adam)
        flow_matching: flow matching utilities with the conditional path
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

            # Sample noise and random path times, then run the flow matching process
            noise = torch.randn_like(images)
            t = torch.rand(images.size(0), device=device)
            x_t, u_t = flow_matching.compute_location_and_velocity(images, t, noise)

            # Predict the velocity and compute the MSE objective
            pred_velocity = model(x_t, t)
            loss = criterion(pred_velocity, u_t)

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
        generate_samples_fm(model, flow_matching, num_gen_images, epoch, num_epochs)

    return loss_log


##############################################################################
#                             GENERATION FUNCTION
##############################################################################


def generate_samples_fm(model, flow_matching, num_gen_images, epoch, num_epochs):
    """
    Generate sample images using the learned model, i.e., the velocity-prediction network.

    Args:
        model: trained velocity-prediction network
        flow_matching: flow matching utilities with the conditional path
        num_gen_images: number of images to generate, a positive integer
        epoch: current epoch index, a non-negative integer
        num_epochs: total number of epochs, a positive integer
    """
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Images are in the tanh range [-1, 1], which will be re-scaled
        gen_images = flow_matching.generate_ode(model, num_gen_images)

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
        description="Train a Flow Matching (FM) model with UNet "
        "for MNIST image generation",
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
        "--num-steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help="Number of ODE integration steps for sampling",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default=DEFAULT_SOLVER,
        choices=SOLVERS,
        help="ODE solver used for sampling",
    )
    parser.add_argument(
        "--sigma-min",
        type=float,
        default=DEFAULT_SIGMA_MIN,
        help="Minimum noise level of the OT conditional path",
    )
    args = parser.parse_args()
    print("Train an FM model with UNet for MNIST image generation")

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
    num_steps = args.num_steps
    solver = args.solver
    sigma_min = args.sigma_min

    # Load dataset
    training_loader = dl.load_mnist_generation_data(batch_size, tanh_range=True)
    print(
        f"Training set: {len(training_loader.dataset)} samples, "
        f"{image_channels}x{image_size}x{image_size} images"
    )

    # Set random seed for reproducibility before model initialization
    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)

    # Initialize the velocity-prediction model and flow matching utilities
    # Continuous times in [0, 1] require condition_scale
    model = unet.UNet(
        image_channels,
        image_size,
        base_channels,
        condition_scale=1000.0,
    ).to(device)
    flow_matching = FlowMatchingUNet(num_steps, sigma_min, solver, device)

    # Define loss function, optimizer, and learning rate
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    print(
        f"Training with Adam, number of epochs = {num_epochs}, "
        f"learning rate = {learning_rate}, batch size = {batch_size}, "
        f"base channels = {base_channels}, sigma min = {sigma_min}, "
        f"solver = {solver}, steps = {num_steps}"
    )
    num_gen_images = NUM_GENERATED_IMAGES  # Number of images to generate for evaluation
    loss_log = train_fm(
        model,
        training_loader,
        num_epochs,
        criterion,
        optimizer,
        flow_matching,
        device,
        num_gen_images,
    )
    print(f"Generated images saved to {pf.GEN_DIR}")

    # Plot training loss curve
    print("Plotting training loss curve")
    pf.plot_training_curve(loss_log, MODEL_NAME, y_label="MSE loss")

    # Show the generated images during the flow matching process
    print("Plotting generation progress")
    num_progress_rows = 8  # One row per trajectory
    num_progress_cols = 8  # One column per snapshot in time

    _, trajs = flow_matching.generate_ode(model, num_progress_rows, return_traj=True)

    pf.plot_generation_progress(
        trajs,
        title=f"{MODEL_NAME}: Generation Progress",
        file_name=pf.default_file_name(MODEL_NAME, "prog"),
        num_rows=num_progress_rows,
        num_cols=num_progress_cols,
        value_range=(-1.0, 1.0),
    )
