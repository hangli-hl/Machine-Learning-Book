##############################################################################
#                       FLOW MATCHING (FM) FOR 1D DATA
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
MODEL_NAME = "FM-1D"
# Random seed for reproducibility
MODEL_SEED = 28
# Supported ODE solvers: first order (euler), second order (midpoint, heun),
# fourth order (rk4)
SOLVERS = ("euler", "midpoint", "heun", "rk4")
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
# Default number of ODE integration steps for sampling
DEFAULT_NUM_STEPS = 100
# Default ODE solver used for sampling
DEFAULT_SOLVER = "rk4"
# Default minimum noise level of the OT conditional path
DEFAULT_SIGMA_MIN = 0.0
# Default number of samples to generate
DEFAULT_NUM_GEN_SAMPLES = 3000

##############################################################################
#                          FLOW MATCHING UTILITIES
##############################################################################


class FlowMatching1D:
    """
    Utilities for 1D flow matching (FM) training and generation.

    Time convention: t = 0 is the Gaussian noise, t = 1 is the data.

    Learning:
        Construction of the conditional path that
        connects a noise sample x_0 to a data sample x_1:
            x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
            u_t = d x_t / dt = x_1 - (1 - sigma_min) * x_0
        It is used to build training pairs (x_t, u_t). No network call
        and no ODE solving are involved.

        The network is trained by regressing the conditional velocity u_t:
            L = E_{t, x_0, x_1} || v_theta(x_t, t) - u_t ||^2
        The minimizer of this loss is the marginal velocity field
            u_t(x)
        which transports the entire noise distribution to the entire data distribution.

    Generation:
        Numerical integration of the ODE defined by the learned velocity field
            dx/dt = v_theta(x, t).
        The direction is set by the sign of the step size dt:
        - dt > 0 (t: 0 -> 1): generation, noise -> data (generate_ode)
        - dt < 0 (t: 1 -> 0): inversion, data -> noise (invert_ode)

    Symbols:
        x_0: Gaussian noise (source) at t = 0
        x_1: clean data (target) at t = 1
        x_t: point on the conditional path at time t
        u_t: conditional velocity target of the path (constant in t for a
            fixed pair (x_0, x_1))
        v_theta: velocity predicted by the network with parameters theta

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
            x_1: clean data samples, shape (batch_size, 1)
            t: path times in [0, 1], shape (batch_size,)
            x_0: Gaussian noise samples, shape (batch_size, 1)

        Returns:
            x_t: points on the conditional path at time t, shape (batch_size, 1)
            u_t: conditional velocity targets d x_t / dt, shape (batch_size, 1)
        """
        # x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
        noise_coeff_t = (1.0 - (1.0 - self.sigma_min) * t).reshape(-1, 1)
        data_coeff_t = t.reshape(-1, 1)
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
            x: states at time t, shape (batch_size, 1)
            t: current time, a float or scalar tensor

        Returns:
            pred_velocity: predicted velocity, shape (batch_size, 1)
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
            x: states at time t, shape (batch_size, 1)
            t: current time, a float
            dt: signed step size, a float

        Returns:
            x_next: approximate states at time t + dt, shape (batch_size, 1)
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
    def generate_ode(self, model, num_gen_samples, return_traj=False):
        """
        Generation: draw x_0 ~ N(0, I) and advance the learned ODE from
        t = 0 to t = 1 (noise -> data) with num_steps steps of size 1/num_steps.

        Args:
            model: trained velocity-prediction network
            num_gen_samples: number of samples to generate, a positive integer
            return_traj: whether to also return the trajectories, a boolean

        Returns:
            x: generated samples at t = 1, shape (num_gen_samples, 1)
            trajs: states at all steps (only if return_traj),
                shape (num_steps + 1, num_gen_samples, 1)
        """
        model.eval()

        num_steps = self.num_steps
        dt = 1.0 / num_steps

        # Start from pure Gaussian noise and integrate step by step
        shape = (num_gen_samples, 1)
        x = torch.randn(shape, device=self.device)
        traj_points = [x.cpu().clone()] if return_traj else None

        for i in range(num_steps):
            t = i * dt
            x = self.ode_step(model, x, t, dt)
            if return_traj:
                traj_points.append(x.cpu().clone())

        if return_traj:
            trajs = torch.stack(traj_points)
            return x, trajs
        return x

    @torch.no_grad()
    def invert_ode(self, model, x_1, return_traj=False):
        """
        Advance the same learned ODE backward in time, from
        t = 1 to t = 0 (data -> noise), with step size -1/num_steps.

        This does not generate new data. It maps the given data x_1 to
        its corresponding noise at t = 0, which is approximately N(0, I)
        if the model is well trained.

        Args:
            model: trained velocity-prediction network
            x_1: data used as the initial condition at t = 1, shape (batch_size, 1)
            return_traj: whether to also return the trajectories, a boolean

        Returns:
            x: inverted latent states at t = 0, shape (batch_size, 1)
            trajs: states at all steps (only if return_traj),
                shape (num_steps + 1, batch_size, 1)
        """
        model.eval()

        num_steps = self.num_steps

        # Negative step size: integrate backwards in time from t = 1 to t = 0
        dt = -1.0 / num_steps

        x = x_1.to(self.device)
        traj_points = [x.cpu().clone()] if return_traj else None

        for i in range(num_steps):
            t = 1.0 + i * dt
            x = self.ode_step(model, x, t, dt)
            if return_traj:
                traj_points.append(x.cpu().clone())

        if return_traj:
            trajs = torch.stack(traj_points)
            return x, trajs
        return x


##############################################################################
#                             TRAINING FUNCTION
##############################################################################


def train_fm(
    model,
    data,
    num_iterations,
    batch_size,
    criterion,
    optimizer,
    flow_matching,
    device,
):
    """
    Train the flow matching model.

    Args:
        model: velocity-prediction network (UNet1D)
        data: training data, shape (num_samples, 1)
        num_iterations: number of training iterations, a positive integer,
            one random mini-batch per iteration, no pass over the full dataset
        batch_size: batch size for training, a positive integer
        criterion: loss function (nn.MSELoss)
        optimizer: optimizer (Adam)
        flow_matching: flow matching utilities with the conditional path
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
        x_1 = data[indices].to(device)

        # Sample noise and random path times, then run the flow matching process
        noise = torch.randn_like(x_1)
        t = torch.rand(x_1.size(0), device=device)
        x_t, u_t = flow_matching.compute_location_and_velocity(x_1, t, noise)

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
        if (iteration + 1) % 500 == 0:
            print(
                f"Iteration [{iteration + 1}/{num_iterations}], Loss: {loss.item():.4f}"
            )

    return loss_log


##############################################################################
#                           GENERATION FUNCTIONS
##############################################################################


@torch.no_grad()
def compute_path_trajs(flow_matching, data, num_trajs):
    """
    Sample conditional paths x_t that linearly interpolate a noise sample
    x_0 (t = 0) and a data sample x_1 (t = 1).

    At t = 1 the path ends at x_1 + sigma_min * x_0, which equals the clean
    data x_1 only when sigma_min = 0.

    The trajectories are the training paths, not outputs of the
    learned model.

    Args:
        flow_matching: flow matching utilities with the conditional path
        data: clean data, shape (num_samples, 1)
        num_trajs: number of trajectories to record, a positive integer

    Returns:
        trajs: stacked trajectories, shape (num_steps + 1, num_trajs, 1)
            Row 0 is t = 1 (data, plus sigma_min * noise) and the last
            row is t = 0 (noise)
    """
    num_steps = flow_matching.num_steps
    device = flow_matching.device

    x_1 = data[:num_trajs].to(device)
    x_0 = torch.randn_like(x_1)
    traj_points = []

    # Interpolate deterministically from data (t = 1) to noise (t = 0)
    for i in range(num_steps, -1, -1):
        t_batch = torch.full((x_1.size(0),), i / num_steps, device=device)
        x_t, _ = flow_matching.compute_location_and_velocity(x_1, t_batch, x_0)
        traj_points.append(x_t.cpu().clone())

    trajs = torch.stack(traj_points)
    return trajs


@torch.no_grad()
def generate_ode_trajs(model, flow_matching, num_trajs):
    """
    Generation: draw x_0 ~ N(0, I) at t = 0 and integrate the learned ODE
    dx/dt = v_theta(x, t) forward in time up to t = 1 (noise -> data).

    The trajectories are produced entirely by the trained model.

    Args:
        model: trained velocity-prediction network
        flow_matching: flow matching utilities with the ODE sampler
        num_trajs: number of trajectories to record, a positive integer

    Returns:
        trajs: stacked trajectories, shape (num_steps + 1, num_trajs, 1)
            Row 0 is t = 0 (noise) and the last row is t = 1 (data)
    """
    _, trajs = flow_matching.generate_ode(model, num_trajs, return_traj=True)
    return trajs


@torch.no_grad()
def invert_ode_trajs(model, flow_matching, data, num_trajs):
    """
    Integrate from data (t = 1) back to noise (t = 0) by inverting the ODE.

    This does not generate new data. It maps the given data to its
    corresponding latent noise.

    Args:
        model: trained velocity-prediction network
        flow_matching: flow matching utilities with the ODE sampler
        data: clean data used as the initial condition, shape (num_samples, 1)
        num_trajs: number of trajectories to record, a positive integer

    Returns:
        trajs: stacked trajectories, shape (num_steps + 1, num_trajs, 1)
            Row 0 is t = 1 (data) and the last row is t = 0 (noise)
    """
    device = flow_matching.device

    x_1 = data[:num_trajs].to(device)
    _, trajs = flow_matching.invert_ode(model, x_1, return_traj=True)
    return trajs


##############################################################################
#                          VELOCITY FIELD FUNCTION
##############################################################################


@torch.no_grad()
def compute_velocity_field(model, x_range, num_xs, num_ts, device):
    """
    Compute the learned velocity field on a grid over (x, t).

    Args:
        model: trained velocity-prediction network
        x_range: lower and upper bounds of x, a tuple of two floats
        num_xs: number of grid points in x, a positive integer
        num_ts: number of grid points in t, a positive integer
        device: device to run computations on (CPU or GPU)

    Returns:
        field: velocity values on the grid, shape (num_xs, num_ts)
    """
    model.eval()

    xs = torch.linspace(x_range[0], x_range[1], num_xs, device=device).reshape(-1, 1)
    ts = torch.linspace(0.0, 1.0, num_ts, device=device)

    field = np.zeros((num_xs, num_ts))
    for j, t in enumerate(ts):
        t_batch = torch.full((num_xs,), float(t), device=device)
        field[:, j] = model(xs, t_batch).cpu().numpy().reshape(-1)

    model.train()

    return field


##############################################################################
#                               MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Flow Matching (FM) model for "
        "1D Gaussian Mixture Model (GMM) data generation",
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
    parser.add_argument(
        "--num-gen-samples",
        type=int,
        default=DEFAULT_NUM_GEN_SAMPLES,
        help="Number of samples to generate",
    )
    args = parser.parse_args()
    print("Train an FM model for 1D GMM data generation")

    # Set device for computing, CUDA, MPS, or CPU
    device = uf.set_device()
    print(f"Using device: {device}")

    # Set hyperparameters
    num_samples = args.num_samples
    num_iterations = args.num_iterations
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_steps = args.num_steps
    solver = args.solver
    sigma_min = args.sigma_min
    num_gen_samples = args.num_gen_samples

    # Create dataset
    print("Loading GMM data")
    data = dl.load_gaussian_mixture_1d_data(num_samples, device)
    print(f"Training set: {num_samples} samples")
    print(f"Data mean: {data.mean().item():.3f}, std: {data.std().item():.3f}")

    # Set random seed for reproducibility before model initialization
    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)

    # Initialize the velocity-prediction model and flow matching utilities
    # Continuous times in [0, 1] require condition_scale
    model = unet.UNet1D(
        hidden_size,
        condition_scale=1000.0,
    ).to(device)
    flow_matching = FlowMatching1D(num_steps, sigma_min, solver, device)

    # Define loss function, optimizer, and learning rate
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    print(
        f"Training with Adam, iterations = {num_iterations}, "
        f"learning rate = {learning_rate}, batch size = {batch_size}, "
        f"hidden size = {hidden_size}, sigma min = {sigma_min}, "
        f"solver = {solver}, steps = {num_steps}"
    )
    loss_log = train_fm(
        model,
        data,
        num_iterations,
        batch_size,
        criterion,
        optimizer,
        flow_matching,
        device,
    )

    # Plot training loss curve
    print("Plotting training loss curve")
    pf.plot_training_curve(loss_log, MODEL_NAME, y_label="MSE loss")

    # Number of trajectories to compute
    num_trajs = min(25, num_samples)

    # Plot the conditional path (no model) from data to noise
    print("Plotting conditional path (no model) from data to noise")
    conditional_path_trajs = compute_path_trajs(flow_matching, data, num_trajs)
    pf.plot_1d_trajs(
        conditional_path_trajs,
        title=f"{MODEL_NAME} Conditional Path: From Data to Noise",
        file_name=pf.default_file_name(MODEL_NAME, "traj_path"),
        color="green",
        xlabel="integration step (t: 1 to 0)",
        num_trajs=num_trajs,
    )

    # Plot the generative ODE from noise to data
    print("Plotting generative ODE from noise to data")
    generative_ode_trajs = generate_ode_trajs(model, flow_matching, num_trajs)
    pf.plot_1d_trajs(
        generative_ode_trajs,
        title=f"{MODEL_NAME} Generative ODE: From Noise to Data",
        file_name=pf.default_file_name(MODEL_NAME, "traj_generate"),
        color="red",
        xlabel="integration step (t: 0 to 1)",
        num_trajs=num_trajs,
    )

    # Plot the inverse ODE from data to noise
    print("Plotting inverse ODE from data to noise")
    inverse_ode_trajs = invert_ode_trajs(model, flow_matching, data, num_trajs)
    pf.plot_1d_trajs(
        inverse_ode_trajs,
        title=f"{MODEL_NAME} Inverse ODE: From Data to Noise",
        file_name=pf.default_file_name(MODEL_NAME, "traj_invert"),
        color="blue",
        xlabel="integration step (t: 1 to 0)",
        num_trajs=num_trajs,
    )

    # Plot the learned velocity field over (x, t)
    print("Plotting learned velocity field")
    x_range = (-10.0, 10.0)
    field = compute_velocity_field(
        model, x_range, num_xs=200, num_ts=200, device=device
    )
    pf.plot_1d_velocity_field(
        field,
        x_range,
        title=f"{MODEL_NAME}: Learned Velocity Field",
        file_name=pf.default_file_name(MODEL_NAME, "velocity_field"),
    )

    # Sample from the learned model
    print("Sampling from learned model")
    samples = flow_matching.generate_ode(model, num_gen_samples)
    samples = samples.detach().cpu()

    # Plot generated distribution and overlap with original data
    print("Plotting generated samples")
    pf.plot_1d_gen_samples(
        data,
        samples,
        title=f"{MODEL_NAME}: Original Data vs Generated Samples",
        file_name=pf.default_file_name(MODEL_NAME, "gen"),
        gmm_pdf=dl.gaussian_mixture_1d_pdf,
    )
