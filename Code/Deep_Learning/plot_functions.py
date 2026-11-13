##############################################################################
#                       PLOT FUNCTIONS FOR DEEP LEARNING
##############################################################################

import os
import sys

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Random seed for reproducibility
MODEL_SEED = 64

# Directory for analysis plots
PLOT_DIR = "./plot"
# Directory for generative results
GEN_DIR = "./gen"


##############################################################################
#                               HELPER FUNCTIONS
##############################################################################


def _save_fig(fig, file_name, directory=PLOT_DIR, show=True):
    """
    Save a figure to a directory, optionally display it, and release it.

    The figure is always saved. It is displayed only when show is True and the
    Linux machine is not headless, i.e. the backend is not Agg and a display
    server is available.

    Args:
        fig: matplotlib figure
        file_name: name of the saved file, without directory and extension
        directory: target directory, PLOT_DIR for analysis plots and
            GEN_DIR for generative results
        show: whether to display the figure; set to False when the figure is
            written once per epoch inside a training loop, so that training is
            not interrupted by dozens of blocking windows
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{file_name}.png")
    fig.savefig(path, dpi=150)

    has_display = (
        sys.platform == "darwin"
        or os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
    )
    if show and has_display and matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def default_file_name(model_name, suffix):
    """
    Build a file name from a model name and a suffix.

    Args:
        model_name: name of the model, e.g. "VAE-FFN"
        suffix: descriptive suffix, e.g. "training_curve"

    Returns:
        file_name: lower-case file name without directory and extension
    """
    return f"{model_name.lower().replace(' ', '_').replace('-', '_')}_{suffix}"


def _to_numpy(array):
    """
    Convert a torch tensor to a numpy array.

    Args:
        array: torch tensor or numpy array

    Returns:
        array: numpy array sharing the shape of the input
    """
    if torch.is_tensor(array):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _to_displayable(image):
    """
    Reorder a single image so that imshow accepts it.

    Args:
        image: numpy array of shape (image_channels, height, width), or
            (height, width) for an image that is already channel-free

    Returns:
        image: array of shape (height, width) for one channel, or
            (height, width, image_channels) otherwise
    """
    if image.ndim == 2:
        return image
    image = np.transpose(image, (1, 2, 0))
    if image.shape[2] == 1:
        return image[:, :, 0]
    return image


##############################################################################
#                            PLOT TRAINING CURVE
##############################################################################


def plot_training_curve(
    loss_log,
    model_name,
    x_label="iterations",
    y_label="loss",
    file_name=None,
    y_log_scale=True,
):
    """
    Plot the training loss curve(s) of a trained model.

    Args:
        loss_log: training losses, either a list of losses for each batch, or a
            dict mapping a curve label to such a list, e.g.
            {"Discriminator": discriminator_loss_log,
             "Generator": generator_loss_log}
        model_name: name of the trained model, e.g. "VAE-FFN"
        x_label: itertions or epochs of training, a string
        y_label: label of the y axis, a string, e.g.
            "cross entropy loss" for a classifier or "noise prediction loss"
            for a generator
        file_name: name of the saved file, without directory and extension;
            derived from model_name when omitted
        y_log_scale: whether y axis is on log scale
    """
    if file_name is None:
        file_name = default_file_name(model_name, "training_curve")

    # Treat a single loss log as a dict with one unlabelled curve
    if not isinstance(loss_log, dict):
        loss_log = {None: loss_log}

    fig, ax = plt.subplots(figsize=(8, 6))

    for curve_label, losses in loss_log.items():
        label = model_name if curve_label is None else f"{model_name} {curve_label}"
        ax.plot(losses, label=label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Training Loss Curve for {model_name}")
    if y_log_scale == True:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    _save_fig(fig, file_name, directory=PLOT_DIR)


##############################################################################
#              VISUALIZE FEED-FORWARD NETWORK FOR CLASSIFICATION
##############################################################################


def visualize_ffn_binary(instances, labels, model, file_name):
    """
    Visualize the feedforward network decision boundary in 2D space.

    Plot the training data points, decision boundary (P=0.5), and
    probability contours showing confidence regions.

    Note that it assumes that the classes are binary and the data is 2D.

    Args:
        instances: 2D data points, shape (num_instances, 2)
        labels: true labels, shape (num_instances, ), labels in {0, 1}
        model: trained Feedforward Network model
        file_name: name of the model visualization file, string
    """
    # Initialize figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Set axis limits with padding
    x_min, x_max = instances[:, 0].min() - 1, instances[:, 0].max() + 1
    y_min, y_max = instances[:, 1].min() - 1, instances[:, 1].max() + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Generate grid for visualization
    x = np.linspace(x_min, x_max, 200)
    y = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.c_[X.ravel(), Y.ravel()]

    # Predict probabilities for all grid points
    probs = model.forward(Z)
    probs = probs.reshape(X.shape)

    # Plot decision boundary at P(y=1|x) = 0.5
    ax.contour(X, Y, probs, levels=[0.5], colors="red", linestyles="-", alpha=0.5)

    # Plot probability regions with different colors
    # White: very likely negative (P < 0.3)
    # Yellow: somewhat negative (0.3 <= P < 0.5)
    # Orange: somewhat positive (0.5 <= P < 0.7)
    # Red: very likely positive (P >= 0.7)
    contourf = ax.contourf(
        X,
        Y,
        probs,
        alpha=0.3,
        levels=[0, 0.3, 0.5, 0.7, 1],
        colors=["white", "yellow", "orange", "red"],
    )
    fig.colorbar(contourf, ax=ax, label="predicted probability")

    # Separate positive and negative instances
    pos_instances = instances[labels == 1]
    neg_instances = instances[labels == 0]

    # Plot instances, colored by their true labels
    ax.scatter(
        pos_instances[:, 0],
        pos_instances[:, 1],
        marker="o",
        c="green",
        label="positive instances",
    )
    ax.scatter(
        neg_instances[:, 0],
        neg_instances[:, 1],
        marker="x",
        c="blue",
        label="negative instances",
    )

    # Configure plot
    ax.set_title("Feedforward Network for Binary Classification")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    # Save the model visualization to the file
    _save_fig(fig, file_name, directory=PLOT_DIR)


def visualize_ffn_multiclass(instances, labels, model, file_name):
    """
    Visualize the feedforward network predictions in 2D space using contour plot.

    Plot the decision boundaries and actual data points colored by their true class.

    Note that it assumes that the data is 2D.

    Args:
        instances: 2D multiclass classification data, shape (num_instances, 2)
        labels: classification labels, shape (num_instances, ), labels in {0, 1,..., num_classes-1}
        model: trained Feedforward Network model
        file_name: name of the model visualization file
    """
    # Initialize figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Set axis limits with padding
    x_min, x_max = (
        instances[:, 0].min() - 1,
        instances[:, 0].max() + 1,
    )
    y_min, y_max = (
        instances[:, 1].min() - 1,
        instances[:, 1].max() + 1,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Generate grid for visualization
    x = np.linspace(x_min, x_max, 500)
    y = np.linspace(y_min, y_max, 500)
    X, Y = np.meshgrid(x, y)
    points = np.c_[X.ravel(), Y.ravel()]

    # Predict class for all grid points
    _, C = model.predict(points)
    C = C.reshape(X.shape)

    levels = np.arange(model.output_size + 1) - 0.5
    # Plot decision regions with different colors
    contourf = ax.contourf(
        X,
        Y,
        C,
        alpha=0.3,
        levels=levels,
        cmap="viridis",
    )

    # Plot decision boundary at class transitions
    ax.contour(
        X,
        Y,
        C,
        levels=levels,
        colors="red",
        linestyles="-",
        alpha=0.5,
    )

    # Plot instances, colored by their true class
    scatter = ax.scatter(
        instances[:, 0],
        instances[:, 1],
        c=labels,
        cmap="viridis",
        marker="o",
        s=50,
        edgecolors="black",
        linewidths=1.5,
    )

    # Add legend for classes
    for class_index in range(model.output_size):
        ax.scatter(
            [],
            [],
            c=[scatter.cmap(scatter.norm(class_index))],
            marker="o",
            s=50,
            edgecolors="black",
            linewidths=1.5,
            label=f"class {class_index}",
        )

    # Configure plot
    ax.set_title("Feedforward Network for Multiclass Classification")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    # Save the model visualization to the file
    _save_fig(fig, file_name, directory=PLOT_DIR)


##############################################################################
#                         DISPLAY MISCLASSIFIED SAMPLES
##############################################################################


def display_misclassified_images(
    images, true_labels, predicted_labels, model_name, file_name=None
):
    """
    Plot misclassified images with their true and predicted labels.

    The images come from a loader that only applies ToTensor, so the pixel
    values are already in the display range [0, 1].

    Args:
        images: misclassified images, a torch tensor, numpy array, or list of
            images, each of shape (image_channels, height, width) or (height, width)
        true_labels: true labels, a torch tensor, numpy array, or list
        predicted_labels: predicted labels, a torch tensor, numpy array, or list
        model_name: name of the model
        file_name: name of the file of misclassified images
    """
    num_images = len(images)
    if num_images == 0:
        print("No misclassified images to display")
        return

    # Labels may arrive as tensors, so convert them before formatting the titles
    true_labels = _to_numpy(true_labels).reshape(-1)
    predicted_labels = _to_numpy(predicted_labels).reshape(-1)

    # Calculate grid size
    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 6))
    axes = np.atleast_1d(axes).ravel()

    for index, ax in enumerate(axes):
        if index < num_images:
            # Drop the channel axis so that imshow accepts the image
            image = _to_displayable(_to_numpy(images[index]))
            ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(
                f"true: {true_labels[index]}\npred: {predicted_labels[index]}",
                fontsize=10,
            )
        ax.axis("off")

    fig.suptitle(f"Misclassified Samples of {model_name}")
    fig.tight_layout()

    if file_name is None:
        file_name = default_file_name(model_name, "misclassified")

    _save_fig(fig, file_name, directory=PLOT_DIR)


##############################################################################
#                             PLOT VAE LATENT SPACE
##############################################################################


def plot_vae_latent_space(means, labels, model_name, file_name=None):
    """
    Plot the latent space using t-SNE dimensionality reduction.

    The result describes the generative model, so it is saved to PLOT_DIR.

    Args:
        means: latent means of shape (num_samples, latent_size), a torch tensor
            or numpy array
        labels: corresponding labels of shape (num_samples,), a torch tensor
            or numpy array
        model_name: name of model, a string
        file_name: name of the saved file, without directory and extension
    """
    if file_name is None:
        file_name = default_file_name(model_name, "latent_space")

    # t-SNE and the colour mapping both need numpy, and the labels are used as
    # discrete colour indices, so they are cast to integers
    means = _to_numpy(means)
    means = means.reshape(means.shape[0], -1)
    labels = _to_numpy(labels).reshape(-1).astype(int)

    # Sample a subset for faster computation
    num_samples = 5000
    num_classes = 10
    if means.shape[0] > num_samples:
        indices = np.random.choice(means.shape[0], num_samples, replace=False)
        means = means[indices]
        labels = labels[indices]

    # Apply t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=MODEL_SEED)
    means_2d = tsne.fit_transform(means)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(means_2d[:, 0], means_2d[:, 1], c=labels, cmap="tab10", s=2)
    scatter.set_clim(-0.5, 9.5)
    fig.colorbar(scatter, ax=ax, ticks=range(num_classes), label="digit")
    ax.set_xlabel("t-SNE component 1")
    ax.set_ylabel("t-SNE component 2")
    ax.set_title(f"Latent Space of {model_name}")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    _save_fig(fig, file_name, directory=PLOT_DIR)


##############################################################################
#              GENERATIVE MODEL PLOTS (1D DATA AND TRAJECTORIES)
##############################################################################


def plot_1d_trajs(
    traj,
    title,
    file_name,
    color="green",
    xlabel="step",
    num_trajs=25,
):
    """
    Plot a subset of 1D trajectories, one figure per call. The caller decides
    what the trajectories are (conditional path, generative ODE, inverse ODE,
    or a diffusion forward/reverse process).

    Args:
        traj: trajectories of shape (num_steps + 1, num_samples, 1), a torch tensor
            or numpy array
        title: title of the plotted trajectories
        file_name: name of the saved file, without directory and extension
        color: colour of the trajectory lines
        xlabel: label of the horizontal axis
        num_trajs: number of trajectories to draw, a positive integer
    """

    traj = _to_numpy(traj)

    # Pick evenly spaced trajectories instead of a hard-coded subset
    num_available = traj.shape[1]
    num_drawn = max(1, min(num_trajs, num_available))
    indices = np.unique(np.linspace(0, num_available - 1, num_drawn).astype(int))

    fig, ax = plt.subplots(figsize=(12, 6))
    for index in indices:
        ax.plot(traj[:, index, 0], color=color, alpha=0.35, linewidth=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("x value")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    _save_fig(fig, file_name, directory=PLOT_DIR, show=True)


def plot_1d_score_function(
    grid, true_score, learned_score, title, file_name, sigma=None
):
    """
    Plot the learned score against the analytic score of the data.

    Args:
        grid: evaluation points, 1D numpy array
        true_score: analytic score values, 1D numpy array
        learned_score: learned score values, 1D numpy array
        title: title of the plotted score functions
        file_name: name of the saved file, without directory and extension
        sigma: optional noise level shown in the legend
    """

    label = (
        f"learned score (sigma = {sigma:.3f})" if sigma is not None else "learned score"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(grid, true_score, label="true score", linewidth=2)
    ax.plot(grid, learned_score, "--", label=label, linewidth=2)
    ax.set_xlabel("x value")
    ax.set_ylabel("d/dx log p(x)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    _save_fig(fig, file_name, directory=PLOT_DIR, show=True)


def plot_1d_velocity_field(field, x_range, title, file_name):
    """
    Plot the learned velocity field v_theta(x, t) as a heatmap.

    Args:
        field: velocity values of shape (num_xs, num_ts)
        x_range: (min, max) range of x values
        title: title of the plotted velocity field
        file_name: name of the saved file, without directory and extension
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    vmax = float(np.abs(field).max())
    image = ax.imshow(
        field,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        extent=[0.0, 1.0, x_range[0], x_range[1]],
    )
    fig.colorbar(image, ax=ax, label="v_theta(x, t)")
    ax.set_xlabel("t (0 = noise, 1 = data)")
    ax.set_ylabel("x")
    ax.set_title(title)
    fig.tight_layout()

    _save_fig(fig, file_name, directory=PLOT_DIR, show=True)


def plot_1d_gen_samples(
    data, samples, title, file_name, gmm_pdf=None, grid_range=(-10, 10)
):
    """
    Plot the overlap of original data and generated samples in one figure.

    Args:
        data: original 1D data, torch tensor or numpy array
        samples: generated 1D samples, torch tensor or numpy array
        title: title of generated samples
        file_name: name of the saved file, without directory and extension
        gmm_pdf: mapping a grid to the value of GMM density function
        grid_range: (min, max) range of the evaluation grid for gmm_pdf
    """

    data = _to_numpy(data).reshape(-1)
    samples = _to_numpy(samples).reshape(-1)

    fig, ax = plt.subplots(figsize=(10, 6))
    low = min(data.min(), samples.min())
    high = max(data.max(), samples.max())
    bins = np.linspace(low, high, 101)
    ax.hist(data, bins=bins, density=True, alpha=0.5, label="original data")
    ax.hist(samples, bins=bins, density=True, alpha=0.5, label="generated samples")

    if gmm_pdf is not None:
        grid = np.linspace(grid_range[0], grid_range[1], 500)
        ax.plot(grid, gmm_pdf(grid), "k--", lw=1.5, label="true pdf")
    ax.set_xlabel("x value")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    _save_fig(fig, file_name, directory=GEN_DIR, show=True)


##############################################################################
#                           PLOT GENERATED IMAGES
##############################################################################


def plot_generated_images(
    gen_images,
    title,
    file_name,
    num_rows=6,
    num_cols=6,
    value_range=(-1.0, 1.0),
    show=False,
):
    """
    Plot generated images as a grid and save it to the generation directory.

    A model with a tanh output layer (GAN) or a sampler that clamps to [-1, 1]
    (DDPM, SMLD, FM) keeps the default value_range=(-1.0, 1.0), while a model
    with a sigmoid output layer (VAE) must pass value_range=(0.0, 1.0).

    Args:
        gen_images: generated images, a torch tensor or numpy array of shape
            (num_images, image_channels, image_size, image_size)
        title: title of the generated images
        file_name: name of the saved file, without directory and extension
        num_rows: number of rows of the grid, a positive integer
        num_cols: number of columns of the grid, a positive integer
        value_range: (low, high) value range of gen_images, rescaled to [0, 1] for display
        show: whether to display the grid, False when called once per epoch
            inside a training loop
    """

    images = _to_numpy(gen_images)

    # Rescale from value_range to the display range [0, 1]
    low, high = value_range
    images = np.clip((images - low) / (high - low), 0.0, 1.0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    axes = np.atleast_1d(axes).ravel()

    for index, ax in enumerate(axes):
        if index < len(images):
            ax.imshow(_to_displayable(images[index]), cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    _save_fig(fig, file_name, directory=GEN_DIR, show=show)


##############################################################################
#                     PLOT IMAGE GENERATION PROCESS
##############################################################################


def plot_generation_progress(
    trajs,
    title,
    file_name,
    num_rows=8,
    num_cols=8,
    value_range=(-1.0, 1.0),
    show=True,
):
    """
    Plot snapshots of generative trajectories as a grid, one row per sample.

    Each row shows the evolution of one sample from noise (left column) to the
    final image (right column), and the columns are snapshots of the trajectory.

    The snapshots are intermediate states, not final samples, so they are not
    confined to value_range and are clipped before display. The columns are
    concentrated near the end of the trajectory, where the image structure
    appears.

    Args:
        trajs: snapshots of the trajectories, a torch tensor of shape
            (num_snapshots, batch_size, image_channels, height, width);
            trajs[0] is the initial noise and trajs[-1] the final images
        title: title of generated samples
        file_name: name of the saved file, without directory and extension
        num_rows: number of samples (rows) to display, a positive integer,
            clamped to the batch size
        num_cols: number of snapshots (columns) to display, a positive integer,
            clamped to the number of snapshots
        value_range: (low, high) value range of gen_images, rescaled to [0, 1] for display
        show: whether to display the grid
    """

    num_snapshots = len(trajs)
    num_rows = max(1, min(num_rows, len(trajs[0])))
    num_cols = max(1, min(num_cols, num_snapshots))

    # Select snapshot indices, denser near the end where structure appears
    fractions = 1.0 - (1.0 - np.linspace(0.0, 1.0, num_cols)) ** 2
    indices = np.unique(np.round(fractions * (num_snapshots - 1)).astype(int))
    num_cols = len(indices)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    axes = np.asarray(axes).reshape(num_rows, num_cols)

    low, high = value_range
    for column, index in enumerate(indices):
        images = _to_numpy(trajs[index])[:num_rows]
        images = np.clip((images - low) / (high - low), 0.0, 1.0)
        for row in range(num_rows):
            ax = axes[row, column]
            ax.imshow(_to_displayable(images[row]), cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    _save_fig(fig, file_name, directory=GEN_DIR, show=show)
