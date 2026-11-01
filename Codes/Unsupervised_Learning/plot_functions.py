##############################################################################
#                   PLOT FUNCTIONS FOR UNSUPERVISED LEARNING
##############################################################################


import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

# Random seed for reproducibility
PLOT_SEED = 64

# Directory in which figures are saved
PLOT_DIR = "./plot"

# Keep every TRAJECTORY_THIN snapshot for the trajectory plot
TRAJECTORY_THIN = 10

# A trajectory needs at least this many distinct snapshots to be worth projecting
MIN_TRAJECTORY_SNAPSHOTS = 3


##############################################################################
#                               HELPER FUNCTIONS
##############################################################################


def _save_and_show(file_name):
    """
    Save the current figure into PLOT_DIR and display it.

    Args:
        file_name: file name of the figure, including the extension
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, file_name)
    plt.savefig(path)
    plt.show()


##############################################################################
#                            PLOT CONVERGENCE CURVE
##############################################################################


def plot_convergence_curve(
    series,
    method_name,
    file_name,
    x_label="iteration",
    y_label="value",
    start_iteration=0,
    twin_series=None,
    twin_y_label=None,
):
    """
    Plot training curves against iterations.

    Args:
        series: mapping a legend label to a 1D array of diagnostic values. All series share
            the primary (left) y axis
        method_name: name of the method, used in the plot title
        file_name: file name for saving the plot
        x_label: label of the x axis
        y_label: label of the primary (left) y axis
        start_iteration: iteration of the first recorded value
        twin_series: optional mapping a legend label to a 1D array of values. All series share
            the secondary (right) y axis
        twin_y_label: label of the secondary (right) y axis
    """
    if not series or all(len(values) == 0 for values in series.values()):
        print("No values recorded, skip convergence curve")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot every series sharing the primary y axis
    for label, values in series.items():
        values = np.asarray(values)
        iterations = np.arange(len(values)) + start_iteration
        ax.plot(iterations, values, linewidth=1.5, label=label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()

    # Plot every series sharing the secondary y axis
    if twin_series:
        ax_twin = ax.twinx()
        for label, values in twin_series.items():
            values = np.asarray(values)
            iterations = np.arange(len(values)) + start_iteration
            ax_twin.plot(
                iterations,
                values,
                linestyle="--",
                linewidth=1.2,
                color="gray",
                label=label,
            )
        ax_twin.set_ylabel(twin_y_label if twin_y_label else "")
        twin_handles, twin_labels = ax_twin.get_legend_handles_labels()
        handles += twin_handles
        labels += twin_labels

    ax.legend(handles, labels, loc="best")
    ax.set_title(f"{method_name}: Convergence")
    plt.tight_layout()

    # Save the figure and show it
    _save_and_show(file_name)
    plt.close(fig)


##############################################################################
#                            VISUALIZE CLUSTERING RESULT
##############################################################################


def _get_model_centers(model):
    """
    Get the cluster centers of either a K-Means or a GMM model.

    Args:
        model: trained model (must have 'centroids' or 'means' attribute)

    Returns:
        centers: cluster centers, shape (num_clusters, num_features)
        center_label: attribute name
    """
    if hasattr(model, "centroids"):
        return model.centroids, "centroids"
    if hasattr(model, "means"):
        return model.means, "means"
    raise AttributeError("Model must have 'centroids' or 'means'.")


def _set_axes(ax, title, x_label="x", y_label="y"):
    """
    Apply the shared title / label / origin-line / grid / legend styling to one subplot.

    Args:
        ax: matplotlib axes to set up
        title: subplot title
        x_label: label of the x axis
        y_label: label of the y axis
    """
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend()


def _scatter_samples(ax, samples, labels, **color_kwargs):
    """
    Scatter 2D samples colored by their cluster ids.

    Args:
        ax: matplotlib axes to draw on
        samples: 2D data points, shape (num_samples, 2)
        labels: cluster labels used as colors, shape (num_samples,)
        color_kwargs: shared color mapping (cmap, vmin, vmax) so that several
            subplots map cluster ids to colors in the same way
    """
    return ax.scatter(
        samples[:, 0],
        samples[:, 1],
        c=labels,
        **color_kwargs,
        marker="o",
        s=30,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.7,
    )


def _scatter_centers(ax, centers, label):
    """
    Scatter cluster centers as red crosses.

    Args:
        ax: matplotlib axes to draw on
        centers: cluster centers, shape (num_clusters, 2)
        label: name of the centers
    """
    return ax.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="x",
        c="red",
        s=100,
        linewidths=3,
        label=label,
    )


def _match_labels_to_true(true_labels, labels, num_clusters):
    """
    Find the permutation of predicted cluster ids that best matches the true ids, so that
    the colors of the two plots are comparable.

    Maximizing the total overlap between predicted and true clusters is a linear
    assignment problem on the contingency table.

    Args:
        true_labels: true cluster labels in {0, ..., num_clusters-1}, shape (num_samples,)
        labels: predicted cluster labels in {0, ..., num_clusters-1}, shape (num_samples,)
        num_clusters: size of the (square) contingency table

    Returns:
        mapping: permutation array of shape (num_clusters,) with mapping[old_id] = new_id
    """
    # overlap[t, p]: number of samples in true cluster t and predicted cluster p
    overlap = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    for true, predicted in zip(true_labels, labels):
        overlap[true, predicted] += 1

    # linear_sum_assignment minimizes, so negate to maximize the overlap
    row_ind, col_ind = linear_sum_assignment(-overlap)

    mapping = np.arange(num_clusters)
    mapping[col_ind] = row_ind
    return mapping


def _compute_cluster_centers(samples, labels):
    """
    Compute the mean center of each cluster.

    Args:
        samples: data points, shape (num_samples, num_features)
        labels:  cluster labels, shape (num_samples,)

    Returns:
        centers: array of centers of shape (num_clusters, num_features),
        ordered by sorted unique labels
    """

    unique_labels = np.unique(labels)
    centers = np.array(
        [samples[labels == label].mean(axis=0) for label in unique_labels]
    )
    return centers


def visualize_clustering_result(
    samples,
    model,
    labels,
    true_labels,
    method_name,
    file_name,
):
    """
    Visualize 2D clustering results: ground truth on top, learned clustering below.

    Args:
        samples: 2D data points, shape (num_samples, 2)
        model: trained clustering model (must have 'centroids' or 'means' attribute)
        labels: predicted cluster labels, shape (num_samples,)
        true_labels: true cluster labels, shape (num_samples,)
        method_name: name of the clustering method for the subplot title
        file_name: file name for saving the plot
    """
    # Initialize figure with 2 subplots (vertically stacked)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))

    # Set axis limits with padding
    x_min, x_max = samples[:, 0].min() - 1, samples[:, 0].max() + 1
    y_min, y_max = samples[:, 1].min() - 1, samples[:, 1].max() + 1

    # Determine cluster centers and label text from the model
    learned_centers, center_label = _get_model_centers(model)

    # Number of colors needed. Counting the centers keeps the scale stable even
    # when some cluster id is never predicted (an empty cluster)
    num_colors = (
        max(
            int(np.max(true_labels)),
            int(np.max(labels)),
            len(learned_centers) - 1,
        )
        + 1
    )

    # Relabel the prediction to match the ground truth, and fix the color scale so
    # that both plots map cluster ids to colors in the same way
    if num_colors > 1:
        labels = _match_labels_to_true(true_labels, labels, num_colors)[labels]
    color_kwargs = dict(cmap="viridis", vmin=0, vmax=max(num_colors - 1, 1))

    # ==================== Plot Ground Truth (TOP) ====================
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)

    _scatter_samples(ax1, samples, true_labels, **color_kwargs)

    # Calculate and plot true centers
    true_centers = _compute_cluster_centers(samples, true_labels)
    _scatter_centers(ax1, true_centers, f"true {center_label}")

    _set_axes(ax1, "Ground Truth")

    # ==================== Plot Learned Clustering (BOTTOM) ====================
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)

    _scatter_samples(ax2, samples, labels, **color_kwargs)
    _scatter_centers(ax2, learned_centers, f"learned {center_label}")

    _set_axes(ax2, method_name)
    plt.tight_layout()

    # Save the figure and show it
    _save_and_show(file_name)
    plt.close(fig)


##############################################################################
#                           VISUALIZE PARAMETER SPACE
##############################################################################


def _best_snapshot_index(values, higher_is_better=True):
    """
    Index of the best snapshot, ignoring non-finite diagnostic values.

    Args:
        values: diagnostic value per snapshot, shape (num_snapshots,)
        higher_is_better: True for maximized quantities (log-likelihood), False for
            minimized ones (inertia)

    Returns:
        best_index: index of the best snapshot, or 0 if no value is finite
    """
    finite = np.isfinite(values)
    if not np.any(finite):
        return 0

    if higher_is_better:
        return int(np.argmax(np.where(finite, values, -np.inf)))
    return int(np.argmin(np.where(finite, values, np.inf)))


def _standardize_columns(array):
    """
    Standardize every column of a 2D array to zero mean and unit variance.

    Args:
        array: values to standardize, shape (num_rows, num_columns)

    Returns:
        standardized: column-wise standardized values, shape (num_rows, num_columns)
    """
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    std[std == 0.0] = 1.0
    return (array - mean) / std


def _fix_component_signs(coordinates):
    """
    Remove the sign ambiguity of the principal axes.

    An eigenvector is only defined up to a scale factor, so the same history can come out
    mirrored on different runs. Forcing the largest-magnitude entry of each axis to be
    positive pins down one deterministic orientation.

    Args:
        coordinates: projected snapshots, shape (num_snapshots, 2)

    Returns:
        coordinates: the same array with each column sign-normalized
    """
    for axis in range(coordinates.shape[1]):
        extreme = int(np.argmax(np.abs(coordinates[:, axis])))
        if coordinates[extreme, axis] < 0.0:
            coordinates[:, axis] *= -1.0
    return coordinates


def _project_to_2d(params):
    """
    Project flattened parameter snapshots down to two dimensions with PCA.

    First, conduct standardization: means are O(data), covariance entries are O(data^2) and
    mixing coefficients are O(1/k).

    PCA is linear, so directions and relative distances keep a meaning and the fraction
    of retained variance is reported on the axes.

    Args:
        params: flattened parameter snapshots, shape (num_snapshots, num_parameters)

    Returns:
        coordinates: projected snapshots of shape (num_snapshots, 2), or None if the
            history is too degenerate to project
        axis_labels: (x_label, y_label) for the plot, or None in the degenerate case
    """
    num_snapshots, num_parameters = params.shape

    # Output dimensions need to be at least two snapshots and two parameters
    if min(num_snapshots, num_parameters) < 2:
        return None, None

    pca = PCA(n_components=2, random_state=PLOT_SEED)
    coordinates = _fix_component_signs(pca.fit_transform(_standardize_columns(params)))
    ratios = pca.explained_variance_ratio_
    axis_labels = (
        f"PC1 ({ratios[0]:.1%} of variance)",
        f"PC2 ({ratios[1]:.1%} of variance)",
    )
    return coordinates, axis_labels


def _thin_and_deduplicate(
    parameter_history, value_history, thin, higher_is_better=True
):
    """
    Keep every 'thin'-th snapshot, always keep the best one, then drop repeated snapshots.

    Thinning matters for MCMC: a rejected proposal leaves the parameters untouched, so
    duplicate snapshots collapse into one tight blob that looks like a converged chain.
    Deduplication is harmless for EM, where consecutive snapshots differ until convergence.

    Args:
        parameter_history: flattened parameter snapshots, shape (num_snapshots, num_parameters)
        value_history: diagnostic value per snapshot, shape (num_snapshots,)
        thin: keep every thin-th snapshot, a positive integer (1 keeps all of them)
        higher_is_better: whether the diagnostic is maximized or minimized

    Returns:
        tuple: (parameter_history, value_history) restricted to the kept snapshots,
            in visiting order
    """
    keep = np.arange(parameter_history.shape[0])[::thin]

    # Always keep the best snapshot
    best_overall = _best_snapshot_index(value_history, higher_is_better)
    if not np.isin(best_overall, keep):
        keep = np.sort(np.append(keep, best_overall))

    # Round before comparing so that snapshots differing only in floating-point noise
    # still count as duplicates, and keep the first occurrence of each distinct snapshot
    _, first_seen = np.unique(
        parameter_history[keep].round(10), axis=0, return_index=True
    )
    keep = keep[np.sort(first_seen)]

    return parameter_history[keep], value_history[keep]


def visualize_parameter_space(
    parameter_history,
    value_history,
    file_name,
    value_label="value",
    value_symbol=None,
    thin=1,
    after_burn_in=False,
    higher_is_better=True,
):
    """
    Visualize a learning trajectory in parameter space, projected to 2D with PCA.

    One implementation shared by K-Means, EM, Gibbs sampling and Metropolis-Hastings:
    they all record a sequence of flattened parameter vectors plus one diagnostic value
    per snapshot. The per-algorithm differences live in the thin wrappers below.

    The plot highlights the start point, the end point, and the best point when it is
    not the end point.

    Args:
        parameter_history: flattened parameter snapshots, shape (num_snapshots, num_parameters)
        value_history: diagnostic value per snapshot (log-likelihood, SSE),
            shape (num_snapshots,)
        file_name: file name for saving the plot
        value_label: full name of the diagnostic, used as the colorbar label
        value_symbol: short name of the diagnostic used in the legend
        thin: keep every thin-th snapshot, a positive integer (1 keeps all of them)
        after_burn_in: True when the history starts after an MCMC burn-in phase
        higher_is_better: True for maximized diagnostics (log-likelihood), False for
            minimized ones (K-Means SSE)

    Raises:
        ValueError: if the two histories have different lengths or thin is not positive
    """
    if value_symbol is None:
        value_symbol = value_label

    if thin < 1:
        raise ValueError(f"thin must be >= 1, got {thin}")

    if len(parameter_history) != len(value_history):
        raise ValueError(
            f"parameter_history ({len(parameter_history)}) and value_history "
            f"({len(value_history)}) must have the same length"
        )

    if len(parameter_history) < MIN_TRAJECTORY_SNAPSHOTS:
        print(
            f"Not enough parameter snapshots for a trajectory plot "
            f"(need at least {MIN_TRAJECTORY_SNAPSHOTS}, got {len(parameter_history)})"
        )
        return

    # Convert the histories to arrays, then thin them and drop repeated snapshots
    params = np.asarray(parameter_history, dtype=float)
    values = np.asarray(value_history, dtype=float)
    params, values = _thin_and_deduplicate(params, values, thin, higher_is_better)
    num_snapshots = params.shape[0]

    if num_snapshots < MIN_TRAJECTORY_SNAPSHOTS:
        print(f"Only {num_snapshots} distinct parameter snapshots after thinning")
        return

    # Project the snapshots down to two dimensions using PCA
    param_2d, axis_labels = _project_to_2d(params)
    if param_2d is None:
        print(
            f"Parameter history of shape {params.shape} cannot be projected to 2D: "
            f"at least two snapshots and two parameters are required"
        )
        return

    # Find indices of special points
    start_idx = 0
    end_idx = num_snapshots - 1
    best_idx = _best_snapshot_index(values, higher_is_better)
    best_word = "max" if higher_is_better else "min"

    # Conduct visualization
    fig = plt.figure(figsize=(8, 6))

    # Plot all snapshots, colored by the diagnostic value
    scatter = plt.scatter(
        param_2d[:, 0],
        param_2d[:, 1],
        c=values,
        cmap="viridis",
        s=30,
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
    )

    # Connect consecutive kept snapshots. After thinning they are not adjacent iterations
    plt.plot(param_2d[:, 0], param_2d[:, 1], color="gray", alpha=0.3, linewidth=0.5)

    # Highlight the start point (green diamond)
    start_text = "start point after burn-in" if after_burn_in else "start point"
    plt.scatter(
        param_2d[start_idx, 0],
        param_2d[start_idx, 1],
        c="green",
        s=100,
        marker="d",
        edgecolors="darkgreen",
        linewidths=1,
        label=f"{start_text} ({value_symbol}={values[start_idx]:.2f})",
    )

    # Highlight the end point (red star), noting when it is also the best point
    if best_idx == end_idx:
        endpoint_label = (
            f"endpoint & {best_word} {value_symbol} ({values[end_idx]:.2f})"
        )
    else:
        endpoint_label = f"endpoint ({value_symbol}={values[end_idx]:.2f})"

    plt.scatter(
        param_2d[end_idx, 0],
        param_2d[end_idx, 1],
        c="red",
        s=100,
        marker="*",
        edgecolors="darkred",
        linewidths=1,
        label=endpoint_label,
    )

    # Highlight the best point (blue hexagon) if different from the endpoint
    if best_idx != end_idx:
        plt.scatter(
            param_2d[best_idx, 0],
            param_2d[best_idx, 1],
            c="blue",
            s=100,
            marker="h",
            edgecolors="darkblue",
            linewidths=1,
            label=f"{best_word} {value_symbol} ({values[best_idx]:.2f})",
        )

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(value_label, rotation=270, labelpad=20)

    # Add titles and labels
    plt.title("Trajectory of Learning in Parameter Space (PCA)")
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_and_show(file_name)
    plt.close(fig)


def visualize_parameter_space_kmeans(model, file_name):
    """
    Visualize the K-Means centroid trajectory, colored by the SSE.

    K-Means often converges within a small number of iterations, so every snapshot is kept.

    Args:
        model: trained K-Means model with parameter_history and sse_history
        file_name: file name for saving the plot
    """
    visualize_parameter_space(
        model.parameter_history,
        model.sse_history,
        file_name=file_name,
        value_label="sum of squared errors",
        value_symbol="SSE",
        thin=1,
        after_burn_in=False,
        higher_is_better=False,
    )


def visualize_parameter_space_em(model, file_name):
    """
    Visualize the EM parameter trajectory, colored by the log-likelihood.

    Args:
        model: trained Gaussian Mixture Model with parameter_history and log_likelihood_history
        file_name: file name for saving the plot
    """
    visualize_parameter_space(
        model.parameter_history,
        model.log_likelihood_history,
        file_name=file_name,
        value_label="log-likelihood",
        value_symbol="LL",
        thin=1,
        after_burn_in=False,
        higher_is_better=True,
    )


def visualize_parameter_space_gs(model, file_name):
    """
    Visualize the Gibbs sampling parameter trajectory, colored by the log-likelihood.

    The history is thinned, and the snapshots start after burn-in.

    Args:
        model: trained Gaussian Mixture Model with parameter_history and log_likelihood_history
        file_name: file name for saving the plot
    """
    visualize_parameter_space(
        model.parameter_history,
        model.log_likelihood_history,
        file_name=file_name,
        value_label="log-likelihood",
        value_symbol="LL",
        thin=TRAJECTORY_THIN,
        after_burn_in=True,
        higher_is_better=True,
    )


def visualize_parameter_space_mh(model, file_name):
    """
    Visualize the Metropolis-Hastings parameter trajectory, colored by the log-likelihood.

    The chain is thinned and starts after burn-in. Rejected proposals additionally leave
    duplicate snapshots, which the shared preprocessing removes.

    Args:
        model: trained Gaussian Mixture Model with parameter_history and log_likelihood_history
        file_name: file name for saving the plot
    """
    visualize_parameter_space(
        model.parameter_history,
        model.log_likelihood_history,
        file_name=file_name,
        value_label="log-likelihood",
        value_symbol="LL",
        thin=TRAJECTORY_THIN,
        after_burn_in=True,
        higher_is_better=True,
    )
