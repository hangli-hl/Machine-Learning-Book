##############################################################################
#                    PLOT FUNCTIONS FOR SUPERVISED LEARNING
##############################################################################

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import norm

# Set directory of plotted results
PLOT_DIR = "./plot"

# Size of figure
FIG_SIZE = (8, 6)

# Number of grid points per axis used to visualize a model over the data range
GRID_SIZE = 100

# Region style for predicted labels: the lowest bin is the negative class,
# the highest bin the positive class
LABEL_LEVELS = [-0.5, 0.5, 1.5]
LABEL_COLORS = ["white", "red"]

# Region style for predicted probabilities, in four bins whose lowest and highest
# bins match the negative and the positive classes
PROBABILITY_LEVELS = [0.0, 0.3, 0.5, 0.7, 1.0]
PROBABILITY_COLORS = ["white", "yellow", "orange", "red"]

# Score at which a binary decision boundary is drawn
DECISION_LEVEL = 0.5

# Number of filled contours and contour lines of a regression surface
SURFACE_FILL_LEVELS = 20
SURFACE_LINE_LEVELS = 10


##############################################################################
#                               HELPER FUNCTIONS
##############################################################################


def _set_fig(instances, fig_size=FIG_SIZE, grid_size=GRID_SIZE):
    """
    Set up the figure of a 2D model visualization and the grid to evaluate on.

    The axis limits and the grid both span the padded data range of the instances.

    Args:
        instances: 2D data points, shape (num_instances, 2)
        fig_size: figure size as (width, height) in inches
        grid_size: number of grid points per axis, a positive integer

    Returns:
        fig: matplotlib figure
        ax: matplotlib axes, with the axis limits set to the padded data range
        X: x-coordinates of the grid, shape (grid_size, grid_size)
        Y: y-coordinates of the grid, shape (grid_size, grid_size)
        grid: grid points, shape (grid_size * grid_size, 2), where each row is
            a 2D point (x, y)
    """
    x_min, x_max, y_min, y_max = _make_range(instances)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    X, Y, grid = _make_grid((x_min, x_max, y_min, y_max), grid_size)
    return fig, ax, X, Y, grid


def _make_range(instances, padding=1.0):
    """
    Compute the axis limits spanning the instances, padded on each side.

    Args:
        instances: 2D data points, shape (num_instances, 2)
        padding: margin added on each side, a positive float

    Returns:
        tuple: axis limits as (x_min, x_max, y_min, y_max)
    """
    if instances.shape[1] != 2:
        raise ValueError("Number of features is not 2: cannot plot in 2D")

    x_min, x_max = instances[:, 0].min() - padding, instances[:, 0].max() + padding
    y_min, y_max = instances[:, 1].min() - padding, instances[:, 1].max() + padding
    return x_min, x_max, y_min, y_max


def _make_grid(limits, grid_size=GRID_SIZE):
    """
    Create a grid of evaluation points covering the given axis limits.

    Args:
        limits: axis limits as (x_min, x_max, y_min, y_max)
        grid_size: number of grid points per axis, a positive integer

    Returns:
        X: x-coordinates of the grid, shape (grid_size, grid_size)
        Y: y-coordinates of the grid, shape (grid_size, grid_size)
        grid: grid points, shape (grid_size * grid_size, 2)
    """
    x_min, x_max, y_min, y_max = limits
    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size)
    )
    return X, Y, np.c_[X.ravel(), Y.ravel()]


def _config_fig(ax, title):
    """
    Configure the figure of a 2D model visualization.

    Set the title and the axis labels, draw the coordinate axes, add the
    gridlines and build the legend from the artists plotted so far.

    Args:
        ax: matplotlib axes to draw on
        title: title of the plot
    """
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.grid(alpha=0.3)
    ax.legend()


def _save_fig(fig, file_name):
    """
    Save a figure to the plot directory, display it and release it.

    Args:
        fig: matplotlib figure
        file_name: name of the saved file, without directory and extension
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, f"{file_name}.png")
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)


def _default_file_name(model_name, suffix):
    """
    Build a file name from a model name and a suffix.

    Args:
        model_name: name of the model, e.g. "VAE-FFN"
        suffix: descriptive suffix, e.g. "training_curve"

    Returns:
        file_name: lower-case file name without directory and extension
    """
    return f"{model_name.lower().replace(' ', '_').replace('-', '_')}_{suffix}"


##############################################################################
#                       PLOT FUNCTION FOR TRAINING CURVE
##############################################################################


def plot_training_curve(loss_log, model_name, loss_name, log_x=True):
    """
    Plot the training curve of a model and save it to a file.

    Args:
        loss_log: losses per iteration or epoch, shape (num_steps,)
        model_name: name of the model, string
        loss_name: name of the loss function, string
        log_x: whether to use a logarithmic x-axis
    """
    if len(loss_log) == 0:
        raise ValueError("The loss log is empty: nothing to plot")

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(np.arange(1, len(loss_log) + 1), loss_log, label=model_name)
    ax.set_title(f"Training Curve for {model_name}")
    ax.set_xlabel("Iterations / Epochs")
    ax.set_ylabel(loss_name)
    if log_x:
        ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    # Save the figure to the plot directory
    file_name = _default_file_name(model_name, "training_curve")
    _save_fig(fig, file_name)


##############################################################################
#              VISUALIZATION FUNCTIONS FOR BINARY CLASSIFICATION
##############################################################################


def visualize_binary_classification(
    instances,
    labels,
    score_function,
    title,
    file_name,
    prediction_type="label",
    overlay=None,
):
    """
    Visualize the predicted regions of a binary classifier together with the data.

    Plot the coordinate axes, the predicted regions, the decision boundary for
    probability scores, and the instances marked as positive and negative.

    Args:
        instances: 2D data points, shape (num_instances, 2)
        labels: class labels, shape (num_instances,), labels in {-1, 1} or {0, 1}
        score_function: maps a grid of shape (num_points, 2) to scores of shape
            (num_points,). For prediction_type 'label' it must return 0/1 or boolean
            indicators of the positive class, not -1/+1. For prediction_type
            'probability' it must return probabilities of the positive class
        title: title of the plot
        file_name: name of the saved file, without directory and extension
        prediction_type: 'label' or 'probability'
        overlay: optional function (ax, X, Y, grid) drawing extra model details
    """
    # Validate the prediction type before any figure is created
    region_style = _select_region_style(prediction_type)

    # Set up the figure and the grid
    fig, ax, X, Y, grid = _set_fig(instances)

    # Evaluate the model on a grid
    scores = np.asarray(score_function(grid), dtype=float).reshape(X.shape)

    # Plot the predicted regions, the model-specific artists and the instances
    _plot_predicted_binary_regions(ax, X, Y, scores, region_style)
    if overlay is not None:
        overlay(ax, X, Y, grid)
    _plot_binary_instances(ax, instances, labels)

    # Configure the figure
    _config_fig(ax, title)
    fig.tight_layout()

    _save_fig(fig, file_name)


def _select_region_style(prediction_type):
    """
    Select the contour style used to shade the predicted regions.

    Args:
        prediction_type: 'label' for predicted labels, 'probability' for predicted
            probabilities of the positive class

    Returns:
        dict: contour levels, colors, colorbar settings and whether to draw
            the decision boundary
    """
    if prediction_type == "label":
        return {
            "levels": LABEL_LEVELS,
            "colors": LABEL_COLORS,
            "ticks": [0, 1],
            "tick_labels": ["negative", "positive"],
            "label": "predicted label",
            "boundary": False,
        }
    if prediction_type == "probability":
        return {
            "levels": PROBABILITY_LEVELS,
            "colors": PROBABILITY_COLORS,
            "ticks": PROBABILITY_LEVELS,
            "tick_labels": None,
            "label": "predicted probability",
            "boundary": True,
        }
    raise ValueError(
        f"Unknown prediction_type. Expected type: 'label' or 'probability'"
    )


def _plot_predicted_binary_regions(ax, X, Y, scores, region_style):
    """
    Shade the predicted regions and, for probability scores, draw the boundary.

    Args:
        ax: axes to draw on
        X, Y: grid coordinates, both of shape (num_rows, num_columns)
        scores: model scores on the grid, shape (num_rows, num_columns)
        region_style: region style from _select_region_style
    """
    regions = ax.contourf(
        X,
        Y,
        scores,
        levels=region_style["levels"],
        colors=region_style["colors"],
        alpha=0.3,
    )
    colorbar = ax.figure.colorbar(
        regions, ax=ax, ticks=region_style["ticks"], label=region_style["label"]
    )
    if region_style["boundary"]:
        ax.contour(X, Y, scores, levels=[DECISION_LEVEL], colors="red", linewidths=1.5)
        ax.plot([], [], c="red", linewidth=1.5, label="boundary")

    if region_style["tick_labels"] is not None:
        colorbar.ax.set_yticklabels(region_style["tick_labels"])


def _plot_binary_instances(ax, instances, labels):
    """
    Plot instances as a scatter plot, colored by their labels.

    Args:
        ax: axes to draw on
        instances: 2D data points, shape (num_instances, 2)
        labels: class labels, shape (num_instances,), positive class encoded as 1
    """
    labels = np.asarray(labels)
    pos_instances = instances[labels == 1]
    neg_instances = instances[labels != 1]
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


##############################################################################
#                          VISUALIZE PERCEPTRON MODEL
##############################################################################


def visualize_perceptron(instances, labels, model):
    """
    Visualize the Perceptron model in 2D space, including its decision boundary.

    Args:
        instances: 2D data points, shape (num_instances, 2)
        labels: class labels, shape (num_instances,), labels in {-1, 1}
        model: trained Perceptron model
    """
    w = np.asarray(model.weights, dtype=float)
    if np.linalg.norm(w) == 0:
        raise ValueError("Zero weight vector: cannot draw a linear boundary")

    def overlay(ax, X, Y, grid):
        # Plot the boundary of the linear model
        d = np.array([-w[1], w[0]]) / np.linalg.norm(w)
        base = -model.bias * w / (w @ w)
        span = 10 * np.abs(instances).max()
        pts = np.stack([base - span * d, base + span * d])

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.plot(pts[:, 0], pts[:, 1], color="red", linewidth=1, label="boundary")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    visualize_binary_classification(
        instances,
        labels,
        lambda grid: model.predict(grid) == 1,
        title="Perceptron",
        file_name="perceptron_model",
        overlay=overlay,
    )


##############################################################################
#                       VISUALIZE NEAREST NEIGHBOR MODEL
##############################################################################


def visualize_nearest_neighbor(instances, labels, model):
    """
    Visualize the Nearest Neighbor Classifier in 2D space.

    The predicted regions are unions of the Voronoi cells of the instances.

    Args:
        instances: 2D data points, shape (num_instances, 2)
        labels: class labels, shape (num_instances,), labels in {0, 1}
        model: a trained NN model with a KD-tree
    """
    visualize_binary_classification(
        instances,
        labels,
        lambda grid: model.predict(grid) == 1,
        title="Nearest Neighbor Classifier",
        file_name="nearest_neighbor_model",
    )


##############################################################################
#                     VISUALIZE GAUSSIAN NAIVE BAYES MODEL
##############################################################################


def visualize_gaussian_naive_bayes(instances, labels, model):
    """
    Visualize the Gaussian Naive Bayes model in 2D space.

    Overlay the learned Gaussian density of each class on the probability regions.

    Args:
        instances: 2D data points, shape (num_instances, 2)
        labels: class labels, shape (num_instances,), labels in {0, 1}
        model: trained Gaussian Naive Bayes model
    """

    def overlay(ax, X, Y, grid):
        # Plot the density contours of each class
        for mean, variance in zip(model.means, model.variances):
            density = norm.pdf(grid[:, 0], mean[0], np.sqrt(variance[0])) * norm.pdf(
                grid[:, 1], mean[1], np.sqrt(variance[1])
            )
            density = density.reshape(X.shape)
            ax.contour(
                X,
                Y,
                density,
                levels=np.linspace(density.min(), density.max(), 6),
                colors="black",
                alpha=0.5,
                linewidths=0.5,
            )
        ax.plot([], [], c="black", alpha=0.5, linewidth=0.5, label="Gaussian densities")

    visualize_binary_classification(
        instances,
        labels,
        lambda grid: model.compute_posteriors(grid)[:, 1],
        title="Gaussian Naive Bayes",
        file_name="gaussian_naive_bayes_model",
        prediction_type="probability",
        overlay=overlay,
    )


##############################################################################
#                     VISUALIZE CLASSIFICATION TREE MODEL
##############################################################################


def visualize_classification_tree(instances, labels, model):
    """
    Visualize the Classification Tree model in 2D space.

    Args:
        instances: 2D data points, shape (num_instances, 2)
        labels: class labels, shape (num_instances,), labels in {0, 1}
        model: trained decision tree model
    """
    visualize_binary_classification(
        instances,
        labels,
        lambda grid: model.predict(grid) == 1,
        title="Classification Tree",
        file_name="classification_tree_model",
    )


##############################################################################
#                 VISUALIZE BINOMIAL LOGISTIC REGRESSION MODEL
##############################################################################


def visualize_binomial_logistic_regression(instances, labels, model):
    """
    Visualize the Binomial Logistic Regression model in 2D space.

    Args:
        instances: 2D data points, shape (num_instances, 2)
        labels: class labels, shape (num_instances,), labels in {0, 1}
        model: trained Binomial Logistic Regression model
    """
    visualize_binary_classification(
        instances,
        labels,
        lambda grid: model.forward(grid),
        title="Binomial Logistic Regression",
        file_name="binomial_logistic_regression_model",
        prediction_type="probability",
    )


##############################################################################
#                          VISUALIZE LINEAR SVM MODEL
##############################################################################


def visualize_linear_svm(instances, labels, model):
    """
    Visualize the linear SVM in 2D space, including its decision boundary and margins.

    Args:
        instances: 2D data points, shape (num_instances, 2)
        labels: class labels, shape (num_instances,), labels in {-1, 1}
        model: trained linear SVM model
    """
    w = np.asarray(model.weights, dtype=float)
    if np.linalg.norm(w) == 0:
        raise ValueError("Zero weight vector: cannot draw a linear boundary")

    def overlay(ax, X, Y, grid):
        # Plot the boundary and margins of the linear model
        d = np.array([-w[1], w[0]]) / np.linalg.norm(w)
        span = 10 * np.abs(instances).max()

        def line(offset):
            base = (offset - model.bias) * w / (w @ w)
            return np.stack([base - span * d, base + span * d])

        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        pts = line(0.0)
        ax.plot(pts[:, 0], pts[:, 1], color="red", linewidth=1, label="boundary")

        for offset, label in ((1.0, "margins"), (-1.0, None)):
            pts = line(offset)
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color="red",
                linestyle="dashed",
                linewidth=1,
                label=label,
            )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    visualize_binary_classification(
        instances,
        labels,
        lambda grid: model.predict(grid) == 1,
        title="Linear SVM",
        file_name="linear_svm_model",
        overlay=overlay,
    )


##############################################################################
#                          VISUALIZE KERNEL SVM MODEL
##############################################################################


def visualize_kernel_svm(instances, labels, model):
    """
    Visualize the Kernel SVM model in 2D space, including its decision boundary,
    margins and support vectors.

    Args:
        instances: 2D data points, shape (num_instances, 2)
        labels: class labels, shape (num_instances,), labels in {-1, 1}
        model: trained Kernel SVM model
    """

    if model.support_vectors is None or len(model.support_vectors) == 0:
        raise ValueError("No support vector: model has no support vectors to visualize")

    def overlay(ax, X, Y, grid):
        # Plot the boundary f(x) = 0 and the margins f(x) = ±1
        scores = model.forward(grid).reshape(X.shape)
        ax.contour(X, Y, scores, levels=[0], colors="red", linewidths=1)
        ax.contour(
            X,
            Y,
            scores,
            levels=[-1, 1],
            colors="red",
            linestyles="dashed",
            linewidths=1,
        )
        ax.plot([], [], c="red", linewidth=1, label="boundary")
        ax.plot([], [], c="red", linestyle="dashed", linewidth=1, label="margins")

        # Highlight the support vectors
        ax.scatter(
            model.support_vectors[:, 0],
            model.support_vectors[:, 1],
            s=100,
            linewidths=1,
            facecolors="none",
            edgecolors="orange",
            label="support vectors",
        )

    visualize_binary_classification(
        instances,
        labels,
        lambda grid: model.predict(grid) == 1,
        title="Kernel SVM",
        file_name="kernel_svm_model",
        overlay=overlay,
    )


##############################################################################
#                           VISUALIZE ADABOOST MODEL
##############################################################################


def visualize_adaboost(instances, labels, model):
    """
    Visualize the AdaBoost model in 2D space.

    Args:
        instances: 2D data points, shape (num_instances, 2)
        labels: class labels, shape (num_instances,), labels in {-1, 1}
        model: trained AdaBoost model
    """
    visualize_binary_classification(
        instances,
        labels,
        lambda grid: model.predict(grid) == 1,
        title="AdaBoost",
        file_name="adaboost_model",
    )


##############################################################################
#             VISUALIZE GRADIENT BOOSTING BINARY CLASSIFIER MODEL
##############################################################################


def visualize_gradient_boosting_binary(instances, labels, model):
    """
    Visualize the Gradient Boosting Binary Classifier in 2D space.

    Args:
        instances: 2D data points, shape (num_instances, 2)
        labels: class labels, shape (num_instances,), labels in {0, 1}
        model: trained Gradient Boosting model
    """
    visualize_binary_classification(
        instances,
        labels,
        lambda grid: model.forward(grid),
        title="Gradient Boosting Binary Classifier",
        file_name="gradient_boosting_binary_model",
        prediction_type="probability",
    )


##############################################################################
#            VISUALIZATION FUNCTIONS FOR MULTICLASS CLASSIFICATION
##############################################################################


def visualize_multiclass_classification(
    instances, labels, label_function, num_classes, title, file_name
):
    """
    Visualize the predicted regions of a multiclass classifier together with the data.

    Plot the coordinate axes, the predicted class regions, and the instances
    colored by their classes.

    Args:
        instances: 2D data points, shape (num_instances, 2)
        labels: class labels, shape (num_instances,)
        label_function: callable mapping grid points of shape (num_points, 2) to
            predicted class labels of shape (num_points,)
        num_classes: number of classes
        title: title of the plot
        file_name: name of the saved file, without directory and extension
    """
    # Set up the figure and the grid
    fig, ax, X, Y, grid = _set_fig(instances)

    # Evaluate the model on a grid
    Z = np.asarray(label_function(grid)).reshape(X.shape)

    # Share one discrete color scale between regions and instances
    colormap = ListedColormap(plt.get_cmap("tab10").colors[:num_classes])
    class_norm = BoundaryNorm(np.arange(num_classes + 1) - 0.5, num_classes)

    _plot_predicted_multiclass_regions(ax, X, Y, Z, colormap, class_norm)
    _plot_multiclass_instances(ax, instances, labels, colormap, class_norm)

    # Configure the figure
    _config_fig(ax, title)
    fig.tight_layout()

    _save_fig(fig, file_name)


def _plot_predicted_multiclass_regions(ax, X, Y, Z, colormap, class_norm):
    """
    Plot the predicted class regions as a filled mesh.

    Args:
        ax: axes to draw on
        X, Y: grid coordinates, both of shape (num_rows, num_columns)
        Z: predicted class labels on the grid, shape (num_rows, num_columns)
        colormap: discrete colormap mapping class labels to colors
        class_norm: normalizer mapping class labels to colormap positions
    """
    ax.pcolormesh(
        X, Y, Z, cmap=colormap, norm=class_norm, alpha=0.3, shading="nearest", zorder=0
    )


def _plot_multiclass_instances(ax, instances, labels, colormap, class_norm):
    """
    Plot instances as a scatter plot, colored by their classes.

    Args:
        ax: axes to draw on
        instances: 2D data points, shape (num_instances, 2)
        labels: class labels, shape (num_instances,)
        colormap: discrete colormap mapping class labels to colors
        class_norm: normalizer mapping class labels to colormap positions
    """
    labels = np.asarray(labels)
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(
            instances[mask, 0],
            instances[mask, 1],
            color=colormap(class_norm(label)),
            s=30,
            edgecolors="black",
            linewidths=1.5,
            label=f"class {label}",
        )


##############################################################################
#               VISUALIZE MULTINOMIAL LOGISTIC REGRESSION MODEL
##############################################################################


def visualize_multinomial_logistic_regression(instances, labels, model):
    """
    Visualize the Multinomial Logistic Regression model in 2D space.

    Args:
        instances: 2D data points, shape (num_instances, 2)
        labels: class labels, shape (num_instances,)
        model: trained Multinomial Logistic Regression model
    """
    visualize_multiclass_classification(
        instances,
        labels,
        lambda grid: model.predict(grid),
        num_classes=model.num_classes,
        title="Multinomial Logistic Regression",
        file_name="multinomial_logistic_regression_model",
    )


##############################################################################
#           VISUALIZE GRADIENT BOOSTING MULTICLASS CLASSIFIER MODEL
##############################################################################


def visualize_gradient_boosting_multiclass(instances, labels, model):
    """
    Visualize the Gradient Boosting Multiclass Classifier in 2D space.

    Args:
        instances: 2D data points, shape (num_instances, 2)
        labels: class labels, shape (num_instances,)
        model: trained Gradient Boosting model
    """
    visualize_multiclass_classification(
        instances,
        labels,
        lambda grid: model.predict(grid),
        num_classes=model.num_classes,
        title="Gradient Boosting Multiclass Classifier",
        file_name="gradient_boosting_multiclass_model",
    )


##############################################################################
#                    VISUALIZATION FUNCTIONS FOR REGRESSION
##############################################################################


def visualize_regression(
    instances, values, value_function, title, file_name, label_contours=False
):
    """
    Visualize a regression model and the data in 2D space.

    Plot the coordinate axes, the prediction surface as filled contours with
    contour lines, and the instances grayscaled by their target values.

    Args:
        instances: 2D data points, shape (num_instances, 2)
        values: target values, shape (num_instances,)
        value_function: callable mapping grid points of shape (num_points, 2) to
            predicted values of shape (num_points,)
        title: title of the plot
        file_name: name of the saved file, without directory and extension
        label_contours: whether to annotate the contour lines with their values
    """
    # Set up the figure and the grid
    fig, ax, X, Y, grid = _set_fig(instances)

    # Evaluate the model on a grid
    Z = np.asarray(value_function(grid), dtype=float).reshape(X.shape)

    _plot_predicted_regression_levels(ax, X, Y, Z, label_contours=label_contours)
    _plot_regression_instances(ax, instances, values)

    # Configure the figure
    _config_fig(ax, title)
    fig.tight_layout()

    _save_fig(fig, file_name)


def _plot_predicted_regression_levels(ax, X, Y, Z, label_contours=False):
    """
    Plot the prediction surface as filled contours with contour lines.

    Args:
        ax: axes to draw on
        X, Y: grid coordinates, both of shape (num_rows, num_columns)
        Z: predicted values on the grid, shape (num_rows, num_columns)
        label_contours: whether to annotate the contour lines with their values
    """
    surface = ax.contourf(X, Y, Z, levels=SURFACE_FILL_LEVELS, cmap="Reds", alpha=0.5)
    ax.figure.colorbar(surface, ax=ax, label="predicted value")

    contour_lines = ax.contour(
        X, Y, Z, levels=SURFACE_LINE_LEVELS, colors="black", alpha=0.5, linewidths=0.5
    )
    if label_contours:
        ax.clabel(contour_lines, inline=True, fontsize=8)


def _plot_regression_instances(ax, instances, values):
    """
    Plot instances as a scatter plot, grayscaled by their target values.

    Args:
        ax: axes to draw on
        instances: 2D data points, shape (num_instances, 2)
        values: target values, shape (num_instances,)
    """
    ax.scatter(
        instances[:, 0],
        instances[:, 1],
        c=values,
        cmap="gray_r",
        s=30,
        edgecolors="black",
        linewidths=1.5,
        label="instances (grayscale: true values)",
    )


##############################################################################
#                      VISUALIZE LINEAR REGRESSION MODEL
##############################################################################


def visualize_linear_regression(instances, values, model):
    """
    Visualize the Linear Regression model in 2D space.

    Args:
        instances: 2D data points, shape (num_instances, 2)
        values: target values, shape (num_instances,)
        model: trained LinearRegression model
    """
    visualize_regression(
        instances,
        values,
        model.predict,
        title="Linear Regression",
        file_name="linear_regression_model",
        label_contours=True,
    )


##############################################################################
#                       VISUALIZE REGRESSION TREE MODEL
##############################################################################


def visualize_regression_tree(instances, values, model):
    """
    Visualize the Regression Tree model in 2D space.

    Args:
        instances: 2D data points, shape (num_instances, 2)
        values: target values, shape (num_instances,)
        model: trained regression tree model
    """
    visualize_regression(
        instances,
        values,
        model.predict,
        title="Regression Tree",
        file_name="regression_tree_model",
    )


##############################################################################
#                 VISUALIZE GRADIENT BOOSTING REGRESSOR MODEL
##############################################################################


def visualize_gradient_boosting_regressor(instances, values, model):
    """
    Visualize the Gradient Boosting Regressor in 2D space.

    Args:
        instances: 2D data points, shape (num_instances, 2)
        values: target values, shape (num_instances,)
        model: trained Gradient Boosting Regressor
    """
    visualize_regression(
        instances,
        values,
        model.predict,
        title="Gradient Boosting Regressor",
        file_name="gradient_boosting_regressor_model",
        label_contours=True,
    )
