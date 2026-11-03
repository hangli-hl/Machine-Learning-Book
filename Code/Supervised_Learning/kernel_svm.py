##############################################################################
#                  KERNEL SUPPORT VECTOR MACHINE (KERNEL SVM)
##############################################################################


import argparse

import numpy as np
from scipy.optimize import minimize

import data_creators as dc
import utility_functions as uf
import plot_functions as pf

# Tolerance on α for identifying (margin) support vectors
SV_EPS = 1e-6
# Maximum number of iterations of the SLSQP solver
MAX_ITERATIONS = 1000

##############################################################################
#                               KERNEL SVM CLASS
##############################################################################


class KernelSVM:
    """
    Kernel SVM (Support Vector Machine) for binary classification.

    The model uses kernel trick to implicitly map data to a high-dimensional feature space,
    then finds the optimal separating hyperplane in the space.

    The decision function is: f(x) = Σ α_i·y_i·K(x_i, x) + b
    where α_i are Lagrange multipliers, y_i are labels, K is the kernel function,
    x_i are support vectors, and b is the bias.

    The learning algorithm solves the dual quadratic program (QP) with SLSQP to obtain α,
    and computes b from the margin support vectors.

    Attributes:
        kernel: kernel type ("linear", "rbf", "poly")
        C: regularization parameter (larger value = less regularization)
        gamma: kernel coefficient for "rbf" and "poly"
        degree: degree for polynomial kernel
        coef0: independent term in polynomial kernel
        alphas: Lagrange multipliers, shape (num_support_vectors,)
        support_vectors: training instances that are support vectors,
            shape (num_support_vectors, num_features)
        support_vector_labels: labels of support vectors,
            shape (num_support_vectors,)
        support_vector_indices: indices of support vectors in training data,
            shape (num_support_vectors,)
        bias: bias term, a float
    """

    def __init__(self, kernel="rbf", C=1.0, gamma="auto", degree=3, coef0=1.0):
        """
        Initialize a Kernel SVM.
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_indices = None
        self.bias = None

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    @staticmethod
    def _check_labels(labels):
        """
        Check that labels are in {-1, 1} and that both classes are present.

        Args:
            labels: training labels of shape (num_instances,)
        """
        if set(np.unique(labels)) != {-1, 1}:
            raise ValueError("Labels must be in {-1, 1} and contain both classes")

    def _resolve_gamma(self, num_features):
        """
        Resolve gamma: set to 1/num_features if 'auto', otherwise cast to float.

        Args:
            num_features: number of features, a positive integer
        """
        if self.gamma == "auto":
            self.gamma = 1.0 / num_features
        else:
            self.gamma = float(self.gamma)

    def _build_qp_matrix(self, instances, labels):
        """
        Build the QP matrix Q where Q_ij = y_i·y_j·K(x_i, x_j).

        Args:
            instances: training instances of shape (num_instances, num_features)
            labels: training labels of shape (num_instances,)

        Returns:
            qp_matrix: matrix of shape (num_instances, num_instances)
        """
        kernel_matrix = self._kernel_function(instances, instances)
        labels_col = labels.reshape(-1, 1)
        return labels_col * kernel_matrix * labels_col.T

    def _compute_bias(self, instances, labels, alphas):
        """
        Compute the bias b using margin support vectors (where 0 < α < C).

        For each margin support vector x_i, the KKT conditions give:
            b = y_i - Σ α_j·y_j·K(x_j, x_i)
        The final bias is averaged over all margin support vectors.
        Falls back to 0 if no margin support vectors are found.

        Args:
            instances: training instances of shape (num_instances, num_features)
            labels: training labels of shape (num_instances,)
            alphas: Lagrange multipliers of shape (num_instances,)

        Returns:
            bias: bias term, a float
        """
        margin_mask = (alphas > SV_EPS) & (alphas < self.C - SV_EPS)
        if not np.any(margin_mask):
            print("Warning: no margin support vectors; bias falls back to 0")
            return 0.0

        # Σ α_j·y_j·K(x_j, x_i) for every margin support vector x_i
        kernel_values = self._kernel_function(instances[margin_mask], instances)
        scores = kernel_values @ (alphas * labels)

        return float(np.mean(labels[margin_mask] - scores))

    def _kernel_function(self, x1, x2):
        """
        Compute kernel function K(x1, x2).

        Args:
            x1: instances of shape (num_instances_1, num_features)
            x2: instances of shape (num_instances_2, num_features)

        Returns:
            kernel_matrix: kernel matrix of shape (num_instances_1, num_instances_2)
        """
        if self.kernel == "linear":
            # Linear kernel: K(x1, x2) = x1·x2
            return x1 @ x2.T

        elif self.kernel == "rbf":
            # RBF (Gaussian) kernel: K(x1, x2) = exp(-γ·||x1 - x2||^2)

            # Compute pairwise squared Euclidean distances
            # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2·x1·x2
            sq_dists = (
                np.sum(x1**2, axis=1).reshape(-1, 1)
                + np.sum(x2**2, axis=1)
                - 2 * x1 @ x2.T
            )
            return np.exp(-self.gamma * sq_dists)

        elif self.kernel == "poly":
            # Polynomial kernel: K(x1, x2) = (γ·x1·x2 + coef0)^d
            return (self.gamma * x1 @ x2.T + self.coef0) ** self.degree

        else:
            raise ValueError(
                f"Unknown kernel: '{self.kernel}'. "
                f"Available kernels: 'linear', 'rbf', 'poly'"
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, instances, labels):
        """
        Train the Kernel SVM using quadratic programming.

        The optimization problem is solved in dual form using constrained optimization (SLSQP):
            minimize: 1/2 Σ Σ α_i α_j y_i y_j · K(x_i, x_j) - Σ α_i
            subject to: 0 <= α_i <= C, Σ α_i · y_i = 0

        Args:
            instances: training instances of shape (num_instances, num_features)
            labels: training labels of shape (num_instances,), labels in {-1, 1}
        """
        self._check_labels(labels)
        num_instances, num_features = instances.shape
        self._resolve_gamma(num_features)

        # Construct matrix for QP: Q_ij = y_i·y_j·K(x_i, x_j)
        qp_matrix = self._build_qp_matrix(instances, labels)

        # Define the objective function: 1/2·α^T·Q·α - 1^T·α
        def objective(alpha):
            return 0.5 * alpha @ (qp_matrix @ alpha) - np.sum(alpha)

        # Define the gradient: ∇f(α) = Q·α - 1
        def gradient(alpha):
            return qp_matrix @ alpha - np.ones(num_instances)

        # Equality constraint: Σ α_i·y_i = 0
        def constraint_eq(alpha):
            return alpha @ labels

        # Constraints for scipy.optimize.minimize
        constraints = {"type": "eq", "fun": constraint_eq}

        # Bounds: 0 <= α_i <= C
        bounds = [(0, self.C) for _ in range(num_instances)]

        # Initial values of alphas (Lagrange multipliers)
        alpha0 = np.zeros(num_instances)

        # Solve the optimization problem using SLSQP
        result = minimize(
            objective,
            alpha0,
            method="SLSQP",
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": MAX_ITERATIONS, "disp": False},
        )

        # Extract Lagrange multipliers
        alphas = result.x

        # Warn if the optimizer did not converge
        if not result.success:
            print(f"Warning: optimization did not converge - {result.message}")

        # Identify support vectors (where α > threshold)
        support_mask = alphas > SV_EPS

        self.alphas = alphas[support_mask]
        self.support_vectors = instances[support_mask]
        self.support_vector_labels = labels[support_mask]
        self.support_vector_indices = np.where(support_mask)[0]

        # Compute bias b using margin support vectors
        self.bias = self._compute_bias(instances, labels, alphas)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, instances):
        """
        Make predictions on multiple instances.

        Args:
            instances: input instances, shape (num_instances, num_features)

        Returns:
            labels: predicted class labels in {-1, 1}, shape (num_instances,), dtype=int
        """
        # Compute decision function scores for all instances
        scores = self.forward(instances)

        # Apply sign function: label = 1 if score >= 0, label = -1 otherwise
        return np.where(scores >= 0, 1, -1)

    def forward(self, instances):
        """
        Forward pass: compute scores of multiple instances.

        Decision function: f(x) = Σ α_i·y_i·K(x_i, x) + b

        Args:
            instances: input instances, shape (num_instances, num_features)

        Returns:
            scores: scores of instances of shape (num_instances,), values are float
        """
        # Compute kernel matrix between instances and support vectors
        kernel_matrix = self._kernel_function(instances, self.support_vectors)

        # Decision function: f(x) = Σ α_i·y_i·K(x_i, x) + b
        scores = kernel_matrix @ (self.alphas * self.support_vector_labels) + self.bias

        return scores


##############################################################################
#                              MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Kernel SVM on a selected dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="gaussians",
        choices=["artificial", "gaussians", "iris", "jigsaw", "circle", "moons"],
        help="Dataset to use: artificial (manual 2D), gaussians (synthetic 2D, two gaussians), "
        "iris (Iris 4D), jigsaw (synthetic 2D, xor patterns), circle (synthetic 2D, "
        "circular patterns) or moons (synthetic 2D, two half moons)",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=["linear", "rbf", "poly"],
        help="Kernel type: linear, rbf, poly",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Regularization parameter (larger value = less regularization)",
    )
    parser.add_argument(
        "--gamma",
        type=str,
        default="auto",
        help="Kernel coefficient for RBF/poly ('auto' or float value)",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=3,
        help="Degree for polynomial kernel",
    )
    parser.add_argument(
        "--coef0",
        type=float,
        default=1.0,
        help="Independent term in polynomial kernel",
    )
    args = parser.parse_args()
    print("Train a Kernel SVM for binary classification")

    # Select dataset, labels in {-1, +1}
    training_instances, training_labels, test_instances, test_labels = (
        dc.select_binary_classification_data(dataset=args.data, signed_label=True)
    )
    num_features = training_instances.shape[1]
    num_classes = 2
    print(f"Using dataset: {args.data}")
    print(
        f"Training set: {training_instances.shape[0]} instances, "
        f"{num_features} features, {num_classes} classes"
    )
    print(
        f"Test set: {test_instances.shape[0]} instances, "
        f"{num_features} features, {num_classes} classes"
    )

    # Initialize the Kernel SVM model
    model = KernelSVM(args.kernel, args.C, args.gamma, args.degree, args.coef0)

    # Train the model using quadratic programming
    print(
        f"Training with kernel = {model.kernel}, "
        f"C = {model.C}, gamma = {model.gamma}, degree = {model.degree}, coef0 = {model.coef0}"
    )
    model.train(training_instances, training_labels)
    print(f"Training completed with {len(model.alphas)} support vectors")

    # Evaluate on training set
    predicted_labels = model.predict(training_instances)
    training_accuracy = uf.calculate_accuracy(training_labels, predicted_labels)
    print(f"Training accuracy: {training_accuracy:.4f}")

    # Evaluate on test set
    predicted_labels = model.predict(test_instances)
    test_accuracy = uf.calculate_accuracy(test_labels, predicted_labels)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Visualize the kernel svm model (only for 2D data)
    if num_features == 2:
        pf.visualize_kernel_svm(training_instances, training_labels, model)
