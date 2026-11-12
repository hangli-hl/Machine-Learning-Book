##############################################################################
#                             GAUSSIAN NAIVE BAYES
##############################################################################


import argparse
import numpy as np

from Data import data_loaders as dl
import utility_functions as uf
import plot_functions as pf

# Small constant added to every variance to avoid division by zero or log(0)
VAR_EPS = 1e-6

##############################################################################
#                          GAUSSIAN NAIVE BAYES CLASS
##############################################################################


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes for binary classification.

    The model assumes that features follow a Gaussian distribution within each class
    and the features are conditionally independent given the class (naive Bayes assumption).

    The model computes posterior probability using Bayes' theorem:
        P(y|x) ∝ P(x|y) · P(y)
    where P(x|y) = ∏_i P(x_i|y) (naive Bayes assumption).

    The model is trained in closed form by maximum likelihood estimation
    (no iterations needed). The class label is predicted as the class with
    the largest posterior probability (MAP - Maximum A Posteriori).

    Attributes:
        num_features: number of features, a positive integer
        num_classes: number of classes, a positive integer
        classes: unique classes, shape (num_classes,)
        means: mean of each feature for each class, shape (num_classes, num_features)
        variances: variance of each feature for each class, shape (num_classes, num_features)
        priors: prior probability of each class, shape (num_classes,)
    """

    def __init__(self):
        """
        Initialize a Gaussian Naive Bayes model.
        """
        self.num_features = None
        self.num_classes = None
        self.classes = None
        self.means = None
        self.variances = None
        self.priors = None

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    @staticmethod
    def _check_labels(labels):
        """
        Check that labels are in {0, 1} and that both classes are present.

        Args:
            labels: training labels of shape (num_instances,)
        """
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Labels must be in {0, 1} and contain both classes")

    def _init_parameters(self, classes, num_features):
        """
        Set classes and feature/class counts, and initialize means, variances,
        and priors to zero.

        Args:
            classes: unique classes, shape (num_classes,)
            num_features: number of features, a positive integer
        """
        self.classes = classes
        self.num_classes = len(classes)
        self.num_features = num_features
        self.means = np.zeros((self.num_classes, num_features))
        self.variances = np.zeros((self.num_classes, num_features))
        self.priors = np.zeros(self.num_classes)

    @staticmethod
    def _estimate_class_parameters(instances_in_class, num_instances):
        """
        Compute the maximum likelihood estimates for one class.

            μ_ki  = mean of feature i in class k
            σ²_ki = variance of feature i in class k (plus VAR_EPS)
            P(k)  = proportion of instances in class k

        Args:
            instances_in_class: instances belonging to the class,
                shape (num_instances_in_class, num_features)
            num_instances: total number of training instances, a positive integer

        Returns:
            tuple: (means, variances, prior) where
                - means: shape (num_features,)
                - variances: shape (num_features,)
                - prior: a float
        """
        means = instances_in_class.mean(axis=0)
        variances = instances_in_class.var(axis=0) + VAR_EPS
        prior = instances_in_class.shape[0] / num_instances
        return means, variances, prior

    @staticmethod
    def _compute_log_priors(priors):
        """
        Compute log P(y) for all classes.

        Args:
            priors: prior probabilities, shape (num_classes,)

        Returns:
            log_priors: shape (num_classes,)
        """
        return np.log(priors)

    @staticmethod
    def _compute_log_likelihood(instances, means, variances):
        """
        Compute log P(x|y) for a single class using the Gaussian PDF
        and the naive Bayes assumption:
            log P(x_i|y) = - 0.5 · log(2πσ²) - 0.5 · (x_i - μ)² / σ²
            log P(x|y)   = Σ_i log P(x_i|y)

        Args:
            instances: instances of shape (num_instances, num_features)
            means: feature means of the class, shape (num_features,)
            variances: feature variances of the class, shape (num_features,)

        Returns:
            log_likelihood: shape (num_instances,)
        """
        log_coefs = -0.5 * np.log(2 * np.pi * variances)
        exponents = -0.5 * ((instances - means) ** 2) / variances
        return np.sum(log_coefs + exponents, axis=1)

    def _compute_log_posteriors(self, instances):
        """
        Compute the unnormalized log posterior probabilities for all instances.

        Uses Bayes' theorem with log probabilities for numerical stability:
            log P(y|x) = log P(x|y) + log P(y) - log P(x)
        Since log P(x) is the same for all classes, it is dropped. The result is
        therefore the log posterior up to an additive constant per instance.

        Args:
            instances: input instances, shape (num_instances, num_features)

        Returns:
            log_posteriors: unnormalized log posterior probabilities,
                shape (num_instances, num_classes)
        """
        log_priors = self._compute_log_priors(self.priors)

        log_likelihoods = np.zeros((instances.shape[0], self.num_classes))
        for class_index in range(self.num_classes):
            log_likelihoods[:, class_index] = self._compute_log_likelihood(
                instances, self.means[class_index, :], self.variances[class_index, :]
            )

        return log_likelihoods + log_priors

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, instances, labels):
        """
        Train the Gaussian Naive Bayes model.

        Estimate the parameters (mean, variance, and prior) for each class
        from the training data by maximum likelihood estimation.

        Args:
            instances: training instances, shape (num_instances, num_features)
            labels: training labels, shape (num_instances,), labels in {0, 1}
        """
        # Check labels; this also guarantees that every class has at least one instance
        self._check_labels(labels)
        self._init_parameters(np.unique(labels), instances.shape[1])

        # Estimate mean, variance, and class prior for each class
        num_instances = instances.shape[0]
        for class_index in range(self.num_classes):
            instances_in_class = instances[labels == self.classes[class_index]]
            means, variances, prior = self._estimate_class_parameters(
                instances_in_class, num_instances
            )
            self.means[class_index, :] = means
            self.variances[class_index, :] = variances
            self.priors[class_index] = prior

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, instances):
        """
        Predict class labels.

        Select the class with the largest posterior probability
        (MAP - Maximum A Posteriori).

        Args:
            instances: input instances, shape (num_instances, num_features)

        Returns:
            labels: predicted class labels in {0, 1}, shape (num_instances,), dtype=int
        """
        log_posteriors = self._compute_log_posteriors(instances)
        class_indices = np.argmax(log_posteriors, axis=1)
        return self.classes[class_indices]

    def forward(self, instances):
        """
        Forward pass: compute the posterior probabilities for all instances.

        Column k corresponds to self.classes[k]; for binary labels in {0, 1},
        column 1 is P(y=1|x).

        Args:
            instances: input instances, shape (num_instances, num_features)

        Returns:
            posteriors: posterior probabilities, shape (num_instances, num_classes)
        """
        log_posteriors = self._compute_log_posteriors(instances)
        return uf.softmax(log_posteriors)


##############################################################################
#                              MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Gaussian Naive Bayes model on a selected dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="gaussians",
        choices=["gaussians", "iris", "jigsaw", "circle", "moons"],
        help="Dataset to use: gaussians (synthetic 2D, two gaussians), "
        "iris (Iris 4D), jigsaw (synthetic 2D, xor patterns), circle (synthetic 2D, "
        "circular patterns) or moons (synthetic 2D, two half moons)",
    )
    args = parser.parse_args()
    print("Train a Gaussian Naive Bayes model for binary classification")

    # Select dataset, labels in {0, 1}
    training_instances, training_labels, test_instances, test_labels = (
        dl.load_binary_classification_data(dataset=args.data, signed_label=False)
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

    # Initialize the Gaussian Naive Bayes model
    model = GaussianNaiveBayes()

    # Train the model (no iterations needed - closed-form solution)
    print("Training with maximum likelihood estimation")
    model.train(training_instances, training_labels)
    print("Training completed")

    # Evaluate on training set
    predicted_training_labels = model.predict(training_instances)
    training_accuracy = uf.evaluate_accuracy(training_labels, predicted_training_labels)
    print(f"Training accuracy: {training_accuracy:.4f}")

    # Evaluate on test set
    predicted_test_labels = model.predict(test_instances)
    test_accuracy = uf.evaluate_accuracy(test_labels, predicted_test_labels)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Visualize the gaussian naive bayes model (only for 2D data)
    if num_features == 2:
        pf.visualize_gaussian_naive_bayes(training_instances, training_labels, model)
