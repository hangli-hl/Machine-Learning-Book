##############################################################################
#                          CLASSIFICATION TREE (CART)
##############################################################################


import argparse

import numpy as np

import data_creators as dc
import utility_functions as uf
import plot_functions as pf

# Default maximum depth of the classification tree
DEFAULT_MAX_TREE_DEPTH = 3
# Default minimum number of instances required to split a node
DEFAULT_MIN_SAMPLE_SPLIT = 5


##############################################################################
#                          CLASSIFICATION NODE CLASS
##############################################################################


class ClassificationNode:
    """
    A node in a classification tree.

    The node is either an internal node representing a split decision or
    a leaf node representing a class prediction.

    Attributes:
        feature: index of the feature to split on (None for leaf nodes), an integer
        threshold: threshold for the split (None for leaf nodes), a float
        left: left child subtree (instances with feature value <= threshold),
            a ClassificationNode or None
        right: right child subtree (instances with feature value > threshold),
            a ClassificationNode or None
        label: predicted class label if this is a leaf node (None for internal nodes),
            an integer or None
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        """
        Initialize a classification node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

    def is_leaf(self):
        """
        Check if this node is a leaf node.

        Returns:
            boolean: True if it is a leaf node, False if it is an internal node
        """
        return self.label is not None


##############################################################################
#                        CLASSIFICATION TREE CLASS
##############################################################################


class ClassificationTree:
    """
    A Classification Tree (CART) for binary classification.

    Internal nodes represent classification (split) decision rules.
    Leaf nodes represent the majority class of training instances.

    The tree is grown recursively (a greedy, top-down procedure): at each internal node,
    the split that minimizes the weighted Gini index is chosen, and splitting continues
    until a stopping criterion is met.

    Attributes:
        root: root node of the classification tree, a ClassificationNode or None
        num_features: number of input features, a positive integer
        max_tree_depth: maximum depth of the tree, a positive integer
        min_sample_split: minimum instances required to split a node, a positive integer
    """

    def __init__(
        self,
        max_tree_depth=DEFAULT_MAX_TREE_DEPTH,
        min_sample_split=DEFAULT_MIN_SAMPLE_SPLIT,
    ):
        """
        Initialize a Classification Tree.
        """
        self.root = None
        self.num_features = None
        self.max_tree_depth = max_tree_depth
        self.min_sample_split = min_sample_split

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    @staticmethod
    def _check_labels(labels):
        """
        Check that labels are in {0, 1} and that both classes are present.

        Args:
            labels: training labels, shape (num_instances,)
        """
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Labels must be in {0, 1} and contain both classes")

    @staticmethod
    def _compute_candidate_thresholds(feature_values):
        """
        Compute candidate split thresholds for one feature.

        The candidates are the midpoints between adjacent unique values.
        A constant feature has no valid split point, so the result is empty.

        Args:
            feature_values: values of one feature, shape (num_instances,)

        Returns:
            thresholds: candidate thresholds, shape (num_unique_values - 1,)
        """
        # Convert to float to avoid integer overflow and integer division
        # when computing midpoints
        unique_values = np.unique(feature_values).astype(float)
        return (unique_values[:-1] + unique_values[1:]) / 2.0

    @staticmethod
    def _compute_gini(labels):
        """
        Compute the Gini index of a set of labels.

        Gini index measures the impurity of a node:
            gini = 1 - Σ_i (p_i²)
        where p_i is the proportion of class i.

        Args:
            labels: class labels of a node, shape (num_instances,)

        Returns:
            gini: Gini index (0 for a pure node), a float
        """
        _, counts = np.unique(labels, return_counts=True)
        proportions = counts / len(labels)
        return 1.0 - np.sum(proportions**2)

    def _compute_weighted_gini(self, feature_values, threshold, labels):
        """
        Compute the weighted Gini index for a threshold.

        The weighted Gini index is:
            total_gini = (num_left/num) · gini_left + (num_right/num) · gini_right

        Args:
            feature_values: values of the feature being split, shape (num_instances,)
            threshold: threshold for the split, a float
            labels: class labels at this node, shape (num_instances,)

        Returns:
            total_gini: weighted Gini index (lower is better), a float
        """
        # Split the labels based on the threshold
        left_labels = labels[feature_values <= threshold]
        right_labels = labels[feature_values > threshold]
        num_left, num_right = len(left_labels), len(right_labels)
        num = num_left + num_right

        # Invalid threshold: empty subset
        if num_left == 0 or num_right == 0:
            return float("inf")

        # Compute weighted Gini index
        gini_left = self._compute_gini(left_labels)
        gini_right = self._compute_gini(right_labels)
        total_gini = (num_left / num) * gini_left + (num_right / num) * gini_right
        return total_gini

    @staticmethod
    def _decide_leaf_label(labels):
        """
        Decide the predicted class label for a leaf node.

        Return the most frequent class in the node (ties go to the smaller label).

        Args:
            labels: labels of this node, shape (num_instances,)

        Returns:
            label: label of the most frequent class, an integer
        """
        values, counts = np.unique(labels, return_counts=True)
        return values[np.argmax(counts)]

    def _find_best_threshold(self, instances, labels):
        """
        Find the best threshold that minimizes the weighted Gini index.

        Search over all features and all midpoints between adjacent unique
        feature values to find the threshold that minimizes the weighted Gini index.

        Args:
            instances: instances at this node, shape (num_instances, num_features)
            labels: class labels at this node, shape (num_instances,)

        Returns:
            tuple: (best_feature, best_threshold) where
                - best_feature: index of the feature to split on, an integer
                    (None if no valid split exists)
                - best_threshold: threshold for the split, a float
                    (None if no valid split exists)
        """
        best_gini = float("inf")
        best_feature, best_threshold = None, None

        # Try all features and thresholds (a constant feature has no thresholds)
        for feature in range(self.num_features):
            feature_values = instances[:, feature]

            for threshold in self._compute_candidate_thresholds(feature_values):
                # Compute weighted Gini index for this threshold
                gini = self._compute_weighted_gini(feature_values, threshold, labels)

                # Update best threshold if this one is better
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, instances, labels):
        """
        Train the Classification Tree on training data.

        The tree is built recursively using greedy splitting that minimizes the weighted Gini index.
        At each node, the algorithm searches over all features and thresholds, selects the
        threshold that minimizes weighted Gini index, and recursively builds left and right subtrees.

        Args:
            instances: training instances, shape (num_instances, num_features)
            labels: training labels, shape (num_instances,), labels in {0, 1}
        """
        self._check_labels(labels)
        self.num_features = instances.shape[1]

        # Build the tree from the root
        self.root = self.grow_tree(instances, labels)

    def grow_tree(self, instances, labels, depth=0):
        """
        Recursively grow the classification tree.

        At each step, either create a leaf node (if stopping criteria are met)
        or find the best threshold and recursively build left/right subtrees.

        Stopping criteria: the maximum depth is reached, the node has fewer
        than min_sample_split instances, the node is pure (only one class),
        or no valid split exists.

        Args:
            instances: instances at this node, shape (num_instances, num_features)
            labels: labels at this node, shape (num_instances,)
            depth: current depth in the tree, an integer (default: 0)
                depth counts the number of splits from the root (root depth = 0)

        Returns:
            subtree: the constructed subtree rooted at this node, a ClassificationNode
        """
        num_instances = instances.shape[0]
        num_classes_in_node = len(np.unique(labels))

        # Search for the best split unless a stopping criterion is already met
        # (max depth reached, too few instances, or pure node)
        best_feature, best_threshold = None, None
        if (
            depth < self.max_tree_depth
            and num_instances >= self.min_sample_split
            and num_classes_in_node > 1
        ):
            best_feature, best_threshold = self._find_best_threshold(instances, labels)

        # Create a leaf if a stopping criterion is met or no valid split exists
        if best_feature is None:
            leaf_label = self._decide_leaf_label(labels)
            return ClassificationNode(label=leaf_label)

        # Split the data and recursively build subtrees
        left_mask = instances[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left = self.grow_tree(instances[left_mask], labels[left_mask], depth + 1)
        right = self.grow_tree(instances[right_mask], labels[right_mask], depth + 1)

        return ClassificationNode(
            feature=best_feature, threshold=best_threshold, left=left, right=right
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, instances):
        """
        Make predictions on multiple instances.

        Traverse the tree from root to leaf for each instance.

        Args:
            instances: input instances, shape (num_instances, num_features)

        Returns:
            labels: predicted class labels in {0, 1}, shape (num_instances,), dtype=int
        """
        return np.array(
            [self.traverse_tree(instance, self.root) for instance in instances]
        )

    def traverse_tree(self, instance, node):
        """
        Traverse the tree to make a prediction for a single instance.

        Recursively follow the tree from root to leaf based on split rules.

        Args:
            instance: single instance, shape (num_features,)
            node: current node in the tree

        Returns:
            label: predicted class label, an integer
        """
        # If reached a leaf node, return its label
        if node.is_leaf():
            return node.label

        # Otherwise, recurse to left or right child
        if instance[node.feature] <= node.threshold:
            return self.traverse_tree(instance, node.left)
        return self.traverse_tree(instance, node.right)


##############################################################################
#                              MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Classification Tree (CART) on a selected dataset",
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
        "--max-tree-depth",
        type=int,
        default=DEFAULT_MAX_TREE_DEPTH,
        help="Maximum depth of the tree",
    )
    parser.add_argument(
        "--min-sample-split",
        type=int,
        default=DEFAULT_MIN_SAMPLE_SPLIT,
        help="Minimum number of instances required to split a node",
    )
    args = parser.parse_args()
    print("Train a Classification Tree (CART)")

    # Select dataset, labels in {0, 1}
    training_instances, training_labels, test_instances, test_labels = (
        dc.select_binary_classification_data(dataset=args.data, signed_label=False)
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

    # Initialize classification tree
    model = ClassificationTree(args.max_tree_depth, args.min_sample_split)

    # Train the classification tree
    print(
        "Training with greedy splitting, "
        f"maximum depth = {model.max_tree_depth}, minimum split size = {model.min_sample_split}"
    )
    model.train(training_instances, training_labels)
    print("Training completed")

    # Evaluate on training set
    predicted_labels = model.predict(training_instances)
    training_accuracy = uf.evaluate_accuracy(training_labels, predicted_labels)
    print(f"Training accuracy: {training_accuracy:.4f}")

    # Evaluate on test set
    predicted_labels = model.predict(test_instances)
    test_accuracy = uf.evaluate_accuracy(test_labels, predicted_labels)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Visualize the classification tree model (only for 2D data)
    if num_features == 2:
        pf.visualize_classification_tree(training_instances, training_labels, model)
