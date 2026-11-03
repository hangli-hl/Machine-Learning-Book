##############################################################################
#                           REGRESSION TREE (CART)
##############################################################################


import argparse

import numpy as np

import data_creators as dc
import utility_functions as uf
import plot_functions as pf

# Default maximum depth of the regression tree
DEFAULT_MAX_TREE_DEPTH = 3
# Default minimum number of instances required to split a node
DEFAULT_MIN_SAMPLE_SPLIT = 5
# Minimum MSE (variance) of a node required to continue splitting
MIN_VARIANCE = 1e-6

##############################################################################
#                            REGRESSION NODE CLASS
##############################################################################


class RegressionNode:
    """
    A node in a regression tree.

    The node is either an internal node representing a split decision or
    a leaf node representing a value prediction.

    Attributes:
        feature: index of the feature to split on (None for leaf nodes), an integer
        threshold: threshold for the split (None for leaf nodes), a float
        left: left child subtree (instances with feature value <= threshold),
            a RegressionNode or None
        right: right child subtree (instances with feature value > threshold),
            a RegressionNode or None
        value: predicted target value if this is a leaf node (None for internal nodes),
            a float or None
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
        Initialize a regression node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        """
        Check if this node is a leaf node.

        Returns:
            boolean: True if this is a leaf node, False if it is an internal node
        """
        return self.value is not None


##############################################################################
#                           REGRESSION TREE CLASS
##############################################################################


class RegressionTree:
    """
    A Regression Tree (CART) for regression.

    Internal nodes represent the regression decision (split) rules.
    Leaf nodes represent the mean of the values of training instances.

    The tree is grown recursively (a greedy, top-down procedure): at each internal node,
    the split that minimizes the weighted MSE of the two child nodes is chosen and
    splitting continues until a stopping criterion is met.

    Attributes:
        root: root node of the regression tree, a RegressionNode or None
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
        Initialize a Regression Tree.
        """
        self.root = None
        self.num_features = None
        self.max_tree_depth = max_tree_depth
        self.min_sample_split = min_sample_split

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

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
    def _compute_mse(values):
        """
        Compute the mean squared error (MSE) of a set of values.

        MSE measures the variance of target values in a node:
            MSE = mean((y - ȳ)²)
        where y are the target values and ȳ is their mean.

        Args:
            values: target values of a node, shape (num_instances,)

        Returns:
            mse: mean squared error (0 for a constant node), a float
        """
        return np.mean((values - np.mean(values)) ** 2)

    def _compute_weighted_mse(self, feature_values, threshold, values):
        """
        Compute the weighted mean squared error (MSE) for a threshold.

        The weighted MSE is:
            total_mse = (num_left/num) · MSE_left + (num_right/num) · MSE_right

        Args:
            feature_values: values of the feature being split, shape (num_instances,)
            threshold: threshold for the split, a float
            values: target values at this node, shape (num_instances,)

        Returns:
            total_mse: weighted MSE (lower is better), a float
        """
        # Split the values based on the threshold
        left_values = values[feature_values <= threshold]
        right_values = values[feature_values > threshold]
        num_left, num_right = len(left_values), len(right_values)
        num = num_left + num_right

        # Invalid threshold: empty subset
        if num_left == 0 or num_right == 0:
            return float("inf")

        # Compute weighted MSE
        mse_left = self._compute_mse(left_values)
        mse_right = self._compute_mse(right_values)
        total_mse = (num_left / num) * mse_left + (num_right / num) * mse_right
        return total_mse

    @staticmethod
    def _decide_leaf_value(values):
        """
        Decide the predicted target value for a leaf node.

        Return the mean of the target values in the node.

        Args:
            values: target values at this node, shape (num_instances,)

        Returns:
            value: mean of target values of this node, a float
        """
        return np.mean(values)

    def _find_best_threshold(self, instances, values):
        """
        Find the best threshold that minimizes the weighted MSE.

        Search over all features and all midpoints between adjacent unique
        feature values to find the threshold that minimizes the weighted MSE.

        Args:
            instances: instances at this node, shape (num_instances, num_features)
            values: values at this node, shape (num_instances,)

        Returns:
            tuple: (best_feature, best_threshold) where
                - best_feature: index of the feature to split on, an integer
                    (None if no valid split exists)
                - best_threshold: threshold for the split, a float
                    (None if no valid split exists)
        """
        best_mse = float("inf")
        best_feature, best_threshold = None, None

        # Try all features and thresholds (a constant feature has no thresholds)
        for feature in range(self.num_features):
            feature_values = instances[:, feature]

            for threshold in self._compute_candidate_thresholds(feature_values):
                # Compute weighted MSE for this threshold
                mse = self._compute_weighted_mse(feature_values, threshold, values)

                # Update best threshold if this one is better
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, instances, values):
        """
        Train the Regression Tree on training data.

        The tree is built recursively using greedy splitting based on mean squared error (MSE) minimization.
        At each node, the algorithm searches over all features and thresholds, selects the
        threshold that minimizes weighted MSE, and recursively builds left and right subtrees.

        Args:
            instances: training instances, shape (num_instances, num_features)
            values: training values, shape (num_instances,)
        """
        self.num_features = instances.shape[1]

        # Build the tree from the root
        self.root = self.grow_tree(instances, values)

    def grow_tree(self, instances, values, depth=0):
        """
        Recursively grow the regression tree.

        At each step, either create a leaf node (if stopping criteria are met)
        or find the best threshold and recursively build left/right subtrees.

        Stopping criteria: the maximum depth is reached, the node has fewer
        than min_sample_split instances, the target values are almost constant
        (MSE below MIN_VARIANCE), or no valid split exists.

        Args:
            instances: instances at this node, shape (num_instances, num_features)
            values: values at this node, shape (num_instances,)
            depth: current depth in the tree, an integer (default: 0)
                depth counts the number of splits from the root (root depth = 0)

        Returns:
            subtree: the constructed subtree rooted at this node, a RegressionNode
        """
        num_instances = instances.shape[0]

        # Search for the best split unless a stopping criterion is already met
        # (max depth reached, too few instances, or near-constant values)
        best_feature, best_threshold = None, None
        if (
            depth < self.max_tree_depth
            and num_instances >= self.min_sample_split
            and self._compute_mse(values) >= MIN_VARIANCE
        ):
            best_feature, best_threshold = self._find_best_threshold(instances, values)

        # Create a leaf if a stopping criterion is met or no valid split exists
        if best_feature is None:
            leaf_value = self._decide_leaf_value(values)
            return RegressionNode(value=leaf_value)

        # Split the data and recursively build subtrees
        left_mask = instances[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left = self.grow_tree(instances[left_mask], values[left_mask], depth + 1)
        right = self.grow_tree(instances[right_mask], values[right_mask], depth + 1)

        return RegressionNode(
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
            values: predicted values, shape (num_instances,), dtype=float
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
            value: predicted value, a float
        """
        # If reached a leaf node, return its predicted value
        if node.is_leaf():
            return node.value

        # Otherwise, recurse to left or right child
        if instance[node.feature] <= node.threshold:
            return self.traverse_tree(instance, node.left)
        return self.traverse_tree(instance, node.right)


##############################################################################
#                               MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Regression Tree (CART) on a selected dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="plane",
        choices=["artificial", "plane", "diabetes"],
        help="Dataset to use: artificial (manual 2D), plane (synthetic 2D) "
        "or diabetes (Diabetes 10D)",
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
    print("Train a Regression Tree (CART)")

    # Select dataset
    training_instances, training_values, test_instances, test_values = (
        dc.select_regression_data(dataset=args.data)
    )
    num_features = training_instances.shape[1]
    print(f"Using dataset: {args.data}")
    print(
        f"Training set: {training_instances.shape[0]} instances, "
        f"{num_features} features"
    )
    print(f"Test set: {test_instances.shape[0]} instances, {num_features} features")

    # Initialize regression tree
    model = RegressionTree(args.max_tree_depth, args.min_sample_split)

    # Train the regression tree
    print(
        "Training with greedy splitting, "
        f"maximum depth = {model.max_tree_depth}, minimum split size = {model.min_sample_split}"
    )
    model.train(training_instances, training_values)
    print("Training completed")

    # Evaluate on training set
    predicted_values = model.predict(training_instances)
    training_mse = uf.calculate_mse(training_values, predicted_values)
    print(f"Training MSE: {training_mse:.4f}")

    # Evaluate on test set
    predicted_values = model.predict(test_instances)
    test_mse = uf.calculate_mse(test_values, predicted_values)
    print(f"Test MSE: {test_mse:.4f}")

    # Visualize the regression tree model (only for 2D data)
    if num_features == 2:
        pf.visualize_regression_tree(training_instances, training_values, model)
