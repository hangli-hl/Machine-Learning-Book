##############################################################################
#                        K NEAREST NEIGHBORS CLASSIFIER
##############################################################################


import argparse
import heapq

import numpy as np

import data_creators as dc
import utility_functions as uf
import plot_functions as pf

# Default number of neighbors (k) used for the majority vote
DEFAULT_NUM_NEIGHBORS = 1


##############################################################################
#                                KD NODE CLASS
##############################################################################


class KDNode:
    """
    A node in a KD-tree (k-dimensional tree).

    The node stores a data point (instance) and splits the space along one axis.

    Attributes:
        point: coordinates of the point stored at this node, shape (num_features,)
        label: class label associated with this point, an integer
        axis: the axis (feature) along which this node splits the space, an integer
        left: left child subtree (points with feature value <= node value on the axis), a KDNode or None
        right: right child subtree (points with feature value >= node value on the axis), a KDNode or None
    """

    def __init__(self, point, label, axis, left=None, right=None):
        """
        Initialize a KD-tree node.
        """
        self.point = point
        self.label = label
        self.axis = axis
        self.left = left
        self.right = right


##############################################################################
#                  K NEAREST NEIGHBORS CLASSIFIER CLASS
##############################################################################


class KNearestNeighborsClassifier:
    """
    A K Nearest Neighbors Classifier for binary classification using a KD-tree.

    The model uses a KD-tree for efficient k nearest neighbors search.
    With num_neighbors = 1 the classifier reduces to the plain nearest neighbor rule.

    The KD-tree is built recursively (a top-down procedure): at each node, the points
    are sorted along the splitting axis (cycling through the features by depth), and
    the median point is stored at the node.

    The prediction is made by a branch-and-bound search: the tree is traversed to
    collect the k closest points in a bounded max-heap, and the majority class
    among these k nearest neighbors is returned.

    Attributes:
        root: root node of the KD-tree, a KDNode or None
        num_features: number of input features, a positive integer
        num_neighbors: number of neighbors used for the majority vote (k), a positive integer
    """

    def __init__(self, num_neighbors=DEFAULT_NUM_NEIGHBORS):
        """
        Initialize a K Nearest Neighbors Classifier.
        """
        self.root = None
        self.num_features = None
        self.num_neighbors = num_neighbors

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

    @staticmethod
    def _split_at_median(points, labels, axis):
        """
        Split points at the median along one axis.

        The points are sorted along the axis, and the median point becomes the
        splitting point. The points before it go to the left subset, and the
        points after it go to the right subset.

        Args:
            points: points to split, shape (num_instances, num_features)
            labels: labels corresponding to the points, shape (num_instances,)
            axis: the axis (feature) to sort and split on, an integer

        Returns:
            tuple: (median_point, median_label, left_subset, right_subset) where
                - median_point: coordinates of the splitting point, shape (num_features,)
                - median_label: label of the splitting point, an integer
                - left_subset: (points, labels) before the median
                - right_subset: (points, labels) after the median
        """
        sort_indices = np.argsort(points[:, axis])
        sorted_points = points[sort_indices]
        sorted_labels = labels[sort_indices]

        # Median point makes the tree balanced
        median = len(sorted_points) // 2

        left_subset = (sorted_points[:median], sorted_labels[:median])
        right_subset = (sorted_points[median + 1 :], sorted_labels[median + 1 :])
        return sorted_points[median], sorted_labels[median], left_subset, right_subset

    @staticmethod
    def _compute_squared_distance(point, target):
        """
        Compute the squared Euclidean distance between two points.

        Args:
            point: a point in the tree, shape (num_features,)
            target: query point, shape (num_features,)

        Returns:
            distance: squared Euclidean distance, a float
        """
        return np.sum((point - target) ** 2)

    @staticmethod
    def _order_subtrees(target, node):
        """
        Order the two subtrees of a node as (near, far) with respect to the query point.

        The near subtree is on the same side of the splitting plane as the target.

        Args:
            target: query point, shape (num_features,)
            node: the node whose subtrees are ordered, a KDNode

        Returns:
            tuple: (near_subtree, far_subtree), both KDNode (or None)
        """
        if target[node.axis] <= node.point[node.axis]:
            return node.left, node.right
        return node.right, node.left

    @staticmethod
    def _worst_distance(neighbor_heap):
        """
        Return the distance of the currently worst candidate in the heap.

        Args:
            neighbor_heap: max-heap of candidates

        Returns:
            distance: squared Euclidean distance of the worst candidate, a float
                (infinity if the heap is empty)
        """
        if not neighbor_heap:
            return float("inf")
        return -neighbor_heap[0][0]

    def _push_neighbor(self, neighbor_heap, distance, label):
        """
        Insert a candidate into the bounded max-heap of the k nearest neighbors.

        The heap is a max-heap on the distance (distances are stored negated), so its
        root is always the current worst of the k best candidates.

        Args:
            neighbor_heap: max-heap of candidates, updated in place.
                Each entry is a tuple (-distance, label)
            distance: squared Euclidean distance from the query point, a float
            label: class label of the candidate point, an integer
        """
        entry = (-distance, label)

        if len(neighbor_heap) < self.num_neighbors:
            # The heap is not full yet: simply add the candidate
            heapq.heappush(neighbor_heap, entry)
        elif distance < self._worst_distance(neighbor_heap):
            # The candidate is better than the current k-th nearest: replace it
            heapq.heapreplace(neighbor_heap, entry)

    def _need_search_far_side(self, target, node, neighbor_heap):
        """
        Decide whether the far side of a splitting plane may contain a closer point.

        The far side must be searched if fewer than k candidates have been collected,
        or if the splitting plane is closer than the current k-th nearest neighbor.

        Args:
            target: query point, shape (num_features,)
            node: the node defining the splitting plane, a KDNode
            neighbor_heap: max-heap of the current best candidates

        Returns:
            boolean: True if the far subtree must be searched, False otherwise
        """
        if len(neighbor_heap) < self.num_neighbors:
            return True
        plane_distance = (target[node.axis] - node.point[node.axis]) ** 2
        return plane_distance < self._worst_distance(neighbor_heap)

    @staticmethod
    def _extract_labels(neighbor_heap):
        """
        Extract the labels of the candidates in the heap.

        Args:
            neighbor_heap: max-heap of candidates, each entry is (-distance, label)

        Returns:
            neighbor_labels: labels of the candidates, shape (at most num_neighbors,)
        """
        return np.array([label for _, label in neighbor_heap])

    @staticmethod
    def _majority_vote(neighbor_labels):
        """
        Decide the predicted class label from the labels of the k nearest neighbors.

        Return the most frequent class among the neighbors. Ties are broken in favour
        of the smallest label among the tied classes.

        Args:
            neighbor_labels: labels of the k nearest neighbors, (at most num_neighbors,)

        Returns:
            label: label of the majority class, an integer
        """
        unique_labels, label_counts = np.unique(neighbor_labels, return_counts=True)
        return unique_labels[np.argmax(label_counts)]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, instances, labels):
        """
        Train the K Nearest Neighbors Classifier on training data.

        Training only builds a balanced KD-tree from the training data, which
        is used later for efficient k nearest neighbors search.

        Args:
            instances: training instances, shape (num_instances, num_features)
            labels: training labels, shape (num_instances,), labels in {0, 1}
        """
        self._check_labels(labels)
        self.num_features = instances.shape[1]

        # Build the KD-tree from the root
        self.root = self._build_kd_tree(instances, labels)

    def _build_kd_tree(self, points, labels, depth=0):
        """
        Recursively build a balanced KD-tree.

        At each step, either return an empty subtree (if there are no points) or
        select the axis by depth, store the median point at the node, and
        recursively build left and right subtrees.

        Args:
            points: points to include in this subtree, shape (num_instances, num_features)
            labels: labels corresponding to the points, shape (num_instances,)
            depth: current depth in the tree, an integer (default: 0)
                depth counts the number of splits from the root (root depth = 0)

        Returns:
            subtree: the constructed subtree rooted at this node (None if points is empty)
        """
        # Stopping criterion: no points to add
        if len(points) == 0:
            return None

        # Select axis based on depth (cycle through all features)
        axis = depth % self.num_features

        # Split the data at the median and recursively build subtrees
        point, label, (left_points, left_labels), (right_points, right_labels) = (
            self._split_at_median(points, labels, axis)
        )
        left = self._build_kd_tree(left_points, left_labels, depth + 1)
        right = self._build_kd_tree(right_points, right_labels, depth + 1)

        return KDNode(point=point, label=label, axis=axis, left=left, right=right)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, instances):
        """
        Make predictions on multiple instances.

        Search the KD-tree for the k nearest neighbors of each instance and
        return the majority class among them.

        Args:
            instances: input instances, shape (num_instances, num_features)

        Returns:
            labels: predicted class labels in {0, 1}, shape (num_instances,), dtype=int
        """
        return np.array([self.predict_one(instance) for instance in instances])

    def predict_one(self, instance):
        """
        Make a prediction for a single instance.

        Args:
            instance: single instance, shape (num_features,)

        Returns:
            label: predicted class label (majority class of the k nearest neighbors), an integer
        """
        neighbor_heap = []
        self.search_kd_tree(instance, self.root, neighbor_heap)
        return self._majority_vote(self._extract_labels(neighbor_heap))

    def search_kd_tree(self, target, node, neighbor_heap):
        """
        Recursively search the KD-tree for the k nearest neighbors.

        Employ a branch-and-bound strategy: traverse down the tree to collect candidates,
        then unwind the recursion, checking if the other side of the splitting plane
        could contain a point closer than the current k-th nearest neighbor.

        Args:
            target: query point, shape (num_features,)
            node: root of the (sub)tree to search, a KDNode
            neighbor_heap: max-heap of the current best candidates, updated in place.
                Each entry is a tuple (-distance, label)
        """
        # Base case: reached an empty subtree
        if node is None:
            return

        # Try to keep the point of the current node as a candidate
        distance = self._compute_squared_distance(node.point, target)
        self._push_neighbor(neighbor_heap, distance, node.label)

        # Search the side of the splitting plane that contains the target first
        near_subtree, far_subtree = self._order_subtrees(target, node)
        self.search_kd_tree(target, near_subtree, neighbor_heap)

        # Search the other side only if it could contain a closer point
        if self._need_search_far_side(target, node, neighbor_heap):
            self.search_kd_tree(target, far_subtree, neighbor_heap)


##############################################################################
#                              MAIN PROGRAM
##############################################################################


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a K Nearest Neighbors Classifier on a selected dataset",
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
        "--num-neighbors",
        type=int,
        default=DEFAULT_NUM_NEIGHBORS,
        help="Number of nearest neighbors (k) used for the majority vote, "
        "an odd value is recommended to avoid ties",
    )
    args = parser.parse_args()
    print("Train a K Nearest Neighbors Classifier")

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

    # Initialize k nearest neighbors classifier
    model = KNearestNeighborsClassifier(num_neighbors=args.num_neighbors)

    # Train the k nearest neighbors classifier
    print(
        "Training with a balanced KD-tree, "
        f"number of neighbors = {model.num_neighbors}"
    )
    model.train(training_instances, training_labels)
    print("Training completed")

    # Evaluate on training set
    predicted_labels = model.predict(training_instances)
    training_accuracy = uf.calculate_accuracy(training_labels, predicted_labels)
    print(f"Training accuracy: {training_accuracy:.4f}")

    # Evaluate on test set
    predicted_labels = model.predict(test_instances)
    test_accuracy = uf.calculate_accuracy(test_labels, predicted_labels)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Visualize the k nearest neighbors classifier (only for 2D data)
    if num_features == 2:
        pf.visualize_nearest_neighbor(training_instances, training_labels, model)
