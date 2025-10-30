import numpy as np


class KDNode:
    """A node in a k-d tree."""

    def __init__(self, data, label, dim, left=None, right=None):
        """
        data: The feature vector of the point at this node.
        label: The class label of the point.
        dim: The dimension used for splitting at this node.
        left: The left child node.
        right: The right child node.
        """
        self.data = data
        self.label = label
        self.dim = dim
        self.left = left
        self.right = right


class KDTree:
    """A k-d tree data structure for efficient nearest neighbor search."""

    def __init__(self, X_train, y_train):
        """
        X_train: The training features.
        y_train: The training labels.

        Attributes:
            - dims: Number of dimensions in the feature space (will be needed for building the tree)
            - root: The root node of the k-d tree
        """
        self.dims = X_train.shape[1]
        self.root = self._build_tree(X_train, y_train, depth=0)

    def _build_tree(self, X_train, y_train, depth):
        """
        Recursively builds the k-d tree.
        X_train: The subset of training features for this node.
        y_train: The corresponding labels.
        depth: The current recursion depth.
        """
        if len(X_train) == 0:
            return None

        dim = depth % self.dims
        sorted_idx = X_train[:, dim].argsort()
        X_train = X_train[sorted_idx]
        y_train = y_train[sorted_idx]
        median_idx = len(X_train) // 2

        node = KDNode(
            data=X_train[median_idx],
            label=y_train[median_idx],
            dim=dim
        )

        node.left = self._build_tree(X_train[:median_idx], y_train[:median_idx], depth + 1)
        node.right = self._build_tree(X_train[median_idx + 1:], y_train[median_idx + 1:], depth + 1)
        return node

    def _find_nearest(self, node, query_point, best_guess=None, best_dist=np.inf):
        """
        Recursively finds the nearest neighbor to a query point.
        node: The current node in the tree.
        query_point: The point for which to find the nearest neighbor.
        best_guess: The current best neighbor found so far.
        best_dist: The distance to the current best guess.

        Returns:
            best_guess: The nearest neighbor found.
            best_dist: The distance to the nearest neighbor.
        """
        if node is None:
            return best_guess, best_dist

        # TODO: Implement the nearest neighbor search logic
        dist = np.linalg.norm(query_point - node.data)
        if dist < best_dist:
            best_guess = node
            best_dist = dist

        dim = node.dim
        if query_point[dim] < node.data[dim]:
            first_branch = node.left
            second_branch = node.right
        else:
            first_branch = node.right
            second_branch = node.left

        best_guess, best_dist = self._find_nearest(first_branch, query_point, best_guess, best_dist)

        if abs(query_point[dim] - node.data[dim]) < best_dist:
            best_guess, best_dist = self._find_nearest(second_branch, query_point, best_guess, best_dist)

        return best_guess, best_dist

    def find_nearest_neighbor(self, query_point):
        """
        Public method to find the nearest neighbor.
        query_point: The point for which to find the nearest neighbor.
        """
        return self._find_nearest(self.root, query_point)[0]


def kdtree_1nn_classifier(X_train, y_train, X_test):
    """
    Classifies a set of test points using a 1-NN search with a k-d tree.
    X_train: The training features. np.ndarray of shape (num_train_samples, num_features)
    y_train: The training labels. np.ndarray of shape (num_train_samples,)
    X_test: The test features. np.ndarray of shape (num_test_samples, num_features)

    Returns:
        predictions: The predicted labels for the test set. np.ndarray of shape (num_test_samples,)
    """
    tree = KDTree(X_train, y_train)

    predictions = []
    for i, query_point in enumerate(X_test):
        nearest_node = tree.find_nearest_neighbor(query_point)
        predictions.append(nearest_node.label)
    return np.array(predictions)


if __name__ == "__main__":
    # Example usage
    from data_process import preprocess_mnist_data, preprocess_credit_card

    # Load and preprocess the MNIST dataset
    # X_train, y_train, X_test, y_test, mean, std = preprocess_mnist_data("data/MNIST/train.csv", "data/MNIST/t10k.csv")
    X_train, y_train, X_test, y_test, mean, std = preprocess_credit_card(
        "data/credit_card_fraud/credit_card_fraud_train.csv", "data/credit_card_fraud/credit_card_fraud_test.csv"
    )
    print("Data loaded and preprocessed.")
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    # Use a subset for quick testing (you MUST remove this in real use)
    X_train_small = X_train[:1000]
    y_train_small = y_train[:1000]
    X_test_small = X_test[:100]
    y_test_small = y_test[:100]

    # Classify using k-d tree 1-NN
    predictions = kdtree_1nn_classifier(X_train, y_train, X_test_small)

    # Print results
    print("Predictions:", predictions)
    print("True labels:", y_test_small)

    accuracy = np.mean(predictions == y_test_small)
    print(f"Accuracy: {accuracy * 100:.2f}%")
