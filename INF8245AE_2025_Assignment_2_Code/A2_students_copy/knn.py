import numpy as np
import scipy
import sklearn.metrics.pairwise as smp

from data_process import preprocess_mnist_data
from utils import visualize_image
import pandas as pd
import os


def euclidean_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Computes the Euclidean distance between two arrays

    Args:
        A (np.ndarray): Numpy array of shape [num_samples_a x num_features]
        B (np.ndarray): Numpy array of shape [num_samples_b x num_features]

    Returns:
        np.ndarray: Numpy array of shape [num_samples_a x num_samples_b] where
                    each column contains the distance between one element in
                    matrix_b and all elements in matrix_a
    """
    distances = None

    ### Implement here
    # You might want to use a metric in sklearn.metrics.pairwise to avoid potential out-of-memory errors.
    # And to speed up the computation.

    return distances


def cosine_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Computes the cosine distance between two arrays

    Args:
        A (np.ndarray): Numpy array of shape [num_samples_a x num_features]
        B (np.ndarray): Numpy array of shape [num_samples_b x num_features]

    Returns:
        np.ndarray: Numpy array of shape [num_samples_a x num_samples_b] where
                    each column contains the cosine distance between one element in
                    matrix_b and all elements in matrix_a
    """
    # NOTE: Similar to the euclidean_distance function, you might want to use
    # scikit-learn function to avoid potential out-of-memory errors.
    distances = None

    ### Implement here

    return distances


def get_k_nearest_neighbors(distances: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Gets the k nearest labels based on the distances

    Args:
        distances (np.ndarray): Numpy array of shape num_train_samples x num_test_samples
                                containing the Euclidean distances
        labels (np.ndarray): Numpy array of shape [num_train_samples, ] containing
                                the training labels
        k (int): Number of nearest neighbours

    Returns:
        np.ndarray: Numpy array of shape [k x num_test_samples] containing the
                    training labels of the k nearest neighbours for each test sample
    """

    # Sort the distances in ascending and get the indices of the first "k" elements
    # HINT: You need to sort the distances in ascending order to get the indices
    # of the first "k" elements. BUT, you would not need to sort the entire array,
    # it would be enough to make sure that the "k"-th element is in the correct position!

    # NOTE: Since the matrix sizes are huge, it would be impractical to run any sort of a
    # loop to get the nearest labels. Think about how you can do it without using loops.

    neighbors = None

    ### Implement here, explain also briefly in a comment how you are doing it.
    ### If you use any external function, such as numpy's functions, please explain what the function does.

    return neighbors


def majority_voting(nearest_labels: np.ndarray) -> np.ndarray:
    """Gets the best prediction, i.e. the label class that occurs most frequently

    Args:
        nearest_labels (np.ndarray): Numpy array of shape [k x num_test_samples] obtained from the output of the get_k_neighbors function

    Returns:
        np.array: Numpy array of shape [num_test_samples] containing the best prediction for each test sample. If there are more than one most frequent labels, return the smallest one.
    """

    predicted = None

    ### Implement here

    return predicted


def knn_classifier(
    training_set: np.ndarray,
    training_labels: np.ndarray,
    test_set: np.ndarray,
    test_labels: np.ndarray,
    k: int,
    dist_func: callable,
) -> float:
    """
    Performs k-nearest neighbour classification

    Args:
    training_set (np.ndarray): Vectorized training images (shape: [num_train_samples x num_features])
    training_labels (np.ndarray): Training labels (shape: [num_train_samples, 1])
    test_set (np.ndarray): Vectorized test images (shape: [num_test_samples x num_features])
    test_labels (np.ndarray): Test labels (shape: [num_test_samples, 1])
    k (int): number of nearest neighbours

    Returns:
    accuracy (float): the accuracy in %
    """
    # compute the distance between each test sample and all training samples
    # Cache distances per distance function and dataset pair to reuse across different k calls
    dists = dist_func(training_set, test_set)

    nearest_labels = get_k_nearest_neighbors(distances=dists, labels=training_labels, k=k)

    # from the nearest labels above choose the label classes that occurs most frequently
    predictions = majority_voting(nearest_labels)

    # calculate and return accuracy of the predicitions
    accuracy = (np.equal(predictions, test_labels).sum()) / len(test_set) * 100.0

    return accuracy


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, mean, std = preprocess_mnist_data(
        os.path.join(os.path.dirname(__file__), "data", "MNIST", "mnist_train.csv"),
        os.path.join(os.path.dirname(__file__), "data", "MNIST", "mnist_test.csv"),
    )

    # define the training set and labels
    X_val, y_val = None, None

    print("Training set shape: ", X_train.shape)
    print("Validation set shape: ", X_val.shape)
    print("Test set shape: ", X_test.shape)
    # dictionary to store the k values as keys and the validation accuracies as the values
    val_accuracy_per_k = {}

    for k in [1]:
        print(f"Calculating validation accuracy for k={k}")
        val_accuracy_per_k[k] = None  # TODO complete that line
        print(f"Validation accuracy of {val_accuracy_per_k[k]} % for k={k}")

    best_k = None
    print(f"Best validation accuracy of {val_accuracy_per_k[best_k]} % for k={best_k}")

    print("Running on the test set...")
    test_accuracy = None  # TODO complete that line
    print(test_accuracy)

    # Do the same for cosine distance
