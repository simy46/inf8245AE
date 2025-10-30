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
    return smp.euclidean_distances(A, B)


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
    return (1.0 - smp.cosine_similarity(A, B))


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

    # Step 1: Use argpartition to get indices of k smallest distances for each column (test sample).
    # argpartition return indices that partition the array so that k smallest elements
    # are in first k positions (unordered). This is much faster than full sort.
    k_smallest_idx = np.argpartition(distances, kth=k-1, axis=0)[:k, :]

    # Step 2: Sort k neighbors in ascending order of distance.
    # We gather the distances of these k neighbors, then argsort within subset.
    # Ensures neighbors[0] corresponds to closest neighbor.
    k_smallest_distances = np.take_along_axis(distances, k_smallest_idx, axis=0)
    order_within_k = np.argsort(k_smallest_distances, axis=0)
    sorted_k_indices = np.take_along_axis(k_smallest_idx, order_within_k, axis=0)
    return labels[sorted_k_indices]

def _most_frequent_label(labels_for_one_test):
    counts = np.bincount(labels_for_one_test)
    return np.argmax(counts)

def majority_voting(nearest_labels: np.ndarray) -> np.ndarray:
    """Gets the best prediction, i.e. the label class that occurs most frequently

    Args:
        nearest_labels (np.ndarray): Numpy array of shape [k x num_test_samples] obtained from the output of the get_k_neighbors function

    Returns:
        np.array: Numpy array of shape [num_test_samples] containing the best prediction for each test sample. If there are more than one most frequent labels, return the smallest one.
    """
    return np.apply_along_axis(_most_frequent_label, axis=0, arr=nearest_labels)


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

def predict_in_batches(X_train, y_train, X_test, k, distance_fn, batch_size=1000):
    num_test = X_test.shape[0]
    all_predictions = []

    for start in range(0, num_test, batch_size):
        end = min(start + batch_size, num_test)
        X_batch = X_test[start:end]
        dist_batch = distance_fn(X_train, X_batch)
        nearest_labels_batch = get_k_nearest_neighbors(dist_batch, y_train, k)
        y_pred_batch = majority_voting(nearest_labels_batch)
        all_predictions.append(y_pred_batch)

    return np.concatenate(all_predictions, axis=0)


def hyperparameter_search_in_batches(X_train, y_train, X_val, y_val, k_values, distance_fn, batch_size=1000):
    num_val = X_val.shape[0]
    correct_per_k = {k: 0 for k in k_values}

    for start in range(0, num_val, batch_size):
        end = min(start + batch_size, num_val)
        X_batch = X_val[start:end]
        y_batch = y_val[start:end]

        dist_batch = distance_fn(X_train, X_batch)

        for k in k_values:
            nearest_labels = get_k_nearest_neighbors(dist_batch, y_train, k)
            y_pred = majority_voting(nearest_labels)
            correct_per_k[k] += np.sum(y_pred == y_batch)


    val_accuracy_per_k = {k: (correct_per_k[k] / num_val) * 100 for k in k_values}
    best_k = max(val_accuracy_per_k, key=val_accuracy_per_k.get)
    best_acc = val_accuracy_per_k[best_k]

    print(f"[{distance_fn.__name__}] Best validation accuracy: {best_acc:.2f}% (k={best_k})")
    return best_k, val_accuracy_per_k


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, mean, std = preprocess_mnist_data(
        os.path.join(os.path.dirname(__file__), "data", "MNIST", "mnist_train.csv"),
        os.path.join(os.path.dirname(__file__), "data", "MNIST", "mnist_test.csv"),
    )

    X_val = X_train[-10000:]
    y_val = y_train[-10000:]
    X_train_small = X_train[:-10000]
    y_train_small = y_train[:-10000]

    print("Training set shape: ", X_train_small.shape)
    print("Validation set shape: ", X_val.shape)
    print("Test set shape: ", X_test.shape)

    k_values = [1, 2, 3, 4, 5, 10, 20]

    print("\n--- Hyperparameter search (Euclidean) ---")
    dist_val = euclidean_distance(X_train_small, X_val)
    val_accuracy_per_k = {}
    for k in k_values:
        nearest_labels = get_k_nearest_neighbors(dist_val, y_train_small, k)
        y_val_pred = majority_voting(nearest_labels)
        accuracy = np.mean(y_val_pred == y_val) * 100
        val_accuracy_per_k[k] = accuracy
        print(f"Validation accuracy (euclidean) of {accuracy:.2f}% for k={k}")

    best_k = max(val_accuracy_per_k, key=val_accuracy_per_k.get)
    print(f"\nBest validation accuracy (euclidean) = {val_accuracy_per_k[best_k]:.2f}% for k={best_k}")

    print("\nRunning on the test set (Euclidean) in batches...")
    y_test_pred = predict_in_batches(
        X_train_small, y_train_small, X_test, best_k, euclidean_distance, batch_size=1000
    )
    test_accuracy = np.mean(y_test_pred == y_test) * 100
    print(f"Test accuracy (Euclidean, k={best_k}) = {test_accuracy:.2f}%")

    print("\n--- Hyperparameter search (Cosine) in batches ---")
    best_k_cos, val_accuracy_per_k_cos = hyperparameter_search_in_batches(
        X_train_small, y_train_small, X_val, y_val, k_values, cosine_distance, batch_size=1000
    )

    for k in k_values:
        print(f"Validation accuracy (cosine) of {val_accuracy_per_k_cos[k]:.2f}% for k={k}")
    print(f"\nBest validation accuracy (cosine) = {val_accuracy_per_k_cos[best_k_cos]:.2f}% for k={best_k_cos}")

    print("\nRunning on the test set (Cosine) in batches...")
    y_test_pred_cos = predict_in_batches(
        X_train_small, y_train_small, X_test, best_k_cos, cosine_distance, batch_size=1000
    )
    test_accuracy_cos = np.mean(y_test_pred_cos == y_test) * 100
    print(f"Test accuracy (Cosine, k={best_k_cos}) = {test_accuracy_cos:.2f}%")
