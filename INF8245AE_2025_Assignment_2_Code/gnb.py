import numpy as np
from scipy.stats import multivariate_normal
import typing


# -----------------------
# Gaussian Naive Bayes
# -----------------------
def gnb_fit_classifier(X: np.ndarray, Y: np.ndarray, smoothing: float = 1e-3) -> typing.Tuple:
    """
    Fits the GNB classifier on the training data
    """
    prior_probs = []
    means = []
    vars_ = []
    # Your implementation here
    return prior_probs, means, vars_


def gnb_predict(
    X: np.ndarray,
    prior_probs: typing.List[float],
    means: typing.List[np.ndarray],
    vars_: typing.List[np.ndarray],
    num_classes: int,
) -> np.ndarray:
    """
    Computes predictions from the GNB classifier
    """
    log_probs = None
    preds = None
    # Your implementation here
    return preds


def gnb_classifier(train_set, train_labels, test_set, test_labels, smoothing=1e-3):
    """
    Runs GNB classifier and computes accuracy
    """
    num_classes = len(np.unique(train_labels))
    priors, means, vars_ = gnb_fit_classifier(train_set, train_labels, smoothing)
    y_pred = gnb_predict(test_set, priors, means, vars_, num_classes)
    accuracy = np.mean(y_pred == test_labels) * 100.0
    return accuracy


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    from data_process import preprocess_mnist_data
    import numpy as np

    # MNIST dataset (from CSVs prepared by data_download.py)
    X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist, _, _ = preprocess_mnist_data(
        "data/MNIST/mnist_train.csv", "data/MNIST/mnist_test.csv"
    )

    print("Evaluating on MNIST...")
    gnb_acc_mnist = gnb_classifier(X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist)
    print(f"MNIST - GNB accuracy: {gnb_acc_mnist:.2f} %")

    # IRIS dataset (CSV created by data_download.py): last column is label
    train_iris = np.loadtxt("data/iris/iris_train.csv", delimiter=",")
    test_iris = np.loadtxt("data/iris/iris_test.csv", delimiter=",")
    X_train_iris, y_train_iris = train_iris[:, :-1], train_iris[:, -1].astype(int)
    X_test_iris, y_test_iris = test_iris[:, :-1], test_iris[:, -1].astype(int)

    print("\nEvaluating on IRIS...")
    gnb_acc_iris = gnb_classifier(X_train_iris, y_train_iris, X_test_iris, y_test_iris)
    print(f"IRIS - GNB accuracy: {gnb_acc_iris:.2f} %")


# # -----------------------
# # Quadratic Discriminant Analysis # You are lucky you don't have to do anything about this!
# # -----------------------
# def qda_fit_model(X: np.ndarray, Y: np.ndarray, reg: float = 1e-3) -> typing.Tuple:
#     """
#     Fit QDA model: compute mu_k and full covariance Sigma_k per class
#     """
#     priors, means, covariances = [], [], []
#     return priors, means, covariances


# def qda_predict(
#     X: np.ndarray, priors: typing.List[float], means: typing.List[np.ndarray], covariances: typing.List[np.ndarray]
# ) -> np.ndarray:
#     """
#     Computes predictions from a QDA classifier
#     """
#     log_probs = None
#     preds = None
#     return preds


# def qda_classifier(train_set, train_labels, test_set, test_labels, reg=1e-3):
#     """
#     Run QDA classifier and return accuracy
#     """
#     priors, means, covariances = qda_fit_model(train_set, train_labels, reg)
#     y_pred = qda_predict(test_set, priors, means, covariances)
#     accuracy = np.mean(y_pred == test_labels) * 100.0
#     return accuracy
