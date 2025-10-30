import os
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
    classes = np.unique(Y)
    prior_probs = []
    means = []
    vars_ = []

    for k in classes:
        X_k = X[Y == k]
        prior_k = X_k.shape[0] / X.shape[0]
        mean_k = np.mean(X_k, axis=0)
        var_k = np.var(X_k, axis=0)
        var_k += smoothing
        prior_probs.append(prior_k)
        means.append(mean_k)
        vars_.append(var_k)
    return np.array(prior_probs), np.array(means), np.array(vars_)


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
    N, d = X.shape
    log_prior = np.log(prior_probs)
    log_likelihood = np.zeros((N, num_classes))

    for k in range(num_classes):
        mu_k = means[k]
        var_k = vars_[k]
        log_pdf = -0.5 * np.sum(
            np.log(2 * np.pi * var_k) + ((X - mu_k) ** 2) / var_k,
            axis=1
        )
        log_likelihood[:, k] = log_prior[k] + log_pdf

    preds = np.argmax(log_likelihood, axis=1)
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
    import numpy as np
    import matplotlib.pyplot as plt
    from data_process import preprocess_mnist_data

    # -----------------------
    # MNIST Evaluation
    # -----------------------
    print("Evaluating on MNIST...")
    X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist, _, _ = preprocess_mnist_data(
        "data/MNIST/mnist_train.csv", "data/MNIST/mnist_test.csv"
    )

    num_classes_mnist = len(np.unique(y_train_mnist))
    priors, means, vars_ = gnb_fit_classifier(X_train_mnist, y_train_mnist)
    y_pred_mnist = gnb_predict(X_test_mnist, priors, means, vars_, num_classes_mnist)
    gnb_acc_mnist = np.mean(y_pred_mnist == y_test_mnist) * 100.0

    print(f"MNIST - GNB accuracy: {gnb_acc_mnist:.2f} %")
    print("\nPer-class error rates (MNIST):")
    for k in range(num_classes_mnist):
        mask = y_test_mnist == k
        class_acc = np.mean(y_pred_mnist[mask] == y_test_mnist[mask])
        class_err = 1 - class_acc
        print(f"Class {k}: error rate = {class_err * 100:.2f} %")

    # -----------------------
    # IRIS Evaluation
    # -----------------------
    print("\nEvaluating on IRIS...")
    train_iris = np.loadtxt("data/iris/iris_train.csv", delimiter=",")
    test_iris = np.loadtxt("data/iris/iris_test.csv", delimiter=",")
    X_train_iris, y_train_iris = train_iris[:, :-1], train_iris[:, -1].astype(int)
    X_test_iris, y_test_iris = test_iris[:, :-1], test_iris[:, -1].astype(int)

    num_classes_iris = len(np.unique(y_train_iris))
    priors_iris, means_iris, vars_iris = gnb_fit_classifier(X_train_iris, y_train_iris)
    y_pred_iris = gnb_predict(X_test_iris, priors_iris, means_iris, vars_iris, num_classes_iris)
    gnb_acc_iris = np.mean(y_pred_iris == y_test_iris) * 100.0

    print(f"IRIS - GNB accuracy: {gnb_acc_iris:.2f} %")

    # -----------------------
    # IRIS Visualization
    # -----------------------
    print("\nGenerating Iris visualization...")
    os.makedirs("images", exist_ok=True)

    plt.figure(figsize=(6, 5))
    for label in np.unique(y_train_iris):
        plt.scatter(
            X_train_iris[y_train_iris == label, 0],
            X_train_iris[y_train_iris == label, 1],
            label=f"Class {label}",
            alpha=0.7
        )
    plt.title("Iris dataset visualization (first two features)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/iris_visualization.png")
    plt.show()



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
