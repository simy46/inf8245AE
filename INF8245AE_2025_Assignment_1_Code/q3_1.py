import numpy as np


# Part (a)
def ridge_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, lamb: float) -> np.ndarray:
    """
    Computes the gradient of Ridge regression loss.
    ∇L(w) = -2/n X^T (y - X w) + 2 λ w
    """
    n_samples = X.shape[0]
    return (-2 / n_samples) * X.T @ (y - X @ w) + 2 * lamb * w


# Part (b)
def learning_rate_exp_decay(eta0: float, t: int, k_decay: float) -> float:
    return eta0 * np.exp(-k_decay * t)



# Part (c)
def learning_rate_cosine_annealing(eta0: float, t: int, T: int) -> float:
    return eta0 * (1 + np.cos(np.pi * t / T)) / 2


# Part (d)
def gradient_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, lamb:float, eta: float) -> np.ndarray:
    return w - eta * ridge_gradient(X, y, w, lamb)


# Part (e)
def ridge_loss(X, y, w, lamb):
    """Compute Ridge regression loss."""
    n = X.shape[0]
    residual = y - X @ w
    return (1 / n) * np.sum(residual ** 2) + lamb * np.sum(w ** 2)


def gradient_descent_ridge(X, y, lamb=1.0, eta0=0.1, T=500, schedule="constant", k_decay=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    losses = []

    for t in range(T):
        if schedule == "constant":
            eta = eta0
        elif schedule == "exp_decay":
            eta = learning_rate_exp_decay(eta0, t, k_decay)
        elif schedule == "cosine":
            eta = learning_rate_cosine_annealing(eta0, t, T)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        w = gradient_step(X, y, w, lamb, eta)
        loss = ridge_loss(X, y, w, lamb)
        losses.append(loss)

    return w, np.array(losses)


# Remove the following line if you are not using it:
if __name__ == "__main__":

    # If you want to test your functions, write your code here.
    # If you write it outside this snippet, the autograder will fail!
    print("Testing gradient_descent_ridge with random data...")
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    w = gradient_descent_ridge(X, y, lamb=0.1, eta0=0.01, T=1000, schedule="cosine")
    print("Learned weights:", w)