import numpy as np


# Part (a)
def ridge_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, lamb: float) -> np.ndarray:
    """
    Computes the gradient of Ridge regression loss.
    ∇L(w) = -2/n X^T (y - X w) + 2 λ w
    """
    # WRITE YOUR CODE HERE...


# Part (b)
def learning_rate_exp_decay(eta0: float, t: int, k_decay: float) -> float:
    # WRITE YOUR CODE HERE...



# Part (c)
def learning_rate_cosine_annealing(eta0: float, t: int, T: int) -> float:
    # WRITE YOUR CODE HERE...


# Part (d)
def gradient_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, lamb:float, eta: float) -> np.ndarray:
    # WRITE YOUR CODE HERE...


# Part (e)
def gradient_descent_ridge(X, y, lamb=1.0, eta0=0.1, T=500, schedule="constant", k_decay=0.01):
    # WRITE YOUR CODE HERE...


# Remove the following line if you are not using it:
if __name__ == "__main__":

    # If you want to test your functions, write your code here.
    # If you write it outside this snippet, the autograder will fail!
