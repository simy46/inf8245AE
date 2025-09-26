import numpy as np

# Part (a)
def data_matrix_bias(X: np.ndarray) -> np.ndarray:
    """Append a bias column of ones as the first column of X."""
    bias_col = np.ones((X.shape[0], 1))
    return np.hstack((bias_col, X))

# Part (b)
def linear_regression_optimize(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form OLS solution"""
    X_T_dot_X = X.T @ X
    inv_X = np.linalg.inv(X_T_dot_X) #? verify if inversible?
    X_T_dot_y = X.T @ y
    return inv_X @ X_T_dot_y

# Part (c)
def ridge_regression_optimize(X: np.ndarray, y: np.ndarray, lamb: float) -> np.ndarray:
    """Closed-form Ridge regression."""
    X_T_dot_X = X.T @ X
    lamb_I = lamb*np.identity(X_T_dot_X.shape[0]) # known inversible if lamb!=0
    add = X_T_dot_X + lamb_I
    inv = np.linalg.inv(add)
    X_T_dot_y = X.T @ y
    return inv @ X_T_dot_y

# Part (e)
def weighted_ridge_regression_optimize(X: np.ndarray, y: np.ndarray, lambda_vec: np.ndarray) -> np.ndarray:
    """Weighted Ridge regression solution."""
    X_T_dot_X = X.T @ X
    Lambda = np.diag(lambda_vec)
    add = X_T_dot_X + Lambda
    inv = np.linalg.inv(add)
    X_T_dot_y = X.T @ y
    return inv @ X_T_dot_y
    

# Part (f)
def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute predictions: y_hat = X w"""
    y_hat = X @ w
    return y_hat.ravel()

# Part (f)
def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    y = y.ravel()
    y_hat = y_hat.ravel()
    expr = np.mean((y - y_hat) ** 2)
    return np.sqrt(expr)