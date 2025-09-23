import numpy as np

# Part (a)
def data_matrix_bias(X: np.ndarray) -> np.ndarray:
    """Append a bias column of ones as the first column of X."""
    bias_col = np.ones((X.shape[0], 1))
    return np.hstack((bias_col, X))

# Part (b)
def linear_regression_optimize(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form OLS solution"""
    print(X)
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

if __name__ == "__main__":
    X = np.array([[0, 65], 
                  [3, 44], 
                  [8,  2]])
    y = np.array([11, 22, 33])

    lambda_vec = np.array([0.0, 2.0, 3.0])

    X_aug = data_matrix_bias(X)

    lin_w       = linear_regression_optimize(X_aug, y)
    for lamb in lambda_vec:
        l2_reg  = ridge_regression_optimize(X_aug, y, lamb)   
        print("\n===== Ridge regression (λ = {}) =====".format(lamb))
        print("w:", l2_reg)
        predict_l2  = predict(X_aug, l2_reg)
        print("Prédictions:", predict_l2)
 
    weighted_reg = weighted_ridge_regression_optimize(X_aug, y, lambda_vec)

    # 
    predict_lin = predict(X_aug, lin_w)
    predict_w   = predict(X_aug, weighted_reg)

    print("\n===== Données =====")
    print("X_aug:\n", X_aug, "\n")
    print("y:", y)
    print("lambda (ridge):", lamb)
    print("lambda_vec (weighted ridge):", lambda_vec)

    print("\n===== Régression linéaire (OLS) =====")
    print("w:", lin_w, "\n")
    print("Prédictions:", predict_lin, "\n")

    print("\n===== Weighted ridge regression (λ_vec = {}) =====".format(lambda_vec))
    print("w:", weighted_reg)
    print("Prédictions:", predict_w)
    print("========================\n")