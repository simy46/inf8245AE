import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from q1_1 import (
    data_matrix_bias,
    linear_regression_optimize,
    ridge_regression_optimize,
    weighted_ridge_regression_optimize,
    predict,
    rmse
)

def plot_predictions(y_true, y_pred, title):
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             color='red', linestyle='--')
    plt.xlabel("Actual y")
    plt.ylabel("Predicted y")
    plt.title(title)
    plt.show()

def load_csv_as_array(path: str, flatten: bool = False) -> np.ndarray:
    arr = pd.read_csv(path, header=None, skiprows=1).to_numpy()
    if flatten:
        arr = arr.ravel()
    return arr

train_x = load_csv_as_array("X_train.csv")
test_x  = load_csv_as_array("X_test.csv")
train_y = load_csv_as_array("y_train.csv", flatten=True)
test_y  = load_csv_as_array("y_test.csv", flatten=True)
lambda_vec = np.array([0.01, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3])

train_x_aug = data_matrix_bias(train_x)
test_x_aug = data_matrix_bias(test_x)

lin_pred = linear_regression_optimize(train_x_aug, train_y)
ridge_pred = ridge_regression_optimize(train_x_aug, train_y, 1.0)
weighted_ridge_pred = weighted_ridge_regression_optimize(train_x_aug, train_y, lambda_vec)

lin_y_hat = predict(test_x_aug, lin_pred)
print("OLS RMSE:", rmse(test_y, lin_y_hat))

ridge_y_hat = predict(test_x_aug, ridge_pred)
print("Ridge RMSE:", rmse(test_y, ridge_y_hat))

weighted_ridge_y_hat = predict(test_x_aug, weighted_ridge_pred)
print("Weighted Ridge RMSE:", rmse(test_y, weighted_ridge_y_hat))

plot_predictions(test_y, lin_y_hat, "OLS: Predicted vs Actual")
plot_predictions(test_y, ridge_y_hat, "Ridge (λ=1.0): Predicted vs Actual")
plot_predictions(test_y, weighted_ridge_y_hat, "Weighted Ridge: Predicted vs Actual")
