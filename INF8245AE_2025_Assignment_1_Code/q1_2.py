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


def load_csv_as_array(path: str, flatten: bool = False) -> np.ndarray:
    arr = pd.read_csv(path, header=None, skiprows=1).to_numpy()
    if flatten:
        arr = arr.ravel()
    return arr


# Write your code here ...
# Not autograded — function names and structure are flexible.

train_x = load_csv_as_array("X_train.csv")
test_x  = load_csv_as_array("X_test.csv")
train_y = load_csv_as_array("y_train.csv", flatten=True)
test_y  = load_csv_as_array("y_test.csv", flatten=True)

train_x_aug = data_matrix_bias(train_x)
test_x_aug = data_matrix_bias(test_x)

lin_pred = linear_regression_optimize(train_x_aug, train_y)
y_hat = predict(test_x_aug, lin_pred)
rmse(test_y, y_hat)
print("OLS RMSE:", rmse(test_y, y_hat))