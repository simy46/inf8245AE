import numpy as np
from q1_1 import rmse, ridge_regression_optimize, data_matrix_bias


# Part (a)
def cv_splitter(X, y, k):
    """
    Splits data into k folds for cross-validation.
    Returns a list of tuples: (X_train_fold, y_train_fold, X_val_fold, y_val_fold)
    """
    # WRITE YOUR CODE HERE...



# Part (b)
def MAE(y, y_hat):
    # WRITE YOUR CODE HERE...



def MaxError(y, y_hat):
    # WRITE YOUR CODE HERE...




# Part (c)
def cross_validate_ridge(X, y, lambda_list, k, metric):
    """
    Performs k-fold CV over lambda_list using the given metric.
    metric: one of "MAE", "MaxError", "RMSE"
    Returns the lambda with best average score and a dictionary of mean scores.
    """
    # WRITE YOUR CODE HERE...



# Remove the following line if you are not using it:
if __name__ == "__main__":
    # If you want to test your functions, write your code here.
    # If you write it outside this snippet, the autograder will fail!
