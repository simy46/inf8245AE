import numpy as np
from q1_1 import predict, rmse, ridge_regression_optimize, data_matrix_bias


import numpy as np

def cv_splitter(X, y, k, seed=None):
    """
    Splits data into k folds for cross-validation.
    Returns a list of tuples: (X_train_fold, y_train_fold, X_val_fold, y_val_fold)
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    X, y = X[indices], y[indices]
    base_size = n // k 
    remainder = n % k    
    splits = []
    cur = 0
    fold_sizes = np.full(k, base_size, dtype=int)
    for i in range(remainder):
        fold_sizes[i] += 1
    for fold_size in fold_sizes:
        start, stop = cur, cur + fold_size
        val_idx = np.arange(start, stop)
        train_idx = np.concatenate([np.arange(0, start), np.arange(stop, n)])

        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_val_fold, y_val_fold     = X[val_idx], y[val_idx]

        splits.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
        cur = stop

    return splits


# Part (b)
def MAE(y, y_hat):
    return np.abs(y - y_hat).mean()


def MaxError(y, y_hat):
    return np.abs(y - y_hat).max()


def get_metric_fn(metric):
    match metric:
        case "MAE":
            return MAE
        case "MaxError":
            return MaxError
        case "RMSE":
            return rmse

# Part (c)
def cross_validate_ridge(X, y, lambda_list, k, metric, seed=None): #! No seed
    """
    Performs k-fold CV over lambda_list using the given metric.
    metric: one of "MAE", "MaxError", "RMSE"
    Returns the lambda with best average score and a dictionary of mean scores.
    """
    
    metric_fn = get_metric_fn(metric)

    scores = {}

    for lam in lambda_list:
        fold_errors = []
        folds = cv_splitter(X, y, k, seed=seed)

        for X_train, y_train, X_val, y_val in folds:
            X_train_aug = data_matrix_bias(X_train)
            X_val_aug   = data_matrix_bias(X_val)
            w = ridge_regression_optimize(X_train_aug, y_train, lam)
            y_hat = predict(X_val_aug, w)
            err = metric_fn(y_val, y_hat)
            fold_errors.append(err)

        scores[lam] = np.mean(fold_errors)

    best_lambda = min(scores, key=scores.get)
    return best_lambda, scores