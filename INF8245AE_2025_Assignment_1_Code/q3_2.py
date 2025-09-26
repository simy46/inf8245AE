import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from q3_1 import gradient_descent_ridge
from q1_1 import rmse, predict, data_matrix_bias

def load_csv_as_array(path: str, flatten: bool = False) -> np.ndarray:
    arr = pd.read_csv(path, header=None, skiprows=1).to_numpy()
    if flatten:
        arr = arr.ravel()
    return arr

X_train = load_csv_as_array("X_train.csv")
y_train = load_csv_as_array("y_train.csv", flatten=True)
X_test = load_csv_as_array("X_test.csv")
y_test = load_csv_as_array("y_test.csv", flatten=True)

X_train_aug = data_matrix_bias(X_train)
X_test_aug = data_matrix_bias(X_test)

eta0 = 0.001
k = 0.001
T = 100
lamb = 1.0

schedules = ["constant", "exp_decay", "cosine"]
results = {}

for schedule in schedules:
    w_final, losses = gradient_descent_ridge(
        X_train_aug, y_train, lamb=lamb, eta0=eta0, T=T, schedule=schedule, k_decay=k
    )

    y_pred = predict(X_test_aug, w_final)
    test_rmse = rmse(y_test, y_pred)
    results[schedule] = {"weights": w_final, "losses": losses, "rmse": test_rmse}
    plt.plot(losses, label=f"{schedule} (RMSE={test_rmse:.4f})")

plt.title("Training Loss for Different Learning Rate Schedules")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.legend()
plt.show()

for schedule, res in results.items():
    print(f"{schedule} → RMSE on test set: {res['rmse']:.4f}")
