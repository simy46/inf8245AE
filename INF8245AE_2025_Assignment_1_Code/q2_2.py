import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from q2_1 import cross_validate_ridge
from q1_2 import load_csv_as_array

def plot_cv_curve(scores, title):
    lambdas = list(scores.keys())
    values = list(scores.values())
    plt.plot(lambdas, values, marker="o")
    plt.xscale("log")
    plt.xlabel("λ")
    plt.ylabel("Validation Score")
    plt.title(title)
    plt.show()

X_train = load_csv_as_array("X_train.csv")
y_train = load_csv_as_array("y_train.csv", flatten=True)

lambda_list = [0.01, 0.1, 1, 10, 100]
metrics = ["MAE", "MaxError", "RMSE"]

results = []
for metric in metrics:
    best_lambda, scores = cross_validate_ridge(X_train, y_train, lambda_list, k=5, metric=metric) # added seed to have consistent results and test better
    best_score = scores[best_lambda]
    results.append({"Metric": metric, "Best λ": best_lambda, "Mean Validation Score": best_score})
    # plot_cv_curve(scores, f"Cross-Validation ({metric})")
    # print(f"Best λ for {metric}: {best_lambda} with score {best_score}")
    # for lam, score in scores.items():
    #     print(f"  λ={lam}: {metric}={score:.4f}")

df = pd.DataFrame(results)
print(df)
