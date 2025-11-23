import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from q1 import data_preprocessing
from q2 import RANDOM_SEED, data_splits, normalize_features
from sklearn.metrics import make_scorer, accuracy_score
import matplotlib.pyplot as plt
import os


# after q3 ----------------
best_params_decision_tree = {
    'criterion': 'gini',
    'max_depth': None,
    'max_leaf_nodes': 100,
    'min_samples_leaf': 8
}

best_params_random_forest = {
    'bootstrap': False,
    'max_depth': 20,
    'n_estimators': 200
}

best_params_svm = {
    'C': 10,
    'gamma': 'scale',
    'kernel': 'rbf',
    'tol': 0.001
}
# -----------------------

param_grid_decision_tree = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'max_leaf_nodes': [None, 10, 20, 50, 100]
}

param_grid_random_forest = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'bootstrap': [True, False]
}

param_grid_svm = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 1, 10],
    'tol': [1e-4, 1e-3, 1e-2],
    'gamma': ['scale', 'auto']
}

decision_tree = DecisionTreeClassifier(random_state=RANDOM_SEED)
random_forest = RandomForestClassifier(random_state=RANDOM_SEED)
svm = SVC(random_state=RANDOM_SEED)

scorer = "accuracy"

def perform_grid_search(model, X_train, y_train, params):
    strat_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring=scorer,
        cv=strat_kfold,
        n_jobs=1,
        verbose=0,
        error_score='raise'
    )

    grid_search.fit(X_train, y_train)


    best_param = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best parameters are:", best_param)
    print("Best score is:", best_score)

    return grid_search, best_param, best_score

def q4_hyperparameter_plots(X_train_scaled, y_train):
    os.makedirs("images", exist_ok=True)

    print("\n=== Q4: Decision Tree (max_depth) ===")
    dt_depths = [None, 5, 10, 20, 30]
    dt_accuracies = []

    for depth in dt_depths:
        print(f"Training Decision Tree with max_depth={depth} ...")
        model = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_SEED)
        model.fit(X_train_scaled, y_train)
        acc = model.score(X_train_scaled, y_train)
        print(f"  → Accuracy = {acc:.4f}")
        dt_accuracies.append(acc)

    plt.figure(figsize=(6,4))
    plt.plot(["None","5","10","20","30"], dt_accuracies, marker="o")
    plt.xlabel("max_depth")
    plt.ylabel("Training Accuracy")
    plt.title("Decision Tree: Effect of max_depth")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/q4_decision_tree.png")
    plt.close()

    print("\n=== Q4: Random Forest (n_estimators) ===")
    rf_estimators = [50, 100, 200]
    rf_accuracies = []

    for n in rf_estimators:
        print(f"Training Random Forest with n_estimators={n} ...")
        model = RandomForestClassifier(n_estimators=n, random_state=RANDOM_SEED)
        model.fit(X_train_scaled, y_train)
        acc = model.score(X_train_scaled, y_train)
        print(f"  → Accuracy = {acc:.4f}")
        rf_accuracies.append(acc)

    plt.figure(figsize=(6,4))
    plt.plot(rf_estimators, rf_accuracies, marker="o")
    plt.xlabel("n_estimators")
    plt.ylabel("Training Accuracy")
    plt.title("Random Forest: Effect of n_estimators")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/q4_random_forest.png")
    plt.close()

    print("\n=== Q4: SVM (kernel) ===")
    svm_kernels = ["linear", "poly", "rbf"]
    svm_accuracies = []

    for kernel in svm_kernels:
        print(f"Training SVM with kernel='{kernel}' ...")
        model = SVC(kernel=kernel, random_state=RANDOM_SEED)
        model.fit(X_train_scaled, y_train)
        acc = model.score(X_train_scaled, y_train)
        print(f"  → Accuracy = {acc:.4f}")
        svm_accuracies.append(acc)

    plt.figure(figsize=(6,4))
    plt.bar(svm_kernels, svm_accuracies)
    plt.xlabel("kernel")
    plt.ylabel("Training Accuracy")
    plt.title("SVM: Effect of kernel type")
    plt.tight_layout()
    plt.savefig("images/q4_svm.png")
    plt.close()

    print("\n[Q4] All plots saved in ./images/")

def q4_test_accuracy_plot(X_train_scaled, X_test_scaled, y_train, y_test,
                          best_params_dt, best_params_rf, best_params_svm):

    dt_best = DecisionTreeClassifier(**best_params_dt, random_state=RANDOM_SEED)
    rf_best = RandomForestClassifier(**best_params_rf, random_state=RANDOM_SEED)
    svm_best = SVC(**best_params_svm, random_state=RANDOM_SEED)

    dt_best.fit(X_train_scaled, y_train)
    rf_best.fit(X_train_scaled, y_train)
    svm_best.fit(X_train_scaled, y_train)

    dt_acc = dt_best.score(X_test_scaled, y_test)
    rf_acc = rf_best.score(X_test_scaled, y_test)
    svm_acc = svm_best.score(X_test_scaled, y_test)

    print("\n=== Test Accuracies Using Best Hyperparameters ===")
    print(f"Decision Tree: {dt_acc:.4f}")
    print(f"Random Forest: {rf_acc:.4f}")
    print(f"SVM: {svm_acc:.4f}")

    model_names = ["Decision Tree", "Random Forest", "SVM"]
    accuracies = [dt_acc, rf_acc, svm_acc]

    plt.figure(figsize=(6, 4))
    plt.bar(model_names, accuracies, color=["orange", "green", "blue"])
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy of Models (Best Hyperparameters)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("images/q4_test_accuracy.png")
    plt.close()

    print("\n[Q4] Test accuracy plot saved to images/q4_test_accuracy.png")

if __name__ == "__main__":
    X, y = data_preprocessing()
    X_train, X_test, y_train, y_test = data_splits(X, y)
    X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)

    grid_decision_tree, best_params_decision_tree, best_score_decision_tree  = perform_grid_search(decision_tree, X_train_scaled, y_train, param_grid_decision_tree)
    grid_random_forest, best_params_random_forest, best_score_random_forest  = perform_grid_search(random_forest, X_train_scaled, y_train, param_grid_random_forest)
    grid_svm, best_params_svm, best_score_svm = perform_grid_search(svm, X_train_scaled, y_train, param_grid_svm)
    # q4_hyperparameter_plots(X_train_scaled, y_train)
    # q4_test_accuracy_plot(
    #     X_train_scaled, X_test_scaled, y_train, y_test,
    #     best_params_decision_tree,
    #     best_params_random_forest,
    #     best_params_svm
    # )

