import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def avg_age_fn(X):
    return X["age"].mean()

def wamen_perc(X):
    if "sex" in X.columns:
        return (X["sex"].str.strip().str.lower().eq("female").mean()) * 100
    return np.nan


def rich_ppl_perc(y):
    if isinstance(y, pd.DataFrame):
        y = y.squeeze()
    return (y.eq(">50K").mean()) * 100


def missing_value_perc(X):
    return (X.isnull().sum().sum() / X.size) * 100

def scatter_hours_and_age(X, y):
    if {"hours-per-week", "age"}.issubset(X.columns):
        plt.figure(figsize=(7,5))

        y_series = y.squeeze()
        colors = y_series.eq(">50K").map({True: "green", False: "red"})

        plt.scatter(
            X["age"],
            X["hours-per-week"],
            c=colors,
            alpha=0.6
        )
        plt.xlabel("Age")
        plt.ylabel("Hours per week")
        plt.title("Relation entre l'âge et les heures de travail par revenu")
        plt.show()
    else:
        print("Colonnes nécessaires manquantes : 'age' ou 'hours-per-week'.")


def prettier_print(avg_age, women_percent, income_percent, missing_values_percent):
    print("\n" + "="*50)
    print("DATA EXPLORATION SUMMARY")
    print("="*50)
    print(f"Average age: {avg_age:.2f} years")
    print(f"Percentage of women: {women_percent:.2f}%")
    print(f"Percentage earning > $50K: {income_percent:.2f}%")
    print(f"Missing values: {missing_values_percent:.2f}%")
    print("="*50 + "\n")
