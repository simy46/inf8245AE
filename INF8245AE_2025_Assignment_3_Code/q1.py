# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from ucimlrepo import fetch_ucirepo
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from utils import avg_age_fn, wamen_perc, rich_ppl_perc, missing_value_perc, scatter_hours_and_age, prettier_print

def downloading_data():
    ''' Downloading data'''
    # fetch dataset
    adult_income = fetch_ucirepo(id=2)
    # data (as pandas dataframes)
    X = adult_income.data.features # shape = (48842, 12)
    y = adult_income.data.targets # shape = (48842, 12)

    # Replace '?' values with NaN
    X = X.replace('?', np.nan)

    # Make labels uniform
    y = y.replace('<=50K.', '<=50K')
    y = y.replace('>50K.', '>50K')

    # metadata
    # print(adult_income.metadata)
    # variable information
    # print(adult_income.variables)

    # Remove the 'fnlwgt' column
    X = X.drop(columns=['fnlwgt', 'education'])

    # print(X.shape)

    return X, y


def data_exploration(X, y):
    """
    Using the data provided calculate: 
    - avg_age --> average age of the individuals, 
    - women_percent --> Percentage of women in the dataset (0-100 %),
    - income_percent --> Percentage of individuals earning more than $50K (0-100 %),
    - missing_values_percent --> Percentage of missing values in the dataset (0-100 %),

    Output: (n_records, n_subscriber, subscriber_percent) -> Tuple of integers
    """
    # TODO : write your code here to calculate the averages and percentages
    avg_age = avg_age_fn(X)
    women_percent = wamen_perc(X)
    income_percent = rich_ppl_perc(y)
    missing_values_percent = missing_value_perc(X)
    
    # TODO: plot a scatter plot with features "hours-per-week" and "age" on the two axes, with samples labeled according to this task (i.e.,  $>\$50K$ or $\leq\$50K$)
    # Hint: Use plt.scatter() with appropriate parameters
    scatter_hours_and_age(X, y)

    return avg_age, women_percent, income_percent, missing_values_percent

def data_imputation(X):
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns

    imputer_cat = SimpleImputer(strategy='most_frequent')
    imputer_num = SimpleImputer(strategy='mean')

    X[categorical_cols] = pd.DataFrame(
        imputer_cat.fit_transform(X[categorical_cols]),
        columns=categorical_cols,
        index=X.index
    )
    X[numerical_cols] = pd.DataFrame(
        imputer_num.fit_transform(X[numerical_cols]),
        columns=numerical_cols,
        index=X.index
    )

    return X

def feature_encoding(X):
    """
    One-hot encode the 'features'.
    Input: X: features (pd.DataFrame)
    Output: X: features_encoded (pd.DataFrame)
    """
    # TODO : write encoding here
    # Hint: Identify categorical columns and use OneHotEncoder
    # Keep numerical columns and concatenate with encoded categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns

    encoder = OneHotEncoder(sparse_output=False, drop=None, handle_unknown='ignore')
    encoded = encoder.fit_transform(X[categorical_cols])

    X_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    X_final = pd.concat([X[numerical_cols].reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)
    # print(f"Final number of features after encoding: {X_final.shape[1]}")
    return X_final


def encode_label(y):
    """
    Encode the 'labels' data to numerical values.
    Input: y: labels (pd.DataFrame) with shape = (48842, 1)
    Output: y: labels_int (pd.DataFrame) with shape = (48842, 1)
    """
    # TODO : write encoding here
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y.values.ravel())
    return pd.Series(y_encoded, name='income')



def data_preprocessing():
    # First download data
    X, y = downloading_data()
    # convert categorical to numerical

    X = data_imputation(X)
    X = feature_encoding(X)
    y = encode_label(y)
    return X, y


if __name__ == "__main__":
    X, y = data_preprocessing()
    # should be inside data_preprocessing, as we have to plot BEFORE encoding data
    # avg_age, women_percent, income_percent, missing_values_percent = data_exploration(X, y)
    # prettier_print(avg_age, women_percent, income_percent, missing_values_percent)

