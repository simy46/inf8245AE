import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from q1 import data_preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


def data_splits(X, y):
    """
    Split the 'features' and 'labels' data into training and testing sets.
    Input(s): X: features (pd.DataFrame), y: labels (pd.DataFrame)
    Output(s): X_train, X_test, y_train, y_test
    """
    # Use random_state = 0 in the train_test_split
    # TODO write data split here
    X_train, X_test, y_train, y_test = None, None, None, None

    return X_train, X_test, y_train, y_test


def normalize_features(X_train, X_test):
    """
    Take the input data and normalize the features.
    Input: X_train: features for train,  X_test: features for test (pd.DataFrame)
    Output: X_train_scaled, X_test_scaled (pd.DataFrame) the same shape of X_train and X_test
    """
    # TODO write normalization here
    # Hint: Use MinMaxScaler, fit on training data, transform both train and test
    X_train_scaled, X_test_scaled = None, None
    return X_train_scaled, X_test_scaled


def train_model(model_name, X_train_scaled, y_train):
    '''
    inputs:
       - model_name: the name of learning algorithm to be trained
       - X_train: features training set
       - y_train: label training set
    output: cls: the trained model
    '''
    if model_name == 'Decision Tree':
        # TODO call classifier here
        cls = None
    elif model_name == 'Random Forest':
        # TODO call classifier here
        cls = None
    elif model_name == 'SVM':
        # TODO call classifier here
        cls = None

    # TODO train the model
    # cls.fit(...)

    return cls


def eval_model(trained_models, X_train, X_test, y_train, y_test):
    '''
    inputs:
       - trained_models: a dictionary of the trained models,
       - X_train: features training set
       - X_test: features test set
       - y_train: label training set
       - y_test: label test set
    outputs:
        - y_train_pred_dict: a dictionary of label predicted for train set of each model
        - y_test_pred_dict: a dictionary of label predicted for test set of each model
        - a dict of accuracy and f1_score of train and test sets for each model
    '''
    evaluation_results = {}
    y_train_pred_dict = {
        'Decision Tree': None,
        'Random Forest': None,
        'SVM': None}
    y_test_pred_dict = {
        'Decision Tree': None,
        'Random Forest': None,
        'SVM': None}

    # Loop through each trained model
    for model_name, model in tqdm(trained_models.items()):
        # Predictions for training and testing sets
        # TODO predict y
        y_train_pred = None
        # TODO predict y
        y_test_pred = None
        # Calculate accuracy
        # TODO find accuracy
        train_accuracy = None
        # TODO find accuracy
        test_accuracy = None
        # Calculate F1-score
        # TODO find f1_score
        train_f1 = None
        # TODO find f1_score
        test_f1 = None
        # Store predictions
        # TODO
        y_train_pred_dict[model_name] = None
        # TODO
        y_test_pred_dict[model_name] = None  
        # Store the evaluation metrics
        evaluation_results[model_name] = {
            'Train Accuracy': None,
            'Test Accuracy': None,
            'Train F1 Score': None,
            'Test F1 Score': None
        }
    # Return the evaluation results
    return y_train_pred_dict, y_test_pred_dict, evaluation_results


def report_model(y_train, y_test, y_train_pred_dict, y_test_pred_dict):
    '''
    inputs:
        - y_train: label training set
        - y_test: label test set
        - y_train_pred_dict: a dictionary of label predicted for train set of each model, len(y_train_pred_dict.keys)=3
        - y_test_pred_dict: a dictionary of label predicted for test set of each model, len(y_train_pred_dict.keys)=3
    '''

    # Loop through each trained model
    for model_name in y_train_pred_dict.keys():
        print(f"\nModel: {model_name}")

        # Predictions for training and testing sets
        # TODO complete it
        y_train_pred = None
        # TODO complete it
        y_test_pred = None
        # Print classification report for training set
        print("\nTraining Set Classification Report:")
        # TODO write Classification Report train
        print(classification_report(y_train, y_train_pred))

        # Print confusion matrix for training set
        print("Training Set Confusion Matrix:")
        # TODO write Confusion Matrix train
        print(confusion_matrix(y_train, y_train_pred))
        # Print classification report for testing set
        print("\nTesting Set Classification Report:")
        # TODO write Classification Report test
        print(classification_report(y_test, y_test_pred))
        # Print confusion matrix for testing set
        print("Testing Set Confusion Matrix:")
        # TODO write Confusion Matrix test
        print(confusion_matrix(y_test, y_test_pred))    

if __name__ == "__main__":
    # TODO call data preprocessing from q1
    X, y = None, None
    X_train, X_test, y_train, y_test = data_splits(X, y)
    X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)

    cls_decision_tree = train_model('Decision Tree', X_train_scaled, y_train)
    cls_randomforest = train_model('Random Forest', X_train_scaled, y_train)
    cls_svm = train_model('SVM', X_train_scaled, y_train)

    # Define a dictionary of model name and their trained model
    trained_models = {
            'Decision Tree': cls_decision_tree,
            'Random Forest': cls_randomforest,
            'SVM': cls_svm }

    # predict labels and calculate accuracy and F1score
    y_train_pred_dict, y_test_pred_dict, evaluation_results = eval_model(trained_models, X_train_scaled, X_test_scaled, y_train, y_test)

    # classification report and calculate confusion matrix
    report_model(y_train, y_test, y_train_pred_dict, y_test_pred_dict)
