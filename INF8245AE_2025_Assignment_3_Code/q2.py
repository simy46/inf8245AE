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

RANDOM_SEED=0
TEST_SIZE=0.2
DECISION_TREES='Decision Tree'
RANDOM_FOREST='Random Forest'
SVM='SVM'

def data_splits(X, y):
    """
    Split the 'features' and 'labels' data into training and testing sets.
    Input(s): X: features (pd.DataFrame), y: labels (pd.DataFrame)
    Output(s): X_train, X_test, y_train, y_test
    """
    # Use random_state = 0 in the train_test_split
    # TODO write data split here
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=True)
    return X_train, X_test, y_train, y_test


def normalize_features(X_train, X_test):
    """
    Take the input data and normalize the features.
    Input: X_train: features for train,  X_test: features for test (pd.DataFrame)
    Output: X_train_scaled, X_test_scaled (pd.DataFrame) the same shape of X_train and X_test
    """
    # TODO write normalization here
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled



def train_model(model_name, X_train_scaled, y_train):
    '''
    inputs:
       - model_name: the name of learning algorithm to be trained
       - X_train: features training set
       - y_train: label training set
    output: cls: the trained model
    '''
    if model_name == DECISION_TREES:
        cls = DecisionTreeClassifier(random_state=RANDOM_SEED)

    elif model_name == RANDOM_FOREST:
        cls = RandomForestClassifier(random_state=RANDOM_SEED)

    elif model_name == SVM:
        cls = SVC(random_state=RANDOM_SEED)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    cls.fit(X_train_scaled, y_train)
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
        DECISION_TREES: None,
        RANDOM_FOREST: None,
        SVM: None
    }
    y_test_pred_dict = {
        DECISION_TREES: None,
        RANDOM_FOREST: None,
        SVM: None
    }

    for model_name, model in tqdm(trained_models.items()):
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        y_train_pred_dict[model_name] = y_train_pred
        y_test_pred_dict[model_name] = y_test_pred  

        evaluation_results[model_name] = {
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy,
            'Train F1 Score': train_f1,
            'Test F1 Score': test_f1
        }

    return y_train_pred_dict, y_test_pred_dict, evaluation_results


def report_model(y_train, y_test, y_train_pred_dict, y_test_pred_dict):
    '''
    inputs:
        - y_train: label training set
        - y_test: label test set
        - y_train_pred_dict: a dictionary of label predicted for train set of each model, len(y_train_pred_dict.keys)=3
        - y_test_pred_dict: a dictionary of label predicted for test set of each model, len(y_train_pred_dict.keys)=3
    '''

    for model_name in y_train_pred_dict.keys():
        print(f"\nModel: {model_name}")

        y_train_pred = y_train_pred_dict[model_name]
        y_test_pred = y_test_pred_dict[model_name]

        print("\nTraining Set Classification Report:")
        print(classification_report(y_train, y_train_pred))
        
        print("Training Set Confusion Matrix:")
        print(confusion_matrix(y_train, y_train_pred))

        print("\nTesting Set Classification Report:")
        print(classification_report(y_test, y_test_pred))

        print("Testing Set Confusion Matrix:")
        print(confusion_matrix(y_test, y_test_pred))    

if __name__ == "__main__":
    # TODO call data preprocessing from q1
    X, y = data_preprocessing()
    X_train, X_test, y_train, y_test = data_splits(X, y)
    X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)

    cls_decision_tree = train_model(DECISION_TREES, X_train_scaled, y_train)
    cls_randomforest = train_model(RANDOM_FOREST, X_train_scaled, y_train)
    cls_svm = train_model(SVM, X_train_scaled, y_train)

    # Define a dictionary of model name and their trained model
    trained_models = {
        DECISION_TREES: cls_decision_tree,
        RANDOM_FOREST: cls_randomforest,
        SVM: cls_svm 
    }

    # predict labels and calculate accuracy and F1score
    y_train_pred_dict, y_test_pred_dict, evaluation_results = eval_model(trained_models, X_train_scaled, X_test_scaled, y_train, y_test)

    print("\n=== Evaluation Results ===")
    df_results = pd.DataFrame(evaluation_results).T
    print(df_results.round(4))

    # classification report and calculate confusion matrix
    report_model(y_train, y_test, y_train_pred_dict, y_test_pred_dict)
