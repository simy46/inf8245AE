import pandas as pd
import numpy as np


def preprocess_mnist_data(train_file_path: str, test_file_path: str):
    """Preprocess the MNIST data from CSV files.
    Args:
        train_file_path (str): Path to the training CSV file.
        test_file_path (str): Path to the testing CSV file.

    Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]: Preprocessed training and testing data, and their statistics.

    Expected types/shapes:
        - X_train: (60000, 784), dtype=np.float16, to save on memory
        - y_train: (60000,)
        - X_test: (10000, 784), dtype=np.float16
        - y_test: (10000,)
        - mean: float
        - std: float
    """
    X_train, y_train, X_test, y_test = None, None, None, None
    mean, std = None, None

    train_df = pd.read_csv(train_file_path, header=None)
    test_df = pd.read_csv(test_file_path, header=None)

    y_train = train_df.iloc[:, 0].to_numpy(dtype=np.int64)
    y_test = test_df.iloc[:, 0].to_numpy(dtype=np.int64)

    train_pixels = train_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    test_pixels = test_df.iloc[:, 1:].to_numpy(dtype=np.float32)

    pixel_mean = train_pixels.mean()
    pixel_std = train_pixels.std()

    train_pixels = (train_pixels - pixel_mean) / pixel_std
    test_pixels = (test_pixels - pixel_mean) / pixel_std

    X_train = train_pixels.astype(np.float32)
    X_test = test_pixels.astype(np.float32)

    return X_train, y_train, X_test, y_test, pixel_mean, pixel_std


def preprocess_credit_card(train_file_path: str, test_file_path: str):
    """
        This function should be very similar to the preprocess mnist data function except that we
        have a header, the classes column has name Class and you can discard the column:
    • Open the pair of csv files as input using pandas, keeping the header and using
    column id as our index
    • extract the column Class for both,
    • perform scaling column wise: substract, for each column, its mean and divide
    by its standard deviation, as computed on the train set.
    • return the result as numpy arrays
        Args:
            train_file_path (str): Path to the training CSV file.
            test_file_path (str): Path to the testing CSV file.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]: Preprocessed training and testing data, and their statistics.
        Expected types/shapes:
            - X_train: (n_samples_train, n_features), dtype=np.float32
            - y_train: (n_samples_train,)
            - X_test: (n_samples_test, n_features), dtype=np.float32
            - y_test: (n_samples_test,)
            - mean: (n_features,), dtype=np.float32
            - std: (n_features,), dtype=np.float32
    """

    train_df = pd.read_csv(train_file_path, index_col="id")
    test_df = pd.read_csv(test_file_path, index_col="id")
    y_train = train_df["Class"].to_numpy(dtype=np.int64)
    y_test = test_df["Class"].to_numpy(dtype=np.int64)
    X_train_df = train_df.drop(columns=["Class"])
    X_test_df = test_df.drop(columns=["Class"])
    mean = X_train_df.mean(axis=0).to_numpy(dtype=np.float32)
    std = X_train_df.std(axis=0, ddof=0).to_numpy(dtype=np.float32)
    std_safe = np.where(std == 0, 1.0, std)
    X_train = ((X_train_df.to_numpy(dtype=np.float32) - mean) / std_safe).astype(np.float32)
    X_test = ((X_test_df.to_numpy(dtype=np.float32) - mean) / std_safe).astype(np.float32)
    return X_train, y_train, X_test, y_test, mean, std_safe