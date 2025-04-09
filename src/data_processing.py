import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self, file_path, output_dir, selected_columns, target_column, feature_columns):
        """
        Initialize the DataProcessor class with the necessary parameters.

        Args:
            file_path (str): Path to the input data file.
            output_dir (str): Directory to save the processed data.
            selected_columns (list): List of columns to retain from the dataset.
            target_column (str): Name of the target column for prediction.
            feature_columns (list): List of feature column names to use for training.
        """
        self.file_path = file_path
        self.output_dir = output_dir
        self.selected_columns = selected_columns
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.columns_to_keep = feature_columns + [target_column]  # Combine features and target for saving later
        self.data = None  # Placeholder for the loaded dataset
        
    def load_data(self):
        """
        Load the dataset from the specified file path and select the relevant columns.

        Returns:
            pd.DataFrame: The loaded dataset with only selected columns.
        """
        self.data = pd.read_csv(self.file_path)[self.selected_columns]  # Read the file and filter columns
        return self.data
    
    def clean_data(self, z_threshold=3):
        """
        Clean the dataset by removing rows with missing values and outliers based on z-scores.

        Args:
            z_threshold (float): Threshold for identifying outliers using z-scores. Default is 3.

        Returns:
            pd.DataFrame: The cleaned dataset with no missing values and fewer outliers.
        """
        self.data = self.data.dropna()  # Drop rows with missing values
        
        # Calculate z-scores for all numeric columns to detect outliers
        numeric_data = self.data.select_dtypes(include=[np.number])  
        z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())  
        outliers = (z_scores > z_threshold).any(axis=1)  # Identify rows with any column exceeding z-threshold
        self.data = self.data[~outliers]  # Keep rows without outliers
        
        return self.data
    
    def normalize_data(self):
        """
        Normalize the feature columns using StandardScaler to standardize the data.

        Returns:
            pd.DataFrame: Dataset with normalized feature values.
        """
        scaler = StandardScaler()  # Initialize StandardScaler
        self.data[self.feature_columns] = scaler.fit_transform(self.data[self.feature_columns])  # Normalize features
        return self.data
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the dataset into training and testing sets.

        Args:
            test_size (float): Proportion of the dataset to be used as the test set. Default is 0.2.
            random_state (int): Random seed for reproducibility. Default is 42.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Training and testing splits for features and target.
        """
        x = self.data[self.feature_columns]  # Feature matrix
        y = self.data[self.target_column]  # Target variable
        # Perform train-test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test
    
    def save_preprocessed_data(self, save_path, x_train, y_train, x_test, y_test):
        """
        Save the entire preprocessed dataset (both training and testing data) to a single CSV file.

        Args:
            save_path (str): Path to save the combined preprocessed dataset.
            x_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            x_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing target.
        """
        # Combine training and testing data
        train_data = np.hstack((x_train, y_train.values.reshape(-1, 1)))  # Stack training data horizontally
        test_data = np.hstack((x_test, y_test.values.reshape(-1, 1)))  # Stack testing data horizontally
        combined_data = np.vstack((train_data, test_data))  # Combine training and testing data vertically
        # Save as a CSV
        combined_df = pd.DataFrame(combined_data, columns=self.columns_to_keep)
        combined_df.to_csv(save_path, index=False)
    
    def save_preprocessed_data_splitted(self, train_path, test_path, x_train, y_train, x_test, y_test):
        """
        Save the preprocessed data into separate training and testing CSV files.

        Args:
            train_path (str): Path to save the training data.
            test_path (str): Path to save the testing data.
            x_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            x_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing target.
        """
        # Combine training and testing data
        train_data = np.hstack((x_train, y_train.values.reshape(-1, 1)))
        test_data = np.hstack((x_test, y_test.values.reshape(-1, 1)))
        
        # Save training data to a CSV
        train_df = pd.DataFrame(train_data, columns=self.columns_to_keep)
        train_df.to_csv(train_path, index=False)
        
        # Save testing data to a CSV
        test_df = pd.DataFrame(test_data, columns=self.columns_to_keep)
        test_df.to_csv(test_path, index=False)
        
    def process_data(self, save_path, save_train_path, save_test_path):
        """
        Complete data processing pipeline:
        - Load the data
        - Clean the data (handle missing values and remove outliers)
        - Normalize feature values
        - Split the data into training and testing sets
        - Save the processed data to CSV files

        Args:
            save_path (str): Path to save the combined processed data.
            save_train_path (str): Path to save the training data.
            save_test_path (str): Path to save the testing data.
        """
        print("Loading data...")
        self.load_data()
        print("Data loaded successfully.")
        print("Data shape:", self.data.shape)

        self.clean_data()  # Handle missing values and outliers
        print("Missing values handled successfully.")

        self.normalize_data()  # Normalize features
        print("Features normalized successfully.")

        # Split the data
        x_train, x_test, y_train, y_test = self.split_data()
        print("Data split into training and testing sets successfully.")
        print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

        # Save preprocessed datasets
        self.save_preprocessed_data(save_path, x_train, y_train, x_test, y_test)
        self.save_preprocessed_data_splitted(save_train_path, save_test_path, x_train, y_train, x_test, y_test)
