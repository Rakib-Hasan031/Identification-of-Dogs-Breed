"""
Data preprocessing utilities for heart disease prediction.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Load the heart disease dataset."""
    return pd.read_csv(filepath)

def preprocess_data(filepath, test_size=0.2, random_state=42):
    """
    Load and preprocess the heart disease dataset.
    
    Args:
        filepath (str): Path to the dataset
        test_size (float): Proportion of dataset to include in the test split
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Load data
    df = load_data(filepath)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test