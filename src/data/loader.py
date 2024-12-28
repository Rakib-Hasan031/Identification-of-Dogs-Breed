"""
Data loading utilities for the dog breed classification project.
"""
import os
import pandas as pd
import tensorflow as tf
from typing import Tuple, List

def load_labels(labels_path: str) -> pd.DataFrame:
    """Load and return the labels DataFrame."""
    return pd.read_csv(labels_path)

def get_file_paths(data_dir: str) -> List[str]:
    """Get list of image file paths from directory."""
    return [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]

def create_data_batches(file_paths: List[str], 
                       labels=None, 
                       batch_size: int = 32, 
                       valid_data: bool = False, 
                       test_data: bool = False) -> tf.data.Dataset:
    """
    Creates batches of data from image file paths and labels.
    
    Args:
        file_paths: List of image file paths
        labels: Labels corresponding to the images
        batch_size: Number of samples per batch
        valid_data: Whether this is validation data
        test_data: Whether this is test data
        
    Returns:
        tf.data.Dataset of batched images and labels
    """
    # Handle test data (no labels)
    if test_data:
        data = tf.data.Dataset.from_tensor_slices((file_paths))
        data_batch = data.map(preprocess_image).batch(batch_size)
        return data_batch
    
    # Handle training/validation data
    data = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    # Shuffle training data
    if not valid_data:
        data = data.shuffle(buffer_size=len(file_paths))
    
    # Preprocess and batch data
    data = data.map(preprocess_image_and_label)
    data_batch = data.batch(batch_size)
    
    return data_batch

def load_data(train_dir: str, 
              labels_path: str, 
              test_size: float = 0.2) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Load and prepare training and validation data.
    
    Args:
        train_dir: Directory containing training images
        labels_path: Path to labels CSV file
        test_size: Proportion of data to use for validation
        
    Returns:
        Tuple of (train_data, val_data) as TensorFlow datasets
    """
    # Load labels
    labels_df = load_labels(labels_path)
    
    # Get file paths
    file_paths = get_file_paths(train_dir)
    
    # Split data
    train_size = int(len(file_paths) * (1 - test_size))
    train_paths = file_paths[:train_size]
    val_paths = file_paths[train_size:]
    
    # Create data batches
    train_data = create_data_batches(train_paths, labels_df[:train_size])
    val_data = create_data_batches(val_paths, labels_df[train_size:], valid_data=True)
    
    return train_data, val_data