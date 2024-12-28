"""
Visualization utilities for dog breed classification.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_predictions(image: np.ndarray,
                    true_label: str,
                    pred_label: str,
                    pred_prob: float) -> None:
    """
    Plot image with prediction information.
    
    Args:
        image: Image array
        true_label: True class label
        pred_label: Predicted class label
        pred_prob: Prediction probability
    """
    plt.imshow(image)
    plt.axis('off')
    
    # Set title color based on prediction accuracy
    color = "green" if pred_label == true_label else "red"
    
    plt.title(f"Pred: {pred_label} ({pred_prob:.2%})\nTrue: {true_label}",
              color=color)

def plot_training_history(history: dict) -> None:
    """
    Plot training history metrics.
    
    Args:
        history: Training history dictionary
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history['accuracy'], label='training')
    ax1.plot(history['val_accuracy'], label='validation')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Loss
    ax2.plot(history['loss'], label='training')
    ax2.plot(history['val_loss'], label='validation')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_prediction_confidence(pred_probs: np.ndarray,
                             class_names: List[str],
                             true_label: str = None,
                             top_k: int = 10) -> None:
    """
    Plot top prediction confidences.
    
    Args:
        pred_probs: Array of prediction probabilities
        class_names: List of class names
        true_label: True class label (optional)
        top_k: Number of top predictions to show
    """
    # Get top k predictions
    top_indices = pred_probs.argsort()[-top_k:][::-1]
    top_probs = pred_probs[top_indices]
    top_names = [class_names[i] for i in top_indices]
    
    # Create bar plot
    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(top_k), top_probs)
    plt.xticks(range(top_k), top_names, rotation=45, ha='right')
    plt.xlabel('Breed')
    plt.ylabel('Confidence')
    
    # Highlight true label if provided
    if true_label and true_label in top_names:
        idx = top_names.index(true_label)
        bars[idx].set_color('green')
    
    plt.tight_layout()
    plt.show()