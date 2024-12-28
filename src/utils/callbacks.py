"""
Custom callbacks for model training.
"""
import os
import datetime
import tensorflow as tf

def create_tensorboard_callback(log_dir: str = "logs") -> tf.keras.callbacks.TensorBoard:
    """
    Create TensorBoard callback for visualizing training metrics.
    
    Args:
        log_dir: Directory to store logs
        
    Returns:
        TensorBoard callback
    """
    log_dir = os.path.join(log_dir, 
                          datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir)

def create_early_stopping_callback(monitor: str = "val_accuracy",
                                 patience: int = 3) -> tf.keras.callbacks.EarlyStopping:
    """
    Create early stopping callback to prevent overfitting.
    
    Args:
        monitor: Metric to monitor
        patience: Number of epochs with no improvement before stopping
        
    Returns:
        EarlyStopping callback
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience
    )

def create_model_checkpoint_callback(model_dir: str = "models",
                                   monitor: str = "val_accuracy") -> tf.keras.callbacks.ModelCheckpoint:
    """
    Create model checkpoint callback to save best models.
    
    Args:
        model_dir: Directory to save model checkpoints
        monitor: Metric to monitor
        
    Returns:
        ModelCheckpoint callback
    """
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "model-{epoch:02d}-{val_accuracy:.2f}.h5"),
        monitor=monitor,
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        save_freq='epoch'
    )