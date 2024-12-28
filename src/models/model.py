"""
Model creation and training utilities for dog breed classification.
"""
import tensorflow as tf
import tensorflow_hub as hub
from typing import Tuple

def create_model(num_classes: int = 120) -> tf.keras.Model:
    """
    Create and return the model architecture.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    # Model URL from TensorFlow Hub
    model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5"
    
    # Create model
    model = tf.keras.Sequential([
        hub.KerasLayer(model_url),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    
    return model

def train_model(model: tf.keras.Model,
                train_data: tf.data.Dataset,
                val_data: tf.data.Dataset,
                epochs: int = 100,
                callbacks: list = None) -> tf.keras.callbacks.History:
    """
    Train the model on the provided data.
    
    Args:
        model: Compiled Keras model
        train_data: Training dataset
        val_data: Validation dataset
        epochs: Number of epochs to train
        callbacks: List of Keras callbacks
        
    Returns:
        Training history
    """
    return model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=callbacks
    )

def predict_breed(model: tf.keras.Model,
                 image_path: str,
                 class_names: list) -> Tuple[str, float]:
    """
    Predict dog breed from image.
    
    Args:
        model: Trained model
        image_path: Path to image file
        class_names: List of class names (breeds)
        
    Returns:
        Tuple of (predicted_breed, confidence)
    """
    # Preprocess image
    from ..data.preprocessor import preprocess_image
    image = preprocess_image(image_path)
    image = tf.expand_dims(image, axis=0)
    
    # Make prediction
    predictions = model.predict(image)
    
    # Get predicted class and confidence
    predicted_class = tf.argmax(predictions[0])
    confidence = tf.reduce_max(predictions[0])
    
    return class_names[predicted_class], confidence.numpy()