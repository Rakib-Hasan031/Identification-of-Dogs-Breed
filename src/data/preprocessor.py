"""
Image preprocessing utilities for dog breed classification.
"""
import tensorflow as tf

IMG_SIZE = 224

def preprocess_image(image_path: str) -> tf.Tensor:
    """
    Preprocess a single image file path into a tensor.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Preprocessed image tensor
    """
    # Read image file
    image = tf.io.read_file(image_path)
    
    # Decode JPEG
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Convert to float32 in [0, 1] range
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # Resize
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    
    return image

def preprocess_image_and_label(image_path: str, label) -> tuple:
    """
    Preprocess image and return with label.
    
    Args:
        image_path: Path to image file
        label: Label for the image
        
    Returns:
        Tuple of (preprocessed_image, label)
    """
    return preprocess_image(image_path), label