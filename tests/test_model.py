"""
Unit tests for the dog breed classification model.
"""
import unittest
import tensorflow as tf
from src.models.model import create_model
from src.data.preprocessor import preprocess_image

class TestModel(unittest.TestCase):
    def setUp(self):
        """Set up test data and model."""
        self.num_classes = 120
        self.model = create_model(self.num_classes)
        
    def test_model_creation(self):
        """Test if model is created with correct output shape."""
        self.assertIsInstance(self.model, tf.keras.Model)
        self.assertEqual(self.model.output_shape[-1], self.num_classes)
        
    def test_model_compile(self):
        """Test if model compiles successfully."""
        self.assertIsNotNone(self.model.optimizer)
        self.assertIsNotNone(self.model.loss)
        
    def test_image_preprocessing(self):
        """Test image preprocessing shape."""
        # Create dummy image tensor
        dummy_image = tf.random.uniform((100, 100, 3))
        processed_image = preprocess_image(dummy_image)
        
        self.assertEqual(processed_image.shape, (224, 224, 3))
        
if __name__ == '__main__':
    unittest.main()