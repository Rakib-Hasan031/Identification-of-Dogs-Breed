# Identification of Dogs Breed

An end-to-end multi-class dog breed classification project using TensorFlow 2.0 and TensorFlow Hub.

## Project Overview

This project aims to classify dog breeds from images using deep learning. The model can identify 120 different dog breeds with approximately 88.5% accuracy using transfer learning with MobileNetV2.

### Features
- Multi-class image classification
- Transfer learning using MobileNetV2
- TensorBoard integration for model monitoring
- Custom image prediction support
- Kaggle competition submission format

## Dataset

The dataset is from Kaggle's dog breed identification competition:
- Training set: ~10,000+ labeled images of 120 different dog breeds
- Test set: ~10,000+ unlabeled images
- Labels file: Contains breed labels for training images


## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dog-breed-classification.git
cd dog-breed-classification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```
dog-breed-classification/
├── data/
│   ├── train/          # Training images
│   ├── test/           # Test images
│   └── labels.csv      # Training labels
├── models/             # Saved model files
├── notebooks/          # Jupyter notebooks
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py
│       └── callbacks.py
├── tests/
│   └── test_model.py
├── README.md
└── requirements.txt
```

## Usage

1. Data preparation:
```python
from src.data.loader import load_data
from src.data.preprocessor import preprocess_images

# Load and preprocess data
train_data, val_data = load_data()
```

2. Train model:
```python
from src.models.model import create_model, train_model

# Create and train model
model = create_model()
history = train_model(model, train_data, val_data)
```

3. Make predictions:
```python
from src.models.model import predict_breed

# Predict dog breed from image
breed = predict_breed(model, "path/to/image.jpg")
```

## Model Architecture

The model uses transfer learning with MobileNetV2 as the base model:
1. MobileNetV2 base (pretrained on ImageNet)
2. Dense layer (120 units, softmax activation)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgments
- Dataset from Kaggle's Dog Breed Identification competition
- TensorFlow Hub for providing pretrained models
- MobileNetV2 architecture
