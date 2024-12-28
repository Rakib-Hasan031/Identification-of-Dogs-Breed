"""
Model training and prediction utilities.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def create_model(C=0.20433597178569418, solver='liblinear'):
    """Create a LogisticRegression model with specified parameters."""
    return LogisticRegression(C=C, solver=solver)

def train_model(X_train, y_train, param_grid=None):
    """
    Train the heart disease prediction model.
    
    Args:
        X_train: Training features
        y_train: Training target
        param_grid: Dictionary with parameters names (string) as keys and lists of
                   parameter settings to try as values
                   
    Returns:
        trained model
    """
    if param_grid is None:
        # Use default parameters
        model = create_model()
        model.fit(X_train, y_train)
        return model
    
    # Use GridSearchCV for hyperparameter tuning
    model = GridSearchCV(
        LogisticRegression(),
        param_grid,
        cv=5,
        verbose=True
    )
    model.fit(X_train, y_train)
    return model

def predict(model, X):
    """Make predictions using the trained model."""
    return model.predict(X)