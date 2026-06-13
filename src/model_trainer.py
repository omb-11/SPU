from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os

def train_model(X, y):
    """Train a RandomForestClassifier."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, cm

def save_model(model, file_path):
    """Save the model to disk."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)

def load_model(file_path):
    """Load the model from disk."""
    return joblib.load(file_path)
