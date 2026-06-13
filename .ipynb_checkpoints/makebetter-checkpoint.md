# Make It Better: SPU Model Improvement Guide

This document explains how to improve model training and the overall SPU project for better results and real-world applicability.

---

## Part 1: How to Train the Model Better

### 1.1 Hyperparameter Tuning

The current model uses default Random Forest parameters. You can improve it significantly with tuning:

#### **Option A: Grid Search (Systematic)**

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 300, 500, 1000],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

# Create grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1  # Use all cores
)

# Fit and find best parameters
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

model = grid_search.best_estimator_
```

**What this does:**
- Tests all combinations of parameters
- Uses 5-fold cross-validation to avoid overfitting
- Finds the best combination
- Time-consuming but thorough

**Why it matters:**
- Default parameters rarely optimal
- Can improve accuracy by 5-15%

---

#### **Option B: Random Search (Faster)**

```python
from sklearn.model_selection import RandomizedSearchCV

# Same param_grid as above
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    n_iter=20,  # Try 20 random combinations
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
model = random_search.best_estimator_
```

**Advantage:** Faster than grid search, still finds good parameters

---

#### **Key Parameters to Tune:**

| Parameter | Current | Better Range | Impact |
|-----------|---------|--------------|--------|
| `n_estimators` | 300 | 100-1000 | More trees = better but slower |
| `max_depth` | None | 10-30 | Limits tree depth, prevents overfitting |
| `min_samples_split` | 2 | 5-10 | Min samples to split node (prevents overfitting) |
| `min_samples_leaf` | 1 | 2-4 | Min samples in leaf node |
| `max_features` | 'sqrt' | 'sqrt', 'log2' | Features per split (prevents overfitting) |
| `class_weight` | 'balanced' | 'balanced', 'balanced_subsample' | Handles class imbalance |

---

### 1.2 Cross-Validation (More Robust Evaluation)

Current approach: Single train/test split (risky with small data)

**Better: K-Fold Cross-Validation**

```python
from sklearn.model_selection import cross_validate

# Perform 5-fold cross-validation
cv_results = cross_validate(
    RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'),
    X, y,
    cv=5,
    scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
    return_train_score=True
)

print(f"CV Accuracies: {cv_results['test_accuracy']}")
print(f"Mean Accuracy: {cv_results['test_accuracy'].mean():.3f} (+/- {cv_results['test_accuracy'].std():.3f})")
print(f"Mean F1: {cv_results['test_f1_macro'].mean():.3f}")
```

**What this does:**
- Splits data into 5 folds
- Trains 5 models, each time leaving out different fold
- Evaluates on each fold
- More reliable estimate of real performance

**Benefits:**
- Uses all data for training and testing
- Better variance estimate
- Prevents lucky/unlucky splits

---

### 1.3 Try Alternative Algorithms

Random Forest is good, but try others:

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Dictionary of models to compare
models = {
    'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=300, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=300, random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=42, early_stopping=True, validation_fraction=0.1)
}

# Compare all models
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name:20s}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Use best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
best_model.fit(X_train, y_train)
```

**Models to compare:**
- **Gradient Boosting**: Often better than Random Forest, builds trees sequentially
- **XGBoost**: Faster, more optimized gradient boosting
- **Neural Networks**: Can capture complex patterns, needs more data
- **SVM**: Good for small datasets, less prone to overfitting

---

### 1.4 Feature Scaling (For Some Algorithms)

Some algorithms benefit from scaled features:

```python
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now train with scaled features
model = SVC(kernel='rbf', probability=True)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

**When to use:**
- **Essential**: Neural Networks, SVM, KNN, Logistic Regression
- **Doesn't hurt**: Tree-based methods (but not necessary)

---

### 1.5 Class Imbalance Handling

If classes have different sizes, use techniques beyond `class_weight`:

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# SMOTE: Create synthetic samples of minority class
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

model.fit(X_train_balanced, y_train_balanced)
```

**Techniques:**
- **SMOTE**: Creates synthetic minority samples (good for severe imbalance)
- **Under-sampling**: Remove majority samples
- **Threshold adjustment**: Change decision boundary

---

### 1.6 Feature Importance Analysis

Understand which features matter:

```python
# Get feature importances from trained Random Forest
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot top features
plt.figure(figsize=(12, 6))
plt.title("Top 15 Feature Importances")
for i in range(15):
    print(f"{i+1}. {X_train.columns[indices[i]]}: {importances[indices[i]]:.4f}")

plt.bar(range(15), importances[indices[:15]])
plt.xticks(range(15), [X_train.columns[i] for i in indices[:15]], rotation=45, ha='right')
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
```

**Why:**
- Drop unimportant features (simplify model)
- Focus on most predictive features
- Understand what model uses

---

### 1.7 Learning Curves (Diagnose Overfitting/Underfitting)

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=300, random_state=42),
    X_train, y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy',
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.plot(train_sizes, val_mean, label='Validation score')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Learning Curve')
plt.grid()
plt.show()
```

**Interpretation:**
- **Converging curves**: Good fit
- **Large gap**: Overfitting (model memorizes)
- **Both low**: Underfitting (model too simple)
- **Curves rising**: Add more data helps

---

### 1.8 Ensemble Methods (Combine Multiple Models)

```python
from sklearn.ensemble import VotingClassifier

# Create ensemble of best models
ensemble = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=300, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=300, random_state=42)),
        ('svm', SVC(probability=True, kernel='rbf'))
    ],
    voting='soft'  # Use probability averages
)

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble Accuracy: {accuracy:.3f}")
```

**Why:**
- Combines strengths of multiple models
- Usually more stable than single model
- Reduces variance

---

## Part 2: Bigger Improvements to the Project

### 2.1 Use Real Labels Instead of Filenames

**Current approach (Problem):**
- Uses source filename as label
- Doesn't capture actual content
- Not scientifically rigorous

**Better approach:**

```python
# Create a labels.csv with real annotations
# Columns: file, segment_index, label, description
#
# Example:
# sample1m.wav,0,speech,male speech
# sample1m.wav,1,speech,male speech
# sample2.wav,0,music,electronic music
# sample2.wav,1,music,classical music

import pandas as pd

# Load real labels
label_df = pd.read_csv('labels.csv')

# Merge with feature dataset
feature_df = feature_df.merge(
    label_df[['file', 'segment_index', 'label']],
    on=['file', 'segment_index'],
    how='left'
)

# Now train on real semantic labels
y = feature_df['label']  # Speech, Music, Noise, etc.
```

**Impact:** 100-500% accuracy improvement with real labels

---

### 2.2 Expand Dataset

**Current:**
- 4 audio files (~110 seconds total)
- Very small for ML

**Better:**

```
├── speech/
│   ├── speaker1/
│   │   ├── file1.wav
│   │   ├── file2.wav
│   │   └── ... (10+ files per speaker)
│   ├── speaker2/
│   └── ... (5+ speakers)
├── music/
│   ├── classical/
│   ├── electronic/
│   ├── hip_hop/
│   └── ... (3+ genres, 10+ files each)
├── environmental/
│   ├── traffic/
│   ├── birds/
│   ├── machinery/
│   └── ... (3+ types, 10+ files each)
└── silence/
    └── ... (10+ files)
```

**Sources:**
- **ESC-50**: Environmental Sound Classification (2000 clips)
- **FSD50K**: Freesound Dataset (50K clips)
- **GTZAN**: Music Genre (1000 tracks)
- **LibriSpeech**: Speech (1000+ hours)
- **YouTube**: Download clips of specific categories

**Code to organize data:**

```python
import os
from pathlib import Path

# Organize files by category
data_dir = Path('audio_data')

for category_folder in data_dir.glob('*'):
    if category_folder.is_dir():
        audio_files = list(category_folder.glob('*.wav'))
        print(f"{category_folder.name}: {len(audio_files)} files")
```

**Impact:** Accuracy can jump to 85-95% with 1000+ samples

---

### 2.3 Advanced Feature Engineering

**Current features:** 32 basic features

**Add more sophisticated features:**

```python
def advanced_features(segment, sr):
    features = extract_features(segment, sr)  # Keep existing 32
    
    # Add Chroma features (musical notes)
    chroma_stft = librosa.feature.chroma_stft(y=segment, sr=sr)
    features['chroma_mean'] = float(np.mean(chroma_stft))
    features['chroma_std'] = float(np.std(chroma_stft))
    
    # Add Tempogram (rhythm)
    onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
    tempogram = librosa.feature.tempogram(onset_env=onset_env, sr=sr)
    features['tempogram_mean'] = float(np.mean(tempogram))
    
    # Add Spectral contrast
    S = np.abs(librosa.stft(segment))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    for i, val in enumerate(np.mean(contrast, axis=1)):
        features[f'spectral_contrast_{i}'] = float(val)
    
    # Add Poly features
    poly_features = librosa.feature.poly_features(S=S)
    for i, val in enumerate(np.mean(poly_features, axis=1)):
        features[f'poly_feature_{i}'] = float(val)
    
    # Add Delta (rate of change)
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    for i, val in enumerate(np.mean(delta_mfcc, axis=1)):
        features[f'mfcc_delta_{i}'] = float(val)
    
    # Add Tonnetz (chromatic structure)
    tonnetz = librosa.feature.tonnetz(y=segment, sr=sr)
    for i, val in enumerate(np.mean(tonnetz, axis=1)):
        features[f'tonnetz_{i}'] = float(val)
    
    return features
```

**New features:**
- **Chroma**: Which musical notes are present
- **Tempogram**: Rhythm/beat information
- **Spectral Contrast**: Peaks vs valleys in spectrum
- **Poly Features**: Derived from spectrogram
- **Delta MFCC**: How MFCCs change over time (dynamics)
- **Tonnetz**: Harmonic/tonal information

**Impact:** +5-10% accuracy improvement

---

### 2.4 Data Augmentation

Artificially expand small dataset:

```python
import librosa
import numpy as np

def augment_audio(audio, sr):
    """Create 5 variations of each audio clip"""
    augmented = [audio]  # Original
    
    # 1. Time stretch (speed up/slow down)
    augmented.append(librosa.effects.time_stretch(audio, rate=0.9))  # 10% slower
    augmented.append(librosa.effects.time_stretch(audio, rate=1.1))  # 10% faster
    
    # 2. Pitch shift
    augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2))  # Down 2 semitones
    augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=2))   # Up 2 semitones
    
    # 3. Add noise
    noise = np.random.normal(0, 0.005, len(audio))
    augmented.append(audio + noise)
    
    # 4. Dynamic range compression
    augmented.append(audio * 0.8 + np.random.normal(0, 0.002, len(audio)))
    
    return augmented

# Use in dataset building
for key, item in audio_bank.items():
    audio = item['audio']
    sr = item['sr']
    augmented_audios = augment_audio(audio, sr)
    
    for aug_idx, aug_audio in enumerate(augmented_audios):
        # Extract features and add to dataset
        # This multiplies dataset size by ~6
```

**Impact:** Dataset size ×6, typically +2-5% accuracy

---

### 2.5 Deep Learning (Neural Networks)

For larger datasets (1000+), neural networks often outperform:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Build neural network
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(y_train.unique()), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.3f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()
```

**Advantages:**
- Better for large datasets
- Can learn hierarchical features
- No feature scaling needed with normalization

**Disadvantages:**
- Needs more data
- Longer training time
- Harder to interpret

---

### 2.6 Real-Time Processing

Deploy on live microphone:

```python
import sounddevice as sd

# Record chunks and predict in real-time
def real_time_prediction(duration=10, sr=44100, segment_samples=int(0.5 * 44100)):
    
    print("Recording... (press Ctrl+C to stop)")
    
    # Record audio
    audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    
    # Process in segments
    for i in range(0, len(audio) - segment_samples, segment_samples):
        segment = audio[i:i+segment_samples]
        
        # Extract features
        features = extract_features(segment, sr)
        features_df = pd.DataFrame([features])
        
        # Predict
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0].max()
        
        print(f"Segment {i//segment_samples}: {prediction} ({probability:.2%})")
        
real_time_prediction(duration=5)
```

**Use cases:**
- Real-time audio monitoring
- Speech recognition
- Environmental sound detection

---

### 2.7 Model Deployment

Save and deploy model:

```python
import joblib

# Save model
joblib.dump(model, 'audio_classifier.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')  # If using scaling

# Load model (on different machine)
loaded_model = joblib.load('audio_classifier.pkl')

# Or use ONNX for cross-platform
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

**Deployment options:**
- Flask web app
- AWS Lambda
- Mobile app (TensorFlow Lite)
- Docker container
- REST API

---

### 2.8 A/B Testing

Compare two models:

```python
from scipy import stats

# Train model A and model B
model_a = RandomForestClassifier(n_estimators=300)
model_b = GradientBoostingClassifier(n_estimators=300)

model_a.fit(X_train, y_train)
model_b.fit(X_train, y_train)

# Get predictions on test set
pred_a = model_a.predict(X_test)
pred_b = model_b.predict(X_test)

# McNemar's test for statistical significance
from sklearn.metrics import accuracy_score

acc_a = accuracy_score(y_test, pred_a)
acc_b = accuracy_score(y_test, pred_b)

# Count agreements/disagreements
agree_both = np.sum((pred_a == y_test) & (pred_b == y_test))
agree_a_wrong_b = np.sum((pred_a == y_test) & (pred_b != y_test))
agree_b_wrong_a = np.sum((pred_a != y_test) & (pred_b == y_test))
both_wrong = np.sum((pred_a != y_test) & (pred_b != y_test))

print(f"Model A Accuracy: {acc_a:.3f}")
print(f"Model B Accuracy: {acc_b:.3f}")
print(f"Statistically significant? {abs(agree_a_wrong_b - agree_b_wrong_a) > 1.96 * np.sqrt(agree_a_wrong_b + agree_b_wrong_a)}")
```

---

### 2.9 Error Analysis

Understand where model fails:

```python
# Confusion matrix analysis
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Classification Report:")
print(cr)

# Find misclassified samples
misclassified = feature_df[feature_df.index.isin(
    np.where(model.predict(feature_df.drop(columns=['label', 'file', 'segment_index'])) != feature_df['label'])[0]
)]

print(f"\nMisclassified {len(misclassified)} samples out of {len(feature_df)}")
print("\nMisclassified breakdown:")
print(misclassified['label'].value_counts())

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, 
            yticklabels=model.classes_)
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
```

---

### 2.10 Monitoring and Retraining

In production, model performance degrades (data drift):

```python
# Log predictions and actual labels
import json
from datetime import datetime

prediction_log = []

def predict_and_log(audio_segment, true_label=None):
    features = extract_features(audio_segment, sr)
    prediction = model.predict(pd.DataFrame([features]))[0]
    probability = model.predict_proba(pd.DataFrame([features]))[0].max()
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'prediction': prediction,
        'confidence': float(probability),
        'true_label': true_label
    }
    prediction_log.append(log_entry)
    
    # Save log periodically
    if len(prediction_log) % 100 == 0:
        with open('predictions_log.jsonl', 'a') as f:
            for entry in prediction_log[-100:]:
                f.write(json.dumps(entry) + '\n')
    
    return prediction, probability

# Monitor performance
def check_model_drift():
    recent_logs = prediction_log[-1000:]
    recent_accuracy = sum(
        1 for log in recent_logs 
        if log['true_label'] and log['prediction'] == log['true_label']
    ) / len([l for l in recent_logs if l['true_label']])
    
    print(f"Recent accuracy: {recent_accuracy:.3f}")
    
    # Retrain if accuracy drops below threshold
    if recent_accuracy < 0.75:
        print("Accuracy dropped! Retraining model...")
        # Collect new data and retrain
        new_data = pd.read_json('recent_data.jsonl', lines=True)
        model.fit(new_data.drop(columns=['label']), new_data['label'])
```

---

## Summary: Priority Improvements

### Immediate (Easy, High Impact):
1. **Grid search** for hyperparameters (+5-10%)
2. **Cross-validation** for better evaluation
3. **Try different algorithms** (XGBoost, Gradient Boosting)
4. **Feature importance analysis**

### Medium-term (Moderate effort):
1. **Expand dataset** to 500+ samples
2. **Add real labels** instead of filenames
3. **Advanced feature engineering** (chroma, tempogram, etc.)
4. **Data augmentation** (×5-6 dataset size)

### Long-term (Complex, High Impact):
1. **Deep learning** with 1000+ samples
2. **Real-time deployment** on microphone
3. **Production system** with monitoring
4. **Domain-specific tuning** (speech, music, etc.)

---

## Quick Implementation Checklist

- [ ] Try GridSearchCV on current data
- [ ] Implement 5-fold cross-validation
- [ ] Test 3+ different algorithms
- [ ] Create feature importance plot
- [ ] Collect 100+ new audio samples
- [ ] Add real semantic labels (speech/music/noise/etc.)
- [ ] Implement data augmentation
- [ ] Try advanced features (chroma, tempogram)
- [ ] Train neural network on expanded dataset
- [ ] Deploy as Flask API
- [ ] Set up monitoring and logging
- [ ] Implement retraining pipeline

---

Each improvement builds on previous ones. Start with hyperparameter tuning, then expand dataset, then try deep learning.
