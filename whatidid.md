# What I Did: SPU Project Code Explanation

This document explains the complete Signal Processing Unit (SPU) pipeline and all code components.

## Project Overview

The SPU is a Python-based audio analysis system that:
1. Loads `.wav` audio files
2. Analyzes them in time and frequency domains
3. Applies digital filtering
4. Extracts audio features (DSP and MFCC) 
# WHAT IS FEATURE EXTRACTION?
Feature extraction means:
Taking important properties from sound.
Like:
- pitch
-  energy
- frequencies
- rhythm
- tone
- speech characteristics
- These become ML input.
# mel scale
image-- in folder(mel.jpeg)
mfcc = mel frequency cepstral coefficients.
# WHAT MFCC CAPTURES
MFCC captures:
- vocal shape
- tone
- pronunciation
- speech texture
- Very useful for:
- speech recognition
- emotion detection
- speaker identification
5. Trains a machine learning model to classify audio segments
6. Makes predictions on unseen audio

---

## Code Components

### 1. **Setup and Requirements** (Cell 1-2)

```python
# req -- pip install numpy scipy matplotlib librosa scikit-learn pandas sounddevice jupyter
```

**What it does:**
- Lists all required Python packages needed to run the notebook
- Install with: `pip install numpy scipy matplotlib librosa scikit-learn pandas sounddevice jupyter`

**Packages:**
- `numpy`: Numerical computing
- `scipy`: Signal processing (filtering)
- `matplotlib`: Plotting and visualization
- `librosa`: Audio loading and feature extraction
- `scikit-learn`: Machine learning models and metrics
- `pandas`: Data manipulation
- `sounddevice`: Audio device support (optional)
- `jupyter`: Notebook environment

---

### 2. **Imports and Configuration** (Cell 3)

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy import signal as spsignal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

plt.rcParams['figure.figsize'] = (12, 4)

DATA_DIR = Path('.')
AUDIO_FILES = sorted(DATA_DIR.glob('sample*.wav'))
if not AUDIO_FILES:
    raise FileNotFoundError('No sample*.wav files were found in the current folder.')

SEGMENT_SECONDS = 0.5
LOWPASS_CUTOFF_HZ = 4000
RANDOM_STATE = 42
```

**What it does:**
- Imports all necessary libraries
- Sets up default plot size
- Finds all `.wav` files in the current folder
- Defines configuration parameters:
  - `SEGMENT_SECONDS = 0.5`: Each audio segment is 0.5 seconds long
  - `LOWPASS_CUTOFF_HZ = 4000`: Filter cutoff frequency at 4000 Hz
  - `RANDOM_STATE = 42`: For reproducible results

**Key Constants:**
- Using `Path('.')` ensures the code works across Windows/Mac/Linux
- `glob('sample*.wav')` finds files like `sample1m.wav`, `sample2.wav`, etc.

---

### 3. **Audio Loading** (Cell 4)

```python
audio, sr = librosa.load("sample1m.wav")
print(sr)
print(audio.shape)
```

**What it does:**
- Loads a single audio file using `librosa`
- `audio`: Array of audio samples (amplitude values)
- `sr`: Sample rate (how many samples per second, typically 44100 Hz)
- Prints the sample rate and shape of the audio array

**Why this matters:**
- Sample rate determines frequency resolution in analysis
- Audio shape tells us how many samples we have

---

### 4. **Waveform Visualization** (Cell 5)

```python
plt.figure(figsize=(12,4))
plt.plot(audio)
plt.title("Audio Signal")
plt.xlabel("sample")
plt.ylabel("amplitude")
plt.show()
```

**What it does:**
- Creates a time-domain plot of the raw audio signal
- X-axis: Sample index (time progression)
- Y-axis: Amplitude (volume at each sample)
- Shows the waveform as a line graph

**Why this matters:**
- Visual inspection reveals signal characteristics
- Shows envelope, transients, silence, and clipping

---

### 5. **FFT Analysis** (Cell 6)

```python
# FFT Formula: X(f) = integrate x(t)*e^(-j*2pi*ft).dt from (-∞ to ∞)
fft_result = np.fft.fft(audio)
magnitude = np.abs(fft_result)
plt.plot(magnitude)
plt.title("FFT Spectrum")
plt.show()
```

**What it does:**
- Computes the Fast Fourier Transform (FFT) of the audio
- FFT transforms time-domain signal → frequency-domain
- Takes absolute value to get magnitude
- Plots frequency spectrum

**The Math:**
- Fourier Transform equation shown as comment
- Tells us which frequencies are present and their strengths
- Higher magnitudes = more energy at that frequency

**Why this matters:**
- Reveals frequency content of the audio
- Shows if there are low frequencies, high frequencies, or dominant tones

---

### 6. **Butterworth Filtering** (Cell 7)

```python
b, a = spsignal.butter(4, 0.1, btype='low')
filtered = spsignal.filtfilt(b, a, audio)
```

**What it does:**
- Creates a 4th-order Butterworth low-pass filter
- `0.1`: Normalized cutoff frequency (4000 Hz ÷ 22050 Hz Nyquist)
- `filtfilt()`: Applies filter forward and backward (zero phase distortion)
- Removes high-frequency noise and unwanted components

**Filter Details:**
- **Low-pass**: Allows low frequencies through, blocks high frequencies
- **Order 4**: Steeper frequency response
- **Butterworth**: Smooth response with no ripple in passband

**Why this matters:**
- Cleans up noisy audio before feature extraction
- Preserves important signal components while removing noise

---

### 7. **MFCC Feature Extraction** (Cell 8)

```python
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
print(mfcc.shape)
```

**What it does:**
- Extracts 13 Mel-Frequency Cepstral Coefficients (MFCCs)
- MFCCs mimic how humans perceive sound
- Converts audio to a compact feature representation
- Shape output: (13, time_steps) - 13 features across time

**MFCC Explanation:**
- Divides audio into frequency bands using mel-scale (human hearing perception)
- Computes power in each band
- Applies logarithmic scaling (how humans hear volume)
- Takes discrete cosine transform to decorrelate features
- Results: 13 coefficients that summarize spectral shape

**Why this matters:**
- Highly effective for audio classification tasks
- Much smaller than raw audio (13 numbers vs millions)
- Captures perceptually relevant information

---

### 8. **Machine Learning Model** (Cell 9)

```python
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

**What it does:**
- Creates a Random Forest classifier
- Trains on labeled training data
- Random Forest: Ensemble of decision trees that vote on predictions

**Why Random Forest:**
- Robust for small tabular feature datasets
- No need for feature scaling
- Handles non-linear relationships
- Provides feature importance insights

---

### 9. **Prediction** (Cell 10)

```python
prediction = model.predict(X_test)
```

**What it does:**
- Uses trained model to predict class labels for test data
- `X_test`: Feature vectors from test set
- Output: Predicted class for each test sample

---

## Complete Pipeline (What Should Run End-to-End)

A complete implementation would execute these steps:

### Step 1: Load All Audio Files
```
Create audio_bank dictionary
For each sample*.wav file:
  - Load with librosa.load(sr=None, mono=True)
  - Store audio, sample rate, and metadata
  - Build metadata table with duration, peak amplitude, etc.
```

### Step 2: Visualize Representative Audio
```
Select shortest audio file as example
Plot 2x1 figure:
  - Top: Waveform with librosa.display.waveshow()
  - Bottom: Spectrogram with STFT and librosa.display.specshow()
```

### Step 3: Apply Filtering and Compare
```
Calculate Nyquist frequency = sr / 2
Design Butterworth filter with normalized cutoff
Apply filtfilt() to preserve phase
Create 2x1 comparison plots:
  - Top: Time domain raw vs filtered
  - Bottom: Frequency domain FFT comparison
```

### Step 4: Extract Features from All Segments
```
For each audio file:
  - Divide into 0.5-second segments
  - For each segment, extract:
    * RMS energy
    * Zero-crossing rate
    * Spectral centroid
    * Spectral bandwidth
    * Spectral rolloff
    * Spectral flatness
    * 13 MFCC means
    * 13 MFCC standard deviations
  - Total: 32 features per segment

Create balanced dataset (same segments per class)
Store in DataFrame with label, file, segment_index
```

### Step 5: Train Machine Learning Model
```
Split data: 75% train, 25% test (stratified by class)
Train RandomForestClassifier with:
  - 300 trees
  - Balanced class weights
  - Fixed random state for reproducibility

Evaluate with:
  - Accuracy score
  - Classification report (precision, recall, F1)
  - Confusion matrix
```

### Step 6: Demo Prediction
```
Sample one random segment from dataset
Extract its features
Predict class with model
Show:
  - Actual label
  - Predicted label
  - Source file
  - Class probabilities for all classes
```

---

## Key Concepts Explained

### **Fourier Transform**
- Converts time-domain signal → frequency-domain
- Shows which frequencies are present in the signal
- Foundation for FFT analysis

### **Digital Filtering**
- Removes unwanted frequency components
- Low-pass filter: passes frequencies below cutoff
- Applied to clean audio before feature extraction

### **Feature Extraction**
- Converts raw audio → numerical descriptors
- MFCC: Perceptually-motivated coefficients
- RMS, ZCR, Spectral features: Energy and frequency characteristics
- Creates compact representation (32 features vs millions of samples)

### **Machine Learning**
- Random Forest: Ensemble of decision trees
- Supervised learning: Trains on labeled examples
- Classification: Predicts which audio clip a segment came from
- No external labels used → uses source filename as label

### **Train/Test Split**
- Prevents overfitting by evaluating on unseen data
- 75% training data, 25% test data
- Stratified: maintains class balance in both sets

---

## Dataset

The project uses 4 audio files:
- `sample1m.wav` (~6 seconds)
- `sample2.wav` (~12 seconds)
- `sample5.wav` (~30 seconds)
- `sample10.wav` (~60 seconds)

Each sampled at 44.1 kHz, stereo initially (converted to mono for processing).

---

## Expected Results

When run successfully, the system produces:
1. Metadata table of all audio files
2. Waveform and spectrogram visualization
3. Time-domain filtering comparison
4. Frequency-domain FFT comparison
5. Balanced feature dataset (32 features × N segments)
6. Model accuracy on test set
7. Classification report with precision/recall/F1
8. Confusion matrix visualization
9. Demo prediction with class probabilities

---

## Files in This Project

- `mainunit.ipynb`: Main working notebook (current implementation)
- `codex.ipynb`: Alternative clean implementation
- `SPU_thesis.md`: Complete project documentation and theory
- `documentation.md`: Additional documentation
- `whatidid.md`: This explanation file
- `sample*.wav`: Audio files for analysis (not included in repo)

---

## How to Run

1. Install dependencies: `pip install numpy scipy matplotlib librosa scikit-learn pandas jupyter`
2. Place `.wav` files in the same folder as the notebook
3. Open `mainunit.ipynb` in Jupyter
4. Run cells sequentially from top to bottom
5. Observe plots and printed outputs

---

This SPU demonstrates the complete workflow of a digital signal processor: acquisition → analysis → filtering → feature extraction → machine learning → prediction.
