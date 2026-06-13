# SPU (Signal Processing Unit) - Audio Intelligence System

SPU is a modular Audio Intelligence System built with Python, leveraging Digital Signal Processing (DSP) and Machine Learning (ML) to analyze and classify audio signals.

## Project Structure

- `data/raw`: Original audio files (.wav)
- `data/processed`: Extracted features and preprocessed data
- `notebooks/`: Step-by-step workflow for exploration, DSP, training, and inference
- `src/`: Modular Python scripts for reusable logic
- `models/`: Saved trained models (e.g., .pkl)
- `outputs/`: Generated plots and evaluation results

## Tech Stack

- **DSP:** Librosa, SciPy, NumPy
- **ML:** Scikit-learn, Pandas
- **Visualization:** Matplotlib
- **Persistence:** Joblib

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open JupyterLab and follow the notebooks in `notebooks/` sequentially.

## Features

- Waveform Visualization
- Fast Fourier Transform (FFT) Analysis
- Mel-frequency Cepstral Coefficients (MFCC) Extraction
- RandomForest Classification
- Real-time Audio Support (Planned)
