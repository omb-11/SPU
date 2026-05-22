# Signal Processing Unit (SPU): A Mini Thesis

## Title
Design and Demonstration of a Signal Processing Unit for Audio Analysis Using Python-Based Digital Signal Processing and Machine Learning

## Abstract
This project demonstrates a compact Signal Processing Unit (SPU) for audio analysis. The system operates on `.wav` recordings available in the project folder and applies a complete signal-processing pipeline: signal acquisition, time-domain inspection, filtering, Fast Fourier Transform (FFT), feature extraction, model training, prediction, and visualization. The implementation is practical rather than purely theoretical. It shows how a Python notebook can emulate the core workflow of a digital signal processor by transforming raw audio into structured numerical features and then using those features for classification. Since the dataset does not contain external human labels, the machine-learning stage uses source clip identity as the prediction target. This keeps the pipeline executable and scientifically interpretable while preserving the structure proposed in the original project notes.

## Introduction
A Digital Signal Processor is designed to manipulate real-world signals efficiently. In this folder, the real-world signal is audio. The project intent is clear from the existing files: load audio, study its waveform, apply filtering, inspect its frequency content, extract features, and attach a predictive model on top of those features. The implementation outlined in this thesis follows the complete signal-processing pipeline from raw audio to machine-learning-based prediction.

## Problem Statement
Raw audio signals are difficult to use directly for pattern recognition because they are high-dimensional and sensitive to noise. A signal-processing pipeline is therefore needed to:

1. standardize the audio representation,
2. reduce unwanted frequency content,
3. transform the signal into an interpretable frequency view,
4. compute stable descriptive features, and
5. use those features for automatic prediction.

## Objectives
The SPU in this project has five concrete objectives:

1. Load and organize the `.wav` files in the folder.
2. Visualize the waveform and spectrogram of a representative signal.
3. Apply a low-pass Butterworth filter to suppress higher-frequency components.
4. Extract both classical DSP and cepstral features from fixed-length signal segments.
5. Train a machine-learning model that predicts which source audio clip produced a segment.

## Dataset Description
The folder contains four stereo `.wav` recordings sampled at 44.1 kHz:

1. `sample1m.wav`
2. `sample2.wav`
3. `sample5.wav`
4. `sample10.wav`

Their durations are approximately 6 seconds, 12 seconds, 30 seconds, and 60 seconds. For analysis, the notebook converts them to mono so that each signal can be processed consistently.

## Methodology
### 1. Signal Acquisition
Each audio file is loaded using `librosa.load(..., sr=None, mono=True)`. Using `sr=None` preserves the native sampling rate, and `mono=True` simplifies downstream DSP operations by averaging the stereo channels into a single waveform.

### 2. Time-Domain Analysis
The waveform display shows amplitude variation over time. This gives immediate visibility into envelope changes, transient behavior, and potential clipping or silence regions.

### 3. Spectral Analysis
A spectrogram is computed with the Short-Time Fourier Transform (STFT). This gives a time-varying view of the signal’s frequency content and supports visual interpretation beyond the raw waveform.

### 4. Filtering
A fourth-order low-pass Butterworth filter is applied. The Butterworth filter is chosen because it provides a smooth frequency response without ripple in the passband. In the notebook, the filtered signal is compared directly against the original signal in both the time domain and the FFT domain.

### 5. FFT
The Fast Fourier Transform provides the signal’s frequency-domain magnitude distribution. This is a central DSP step because it reveals how signal energy is distributed across frequencies. Comparing the FFT before and after filtering verifies that the low-pass filter suppresses higher-frequency content as intended.

### 6. Feature Extraction
Fixed-length signal windows are created from each audio clip. For each window, the notebook extracts:

1. RMS energy
2. Zero-crossing rate
3. Spectral centroid
4. Spectral bandwidth
5. Spectral rolloff
6. Spectral flatness
7. 13 MFCC mean values
8. 13 MFCC standard deviations

These features combine time-domain energy, spectral shape, and cepstral structure, which makes them suitable for lightweight audio classification.

### 7. Machine Learning
Because the project folder does not provide semantic labels such as speech, music, machinery, or emotion, the source file name is used as the class label. This means the classifier learns to distinguish the acoustic profile of segments originating from different clips. A Random Forest classifier is used because it is robust for small tabular feature datasets and does not require feature scaling to perform reasonably well.

## Mathematical Background
The main frequency-domain idea behind the project is the Fourier Transform:

X(f) = integral of x(t)e^(-j2pift)dt

In discrete computation, the FFT is used instead of the continuous transform because the audio is sampled digitally. The classifier never consumes the raw FFT directly in this implementation; instead, it uses summary features derived from short windows, which is usually more stable for small datasets.

## Results Interpretation
A complete implementation of this system should produce:

1. a metadata table for all audio files,
2. a waveform plot,
3. a spectrogram,
4. a time-domain raw vs filtered comparison,
5. an FFT comparison,
6. a balanced feature dataset,
7. model accuracy and classification report,
8. a confusion matrix, and
9. a demonstration prediction on a held feature row.

This makes the notebook usable as both a lab notebook and a project demonstration.

## Limitations
The current system has several limitations:

1. The dataset is small.
2. The labels are proxy labels derived from file identity, not external annotation.
3. The classifier is a demonstration model, not a production-ready recognizer.
4. Real-time streaming and deployment are not yet implemented.

These are important because they affect how the results should be interpreted. The notebook proves the pipeline works, but it does not claim broad generalization beyond the available signals.

## Future Work
Several realistic next steps follow naturally from the current structure:

1. Add real task labels for supervised learning.
2. Expand the dataset with more recordings per class.
3. Introduce denoising, band-pass filtering, or adaptive filtering.
4. Add real-time microphone capture.
5. Build a drag-and-drop UI for audio analysis.
6. Compare classical ML models with PyTorch-based neural networks.

## Conclusion
This project successfully demonstrates the architecture of a small Signal Processing Unit for audio. The thesis outlines a complete DSP pipeline that starts from raw `.wav` audio, performs core digital signal processing, extracts informative descriptors, and finishes with machine-learning-based prediction.
