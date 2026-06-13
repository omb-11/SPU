import librosa
import numpy as np
import scipy.fftpack

def load_audio(file_path, sr=22050):
    """Load audio file using librosa."""
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

def get_fft(y, sr):
    """Compute FFT of the signal."""
    n = len(y)
    freq = np.fft.fftfreq(n, d=1/sr)
    fft_values = np.abs(scipy.fftpack.fft(y))
    return freq[:n//2], fft_values[:n//2]

def extract_mfcc(y, sr, n_mfcc=13):
    """Extract MFCC features and use mean pooling."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs.T, axis=0)
    return mfcc_mean
