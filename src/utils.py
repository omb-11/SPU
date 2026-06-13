import matplotlib.pyplot as plt
import librosa.display
import os

def plot_waveform(y, sr, title="Waveform", save_path=None):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_fft(freq, fft_values, title="FFT Spectrum", save_path=None):
    plt.figure(figsize=(10, 4))
    plt.plot(freq, fft_values)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_mfcc(mfcc, title="MFCC", save_path=None):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()
