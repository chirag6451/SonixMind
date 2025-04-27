"""
Visualization utilities for audio waveforms, spectrograms, and frequency analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
import librosa
import librosa.display
import io
from typing import Tuple, Optional, Dict, List, Union
import base64
from matplotlib.collections import LineCollection
import scipy.signal as signal

def create_waveform(audio: np.ndarray, sr: int, title: str = None, 
                    color: str = '#1DB954', figsize: Tuple[int, int] = (10, 2)) -> Figure:
    """
    Create a waveform visualization of audio data.
    
    Args:
        audio: Audio data as numpy array
        sr: Sample rate
        title: Title for the plot
        color: Color for the waveform
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure with waveform plot
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    
    # Calculate time array
    time = np.linspace(0, len(audio) / sr, len(audio))
    
    # Plot waveform
    ax.plot(time, audio, color=color, linewidth=0.5)
    
    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    if title:
        ax.set_title(title)
    
    # Set y-axis limits with some padding
    max_amp = max(0.01, np.max(np.abs(audio))) * 1.1
    ax.set_ylim(-max_amp, max_amp)
    
    # Remove spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # Tight layout and grid
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    
    return fig

def create_spectrogram(audio: np.ndarray, sr: int, title: str = None,
                       cmap: str = 'viridis', figsize: Tuple[int, int] = (10, 4)) -> Figure:
    """
    Create a spectrogram visualization of audio data.
    
    Args:
        audio: Audio data as numpy array
        sr: Sample rate
        title: Title for the plot
        cmap: Colormap for spectrogram
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure with spectrogram plot
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    
    # Calculate spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    
    # Plot spectrogram
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax, cmap=cmap)
    
    # Add colorbar
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    if title:
        ax.set_title(title)
    
    # Tight layout
    fig.tight_layout()
    
    return fig

def create_eq_response(eq_bands: Dict[float, float], sr: int = 44100, n_points: int = 1000,
                        color: str = '#1DB954', figsize: Tuple[int, int] = (10, 4)) -> Figure:
    """
    Create a frequency response visualization for an equalizer.
    
    Args:
        eq_bands: Dictionary of frequency bands and gain values
        sr: Sample rate
        n_points: Number of frequency points to calculate
        color: Color for the response curve
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure with frequency response plot
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    
    # Calculate frequency array (logarithmic)
    freqs = np.logspace(np.log10(20), np.log10(sr/2), n_points)
    
    # Initialize response array
    response = np.zeros(n_points)
    
    # Calculate response for each EQ band
    for freq, gain in eq_bands.items():
        # Skip if no gain
        if gain == 0:
            continue
            
        # Convert to float
        try:
            freq = float(freq)
            
            # Calculate bandwidth based on frequency
            bandwidth = max(freq * 0.15, 30)
            
            # Calculate frequency response for this band
            freq_response = np.zeros(n_points)
            for i, f in enumerate(freqs):
                # Simplified bell filter response calculation
                f_ratio = f / freq
                if f_ratio > 0:
                    # Calculate response using a simple bell curve approximation
                    # Width of bell curve varies with frequency
                    bell_width = 1.0 / (bandwidth / freq)
                    freq_response[i] = gain * np.exp(-((np.log10(f_ratio) ** 2) / (bell_width ** 2)))
            
            # Add to total response
            response += freq_response
        except Exception as e:
            print(f"Error calculating response for {freq}Hz: {e}")
    
    # Plot frequency response
    ax.semilogx(freqs, response, color=color, linewidth=2)
    
    # Add horizontal line at 0 dB
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Set labels and limits
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    ax.set_title('EQ Frequency Response')
    ax.set_xlim(20, sr/2)
    
    # Set reasonable y-axis limits based on the max absolute gain
    max_gain = max(12, np.max(np.abs(response)) * 1.2)
    ax.set_ylim(-max_gain, max_gain)
    
    # Add grid
    ax.grid(True, which='both', alpha=0.2)
    
    # Add frequency markers
    for f in [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
        if f <= sr/2:
            ax.axvline(x=f, color='lightgray', linestyle='-', alpha=0.2)
    
    # Tight layout
    fig.tight_layout()
    
    return fig

def create_comparison_plot(original: np.ndarray, processed: np.ndarray, 
                          sr: int, title: str = "Original vs Processed",
                          figsize: Tuple[int, int] = (10, 6)) -> Figure:
    """
    Create a comparison plot of original and processed audio.
    
    Args:
        original: Original audio data
        processed: Processed audio data
        sr: Sample rate
        title: Title for the plot
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure with comparison plot
    """
    fig = plt.figure(figsize=figsize)
    
    # Ensure both arrays are the same length
    min_len = min(len(original), len(processed))
    original = original[:min_len]
    processed = processed[:min_len]
    
    # Calculate time array
    time = np.linspace(0, min_len / sr, min_len)
    
    # Calculate spectrograms
    D_original = librosa.amplitude_to_db(np.abs(librosa.stft(original)), ref=np.max)
    D_processed = librosa.amplitude_to_db(np.abs(librosa.stft(processed)), ref=np.max)
    
    # Plot waveforms
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(time, original, color='#1DB954', linewidth=0.5)
    ax1.set_title('Original Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(time, processed, color='#2E77B5', linewidth=0.5)
    ax2.set_title('Processed Waveform')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    
    # Plot spectrograms
    ax3 = fig.add_subplot(2, 2, 3)
    img1 = librosa.display.specshow(D_original, sr=sr, x_axis='time', y_axis='log', ax=ax3, cmap='viridis')
    ax3.set_title('Original Spectrogram')
    
    ax4 = fig.add_subplot(2, 2, 4)
    img2 = librosa.display.specshow(D_processed, sr=sr, x_axis='time', y_axis='log', ax=ax4, cmap='viridis')
    ax4.set_title('Processed Spectrogram')
    
    # Add colorbar
    fig.colorbar(img2, ax=[ax3, ax4], format='%+2.0f dB')
    
    # Set main title
    fig.suptitle(title, fontsize=16)
    
    # Tight layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the main title
    
    return fig

def fig_to_base64(fig: Figure) -> str:
    """
    Convert a matplotlib figure to a base64-encoded string.
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Base64-encoded string of the figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_str

def create_compression_visualization(audio: np.ndarray, sr: int, threshold_db: float, 
                                    ratio: float, title: str = "Compression Visualization",
                                    figsize: Tuple[int, int] = (10, 6)) -> Figure:
    """
    Create a visualization of the compression effect on audio dynamics.
    
    Args:
        audio: Audio data
        sr: Sample rate
        threshold_db: Threshold in dB
        ratio: Compression ratio
        title: Title for the plot
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure with compression visualization
    """
    fig = plt.figure(figsize=figsize)
    
    # Calculate decibel amplitude
    eps = 1e-10  # To avoid log(0)
    db = 20 * np.log10(np.abs(audio) + eps)
    
    # Calculate compression curve
    x_db = np.linspace(-80, 0, 1000)
    y_db = np.copy(x_db)
    
    # Apply compression to the curve
    mask = x_db > threshold_db
    y_db[mask] = threshold_db + (x_db[mask] - threshold_db) / ratio
    
    # Plot histogram of original audio levels
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.hist(db[db > -80], bins=100, alpha=0.7, color='#1DB954')
    ax1.set_xlabel('Level (dB)')
    ax1.set_ylabel('Count')
    ax1.set_title('Audio Level Distribution')
    ax1.set_xlim(-80, 0)
    
    # Plot compression curve
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x_db, x_db, '--', color='gray', alpha=0.7, label='No Compression')
    ax2.plot(x_db, y_db, '-', color='#2E77B5', linewidth=2, label=f'Ratio: {ratio}:1')
    
    # Add threshold line
    ax2.axvline(x=threshold_db, color='red', linestyle='--', alpha=0.7, 
                label=f'Threshold: {threshold_db} dB')
    
    ax2.set_xlabel('Input Level (dB)')
    ax2.set_ylabel('Output Level (dB)')
    ax2.set_title('Compression Curve')
    ax2.set_xlim(-80, 0)
    ax2.set_ylim(-80, 0)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Set main title
    fig.suptitle(title, fontsize=16)
    
    # Tight layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the main title
    
    return fig 