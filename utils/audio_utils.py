"""
Audio utility functions for processing, analyzing, and manipulating audio files.
"""
import os
import tempfile
import subprocess
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple, Optional, Union
import ffmpeg
import scipy.signal as signal
import time
import psutil

def load_audio(file_path: str, sr: int = 44100) -> Tuple[np.ndarray, int]:
    """
    Load an audio file with memory optimization.
    
    Args:
        file_path: Path to the audio file
        sr: Sample rate for resampling
        
    Returns:
        Tuple of audio data as numpy array and sample rate
    """
    try:
        # For memory efficiency, we'll use librosa's load with sr=None first 
        # to check the file's native sample rate
        native_sr = librosa.get_samplerate(file_path)
        
        # If the native sample rate matches our target, use soundfile which is more memory efficient
        if native_sr == sr:
            audio, _ = sf.read(file_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # Convert to mono if stereo
        else:
            # If resampling is needed, use librosa which handles this well
            audio, _ = librosa.load(file_path, sr=sr, mono=True)
        
        return audio, sr
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return np.array([]), sr

def extract_audio(video_path: str, output_path: Optional[str] = None, 
                  start_time: Optional[float] = None, 
                  end_time: Optional[float] = None,
                  sr: int = 44100) -> str:
    """
    Extract audio from a video file with memory optimization.
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the extracted audio (if None, uses a temp file)
        start_time: Start time in seconds for extraction (optional)
        end_time: End time in seconds for extraction (optional)
        sr: Sample rate for the output audio
        
    Returns:
        Path to the extracted audio file
    """
    # Set process priority to low to prevent memory issues
    p = psutil.Process(os.getpid())
    original_nice = p.nice()
    try:
        p.nice(10)  # Lower priority
        
        # Create temp file if no output path provided
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
        
        # Build ffmpeg command
        ffmpeg_cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
                      '-ar', str(sr), '-ac', '1']
        
        # Add time segment if specified
        if start_time is not None:
            ffmpeg_cmd.extend(['-ss', str(start_time)])
        
        if end_time is not None:
            ffmpeg_cmd.extend(['-to', str(end_time)])
        
        # Add output path and overwrite flag
        ffmpeg_cmd.extend(['-y', output_path])
        
        # Run as subprocess with minimal memory usage
        process = subprocess.Popen(
            ffmpeg_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            bufsize=10**5  # Buffer size to reduce memory usage
        )
        
        # Wait for completion
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error extracting audio: {stderr.decode()}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return ""
        
        return output_path
    except Exception as e:
        print(f"Error in extract_audio: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return ""
    finally:
        # Restore original process priority
        p.nice(original_nice)

def process_audio(audio: np.ndarray, sr: int, 
                  eq_bands: dict = None, 
                  gain_db: float = 0.0,
                  compression_ratio: float = 1.0, 
                  threshold_db: float = 0.0) -> np.ndarray:
    """
    Process audio with EQ, gain, and compression.
    
    Args:
        audio: Audio data as numpy array
        sr: Sample rate
        eq_bands: Dictionary with frequency bands and gain values
        gain_db: Output gain in dB
        compression_ratio: Compression ratio (1.0 = no compression)
        threshold_db: Compression threshold in dB
        
    Returns:
        Processed audio data
    """
    processed = audio.copy()
    
    # Apply EQ if provided
    if eq_bands and len(eq_bands) > 0:
        processed = apply_eq(processed, sr, eq_bands)
    
    # Apply compression if ratio > 1.0
    if compression_ratio > 1.0:
        processed = apply_compression(processed, threshold_db, compression_ratio)
    
    # Apply gain
    if gain_db != 0:
        processed = apply_gain(processed, gain_db)
    
    return processed

def apply_eq(audio: np.ndarray, sr: int, eq_bands: dict) -> np.ndarray:
    """
    Apply equalization to audio using specified EQ bands.
    
    Args:
        audio: Audio data as numpy array
        sr: Sample rate
        eq_bands: Dictionary with frequency bands and gain values
        
    Returns:
        Equalized audio data
    """
    # Create a copy to avoid modifying the original
    equalized = audio.copy()
    
    # Process each EQ band
    for freq, gain in eq_bands.items():
        if gain == 0:  # Skip if no gain is applied
            continue
            
        # Convert gain from dB to linear
        gain_linear = 10 ** (gain / 20.0)
        
        # Design a bandpass filter for the frequency
        try:
            freq = float(freq)
            # Set bandwidth based on frequency (narrower at lower frequencies)
            bandwidth = max(freq * 0.15, 30)  # At least 30 Hz bandwidth
            
            # Create bandpass filter
            low_cutoff = max(20, freq - bandwidth/2)  # Ensure not below 20 Hz
            high_cutoff = min(sr/2 - 1, freq + bandwidth/2)  # Ensure not above Nyquist
            
            b, a = signal.butter(2, [low_cutoff/(sr/2), high_cutoff/(sr/2)], btype='band')
            
            # Apply filter to get the band-isolated signal
            band_signal = signal.lfilter(b, a, equalized)
            
            # Apply gain to the isolated band and add back to the signal
            equalized = equalized - band_signal + (band_signal * gain_linear)
        except Exception as e:
            print(f"Error applying EQ at {freq}Hz: {e}")
    
    return equalized

def apply_compression(audio: np.ndarray, threshold_db: float, ratio: float) -> np.ndarray:
    """
    Apply dynamic range compression to the audio.
    
    Args:
        audio: Audio data as numpy array
        threshold_db: Threshold in dB
        ratio: Compression ratio
        
    Returns:
        Compressed audio data
    """
    # Convert threshold from dB to amplitude
    threshold = 10 ** (threshold_db / 20.0)
    
    # Calculate amplitude of audio
    amplitude = np.abs(audio)
    
    # Initialize output array
    compressed = np.zeros_like(audio)
    
    # Apply compression - above threshold, reduce by ratio
    mask_above = amplitude > threshold
    gain_above = threshold + (amplitude[mask_above] - threshold) / ratio
    compressed_above = np.sign(audio[mask_above]) * gain_above
    
    # Below threshold, keep the same
    compressed_below = audio[~mask_above]
    
    # Combine results
    compressed[mask_above] = compressed_above
    compressed[~mask_above] = compressed_below
    
    return compressed

def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """
    Apply gain to audio.
    
    Args:
        audio: Audio data as numpy array
        gain_db: Gain in dB
        
    Returns:
        Amplified audio data
    """
    # Convert dB to linear gain
    gain = 10 ** (gain_db / 20.0)
    
    # Apply gain
    return audio * gain

def save_audio(audio: np.ndarray, sr: int, file_path: str, format: str = 'wav', 
               bit_depth: int = 16) -> bool:
    """
    Save audio to file with specified format and bit depth.
    
    Args:
        audio: Audio data as numpy array
        sr: Sample rate
        file_path: Output file path
        format: Output format ('wav', 'mp3', etc.)
        bit_depth: Bit depth for output file
    
    Returns:
        Success status (bool)
    """
    try:
        # Set subtype based on bit depth
        if bit_depth == 16:
            subtype = 'PCM_16'
        elif bit_depth == 24:
            subtype = 'PCM_24'
        elif bit_depth == 32:
            subtype = 'FLOAT'
        else:
            subtype = 'PCM_16'  # Default to 16-bit
            
        # Save using soundfile
        sf.write(file_path, audio, sr, format=format, subtype=subtype)
        return True
    except Exception as e:
        print(f"Error saving audio: {e}")
        return False 