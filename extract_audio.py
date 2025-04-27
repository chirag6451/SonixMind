#!/usr/bin/env python3
"""
Purpose: Extract high-quality audio from video files (MP4 or MOV) to MP3 or WAV formats
Objective: Provide a flexible utility for extracting audio tracks from video files while 
           maintaining high audio quality through configurable parameters
"""

import os
import ffmpeg
from typing import Optional, Dict, Any
import subprocess
import tempfile


def extract_audio(
    input_file: str,
    output_file: Optional[str] = None,
    format: str = "mp3",
    quality: str = "medium",
    normalize_audio: bool = False,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> bool:
    """
    Extract audio from a video file using FFmpeg.
    
    Args:
        input_file: Path to the input video file
        output_file: Path where the output audio file will be saved
        format: Output format (mp3 or wav)
        quality: Audio quality preset (high, medium, low)
        normalize_audio: Whether to normalize audio levels
        start_time: Start time in seconds for extraction (optional)
        end_time: End time in seconds for extraction (optional)
    
    Returns:
        True if successful, False if an error occurred
    """
    # Define quality presets
    quality_presets = {
        "high": {
            "mp3": ["-c:a", "libmp3lame", "-b:a", "320k", "-ar", "44100"],
            "wav": ["-c:a", "pcm_s16le", "-ar", "44100"]
        },
        "medium": {
            "mp3": ["-c:a", "libmp3lame", "-b:a", "192k", "-ar", "44100"],
            "wav": ["-c:a", "pcm_s16le", "-ar", "44100"]
        },
        "low": {
            "mp3": ["-c:a", "libmp3lame", "-b:a", "128k", "-ar", "44100"],
            "wav": ["-c:a", "pcm_s16le", "-ar", "44100"]
        }
    }
    
    # Validate format
    if format not in ["mp3", "wav"]:
        print(f"Error: Unsupported format {format}. Using mp3 instead.")
        format = "mp3"
    
    # Validate quality
    if quality not in quality_presets:
        print(f"Error: Unsupported quality {quality}. Using medium instead.")
        quality = "medium"
    
    # Build the ffmpeg command
    command = ["ffmpeg", "-y", "-i", input_file]
    
    # Add segment extraction if specified
    if start_time is not None:
        command.extend(["-ss", str(start_time)])
    
    if end_time is not None:
        duration = end_time - (start_time or 0)
        command.extend(["-t", str(duration)])
    
    # Add quality settings
    command.extend(quality_presets[quality][format])
    
    # Add normalization if requested
    if normalize_audio:
        # For normalization, we'll use the loudnorm filter
        command.extend(["-af", "loudnorm=I=-16:LRA=11:TP=-1.5"])
    
    # Add output file
    command.append(output_file)
    
    # Execute the command
    try:
        subprocess.run(command, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        print(f"FFmpeg stderr: {e.stderr.decode()}")
        return False


# Example usage
if __name__ == "__main__":
    # Example usage
    input_video = "input.mp4"
    output_audio = "output.mp3"
    
    if os.path.exists(input_video):
        success = extract_audio(input_video, output_audio, quality="high", normalize_audio=True)
        if success:
            print(f"Audio extracted to {output_audio}")
        else:
            print("Audio extraction failed")
    else:
        print(f"Input file {input_video} does not exist")
    
    print("This module provides the extract_audio function for extracting high-quality audio from video files.")
    print("Import and use the function in your own scripts or modify the examples in this file.")
