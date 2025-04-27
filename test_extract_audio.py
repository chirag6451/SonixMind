#!/usr/bin/env python3
"""
Purpose: Test script for the extract_audio module
Objective: Demonstrate usage of the extract_audio function with various parameters
"""

from extract_audio import extract_audio
import os

def test_basic_extraction():
    """Test basic MP3 extraction with high quality"""
    # Using an existing video file in the directory
    video_file = "s1.MP4"
    
    if not os.path.exists(video_file):
        print(f"Please update the script with an existing video file path. '{video_file}' not found.")
        return
    
    # Basic extraction with default settings (high quality MP3)
    output_file = extract_audio(video_file)
    print(f"Basic extraction test completed. Output file: {output_file}")


def test_wav_extraction():
    """Test WAV extraction with high quality"""
    # Using an existing video file in the directory
    video_file = "s1.MP4"
    
    if not os.path.exists(video_file):
        print(f"Please update the script with an existing video file path. '{video_file}' not found.")
        return
    
    # Extract to WAV with high quality
    output_file = extract_audio(
        video_file,
        format="wav",
        quality="high"
    )
    print(f"WAV extraction test completed. Output file: {output_file}")


def test_partial_extraction():
    """Test extracting a portion of the audio with normalization"""
    # Using an existing video file in the directory
    video_file = "aks.mov"
    
    if not os.path.exists(video_file):
        print(f"Please update the script with an existing video file path. '{video_file}' not found.")
        return
    
    # Extract a portion of audio (10 seconds to 20 seconds) with normalization
    output_file = extract_audio(
        video_file,
        output_file="partial_audio.mp3",
        start_time=10.0,
        end_time=20.0,
        normalize_audio=True
    )
    print(f"Partial extraction test completed. Output file: {output_file}")


def test_custom_output():
    """Test extraction with custom output path"""
    # Using an existing video file in the directory
    video_file = "enhanced.mp4"
    
    if not os.path.exists(video_file):
        print(f"Please update the script with an existing video file path. '{video_file}' not found.")
        return
    
    # Extract with custom output path
    output_dir = "extracted_audio"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = extract_audio(
        video_file,
        output_file=os.path.join(output_dir, "custom_output.mp3"),
        quality="medium"
    )
    print(f"Custom output test completed. Output file: {output_file}")


if __name__ == "__main__":
    print("Running audio extraction tests...")
    
    # Uncomment the tests you want to run
    test_basic_extraction()
    # test_wav_extraction()
    # test_partial_extraction()
    # test_custom_output()
    
    print("Tests completed. To run additional tests, uncomment the test functions in this file.")
