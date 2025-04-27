#!/usr/bin/env python3
"""
Purpose: Test audio extraction from s1.MP4 file
Objective: Verify the extract_audio function works correctly with the s1.MP4 file
"""

from extract_audio import extract_audio
import os

def main():
    # Test file
    video_file = "s1.MP4"
    
    if not os.path.exists(video_file):
        print(f"Error: Test file '{video_file}' not found.")
        return
    
    print(f"Starting audio extraction from {video_file}...")
    
    # Test 1: Extract to MP3 with high quality
    mp3_output = extract_audio(
        input_file=video_file,
        output_file="s1_high_quality.mp3",
        format="mp3",
        quality="high",
        normalize_audio=True
    )
    print(f"MP3 extraction completed: {mp3_output}")
    
    # Test 2: Extract to WAV with high quality
    wav_output = extract_audio(
        input_file=video_file,
        output_file="s1_high_quality.wav",
        format="wav",
        quality="high",
        normalize_audio=True
    )
    print(f"WAV extraction completed: {wav_output}")
    
    print("Audio extraction tests completed successfully!")

if __name__ == "__main__":
    main()
