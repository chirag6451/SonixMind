# SonixMind User Guide

Welcome to SonixMind, a powerful audio and video processing application. This guide will walk you through all features and options available in the app.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Audio Extraction](#audio-extraction)
3. [Audio Processing](#audio-processing)
4. [Audio Mastering](#audio-mastering)
5. [YouTube Reference](#youtube-reference)
6. [Visualization Features](#visualization-features)
7. [Advanced Options](#advanced-options)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### Launching the Application
1. Open your terminal
2. Navigate to the SonixMind directory
3. Run the command: `streamlit run app.py`
4. The application will open in your default web browser at http://localhost:8501

### Interface Overview
- The left sidebar contains navigation and settings
- The main area displays the active tool and its controls
- The top bar shows the SonixMind logo and application title

## Audio Extraction

### Extracting Audio from Video
1. Navigate to the "Audio Extraction" section in the sidebar
2. Click "Upload Video" to select a video file (supported formats: MP4, MOV)
3. Once uploaded, the video preview will appear
4. Click "Extract Audio" to process the file
5. Wait for the extraction to complete (progress bar will show)
6. Upon completion, the extracted audio will be available for playback and download

### Extraction Settings
- **Format**: Choose between WAV and MP3 output formats
- **Sample Rate**: Select audio sample rate (default: 44.1kHz)
- **Channels**: Choose mono or stereo output
- **Normalize Audio**: Toggle to automatically normalize audio levels

## Audio Processing

### Basic Audio Processing
1. Navigate to the "Audio Processing" section
2. Upload an audio file or use previously extracted audio
3. Use the sliders to adjust audio parameters:
   - Volume (gain)
   - Bass boost
   - Treble
   - Clarity enhancement
4. Preview changes with the "Play Processed Audio" button
5. Click "Apply Processing" to finalize changes
6. Download the processed audio using the download button

### Noise Reduction
1. In the Audio Processing section, select the "Noise Reduction" tab
2. Choose noise reduction level (Low, Medium, High)
3. Optionally, configure advanced parameters:
   - Noise floor threshold
   - Attack time
   - Release time
4. Click "Preview" to hear a sample of the noise-reduced audio
5. Click "Apply Noise Reduction" to process the entire file

### Voice Enhancement
1. Select the "Voice Enhancement" tab in Audio Processing
2. Upload or select a voice recording
3. Choose enhancement type:
   - Clarity improvement
   - Voice isolation
   - VoiceFixer (AI-powered restoration)
4. Adjust settings for the selected enhancement type
5. Preview and apply changes

## Audio Mastering

### Master Using Reference Track
1. Navigate to the "Mastering" section
2. Upload or select your target audio for mastering
3. Upload a reference track or select from available references
4. Click "Analyze Both Tracks" to prepare for mastering
5. Adjust mastering settings:
   - Match loudness (LUFS target)
   - Match frequency balance
   - Match dynamics
6. Click "Apply Mastering" to process
7. Preview and download the mastered track

### Custom Mastering
1. In the Mastering section, select the "Custom" tab
2. Upload or select your audio file
3. Adjust the following settings:
   - EQ (multi-band equalizer)
   - Compression ratio and threshold
   - Limiting and saturation
   - Stereo width
4. Use the visualization to guide your adjustments
5. Click "Apply Custom Mastering" to process
6. Preview and download the result

## YouTube Reference

### Downloading References from YouTube
1. Navigate to the "YouTube Reference" section
2. Enter a YouTube URL or search term
3. Click "Search" to display results
4. Select a video from the results
5. Choose download quality (audio only)
6. Click "Download Reference"
7. The downloaded audio will be available in the reference library

### Managing Reference Library
1. Navigate to the "Reference Library" tab
2. View all downloaded references
3. Filter by genre, artist, or quality
4. Play references directly in the app
5. Delete unwanted references
6. Use references for audio mastering

## Visualization Features

### Waveform Analysis
1. Upload or select any audio file
2. Navigate to the "Visualization" section
3. View the detailed waveform display
4. Zoom in/out using the controls
5. Check levels with the level meter overlay

### Spectrogram View
1. In the Visualization section, select "Spectrogram"
2. View the frequency content of your audio over time
3. Adjust spectrogram settings:
   - Resolution
   - Color map
   - Frequency scale (linear or logarithmic)
4. Identify frequency issues or areas for improvement

### Audio Statistics
1. Select the "Statistics" tab in Visualization
2. View detailed audio metrics:
   - RMS level
   - Peak level
   - Loudness (LUFS)
   - Dynamic range
   - Crest factor
   - Frequency response graph

## Advanced Options

### Batch Processing
1. Navigate to "Advanced Options" in the sidebar
2. Select "Batch Processing"
3. Upload multiple files or select a folder
4. Configure processing settings to apply to all files
5. Click "Process All Files"
6. Download processed files individually or as a zip archive

### Custom Presets
1. In Advanced Options, select "Presets"
2. Create a new preset with your favorite settings
3. Save the preset with a descriptive name
4. Apply saved presets to future processing tasks
5. Export/import presets to share with others

### Project Management
1. Select "Projects" in Advanced Options
2. Create a new project
3. Add files and processing steps to your project
4. Save project progress
5. Resume work on saved projects

## Troubleshooting

### Common Issues
- **Application doesn't launch**: Ensure Python and all dependencies are installed
- **Upload fails**: Check file format compatibility and file size limits
- **Processing errors**: Check log for details, often related to file permissions
- **YouTube download issues**: May be related to YouTube API changes, update yt-dlp
- **Audio playback issues**: Check browser audio settings and permissions

### VoiceFixer Model Issues
If you experience problems with the VoiceFixer functionality:
1. Verify model files are correctly downloaded and placed in the appropriate directories
2. Check console for specific error messages
3. Try redownloading model files from the official source
4. Ensure your system meets the minimum requirements for running AI models

### Getting Help
- Check the GitHub repository for known issues
- Submit detailed bug reports including log information
- Join the community forum for user discussions
- Contact support at support@indapoint.com

---

For more information, visit [www.indapoint.com](https://www.indapoint.com/) or contact us at info@indapoint.com.

Copyright © 2023-2025 IndaPoint Technologies Pvt. Ltd. All rights reserved. 