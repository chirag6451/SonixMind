#!/usr/bin/env python3
"""
Purpose: Streamlit app for audio extraction and processing from video files
Objective: Provide a user-friendly interface to extract high-quality audio from videos
           and apply basic audio processing
"""

import streamlit as st
import tempfile
import os
import soundfile as sf
import numpy as np
from extract_audio import extract_audio

# Set page configuration
st.set_page_config(
    page_title="Video Audio Extractor",
    page_icon="🎵",
    layout="wide"
)

def apply_normalization(audio_data, sample_rate, output_path):
    """Apply audio normalization to make volume consistent"""
    # Calculate the RMS value
    rms = np.sqrt(np.mean(audio_data**2))
    
    # Calculate the target RMS (adjust as needed for desired loudness)
    target_rms = 0.1
    
    # Calculate the gain needed
    if rms > 0:
        gain = target_rms / rms
    else:
        gain = 1.0
    
    # Apply the gain
    normalized_audio = audio_data * gain
    
    # Clip to prevent distortion
    normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
    
    # Save the normalized audio
    sf.write(output_path, normalized_audio, sample_rate)
    return output_path

def apply_noise_reduction(audio_data, sample_rate, output_path, threshold=0.01):
    """Apply simple noise reduction by thresholding low amplitudes"""
    # Calculate the absolute values of the audio samples
    abs_audio = np.abs(audio_data)
    
    # Apply a simple threshold to reduce background noise
    noise_reduced = np.where(abs_audio > threshold, audio_data, 0)
    
    # Save the noise-reduced audio
    sf.write(output_path, noise_reduced, sample_rate)
    return output_path

# Main UI
st.title("🎵 Video Audio Extractor & Processor")
st.write("Upload your video file, extract high-quality audio, and apply optional processing")

# File upload
uploaded_file = st.file_uploader("Choose a video file (MP4 or MOV)", type=["mp4", "mov"])

if uploaded_file:
    # Save uploaded file temporarily
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
    temp_video.write(uploaded_file.read())
    temp_video.close()
    
    # Display video
    st.video(temp_video.name)
    
    # Settings
    st.header("⚙️ Extraction Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        audio_format = st.selectbox("Output Format", ["mp3", "wav"], index=0)
        audio_quality = st.selectbox("Audio Quality", ["high", "medium", "low"], index=0)
    
    with col2:
        normalize_audio = st.checkbox("Normalize Audio Levels", value=True)
        noise_reduction = st.checkbox("Apply Noise Reduction", value=False)
        
    # Time range selection (optional)
    use_time_range = st.checkbox("Extract Specific Time Range", value=False)
    
    start_time = None
    end_time = None
    
    if use_time_range:
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.number_input("Start Time (seconds)", min_value=0.0, value=0.0, step=1.0)
        with col2:
            end_time = st.number_input("End Time (seconds)", min_value=0.0, value=60.0, step=1.0)
    
    # Process button
    if st.button("🚀 Extract Audio"):
        with st.spinner("Extracting audio from video..."):
            # Create temporary files for processing
            temp_extracted = tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}")
            temp_processed = tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}")
            
            try:
                # Step 1: Extract audio from video
                extract_audio(
                    input_file=temp_video.name,
                    output_file=temp_extracted.name,
                    format=audio_format,
                    quality=audio_quality,
                    normalize_audio=False,  # We'll handle normalization separately
                    start_time=start_time if use_time_range else None,
                    end_time=end_time if use_time_range else None
                )
                
                # Step 2: Apply additional processing if selected
                current_file = temp_extracted.name
                
                # Load the extracted audio
                audio_data, sample_rate = sf.read(current_file)
                
                # Apply noise reduction if selected
                if noise_reduction:
                    with st.spinner("Applying noise reduction..."):
                        current_file = apply_noise_reduction(
                            audio_data, sample_rate, temp_processed.name
                        )
                        # Reload the processed audio for further processing
                        audio_data, sample_rate = sf.read(current_file)
                
                # Apply normalization if selected
                if normalize_audio:
                    with st.spinner("Normalizing audio levels..."):
                        current_file = apply_normalization(
                            audio_data, sample_rate, temp_processed.name
                        )
                else:
                    # If no processing was applied, use the extracted file
                    if not noise_reduction:
                        current_file = temp_extracted.name
                
                st.success("Audio extraction and processing complete!")
                
                # Display results
                st.header("🎧 Results")
                
                # Audio player
                st.subheader("Extracted Audio")
                st.audio(current_file, format=f"audio/{audio_format}")
                
                # Download button
                with open(current_file, "rb") as f:
                    file_extension = "mp3" if audio_format == "mp3" else "wav"
                    output_filename = f"{os.path.splitext(uploaded_file.name)[0]}.{file_extension}"
                    st.download_button(
                        label="💾 Download Processed Audio",
                        data=f,
                        file_name=output_filename,
                        mime=f"audio/{audio_format}"
                    )
            
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
            
            finally:
                # Clean up temporary files
                try:
                    os.unlink(temp_video.name)
                    os.unlink(temp_extracted.name)
                    if os.path.exists(temp_processed.name):
                        os.unlink(temp_processed.name)
                except Exception as e:
                    st.warning(f"Could not clean up temporary files: {str(e)}")

else:
    # Display instructions when no file is uploaded
    st.info("👆 Upload a video file to get started")
    
    st.markdown("""
    ### Features:
    - Extract high-quality audio from MP4 or MOV video files
    - Choose between MP3 or WAV output formats
    - Select quality level (bitrate and sample rate)
    - Extract specific time ranges from longer videos
    - Apply audio normalization to balance volume levels
    - Simple noise reduction for cleaner audio
    
    ### How it works:
    1. Upload your video file
    2. Configure extraction settings
    3. Click "Extract Audio"
    4. Preview and download the processed audio
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and FFmpeg")
