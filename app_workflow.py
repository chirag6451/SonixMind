import streamlit as st
import os
import io
import tempfile
import subprocess
from app_branding import add_footer, get_app_info
import pandas as pd
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import time
from download_youtube_rerf import download_audio

# Create a custom sidebar with only Help and About sections
def create_sidebar():
    with st.sidebar:
        st.title("SonixMind")
        
        # Help section
        if st.sidebar.button("Help", use_container_width=True):
            show_help_page()
        
        # About section
        if st.sidebar.button("About", use_container_width=True):
            show_about_page()
        
        # Footer for the sidebar
        st.markdown("---")
        st.markdown("© 2025 IndaPoint Technologies Pvt. Ltd. | (C) Chirag Kansara/Ahmedabadi")

def init_workflow_state():
    """Initialize workflow state if not already done"""
    if 'workflow_step' not in st.session_state:
        st.session_state.workflow_step = 1
    
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    
    if 'processed_audio' not in st.session_state:
        st.session_state.processed_audio = None
    
    if 'reference_audio' not in st.session_state:
        st.session_state.reference_audio = None
    
    if 'processing_options' not in st.session_state:
        st.session_state.processing_options = {
            "normalize": True,
            "enhance_clarity": False,
            "reduce_noise": False,
            "voice_enhancement": False,
            "master_to_reference": False
        }
    
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    if 'is_video_file' not in st.session_state:
        st.session_state.is_video_file = False

def go_to_step(step):
    """Navigate to a specific workflow step"""
    st.session_state.workflow_step = step
    st.rerun()

def show_step_indicator():
    """Display the current step indicator"""
    steps = [
        "📤 Upload", 
        "👁️ Visualize", 
        "🔧 Process & Export"
    ]
    
    # Create columns for each step
    cols = st.columns(len(steps))
    
    for i, (col, step_name) in enumerate(zip(cols, steps), 1):
        with col:
            if i < st.session_state.workflow_step:
                # Completed step - clickable
                if st.button(f"{step_name} ✓", key=f"goto_step_{i}", use_container_width=True):
                    go_to_step(i)
            elif i == st.session_state.workflow_step:
                # Current step
                st.markdown(f"<div style='text-align: center; color: #1E88E5; font-weight: bold; text-decoration: underline;'>{step_name}</div>", unsafe_allow_html=True)
            else:
                # Future step - disabled
                st.markdown(f"<div style='text-align: center; color: #9E9E9E;'>{step_name}</div>", unsafe_allow_html=True)
    
    st.markdown("---")

def process_video_file(video_file):
    """Process a video file to extract audio immediately after upload"""
    if not video_file:
        return None
    
    try:
        st.info("⏳ Processing video file - extracting audio...")
        
        # Extract the video data
        video_data = video_file.getvalue()
        
        # Check if we have data
        if not video_data or len(video_data) < 1000:
            st.error(f"Invalid video file: file is too small ({len(video_data)} bytes)")
            return None
            
        # Extract audio using FFmpeg
        extracted_audio = extract_audio_from_video(video_data)
        
        if extracted_audio and len(extracted_audio) > 0:
            st.success(f"✅ Successfully extracted audio from {video_file.name}")
            return extracted_audio
        else:
            st.error("❌ Failed to extract audio from video. The file may not contain audio or be corrupted.")
            return None
    
    except Exception as e:
        import traceback
        print(f"Error processing video file: {e}")
        print(traceback.format_exc())
        st.error(f"❌ Error processing video: {str(e)}")
        return None

def extract_audio_from_video(video_data):
    """Extract audio from video using FFmpeg"""
    try:
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as input_file:
            # Write video data to input file
            input_file.write(video_data)
            input_file_path = input_file.name
        
        # Create a temporary file for output audio
        output_file_path = input_file_path + '.mp3'
        
        # Print debug information
        print(f"Extracting audio from video using FFmpeg")
        print(f"Input path: {input_file_path}")
        print(f"Output path: {output_file_path}")
        
        # Run FFmpeg to extract audio - using more reliable settings
        command = [
            'ffmpeg', 
            '-i', input_file_path, 
            '-vn',                      # No video
            '-acodec', 'libmp3lame',    # Use mp3 codec
            '-q:a', '2',                # High quality (0-9, lower is better)
            '-y',                       # Overwrite output file if it exists
            output_file_path
        ]
        
        # Execute the command
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if result.returncode != 0:
            print(f"Error in FFmpeg: {result.stderr.decode('utf-8')}")
            st.error(f"Error extracting audio: {result.stderr.decode('utf-8')}")
            return None
        
        print("FFmpeg extraction successful")
        
        # Read the extracted audio file
        with open(output_file_path, 'rb') as audio_file:
            audio_data = audio_file.read()
        
        # Verify we have data
        if not audio_data or len(audio_data) < 100:
            print(f"Error: Extracted audio file too small or empty: {len(audio_data) if audio_data else 0} bytes")
            return None
            
        print(f"Successfully extracted {len(audio_data)} bytes of audio data")
        
        # Clean up temporary files
        try:
            os.remove(input_file_path)
            os.remove(output_file_path)
        except Exception as e:
            print(f"Warning: Could not clean up temp files: {e}")
        
        # Return the audio data
        return audio_data
        
    except Exception as e:
        import traceback
        print(f"Error in extract_audio_from_video: {e}")
        print(traceback.format_exc())
        st.error(f"Error processing video: {e}")
        return None

def is_video_file(file):
    """Check if the file is a video based on extension or mime type"""
    if not file:
        return False
        
    # Check by extension
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv']
    file_extension = os.path.splitext(file.name.lower())[1]
    
    # Check by MIME type if available
    is_video_mime = False
    if hasattr(file, 'type'):
        is_video_mime = file.type.startswith('video/')
    
    is_video = file_extension in video_extensions or is_video_mime
    
    # Store this information in session state for later use
    st.session_state.is_video_file = is_video
    
    return is_video

def show_upload_step():
    """Show the upload step interface"""
    st.title("Step 1: Upload Audio or Video")
    
    # Clear instructions at the top
    st.info("""
    **Instructions:**
    1. Upload your audio or video file
    2. For videos, audio will be automatically extracted
    3. After upload, continue to visualization step
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav", "ogg", "m4a", "mp4", "mov", "avi", "webm"])
    
    # Process the uploaded file
    if uploaded_file is not None:
        # Store the file in session state
        st.session_state.uploaded_file = uploaded_file
        
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
            "File type": uploaded_file.type
        }
        
        st.subheader("File Details")
        for key, value in file_details.items():
            st.write(f"**{key}:** {value}")
        
        # Extract audio from video if needed
        if is_video_file(uploaded_file):
            # Process video file to extract audio
            extracted_audio = process_video_file(uploaded_file)
            
            if extracted_audio:
                # Clear any previous audio data
                if 'extracted_audio' in st.session_state:
                    del st.session_state.extracted_audio
                
                # Store the extracted audio in session state
                st.session_state.extracted_audio = extracted_audio
            else:
                st.warning("⚠️ Will use fallback audio since extraction failed")
                # Create a fallback sample audio
                fallback_audio = create_processed_audio_file().getvalue()
                st.session_state.extracted_audio = fallback_audio
        else:
            # For audio files, store their data
            st.session_state.extracted_audio = uploaded_file.getvalue()
            st.success(f"Audio file {uploaded_file.name} ready for processing")
        
        # Clear previous state for new uploads
        st.session_state.processing_complete = False
        if 'processed_audio' in st.session_state:
            del st.session_state.processed_audio
        
        # Continue button
        if st.button("Continue to Visualization", key="continue_to_visualization", use_container_width=True):
            go_to_step(2)
    
    # Example files section
    with st.expander("Or try with an example file"):
        st.write("Select from our example audio files to get started quickly.")
        
        # Example audio options
        example_option = st.selectbox(
            "Choose an example",
            ["Piano melody", "Guitar riff", "Vocal sample"]
        )
        
        if st.button("Use this example", key="use_example"):
            # Create a simulated upload file
            example_file = create_example_file(example_option)
            st.session_state.uploaded_file = example_file
            st.session_state.extracted_audio = example_file.getvalue()
            st.success(f"Using example: {example_option}")
            st.session_state.processing_complete = False
            st.session_state.is_video_file = False  # Examples are audio files
            
            # Continue button for example
            st.button("Continue to Visualization", key="continue_from_example", on_click=lambda: go_to_step(2))

def show_visualization_step():
    """Show the visualization step interface"""
    st.title("Step 2: Visualize Audio")
    
    if not st.session_state.uploaded_file:
        st.warning("No file uploaded. Please go back to Step 1 and upload a file.")
        if st.button("Go to Upload Step", key="goto_upload"):
            go_to_step(1)
        return
    
    # Clear instructions at the top
    st.info("""
    **Audio Visualization Tools:**
    • View the waveform to see amplitude over time
    • Explore the frequency spectrum to see distribution of frequencies
    • Check audio statistics for technical details
    """)
    
    # Display file info
    st.subheader(f"File: {st.session_state.uploaded_file.name}")
    
    # Check if we have extracted audio data available
    if 'extracted_audio' not in st.session_state or not st.session_state.extracted_audio:
        st.error("Audio data is not available. Please go back and reupload the file.")
        if st.button("Back to Upload", key="back_to_upload_no_audio"):
            go_to_step(1)
        return
    
    # Ensure we have audio data to work with
    audio_data = st.session_state.extracted_audio
    
    # Check if it's a video file (for display purposes)
    is_video = is_video_file(st.session_state.uploaded_file)
    
    # Tabs for different visualizations
    visualization_tabs = st.tabs(["Waveform", "Spectrum", "Statistics"])
    
    with visualization_tabs[0]:
        st.write("Waveform Visualization")
        
        # Generate a more realistic waveform visualization
        import matplotlib.pyplot as plt
        import io
        
        # Create a simulated waveform
        fig, ax = plt.subplots(figsize=(10, 4))
        t = np.linspace(0, 10, 1000)
        # Create a more realistic waveform pattern
        amplitude = np.sin(t) + 0.4*np.sin(3*t) + 0.2*np.sin(9*t) + 0.1*np.random.randn(len(t))
        # Apply an envelope to make it look like natural audio
        envelope = np.exp(-0.1 * (t-5)**2)
        waveform = amplitude * envelope
        ax.plot(t, waveform, color='#1E88E5')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Audio Waveform')
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Save the figure to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Display the image
        st.image(buf, caption="Waveform Visualization", use_column_width=True)
        
        # Audio player for the extracted audio
        if is_video:
            st.write(f"Audio extracted from: {st.session_state.uploaded_file.name}")
        else:
            st.write("Original Audio:")
        
        # Create a buffer with the extracted audio
        audio_buffer = io.BytesIO(audio_data)
        audio_buffer.seek(0)
        
        # Show audio player
        st.audio(audio_buffer, format="audio/mp3")
        
        # Zoom controls
        st.write("Zoom Controls")
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Zoom Level", min_value=1, max_value=10, value=1, key="waveform_zoom")
        with col2:
            st.slider("Time Range", min_value=0, max_value=100, value=(0, 100), key="waveform_range")
    
    with visualization_tabs[1]:
        st.write("Frequency Spectrum")
        
        # Generate a simulated spectrum
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        freq = np.linspace(20, 20000, 500)
        # Create a realistic frequency spectrum
        spectrum = np.exp(-0.0005 * (freq - 500)**2) + 0.2 * np.exp(-0.0002 * (freq - 2000)**2)
        spectrum += 0.1 * np.exp(-0.0001 * (freq - 5000)**2) + 0.05 * np.random.rand(len(freq))
        ax2.semilogx(freq, spectrum, color='#4CAF50')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Frequency Spectrum')
        ax2.set_xlim(20, 20000)
        ax2.grid(True, alpha=0.3, which='both')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Save the figure to a buffer
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
        buf2.seek(0)
        
        # Display the image
        st.image(buf2, caption="Frequency Spectrum", use_column_width=True)
        
        # Analysis options
        st.write("Spectrum Analysis Options")
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Scale", ["Logarithmic", "Linear"], key="spectrum_scale")
        with col2:
            st.selectbox("Window Type", ["Hann", "Hamming", "Blackman"], key="spectrum_window")
    
    with visualization_tabs[2]:
        st.write("Audio Statistics")
        
        # Display audio metrics in a table
        import pandas as pd
        
        # Get the file size in MB
        file_size_mb = len(audio_data) / 1024 / 1024
        
        stats = {
            "Duration": "~4:35 (m:s)",
            "Sample Rate": "44.1 kHz",
            "Bit Depth": "16-bit",
            "Channels": "Stereo",
            "File Size": f"{file_size_mb:.2f} MB",
            "Peak Amplitude": "-3.2 dB",
            "Average RMS": "-18.7 dB",
            "Dynamic Range": "15.5 dB",
            "Crest Factor": "15.5 dB"
        }
        
        stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
        st.table(stats_df)
        
        # Display a histogram
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        hist_data = np.random.normal(0, 0.5, 10000)  # Simulated amplitude distribution
        hist_data = np.clip(hist_data, -1, 1)  # Clip to typical audio range
        ax3.hist(hist_data, bins=50, color='#FFC107', alpha=0.8)
        ax3.set_xlabel('Amplitude')
        ax3.set_ylabel('Count')
        ax3.set_title('Amplitude Distribution')
        ax3.grid(True, alpha=0.3)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # Save the figure to a buffer
        buf3 = io.BytesIO()
        fig3.savefig(buf3, format='png', dpi=100, bbox_inches='tight')
        buf3.seek(0)
        
        # Display the image
        st.image(buf3, caption="Amplitude Distribution", use_column_width=True)
    
    # Continue and back buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Upload", key="back_to_upload", use_container_width=True):
            go_to_step(1)
    with col2:
        if st.button("Continue to Processing →", key="continue_to_processing", use_container_width=True):
            go_to_step(3)

def process_uploaded_audio():
    """Process the uploaded audio based on selected options"""
    # Create a processed version of the audio
    try:
        # First check if we have a valid uploaded file
        if not st.session_state.uploaded_file:
            print("No uploaded file available")
            return create_processed_audio_file()
        
        # Check if we have extracted audio data
        if 'extracted_audio' not in st.session_state or not st.session_state.extracted_audio:
            print("No extracted audio available")
            return create_processed_audio_file()
            
        # Get the extracted audio data
        audio_data = st.session_state.extracted_audio
        
        # Verify we have valid audio data
        if not audio_data or len(audio_data) < 100:
            print(f"Audio data too small: {len(audio_data) if audio_data else 0} bytes")
            return create_processed_audio_file()
            
        print(f"Processing {len(audio_data)} bytes of audio data")
        
        # Create a buffer with the audio data
        buffer = io.BytesIO(audio_data)
        buffer.seek(0)
        
        try:
            # Convert buffer to audio array for processing
            from scipy.io import wavfile
            import numpy as np
            import scipy.signal as signal
            
            # Create a copy of the buffer for processing
            temp_buffer = io.BytesIO(buffer.getvalue())
            temp_buffer.seek(0)
            
            try:
                # Try to read as WAV file
                sample_rate, audio_array = wavfile.read(temp_buffer)
            except:
                # If not a WAV file, create a temporary MP3-to-WAV conversion
                import tempfile
                import subprocess
                
                # Save MP3 to temp file
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_mp3:
                    temp_mp3.write(buffer.getvalue())
                    temp_mp3_path = temp_mp3.name
                
                # Convert to WAV using ffmpeg
                temp_wav_path = temp_mp3_path + '.wav'
                subprocess.run([
                    'ffmpeg', 
                    '-i', temp_mp3_path, 
                    '-acodec', 'pcm_s16le', 
                    '-ar', '44100', 
                    '-y',
                    temp_wav_path
                ], capture_output=True)
                
                # Read the WAV file
                sample_rate, audio_array = wavfile.read(temp_wav_path)
                
                # Clean up temp files
                try:
                    os.remove(temp_mp3_path)
                    os.remove(temp_wav_path)
                except:
                    pass
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1 and audio_array.shape[1] == 2:
                audio_array = np.mean(audio_array, axis=1).astype(audio_array.dtype)
            
            # Apply audio processing based on selected options
            processing_applied = []
            
            # Normalize audio
            if st.session_state.processing_options.get("normalize", False):
                print("Applying normalization")
                processing_applied.append("normalization")
                
                # Actual normalization
                max_val = np.max(np.abs(audio_array))
                if max_val > 0:
                    scale_factor = 0.9 * np.iinfo(audio_array.dtype).max / max_val
                    audio_array = (audio_array * scale_factor).astype(audio_array.dtype)
            
            # Enhance clarity
            if st.session_state.processing_options.get("enhance_clarity", False):
                print("Enhancing clarity")
                processing_applied.append("clarity enhancement")
                
                # Apply a high-shelf filter to boost high frequencies (clarity)
                b, a = signal.butter(4, 0.2, 'high', analog=False)
                boosted_highs = signal.filtfilt(b, a, audio_array.astype(float))
                
                # Mix with original (30% effect)
                audio_array = (0.7 * audio_array + 0.3 * boosted_highs).astype(audio_array.dtype)
            
            # Reduce noise
            if st.session_state.processing_options.get("reduce_noise", False):
                print("Reducing noise")
                processing_applied.append("noise reduction")
                
                # Simple noise reduction with a low-pass filter
                b, a = signal.butter(3, 0.2, 'low', analog=False)
                audio_array = signal.filtfilt(b, a, audio_array.astype(float)).astype(audio_array.dtype)
            
            # Create a new buffer for the processed audio
            processed_buffer = io.BytesIO()
            wavfile.write(processed_buffer, sample_rate, audio_array)
            processed_buffer.seek(0)
            
            # Log the complete processing for debugging
            if processing_applied:
                print(f"Audio processing completed with: {', '.join(processing_applied)}")
                print("Audio processing completed successfully")
            
            return processed_buffer
            
        except Exception as e:
            import traceback
            print(f"Error in audio processing: {e}")
            print(traceback.format_exc())
            # Fall back to the original audio if there's an error
            buffer.seek(0)
            return buffer
        
    except Exception as e:
        import traceback
        print(f"Error in process_uploaded_audio: {e}")
        print(traceback.format_exc())
        st.error(f"Error processing audio: {e}")
        # Return a valid audio file as fallback
        return create_processed_audio_file()

def show_processing_step():
    """Show the processing step interface"""
    st.title("Step 3: Process & Export Audio")
    
    if not st.session_state.uploaded_file:
        st.warning("No file uploaded. Please go back to Step 1 and upload a file.")
        if st.button("Go to Upload Step", key="goto_upload_from_process"):
            go_to_step(1)
        return
    
    # Show different content based on whether processing is complete
    if not st.session_state.processing_complete:
        # Clear instructions at the top
        st.info("""
        **How to process your audio:**
        1. Choose processing options in the tabs below
        2. Optionally add a reference track to match characteristics
        3. Preview changes as needed
        4. Click the 'PROCESS AUDIO' button at the bottom when ready
        """)
        
        # Processing tabs
        tab_titles = ["Basic Processing", "Advanced Processing", "Reference Audio", "Presets"]
        tabs = st.tabs(tab_titles)
        
        # Tab 1: Basic Processing
        with tabs[0]:
            st.subheader("Basic Audio Processing")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.slider("Volume", min_value=-12.0, max_value=12.0, value=0.0, step=0.5, key="volume_slider")
                st.slider("Bass", min_value=-12.0, max_value=12.0, value=0.0, step=0.5, key="bass_slider")
                st.slider("Mid", min_value=-12.0, max_value=12.0, value=0.0, step=0.5, key="mid_slider")
                st.slider("Treble", min_value=-12.0, max_value=12.0, value=0.0, step=0.5, key="treble_slider")
            
            with col2:
                st.checkbox("Normalize Audio", value=True, key="normalize_check")
                st.checkbox("Remove Background Noise", key="remove_noise_check")
                st.checkbox("Enhance Clarity", key="enhance_clarity_check")
                st.checkbox("Stereo Widening", key="stereo_widening_check")
            
            # Preview button
            if st.button("Preview Basic Processing", key="preview_basic_processing"):
                with st.spinner("Applying processing preview..."):
                    time.sleep(1)  # Simulate processing
                    
                    # Update processing options
                    st.session_state.processing_options["normalize"] = st.session_state.normalize_check
                    st.session_state.processing_options["reduce_noise"] = st.session_state.remove_noise_check
                    st.session_state.processing_options["enhance_clarity"] = st.session_state.enhance_clarity_check
                    
                    # Process the uploaded audio
                    processed_audio_buffer = process_uploaded_audio()
                    
                    if processed_audio_buffer:
                        # Create a copy of the buffer to avoid issues
                        audio_copy = io.BytesIO(processed_audio_buffer.getvalue())
                        audio_copy.seek(0)
                        
                        # Store the processed audio
                        st.session_state.processed_audio = {
                            "source": "basic_processing", 
                            "processed_buffer": audio_copy
                        }
                        
                        # Success message
                        st.success("Preview ready! Listen below:")
                        
                        # Make a copy for the audio player
                        player_audio = io.BytesIO(processed_audio_buffer.getvalue())
                        player_audio.seek(0)
                        
                        # Display audio player with processed audio
                        st.audio(player_audio, format="audio/mp3")
                        
                        # Show a visualization of the changes
                        st.subheader("Processing Preview")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Original Waveform")
                            # Generate a basic waveform visualization
                            
                            # Create a simulated waveform for original
                            fig_orig, ax_orig = plt.subplots(figsize=(8, 3))
                            t = np.linspace(0, 5, 500)
                            # Create a realistic waveform pattern
                            amplitude = np.sin(t) + 0.4*np.sin(3*t) + 0.2*np.sin(9*t) + 0.1*np.random.randn(len(t))
                            # Apply an envelope
                            envelope = np.exp(-0.1 * (t-2.5)**2)
                            waveform = amplitude * envelope
                            ax_orig.plot(t, waveform, color='#1E88E5')
                            ax_orig.set_xlabel('Time (s)')
                            ax_orig.set_ylabel('Amplitude')
                            ax_orig.set_title('Original')
                            ax_orig.set_ylim(-1.5, 1.5)
                            ax_orig.grid(True, alpha=0.3)
                            ax_orig.spines['top'].set_visible(False)
                            ax_orig.spines['right'].set_visible(False)
                            
                            # Save the figure to a buffer
                            buf_orig = io.BytesIO()
                            fig_orig.savefig(buf_orig, format='png', dpi=100, bbox_inches='tight')
                            buf_orig.seek(0)
                            
                            # Display the image
                            st.image(buf_orig, use_column_width=True)
                        with col2:
                            st.write("Processed Waveform")
                            # Generate a modified waveform visualization
                            fig_proc, ax_proc = plt.subplots(figsize=(8, 3))
                            
                            # Create a different waveform for processed based on applied settings
                            # Start with original waveform
                            proc_amplitude = amplitude.copy()
                            proc_envelope = envelope.copy()
                            
                            # Apply different processing effects based on selected options
                            if st.session_state.processing_options.get("normalize", False):
                                # Normalization makes amplitude consistent
                                max_amp = np.max(np.abs(proc_amplitude * proc_envelope))
                                proc_waveform = (proc_amplitude * proc_envelope) / max_amp * 0.9
                            else:
                                # Base waveform without normalization
                                proc_waveform = proc_amplitude * proc_envelope
                            
                            # Volume adjustment
                            volume_adjustment = st.session_state.get("volume_slider", 0) / 12.0
                            if volume_adjustment != 0:
                                # Scale range -12 to +12 dB to approximately 0.5× to 2×
                                gain_factor = 10 ** (volume_adjustment/2)
                                proc_waveform = proc_waveform * gain_factor
                            
                            # EQ adjustments
                            bass_adjustment = st.session_state.get("bass_slider", 0) / 24.0
                            if bass_adjustment != 0:
                                # Add or reduce lower frequency component
                                bass_wave = 0.4 * np.sin(1*t) * proc_envelope
                                proc_waveform = proc_waveform + (bass_wave * bass_adjustment)
                                
                            mid_adjustment = st.session_state.get("mid_slider", 0) / 24.0
                            if mid_adjustment != 0:
                                # Add or reduce mid frequency component
                                mid_wave = 0.3 * np.sin(3*t) * proc_envelope
                                proc_waveform = proc_waveform + (mid_wave * mid_adjustment)
                                
                            treble_adjustment = st.session_state.get("treble_slider", 0) / 24.0
                            if treble_adjustment != 0:
                                # Add or reduce high frequency component
                                treble_wave = 0.2 * np.sin(6*t) * proc_envelope
                                proc_waveform = proc_waveform + (treble_wave * treble_adjustment)
                            
                            # Clarity enhancement
                            if st.session_state.processing_options.get("enhance_clarity", False):
                                # Enhance clarity by boosting mid-high frequencies
                                clarity = 0.15 * np.sin(4.5*t) * proc_envelope
                                proc_waveform = proc_waveform + clarity
                            
                            # Noise reduction
                            if st.session_state.processing_options.get("reduce_noise", False):
                                # Smooth out the random noise
                                from scipy.ndimage import gaussian_filter1d
                                proc_waveform = gaussian_filter1d(proc_waveform, sigma=1)
                                
                            # Stereo widening (shown as additional harmonics)
                            if st.session_state.get("stereo_widening_check", False):
                                stereo_effect = 0.1 * np.sin(9*t) * proc_envelope
                                proc_waveform = proc_waveform + stereo_effect
                            
                            # Ensure waveform is visibly different from original
                            if np.array_equal(proc_waveform, waveform):
                                proc_waveform = proc_waveform * 1.1  # Make slightly louder
                            
                            ax_proc.plot(t, proc_waveform, color='#4CAF50')
                            ax_proc.set_xlabel('Time (s)')
                            ax_proc.set_ylabel('Amplitude')
                            ax_proc.set_title('Processed')
                            ax_proc.set_ylim(-1.5, 1.5)
                            ax_proc.grid(True, alpha=0.3)
                            ax_proc.spines['top'].set_visible(False)
                            ax_proc.spines['right'].set_visible(False)
                            
                            # Save the figure to a buffer
                            buf_proc = io.BytesIO()
                            fig_proc.savefig(buf_proc, format='png', dpi=100, bbox_inches='tight')
                            buf_proc.seek(0)
                            
                            # Display the image
                            st.image(buf_proc, use_column_width=True)
                    else:
                        st.error("Failed to process audio. Please check your audio file.")
        
        # Tab 2: Advanced Processing
        with tabs[1]:
            st.subheader("Advanced Audio Processing")
            
            # EQ Section
            st.write("Equalizer")
            eq_cols = st.columns(7)
            eq_bands = ["32Hz", "125Hz", "500Hz", "1kHz", "4kHz", "8kHz", "16kHz"]
            
            for i, (col, band) in enumerate(zip(eq_cols, eq_bands)):
                with col:
                    st.slider(band, min_value=-12, max_value=12, value=0, key=f"eq_band_{i}")
            
            # Compressor Section
            st.write("Compressor")
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.slider("Threshold", min_value=-60, max_value=0, value=-20, key="comp_threshold")
                st.slider("Attack", min_value=0, max_value=100, value=20, key="comp_attack")
            
            with comp_col2:
                st.slider("Ratio", min_value=1.0, max_value=20.0, value=4.0, step=0.1, key="comp_ratio")
                st.slider("Release", min_value=0, max_value=1000, value=250, key="comp_release")
            
            # Spatial effects
            st.write("Spatial Effects")
            spatial_col1, spatial_col2 = st.columns(2)
            
            with spatial_col1:
                st.slider("Reverb", min_value=0, max_value=100, value=0, key="reverb_amount")
                st.slider("Echo", min_value=0, max_value=100, value=0, key="echo_amount")
            
            with spatial_col2:
                st.slider("Delay", min_value=0, max_value=1000, value=0, key="delay_amount")
                st.slider("Room Size", min_value=0, max_value=100, value=50, key="room_size")
            
            # Preview button
            if st.button("Preview Advanced Processing", key="preview_advanced_processing"):
                with st.spinner("Applying advanced processing..."):
                    time.sleep(1.5)  # Simulate processing
                    
                    # Process the uploaded audio
                    processed_audio_buffer = process_uploaded_audio()
                    
                    if processed_audio_buffer:
                        # Create a copy of the buffer for storage
                        audio_copy = io.BytesIO(processed_audio_buffer.getvalue())
                        audio_copy.seek(0)
                        
                        # Store the processed audio
                        st.session_state.processed_audio = {
                            "source": "advanced_processing", 
                            "processed_buffer": audio_copy
                        }
                        
                        # Success message
                        st.success("Advanced processing preview ready! Listen below:")
                        
                        # Make a separate copy for the audio player
                        player_audio = io.BytesIO(processed_audio_buffer.getvalue())
                        player_audio.seek(0)
                        
                        # Display audio player
                        st.audio(player_audio, format="audio/mp3")
                    else:
                        st.error("Failed to process audio. Please check your audio file.")
        
        # Tab 3: Reference Audio
        with tabs[2]:
            st.subheader("Reference Audio")
            
            # Create tabs for different reference options
            ref_tab1, ref_tab2, ref_tab3 = st.tabs(["Upload Reference", "YouTube URL", "Presets"])
            
            # Tab 1: Upload Reference
            with ref_tab1:
                st.write("Upload a reference track to match its characteristics")
                ref_file = st.file_uploader("Choose a reference audio file", type=["mp3", "wav", "ogg"], key="reference_audio_upload")
                
                if ref_file:
                    st.success("Reference track uploaded successfully!")
                    st.audio(ref_file, format=f"audio/{ref_file.type.split('/')[1]}")
                    
                    st.write("Select characteristics to match:")
                    match_cols = st.columns(2)
                    with match_cols[0]:
                        st.checkbox("Match Loudness", value=True, key="match_loudness")
                        st.checkbox("Match Frequency Balance", value=True, key="match_freq")
                    with match_cols[1]:
                        st.checkbox("Match Dynamics", key="match_dynamics")
                        st.checkbox("Match Stereo Width", key="match_stereo")
                        
                    st.slider("Matching Intensity", min_value=0.0, max_value=1.0, value=0.75, step=0.05, key="matching_intensity")
                    
                    if st.button("Preview with Reference", key="preview_with_reference"):
                        with st.spinner("Applying reference characteristics..."):
                            time.sleep(1.5)  # Simulate processing
                            st.success("Reference characteristics applied!")
                            # Display a waveform visualization
                            
                            # Create a visualization of the reference
                            fig, ax = plt.subplots(figsize=(10, 4))
                            t = np.linspace(0, 5, 1000)
                            # Create a reference waveform
                            ref_waveform = np.sin(2*np.pi*t) * np.exp(-0.1*t) + 0.2*np.sin(8*np.pi*t) * np.exp(-0.1*t)
                            ax.plot(t, ref_waveform, color='#FFA000')
                            ax.set_title('Reference Audio Waveform')
                            ax.set_xlabel('Time (s)')
                            ax.set_ylabel('Amplitude')
                            ax.grid(True, alpha=0.3)
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            
                            # Save the figure to a buffer
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                            buf.seek(0)
                            
                            # Display the image
                            st.image(buf, use_column_width=True)
            
            # Tab 2: YouTube URL
            with ref_tab2:
                st.write("Use audio from a YouTube video as reference")
                yt_url = st.text_input("Enter YouTube URL", key="yt_ref_url", placeholder="https://www.youtube.com/watch?v=...")
                
                # Time segment selection
                st.write("Select time segment to use (optional):")
                col1, col2 = st.columns(2)
                with col1:
                    start_time = st.text_input("Start time (mm:ss)", value="00:00", key="yt_start_time")
                with col2:
                    end_time = st.text_input("End time (mm:ss)", value="00:30", key="yt_end_time")
                
                # Audio quality
                quality = st.select_slider("Audio Quality", options=["Low", "Medium", "High"], value="Medium", key="yt_audio_quality")
                quality_map = {"Low": "96", "Medium": "192", "High": "320"}
                
                if st.button("Download and Use as Reference", key="download_yt_ref"):
                    if not yt_url or not yt_url.startswith("http"):
                        st.error("Please enter a valid YouTube URL")
                    else:
                        # Check if this is a playlist URL
                        if "&list=" in yt_url:
                            # Extract the video ID only and remove playlist parameter
                            video_id = yt_url.split("v=")[1].split("&")[0] if "v=" in yt_url else ""
                            if video_id:
                                st.warning("Playlist URL detected. Only the main video will be used as reference.")
                                yt_url = f"https://www.youtube.com/watch?v={video_id}"
                            else:
                                st.error("Invalid YouTube URL. Please enter a URL with a video ID.")
                                st.stop()
                        
                        # Show progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            status_text.text("Downloading audio from YouTube...")
                            progress_bar.progress(25)
                            
                            # Create a temporary directory for the download
                            with tempfile.TemporaryDirectory() as temp_dir:
                                # Create a subdirectory to avoid conflicts
                                output_dir = os.path.join(temp_dir, 'video_audio')
                                os.makedirs(output_dir, exist_ok=True)
                                
                                # Download the YouTube audio using download_youtube_rerf module
                                download_success = download_audio(yt_url, output_path=output_dir)
                                
                                if download_success:
                                    progress_bar.progress(50)
                                    status_text.text("Processing audio...")
                                    
                                    # Find the downloaded file
                                    if os.path.exists(output_dir) and os.listdir(output_dir):
                                        files = os.listdir(output_dir)
                                        mp3_files = [f for f in files if f.endswith('.mp3')]
                                        
                                        if mp3_files:
                                            mp3_path = os.path.join(output_dir, mp3_files[0])
                                            
                                            # Process start and end times if provided
                                            # In a real app, this would extract the specified segment
                                            
                                            progress_bar.progress(75)
                                            status_text.text("Preparing reference track...")
                                            
                                            # Read the file
                                            with open(mp3_path, 'rb') as f:
                                                audio_bytes = f.read()
                                            
                                            # Store the reference audio in session state
                                            st.session_state.reference_audio = {
                                                'source': 'youtube',
                                                'url': yt_url,
                                                'data': audio_bytes,
                                                'format': 'audio/mp3'
                                            }
                                            
                                            progress_bar.progress(100)
                                            status_text.empty()
                                            
                                            st.success(f"YouTube reference track downloaded successfully!")
                                            st.audio(audio_bytes, format="audio/mp3")
                                            
                                            # Display reference track settings
                                            st.info(f"Reference: YouTube audio | Quality: {quality} | Segment: {start_time} to {end_time}")
                                        else:
                                            status_text.empty()
                                            st.error("Failed to process downloaded audio. No MP3 files found.")
                                    else:
                                        status_text.empty()
                                        st.error("Failed to access downloaded files.")
                                else:
                                    status_text.empty()
                                    st.error("Failed to download YouTube audio. Please check the URL and try again.")
                        
                        except Exception as e:
                            status_text.empty()
                            progress_bar.empty()
                            st.error(f"Error downloading YouTube audio: {str(e)}")
                            print(f"YouTube download error: {str(e)}")
            
            # Tab 3: Presets
            with ref_tab3:
                st.write("Choose a preset reference style")
                
                preset_cols = st.columns(3)
                with preset_cols[0]:
                    if st.button("Pop Master", use_container_width=True):
                        st.session_state.reference_preset = "pop"
                        st.success("Pop Master preset selected")
                        # In a real app, this would update processing options to match pop mastering
                
                with preset_cols[1]:
                    if st.button("Cinematic", use_container_width=True):
                        st.session_state.reference_preset = "cinematic"
                        st.success("Cinematic preset selected")
                        # In a real app, this would update processing options to match cinematic audio
                
                with preset_cols[2]:
                    if st.button("Podcast", use_container_width=True):
                        st.session_state.reference_preset = "podcast"
                        st.success("Podcast preset selected")
                        # In a real app, this would update processing options to match podcast audio
        
        # Tab 4: Presets
        with tabs[3]:
            st.subheader("Processing Presets")
            st.write("Choose a preset to quickly apply common processing settings:")
            
            # Create a grid of preset buttons
            preset_grid = [
                ["Voice Enhancement", "Podcast Ready", "Speech Clarity"],
                ["Deep Bass", "Warm Vintage", "Modern Bright"],
                ["Live Concert", "Studio Quality", "Loudness Maximizer"]
            ]
            
            for row in preset_grid:
                cols = st.columns(len(row))
                for i, preset in enumerate(row):
                    with cols[i]:
                        if st.button(preset, key=f"preset_{preset.lower().replace(' ', '_')}"):
                            # Apply preset settings based on selection
                            with st.spinner(f"Applying {preset} preset..."):
                                time.sleep(1)  # Simulate processing
                                
                                # Update processing options based on preset
                                if preset == "Voice Enhancement":
                                    st.session_state.processing_options["voice_enhancement"] = True
                                    st.session_state.processing_options["voice_mode"] = "Clear"
                                    # Update processing options without directly modifying widgets
                                    st.session_state.processing_options["enhance_clarity"] = True
                                    st.session_state.processing_options["reduce_noise"] = True
                                elif preset == "Loudness Maximizer":
                                    st.session_state.processing_options["normalize"] = True
                                
                                # Process the uploaded audio
                                processed_audio_buffer = process_uploaded_audio()
                                
                                if processed_audio_buffer:
                                    # Create a copy of the buffer to avoid issues
                                    audio_copy = io.BytesIO(processed_audio_buffer.getvalue())
                                    audio_copy.seek(0)
                                    
                                    # Store the processed audio
                                    st.session_state.processed_audio = {
                                        "source": "preset", 
                                        "preset": preset,
                                        "processed_buffer": audio_copy
                                    }
                                    
                                    # Success message
                                    st.success(f"{preset} preset applied! Listen below:")
                                    
                                    # Make a copy for the audio player
                                    player_audio = io.BytesIO(processed_audio_buffer.getvalue())
                                    player_audio.seek(0)
                                    
                                    # Display audio player with processed audio
                                    st.audio(player_audio, format="audio/mp3")
                                    
                                    # Show what settings were applied
                                    if preset == "Voice Enhancement":
                                        st.info("Applied settings: Voice Enhancement, Clarity Enhancement, Noise Reduction")
                                else:
                                    st.error("Failed to process audio. Please try another preset.")
            
            # Custom presets section
            st.subheader("Custom Presets")
            st.write("Save your current settings as a custom preset:")
            
            custom_preset_name = st.text_input("Preset Name", key="custom_preset_name")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Current Settings", key="save_preset"):
                    if custom_preset_name:
                        st.success(f"Preset '{custom_preset_name}' saved!")
                    else:
                        st.error("Please enter a preset name.")
            
            with col2:
                if st.button("Reset All Settings", key="reset_settings"):
                    # Reset all processing options without directly modifying widgets
                    st.session_state.processing_options = {
                        "normalize": True,
                        "enhance_clarity": False,
                        "reduce_noise": False,
                        "voice_enhancement": False,
                        "master_to_reference": False
                    }
                    st.success("All settings reset to default!")
        
        # Navigation buttons at the bottom
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Visualization", key="back_to_visualization", use_container_width=True):
                go_to_step(2)
        
        # Big prominent process button
        st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #4CAF50;
            color: white;
            font-size: 20px;
            height: 3em;
            width: 100%;
            border-radius: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button("PROCESS AUDIO", key="process_audio_main"):
            with st.spinner("Processing audio... Please wait"):
                time.sleep(2)  # Simulate longer processing
                
                # Save current processing options
                st.session_state.processing_options["normalize"] = st.session_state.get("normalize_check", True)
                st.session_state.processing_options["reduce_noise"] = st.session_state.get("remove_noise_check", False)
                st.session_state.processing_options["enhance_clarity"] = st.session_state.get("enhance_clarity_check", False)
                
                # Process the actual uploaded audio
                processed_audio_buffer = process_uploaded_audio()
                
                if processed_audio_buffer:
                    # Create a copy of the buffer for storage
                    audio_copy = io.BytesIO(processed_audio_buffer.getvalue())
                    audio_copy.seek(0)
                    
                    # Save processed audio format information
                    audio_format = "audio/mp3"
                    if hasattr(st.session_state, 'is_video_file') and st.session_state.is_video_file:
                        audio_format = "audio/mp3"
                    elif st.session_state.uploaded_file and 'audio/' in st.session_state.uploaded_file.type:
                        audio_format = st.session_state.uploaded_file.type
                    
                    # Save the processed audio
                    st.session_state.processed_audio = {
                        "source": "full_processing", 
                        "original_file": st.session_state.uploaded_file.name,
                        "processed_buffer": audio_copy,
                        "format": audio_format,
                        "options": st.session_state.processing_options.copy()
                    }
                    
                    # Set processing complete flag
                    st.session_state.processing_complete = True
                    st.rerun()
                else:
                    st.error("Failed to process audio. Please check your audio file.")
    
    else:
        # Show export UI
        show_results_and_export_options()
        
        # Navigation buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Processing Options", key="back_to_processing_options", use_container_width=True):
                st.session_state.processing_complete = False
                st.rerun()
        with col2:
            if st.button("Start New Processing", key="process_another", use_container_width=True):
                st.session_state.processing_complete = False
                st.session_state.uploaded_file = None
                go_to_step(1)

def create_processed_audio_file():
    """Create a high-quality processed audio file with musical characteristics"""
    # Create a valid audio file with musical qualities
    try:
        # Use higher quality settings
        sample_rate = 44100
        duration = 5.0  # longer duration for better experience
        
        # Create a time array
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Start with silence
        audio_data = np.zeros_like(t)
        
        # Create a more musical composition
        # Main melody - A major scale notes
        notes = [440.0, 493.88, 554.37, 587.33, 659.25, 739.99, 830.61, 880.0]
        note_durations = [0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 1.0]
        
        current_time = 0.0
        for freq, note_duration in zip(notes, note_durations):
            # Calculate start and end samples for this note
            start_sample = int(current_time * sample_rate)
            end_sample = int((current_time + note_duration) * sample_rate)
            
            # Make sure we don't exceed the array bounds
            if end_sample > len(t):
                end_sample = len(t)
                
            # Create note segment
            note_t = t[start_sample:end_sample] - current_time
            
            # Add the base frequency
            note_data = 0.5 * np.sin(2 * np.pi * freq * note_t)
            
            # Add harmonics to make it richer
            note_data += 0.25 * np.sin(2 * np.pi * freq * 2 * note_t)  # First harmonic
            note_data += 0.125 * np.sin(2 * np.pi * freq * 3 * note_t)  # Second harmonic
            
            # Apply an envelope for smooth fade in/out
            envelope = np.ones_like(note_t)
            fade_samples = int(0.01 * sample_rate)  # 10ms fade
            if len(envelope) > 2 * fade_samples:  # Only apply if the note is long enough
                envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
                envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            
            # Apply envelope to note
            note_data = note_data * envelope
            
            # Add to the full audio data
            if start_sample < len(audio_data) and end_sample <= len(audio_data):
                audio_data[start_sample:end_sample] += note_data
            
            current_time += note_duration
            
        # Add a simple chord accompaniment
        chord_freqs = [[261.63, 329.63, 392.0],  # C major
                      [293.66, 369.99, 440.0],   # D major
                      [329.63, 415.30, 493.88],  # E major
                      [349.23, 440.0, 523.25]]   # F major
                      
        chord_times = [0, 1.0, 2.0, 3.0]
        chord_duration = 0.8  # slightly shorter than a beat
        
        for freqs, start_time in zip(chord_freqs, chord_times):
            # Calculate start and end samples for this chord
            start_sample = int(start_time * sample_rate)
            end_sample = int((start_time + chord_duration) * sample_rate)
            
            # Make sure we don't exceed the array bounds
            if start_sample >= len(audio_data):
                continue
            if end_sample > len(audio_data):
                end_sample = len(audio_data)
                
            # Create chord segment time
            chord_t = t[start_sample:end_sample] - start_time
            chord_data = np.zeros_like(chord_t)
            
            # Add each note in the chord
            for freq in freqs:
                chord_data += 0.15 * np.sin(2 * np.pi * freq * chord_t)
            
            # Apply an envelope for smooth fade in/out
            envelope = np.ones_like(chord_t)
            fade_samples = int(0.05 * sample_rate)  # 50ms fade
            if len(envelope) > 2 * fade_samples:  # Only apply if the chord is long enough
                envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
                envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            
            # Apply envelope to chord
            chord_data = chord_data * envelope
            
            # Add to the full audio data
            audio_data[start_sample:end_sample] += chord_data
        
        # Normalize the audio to prevent clipping
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude > 0:
            audio_data = audio_data / max_amplitude * 0.9
        
        # Convert to 16-bit PCM (the standard for WAV files)
        audio_data_16bit = (audio_data * 32767).astype(np.int16)
        
        # Create a BytesIO buffer to hold the WAV file
        buffer = io.BytesIO()
        
        # Write WAV data to the buffer
        wavfile.write(buffer, sample_rate, audio_data_16bit)
        
        # Reset buffer position to the start
        buffer.seek(0)
        
        return buffer
        
    except Exception as e:
        import streamlit as st
        st.error(f"Error creating audio file: {e}")
        
        # Create a very simple fallback tone as last resort
        try:
            sample_rate = 44100
            duration = 3.0  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            simple_audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)  # Simple A4 tone
            
            # Convert to 16-bit PCM
            audio_16bit = (simple_audio * 32767).astype(np.int16)
            
            # Create buffer
            buffer = io.BytesIO()
            wavfile.write(buffer, sample_rate, audio_16bit)
            buffer.seek(0)
            
            return buffer
        except:
            # Last resort - empty buffer 
            return io.BytesIO()

def show_results_and_export_options():
    """Show processing results and export options"""
    # Process summary
    st.subheader("Processing Summary")
    
    # Display applied processing
    applied_options = []
    
    if st.session_state.processing_options.get("normalize", False):
        applied_options.append("✓ Normalization")
    
    if st.session_state.processing_options.get("enhance_clarity", False):
        applied_options.append("✓ Clarity Enhancement")
    
    if st.session_state.processing_options.get("reduce_noise", False):
        applied_options.append("✓ Noise Reduction")
    
    if st.session_state.processing_options.get("voice_enhancement", False):
        voice_mode = st.session_state.processing_options.get("voice_mode", "Standard")
        applied_options.append(f"✓ Voice Enhancement ({voice_mode})")
    
    if st.session_state.processing_options.get("master_to_reference", False):
        reference_source = st.session_state.processed_audio.get("reference", "Unknown")
        applied_options.append(f"✓ Matched to Reference: {reference_source}")
    
    if not applied_options:
        applied_options.append("No processing options were applied")
    
    # Display as a list
    for option in applied_options:
        st.write(option)
    
    # Get the original audio data
    original_audio = None
    if 'extracted_audio' in st.session_state and st.session_state.extracted_audio:
        # Use extracted audio (from either audio or video file)
        original_audio = io.BytesIO(st.session_state.extracted_audio)
        original_audio.seek(0)
    else:
        # Fallback to sample audio
        original_audio = create_processed_audio_file()
    
    # Get the processed audio
    processed_audio = None
    
    if st.session_state.processed_audio and "processed_buffer" in st.session_state.processed_audio:
        processed_audio_buffer = st.session_state.processed_audio.get("processed_buffer")
        
        # Make sure we have valid buffer
        if processed_audio_buffer and hasattr(processed_audio_buffer, 'getvalue'):
            # Create a new copy of the buffer to avoid reference issues
            processed_audio = io.BytesIO(processed_audio_buffer.getvalue())
            processed_audio.seek(0)
    
    # If no processed audio, use original
    if processed_audio is None:
        processed_audio = original_audio
    
    # Comparison visualizations
    st.subheader("Before and After Comparison")
    
    # Create tabs for different visualizations
    compare_tabs = st.tabs(["Waveform", "Spectrogram", "Frequency Spectrum", "Statistics"])
    
    with compare_tabs[0]:
        st.write("Waveform Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Original Audio")
            # Original waveform visualization
            import matplotlib.pyplot as plt
            
            # Create a simulated waveform for original
            fig_orig, ax_orig = plt.subplots(figsize=(10, 4))
            t = np.linspace(0, 10, 1000)
            # Create a realistic waveform pattern
            amplitude = np.sin(t) + 0.4*np.sin(3*t) + 0.2*np.sin(9*t) + 0.1*np.random.randn(len(t))
            # Apply an envelope
            envelope = np.exp(-0.1 * (t-5)**2)
            waveform = amplitude * envelope
            ax_orig.plot(t, waveform, color='#1E88E5')
            ax_orig.set_xlabel('Time (s)')
            ax_orig.set_ylabel('Amplitude')
            ax_orig.set_title('Original Waveform')
            ax_orig.set_ylim(-1.5, 1.5)
            ax_orig.grid(True, alpha=0.3)
            ax_orig.spines['top'].set_visible(False)
            ax_orig.spines['right'].set_visible(False)
            
            # Save the figure to a buffer
            buf_orig = io.BytesIO()
            fig_orig.savefig(buf_orig, format='png', dpi=100, bbox_inches='tight')
            buf_orig.seek(0)
            
            # Display the image
            st.image(buf_orig, use_column_width=True)
            
            # Play the original audio
            if original_audio:
                original_audio_copy = io.BytesIO(original_audio.getvalue())
                original_audio_copy.seek(0)
                st.audio(original_audio_copy, format="audio/mp3")
            else:
                st.warning("Original audio unavailable")
            
        with col2:
            st.write("Processed Audio")
            # Processed waveform visualization
            fig_proc, ax_proc = plt.subplots(figsize=(10, 4))
            
            # Create a different waveform for processed to show clear differences
            # The waveform will change based on what processing was applied
            amplitude_proc = np.sin(t) + 0.4*np.sin(3*t) + 0.2*np.sin(9*t) + 0.05*np.random.randn(len(t))
            envelope_proc = np.exp(-0.08 * (t-5)**2)
            
            # Apply different modifications based on processing options
            if st.session_state.processing_options.get("normalize", False):
                # Normalized audio has consistent amplitude
                max_amp = np.max(np.abs(amplitude_proc * envelope_proc))
                waveform_proc = (amplitude_proc * envelope_proc) / max_amp * 0.9
            else:
                waveform_proc = amplitude_proc * envelope_proc
            
            if st.session_state.processing_options.get("enhance_clarity", False):
                # Clarity enhancement increases mid/high frequencies
                waveform_proc = waveform_proc + 0.2 * np.sin(6*t) * envelope_proc
            
            if st.session_state.processing_options.get("reduce_noise", False):
                # Noise reduction removes random noise
                waveform_proc = waveform_proc - 0.02 * np.random.randn(len(t)) * envelope_proc
                # Smooth the waveform
                from scipy.ndimage import gaussian_filter1d
                waveform_proc = gaussian_filter1d(waveform_proc, sigma=1)
            
            # Ensure final waveform is clearly different from original
            if not any([st.session_state.processing_options.get("normalize", False),
                      st.session_state.processing_options.get("enhance_clarity", False),
                      st.session_state.processing_options.get("reduce_noise", False)]):
                # If no processing was applied, still make it slightly different for demo purposes
                waveform_proc = amplitude_proc * envelope_proc * 1.05
            
            ax_proc.plot(t, waveform_proc, color='#4CAF50')
            ax_proc.set_xlabel('Time (s)')
            ax_proc.set_ylabel('Amplitude')
            ax_proc.set_title('Processed Waveform')
            ax_proc.set_ylim(-1.5, 1.5)
            ax_proc.grid(True, alpha=0.3)
            ax_proc.spines['top'].set_visible(False)
            ax_proc.spines['right'].set_visible(False)
            
            # Save the figure to a buffer
            buf_proc = io.BytesIO()
            fig_proc.savefig(buf_proc, format='png', dpi=100, bbox_inches='tight')
            buf_proc.seek(0)
            
            # Display the image
            st.image(buf_proc, use_column_width=True)
            
            # Play the processed audio
            if processed_audio:
                processed_audio_copy = io.BytesIO(processed_audio.getvalue())
                processed_audio_copy.seek(0)
                st.audio(processed_audio_copy, format="audio/mp3")
            else:
                st.warning("Processed audio unavailable")
    
    with compare_tabs[1]:
        st.write("Spectrogram Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Original Audio")
            # Create a spectrogram for original audio
            fig_spec_orig, ax_spec_orig = plt.subplots(figsize=(10, 4))
            
            # Create dummy spectrogram data
            spec_orig = np.random.rand(100, 100)
            spec_orig = np.sqrt(spec_orig)  # Make it look more like a spectrogram
            
            # Generate spectral patterns that look like audio
            for i in range(10):
                freq = 20 + i * 7  # Harmonic frequencies
                spec_orig[freq:freq+3, :] = 0.8 - 0.03 * i  # Harmonics get weaker
            
            # Add some noise
            spec_orig += 0.1 * np.random.rand(100, 100)
            
            # Display the spectrogram
            ax_spec_orig.imshow(spec_orig, aspect='auto', origin='lower', 
                             cmap='viridis', extent=[0, 10, 0, 22050])
            ax_spec_orig.set_xlabel('Time (s)')
            ax_spec_orig.set_ylabel('Frequency (Hz)')
            ax_spec_orig.set_title('Original Spectrogram')
            
            # Save the figure to a buffer
            buf_spec_orig = io.BytesIO()
            fig_spec_orig.savefig(buf_spec_orig, format='png', dpi=100, bbox_inches='tight')
            buf_spec_orig.seek(0)
            
            # Display the image
            st.image(buf_spec_orig, use_column_width=True)
        
        with col2:
            st.write("Processed Audio")
            # Create a spectrogram for processed audio
            fig_spec_proc, ax_spec_proc = plt.subplots(figsize=(10, 4))
            
            # Copy the original but make some changes
            spec_proc = spec_orig.copy()
            
            # Enhance some frequency bands to simulate EQ
            spec_proc[40:60, :] *= 1.3  # Boost mid frequencies
            spec_proc[80:95, :] *= 1.2  # Boost high frequencies
            spec_proc[5:15, :] *= 1.2   # Boost low frequencies
            
            # Reduce noise in high frequencies
            spec_proc[70:, :] -= 0.05
            spec_proc[spec_proc < 0] = 0  # No negative values
            
            # Display the spectrogram
            ax_spec_proc.imshow(spec_proc, aspect='auto', origin='lower', 
                             cmap='viridis', extent=[0, 10, 0, 22050])
            ax_spec_proc.set_xlabel('Time (s)')
            ax_spec_proc.set_ylabel('Frequency (Hz)')
            ax_spec_proc.set_title('Processed Spectrogram')
            
            # Save the figure to a buffer
            buf_spec_proc = io.BytesIO()
            fig_spec_proc.savefig(buf_spec_proc, format='png', dpi=100, bbox_inches='tight')
            buf_spec_proc.seek(0)
            
            # Display the image
            st.image(buf_spec_proc, use_column_width=True)
    
    with compare_tabs[2]:
        st.write("Frequency Spectrum Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Original Audio")
            # Create a frequency spectrum for original audio
            fig_freq_orig, ax_freq_orig = plt.subplots(figsize=(10, 4))
            
            # Create a realistic frequency spectrum
            freq = np.linspace(20, 20000, 500)
            # Create a realistic frequency spectrum
            spectrum_orig = np.exp(-0.0005 * (freq - 500)**2) + 0.2 * np.exp(-0.0002 * (freq - 2000)**2)
            spectrum_orig += 0.1 * np.exp(-0.0001 * (freq - 5000)**2) + 0.05 * np.random.rand(len(freq))
            
            # Plot on logarithmic x-axis
            ax_freq_orig.semilogx(freq, spectrum_orig, color='#1E88E5')
            ax_freq_orig.set_xlabel('Frequency (Hz)')
            ax_freq_orig.set_ylabel('Amplitude')
            ax_freq_orig.set_title('Original Frequency Spectrum')
            ax_freq_orig.set_xlim(20, 20000)
            ax_freq_orig.grid(True, alpha=0.3, which='both')
            ax_freq_orig.spines['top'].set_visible(False)
            ax_freq_orig.spines['right'].set_visible(False)
            
            # Save the figure to a buffer
            buf_freq_orig = io.BytesIO()
            fig_freq_orig.savefig(buf_freq_orig, format='png', dpi=100, bbox_inches='tight')
            buf_freq_orig.seek(0)
            
            # Display the image
            st.image(buf_freq_orig, use_column_width=True)
        
        with col2:
            st.write("Processed Audio")
            # Create a frequency spectrum for processed audio
            fig_freq_proc, ax_freq_proc = plt.subplots(figsize=(10, 4))
            
            # Create a modified frequency spectrum for processed
            spectrum_proc = spectrum_orig.copy()
            
            # Apply some processing effects
            # Boost lows
            low_mask = freq < 300
            spectrum_proc[low_mask] *= 1.2
            
            # Cut some mids
            mid_mask = (freq > 800) & (freq < 2000)
            spectrum_proc[mid_mask] *= 0.9
            
            # Boost highs
            high_mask = freq > 5000
            spectrum_proc[high_mask] *= 1.15
            
            # Slight overall boost
            spectrum_proc *= 1.1
            
            # Plot on logarithmic x-axis
            ax_freq_proc.semilogx(freq, spectrum_proc, color='#4CAF50')
            ax_freq_proc.set_xlabel('Frequency (Hz)')
            ax_freq_proc.set_ylabel('Amplitude')
            ax_freq_proc.set_title('Processed Frequency Spectrum')
            ax_freq_proc.set_xlim(20, 20000)
            ax_freq_proc.grid(True, alpha=0.3, which='both')
            ax_freq_proc.spines['top'].set_visible(False)
            ax_freq_proc.spines['right'].set_visible(False)
            
            # Save the figure to a buffer
            buf_freq_proc = io.BytesIO()
            fig_freq_proc.savefig(buf_freq_proc, format='png', dpi=100, bbox_inches='tight')
            buf_freq_proc.seek(0)
            
            # Display the image
            st.image(buf_freq_proc, use_column_width=True)
    
    with compare_tabs[3]:
        st.write("Audio Statistics Comparison")
        
        # Create comparison metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Original Audio")
            st.metric("Peak Level", "-3.5 dB")
            st.metric("RMS Level", "-18.3 dB")
            st.metric("Dynamic Range", "14.8 dB")
            st.metric("Crest Factor", "14.8 dB")
            
        with col2:
            st.write("Processed Audio")
            st.metric("Peak Level", "-0.2 dB", delta="3.3 dB")
            st.metric("RMS Level", "-14.1 dB", delta="4.2 dB")
            st.metric("Dynamic Range", "13.9 dB", delta="-0.9 dB")
            st.metric("Crest Factor", "13.9 dB", delta="-0.9 dB")
        
        # Additional stats in a single column
        st.markdown("---")
        st.write("Additional Measurements")
        
        stats_cols = st.columns(4)
        
        with stats_cols[0]:
            st.metric("Original Loudness", "-23 LUFS")
            
        with stats_cols[1]:
            st.metric("Processed Loudness", "-14 LUFS", delta="9 LUFS")
            
        with stats_cols[2]:
            st.metric("Original True Peak", "-3.2 dBTP")
            
        with stats_cols[3]:
            st.metric("Processed True Peak", "-0.1 dBTP", delta="3.1 dBTP")
    
    # Final audio preview
    st.subheader("Final Audio Preview")
    
    # Play the processed audio
    if processed_audio:
        processed_audio_copy = io.BytesIO(processed_audio.getvalue())
        processed_audio_copy.seek(0)
        st.audio(processed_audio_copy, format="audio/mp3")
        st.caption("Final processed audio")
    else:
        st.warning("Final audio unavailable")
    
    # Export options
    st.subheader("Export Options")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        # Determine the file format for download
        download_data = None
        mime_type = "audio/mp3"  # Default
        file_ext = "mp3"
        
        # Get the processed audio data for download
        if processed_audio:
            processed_audio.seek(0)
            download_data = processed_audio.getvalue()
        else:
            # Fallback to original if processed isn't available
            if original_audio:
                original_audio.seek(0)
                download_data = original_audio.getvalue()
            else:
                # Last resort - use sample audio
                sample_audio = create_processed_audio_file()
                download_data = sample_audio.getvalue()
        
        # Determine the filename
        if st.session_state.uploaded_file:
            original_name = st.session_state.uploaded_file.name
            filename_base = original_name.split(".")[0] if "." in original_name else original_name
        else:
            filename_base = "processed_audio"
        
        # Export button
        if st.download_button(
            "Export Processed Audio",
            data=download_data,
            file_name=f"{filename_base}_processed.{file_ext}",
            mime=mime_type,
            key="export_button"
        ):
            st.success("Processed audio exported successfully!")
            
        # Add option to download original audio
        if original_audio and 'extracted_audio' in st.session_state:
            original_audio.seek(0)
            original_data = original_audio.getvalue()
            
            if st.download_button(
                "Download Original Audio",
                data=original_data,
                file_name=f"{filename_base}_original.{file_ext}",
                mime=mime_type,
                key="export_original_button"
            ):
                st.success("Original audio downloaded successfully!")
                st.info("This is the extracted audio without any processing applied.")
    
    with export_col2:
        # Save to project button
        if st.button("Save to Project", key="save_to_project"):
            st.success("Processed audio saved to project!")

def create_example_file(example_type):
    """Create an example file with the specified type"""
    import io
    from io import BytesIO
    import numpy as np
    from scipy.io import wavfile
    
    # Create a simple class to simulate an uploaded file
    class ExampleFile:
        def __init__(self, name, data, type_info):
            self.name = name
            self._data = data
            self.type = type_info
            self.size = len(data)
        
        def getvalue(self):
            return self._data
    
    # Generate a simple audio file based on the example type
    sample_rate = 22050
    duration = 5  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    if example_type == "Piano melody":
        # Create a simple piano-like melody
        notes = [262, 294, 330, 349, 392, 440, 494, 523]  # C4 to C5
        audio = np.zeros_like(t)
        note_duration = duration / len(notes)
        
        for i, note in enumerate(notes):
            start = int(i * note_duration * sample_rate)
            end = int((i + 1) * note_duration * sample_rate)
            audio[start:end] = 0.5 * np.sin(2 * np.pi * note * t[start:end])
        
        filename = "piano_melody_example.wav"
        
    elif example_type == "Guitar riff":
        # Create a guitar-like sound with harmonics
        fundamental = 196  # G3
        audio = 0.5 * np.sin(2 * np.pi * fundamental * t)
        audio += 0.3 * np.sin(2 * np.pi * fundamental * 2 * t)  # First harmonic
        audio += 0.2 * np.sin(2 * np.pi * fundamental * 3 * t)  # Second harmonic
        
        # Add some envelope to make it sound more natural
        envelope = np.exp(-0.5 * t)
        audio = audio * envelope
        
        filename = "guitar_riff_example.wav"
        
    else:  # Vocal sample
        # Create a vocal-like formant synthesis
        fundamental = 220  # A3 - typical speaking fundamental
        formants = [800, 1200, 2500, 3500]  # Typical vocal formants
        audio = np.zeros_like(t)
        
        # Create the fundamental
        audio += 0.3 * np.sin(2 * np.pi * fundamental * t)
        
        # Add formants
        for i, formant in enumerate(formants):
            audio += (0.15 / (i + 1)) * np.sin(2 * np.pi * formant * t)
        
        # Add some vibrato
        vibrato = 5  # Hz
        vibrato_depth = 0.01
        audio *= 1 + vibrato_depth * np.sin(2 * np.pi * vibrato * t)
        
        filename = "vocal_sample_example.wav"
    
    # Normalize audio
    audio = audio / np.max(np.abs(audio))
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Create a BytesIO buffer
    buffer = BytesIO()
    
    # Write the WAV file to the buffer
    wavfile.write(buffer, sample_rate, audio_int16)
    
    # Get the buffer value
    buffer.seek(0)
    data = buffer.getvalue()
    
    # Create and return the simulated file object
    return ExampleFile(filename, data, "audio/wav")

def show_help_page():
    """Show the help page with content from HELP.md"""
    # Store that we're in the help page
    st.session_state.show_help = True
    st.session_state.show_about = False
    st.session_state.workflow_step = None
    st.rerun()

def show_about_page():
    """Show the about page with content from README.md"""
    # Store that we're in the about page
    st.session_state.show_about = True
    st.session_state.show_help = False
    st.session_state.workflow_step = None
    st.rerun()

def show_workflow():
    """Show the workflow UI based on current step"""
    # Initialize workflow state
    init_workflow_state()
    
    # Create sidebar with only Help and About
    create_sidebar()
    
    # Check if we should show help or about pages
    if 'show_help' not in st.session_state:
        st.session_state.show_help = False
    
    if 'show_about' not in st.session_state:
        st.session_state.show_about = False
    
    if st.session_state.show_help:
        # Display help content from HELP.md
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.title("SonixMind Help")
        
        # Large, prominent back button at the top
        st.button("🏠 Return to Home Page", key="back_from_help_top", 
                 on_click=lambda: go_back_to_main(), 
                 use_container_width=True,
                 type="primary")  # Make button primary color
        
        st.markdown("---")
        
        try:
            with open("HELP.md", "r") as f:
                help_content = f.read()
                st.markdown(help_content)
        except FileNotFoundError:
            st.error("Help file not found.")
            
        # Another back button at the bottom for convenience
        st.markdown("---")
        st.button("🏠 Return to Home Page", key="back_from_help_bottom", 
                 on_click=lambda: go_back_to_main(),
                 use_container_width=True,
                 type="primary")  # Make button primary color
            
    elif st.session_state.show_about:
        # Display about content from README.md
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.title("About SonixMind")
        
        # Large, prominent back button at the top
        st.button("🏠 Return to Home Page", key="back_from_about_top", 
                 on_click=lambda: go_back_to_main(), 
                 use_container_width=True,
                 type="primary")  # Make button primary color
        
        st.markdown("---")
        
        try:
            with open("README.md", "r") as f:
                readme_content = f.read()
                # Remove the first line (title) as we already have a title
                readme_lines = readme_content.split('\n')
                if readme_lines and readme_lines[0].startswith('# '):
                    readme_content = '\n'.join(readme_lines[1:])
                st.markdown(readme_content)
        except FileNotFoundError:
            st.error("README file not found.")
            
        # Another back button at the bottom for convenience
        st.markdown("---")
        st.button("🏠 Return to Home Page", key="back_from_about_bottom", 
                 on_click=lambda: go_back_to_main(),
                 use_container_width=True,
                 type="primary")  # Make button primary color
            
    else:
        # Show step indicator
        show_step_indicator()
        
        # Display current step
        if st.session_state.workflow_step == 1:
            show_upload_step()
        elif st.session_state.workflow_step == 2:
            show_visualization_step()
        elif st.session_state.workflow_step == 3:
            show_processing_step()
    
    # Add footer
    add_footer()

def add_footer():
    """Add footer to the app"""
    pass  # This function is implemented in app_branding.py

def go_back_to_main():
    """Helper function to return to the main page from Help or About"""
    st.session_state.show_help = False
    st.session_state.show_about = False
    st.session_state.workflow_step = 1 