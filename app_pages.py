import streamlit as st
from app_branding import show_about_section, show_help_section, add_footer, get_app_info

def load_pages():
    """Define and load app pages"""
    
    # Define page dictionary
    pages = {
        "Home": show_home_page,
        "Audio Extraction": show_audio_extraction_page,
        "Audio Processing": show_audio_processing_page,
        "Audio Mastering": show_audio_mastering_page,
        "YouTube Reference": show_youtube_reference_page,
        "Visualization": show_visualization_page,
        "Help": show_help_section,
        "About": show_about_section
    }
    
    # Add selector to sidebar
    app_info = get_app_info()
    
    st.sidebar.title(f"{app_info['name']}")
    st.sidebar.markdown("---")
    
    # Initialize session state for navigation
    if 'navigation' not in st.session_state:
        st.session_state.navigation = "Home"
    
    # Page selection
    selected_page = st.sidebar.radio("Navigate", list(pages.keys()), index=list(pages.keys()).index(st.session_state.navigation))
    
    # Update session state when sidebar selection changes
    if selected_page != st.session_state.navigation:
        st.session_state.navigation = selected_page
    
    # Display the selected page
    pages[st.session_state.navigation]()
    
    # Add footer
    add_footer()

def show_home_page():
    """Display the home page"""
    st.title("Welcome to SonixMind")
    
    st.info("""
    **Modular Interface Mode**
    
    You're currently using the modular interface where you can access any feature directly.
    For a guided step-by-step experience, switch to the "Step-by-Step Workflow" mode in the sidebar.
    """)
    
    st.markdown("""
    SonixMind is your complete solution for professional audio processing. 
    
    ## What can you do with SonixMind?
    
    - Extract high-quality audio from video files
    - Process and enhance audio with professional tools
    - Master your audio to match reference tracks
    - Download reference tracks from YouTube
    - Visualize audio with detailed waveforms and spectrograms
    """)
    
    # Display feature cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Popular Features")
        st.markdown("""
        - **Voice Enhancement**: Improve speech clarity
        - **Audio Normalization**: Achieve consistent levels
        - **Noise Reduction**: Clean up recordings
        - **Audio Extraction**: Pull audio from videos
        """)
        
        if st.button("Get Started", key="get_started_btn"):
            st.session_state.navigation = "Audio Extraction"
            st.rerun()
    
    with col2:
        st.subheader("Recent Updates")
        st.markdown("""
        - Added AI-powered voice restoration
        - Improved mastering algorithm
        - Enhanced visualization tools
        - Added batch processing capability
        - YouTube reference track integration
        """)
        
        if st.button("Learn More", key="learn_more_btn"):
            st.session_state.navigation = "Help"
            st.rerun()
    
    # Quick access section
    st.markdown("---")
    st.subheader("Quick Access")
    
    qa_col1, qa_col2, qa_col3, qa_col4 = st.columns(4)
    
    with qa_col1:
        if st.button("Extract Audio", key="extract_audio_btn"):
            st.session_state.navigation = "Audio Extraction"
            st.rerun()
    
    with qa_col2:
        if st.button("Process Audio", key="process_audio_btn"):
            st.session_state.navigation = "Audio Processing"
            st.rerun()
    
    with qa_col3:
        if st.button("Master Audio", key="master_audio_btn"):
            st.session_state.navigation = "Audio Mastering"
            st.rerun()
    
    with qa_col4:
        if st.button("YouTube Reference", key="youtube_ref_btn"):
            st.session_state.navigation = "YouTube Reference"
            st.rerun()

def show_audio_extraction_page():
    """Display the audio extraction page - placeholder for integration with actual functionality"""
    st.title("Audio Extraction")
    
    st.markdown("""
    Extract high-quality audio from video files.
    
    > Note: This is a placeholder page that will be integrated with the actual audio extraction functionality from app.py.
    """)
    
    # Placeholder for file uploader
    st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    
    # Placeholder for extraction options
    st.subheader("Extraction Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox("Output Format", ["WAV", "MP3"])
        st.selectbox("Sample Rate", ["44.1 kHz", "48 kHz", "96 kHz"])
    
    with col2:
        st.selectbox("Channels", ["Stereo", "Mono"])
        st.checkbox("Normalize Audio")
    
    # Placeholder for extraction button
    st.button("Extract Audio")

def show_audio_processing_page():
    """Display the audio processing page - placeholder for integration with actual functionality"""
    st.title("Audio Processing")
    
    st.markdown("""
    Process and enhance your audio with professional tools.
    
    > Note: This is a placeholder page that will be integrated with the actual audio processing functionality from app.py.
    """)
    
    # Placeholder tabs for different processing options
    tabs = st.tabs(["Basic Processing", "Noise Reduction", "Voice Enhancement"])
    
    with tabs[0]:
        st.subheader("Basic Processing")
        st.slider("Volume", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        st.slider("Bass Boost", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        st.slider("Treble", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        st.slider("Clarity", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    
    with tabs[1]:
        st.subheader("Noise Reduction")
        st.selectbox("Noise Reduction Level", ["Low", "Medium", "High"])
        st.checkbox("Show Advanced Options")
    
    with tabs[2]:
        st.subheader("Voice Enhancement")
        st.selectbox("Enhancement Type", ["Clarity Improvement", "Voice Isolation", "VoiceFixer (AI)"])
        st.button("Preview")
    
    # Placeholder for processing button
    st.button("Apply Processing")

def show_audio_mastering_page():
    """Display the audio mastering page - placeholder for integration with actual functionality"""
    st.title("Audio Mastering")
    
    st.markdown("""
    Master your audio to match reference tracks.
    
    > Note: This is a placeholder page that will be integrated with the actual audio mastering functionality from app.py.
    """)
    
    # Placeholder tabs for different mastering options
    tabs = st.tabs(["Reference Mastering", "Custom Mastering"])
    
    with tabs[0]:
        st.subheader("Reference Mastering")
        st.file_uploader("Upload Target Audio", type=["wav", "mp3"])
        st.file_uploader("Upload Reference Track", type=["wav", "mp3"])
        st.button("Analyze Both Tracks")
        
        st.checkbox("Match Loudness")
        st.checkbox("Match Frequency Balance")
        st.checkbox("Match Dynamics")
    
    with tabs[1]:
        st.subheader("Custom Mastering")
        st.slider("Low End", min_value=-12.0, max_value=12.0, value=0.0, step=0.5)
        st.slider("Mid Range", min_value=-12.0, max_value=12.0, value=0.0, step=0.5)
        st.slider("High End", min_value=-12.0, max_value=12.0, value=0.0, step=0.5)
        st.slider("Compression", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        st.slider("Limiting", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    
    # Placeholder for mastering button
    st.button("Apply Mastering")

def show_youtube_reference_page():
    """Display the YouTube reference page - placeholder for integration with actual functionality"""
    st.title("YouTube Reference")
    
    st.markdown("""
    Download reference tracks from YouTube.
    
    > Note: This is a placeholder page that will be integrated with the actual YouTube reference functionality from app.py.
    """)
    
    # Placeholder tabs
    tabs = st.tabs(["Search", "Reference Library"])
    
    with tabs[0]:
        st.text_input("Enter YouTube URL or Search Term")
        st.button("Search")
        
        # Placeholder for search results
        st.subheader("Search Results")
        st.info("Search results will appear here.")
    
    with tabs[1]:
        st.subheader("Your Reference Library")
        st.info("Your downloaded reference tracks will appear here.")
    
    # Placeholder for download button
    st.button("Download Selected")

def show_visualization_page():
    """Display the visualization page - placeholder for integration with actual functionality"""
    st.title("Audio Visualization")
    
    st.markdown("""
    Visualize your audio with detailed waveforms and spectrograms.
    
    > Note: This is a placeholder page that will be integrated with the actual visualization functionality from app.py.
    """)
    
    # Placeholder tabs
    tabs = st.tabs(["Waveform", "Spectrogram", "Statistics"])
    
    with tabs[0]:
        st.subheader("Waveform View")
        st.info("Waveform visualization will appear here.")
        
        # Placeholder for options
        st.checkbox("Show Peaks")
        st.checkbox("Show RMS")
    
    with tabs[1]:
        st.subheader("Spectrogram View")
        st.info("Spectrogram visualization will appear here.")
        
        # Placeholder for options
        st.selectbox("Color Map", ["Viridis", "Magma", "Inferno", "Plasma"])
        st.selectbox("Scale", ["Linear", "Logarithmic"])
    
    with tabs[2]:
        st.subheader("Audio Statistics")
        st.info("Audio statistics will appear here.")
    
    # Placeholder for file upload
    st.file_uploader("Upload Audio for Visualization", type=["wav", "mp3"])
    
    # Placeholder for visualization button
    st.button("Generate Visualization") 