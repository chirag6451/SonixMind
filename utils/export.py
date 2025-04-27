"""
Export utilities for audio files and processing results.
"""
import os
import json
import numpy as np
import soundfile as sf
import tempfile
import subprocess
import base64
from typing import Dict, Any, List, Tuple, Optional, Union
import datetime
import zipfile
import io

def export_audio(audio: np.ndarray, sr: int, output_path: str, 
                format: str = 'wav', subtype: str = 'PCM_24') -> str:
    """
    Export audio data to a file.
    
    Args:
        audio: Audio data as numpy array
        sr: Sample rate
        output_path: Path to save the audio file
        format: Audio format (wav, flac, ogg)
        subtype: Audio subtype/bit depth
        
    Returns:
        Path to the saved audio file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save audio file
    sf.write(output_path, audio, sr, format=format, subtype=subtype)
    
    return output_path

def export_mp3(audio: np.ndarray, sr: int, output_path: str, 
              bitrate: str = '320k') -> str:
    """
    Export audio data to MP3 format using ffmpeg.
    
    Args:
        audio: Audio data as numpy array
        sr: Sample rate
        output_path: Path to save the MP3 file
        bitrate: MP3 bitrate
        
    Returns:
        Path to the saved MP3 file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Create temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        temp_wav_path = temp_wav.name
        sf.write(temp_wav_path, audio, sr, format='wav', subtype='PCM_24')
    
    try:
        # Convert to MP3 using ffmpeg
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', temp_wav_path,
            '-b:a', bitrate,
            '-f', 'mp3',
            output_path
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return output_path
    finally:
        # Clean up temporary file
        if os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)

def export_processing_settings(settings: Dict[str, Any], output_path: str) -> str:
    """
    Export processing settings to a JSON file.
    
    Args:
        settings: Dictionary of processing settings
        output_path: Path to save the JSON file
        
    Returns:
        Path to the saved JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_settings = {}
    for key, value in settings.items():
        if isinstance(value, np.ndarray):
            serializable_settings[key] = value.tolist()
        elif isinstance(value, dict):
            # Handle nested dictionaries
            serializable_settings[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    serializable_settings[key][k] = v.tolist()
                else:
                    serializable_settings[key][k] = v
        else:
            serializable_settings[key] = value
    
    # Add timestamp
    serializable_settings['timestamp'] = datetime.datetime.now().isoformat()
    
    # Save JSON file
    with open(output_path, 'w') as f:
        json.dump(serializable_settings, f, indent=2)
    
    return output_path

def create_processing_report(original_audio: np.ndarray, processed_audio: np.ndarray, 
                             sr: int, settings: Dict[str, Any], 
                             output_dir: str) -> str:
    """
    Create a complete processing report with audio files and settings.
    
    Args:
        original_audio: Original audio data
        processed_audio: Processed audio data
        sr: Sample rate
        settings: Dictionary of processing settings
        output_dir: Directory to save the report
        
    Returns:
        Path to the ZIP file containing the report
    """
    # Create temporary directory for report files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Export audio files
        original_path = os.path.join(temp_dir, 'original.wav')
        processed_path = os.path.join(temp_dir, 'processed.wav')
        processed_mp3_path = os.path.join(temp_dir, 'processed.mp3')
        settings_path = os.path.join(temp_dir, 'settings.json')
        
        export_audio(original_audio, sr, original_path)
        export_audio(processed_audio, sr, processed_path)
        export_mp3(processed_audio, sr, processed_mp3_path)
        export_processing_settings(settings, settings_path)
        
        # Create HTML report
        html_path = os.path.join(temp_dir, 'report.html')
        create_html_report(original_audio, processed_audio, sr, settings, html_path)
        
        # Create ZIP file
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_path = os.path.join(output_dir, f'audio_processing_report_{timestamp}.zip')
        
        os.makedirs(os.path.dirname(os.path.abspath(zip_path)), exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
    
    return zip_path

def create_html_report(original_audio: np.ndarray, processed_audio: np.ndarray, 
                       sr: int, settings: Dict[str, Any], output_path: str) -> str:
    """
    Create an HTML report with audio visualization and settings.
    
    Args:
        original_audio: Original audio data
        processed_audio: Processed audio data
        sr: Sample rate
        settings: Dictionary of processing settings
        output_path: Path to save the HTML file
        
    Returns:
        Path to the saved HTML file
    """
    # Import visualization utilities
    from utils.visualization import (
        create_waveform, create_spectrogram, 
        create_comparison_plot, fig_to_base64
    )
    
    # Create visualizations
    waveform_original = fig_to_base64(create_waveform(original_audio, sr, "Original Waveform"))
    waveform_processed = fig_to_base64(create_waveform(processed_audio, sr, "Processed Waveform"))
    spectrogram_original = fig_to_base64(create_spectrogram(original_audio, sr, "Original Spectrogram"))
    spectrogram_processed = fig_to_base64(create_spectrogram(processed_audio, sr, "Processed Spectrogram"))
    comparison = fig_to_base64(create_comparison_plot(original_audio, processed_audio, sr))
    
    # Create audio elements
    def audio_to_base64(audio, sr):
        with io.BytesIO() as buf:
            sf.write(buf, audio, sr, format='wav', subtype='PCM_16')
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
    
    original_audio_b64 = audio_to_base64(original_audio, sr)
    processed_audio_b64 = audio_to_base64(processed_audio, sr)
    
    # Format settings for display
    settings_html = '<table class="settings-table">'
    settings_html += '<tr><th>Setting</th><th>Value</th></tr>'
    
    for key, value in settings.items():
        if isinstance(value, dict):
            # Handle nested dictionaries
            settings_html += f'<tr><td colspan="2" class="section-header">{key}</td></tr>'
            for k, v in value.items():
                if isinstance(v, np.ndarray) and v.size > 10:
                    # Summarize large arrays
                    settings_html += f'<tr><td>{k}</td><td>Array[{v.size}]</td></tr>'
                else:
                    settings_html += f'<tr><td>{k}</td><td>{v}</td></tr>'
        elif isinstance(value, np.ndarray) and value.size > 10:
            # Summarize large arrays
            settings_html += f'<tr><td>{key}</td><td>Array[{value.size}]</td></tr>'
        else:
            settings_html += f'<tr><td>{key}</td><td>{value}</td></tr>'
    
    settings_html += '</table>'
    
    # Create HTML document
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Audio Processing Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2E77B5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }}
            .image-container {{
                margin: 20px 0;
                text-align: center;
            }}
            .image-container img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .audio-container {{
                margin: 20px 0;
            }}
            .audio-container audio {{
                width: 100%;
            }}
            .settings-table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .settings-table th, .settings-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            .settings-table th {{
                background-color: #f2f2f2;
            }}
            .section-header {{
                background-color: #e9e9e9;
                font-weight: bold;
            }}
            .timestamp {{
                color: #777;
                font-style: italic;
                text-align: right;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Audio Processing Report</h1>
            <p class="timestamp">Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Audio Comparison</h2>
                <div class="audio-container">
                    <h3>Original Audio</h3>
                    <audio controls>
                        <source src="data:audio/wav;base64,{original_audio_b64}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                </div>
                <div class="audio-container">
                    <h3>Processed Audio</h3>
                    <audio controls>
                        <source src="data:audio/wav;base64,{processed_audio_b64}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                </div>
                <div class="image-container">
                    <h3>Comparison Overview</h3>
                    <img src="data:image/png;base64,{comparison}" alt="Audio Comparison">
                </div>
            </div>
            
            <div class="section">
                <h2>Waveforms</h2>
                <div class="image-container">
                    <img src="data:image/png;base64,{waveform_original}" alt="Original Waveform">
                </div>
                <div class="image-container">
                    <img src="data:image/png;base64,{waveform_processed}" alt="Processed Waveform">
                </div>
            </div>
            
            <div class="section">
                <h2>Spectrograms</h2>
                <div class="image-container">
                    <img src="data:image/png;base64,{spectrogram_original}" alt="Original Spectrogram">
                </div>
                <div class="image-container">
                    <img src="data:image/png;base64,{spectrogram_processed}" alt="Processed Spectrogram">
                </div>
            </div>
            
            <div class="section">
                <h2>Processing Settings</h2>
                {settings_html}
            </div>
        </div>
    </body>
    </html>
    '''
    
    # Save HTML file
    with open(output_path, 'w') as f:
        f.write(html)
    
    return output_path

def get_download_link(file_path: str, link_text: str = "Download") -> str:
    """
    Create a download link for a file using base64 encoding.
    For use in Streamlit apps.
    
    Args:
        file_path: Path to the file
        link_text: Text for the download link
        
    Returns:
        HTML string with download link
    """
    with open(file_path, 'rb') as f:
        data = f.read()
    
    b64 = base64.b64encode(data).decode('utf-8')
    file_name = os.path.basename(file_path)
    file_extension = os.path.splitext(file_name)[1]
    
    mime_types = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg',
        '.zip': 'application/zip',
        '.json': 'application/json',
        '.html': 'text/html',
    }
    
    mime_type = mime_types.get(file_extension.lower(), 'application/octet-stream')
    
    href = f'<a href="data:{mime_type};base64,{b64}" download="{file_name}">{link_text}</a>'
    return href 