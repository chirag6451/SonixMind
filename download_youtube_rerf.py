"""
YouTube Audio Download Module for SonixMind

This module provides functionality to download audio from YouTube videos
using the yt-dlp library (a fork of youtube-dl with additional features).
"""

import os
import yt_dlp
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('youtube_downloader')

def download_audio(url, output_path='./'):
    """
    Download audio from a YouTube video URL.
    
    Args:
        url (str): The YouTube video URL
        output_path (str): Directory where the audio will be saved
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Clean the URL to ensure we only get a single video
    if "&list=" in url:
        # Extract just the video ID and remove playlist parameters
        video_id = url.split("v=")[1].split("&")[0] if "v=" in url else ""
        if video_id:
            url = f"https://www.youtube.com/watch?v={video_id}"
            logger.info(f"Playlist URL detected. Using only video ID: {video_id}")
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'quiet': False,       # Set to True to reduce output
        'no_warnings': False, # Set to True to reduce output
        'ignoreerrors': True,
        'noplaylist': True,   # Only download the video, not playlist
    }
    
    try:
        logger.info(f"Downloading audio from {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract information first to get title
            info = ydl.extract_info(url, download=False)
            if info is None:
                logger.error("Failed to extract video information")
                return False
                
            # Download the video
            ydl.download([url])
            
        logger.info(f"Successfully downloaded audio from {url}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading from YouTube: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage when script is run directly
    video_url = input("Enter YouTube URL: ")
    success = download_audio(video_url)
    if success:
        print("Download completed successfully!")
    else:
        print("Download failed. Please check the URL and try again.")