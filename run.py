#!/usr/bin/env python

"""
SonixMind Application Runner Script
-----------------------------------
This script provides a convenient way to start the SonixMind application.

Usage:
    python run.py [--port PORT]

Options:
    --port PORT    Specify the port to run on (default: 8501)
"""

import os
import sys
import argparse
import subprocess

def main():
    """Run the SonixMind application with Streamlit."""
    parser = argparse.ArgumentParser(description="Run the SonixMind application")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the application on")
    
    args = parser.parse_args()
    
    # Check for FFmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: FFmpeg not found. SonixMind requires FFmpeg for audio/video processing.")
        print("Please install FFmpeg before running the application.")
        print("Installation instructions: https://github.com/indapoint/sonixmind#installation")
        sys.exit(1)
    
    # Run the Streamlit app
    print(f"Starting SonixMind on port {args.port}...")
    os.environ["STREAMLIT_SERVER_PORT"] = str(args.port)
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)

if __name__ == "__main__":
    main() 