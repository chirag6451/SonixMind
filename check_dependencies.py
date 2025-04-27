#!/usr/bin/env python

"""
SonixMind Dependency Checker
----------------------------
This script checks if all required dependencies for SonixMind are installed.
"""

import sys
import subprocess
import importlib.util
import pkg_resources
from pkg_resources import VersionConflict, DistributionNotFound

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        print(f"❌ Python {required_version[0]}.{required_version[1]} or higher is required.")
        print(f"   Current version: {current_version[0]}.{current_version[1]}")
        return False
    
    print(f"✅ Python {current_version[0]}.{current_version[1]} detected.")
    return True

def check_ffmpeg():
    """Check if FFmpeg is installed."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        version_line = result.stdout.split('\n')[0]
        print(f"✅ FFmpeg detected: {version_line}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg not found. Please install FFmpeg.")
        print("   Installation instructions: https://ffmpeg.org/download.html")
        return False

def check_python_packages():
    """Check if required Python packages are installed."""
    requirements_file = "requirements.txt"
    try:
        with open(requirements_file, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        missing = []
        for requirement in requirements:
            try:
                pkg_resources.require(requirement)
            except (VersionConflict, DistributionNotFound):
                missing.append(requirement)
        
        if missing:
            print("❌ Missing required Python packages:")
            for package in missing:
                print(f"   - {package}")
            print("\nInstall missing packages using:")
            print(f"   pip install -r {requirements_file}")
            return False
        else:
            print("✅ All required Python packages are installed.")
            return True
    except FileNotFoundError:
        print(f"❌ Requirements file '{requirements_file}' not found.")
        return False

def main():
    """Run all dependency checks."""
    print("SonixMind Dependency Checker")
    print("============================")
    
    python_ok = check_python_version()
    ffmpeg_ok = check_ffmpeg()
    packages_ok = check_python_packages()
    
    if python_ok and ffmpeg_ok and packages_ok:
        print("\n✅ All dependencies are satisfied! You're ready to run SonixMind.")
        return 0
    else:
        print("\n❌ Some dependencies are missing. Please install them before running SonixMind.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 