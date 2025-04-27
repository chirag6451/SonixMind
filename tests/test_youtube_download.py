#!/usr/bin/env python

"""
Test YouTube download functionality
"""

import os
import tempfile
import unittest
from download_youtube_rerf import download_audio

class TestYoutubeDownload(unittest.TestCase):
    """Test cases for YouTube download functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove downloaded files
        for file in os.listdir(self.temp_dir):
            if file.endswith('.mp3'):
                os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
        
    def test_download_function_returns_status(self):
        """Test that download_audio function returns a boolean status."""
        # Use a non-existent URL to test negative case (should return False)
        result = download_audio("https://www.youtube.com/watch?v=non_existent_video", self.temp_dir + "/")
        self.assertIsInstance(result, bool)
        
    # Skip actual download test in CI environments
    @unittest.skipIf(os.environ.get('CI') == 'true', "Skipping download test in CI environment")
    def test_successful_download_creates_file(self):
        """Test that a successful download creates an MP3 file."""
        # Use a known valid YouTube short test video
        test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # First YouTube video ever
        
        # Download the audio
        result = download_audio(test_url, self.temp_dir + "/")
        
        # Check if download was successful and file exists
        self.assertTrue(result)
        files = [f for f in os.listdir(self.temp_dir) if f.endswith('.mp3')]
        self.assertGreaterEqual(len(files), 1)
        
        # Check if the file has content
        file_path = os.path.join(self.temp_dir, files[0])
        self.assertTrue(os.path.exists(file_path))
        self.assertGreater(os.path.getsize(file_path), 1000)  # File should be larger than 1KB

if __name__ == '__main__':
    unittest.main() 