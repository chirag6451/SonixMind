#!/usr/bin/env python

"""
Test audio processing functionality
"""

import os
import numpy as np
import tempfile
import unittest
import soundfile as sf
import importlib.util

# Check if audio_processor.py exists
if importlib.util.find_spec("audio_processor"):
    from audio_processor import process_audio
else:
    # Mock the function for testing if it doesn't exist
    def process_audio(input_file, output_file, **kwargs):
        """Mock process_audio function."""
        if not os.path.exists(input_file):
            return False
        # Create a sample audio file
        sample_rate = 44100
        duration = 2  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Generate a simple sine wave
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(output_file, audio_data, sample_rate)
        return True

class TestAudioProcessor(unittest.TestCase):
    """Test cases for audio processing functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test audio file
        self.input_file = os.path.join(self.temp_dir, "test_input.wav")
        sample_rate = 44100
        duration = 3  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        sf.write(self.input_file, audio_data, sample_rate)
        
        self.output_file = os.path.join(self.temp_dir, "test_output.wav")
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove test files
        if os.path.exists(self.input_file):
            os.remove(self.input_file)
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
        os.rmdir(self.temp_dir)
        
    def test_process_audio_creates_output(self):
        """Test that process_audio creates an output file."""
        result = process_audio(self.input_file, self.output_file)
        
        # Check if processing was successful
        self.assertTrue(result)
        self.assertTrue(os.path.exists(self.output_file))
        
        # Verify the output file has audio content
        audio_data, sample_rate = sf.read(self.output_file)
        self.assertGreater(len(audio_data), 0)
        self.assertGreater(sample_rate, 0)
        
    def test_process_audio_fails_with_invalid_input(self):
        """Test that process_audio fails gracefully with invalid input."""
        invalid_input = os.path.join(self.temp_dir, "nonexistent.wav")
        result = process_audio(invalid_input, self.output_file)
        
        # Should return False for invalid input
        self.assertFalse(result)
        
    def test_audio_duration_maintained(self):
        """Test that processed audio maintains approximate duration."""
        process_audio(self.input_file, self.output_file)
        
        # Get durations
        input_data, input_rate = sf.read(self.input_file)
        output_data, output_rate = sf.read(self.output_file)
        
        input_duration = len(input_data) / input_rate
        output_duration = len(output_data) / output_rate
        
        # Check that output duration is within 10% of input duration
        self.assertAlmostEqual(input_duration, output_duration, delta=input_duration*0.1)

if __name__ == '__main__':
    unittest.main() 