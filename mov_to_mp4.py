#!/usr/bin/env python3
import os
import sys
import ffmpeg
import argparse

def convert_mov_to_mp4(input_file, output_file=None):
    """
    Convert a MOV file to MP4 format using ffmpeg
    
    Args:
        input_file: Path to the input MOV file
        output_file: Path to the output MP4 file (optional)
    
    Returns:
        The path to the converted file
    """
    # Check if input file exists
    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return None
    
    # Generate output filename if not provided
    if not output_file:
        filename, _ = os.path.splitext(input_file)
        output_file = f"{filename}.mp4"
    
    print(f"Converting {input_file} to {output_file}...")
    
    try:
        # Run the ffmpeg conversion
        ffmpeg.input(input_file).output(
            output_file, 
            vcodec='libx264',  # Video codec
            acodec='aac',      # Audio codec
            preset='medium',   # Encoding speed/quality balance
            crf=23,            # Quality (lower is better, 18-28 is reasonable)
            audio_bitrate='192k'  # Audio quality
        ).run(capture_stdout=True, capture_stderr=True)
        
        print(f"Conversion complete: {output_file}")
        return output_file
    
    except ffmpeg.Error as e:
        print(f"Error during conversion: {e.stderr.decode('utf8')}")
        return None
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert MOV files to MP4 format")
    parser.add_argument("input_file", help="Path to the input MOV file")
    parser.add_argument("-o", "--output", help="Path to the output MP4 file (optional)")
    
    args = parser.parse_args()
    
    convert_mov_to_mp4(args.input_file, args.output)

if __name__ == "__main__":
    main()
