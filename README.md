# SonixMind

SonixMind is a powerful audio and video processing application that helps you extract, enhance, and master audio with professional-quality results.

## AI-Powered Audio Processing

SonixMind harnesses the power of cutting-edge AI models to transform complex audio engineering tasks into a simple, three-step process:

1. **Upload** - Simply upload your audio or video file
2. **Visualize** - See detailed waveforms and insights about your audio
3. **Process & Export** - Apply AI-powered enhancements and export high-quality audio

Behind the scenes, sophisticated neural networks analyze your audio to identify the optimal processing parameters. The system uses machine learning models trained on thousands of professional audio samples to:

- Extract clean audio from video files with intelligent noise suppression
- Apply adaptive equalization based on content type (voice, music, etc.)
- Match the spectral and dynamic characteristics of professional reference tracks
- Enhance clarity and reduce background noise through neural processing

All of this advanced technology works automatically, without requiring deep technical knowledge or expensive studio equipment.

## Features

- **Audio Extraction**: Extract high-quality audio from video files (MP4, MOV)
- **Audio Processing**: Normalize audio levels, enhance clarity, and apply custom EQ
- **Professional Mastering**: Match your audio to professional reference tracks
- **Visualization**: View detailed waveforms, spectrograms, and audio statistics
- **YouTube Reference**: Download audio from YouTube to use as reference tracks

## Creator's Story

I am a big fan of Kishore Kumar, the famous Indian film singer. I also love singing very much. I often sing on different platforms.

While uploading my songs and music videos on sites like YouTube, I had to do a lot of post-processing using different tools. Going to a recording studio was very costly. Even the AI tool websites I used for song mastering were also very expensive.

One fine weekend, I decided to create something to meet my own needs. To my surprise, it also became useful for other singers. This is how SonixMind was born — from my love for singing and the need for affordable, high-quality audio processing tools that anyone can use.

## Requirements

- Python 3.8 or higher
- FFmpeg
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/chirag6451/SonixMind.git
cd SonixMind
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg if not already installed:
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

4. Set up VoiceFixer model files:
   - The application already contains the `voicefixer` directory
   - Download the required model checkpoints from [VoiceFixer's releases](https://github.com/haoheliu/voicefixer/releases) or using the instructions in `voicefixer/README.md`
   - For users in mainland China, alternative download links are available:
     ```
     百度网盘: https://pan.baidu.com/s/194ufkUR_PYf1nE1KqkEZjQ (提取密码: qis6)
     ```
   - Place the downloaded files according to this structure:
     ```
     voicefixer/
     ├── vf.ckpt (place in ~/.cache/voicefixer/analysis_module/checkpoints/)
     └── model.ckpt-1490000_trimed.pt (place in ~/.cache/voicefixer/synthesis_module/44100/)
     ```

5. Check dependencies:
```bash
python check_dependencies.py
```

## Usage

Run the application:
```bash
streamlit run app.py
```

For detailed usage instructions, please refer to the [HELP.md](HELP.md) file.

## AI Models Used

SonixMind integrates several state-of-the-art AI models to deliver professional audio processing:

### 1. VoiceFixer
- **Purpose**: Speech restoration and enhancement
- **Capabilities**: Restores degraded speech regardless of severity, handling noise, reverberation, and low resolution
- **Paper**: [VoiceFixer: Toward General Speech Restoration With Neural Vocoder](https://arxiv.org/abs/2109.13731)
- **Model Architecture**: Neural vocoder-based restoration
- **Author**: Haohe Liu et al.

### 2. Matchering
- **Purpose**: Audio mastering to match professional reference tracks
- **Capabilities**: Analyzes target and reference audio to match spectral balance, loudness, and dynamics
- **GitHub**: [Sergree/matchering](https://github.com/sergree/matchering)
- **Technology**: Advanced DSP algorithms and psychoacoustic models

### 3. Streamlit ML Components
- **Purpose**: Interactive visualization and processing
- **Capabilities**: Real-time audio processing with visual feedback
- **Technology**: Streamlit's machine learning-optimized components

### 4. Custom Neural Processing
- **Purpose**: Audio enhancement and noise reduction
- **Capabilities**: Custom-tuned algorithms for audio clarity enhancement
- **Implementation**: Based on spectral processing techniques with neural optimizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please see the CONTRIBUTING.md file for guidelines.

## Technologies

- Streamlit for the web interface
- FFmpeg for audio/video conversion
- Matchering for professional mastering
- Numpy, SciPy, and Librosa for audio processing
- youtube_dl/yt-dlp for YouTube audio reference extraction

## Acknowledgements

- [VoiceFixer](https://github.com/haoheliu/voicefixer) by Haohe Liu for the speech restoration model
- [Matchering](https://github.com/sergree/matchering) for the mastering engine
- FFmpeg for audio/video processing capabilities
- Streamlit for the interactive web interface framework

## Authors

- **Chirag Kansara** - Lead Developer - [@indapoint](https://www.linkedin.com/in/indapoint/)
- Contact: chirag@indapoint.com

## Company

- **IndaPoint Technologies Pvt. Ltd.**
- Website: [www.indapoint.com](https://www.indapoint.com/)
- Email: info@indapoint.com

Copyright ©2025 IndaPoint Technologies Pvt. Ltd. All rights reserved.
