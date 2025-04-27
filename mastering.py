# Install needed libraries first:
# pip install noisereduce pedalboard librosa numpy soundfile

import noisereduce as nr
import pedalboard
from pedalboard import Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile
import librosa
import soundfile as sf
import numpy as np

# 1. Load your audio
input_file = 's1_high_quality.wav'
y, sr = librosa.load(input_file, sr=None)

# 2. Noise reduction
y_denoised = nr.reduce_noise(y=y, sr=sr)

# 3. (Optional) Breath removal — simple low-energy removal
energy = np.abs(y_denoised)
threshold = np.percentile(energy, 5)  # breaths are usually in low-energy parts
y_cleaned = np.where(energy > threshold, y_denoised, 0)

# 4. Apply professional effects
with AudioFile(input_file) as f:
    audio = f.read(f.frames)
    samplerate = f.samplerate

board = pedalboard.Pedalboard([
    HighpassFilter(cutoff_frequency_hz=80.0),   # cut low-end rumble
    Compressor(threshold_db=-24, ratio=3.0),
    Reverb(room_size=0.2),
])

processed = board(y_cleaned, sr)

# 5. Save the output
sf.write('final_song.wav', processed, sr)
print("✅ Your pro song is ready!")
