import numpy as np
import soundfile as sf
import os

def create_sine_wave(freq, duration, sample_rate=44100):
    """Create a sine wave at the given frequency with the given duration."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return 0.5 * np.sin(2 * np.pi * freq * t)

def create_pop_reference():
    """Create a reference track with pop music characteristics."""
    sr = 44100
    duration = 5.0
    
    # Create base frequencies
    kick = np.zeros(int(sr * duration))
    for i in range(0, int(sr * duration), sr // 2):  # Kick every half second
        if i + 5000 < len(kick):
            kick[i:i+5000] = np.sin(2 * np.pi * 60 * np.linspace(0, 5000/sr, 5000)) * np.exp(-np.linspace(0, 10, 5000))
    
    # Snare on 2 and 4
    snare = np.zeros(int(sr * duration))
    for i in range(sr//2, int(sr * duration), sr):  # Snare on 2 and 4
        if i + 3000 < len(snare):
            snare[i:i+3000] = np.random.normal(0, 0.3, 3000) * np.exp(-np.linspace(0, 10, 3000))
    
    # Bass line
    bass = np.zeros(int(sr * duration))
    bass_seq = [60, 60, 67, 65]  # Simple bass sequence in MIDI notes
    for i, note in enumerate(bass_seq * 5):
        start = (i * sr) // 4
        if start + sr//4 < len(bass):
            freq = 440 * 2**((note - 69) / 12)  # Convert MIDI to Hz
            bass[start:start+sr//4] = create_sine_wave(freq, 0.25, sr)
    
    # Pad sound
    pad = np.zeros(int(sr * duration))
    for t in range(5):
        chord = [60, 64, 67]  # C major chord
        for note in chord:
            freq = 440 * 2**((note - 69) / 12)
            sine = create_sine_wave(freq, 1.0, sr)
            pad[t*sr:(t+1)*sr] += sine * 0.2
    
    # Apply some compression and EQ (simplified)
    audio = (kick * 0.7) + (snare * 0.6) + (bass * 0.8) + (pad * 0.3)
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.9
    
    # Apply "pop" mastering curve - boost highs and lows
    audio = np.clip(audio * 1.2, -1.0, 1.0)
    
    return audio

def create_rock_reference():
    """Create a reference track with rock music characteristics."""
    sr = 44100
    duration = 5.0
    
    # Create distorted guitar-like sound
    guitar = np.zeros(int(sr * duration))
    guitar_seq = [52, 55, 59, 55]  # Power chord sequence
    for i, note in enumerate(guitar_seq * 5):
        start = (i * sr) // 4
        if start + sr//4 < len(guitar):
            freq = 440 * 2**((note - 69) / 12)
            sine = create_sine_wave(freq, 0.25, sr)
            # Add harmonics for distortion
            sine += 0.5 * create_sine_wave(freq * 2, 0.25, sr)
            sine += 0.3 * create_sine_wave(freq * 3, 0.25, sr)
            guitar[start:start+sr//4] = np.clip(sine * 1.5, -1.0, 1.0)  # Clip for distortion
    
    # Drum pattern
    kick = np.zeros(int(sr * duration))
    for i in range(0, int(sr * duration), sr // 2):  # Kick drum
        if i + 5000 < len(kick):
            kick[i:i+5000] = np.sin(2 * np.pi * 50 * np.linspace(0, 5000/sr, 5000)) * np.exp(-np.linspace(0, 10, 5000))
    
    # Hard-hitting snare
    snare = np.zeros(int(sr * duration))
    for i in range(sr//2, int(sr * duration), sr):  # Snare on 2 and 4
        if i + 4000 < len(snare):
            snare[i:i+4000] = np.random.normal(0, 0.5, 4000) * np.exp(-np.linspace(0, 8, 4000))
    
    # Bass line
    bass = np.zeros(int(sr * duration))
    bass_seq = [40, 40, 47, 45]  # Heavy bass line
    for i, note in enumerate(bass_seq * 5):
        start = (i * sr) // 4
        if start + sr//4 < len(bass):
            freq = 440 * 2**((note - 69) / 12)
            sine = create_sine_wave(freq, 0.25, sr)
            # Add some harmonics for richness
            sine += 0.3 * create_sine_wave(freq * 2, 0.25, sr)
            bass[start:start+sr//4] = sine
    
    # Mix with rock-style compression
    audio = (kick * 0.8) + (snare * 0.7) + (bass * 0.7) + (guitar * 0.6)
    
    # Hard compression for rock sound
    threshold = 0.3
    ratio = 4.0
    mask = np.abs(audio) > threshold
    audio[mask] = np.sign(audio[mask]) * (threshold + (np.abs(audio[mask]) - threshold) / ratio)
    
    # Normalize with rock-style limiting
    audio = audio / np.max(np.abs(audio)) * 0.95
    
    return audio

def create_electronic_reference():
    """Create a reference track with electronic music characteristics."""
    sr = 44100
    duration = 5.0
    
    # Create synth bass
    bass = np.zeros(int(sr * duration))
    bass_seq = [36, 36, 36, 43]  # EDM-style bass sequence
    for i, note in enumerate(bass_seq * 5):
        start = (i * sr) // 4
        if start + sr//4 < len(bass):
            freq = 440 * 2**((note - 69) / 12)
            # Saw wave for electronic bass
            t = np.linspace(0, 0.25, sr//4)
            saw = 2 * (t * freq - np.floor(0.5 + t * freq))
            # Apply filter envelope
            env = np.exp(-np.linspace(0, 5, sr//4))
            bass[start:start+sr//4] = saw * env
    
    # Four-on-the-floor kick
    kick = np.zeros(int(sr * duration))
    for i in range(0, int(sr * duration), sr // 4):  # Kick on every beat
        if i + 3000 < len(kick):
            kick[i:i+3000] = np.sin(2 * np.pi * 60 * np.linspace(0, 3000/sr, 3000)) * np.exp(-np.linspace(0, 12, 3000))
    
    # Hi-hat pattern
    hihat = np.zeros(int(sr * duration))
    for i in range(0, int(sr * duration), sr // 8):  # Eighth note hi-hats
        if i + 1000 < len(hihat):
            hihat[i:i+1000] = np.random.normal(0, 0.2, 1000) * np.exp(-np.linspace(0, 15, 1000))
    
    # Synth pad
    pad = np.zeros(int(sr * duration))
    chord = [60, 64, 67, 71]  # 7th chord
    for note in chord:
        freq = 440 * 2**((note - 69) / 12)
        sine = create_sine_wave(freq, duration, sr)
        # Add some movement with LFO
        lfo = 0.1 * np.sin(2 * np.pi * 0.5 * np.linspace(0, duration, int(sr * duration)))
        pad += sine * (0.15 + lfo)
    
    # Mix with electronic-style sidechaining
    audio = (kick * 0.9) + (hihat * 0.3) + (bass * 0.8) + (pad * 0.3)
    
    # Apply sidechain compression effect
    for i in range(0, int(sr * duration), sr // 4):
        if i + sr//8 < len(audio):
            sidechain_env = np.exp(np.linspace(-5, 0, sr//8))
            audio[i:i+sr//8] = audio[i:i+sr//8] * sidechain_env.reshape(-1, 1) if len(audio.shape) > 1 else audio[i:i+sr//8] * sidechain_env
    
    # Apply limiting for EDM loudness
    audio = np.clip(audio * 1.5, -1.0, 1.0)
    
    # Normalize with brick-wall limiting
    audio = audio / np.max(np.abs(audio)) * 0.99
    
    return audio

def create_hiphop_reference():
    """Create a reference track with hip-hop characteristics."""
    sr = 44100
    duration = 5.0
    
    # Create boom-bap style kick
    kick = np.zeros(int(sr * duration))
    for i in range(0, int(sr * duration), sr // 2):  # Half-note kicks
        if i + 6000 < len(kick):
            kick[i:i+6000] = np.sin(2 * np.pi * np.linspace(80, 40, 6000) * np.linspace(0, 6000/sr, 6000)) * np.exp(-np.linspace(0, 8, 6000))
    
    # Snare on 2 and 4
    snare = np.zeros(int(sr * duration))
    for i in range(sr//2, int(sr * duration), sr):
        if i + 4000 < len(snare):
            snare[i:i+4000] = np.random.normal(0, 0.4, 4000) * np.exp(-np.linspace(0, 7, 4000))
    
    # Hi-hat pattern
    hihat = np.zeros(int(sr * duration))
    for i in range(0, int(sr * duration), sr // 8):  # Eighth note hi-hats
        if i + 800 < len(hihat):
            hihat[i:i+800] = np.random.normal(0, 0.15, 800) * np.exp(-np.linspace(0, 20, 800))
    
    # Bass line
    bass = np.zeros(int(sr * duration))
    bass_seq = [36, 36, 41, 43]  # Hip-hop bass line
    for i, note in enumerate(bass_seq * 5):
        start = (i * sr) // 4
        if start + sr//4 < len(bass):
            freq = 440 * 2**((note - 69) / 12)
            t = np.linspace(0, 0.25, sr//4)
            # Sine with slight distortion for warmth
            sine = np.sin(2 * np.pi * freq * t)
            sine = np.clip(sine * 1.2, -1.0, 1.0)
            bass[start:start+sr//4] = sine * np.exp(-np.linspace(0, 1, sr//4))
    
    # Sample-like melodic element (simplified)
    sample = np.zeros(int(sr * duration))
    for t in range(5):
        chord = [65, 69, 72]  # Sample chord
        for note in chord:
            freq = 440 * 2**((note - 69) / 12)
            sine = create_sine_wave(freq, 1.0, sr)
            # Add some vinyl-like artifacts
            crackle = np.random.normal(0, 0.02, sr)
            sample[t*sr:(t+1)*sr] += (sine * 0.3) + crackle
    
    # Mix with hip-hop style processing
    audio = (kick * 0.9) + (snare * 0.7) + (hihat * 0.25) + (bass * 0.85) + (sample * 0.6)
    
    # Apply hip-hop style compression
    threshold = 0.4
    ratio = 3.0
    mask = np.abs(audio) > threshold
    audio[mask] = np.sign(audio[mask]) * (threshold + (np.abs(audio[mask]) - threshold) / ratio)
    
    # Apply low-pass filter for warmth
    audio_fft = np.fft.rfft(audio)
    freq = np.fft.rfftfreq(len(audio), 1/sr)
    filter_curve = 1 / (1 + (freq / 10000) ** 4)  # Simple low-pass filter
    audio_fft *= filter_curve
    audio = np.fft.irfft(audio_fft, len(audio))
    
    # Normalize with slight saturation
    audio = np.tanh(audio * 1.2) * 0.9
    
    return audio

def create_classical_reference():
    """Create a reference track with classical/acoustic characteristics."""
    sr = 44100
    duration = 5.0
    
    # Create string ensemble
    strings = np.zeros(int(sr * duration))
    chord_seq = [
        [60, 64, 67],  # C major
        [62, 65, 69],  # D minor
        [64, 67, 71],  # E minor
        [65, 69, 72],  # F major
    ]
    
    for i, chord in enumerate(chord_seq):
        start = i * sr
        if start + sr < len(strings):
            for note in chord:
                freq = 440 * 2**((note - 69) / 12)
                sine = create_sine_wave(freq, 1.0, sr)
                # Add vibrato
                vibrato = np.sin(2 * np.pi * 5 * np.linspace(0, 1.0, sr)) * 0.005
                vib_time = np.linspace(0, 1.0, sr) + vibrato
                vib_time = np.clip(vib_time * sr, 0, sr-1).astype(int)
                sine_vib = sine[vib_time]
                # Add harmonics for string-like sound
                sine_vib += 0.3 * create_sine_wave(freq * 2, 1.0, sr)
                sine_vib += 0.15 * create_sine_wave(freq * 3, 1.0, sr)
                sine_vib += 0.075 * create_sine_wave(freq * 4, 1.0, sr)
                strings[start:start+sr] += sine_vib * 0.2
    
    # Piano melody
    piano = np.zeros(int(sr * duration))
    melody = [72, 71, 69, 71, 72, 72, 72, 0, 71, 71, 71, 0, 72, 76, 74]  # Simple melody
    note_duration = sr // 8  # Eighth notes
    
    for i, note in enumerate(melody):
        start = (i * note_duration) % (int(sr * duration) - note_duration)
        if note > 0:  # Not a rest
            freq = 440 * 2**((note - 69) / 12)
            # Piano-like envelope
            env = np.exp(-np.linspace(0, 10, note_duration))
            sine = create_sine_wave(freq, note_duration/sr, sr)
            # Add harmonics for piano-like sound
            sine += 0.5 * create_sine_wave(freq * 2, note_duration/sr, sr)
            sine += 0.25 * create_sine_wave(freq * 3, note_duration/sr, sr)
            piano[start:start+note_duration] = sine[:note_duration] * env
    
    # Acoustic bass
    bass = np.zeros(int(sr * duration))
    bass_notes = [48, 50, 52, 53]  # Simple bass line
    for i, note in enumerate(bass_notes):
        start = i * sr
        if start + sr < len(bass):
            freq = 440 * 2**((note - 69) / 12)
            # Double bass like sound
            sine = create_sine_wave(freq, 1.0, sr)
            # Body resonance
            sine += 0.4 * create_sine_wave(freq * 2, 1.0, sr)
            bass[start:start+sr] = sine * np.exp(-np.linspace(0, 3, sr))
    
    # Mix with classical dynamic range
    audio = (strings * 0.7) + (piano * 0.6) + (bass * 0.5)
    
    # Apply gentle mastering (preserve dynamics)
    audio = audio / np.max(np.abs(audio)) * 0.85
    
    # Add subtle room ambience
    reverb_time = int(0.5 * sr)  # 500ms reverb
    reverb = np.zeros(len(audio) + reverb_time)
    for i in range(len(audio)):
        # Simple convolution-based reverb approximation
        decay = np.exp(-np.linspace(0, 10, reverb_time))
        if i < len(reverb) - reverb_time:
            reverb[i:i+reverb_time] += audio[i] * decay * 0.3
    
    # Mix direct sound with reverb
    audio_with_reverb = audio + reverb[:len(audio)] * 0.4
    
    # Normalize again
    audio_with_reverb = audio_with_reverb / np.max(np.abs(audio_with_reverb)) * 0.85
    
    return audio_with_reverb

def main():
    # Create directory for reference tracks if it doesn't exist
    if not os.path.exists('references'):
        os.makedirs('references')
    
    # Generate and save reference tracks
    references = {
        'pop_reference.wav': create_pop_reference(),
        'rock_reference.wav': create_rock_reference(),
        'electronic_reference.wav': create_electronic_reference(),
        'hiphop_reference.wav': create_hiphop_reference(),
        'classical_reference.wav': create_classical_reference()
    }
    
    for filename, audio in references.items():
        print(f"Creating {filename}...")
        sf.write(filename, audio, 44100)
        print(f"Saved {filename}")

if __name__ == "__main__":
    main() 