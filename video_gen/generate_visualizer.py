import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
from moviepy.editor import ImageSequenceClip, AudioFileClip

# CONFIG
AUDIO_FILE = "man_kare_processed.mp3"       # <-- Replace with your file
FRAME_RATE = 30                    # FPS for the video
OUTPUT_VIDEO = "man_kare_processed.mp4"
FRAME_FOLDER = "frames"

def create_frames(audio_path, frame_folder, frame_rate=30):
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    tempo_value = float(np.atleast_1d(tempo)[0])
    print(f"Estimated tempo: {tempo_value:.2f} BPM")
    print(f"Total duration: {duration:.2f} seconds")

    os.makedirs(frame_folder, exist_ok=True)

    times = np.arange(0, duration, 1/frame_rate)
    frame_idx = 0

    for t in times:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # React to beat
        if np.any(np.abs(beat_times - t) < 0.05):  # 50ms window
            color = np.random.choice(["red", "yellow", "cyan", "lime", "magenta"])
            size = 0.7
        else:
            color = "blue"
            size = 0.3

        circle = plt.Circle((0.5, 0.5), size, color=color)
        ax.add_artist(circle)

        # Save frame
        frame_path = os.path.join(frame_folder, f"frame_{frame_idx:05d}.png")
        plt.savefig(frame_path)
        plt.close()
        frame_idx += 1

def create_video(frame_folder, audio_path, output_path, frame_rate=30):
    # Create video from frames
    clip = ImageSequenceClip(frame_folder, fps=frame_rate)
    audio = AudioFileClip(audio_path)
    clip = clip.set_audio(audio)
    clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

def main():
    create_frames(AUDIO_FILE, FRAME_FOLDER, FRAME_RATE)
    create_video(FRAME_FOLDER, AUDIO_FILE, OUTPUT_VIDEO)
    print(f"✅ Music visualizer created successfully: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
