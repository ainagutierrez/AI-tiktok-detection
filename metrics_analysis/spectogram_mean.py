import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

folder = r"path"

audio_extensions = (".mp3", ".wav", ".flac", ".m4a", ".ogg")

files = [
    f for f in os.listdir(folder)
    if f.lower().endswith(audio_extensions)
]

SR = 16000
DURATION = 60
HOP_LENGTH = 512
MIN_SAMPLES = SR * DURATION

raw_specs = []
skipped = 0

for file in tqdm(files, desc="Processing audio files"):
    path = os.path.join(folder, file)
    y, sr = librosa.load(path, sr=SR, mono=True, duration=DURATION)
    if len(y) < MIN_SAMPLES:
        skipped += 1
        continue
    # Trim to exactly MIN_SAMPLES so all spectrograms have identical frame counts
    y = y[:MIN_SAMPLES]
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP_LENGTH)
    S_db = librosa.power_to_db(S, ref=np.max)
    raw_specs.append(S_db)

print(f"Skipped {skipped} files shorter than 1 minute. Using {len(raw_specs)} files.")

specs = np.stack(raw_specs, axis=0)
avg = np.mean(specs, axis=0)

plt.figure(figsize=(8, 4))
librosa.display.specshow(avg, sr=SR, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel")
plt.title("Average Suno Spectrogram (First 60s)")
plt.colorbar()
plt.tight_layout()
plt.show()