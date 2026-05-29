import os
import random
from tqdm import tqdm
from pydub import AudioSegment
import librosa
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import hashlib

input_folder = r"input/path"
output_folder = r"output/path"
min_stretch = 0.5
max_stretch = 1.5
bitrate = "320k"

os.makedirs(output_folder, exist_ok=True)

def safe_filename(filename):
    name_hash = hashlib.sha1(filename.encode()).hexdigest()[:16]
    return f"{name_hash}.mp3"


mp3_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp3")]

def high_quality_time_stretch(y, sr, rate):
    n_fft = 4096
    hop_length = n_fft // 4
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    D_stretched = librosa.phase_vocoder(D, rate=rate, hop_length=hop_length)
    y_stretched = librosa.istft(D_stretched, hop_length=hop_length)
    return y_stretched

def process_file(filename):
    input_path = os.path.join(input_folder, filename)
    output_filename = safe_filename(filename)
    output_path = os.path.join(output_folder, output_filename)

    stretch_factor = random.uniform(min_stretch, max_stretch)

    # Load audio (stereo if available)
    y, sr = librosa.load(input_path, sr=None, mono=False)

    if y.ndim == 1:
        y_stretched = high_quality_time_stretch(y, sr, stretch_factor)
    else:
        left = high_quality_time_stretch(y[0], sr, stretch_factor)
        right = high_quality_time_stretch(y[1], sr, stretch_factor)
        min_len = min(len(left), len(right))
        left = left[:min_len]
        right = right[:min_len]
        y_stretched = np.vstack([left, right])

    # Normalize
    max_val = np.max(np.abs(y_stretched)) + 1e-9
    y_stretched = y_stretched / max_val

    # Convert to 16-bit PCM
    y_int16 = (y_stretched * 32767).astype(np.int16)

    # Stereo or mono
    if y_int16.ndim == 2:
        interleaved = np.empty((y_int16.shape[1]*2,), dtype=np.int16)
        interleaved[0::2] = y_int16[0]
        interleaved[1::2] = y_int16[1]
        audio_segment = AudioSegment(
            interleaved.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=2
        )
    else:
        audio_segment = AudioSegment(
            y_int16.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )

    # Export to MP3
    audio_segment.export(output_path, format="mp3", bitrate=bitrate)

    return filename

if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count()
    print(f"Using {cpu_count} CPU cores")

    with ProcessPoolExecutor(max_workers=cpu_count-4) as executor:
        list(tqdm(executor.map(process_file, mp3_files), total=len(mp3_files)))

    print("All files processed successfully!")