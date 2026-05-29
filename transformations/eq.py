import os
import random
from tqdm import tqdm
from pydub import AudioSegment
import librosa
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import lfilter

# Parameters
input_folder = "path/input" 
output_folder = "path/output"

bitrate = "320k"

os.makedirs(output_folder, exist_ok=True)

mp3_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp3")][1:15]


def design_peaking_eq(fs, f0, Q, gain_db):
    A = 10**(gain_db / 40)
    omega = 2 * np.pi * f0 / fs
    alpha = np.sin(omega) / (2 * Q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(omega)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(omega)
    a2 = 1 - alpha / A

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])
    return b, a

def apply_random_eq(y, sr):
    # Random EQ parameters
    gain_db = random.uniform(-6, 6)
    center_freq = random.uniform(100, sr // 2 - 1000)
    Q = random.uniform(0.5, 2.0)

    b, a = design_peaking_eq(sr, center_freq, Q, gain_db)
    y_eq = lfilter(b, a, y)

    return y_eq

def process_file(filename):
    input_path = os.path.join(input_folder, filename)
    base_name = os.path.splitext(filename)[0] + "eq"

    # Keep only first part before first underscore (UUID)
    short_name = base_name.split("_")[0]

    output_filename = f"{short_name}.mp3"
    output_path = os.path.join(output_folder, output_filename)

    # Load audio (keep stereo)
    y, sr = librosa.load(input_path, sr=None, mono=False)

    if y.ndim == 1:
        y_eq = apply_random_eq(y, sr)
    else:
        left = apply_random_eq(y[0], sr)
        right = apply_random_eq(y[1], sr)
        min_len = min(len(left), len(right))
        left = left[:min_len]
        right = right[:min_len]
        y_eq = np.vstack([left, right])

    # Normalize
    max_val = np.max(np.abs(y_eq)) + 1e-9
    y_eq = y_eq / max_val

    # Convert to 16-bit PCM
    y_int16 = (y_eq * 32767).astype(np.int16)

    # Convert stereo to interleaved
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

    # Export MP3
    audio_segment.export(output_path, format="mp3", bitrate=bitrate)

    return filename

if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count()
    print(f"Using {cpu_count} CPU cores")

    with ProcessPoolExecutor(max_workers=cpu_count-4) as executor:
        list(tqdm(executor.map(process_file, mp3_files), total=len(mp3_files)))

    print("All files processed successfully!")