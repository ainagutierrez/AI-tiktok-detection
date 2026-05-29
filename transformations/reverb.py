import os
import random
from tqdm import tqdm
from pydub import AudioSegment
import librosa
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import hashlib
from scipy.signal import fftconvolve

# =======================
# PARAMETERS
# =======================
input_folder = "input/path"
output_folder = "output/path"

min_decay = 0.2   # seconds
max_decay = 1.0   # seconds
bitrate = "320k"

os.makedirs(output_folder, exist_ok=True)

def safe_filename(filename):
    name_hash = hashlib.sha1(filename.encode()).hexdigest()[:16]
    return f"{name_hash}.mp3"

mp3_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp3")]

def generate_ir(sr, decay_time):
    length = int(sr * decay_time)
    t = np.linspace(0, decay_time, length)
    ir = np.exp(-3 * t / decay_time)
    ir *= np.random.uniform(0.9, 1.0, size=length)  # slight diffusion
    return ir / np.max(np.abs(ir))

def apply_reverb(y, sr, decay_time):
    ir = generate_ir(sr, decay_time)
    return fftconvolve(y, ir, mode="full")

def process_file(filename):
    input_path = os.path.join(input_folder, filename)
    output_filename = safe_filename(filename)
    output_path = os.path.join(output_folder, output_filename)

    decay_time = random.uniform(min_decay, max_decay)

    y, sr = librosa.load(input_path, sr=None, mono=False)

    if y.ndim == 1:
        y_rev = apply_reverb(y, sr, decay_time)
    else:
        left = apply_reverb(y[0], sr, decay_time)
        right = apply_reverb(y[1], sr, decay_time)
        min_len = min(len(left), len(right))
        y_rev = np.vstack([left[:min_len], right[:min_len]])

    # Normalize
    max_val = np.max(np.abs(y_rev)) + 1e-9
    y_rev = y_rev / max_val

    # Convert to 16-bit PCM
    y_int16 = (y_rev * 32767).astype(np.int16)

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

    audio_segment.export(output_path, format="mp3", bitrate=bitrate)

    return filename

if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count()
    print(f"Using {cpu_count} CPU cores")

    with ProcessPoolExecutor(max_workers=cpu_count-4) as executor:
        list(tqdm(executor.map(process_file, mp3_files), total=len(mp3_files)))

    print("All files processed successfully!")