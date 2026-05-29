import os
import random
from tqdm import tqdm
from pydub import AudioSegment
import librosa
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

input_folder = "input/path"
output_folder = "output/path"

bitrate = "320k"

# Noise strength (SNR range in dB)
min_snr = 10
max_snr = 30


mp3_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp3")]

def add_white_noise(y, snr_db):
    signal_power = np.mean(y ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = np.random.normal(0, np.sqrt(noise_power), y.shape)
    return y + noise

def process_file(filename):

    input_path = os.path.join(input_folder, filename)
    base = os.path.splitext(filename)[0][:40]  # keep first 40 chars
    output_filename = base + ".mp3"
    output_path = os.path.join(output_folder, output_filename)

    snr = random.uniform(min_snr, max_snr)

    # Load audio (keep stereo)
    y, sr = librosa.load(input_path, sr=None, mono=False)

    # Add noise
    y_noisy = add_white_noise(y, snr)

    # Normalize
    max_val = np.max(np.abs(y_noisy)) + 1e-9
    y_noisy = y_noisy / max_val

    # Convert to int16
    y_int16 = (y_noisy * 32767).astype(np.int16)

    # Handle stereo
    if y_int16.ndim == 2:
        interleaved = np.empty((y_int16.shape[1] * 2,), dtype=np.int16)
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
    os.makedirs(output_folder, exist_ok=True)
    cpu_count = multiprocessing.cpu_count()
    print(f"Using {cpu_count} CPU cores")

    with ProcessPoolExecutor(max_workers=cpu_count-4) as executor:
        list(tqdm(executor.map(process_file, mp3_files), total=len(mp3_files)))

    print("All files processed successfully!")