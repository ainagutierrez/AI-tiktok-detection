import os
import random
import hashlib
from tqdm import tqdm
from pydub import AudioSegment
import librosa
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

input_folder = r"path/input"
output_folder = r"path/output"

min_steps = -5
max_steps = 5
bitrate = "320k"

os.makedirs(output_folder, exist_ok=True)

mp3_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp3")][:10]

MIN_LEN = 2048  

def to_audiosegment(y, sr):
    y = np.nan_to_num(y)

    # mono
    if y.ndim == 1:
        audio = AudioSegment(
            (y * 32767).astype(np.int16).tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )
        return audio

    # stereo-safe
    left = AudioSegment(
        (y[0] * 32767).astype(np.int16).tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )

    right = AudioSegment(
        (y[1] * 32767).astype(np.int16).tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )

    return AudioSegment.from_mono_audiosegments(left, right)


def process_file(filename):
    try:
        input_path = os.path.join(input_folder, filename)

        base_name = os.path.splitext(filename)[0] + "pitchshift"
        short_hash = hashlib.md5(base_name.encode()).hexdigest()[:12]
        output_path = os.path.join(output_folder, short_hash + ".mp3")

        pitch_steps = random.uniform(min_steps, max_steps)

        y, sr = librosa.load(input_path, sr=None, mono=False)

        if y is None or len(y) == 0:
            return f"ERROR (empty load): {filename}"

        if y.ndim == 1:
            is_stereo = False
            length_ok = len(y) >= MIN_LEN
        else:
            # ensure (channels, samples)
            if y.shape[0] > y.shape[1]:
                y = y.T

            is_stereo = True
            length_ok = y.shape[1] >= MIN_LEN

        if not length_ok:
            y_shifted = y  # fallback: no processing
        else:
            if not is_stereo:
                y_shifted = librosa.effects.pitch_shift(
                    y, sr=sr, n_steps=pitch_steps
                )
            else:
                left = librosa.effects.pitch_shift(y[0], sr=sr, n_steps=pitch_steps)
                right = librosa.effects.pitch_shift(y[1], sr=sr, n_steps=pitch_steps)

                min_len = min(len(left), len(right))
                y_shifted = np.vstack([left[:min_len], right[:min_len]])


        if y_shifted is None or len(y_shifted) == 0:
            return f"ERROR (empty after processing): {filename}"

        if np.allclose(y_shifted, 0):
            return f"ERROR (silent output): {filename}"

        max_val = np.max(np.abs(y_shifted))
        if max_val > 1e-9:
            y_shifted = y_shifted / max_val

        audio_segment = to_audiosegment(y_shifted, sr)

        if audio_segment.duration_seconds == 0:
            return f"ERROR (zero duration export): {filename}"

        audio_segment.export(output_path, format="mp3", bitrate=bitrate)

        return f"OK: {filename}"

    except Exception as e:
        import traceback
        return f"ERROR: {filename}\n{traceback.format_exc()}"



if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count()
    print(f"Using {cpu_count} CPU cores")

    with ProcessPoolExecutor(max_workers=cpu_count-4) as executor:
        results = list(tqdm(executor.map(process_file, mp3_files), total=len(mp3_files)))

    errors = [r for r in results if r.startswith("ERROR")]

    if errors:
        print("\nSome files failed:")
        for e in errors:
            print(e)
    else:
        print("\nAll files processed successfully!")