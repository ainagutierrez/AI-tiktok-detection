import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

original_folder = r"original/path"
processed_folder = r"output/path"

output_csv = "pitch_shift_validation.csv"

TARGET_SR = 22050
DURATION = 30  # seconds

assert os.path.exists(original_folder)
assert os.path.exists(processed_folder)

def fast_pitch_ratio(y, y_shifted, sr):
    # proxy: spectral centroid shift
    c1 = librosa.feature.spectral_centroid(y=y, sr=sr)
    c2 = librosa.feature.spectral_centroid(y=y_shifted, sr=sr)

    return np.mean(c2) / (np.mean(c1) + 1e-9)


def mel_correlation(y, y_shifted, sr):
    mel1 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel2 = librosa.feature.melspectrogram(y=y_shifted, sr=sr, n_mels=64)

    mel1 = librosa.power_to_db(mel1)
    mel2 = librosa.power_to_db(mel2)

    min_len = min(mel1.shape[1], mel2.shape[1])
    mel1 = mel1[:, :min_len]
    mel2 = mel2[:, :min_len]

    return np.corrcoef(mel1.flatten(), mel2.flatten())[0, 1]


def energy_ratio(y, y_shifted):
    return np.mean(np.abs(y_shifted)) / (np.mean(np.abs(y)) + 1e-9)


def zcr_diff(y, y_shifted):
    z1 = librosa.feature.zero_crossing_rate(y)
    z2 = librosa.feature.zero_crossing_rate(y_shifted)
    return np.mean(z2) - np.mean(z1)


results = []
missing_files = 0

files = [f for f in os.listdir(original_folder) if f.endswith(".mp3")]

for f in tqdm(files):
    try:
        orig_path = os.path.join(original_folder, f)
        shift_path = os.path.join(processed_folder, f)

        if not os.path.exists(shift_path):
            missing_files += 1
            continue

        # Load only first N seconds, downsample
        y, sr = librosa.load(orig_path, sr=TARGET_SR, mono=True, duration=DURATION)
        y_shifted, _ = librosa.load(shift_path, sr=TARGET_SR, mono=True, duration=DURATION)

        length_match = len(y) == len(y_shifted)

        pitch_ratio = fast_pitch_ratio(y, y_shifted, TARGET_SR)
        mel_corr = mel_correlation(y, y_shifted, TARGET_SR)
        energy = energy_ratio(y, y_shifted)
        zcr = zcr_diff(y, y_shifted)


        bad_pitch = (
            np.isnan(pitch_ratio) or
            pitch_ratio < 0.7 or
            pitch_ratio > 1.5
        )

        bad_mel = mel_corr < 0.5
        bad_length = not length_match

        results.append({
            "file": f,
            "length_match": length_match,
            "pitch_ratio_proxy": pitch_ratio,
            "mel_correlation": mel_corr,
            "energy_ratio": energy,
            "zcr_diff": zcr,
            "bad_pitch": bad_pitch,
            "bad_mel": bad_mel,
            "bad_length": bad_length
        })

    except Exception as e:
        print(f"Error processing {f}: {type(e).__name__} - {e}")

df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

print("\nValidation complete.")
print(f"Saved to {output_csv}")
print("Missing processed files:", missing_files)


if df.empty:
    print("\nNo valid files were processed.")
    exit()

print("\n=== SUMMARY ===")
print("Total files:", len(df))
print("Bad pitch:", df["bad_pitch"].sum())
print("Bad mel:", df["bad_mel"].sum())
print("Length mismatch:", df["bad_length"].sum())

print("\nAverages:")
print(df.mean(numeric_only=True))