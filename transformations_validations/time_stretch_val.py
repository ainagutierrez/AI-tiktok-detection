import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

original_folder = r"original/path"
processed_folder = r"output/path"

output_folder = os.path.join(processed_folder, "validation_results_fast")
os.makedirs(output_folder, exist_ok=True)

excerpt_duration = 20

expected_min_ratio = 1 / 1.5  
expected_max_ratio = 1 / 0.5  
ratio_tolerance = 0.03

def load_audio(path):
    y, sr = librosa.load(path, sr=None, mono=False)
    return y, sr

def to_mono(y):
    if y.ndim == 2:
        return np.mean(y, axis=0)
    return y

def get_num_channels(y):
    return 1 if y.ndim == 1 else y.shape[0]

def duration(y, sr):
    if y.ndim == 2:
        return y.shape[1] / sr
    return len(y) / sr

def get_excerpt(y, sr, seconds=20):
    y = to_mono(y)
    target_len = int(seconds * sr)

    if len(y) <= target_len:
        return y

    start = max(0, (len(y) - target_len) // 2)
    end = start + target_len
    return y[start:end]

def basic_stats(y):
    peak = np.max(np.abs(y))
    rms = np.sqrt(np.mean(y**2))
    silence_ratio = np.mean(np.abs(y) < 1e-4)
    clipping_ratio = np.mean(np.abs(y) >= 0.999)
    has_nan = np.isnan(y).any()
    has_inf = np.isinf(y).any()

    return {
        "peak": peak,
        "rms": rms,
        "silence_ratio": silence_ratio,
        "clipping_ratio": clipping_ratio,
        "has_nan": has_nan,
        "has_inf": has_inf
    }


def compare_features(y1, sr1, y2, sr2):
    if sr1 != sr2:
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
        sr2 = sr1

    # MFCC distance 
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)

    mfcc1_mean = np.mean(mfcc1, axis=1)
    mfcc2_mean = np.mean(mfcc2, axis=1)
    mfcc_distance = np.linalg.norm(mfcc1_mean - mfcc2_mean)

    # Spectral centroid 
    cent1 = librosa.feature.spectral_centroid(y=y1, sr=sr1)[0]
    cent2 = librosa.feature.spectral_centroid(y=y2, sr=sr2)[0]
    centroid_diff = abs(np.mean(cent1) - np.mean(cent2))

    # Onset envelope correlation 
    onset1 = librosa.onset.onset_strength(y=y1, sr=sr1)
    onset2 = librosa.onset.onset_strength(y=y2, sr=sr2)

    min_len = min(len(onset1), len(onset2))
    onset1 = onset1[:min_len]
    onset2 = onset2[:min_len]

    if min_len < 5 or np.std(onset1) < 1e-8 or np.std(onset2) < 1e-8:
        onset_corr = 0.0
    else:
        onset_corr = np.corrcoef(onset1, onset2)[0, 1]
        if np.isnan(onset_corr):
            onset_corr = 0.0

    return {
        "mfcc_distance": mfcc_distance,
        "centroid_diff": centroid_diff,
        "onset_corr": onset_corr
    }

def classify_file(row):
    problems = []

    if not (expected_min_ratio - ratio_tolerance <= row["duration_ratio"] <= expected_max_ratio + ratio_tolerance):
        problems.append("duration_out_of_expected_range")

    if row["has_nan"] or row["has_inf"]:
        problems.append("invalid_audio_values")

    if row["silence_ratio"] > 0.95:
        problems.append("mostly_silent")

    if row["peak"] < 0.05:
        problems.append("too_quiet")

    if row["clipping_ratio"] > 0.05:
        problems.append("possible_clipping")

    if row["mfcc_distance"] > 80:
        problems.append("high_timbre_difference")

    if row["onset_corr"] < 0.20:
        problems.append("low_rhythm_similarity")

    status = "PASS" if len(problems) <= 1 else "REVIEW"
    return status, "; ".join(problems) if problems else "ok"

def process_file(f):
    orig_path = os.path.join(original_folder, f)
    proc_path = os.path.join(processed_folder, f)

    if not os.path.exists(proc_path):
        return {
            "file": f,
            "status": "ERROR",
            "notes": "processed file missing"
        }

    try:
        y1, sr1 = load_audio(orig_path)
        y2, sr2 = load_audio(proc_path)

        # Full-file duration check
        dur1 = duration(y1, sr1)
        dur2 = duration(y2, sr2)
        duration_ratio = dur2 / (dur1 + 1e-9)

        orig_channels = get_num_channels(y1)
        proc_channels = get_num_channels(y2)

        # Excerpts only for feature analysis
        y1_excerpt = get_excerpt(y1, sr1, excerpt_duration)
        y2_excerpt = get_excerpt(y2, sr2, excerpt_duration)

        stats = basic_stats(y2_excerpt)
        feats = compare_features(y1_excerpt, sr1, y2_excerpt, sr2)

        row = {
            "file": f,
            "orig_sr": sr1,
            "proc_sr": sr2,
            "orig_channels": orig_channels,
            "proc_channels": proc_channels,
            "orig_duration": dur1,
            "proc_duration": dur2,
            "duration_ratio": duration_ratio,
            **stats,
            **feats
        }

        status, notes = classify_file(row)
        row["status"] = status
        row["notes"] = notes

        return row

    except Exception as e:
        return {
            "file": f,
            "status": "ERROR",
            "notes": str(e)
        }

if __name__ == "__main__":
    files = [f for f in os.listdir(original_folder) if f.lower().endswith(".mp3")]

    print(f"Using {multiprocessing.cpu_count()} CPU cores")

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()-4) as executor:
        results = list(tqdm(executor.map(process_file, files), total=len(files), desc="Validating"))

    df = pd.DataFrame(results)

    csv_path = os.path.join(output_folder, "validation_summary_fast.csv")
    df.to_csv(csv_path, index=False)

    def save_hist(data, title, xlabel, filename, bins=30):
        data = [x for x in data if pd.notnull(x)]
        if len(data) == 0:
            return
        plt.figure()
        plt.hist(data, bins=bins)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename))
        plt.close()

    valid_df = df[df["status"] != "ERROR"].copy()

    if len(valid_df) > 0:
        save_hist(valid_df["duration_ratio"], "Duration Ratio Distribution", "Processed / Original Duration", "hist_duration_ratio.png")
        save_hist(valid_df["peak"], "Peak Distribution", "Peak", "hist_peak.png")
        save_hist(valid_df["rms"], "RMS Distribution", "RMS", "hist_rms.png")
        save_hist(valid_df["silence_ratio"], "Silence Ratio Distribution", "Silence Ratio", "hist_silence_ratio.png")
        save_hist(valid_df["clipping_ratio"], "Clipping Ratio Distribution", "Clipping Ratio", "hist_clipping_ratio.png")
        save_hist(valid_df["mfcc_distance"], "MFCC Distance Distribution", "MFCC Distance", "hist_mfcc_distance.png")
        save_hist(valid_df["onset_corr"], "Onset Correlation Distribution", "Onset Correlation", "hist_onset_corr.png")
        save_hist(valid_df["centroid_diff"], "Spectral Centroid Difference Distribution", "Centroid Difference", "hist_centroid_diff.png")

    print("\n===== FAST VALIDATION SUMMARY =====")
    print(f"Files analyzed: {len(valid_df)}")
    print(f"Errors: {(df['status'] == 'ERROR').sum()}")

    if len(valid_df) > 0:
        print("\n--- Duration Ratio ---")
        print(f"Min:  {valid_df['duration_ratio'].min():.3f}")
        print(f"Max:  {valid_df['duration_ratio'].max():.3f}")
        print(f"Mean: {valid_df['duration_ratio'].mean():.3f}")

        print("\n--- Audio Health ---")
        print(f"Mean Peak:            {valid_df['peak'].mean():.3f}")
        print(f"Mean RMS:             {valid_df['rms'].mean():.3f}")
        print(f"Mean Silence Ratio:   {valid_df['silence_ratio'].mean():.3f}")
        print(f"Mean Clipping Ratio:  {valid_df['clipping_ratio'].mean():.3f}")

        print("\n--- Similarity Metrics ---")
        print(f"Mean MFCC Distance: {valid_df['mfcc_distance'].mean():.3f}")
        print(f"Mean Onset Corr:    {valid_df['onset_corr'].mean():.3f}")
        print(f"Mean Centroid Diff: {valid_df['centroid_diff'].mean():.3f}")

        print("\n--- Status Counts ---")
        print(valid_df["status"].value_counts())

    print(f"\nCSV saved to: {csv_path}")
    print(f"Plots saved to: {output_folder}")