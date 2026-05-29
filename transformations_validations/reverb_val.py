import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import hashlib

original_folder = r"original/path"
processed_folder = r"putput/path"

output_folder = os.path.join(processed_folder, "validation_results_reverb")
os.makedirs(output_folder, exist_ok=True)

excerpt_duration = 20

max_duration_drift = 2.0 # seconds (allowing for the reverb tail)

def safe_filename(filename):
    name_hash = hashlib.sha1(filename.encode()).hexdigest()[:16]
    return f"{name_hash}.mp3"

def load_audio(path):
    y, sr = librosa.load(path, sr=None, mono=False)
    return y, sr

def to_mono(y):
    if y.ndim == 2:
        return np.mean(y, axis=0)
    return y

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
    # Reverb usually increases spectral flatness (makes it more 'noise-like' or smeared)
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    return {
        "peak": peak,
        "rms": rms,
        "spectral_flatness": flatness,
        "has_nan": np.isnan(y).any()
    }

def compare_reverb_features(y1, sr1, y2, sr2):
    if sr1 != sr2:
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
    
    # MFCC Distance 
    mfcc1 = np.mean(librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13), axis=1)
    mfcc2 = np.mean(librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13), axis=1)
    mfcc_dist = np.linalg.norm(mfcc1 - mfcc2)

    # Zero Crossing Rate 
    zcr1 = np.mean(librosa.feature.zero_crossing_rate(y1))
    zcr2 = np.mean(librosa.feature.zero_crossing_rate(y2))
    zcr_ratio = zcr2 / (zcr1 + 1e-9)

    # Onset Strength Correlation
    o1 = librosa.onset.onset_strength(y=y1, sr=sr1)
    o2 = librosa.onset.onset_strength(y=y2, sr=sr1)
    min_l = min(len(o1), len(o2))
    onset_corr = np.corrcoef(o1[:min_l], o2[:min_l])[0, 1]

    return {
        "mfcc_distance": mfcc_dist,
        "zcr_ratio": zcr_ratio,
        "onset_corr": onset_corr
    }

def classify_reverb(row):
    problems = []
    
    #If onset correlation is 1.0, reverb might not have been applied
    if row["onset_corr"] > 0.99:
        problems.append("no_audible_change")
    
    # If correlation is too low, the file might be corrupted or completely different
    if row["onset_corr"] < 0.30:
        problems.append("extreme_distortion_or_mismatch")

    if row["peak"] < 0.05:
        problems.append("too_quiet")
        
    if row["has_nan"]:
        problems.append("nan_values")

    status = "PASS" if len(problems) == 0 else "REVIEW"
    return status, "; ".join(problems) if problems else "ok"

def process_file(f):
    orig_path = os.path.join(original_folder, f)
    proc_filename = safe_filename(f)
    proc_path = os.path.join(processed_folder, proc_filename)

    if not os.path.exists(proc_path):
        return {"file": f, "status": "ERROR", "notes": "processed file missing"}

    try:
        y1, sr1 = load_audio(orig_path)
        y2, sr2 = load_audio(proc_path)

        dur1 = len(y1) / sr1 if y1.ndim == 1 else y1.shape[1] / sr1
        dur2 = len(y2) / sr2 if y2.ndim == 1 else y2.shape[1] / sr2
        
        y1_ex = get_excerpt(y1, sr1, excerpt_duration)
        y2_ex = get_excerpt(y2, sr2, excerpt_duration)

        stats = basic_stats(y2_ex)
        feats = compare_reverb_features(y1_ex, sr1, y2_ex, sr2)

        row = {
            "file": f,
            "hashed_file": proc_filename,
            "orig_duration": dur1,
            "proc_duration": dur2,
            **stats,
            **feats
        }

        status, notes = classify_reverb(row)
        row["status"] = status
        row["notes"] = notes
        return row

    except Exception as e:
        return {"file": f, "status": "ERROR", "notes": str(e)}

if __name__ == "__main__":
    files = [f for f in os.listdir(original_folder) if f.lower().endswith(".mp3")]
    
    print(f"Analyzing {len(files)} files using {multiprocessing.cpu_count()} cores...")

    with ProcessPoolExecutor(max_workers= multiprocessing.cpu_count()-4) as executor:
        results = list(tqdm(executor.map(process_file, files), total=len(files), desc="Validating Reverb"))

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_folder, "reverb_validation_summary.csv")
    df.to_csv(csv_path, index=False)

    # Filtering for plots
    valid_df = df[df["status"] != "ERROR"].copy()

    if len(valid_df) > 0:
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Onset Correlation (How much did the rhythm smear?)
        plt.subplot(2, 2, 1)
        plt.hist(valid_df["onset_corr"], bins=20, color='skyblue', edgecolor='black')
        plt.title("Onset Correlation (Rhythmic Similarity)")
        
        # Plot 2: MFCC Distance (Timbral change)
        plt.subplot(2, 2, 2)
        plt.hist(valid_df["mfcc_distance"], bins=20, color='salmon', edgecolor='black')
        plt.title("MFCC Distance (Timbral Change)")

        # Plot 3: Spectral Flatness
        plt.subplot(2, 2, 3)
        plt.hist(valid_df["spectral_flatness"], bins=20, color='lightgreen', edgecolor='black')
        plt.title("Processed Spectral Flatness")

        # Plot 4: ZCR Ratio
        plt.subplot(2, 2, 4)
        plt.hist(valid_df["zcr_ratio"], bins=20, color='plum', edgecolor='black')
        plt.title("ZCR Ratio (Proc/Orig)")

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "reverb_metrics_dist.png"))
        
    print("\n===== REVERB VALIDATION SUMMARY =====")
    print(df["status"].value_counts())
    print(f"\nCSV Report: {csv_path}")
    print(f"Plots: {output_folder}")