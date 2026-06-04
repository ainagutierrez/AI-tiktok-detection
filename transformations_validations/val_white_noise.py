import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

original_folder  = r"input/path"
processed_folder = r"output/path"

output_folder = os.path.join(processed_folder, "validation_results_whitenoise")
os.makedirs(output_folder, exist_ok=True)

excerpt_duration = 20   # seconds

# Expected SNR range from the noise script (dB)
EXPECTED_SNR_MIN = 10.0
EXPECTED_SNR_MAX = 30.0

# Thresholds — tune after a first pass over distributions
FLATNESS_INCREASE_MIN   = 0.005   # processed must be flatter than original
ONSET_CORR_MIN          = 0.80    # noise shouldn't destroy rhythmic structure
HF_RATIO_INCREASE_MIN   = 0.005   # high-freq energy ratio must increase
SNR_ESTIMATE_MAX        = 35.0    # above this → noise probably not added
SNR_ESTIMATE_MIN        = 5.0     # below this → noise is way too strong

def processed_filename(original_name: str) -> str:
    """Mirrors the noise script: first 40 chars of stem + '.mp3'"""
    stem = os.path.splitext(original_name)[0]
    return stem[:40] + ".mp3"


def load_audio(path):
    y, sr = librosa.load(path, sr=None, mono=False)
    return y, sr


def to_mono(y):
    return np.mean(y, axis=0) if y.ndim == 2 else y


def get_excerpt(y, sr, seconds=20):
    y = to_mono(y)
    target_len = int(seconds * sr)
    if len(y) <= target_len:
        return y
    start = max(0, (len(y) - target_len) // 2)
    return y[start : start + target_len]


def basic_stats(y):
    return {
        "peak"             : float(np.max(np.abs(y))),
        "rms"              : float(np.sqrt(np.mean(y ** 2))),
        "spectral_flatness": float(np.mean(librosa.feature.spectral_flatness(y=y))),
        "has_nan"          : bool(np.isnan(y).any()),
    }


def estimate_snr(y_clean, y_noisy):
    min_len   = min(len(y_clean), len(y_noisy))
    clean     = y_clean[:min_len]
    noisy     = y_noisy[:min_len]
    noise_est = noisy - clean
    sig_pwr   = np.mean(clean ** 2)
    nse_pwr   = np.mean(noise_est ** 2)
    if nse_pwr < 1e-12:
        return 99.0
    return float(10 * np.log10(sig_pwr / (nse_pwr + 1e-12)))


def high_freq_energy_ratio(y, sr, threshold_hz=4000):
    S     = np.abs(librosa.stft(y)) ** 2
    freqs = librosa.fft_frequencies(sr=sr)
    total = S.sum() + 1e-12
    return float(S[freqs >= threshold_hz].sum() / total)


def compare_noise_features(y1, sr1, y2, sr2):
    if sr1 != sr2:
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
    sr = sr1

    flat1 = float(np.mean(librosa.feature.spectral_flatness(y=y1)))
    flat2 = float(np.mean(librosa.feature.spectral_flatness(y=y2)))

    hf1 = high_freq_energy_ratio(y1, sr)
    hf2 = high_freq_energy_ratio(y2, sr)

    def norm(y): return y / (np.max(np.abs(y)) + 1e-9)
    snr_db = estimate_snr(norm(y1), norm(y2))

    o1    = librosa.onset.onset_strength(y=y1, sr=sr)
    o2    = librosa.onset.onset_strength(y=y2, sr=sr)
    min_l = min(len(o1), len(o2))
    onset_corr = float(np.corrcoef(o1[:min_l], o2[:min_l])[0, 1])

    mfcc1     = np.mean(librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=13), axis=1)
    mfcc2     = np.mean(librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=13), axis=1)
    mfcc_dist = float(np.linalg.norm(mfcc1 - mfcc2))

    sc1 = float(np.mean(librosa.feature.spectral_centroid(y=y1, sr=sr)))
    sc2 = float(np.mean(librosa.feature.spectral_centroid(y=y2, sr=sr)))

    return {
        "flat_orig"        : flat1,
        "flat_proc"        : flat2,
        "flatness_increase": flat2 - flat1,
        "hf_ratio_orig"    : hf1,
        "hf_ratio_proc"    : hf2,
        "hf_increase"      : hf2 - hf1,
        "snr_db"           : snr_db,
        "onset_corr"       : onset_corr,
        "mfcc_dist"        : mfcc_dist,
        "sc_shift"         : sc2 - sc1,
    }

def classify_noise(row):
    problems = []

    if row["flatness_increase"] < FLATNESS_INCREASE_MIN:
        problems.append("no_flatness_increase")

    if row["hf_increase"] < HF_RATIO_INCREASE_MIN:
        problems.append("no_hf_energy_increase")

    if row["snr_db"] > SNR_ESTIMATE_MAX:
        problems.append("snr_too_high_noise_not_applied")
    if row["snr_db"] < SNR_ESTIMATE_MIN:
        problems.append("snr_too_low_extreme_noise")

    if row["onset_corr"] < ONSET_CORR_MIN:
        problems.append("onset_distortion")

    dur_diff = abs(row["orig_duration"] - row["proc_duration"])
    if dur_diff > 0.5:
        problems.append(f"duration_mismatch_{dur_diff:.2f}s")

    if row["peak"] < 0.05:
        problems.append("too_quiet")
    if row["has_nan"]:
        problems.append("nan_values")

    status = "PASS" if not problems else "REVIEW"
    return status, "; ".join(problems) if problems else "ok"

def process_file(f):
    orig_path     = os.path.join(original_folder, f)
    proc_filename = processed_filename(f)
    proc_path     = os.path.join(processed_folder, proc_filename)

    if not os.path.exists(proc_path):
        return {"file": f, "proc_file": proc_filename,
                "status": "ERROR", "notes": "processed file missing"}

    try:
        y1, sr1 = load_audio(orig_path)
        y2, sr2 = load_audio(proc_path)

        dur1 = (y1.shape[-1] if y1.ndim == 2 else len(y1)) / sr1
        dur2 = (y2.shape[-1] if y2.ndim == 2 else len(y2)) / sr2

        y1_ex = get_excerpt(y1, sr1, excerpt_duration)
        y2_ex = get_excerpt(y2, sr2, excerpt_duration)

        stats = basic_stats(y2_ex)
        feats = compare_noise_features(y1_ex, sr1, y2_ex, sr2)

        row = {
            "file"          : f,
            "proc_file"     : proc_filename,
            "orig_duration" : dur1,
            "proc_duration" : dur2,
            **stats,
            **feats,
        }

        status, notes = classify_noise(row)
        row["status"] = status
        row["notes"]  = notes
        return row

    except Exception as e:
        return {"file": f, "proc_file": proc_filename,
                "status": "ERROR", "notes": str(e)}

if __name__ == "__main__":
    files = [f for f in os.listdir(original_folder) if f.lower().endswith(".mp3")]
    print(f"Analyzing {len(files)} files using {multiprocessing.cpu_count()} cores...")

    with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count() - 4)) as executor:
        results = list(
            tqdm(executor.map(process_file, files),
                 total=len(files), desc="Validating White Noise")
        )

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_folder, "whitenoise_validation_summary.csv")
    df.to_csv(csv_path, index=False)

    valid_df = df[df["status"] != "ERROR"].copy()

    if len(valid_df) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        ax = axes.ravel()

        configs = [
            ("snr_db",           "Estimated SNR (dB)\n(expected 10–30)",         "steelblue"),
            ("flatness_increase","Spectral Flatness Increase\n(should be > 0)",   "salmon"),
            ("hf_increase",      "HF Energy Ratio Increase\n(should be > 0)",     "gold"),
            ("onset_corr",       "Onset Correlation\n(should stay high)",         "mediumseagreen"),
            ("mfcc_dist",        "MFCC Distance\n(timbral change from noise)",    "mediumpurple"),
            ("sc_shift",         "Spectral Centroid Shift (Hz)",                  "coral"),
        ]

        for i, (col, title, color) in enumerate(configs):
            if col in valid_df.columns:
                data = valid_df[col].dropna()
                ax[i].hist(data, bins=25, color=color, edgecolor="black", alpha=0.85)
                ax[i].set_title(title, fontsize=10)
                ax[i].set_xlabel(col)
                ax[i].set_ylabel("Count")
                if col == "snr_db":
                    ax[i].axvline(EXPECTED_SNR_MIN, color="red",   linestyle="--",
                                  linewidth=1.2, label=f"min {EXPECTED_SNR_MIN} dB")
                    ax[i].axvline(EXPECTED_SNR_MAX, color="green", linestyle="--",
                                  linewidth=1.2, label=f"max {EXPECTED_SNR_MAX} dB")
                    ax[i].legend(fontsize=8)

        plt.suptitle("White Noise Transformation Validation Metrics", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "whitenoise_metrics_dist.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    print("\n===== WHITE NOISE VALIDATION SUMMARY =====")
    print(df["status"].value_counts().to_string())

    review_df = df[df["status"] == "REVIEW"]
    if len(review_df):
        print("\nREVIEW flag breakdown:")
        counts = Counter(
            note.strip()
            for entry in review_df["notes"].dropna()
            for note in entry.split(";")
            if note.strip()
        )
        for note, cnt in counts.most_common():
            print(f"  {note}: {cnt}")

    print(f"\nCSV  → {csv_path}")
    print(f"Plots → {output_folder}")
