import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

original_folder   = r"path/input"
processed_folder  = r"path/output"

output_folder = os.path.join(processed_folder, "validation_results_eq")
os.makedirs(output_folder, exist_ok=True)

excerpt_duration = 20        

# Thresholds — tune these after a first pass
ONSET_CORR_MIN        = 0.85   # EQ should NOT smear onsets; very high corr expected
SPECTRAL_CENTROID_MAX = 0.30   # relative shift threshold (30 %)
SPECTRAL_DIFF_MIN     = 0.002  # below this → EQ probably had no effect
BAND_RATIO_DELTA_MIN  = 0.05   # minimum energy ratio change in any sub-band


def processed_filename(original_name: str) -> str:
    """
    Mirrors the naming logic in the EQ script:
        base_name  = stem + 'eq'
        short_name = base_name.split('_')[0]
        output     = short_name + '.mp3'
    """
    stem      = os.path.splitext(original_name)[0]
    base_name = stem + "eq"
    short_name = base_name.split("_")[0]
    return short_name + ".mp3"


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
    peak = float(np.max(np.abs(y)))
    rms  = float(np.sqrt(np.mean(y ** 2)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    return {
        "peak"             : peak,
        "rms"              : rms,
        "spectral_flatness": flatness,
        "has_nan"          : bool(np.isnan(y).any()),
    }


def band_energy_ratio(y, sr, bands=None):
    if bands is None:
        bands = [(20, 250), (250, 2000), (2000, 6000), (6000, sr // 2)]

    S = np.abs(librosa.stft(y)) ** 2          # power spectrogram
    freqs = librosa.fft_frequencies(sr=sr)
    total = S.sum() + 1e-12

    ratios = {}
    for (lo, hi) in bands:
        mask    = (freqs >= lo) & (freqs < hi)
        ratios[f"band_{lo}_{hi}"] = float(S[mask].sum() / total)
    return ratios


def compare_eq_features(y1, sr1, y2, sr2):
    if sr1 != sr2:
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
    sr = sr1

    sc1 = float(np.mean(librosa.feature.spectral_centroid(y=y1, sr=sr)))
    sc2 = float(np.mean(librosa.feature.spectral_centroid(y=y2, sr=sr)))
    sc_rel_shift = abs(sc2 - sc1) / (sc1 + 1e-9)   # relative change

    # Mean contrast per sub-band; EQ changes the valleys/peaks differently
    contrast1 = np.mean(librosa.feature.spectral_contrast(y=y1, sr=sr), axis=1)
    contrast2 = np.mean(librosa.feature.spectral_contrast(y=y2, sr=sr), axis=1)
    contrast_dist = float(np.linalg.norm(contrast1 - contrast2))

    mfcc1 = np.mean(librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=13), axis=1)
    mfcc2 = np.mean(librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=13), axis=1)
    mfcc_dist = float(np.linalg.norm(mfcc1 - mfcc2))

    # EQ should NOT shift onsets; low correlation would indicate a problem
    o1  = librosa.onset.onset_strength(y=y1, sr=sr)
    o2  = librosa.onset.onset_strength(y=y2, sr=sr)
    min_l = min(len(o1), len(o2))
    onset_corr = float(np.corrcoef(o1[:min_l], o2[:min_l])[0, 1])

    er1 = band_energy_ratio(y1, sr)
    er2 = band_energy_ratio(y2, sr)
    band_deltas = {f"delta_{k}": abs(er2[k] - er1[k]) for k in er1}
    max_band_delta = float(max(band_deltas.values()))

    S1 = np.abs(librosa.stft(y1))
    S2 = np.abs(librosa.stft(y2))
    min_frames = min(S1.shape[1], S2.shape[1])
    spectral_diff = float(np.mean(np.abs(S1[:, :min_frames] - S2[:, :min_frames])))

    return {
        "sc_orig"         : sc1,
        "sc_proc"         : sc2,
        "sc_rel_shift"    : sc_rel_shift,
        "contrast_dist"   : contrast_dist,
        "mfcc_dist"       : mfcc_dist,
        "onset_corr"      : onset_corr,
        "max_band_delta"  : max_band_delta,
        "spectral_diff"   : spectral_diff,
        **band_deltas,
    }


def classify_eq(row):
    problems = []

    # EQ should change the spectrum; if nothing changed, it was not applied
    if row["spectral_diff"] < SPECTRAL_DIFF_MIN:
        problems.append("no_spectral_change")
    if row["max_band_delta"] < BAND_RATIO_DELTA_MIN:
        problems.append("no_band_energy_change")

    # EQ must NOT smear timing — onset correlation should stay high
    if row["onset_corr"] < ONSET_CORR_MIN:
        problems.append("onset_distortion")

    # Sanity checks
    if row["peak"] < 0.05:
        problems.append("too_quiet")
    if row["has_nan"]:
        problems.append("nan_values")

    # Duration mismatch (EQ is length-preserving)
    dur_diff = abs(row["orig_duration"] - row["proc_duration"])
    if dur_diff > 0.5:
        problems.append(f"duration_mismatch_{dur_diff:.2f}s")

    status = "PASS" if not problems else "REVIEW"
    return status, "; ".join(problems) if problems else "ok"


def process_file(f):
    orig_path      = os.path.join(original_folder, f)
    proc_filename  = processed_filename(f)
    proc_path      = os.path.join(processed_folder, proc_filename)

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
        feats = compare_eq_features(y1_ex, sr1, y2_ex, sr2)

        row = {
            "file"          : f,
            "proc_file"     : proc_filename,
            "orig_duration" : dur1,
            "proc_duration" : dur2,
            **stats,
            **feats,
        }

        status, notes = classify_eq(row)
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
                 total=len(files), desc="Validating EQ")
        )

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_folder, "eq_validation_summary.csv")
    df.to_csv(csv_path, index=False)

    valid_df = df[df["status"] != "ERROR"].copy()

    if len(valid_df) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        ax = axes.ravel()

        configs = [
            ("spectral_diff",  "Spectral Difference\n(should be > 0 if EQ applied)", "steelblue"),
            ("mfcc_dist",      "MFCC Distance\n(timbral change)",                     "salmon"),
            ("sc_rel_shift",   "Spectral Centroid Relative Shift",                    "gold"),
            ("contrast_dist",  "Spectral Contrast Distance",                          "mediumpurple"),
            ("onset_corr",     "Onset Correlation\n(should stay high for EQ)",        "mediumseagreen"),
            ("max_band_delta", "Max Band Energy Ratio Δ\n(any sub-band)",             "coral"),
        ]

        for i, (col, title, color) in enumerate(configs):
            if col in valid_df.columns:
                ax[i].hist(valid_df[col].dropna(), bins=25,
                           color=color, edgecolor="black", alpha=0.85)
                ax[i].set_title(title, fontsize=10)
                ax[i].set_xlabel(col)
                ax[i].set_ylabel("Count")

        plt.suptitle("EQ Transformation Validation Metrics", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "eq_metrics_dist.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()

        band_cols = [c for c in valid_df.columns if c.startswith("delta_band_")]
        if band_cols:
            means = valid_df[band_cols].mean()
            labels = [c.replace("delta_band_", "").replace("_", "–") + " Hz"
                      for c in band_cols]

            plt.figure(figsize=(8, 4))
            plt.bar(labels, means.values, color="steelblue", edgecolor="black")
            plt.title("Mean Band Energy Ratio Δ per Sub-band")
            plt.ylabel("Mean |Δ energy ratio|")
            plt.xlabel("Frequency Band")
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, "eq_band_energy_delta.png"),
                        dpi=150)
            plt.close()

    print("\n===== EQ VALIDATION SUMMARY =====")
    print(df["status"].value_counts().to_string())
    if "notes" in df.columns:
        review_df = df[df["status"] == "REVIEW"]
        if len(review_df):
            print("\nREVIEW flag breakdown:")
            all_notes = "; ".join(review_df["notes"].dropna())
            from collections import Counter
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
