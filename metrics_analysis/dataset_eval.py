BASE_DIR = "your/base/dir"

CLASS_DIRS = {
    "tiktok_real": r"tiktokreal/dir",
    "tiktok_fake": r"titktokfake/dir",
}

OUTPUT_DIR = "figures"
SR         = 22050
N_MFCC     = 13
DURATION   = 30        # seconds per file (None = full file)
SEED       = 42

import os, warnings, random, base64, json
from io import BytesIO
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import librosa
import librosa.display

from scipy.spatial.distance import cosine
from scipy.special import kl_div
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from tqdm import tqdm

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[INFO] umap-learn not installed – only t-SNE will run")

random.seed(SEED); np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAL = {
    "real":        "#4C72B0",
    "fake":        "#C44E52",
    "tiktok_real": "#55A868",
    "tiktok_fake": "#DD8452",
}

def col(label, idx=0):
    return PAL.get(label, list(plt.cm.tab10.colors)[idx % 10])

def list_audio(folder):
    exts = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
    return sorted(os.path.join(folder, f)
                  for f in os.listdir(folder) if f.lower().endswith(exts))


def load_audio(path):
    try:
        y, sr = librosa.load(path, sr=SR, duration=DURATION, mono=True)
        return y, sr
    except Exception as e:
        print(f"  [WARN] {path}: {e}")
        return None, SR


def extract(y, sr):
    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    zcr       = librosa.feature.zero_crossing_rate(y)[0]
    rms       = librosa.feature.rms(y=y)[0]
    # fundamental frequency (pitch) via piptrack
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[mags > mags.mean()]
    pitch_vals = pitch_vals[pitch_vals > 50]   # remove sub-bass noise

    return {
        # embedding vectors
        "mfcc_mean":      mfcc.mean(axis=1),
        "mfcc_var":       mfcc.var(axis=1),
        "mfcc_flat":      np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]),
        # scalar summaries
        "centroid_mean":  float(centroid.mean()),
        "centroid_std":   float(centroid.std()),
        "bandwidth_mean": float(bandwidth.mean()),
        "rolloff_mean":   float(rolloff.mean()),
        "zcr_mean":       float(zcr.mean()),
        "rms_mean":       float(rms.mean()),
        "pitch_mean":     float(pitch_vals.mean()) if len(pitch_vals) else 0.0,
        "pitch_std":      float(pitch_vals.std())  if len(pitch_vals) else 0.0,
        "duration":       float(librosa.get_duration(y=y, sr=sr)),
        "sample_rate":    int(sr),
        # raw waveform for spectrogram
        "_y":             y,
    }


def load_all():
    datasets = {}
    for label, subfolder in CLASS_DIRS.items():
        folder = subfolder
        if not os.path.isdir(folder):
            print(f"[WARN] {folder} not found – skipping '{label}'")
            continue
        print(f"\n── [{label}] ─────────────────────────────────────────────────")
        recs = []
        for path in tqdm(list_audio(folder), desc=f"  {label}"):
            y, sr = load_audio(path)
            if y is None:
                continue
            feat = extract(y, sr)
            feat["path"]  = path
            feat["label"] = label
            feat["name"]  = os.path.splitext(os.path.basename(path))[0]
            recs.append(feat)
        datasets[label] = recs
    return datasets


_b64 = {}   # fname → base64 PNG

def savefig(name):
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=150, bbox_inches="tight")
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    _b64[name] = base64.b64encode(buf.read()).decode()
    plt.close()
    print(f"  ✔  {name}")


def plot_class_balance(datasets):
    """Bar chart: how many files per class."""
    print("\n[01] Class balance")
    labels = list(datasets.keys())
    counts = [len(datasets[l]) for l in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts,
                  color=[col(l, i) for i, l in enumerate(labels)],
                  edgecolor="white", linewidth=0.6)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(cnt), ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of files")
    ax.set_title("Class Balance")
    ax.grid(axis="y", alpha=0.3)
    savefig("01_class_balance.png")

def plot_duration(datasets):
    """Histogram/KDE of file durations per class."""
    print("[02] Duration distribution")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # KDE
    for i, (lbl, recs) in enumerate(datasets.items()):
        durations = [r["duration"] for r in recs]
        sns.kdeplot(durations, ax=axes[0], label=lbl, color=col(lbl, i),
                    fill=True, alpha=0.22)
    axes[0].set_xlabel("Duration (s)")
    axes[0].set_title("Duration KDE per class")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Boxplot
    data   = [[r["duration"] for r in datasets[l]] for l in datasets]
    labels = list(datasets.keys())
    bp = axes[1].boxplot(data, patch_artist=True, labels=labels)
    for patch, lbl in zip(bp["boxes"], labels):
        patch.set_facecolor(col(lbl)); patch.set_alpha(0.7)
    axes[1].set_ylabel("Duration (s)")
    axes[1].set_title("Duration boxplot per class")
    axes[1].grid(axis="y", alpha=0.3)

    savefig("02_duration.png")


def plot_sample_rate(datasets):
    """Bar chart showing the distribution of sample rates — should be uniform."""
    print("[03] Sample rate consistency")
    from collections import Counter
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]
    for ax, (lbl, recs) in zip(axes, datasets.items()):
        counts = Counter(r["sample_rate"] for r in recs)
        ax.bar([str(k) for k in counts.keys()], counts.values(),
               color=col(lbl), edgecolor="white")
        ax.set_title(f"[{lbl}]")
        ax.set_xlabel("Sample rate (Hz)"); ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Sample Rate Consistency per Class", y=1.02, fontsize=12)
    savefig("03_sample_rate.png")


def plot_spectral_centroid(datasets):
    """KDE of spectral centroid — brightness of the audio."""
    print("[04] Spectral centroid")
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (lbl, recs) in enumerate(datasets.items()):
        vals = [r["centroid_mean"] for r in recs]
        sns.kdeplot(vals, ax=ax, label=lbl, color=col(lbl, i), fill=True, alpha=0.22)
    ax.set_xlabel("Spectral centroid (Hz)")
    ax.set_title("Spectral Centroid Distribution — real vs fake separation")
    ax.legend(); ax.grid(alpha=0.3)
    savefig("04_spectral_centroid.png")


def plot_rms(datasets):
    """KDE + boxplot of RMS energy — loudness characteristics."""
    print("[05] RMS energy")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for i, (lbl, recs) in enumerate(datasets.items()):
        vals = [r["rms_mean"] for r in recs]
        sns.kdeplot(vals, ax=axes[0], label=lbl, color=col(lbl, i), fill=True, alpha=0.22)
    axes[0].set_xlabel("Mean RMS energy")
    axes[0].set_title("RMS Energy KDE")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    data   = [[r["rms_mean"] for r in datasets[l]] for l in datasets]
    labels = list(datasets.keys())
    bp = axes[1].boxplot(data, patch_artist=True, labels=labels)
    for patch, lbl in zip(bp["boxes"], labels):
        patch.set_facecolor(col(lbl)); patch.set_alpha(0.7)
    axes[1].set_ylabel("Mean RMS"); axes[1].set_title("RMS Energy Boxplot")
    axes[1].grid(axis="y", alpha=0.3)
    savefig("05_rms.png")


def plot_zcr(datasets):
    """ZCR — related to noisiness and HF content. Differs between real/fake speech."""
    print("[06] Zero-crossing rate")
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (lbl, recs) in enumerate(datasets.items()):
        vals = [r["zcr_mean"] for r in recs]
        sns.kdeplot(vals, ax=ax, label=lbl, color=col(lbl, i), fill=True, alpha=0.22)
    ax.set_xlabel("Mean ZCR")
    ax.set_title("Zero-Crossing Rate Distribution — noise / HF content")
    ax.legend(); ax.grid(alpha=0.3)
    savefig("06_zcr.png")


def plot_mfcc_means(datasets):
    """Line plot of mean MFCC per class ± 1 std. Core acoustic fingerprint."""
    print("[07] MFCC mean vectors")
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (lbl, recs) in enumerate(datasets.items()):
        M = np.array([r["mfcc_mean"] for r in recs])
        avg, std = M.mean(0), M.std(0)
        x = np.arange(1, N_MFCC + 1)
        ax.plot(x, avg, marker="o", label=lbl, color=col(lbl, i))
        ax.fill_between(x, avg - std, avg + std, alpha=0.18, color=col(lbl, i))
    ax.set_xlabel("MFCC index"); ax.set_ylabel("Coefficient value")
    ax.set_title("MFCC Mean Vectors ± 1 std — class acoustic fingerprints")
    ax.legend(); ax.grid(alpha=0.3)
    savefig("07_mfcc_means.png")


def plot_mfcc_variance(datasets):
    """Per-coefficient variance across files — shows how diverse each class is."""
    print("[08] MFCC variance per coefficient")
    all_labels = list(datasets.keys())
    n_coef     = N_MFCC
    fig, axes  = plt.subplots(1, n_coef, figsize=(n_coef * 2.2, 4))
    for ax, coef_idx in zip(axes, range(n_coef)):
        data   = [np.array([r["mfcc_var"][coef_idx] for r in datasets[l]]) for l in all_labels]
        colors = [col(l, i) for i, l in enumerate(all_labels)]
        bp = ax.boxplot(data, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.set_title(f"c{coef_idx+1}", fontsize=8)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.3)
    # shared legend
    from matplotlib.patches import Patch
    handles = [Patch(color=col(l, i), label=l) for i, l in enumerate(all_labels)]
    fig.legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle("MFCC Variance per Coefficient (intra-class diversity)", fontsize=11, y=1.02)
    savefig("08_mfcc_variance.png")


def plot_mfcc_distance(datasets):
    """Pairwise cosine distance heatmap between per-file MFCC mean vectors."""
    print("[09] MFCC cosine distance matrix")
    all_recs = [r for recs in datasets.values() for r in recs]
    V = np.array([r["mfcc_mean"] for r in all_recs])
    n = len(V)
    D = np.array([[cosine(V[i], V[j]) for j in range(n)] for i in range(n)])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(D, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="cosine distance")
    counts = [len(datasets[l]) for l in datasets]
    bounds = np.cumsum(counts)[:-1]
    for b in bounds:
        ax.axhline(b - 0.5, color="white", lw=1)
        ax.axvline(b - 0.5, color="white", lw=1)
    ticks = [0] + list(bounds) + [n]
    mid   = [(ticks[i] + ticks[i+1]) // 2 for i in range(len(ticks) - 1)]
    ax.set_xticks(mid); ax.set_xticklabels(list(datasets.keys()), rotation=20)
    ax.set_yticks(mid); ax.set_yticklabels(list(datasets.keys()))
    ax.set_title("Pairwise MFCC Cosine Distance — block structure = class cohesion")
    savefig("09_mfcc_distance_matrix.png")


def plot_embedding(datasets):
    """2-D projection of MFCC embeddings — visual class separability."""
    print("[10] Embedding (t-SNE / UMAP)")
    all_recs   = [r for recs in datasets.values() for r in recs]
    labels_arr = np.array([r["label"] for r in all_recs])
    X = StandardScaler().fit_transform(np.array([r["mfcc_flat"] for r in all_recs]))

    n_plots = 2 if HAS_UMAP else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    print("  t-SNE …")
    tsne = TSNE(n_components=2, random_state=SEED,
                perplexity=min(30, len(X) - 1)).fit_transform(X)
    _scatter(axes[0], tsne, labels_arr, datasets, "t-SNE of MFCC embeddings")

    if HAS_UMAP:
        print("  UMAP …")
        emb = umap_lib.UMAP(n_components=2, random_state=SEED).fit_transform(X)
        _scatter(axes[1], emb, labels_arr, datasets, "UMAP of MFCC embeddings")

    savefig("10_embedding.png")


def _scatter(ax, emb, labels, datasets, title):
    for i, lbl in enumerate(datasets.keys()):
        mask = labels == lbl
        ax.scatter(emb[mask, 0], emb[mask, 1], c=col(lbl, i),
                   label=lbl, s=30, alpha=0.75, edgecolors="none")
    ax.set_title(title); ax.legend(markerscale=1.5)
    ax.set_xticks([]); ax.set_yticks([])


def compute_cluster_metrics(datasets):
    """Returns (sil, db) and prints them."""
    print("[11-12] Cluster quality metrics")
    all_recs = [r for recs in datasets.values() for r in recs]
    X = StandardScaler().fit_transform(np.array([r["mfcc_flat"] for r in all_recs]))
    unique = list(dict.fromkeys(r["label"] for r in all_recs))
    y = np.array([unique.index(r["label"]) for r in all_recs])
    if len(unique) < 2:
        print("  [SKIP] Need ≥ 2 classes.")
        return None, None
    sil = silhouette_score(X, y)
    db  = davies_bouldin_score(X, y)
    print(f"  Silhouette score    : {sil:.4f}  (↑ better, max=1)")
    print(f"  Davies-Bouldin index: {db:.4f}  (↓ better)")

    # Bar chart visualisation
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(["Silhouette"], [sil], color="#4C72B0", width=0.4)
    axes[0].axhline(0, color="black", lw=0.8, ls="--")
    axes[0].set_ylim(-1, 1); axes[0].set_ylabel("Score")
    axes[0].set_title("Silhouette score\n(↑ better, max=1)")
    axes[0].grid(axis="y", alpha=0.3)
    for sp in ["top","right"]: axes[0].spines[sp].set_visible(False)

    axes[1].bar(["Davies-Bouldin"], [db], color="#C44E52", width=0.4)
    axes[1].axhline(1, color="gray", lw=0.8, ls="--", label="typical threshold=1")
    axes[1].set_ylabel("Index"); axes[1].set_title("Davies-Bouldin index\n(↓ better)")
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.3)
    for sp in ["top","right"]: axes[1].spines[sp].set_visible(False)

    fig.suptitle("Cluster Quality in MFCC Embedding Space", fontsize=12)
    savefig("11_cluster_metrics.png")
    return sil, db


def plot_js_divergence(datasets):
    """JS divergence between all class pairs for 4 features."""
    print("[13] Jensen-Shannon divergence")
    feats  = ["centroid_mean", "rms_mean", "zcr_mean", "bandwidth_mean"]
    labels = list(datasets.keys())
    n = len(labels)

    def hp(vals, bins=50):
        h, _ = np.histogram(vals, bins=bins, density=True)
        h = h + 1e-10; return h / h.sum()

    fig, axes = plt.subplots(1, len(feats), figsize=(5 * len(feats), 4))
    for ax, feat in zip(axes, feats):
        mat = np.zeros((n, n))
        for i, li in enumerate(labels):
            pi = hp([r[feat] for r in datasets[li]])
            for j, lj in enumerate(labels):
                pj = hp([r[feat] for r in datasets[lj]])
                m  = 0.5 * (pi + pj)
                mat[i, j] = (0.5 * np.sum(kl_div(pi, m))
                            + 0.5 * np.sum(kl_div(pj, m)))
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, aspect="auto")
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=20)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels)
        plt.colorbar(im, ax=ax)
        ax.set_title(feat.replace("_mean", ""))
    fig.suptitle("Jensen-Shannon Divergence Between Class Distributions", fontsize=12)
    savefig("12_js_divergence.png")


def plot_feature_boxplots(datasets):
    """Side-by-side boxplot of all spectral/temporal features per class."""
    print("[14] Feature boxplots")
    feats  = ["centroid_mean", "bandwidth_mean", "rolloff_mean",
              "zcr_mean", "rms_mean", "pitch_mean"]
    labels = list(datasets.keys())

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, feat in zip(axes.flatten(), feats):
        data   = [np.array([r[feat] for r in datasets[l]]) for l in labels]
        colors = [col(l, i) for i, l in enumerate(labels)]
        bp = ax.boxplot(data, patch_artist=True, labels=labels)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.set_title(feat.replace("_mean", "").replace("_", " "))
        ax.tick_params(axis="x", rotation=18); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Spectral & Temporal Feature Distributions per Class",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    savefig("13_feature_boxplots.png")


def plot_spectrogram_grid(datasets, n_per_class=3):
    """Log-power spectrogram grid: n_per_class columns × 4 rows."""
    print("[15] Spectrogram sample grid")
    labels = list(datasets.keys())
    fig, axes = plt.subplots(len(labels), n_per_class,
                             figsize=(n_per_class * 4, len(labels) * 3))
    if len(labels) == 1:
        axes = [axes]

    for row, lbl in enumerate(labels):
        recs   = datasets[lbl]
        sample = recs[:n_per_class]
        for col_idx, rec in enumerate(sample):
            ax   = axes[row][col_idx]
            y    = rec["_y"]
            S    = librosa.power_to_db(
                       librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64),
                       ref=np.max)
            librosa.display.specshow(S, sr=SR, ax=ax,
                                     x_axis=None, y_axis=None,
                                     cmap="magma")
            ax.set_xticks([]); ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(lbl, fontsize=11, fontweight="bold",
                              color=col(lbl))
            if row == 0:
                ax.set_title(f"sample {col_idx+1}", fontsize=9)

    fig.suptitle("Mel Spectrogram Samples per Class", fontsize=13, y=1.01)
    plt.tight_layout()
    savefig("14_spectrograms.png")


def plot_pitch(datasets):
    """Histogram of estimated fundamental frequency per class."""
    print("[16] Pitch (F0) distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (lbl, recs) in enumerate(datasets.items()):
        vals = [r["pitch_mean"] for r in recs if r["pitch_mean"] > 0]
        if not vals:
            continue
        sns.kdeplot(vals, ax=ax, label=lbl, color=col(lbl, i), fill=True, alpha=0.22)
    ax.set_xlabel("Mean F0 (Hz)")
    ax.set_title("Pitch (Fundamental Frequency) Distribution per Class")
    ax.legend(); ax.grid(alpha=0.3)
    savefig("15_pitch.png")


def plot_intra_diversity(datasets):
    """
    How diverse is each class internally?
    Coefficient of variation (std/mean) for each feature per class.
    High CV = high diversity = good for training.
    """
    print("[17] Intra-class diversity")
    feats  = ["centroid_mean", "bandwidth_mean", "zcr_mean",
              "rms_mean", "pitch_mean", "duration"]
    labels = list(datasets.keys())

    cv_table = {}   # label → [cv per feature]
    for lbl, recs in datasets.items():
        cvs = []
        for f in feats:
            vals = np.array([r[f] for r in recs])
            mu   = vals.mean()
            cv   = (vals.std() / mu) if mu != 0 else 0.0
            cvs.append(cv)
        cv_table[lbl] = cvs

    x = np.arange(len(feats))
    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, lbl in enumerate(labels):
        offset = (i - len(labels) / 2 + 0.5) * width
        ax.bar(x + offset, cv_table[lbl], width=width * 0.9,
               label=lbl, color=col(lbl, i), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_mean", "") for f in feats], rotation=15)
    ax.set_ylabel("Coefficient of variation (std / mean)")
    ax.set_title("Intra-class Feature Diversity — higher = more varied samples")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    savefig("16_intra_diversity.png")


def plot_feature_correlation(datasets):
    """
    Pearson correlation matrix across ALL features for each class.
    Reveals redundancy between features and whether the feature set is informative.
    """
    print("[18] Feature correlation heatmap")
    feats = ["centroid_mean", "bandwidth_mean", "rolloff_mean",
             "zcr_mean", "rms_mean", "pitch_mean", "duration"]
    labels = list(datasets.keys())

    fig, axes = plt.subplots(1, len(labels),
                             figsize=(5 * len(labels), 4.5))
    if len(labels) == 1:
        axes = [axes]

    short = [f.replace("_mean", "").replace("_", "\n") for f in feats]
    for ax, lbl in zip(axes, labels):
        recs = datasets[lbl]
        mat  = np.array([[r[f] for f in feats] for r in recs])
        corr = np.corrcoef(mat.T)
        im   = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(feats))); ax.set_xticklabels(short, fontsize=8)
        ax.set_yticks(range(len(feats))); ax.set_yticklabels(short, fontsize=8)
        ax.set_title(lbl); plt.colorbar(im, ax=ax, shrink=0.7)
        # annotate cells
        for ri in range(len(feats)):
            for ci in range(len(feats)):
                ax.text(ci, ri, f"{corr[ri,ci]:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if abs(corr[ri, ci]) > 0.5 else "black")

    fig.suptitle("Feature Correlation Matrix per Class", fontsize=12, y=1.01)
    plt.tight_layout()
    savefig("17_feature_correlation.png")


def print_summary(datasets):
    feats = ["centroid_mean", "bandwidth_mean", "zcr_mean", "rms_mean",
             "pitch_mean", "duration"]
    print("\n" + "═" * 100)
    print("  DATASET SUMMARY STATISTICS")
    print("═" * 100)
    header = f"{'Class':<16}  {'N':>4}"
    for f in feats:
        header += f"  {f[:13]:>15}"
    print(header)
    print("─" * 100)
    rows = []
    for lbl, recs in datasets.items():
        row = f"{lbl:<16}  {len(recs):>4}"
        r_dict = {"label": lbl, "n": len(recs)}
        for f in feats:
            vals = [r[f] for r in recs]
            row += f"  {np.mean(vals):>6.2f}±{np.std(vals):.2f}"
            r_dict[f] = {"mean": round(float(np.mean(vals)), 4),
                         "std":  round(float(np.std(vals)),  4)}
        print(row)
        rows.append(r_dict)
    print("═" * 100)
    return rows


PLOT_META = [
    ("01_class_balance.png",       "Class balance",
     "File count per class. Confirms dataset balance or reveals class imbalance that may bias a classifier."),
    ("02_duration.png",            "Duration distribution",
     "KDE and boxplot of audio clip durations. Uniform duration is ideal; high variance may indicate data quality issues."),
    ("03_sample_rate.png",         "Sample rate consistency",
     "All files should share the same sample rate. Mixed rates indicate preprocessing problems."),
    ("04_spectral_centroid.png",   "Spectral centroid",
     "Brightness of the audio signal. Fake speech/music synthesised by different models often sits at a different spectral centre of mass."),
    ("05_rms.png",                 "RMS energy",
     "Loudness profile per class. Consistent energy ranges confirm normalisation was applied; divergence may reveal class-specific artefacts."),
    ("06_zcr.png",                 "Zero-crossing rate",
     "Proxy for noisiness and high-frequency content. AI-generated audio often has a characteristically different ZCR from recorded audio."),
    ("07_mfcc_means.png",          "MFCC mean vectors",
     "The acoustic 'fingerprint' of each class. Divergence between real and fake classes validates the discrimination potential of MFCC features."),
    ("08_mfcc_variance.png",       "MFCC variance per coefficient",
     "Intra-class variance per MFCC coefficient. High variance = diverse samples within the class, which supports generalisation."),
    ("09_mfcc_distance_matrix.png","MFCC cosine distance matrix",
     "Pairwise distances between every file. A block-diagonal structure proves intra-class cohesion and inter-class separation."),
    ("10_embedding.png",           "t-SNE / UMAP embedding",
     "Non-linear 2-D projection of MFCC embeddings. Visually confirms whether the four classes are acoustically distinct."),
    ("11_cluster_metrics.png",     "Cluster quality metrics",
     "Silhouette score and Davies-Bouldin index quantify class separability in embedding space — key thesis-level evidence."),
    ("12_js_divergence.png",       "Jensen-Shannon divergence",
     "Symmetric measure of distribution shift between all class pairs. High JS divergence = clear domain boundaries."),
    ("13_feature_boxplots.png",    "Feature boxplots",
     "Side-by-side comparison of six acoustic features across all classes. The most intuitive class-comparison plot for a thesis chapter."),
    ("14_spectrograms.png",        "Spectrogram samples",
     "Visual inspection of Mel spectrograms per class. Qualitative evidence of structural differences (e.g. synthesis artefacts in fake audio)."),
    ("15_pitch.png",               "Pitch (F0) distribution",
     "Fundamental frequency distribution. AI vocoders and TTS systems often produce different pitch statistics compared to natural speech."),
    ("16_intra_diversity.png",     "Intra-class diversity",
     "Coefficient of variation (std/mean) per feature per class. High CV confirms the dataset is not artificially uniform within classes."),
    ("17_feature_correlation.png", "Feature correlation heatmap",
     "Pearson correlation between all features within each class. Reveals redundant features and validates that the feature set is informative."),
]

FIG_CATS = {
    "integrity":  ["01_", "02_", "03_"],
    "spectral":   ["04_", "05_", "06_"],
    "mfcc":       ["07_", "08_", "09_"],
    "embedding":  ["10_", "11_"],
    "separation": ["12_", "13_", "14_"],
    "diversity":  ["15_", "16_", "17_"],
}


def build_dashboard(summary_rows, sil, db):
    print("\n── Building HTML dashboard ──────────────────────────────────")

    metric_html = ""
    for s in summary_rows:
        c = PAL.get(s["label"], "#888")
        metric_html += (
            '<div class="mc" style="border-top:3px solid ' + c + '">'
            '<div class="mc-label">' + s["label"] + '</div>'
            '<div class="mc-val">' + str(s["n"]) + '</div>'
            '<div class="mc-note">files</div>'
            '<div class="mc-sub">'
            "centroid " + str(round(s["centroid_mean"]["mean"])) + " Hz &nbsp;&middot;&nbsp; "
            "rms " + str(round(s["rms_mean"]["mean"], 4)) + " &nbsp;&middot;&nbsp; "
            "zcr " + str(round(s["zcr_mean"]["mean"], 4)) +
            "</div></div>"
        )

    gauge_html = ""
    if sil is not None:
        sp = max(0, min(100, int(sil * 100)))
        dp = max(0, int((1 - min(db, 3.0) / 3.0) * 100))
        gauge_html = (
            '<div class="gauge-row">'
            '<div class="gauge-card">'
            '<div class="g-title">Silhouette score <span class="hint">(&uarr; better &middot; max=1)</span></div>'
            '<div class="g-track"><div class="g-fill gn" style="width:' + str(sp) + '%"></div></div>'
            '<div class="g-val">' + f"{sil:.4f}" + '</div>'
            '</div>'
            '<div class="gauge-card">'
            '<div class="g-title">Davies-Bouldin index <span class="hint">(&darr; better)</span></div>'
            '<div class="g-track"><div class="g-fill go" style="width:' + str(dp) + '%"></div></div>'
            '<div class="g-val">' + f"{db:.4f}" + '</div>'
            '</div></div>'
        )

    fig_html = ""
    for fname, title, caption in PLOT_META:
        if fname not in _b64:
            continue
        cats = " ".join(k for k, prefixes in FIG_CATS.items()
                        if any(fname.startswith(p) for p in prefixes))
        fig_html += (
            '<div class="fig-card" data-cat="' + cats + '" onclick="openLb(\'' + fname + '\')">'
            '<img src="data:image/png;base64,' + _b64[fname] + '" loading="lazy" alt="' + title + '"/>'
            '<div class="fig-title">' + title + '</div>'
            '<div class="fig-cap">' + caption + '</div>'
            '</div>'
        )

    feats = ["centroid_mean", "bandwidth_mean", "zcr_mean",
             "rms_mean", "pitch_mean", "duration"]
    trows = ""
    for s in summary_rows:
        c = PAL.get(s["label"], "#888")
        cells = "".join(
            "<td>" + str(round(s[f]["mean"], 3)) + " &plusmn; " + str(round(s[f]["std"], 3)) + "</td>"
            for f in feats
        )
        trows += (
            "<tr>"
            '<td><span class="pill" style="background:' + c + '22;color:' + c + ';border:1px solid ' + c + '55">'
            + s["label"] + "</span></td>"
            "<td>" + str(s["n"]) + "</td>"
            + cells +
            "</tr>"
        )

    featured_html = ""
    for f, t, cap in PLOT_META[:6]:
        if f not in _b64:
            continue
        featured_html += (
            '<div class="fig-card" onclick="openLb(\'' + f + '\')">'
            '<img src="data:image/png;base64,' + _b64[f] + '" loading="lazy"/>'
            '<div class="fig-title">' + t + '</div>'
            '<div class="fig-cap">' + cap + '</div>'
            '</div>'
        )

    plots_json = json.dumps([{"fname": f, "title": t}
                              for f, t, _ in PLOT_META if f in _b64])
    imgs_json  = json.dumps({f: b for f, b in _b64.items()})

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Dataset Validation &mdash; Deepfake Audio Detection</title>
<style>
:root{
  --bg:#0d1117;--surf:#161b22;--card:#1c2130;
  --acc:#4C72B0;--grn:#55A868;--org:#DD8452;
  --txt:#e6edf3;--mut:#8b949e;--bdr:#30363d;
  --f:'Segoe UI',system-ui,sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:var(--f);min-height:100vh}
header{background:linear-gradient(135deg,#091525 0%,#122040 55%,#091525 100%);
       padding:52px 48px 40px;border-bottom:1px solid var(--bdr)}
header h1{font-size:2.1rem;font-weight:700;letter-spacing:-.5px}
header h1 em{font-style:normal;color:var(--acc)}
header p{color:var(--mut);margin-top:10px;font-size:.93rem;max-width:680px;line-height:1.65}
.tags{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}
.tag{background:rgba(255,255,255,.04);border:1px solid var(--bdr);
     padding:4px 12px;border-radius:999px;font-size:.77rem;color:var(--mut)}
.tag.hi{border-color:var(--acc);color:var(--acc)}
nav{background:var(--surf);border-bottom:1px solid var(--bdr);
    display:flex;padding:0 48px;gap:0;overflow-x:auto}
nav button{background:none;border:none;border-bottom:2px solid transparent;
           color:var(--mut);padding:14px 20px;cursor:pointer;
           font-size:.87rem;white-space:nowrap;transition:all .18s}
nav button:hover{color:var(--txt)}
nav button.on{color:var(--txt);border-color:var(--acc)}
.sec{display:none;padding:44px 48px;max-width:1640px;margin:0 auto}
.sec.on{display:block}
h2.sh{font-size:.9rem;text-transform:uppercase;letter-spacing:.08em;
      color:var(--mut);margin-bottom:22px;font-weight:600}
.mgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:16px;margin-bottom:30px}
.mc{background:var(--card);border:1px solid var(--bdr);border-radius:10px;padding:18px 20px}
.mc-label{font-size:.74rem;text-transform:uppercase;letter-spacing:.07em;color:var(--mut)}
.mc-val{font-size:2.1rem;font-weight:700;margin:5px 0 0}
.mc-note{font-size:.78rem;color:var(--mut);margin-bottom:9px}
.mc-sub{font-size:.73rem;color:var(--mut);border-top:1px solid var(--bdr);padding-top:8px;line-height:1.75}
.gauge-row{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:30px}
.gauge-card{background:var(--card);border:1px solid var(--bdr);border-radius:10px;
            padding:20px;flex:1;min-width:250px}
.g-title{font-size:.83rem;color:var(--mut);margin-bottom:12px}
.g-title .hint{font-size:.73rem}
.g-track{height:7px;background:var(--bdr);border-radius:4px;overflow:hidden;margin-bottom:8px}
.g-fill{height:100%;border-radius:4px;transition:width 1s ease}
.g-fill.gn{background:var(--grn)}
.g-fill.go{background:var(--org)}
.g-val{font-size:1.55rem;font-weight:700}
.fbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px}
.fb{background:var(--card);border:1px solid var(--bdr);color:var(--mut);
    padding:5px 15px;border-radius:999px;cursor:pointer;font-size:.81rem;transition:all .15s}
.fb.on{background:var(--acc);border-color:var(--acc);color:#fff}
.fgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:18px}
.fig-card{background:var(--card);border:1px solid var(--bdr);border-radius:10px;
          overflow:hidden;cursor:zoom-in;transition:transform .15s,border-color .15s}
.fig-card:hover{transform:translateY(-3px);border-color:var(--acc)}
.fig-card img{width:100%;display:block;border-bottom:1px solid var(--bdr)}
.fig-title{font-size:.87rem;font-weight:600;padding:11px 13px 3px}
.fig-cap{font-size:.75rem;color:var(--mut);padding:0 13px 13px;line-height:1.55}
#lb{display:none;position:fixed;inset:0;background:rgba(0,0,0,.9);
    z-index:9999;align-items:center;justify-content:center;flex-direction:column}
#lb.on{display:flex}
#lb img{max-width:90vw;max-height:78vh;border-radius:8px}
#lb .lbt{color:#fff;margin-top:14px;font-size:.93rem;font-weight:600;text-align:center}
#lb .lbn{display:flex;gap:14px;margin-top:12px}
#lb .lbn button{background:var(--card);border:1px solid var(--bdr);color:var(--txt);
                padding:7px 20px;border-radius:6px;cursor:pointer;font-size:.87rem}
#lb .lbn button:hover{background:var(--acc);border-color:var(--acc)}
#lb .lbx{position:absolute;top:16px;right:22px;background:none;border:none;
          color:#fff;font-size:2rem;cursor:pointer;line-height:1}
.tbl-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.83rem}
th{background:var(--card);color:var(--mut);text-align:left;
   padding:10px 13px;border-bottom:1px solid var(--bdr);font-weight:600}
td{padding:10px 13px;border-bottom:1px solid var(--bdr)}
tr:hover td{background:var(--card)}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:.75rem;font-weight:600}
.explain{background:var(--card);border:1px solid var(--bdr);border-radius:10px;
         padding:22px;max-width:740px;margin-top:6px;font-size:.87rem;
         color:var(--mut);line-height:1.75}
.explain strong{color:var(--txt)}
@media(max-width:640px){header,.sec{padding:22px 14px} nav{padding:0 14px} .fgrid{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>Deepfake Audio &mdash; <em>Dataset Validation</em></h1>
  <p>18 analyses characterising the dataset on its own terms: class balance,
     spectral diversity, acoustic separability, intra-class variety,
     and cluster quality.</p>
  <div class="tags">
    <span class="tag hi">real &middot; fake &middot; tiktok_real &middot; tiktok_fake</span>
    <span class="tag">MFCC &middot; Centroid &middot; RMS &middot; ZCR &middot; Pitch</span>
    <span class="tag">t-SNE &middot; UMAP &middot; JS Divergence &middot; Silhouette</span>
    <span class="tag">Spectrogram grid &middot; Intra-class diversity</span>
  </div>
</header>
<nav>
  <button class="on" onclick="tab('overview',this)">Overview</button>
  <button onclick="tab('figures',this)">All Figures</button>
  <button onclick="tab('cluster',this)">Cluster Metrics</button>
  <button onclick="tab('stats',this)">Statistics</button>
</nav>

<div class="sec on" id="overview">
  <h2 class="sh">Dataset at a glance</h2>
  <div class="mgrid">""" + metric_html + """</div>
  """ + gauge_html + """
  <h2 class="sh">Featured plots</h2>
  <div class="fgrid">""" + featured_html + """</div>
</div>

<div class="sec" id="figures">
  <div class="fbar">
    <button class="fb on" onclick="filt('all',this)">All</button>
    <button class="fb" onclick="filt('integrity',this)">Integrity</button>
    <button class="fb" onclick="filt('spectral',this)">Spectral</button>
    <button class="fb" onclick="filt('mfcc',this)">MFCC</button>
    <button class="fb" onclick="filt('embedding',this)">Embedding</button>
    <button class="fb" onclick="filt('separation',this)">Separation</button>
    <button class="fb" onclick="filt('diversity',this)">Diversity</button>
  </div>
  <div class="fgrid" id="fgrid">""" + fig_html + """</div>
</div>

<div class="sec" id="cluster">
  <h2 class="sh">Cluster quality in MFCC space</h2>
  """ + (gauge_html if gauge_html else '<p style="color:var(--mut)">Not enough samples.</p>') + """
  <div class="explain">
    <strong>Silhouette score</strong> measures how similar each sample is to its own class
    compared to other classes. A score near +1 proves that the four classes occupy clearly
    separated regions in MFCC space &mdash; the primary acoustic evidence for your thesis.<br/><br/>
    <strong>Davies-Bouldin index</strong> is the average ratio of within-class scatter to
    between-class distance. Values below 1 are considered good; lower is better. Together these
    two metrics provide an objective, numeric proof that the dataset classes are meaningfully
    distinct without needing any pre/post TikTok comparison.
  </div>
</div>

<div class="sec" id="stats">
  <h2 class="sh">Per-class feature statistics (mean &plusmn; std)</h2>
  <div class="tbl-wrap">
  <table>
    <thead><tr>
      <th>Class</th><th>N</th>
      <th>Centroid (Hz)</th><th>Bandwidth (Hz)</th>
      <th>ZCR</th><th>RMS</th><th>Pitch (Hz)</th><th>Duration (s)</th>
    </tr></thead>
    <tbody>""" + trows + """</tbody>
  </table>
  </div>
</div>

<div id="lb">
  <button class="lbx" onclick="closeLb()">&#x2715;</button>
  <div id="lb-img"></div>
  <div class="lbt" id="lb-ttl"></div>
  <div class="lbn">
    <button onclick="lbStep(-1)">&larr; Prev</button>
    <button onclick="lbStep(1)">Next &rarr;</button>
  </div>
</div>

<script>
const PLOTS = """ + plots_json + """;
const IMGS  = """ + imgs_json  + """;
let cur = 0;
function tab(id, btn) {
  document.querySelectorAll('.sec').forEach(s => s.classList.remove('on'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('on'));
  document.getElementById(id).classList.add('on');
  btn.classList.add('on');
}
function openLb(f) {
  cur = PLOTS.findIndex(p => p.fname === f);
  renderLb();
  document.getElementById('lb').classList.add('on');
}
function closeLb() { document.getElementById('lb').classList.remove('on'); }
function lbStep(d) { cur = (cur + d + PLOTS.length) % PLOTS.length; renderLb(); }
function renderLb() {
  const p = PLOTS[cur];
  document.getElementById('lb-img').innerHTML =
    '<img src="data:image/png;base64,' + IMGS[p.fname] + '" alt="' + p.title + '"/>';
  document.getElementById('lb-ttl').textContent = p.title;
}
document.getElementById('lb').addEventListener('click', e => {
  if (e.target === document.getElementById('lb')) closeLb();
});
document.addEventListener('keydown', e => {
  if (!document.getElementById('lb').classList.contains('on')) return;
  if (e.key === 'Escape') closeLb();
  if (e.key === 'ArrowRight') lbStep(1);
  if (e.key === 'ArrowLeft')  lbStep(-1);
});
function filt(cat, btn) {
  document.querySelectorAll('.fb').forEach(b => b.classList.remove('on'));
  btn.classList.add('on');
  document.querySelectorAll('#fgrid .fig-card').forEach(card => {
    card.style.display = (cat === 'all' || card.dataset.cat.includes(cat)) ? '' : 'none';
  });
}
</script>
</body>
</html>"""

    out = os.path.join(OUTPUT_DIR, "dashboard.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"  ✔  dashboard.html")


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  TikTok Deepfake Audio — Dataset Validation (standalone)    ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    datasets = load_all()
    if not any(datasets.values()):
        print("[ERROR] No files loaded. Check BASE_DIR and CLASS_DIRS.")
        return

    rows    = print_summary(datasets)
    sil, db = compute_cluster_metrics(datasets)

    print("\n── Generating figures ──────────────────────────────────────────")
    plot_class_balance(datasets)
    plot_duration(datasets)
    plot_sample_rate(datasets)
    plot_spectral_centroid(datasets)
    plot_rms(datasets)
    plot_zcr(datasets)
    plot_mfcc_means(datasets)
    plot_mfcc_variance(datasets)
    plot_mfcc_distance(datasets)
    plot_embedding(datasets)
    # cluster metrics bar chart already saved inside compute_cluster_metrics()
    plot_js_divergence(datasets)
    plot_feature_boxplots(datasets)
    plot_spectrogram_grid(datasets)
    plot_pitch(datasets)
    plot_intra_diversity(datasets)
    plot_feature_correlation(datasets)

    build_dashboard(rows, sil, db)

    print(f"\n✅  Done — all outputs in → ./{OUTPUT_DIR}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    {f}")


if __name__ == "__main__":
    main()