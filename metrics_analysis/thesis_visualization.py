import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import label_binarize
from collections import OrderedDict

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
PALETTE = sns.color_palette("Set2")
OUTPUT_DIR = "putput_dir"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")

def load_checkpoint(path):
    """Load a .pth file and return (state_dict, extras)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        return state, ckpt
    # Raw state_dict
    return ckpt, {}

def plot_architecture_table(state_dict, extras):
    print("[1] Architecture summary table …")
    rows = []
    total_params = 0
    for name, tensor in state_dict.items():
        n = tensor.numel()
        total_params += n
        rows.append({
            "Layer": name,
            "Shape": str(list(tensor.shape)),
            "Parameters": f"{n:,}",
            "Dtype": str(tensor.dtype).replace("torch.", ""),
        })

    df = pd.DataFrame(rows)
    # Truncate if too many layers
    show = df if len(df) <= 30 else pd.concat([df.head(15), df.tail(15)])

    fig, ax = plt.subplots(figsize=(14, max(4, len(show) * 0.35 + 1.5)))
    ax.axis("off")
    tbl = ax.table(
        cellText=show.values,
        colLabels=show.columns,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    # Header style
    for j in range(len(show.columns)):
        tbl[(0, j)].set_facecolor("#4C72B0")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    # Alternating rows
    for i in range(1, len(show) + 1):
        color = "#EEF2FF" if i % 2 == 0 else "white"
        for j in range(len(show.columns)):
            tbl[(i, j)].set_facecolor(color)

    ax.set_title(
        f"Model Architecture  ·  {len(rows)} layers  ·  "
        f"{total_params:,} total parameters",
        fontsize=13, pad=12, fontweight="bold"
    )
    save(fig, "01_architecture_table.png")

def plot_param_distribution(state_dict):
    print("[2] Parameter distribution …")
    layer_names, param_counts = [], []
    for name, tensor in state_dict.items():
        layer_names.append(name.replace(".", "\n"))
        param_counts.append(tensor.numel())

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Bar chart
    ax = axes[0]
    colors = PALETTE[:len(layer_names)] if len(layer_names) <= 8 \
        else sns.color_palette("coolwarm", len(layer_names))
    bars = ax.barh(layer_names, param_counts, color=colors)
    ax.set_xlabel("Number of Parameters")
    ax.set_title("Parameters per Layer", fontweight="bold")
    ax.bar_label(bars, labels=[f"{v:,}" for v in param_counts],
                 padding=4, fontsize=7)

    # Pie chart (top-10 layers)
    ax2 = axes[1]
    sorted_idx = np.argsort(param_counts)[::-1]
    top_n = min(10, len(layer_names))
    top_vals = [param_counts[i] for i in sorted_idx[:top_n]]
    top_labs = [layer_names[i].replace("\n", ".") for i in sorted_idx[:top_n]]
    wedges, texts, autotexts = ax2.pie(
        top_vals, labels=top_labs, autopct="%1.1f%%",
        colors=sns.color_palette("tab10", top_n), startangle=140,
        textprops={"fontsize": 7}
    )
    ax2.set_title(f"Top-{top_n} Layers by Parameter Share", fontweight="bold")

    fig.suptitle("Model Parameter Distribution", fontsize=15, fontweight="bold", y=1.01)
    save(fig, "02_param_distribution.png")


def plot_weight_stats(state_dict):
    print("[3] Weight statistics …")
    records = []
    for name, tensor in state_dict.items():
        w = tensor.float().cpu().numpy().ravel()
        records.append({
            "Layer": name,
            "Mean": float(np.mean(w)),
            "Std": float(np.std(w)),
            "Min": float(np.min(w)),
            "Max": float(np.max(w)),
            "Sparsity (%)": float(100 * np.mean(w == 0)),
        })
    df = pd.DataFrame(records).set_index("Layer")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ["Mean", "Std", "Min", "Max"]
    colors_map = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for ax, metric, c in zip(axes.flat, metrics, colors_map):
        vals = df[metric].values
        names = [n.split(".")[-1] for n in df.index]
        ax.bar(names, vals, color=c, alpha=0.8)
        ax.set_title(f"Weight {metric} per Layer", fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Weight Statistics Across Layers", fontsize=15,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "03_weight_statistics.png")

    # Sparsity separately
    fig2, ax = plt.subplots(figsize=(10, 4))
    colors_s = ["#e74c3c" if v > 50 else "#2ecc71" for v in df["Sparsity (%)"].values]
    ax.bar([n.split(".")[-1] for n in df.index], df["Sparsity (%)"].values, color=colors_s)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="50 % threshold")
    ax.set_ylabel("Sparsity (%)")
    ax.set_xlabel("Layer")
    ax.set_title("Weight Sparsity per Layer (% of Zero Weights)", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    save(fig2, "03b_weight_sparsity.png")


def plot_weight_histograms(state_dict):
    print("[4] Weight histograms …")
    weight_layers = [(n, t) for n, t in state_dict.items()
                     if t.ndim >= 2]  # skip 1-D biases for clarity
    n = min(len(weight_layers), 9)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    axes = axes.flatten() if n > 1 else [axes]

    for i, (name, tensor) in enumerate(weight_layers[:n]):
        w = tensor.float().cpu().numpy().ravel()
        ax = axes[i]
        ax.hist(w, bins=60, color=PALETTE[i % len(PALETTE)], edgecolor="none", alpha=0.85)
        ax.axvline(w.mean(), color="red", linewidth=1.2, linestyle="--", label=f"μ={w.mean():.3f}")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Weight Value Distributions per Layer", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    save(fig, "04_weight_histograms.png")


def main():
    parser = argparse.ArgumentParser(description="Thesis visualizations for a .pth model")
    parser.add_argument("--model", default="model_tiktok.pth",
                        help="Path to .pth checkpoint")
    parser.add_argument(
        "--classes", nargs="+",
        default=["Viral", "High", "Medium", "Low", "Flop"],
        help="Class names for classification plots"
    )
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Thesis Visualizations  —  {args.model}")
    print(f"  Output directory: ./{OUTPUT_DIR}/")
    print(f"{'='*55}\n")

    state_dict, extras = load_checkpoint(args.model)
    print(f"  Loaded {len(state_dict)} tensors, "
          f"{sum(t.numel() for t in state_dict.values()):,} total parameters\n")

    plot_architecture_table(state_dict, extras)
    plot_param_distribution(state_dict)
    plot_weight_stats(state_dict)
    plot_weight_histograms(state_dict)

    print(f"\n✅  All plots saved to ./{OUTPUT_DIR}/")



if __name__ == "__main__":
    main()