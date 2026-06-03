import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def get_scores(path):
    with open(path) as f:
        d = json.load(f)
    scores = []
    for x in d["results"]["files"]:
        preds = x.get("predictions", {})
        cnn = preds.get("cnn") or next(iter(preds.values()), None)
        if cnn and cnn.get("success"):
            scores.append(cnn["probability"])
    return scores

fake_sonics  = get_scores(r"your/path/to/the/scores/of/the/cnn_trained_on_sonics")
real_sonics  = get_scores(r"your/path/to/the/scores/of/the/cnn_trained_on_sonics")
real_laura    = get_scores(r"your/path/to/the/scores/of/the/cnn_trained_on_laura")
suno_laura    = get_scores(r"your/path/to/the/scores/of/the/cnn_trained_on_laura")
udio_laura    = get_scores(r"your/path/to/the/scores/of/the/cnn_trained_on_laura")
real_tik600  = get_scores(r"your/path/to/the/scores/of/the/cnn_trained_on_tiktok_600")
fake_tik600  = get_scores(r"your/path/to/the/scores/of/the/cnn_trained_on_tiktok_600")
real_tik700  = get_scores(r"your/path/to/the/scores/of/the/cnn_trained_on_tiktok_700")
fake_tik700  = get_scores(r""your/path/to/the/scores/of/the/cnn_trained_on_tiktok_700"")

def plot_kde(ax, scores, color, label, bw_method=0.05):
    scores = np.array(scores)
    kde = gaussian_kde(scores, bw_method=bw_method)
    xs = np.linspace(0, 1, 500)
    ys = kde(xs)
    ax.plot(xs, ys, color=color, linewidth=2.5, label=f"{label} (n={len(scores):,})")
    ax.fill_between(xs, ys, alpha=0.12, color=color)

def add_clip_note(ax, full_peak, ylim_top):
    ax.annotate(
        f"↑ peak ≈ {full_peak:.0f}",
        xy=(0.01, ylim_top),
        xytext=(0.08, ylim_top * 0.88),
        fontsize=9,
        color="#888780",
        arrowprops=dict(arrowstyle="->", color="#888780", lw=0.8),
    )

def style_ax(ax, title, ylim=None, clip_scores=None, bw_clip=0.03):
    ax.axvline(0.5, color="#888780", linestyle="--", linewidth=1.5, label="τ = 0.5")
    ax.set_title(title, fontsize=13, fontweight="500")
    ax.set_xlabel("Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    if ylim:
        ax.set_ylim(0, ylim)
        if clip_scores is not None:
            peak = float(gaussian_kde(np.array(clip_scores), bw_method=bw_clip)(0.0))
            add_clip_note(ax, peak, ylim)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
plot_kde(ax, real_sonics, color="#1D9E75", label="Real", bw_method=0.03)
plot_kde(ax, fake_sonics, color="#D85A30", label="Fake", bw_method=0.05)
style_ax(ax, "CNN trained on Sonics", ylim=20, clip_scores=real_sonics, bw_clip=0.03)

ax = axes[0, 1]
plot_kde(ax, real_laura, color="#1D9E75", label="Real", bw_method=0.03)
plot_kde(ax, suno_laura, color="#D85A30", label="Suno", bw_method=0.05)
plot_kde(ax, udio_laura, color="#378ADD", label="Udio", bw_method=0.03)
style_ax(ax, "CNN trained on Laura Cros", ylim=20, clip_scores=real_laura, bw_clip=0.03)

ax = axes[1, 0]
plot_kde(ax, real_tik600, color="#1D9E75", label="Real", bw_method=0.05)
plot_kde(ax, fake_tik600, color="#D85A30", label="Fake", bw_method=0.05)
style_ax(ax, " CNN TikTok — model 600")

ax = axes[1, 1]
plot_kde(ax, real_tik700, color="#1D9E75", label="Real", bw_method=0.05)
plot_kde(ax, fake_tik700, color="#D85A30", label="Fake", bw_method=0.05)
style_ax(ax, "CNN TikTok — model 700")

plt.tight_layout()
plt.savefig("score_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
