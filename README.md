# AI-Generated Music Detection Under TikTok Audio Processing Conditions

Bachelor's thesis by **Aina Gutiérrez Llaó** — Audiovisual Systems Engineering, Universitat Pompeu Fabra (2025–2026).

**Supervisor:** Martín Rocamora

---

## Overview

State-of-the-art AI music detectors collapse when audio passes through TikTok's processing pipeline.

This repository:

* Replicates three published AI music detectors
* Benchmarks them under five audio transformations and TikTok AAC re-encoding
* Develops a TikTok-specific CNN detector
* Provides a Chrome extension for real-time detection while browsing TikTok

---

## Dataset and Dashboard

**Kaggle Dataset**

https://www.kaggle.com/datasets/ainagutirrezlla/tiktok-ai-detection/data

**Interactive Dashboard**

https://audio-model-comparison-dashboard-iyb93ncxjgjnuyftpy2xd6.streamlit.app/

---

## Repository Structure

```text
ai_tiktok_detector/
├── chrome extension/            # Chrome extension backend: Flask server + CNN + SONICS pretrained detector (replication of Rahman et al.)
├── sonics & cnn/                # Standalone offline pipeline (identical logic to chrome_extension, no browser dependency)
├── deezer/                      # Deezer spectrogram-amplitude TensorFlow classifier (replication of D.Afchar et al.)
├── laura_cros_vila/             # CLAP/MusiCNN embedding-based classifier (replication of Cros Vila et al.)
├── tiktok_pipeline/             # Dataset construction: video wrapping + TikTok upload/download via Selenium
├── transformations/             # Audio augmentation scripts (pitch shift, reverb, EQ, time stretch, noise)
├── transformations_validations/ # Validation checks for each augmentation
└── metrics_analysis/            # Dataset evaluation, t-SNE/UMAP projections, thesis plots.
```

---

## Pretrained Models

The original pretrained models are **not included** in this repository because of their size.

### SONICS (Rahman et al.)

Install SONICS directly:

```bash
pip install git+https://github.com/awsaf49/sonics.git
```

### The AI Music Arms Race (Cros Vila et al.)

Download the pretrained resources from the original repository:

https://github.com/lcrosvila/ai-music-detection

### AI-Generated Music Detection and its Challenges (Afchar et al.)

Download the Deezer pretrained model from:

https://github.com/deezer/deepfake-detector

### Trained Models from This Thesis

The detectors trained as part of this thesis are available in the repository **Releases** section.

---

## Citation

If you use this code or dataset, please cite:

```text
Gutiérrez Llaó, A. (2026).

Benchmarking AI-Generated Music Detection Under TikTok Audio
Processing Conditions.

Bachelor's Thesis,
Universitat Pompeu Fabra,
Barcelona.
```

---

## References

Rahman et al. (2024)
*SONICS: Synthetic Or Not — Identifying Counterfeit Songs*
arXiv:2408.14080



Afchar et al. (2025)
*AI-Generated Music Detection and its Challenges*
arXiv:2501.10111



Cros Vila et al. (2025)
*The AI Music Arms Race*
Transactions of the International Society for Music Information Retrieval (TISMIR), 8(1), 179–194.
