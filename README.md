**AI-Generated Music Detection Under TikTok Audio Processing Conditions**


Bachelor's thesis by Aina Gutiérrez Llaó — Audiovisual Systems Engineering, Universitat Pompeu Fabra (2025–2026).
Supervisor: Martín Rocamora.


TL;DR: State-of-the-art AI music detectors collapse when audio passes through TikTok's pipeline. This repo replicates three published detectors, benchmarks them under five audio transformations and TikTok's own AAC re-encoding, builds a TikTok-specific CNN, and ships a Chrome extension for real-time detection while browsing.


Dataset (Kaggle): https://www.kaggle.com/datasets/ainagutirrezlla/tiktok-ai-detection/data
Interactive dashboard: https://audio-model-comparison-dashboard-iyb93ncxjgjnuyftpy2xd6.streamlit.app/



**Repository structure:**

ai_tiktok_detector/

├── chrome extension/ 
# Chrome extension backend: Flask server + CNN + SONICS pretrained detector (replication of Rahman et al.)
├── sonics & cnn/    
# Standalone offline pipeline (identical logic, no browser dependency) 
├── deezer/  
# Deezer spectrogram-amplitude TensorFlow classifier (replication of D.Afchar et al.)
├── laura_cros_vila/   
# CLAP/MusiCNN embedding-based classifier (replication of Cros Vila et al.)
├── tiktok_pipeline/   
# Dataset construction: video wrapping + TikTok upload/download via Selenium
├── transformations/ 
# Audio augmentation scripts (pitch shift, reverb, EQ, time stretch, noise)
├── transformations_validations/ 
# Validation checks for each augmentation
└── metrics_analysis/        # Dataset evaluation, t-SNE/UMAP projections, thesis plots



**Citation:**

If you use this code or dataset, please cite:
Gutiérrez Llaó, A. (2026). Benchmarking AI-Generated Music Detection Under TikTok Audio
Processing Conditions. Bachelor's thesis, Universitat Pompeu Fabra, Barcelona.



**References:**

Rahman et al. (2024). SONICS: Synthetic Or Not — Identifying Counterfeit Songs. arXiv:2408.14080
Afchar et al. (2025). AI-Generated Music Detection and its Challenges. arXiv:2501.10111
Cros Vila et al. (2025). The AI Music Arms Race. TISMIR, 8(1):179–194
