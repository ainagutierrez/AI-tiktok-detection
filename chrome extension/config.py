"""
Configuration file for the audio detection pipeline
Modify these values to control pipeline behavior
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CNN Model paths
CNN_MODEL_PATH = "your/cnn/model"

# Output paths
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# CNN Configuration
CNN_CONFIG = {
    'sr': 16000,
    'duration': 3,
    'n_mels': 64,
    'batch_size': 16,
    'learning_rate': 1e-4,
    'epochs': 30
}

# Pretrained Model Configuration
PRETRAINED_MODELS = [
    "awsaf49/sonics-spectttra-alpha-5s",
    "awsaf49/sonics-spectttra-beta-5s",
    "awsaf49/sonics-spectttra-gamma-5s",
    "awsaf49/sonics-spectttra-alpha-120s",
    "awsaf49/sonics-spectttra-beta-120s",
    "awsaf49/sonics-spectttra-gamma-120s"
]

# Default pretrained model to use
DEFAULT_PRETRAINED_MODEL = "awsaf49/sonics-spectttra-gamma-120s"

# Which detection methods to run
ENABLED_METHODS = {
    'cnn': True,
    'pretrained': True
}

OUTPUT_FORMAT = 'json'  # Options: 'json', 'csv', 'table'

DEVICE = "cuda"  # Options: 'cuda', 'cpu', 'auto'

SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']

GROUND_TRUTH = {
    'prova.mp3': 'FAKE',
    'edsheeran.mp3': 'REAL',
    'real.mp3': 'REAL'
}

from pathlib import Path

# Folder containing audio files
AUDIO_DIR = r"your/audio/path"

for audio_file in Path(AUDIO_DIR).glob("*"):
    if audio_file.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
        GROUND_TRUTH[audio_file.name] = 'ground_truth'
