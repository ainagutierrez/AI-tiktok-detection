import os
import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset
import torchaudio.transforms as T

class AudioDataset(Dataset):
    def __init__(self, real_dir, fake_dir, sr=16000, duration=3):
        self.samples = []
        self.sr = sr
        self.num_samples = sr * duration
        self.mel = T.MelSpectrogram(sample_rate=sr, n_mels=64)

        for root, _, files in os.walk(real_dir):
            for f in files:
                if f.lower().endswith(".mp3"):
                    self.samples.append((os.path.join(root, f), 0))  # label 0 = real

        for root, _, files in os.walk(fake_dir):
            for f in files:
                if f.lower().endswith(".mp3"):
                    self.samples.append((os.path.join(root, f), 1))  # label 1 = fake

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]


        try:
            y, sr = librosa.load(path, sr=self.sr, mono=True)   # already resampled + mono
            waveform = torch.from_numpy(y).unsqueeze(0)          # shape: (1, samples)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            waveform = torch.zeros(1, self.num_samples)
            sr = self.sr

        if waveform.shape[1] >= self.num_samples:
            max_start = waveform.shape[1] - self.num_samples
            start = torch.randint(0, max_start + 1, (1,)).item()
            waveform = waveform[:, start:start + self.num_samples]
        else:
            pad_len = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()

        mel_spec = self.mel(waveform.squeeze(0))
        mel_spec = mel_spec.clamp(min=1e-9)   # avoid zeros
        log_mel_spec = torch.log(mel_spec)
        spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-9)
        spec = spec.unsqueeze(0)

        return spec, torch.tensor(label, dtype=torch.float32)
