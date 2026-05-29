import torch
import numpy as np
import soundfile as sf
import librosa


class PretrainedDetector:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            from sonics import HFAudioClassifier
        except ImportError:
            raise ImportError(
                "Install SONICS:\n"
                "pip install git+https://github.com/awsaf49/sonics.git"
            )

        self.model = HFAudioClassifier.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.target_sr = 16000

        # Infer clip duration from model name
        if "5s" in model_name:
            self.clip_seconds = 5
        elif "120s" in model_name:
            self.clip_seconds = 120
        else:
            raise ValueError(f"Unknown model duration in name: {model_name}")

    def fix_length(self, waveform):
        target_len = int(self.target_sr * self.clip_seconds)

        if len(waveform) > target_len:
            # Center crop (important for music)
            start = (len(waveform) - target_len) // 2
            waveform = waveform[start:start + target_len]
        else:
            # Zero pad
            pad_len = target_len - len(waveform)
            waveform = np.pad(waveform, (0, pad_len))

        return waveform

    def load_audio(self, path):
        waveform, sr = sf.read(path, dtype="float32")

        # Stereo → mono
        if waveform.ndim == 2:
            waveform = np.mean(waveform, axis=1)

        # Resample
        if sr != self.target_sr:
            waveform = librosa.resample(
                waveform,
                orig_sr=sr,
                target_sr=self.target_sr
            )

        # Fix duration (CRITICAL for SONICS)
        waveform = self.fix_length(waveform)

        # Convert to tensor [B, T]
        waveform = torch.from_numpy(waveform).float().unsqueeze(0)

        return waveform.to(self.device)

    @torch.no_grad()
    def predict(self, audio_path):
        try:
            waveform = self.load_audio(audio_path)

            logits = self.model(waveform)

            # ---- Robust output handling ----
            if logits.ndim == 2 and logits.shape[-1] == 1:
                # Binary logit
                prob = torch.sigmoid(logits).item()

            elif logits.ndim == 2 and logits.shape[-1] == 2:
                # Softmax output [real, fake]
                prob = torch.softmax(logits, dim=-1)[0, 1].item()

            else:
                # Temporal logits or unknown shape
                prob = torch.sigmoid(logits).mean().item()

            label = "FAKE" if prob >= 0.5 else "REAL"

            return {
                "label": label,
                "probability": float(prob),
                "success": True,
                "error": None
            }

        except Exception as e:
            return {
                "label": None,
                "probability": None,
                "success": False,
                "error": str(e)
            }