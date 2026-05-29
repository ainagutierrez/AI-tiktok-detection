import torch
import torchaudio
import torchaudio.transforms as T
from model import SimpleSpectrogramCNN
import os
import warnings

class CNNDetector:
    """Wrapper for CNN-based detection"""
    
    def __init__(self, model_path, device=None, sr=16000, duration=3, n_mels=64):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        
        # Load model
        self.model = SimpleSpectrogramCNN(n_classes=1).to(self.device)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def load_audio_with_soundfile(self, audio_path):
        """Load audio using soundfile (doesn't require FFmpeg)"""
        try:
            import soundfile as sf
            waveform, sr = sf.read(audio_path, dtype='float32')
            # Convert to torch tensor and ensure correct shape [channels, samples]
            waveform = torch.from_numpy(waveform)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # Add channel dimension
            else:
                waveform = waveform.T  # soundfile returns [samples, channels]
            return waveform, sr
        except ImportError:
            raise ImportError("soundfile not installed. Run: pip install soundfile")
    
    def load_audio_with_librosa(self, audio_path):
        """Load audio using librosa (alternative fallback)"""
        try:
            import librosa
            waveform, sr = librosa.load(audio_path, sr=None, mono=False)
            # Convert to torch tensor
            if waveform.ndim == 1:
                waveform = torch.from_numpy(waveform).unsqueeze(0)
            else:
                waveform = torch.from_numpy(waveform)
            return waveform, sr
        except ImportError:
            raise ImportError("librosa not installed. Run: pip install librosa")
    
    def load_audio(self, audio_path):
        """Load audio with multiple fallback methods"""
        errors = []
        
        # Try soundfile first (no FFmpeg required)
        try:
            return self.load_audio_with_soundfile(audio_path)
        except Exception as e:
            errors.append(f"soundfile: {str(e)}")
        
        # Try librosa as fallback
        try:
            return self.load_audio_with_librosa(audio_path)
        except Exception as e:
            errors.append(f"librosa: {str(e)}")
        
        # Try torchaudio as last resort (requires FFmpeg)
        try:
            return torchaudio.load(audio_path)
        except Exception as e:
            errors.append(f"torchaudio: {str(e)}")
        
        # If all methods failed
        raise RuntimeError(
            f"Failed to load audio file '{audio_path}' with all available methods.\n"
            f"Errors:\n" + "\n".join(f"  - {err}" for err in errors) +
            f"\n\nSolutions:\n"
            f"  1. Install soundfile: pip install soundfile\n"
            f"  2. Install librosa: pip install librosa\n"
            f"  3. Install FFmpeg for torchaudio support"
        )
        
    def predict(self, audio_path):
        try:
            waveform, sr = self.load_audio(audio_path)
            
            if sr != self.sr:
                waveform = torchaudio.functional.resample(waveform, sr, self.sr)
            waveform = waveform.mean(dim=0)  # mono

            num_samples = self.sr * self.duration
            if waveform.shape[0] > num_samples:
                waveform = waveform[:num_samples]
            else:
                pad_len = num_samples - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))

            # Mel Spectrogram
            mel = T.MelSpectrogram(sample_rate=self.sr, n_mels=self.n_mels)
            spec = torch.log(mel(waveform) + 1e-6).unsqueeze(0).unsqueeze(0)

            # Normalize
            spec = (spec - spec.mean()) / (spec.std() + 1e-9)
            spec = spec.to(self.device)

            with torch.no_grad():
                output = self.model(spec)
                prob = torch.sigmoid(output).item()
                label = "FAKE" if prob > 0.5 else "REAL"
                
            return {
                'label': label,
                'probability': prob,
                'method': 'CNN',
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'label': None,
                'probability': None,
                'method': 'CNN',
                'success': False,
                'error': str(e)
            }

# Backward compatibility
def predict_audio(path):
    """Legacy function for backward compatibility"""
    detector = CNNDetector(MODEL_PATH="model.pth")
    result = detector.predict(path)
    print(f"Prediction: {result['label']} (probability: {result['probability']:.2f})")
    return result['label'], result['probability']

if __name__ == "__main__":
    # Example usage
    detector = CNNDetector(model_path="model.pth")
    song_path = "test_audio.mp3"
    result = detector.predict(song_path)
    print(f"Prediction: {result['label']} (probability: {result['probability']:.2f})")