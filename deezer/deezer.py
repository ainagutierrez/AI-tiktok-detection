"""
Usage:
    python deezer.py --folder /path/to/songs --model specnn_amplitude --truth REAL

Optional flags:
    --slice  3.0           audio slice length in seconds (default: 3.0)
    --slices 10            number of random slices per file (default: 10)
    --sr     44100         sampling rate (default: 44100)
    --effects stft_db normalise
                          augmenter effects (default: stft_db normalise)
"""

import argparse
import os
import numpy as np
import librosa
import tensorflow as tf

class Augmenter:
    def __init__(self, sr=44100, effects=("stft_db", "normalise"),
                 win=512, hop=256, n_fft=1024,
                 normalise_mean=0.0, normalise_std=1.0):
        self.sr = sr
        self.effects = list(effects)
        self.fft_win = win
        self.fft_hop = hop
        self.fft_n   = n_fft
        self.norm_mean = tf.constant(normalise_mean, tf.float32)
        self.norm_std  = tf.constant(normalise_std,  tf.float32)

    @staticmethod
    def _switch_channels(x):
        return tf.transpose(x, [1, 0])

    def stft(self, x, mode="dB"):
        if x.shape[-1] == 2:
            x = self._switch_channels(x)
        cx = tf.signal.stft(x, self.fft_win, self.fft_hop, fft_length=self.fft_n)
        if mode == "dB":
            power = tf.square(tf.abs(cx))
            return tf.expand_dims(tf.math.log(tf.clip_by_value(power, 1e-10, 1e6))
                                  / tf.math.log(tf.constant(10.0, tf.float32)), -1)
        elif mode == "magnitude":
            return tf.expand_dims(tf.abs(cx), -1)
        else:
            return tf.stack((tf.math.real(cx), tf.math.imag(cx)), -1)

    def normalise(self, x):
        return (x - self.norm_mean) / self.norm_std

    def transform(self, x):
        y = x
        for effect in self.effects:
            if effect == "stft_db":
                y = self.stft(y, "dB")
            elif effect == "stft_mag":
                y = self.stft(y, "magnitude")
            elif effect == "stft_complex":
                y = self.stft(y, "complex")
            elif effect == "normalise":
                y = self.normalise(y)
        return y

SUPPORTED_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}

def load_audio(path: str, target_sr: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=target_sr, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)
    else:
        audio = audio.T
        if audio.shape[-1] == 1:
            audio = np.repeat(audio, 2, axis=-1)
        elif audio.shape[-1] > 2:
            audio = audio[:, :2]
    return audio.astype(np.float32)

def random_slices(audio: np.ndarray, sr: int, slice_sec: float, n: int) -> list:
    target_len = int(slice_sec * sr)
    total_len = audio.shape[0]
    if total_len < target_len:
        pad = np.zeros((target_len - total_len, 2), dtype=np.float32)
        audio = np.concatenate([audio, pad], axis=0)
        total_len = target_len
    slices = []
    for _ in range(n):
        offset = np.random.randint(0, total_len - target_len + 1)
        slices.append(audio[offset:offset + target_len])
    return slices

def load_model(model_path: str):
    saved_model = tf.saved_model.load(model_path)
    sig = saved_model.signatures["serving_default"]
    input_name = list(sig.structured_input_signature[1].keys())[0]
    spec = sig.structured_input_signature[1][input_name]
    inp_shape = spec.shape.as_list()[1:]
    inp_dtype = spec.dtype

    from keras.layers import TFSMLayer
    from keras import Input, Model
    layer = TFSMLayer(model_path, call_endpoint="serving_default")
    inp = Input(shape=inp_shape, dtype=inp_dtype)
    out = layer(inp)
    model = Model(inputs=inp, outputs=out)
    return model, inp_shape, inp_dtype

def predict_file(path, model, augmenter, inp_shape, sr, slice_sec, n_slices):
    result = {"file": os.path.basename(path), "path": path,
              "score": None, "label": None, "slices_used": 0, "error": ""}
    try:
        audio = load_audio(path, sr)
    except Exception as e:
        result["error"] = f"load error: {e}"
        return result

    patch_h, patch_w = inp_shape[0], inp_shape[1]
    slices = random_slices(audio, sr, slice_sec, n_slices)
    batch = []

    for s in slices:
        # Full spectrogram
        x = tf.constant(s, dtype=tf.float32)
        spec = augmenter.transform(x).numpy()
        if spec.ndim == 4:  # handle batch dim
            spec = spec[0]

        H, W, C = spec.shape

        # Crop multiple overlapping/random patches from spectrogram
        step_h = max(1, (H - patch_h) // 4)
        step_w = max(1, (W - patch_w) // 4)

        for oh in range(0, H - patch_h + 1, step_h):
            for ow in range(0, W - patch_w + 1, step_w):
                patch = spec[oh:oh + patch_h, ow:ow + patch_w, :]
                if patch.shape[0] < patch_h or patch.shape[1] < patch_w:
                    pad = np.zeros((patch_h, patch_w, C), np.float32)
                    pad[:patch.shape[0], :patch.shape[1], :] = patch
                    patch = pad
                batch.append(patch)

    if not batch:
        result["error"] = "No patches generated"
        return result

    batch = np.stack(batch, axis=0)
    preds = model(batch, training=False)
    if isinstance(preds, dict):
        key = next((k for k in ("deepfake", "output_0") if k in preds), list(preds.keys())[0])
        scores = preds[key].numpy().flatten()
    else:
        scores = np.array(preds).flatten()

    mean_score = float(np.mean(scores))
    result["score"] = round(mean_score, 4)
    result["label"] = "REAL" if mean_score >= 0.5 else "FAKE"
    result["slices_used"] = len(slices)
    return result

# --- Main ---
def find_audio_files(folder: str) -> list:
    files = []
    for root, _, fnames in os.walk(folder):
        for f in fnames:
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
                files.append(os.path.join(root, f))
    return sorted(files)

def main():
    parser = argparse.ArgumentParser(description="Deepfake audio detector with optional ground truth")
    parser.add_argument("--folder", required=True, help="Folder containing audio files")
    parser.add_argument("--model", default="specnn_amplitude", help="SavedModel path")
    parser.add_argument("--truth", choices=["REAL","FAKE"], help="Ground truth of the folder")
    parser.add_argument("--slice", type=float, default=3.0)
    parser.add_argument("--slices", type=int, default=10)
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--effects", nargs="+", default=["stft_db","normalise"])
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Model  : {args.model}")
    print(f"  Folder : {args.folder}")
    print(f"  Truth  : {args.truth}")
    print(f"  Effects: {args.effects}")
    print(f"  Slices : {args.slices} × {args.slice}s @ {args.sr}Hz")
    print(f"{'='*60}\n")

    print("Loading model …")
    model, inp_shape, inp_dtype = load_model(args.model)
    print(f"  Input shape: {inp_shape}  dtype: {inp_dtype}\n")
    augmenter = Augmenter(sr=args.sr, effects=args.effects)

    files = find_audio_files(args.folder)
    if not files:
        print(f"No audio files found in: {args.folder}")
        return

    print(f"Found {len(files)} file(s)\n")
    print(f"  {'File':<44} {'Score':>7}  Label")
    print("  " + "-"*58)

    results = []
    correct = 0
    for path in files:
        r = predict_file(path, model, augmenter, inp_shape, args.sr, args.slice, args.slices)
        results.append(r)
        name = r["file"][:44]
        if r["error"]:
            print(f"  !! {name:<43} ERROR: {r['error']}")
        else:
            flag = "✓" if r["label"] == "REAL" else "✗"
            print(f"  {flag}  {name:<43} {r['score']:>7.4f}  {r['label']}")
            if args.truth:
                correct += int(r["label"] == args.truth)

    ok = [r for r in results if not r["error"]]
    print("  " + "-"*58)
    print(f"  {len(ok)} files processed — REAL: {sum(1 for r in ok if r['label']=='REAL')}  "
          f"FAKE: {sum(1 for r in ok if r['label']=='FAKE')}  Errors: {len(results)-len(ok)}")
    if args.truth:
        accuracy = correct / len(ok) if ok else 0
        print(f"  Accuracy vs ground truth ({args.truth}): {accuracy*100:.2f}%\n")

if __name__ == "__main__":
    main()