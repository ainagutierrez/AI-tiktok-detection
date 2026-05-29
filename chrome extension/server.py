import os
import sys
import tempfile
import subprocess
import logging
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict import CNNDetector
import config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=[
    "chrome-extension://*",       # popup / options pages
    "https://www.tiktok.com",     # content.js runs in the TikTok page context
    "https://*.tiktok.com",       # covers regional TikTok domains
])  # allow requests from the extension and from TikTok pages (where content.js runs)

# Load detector once at startup
logger.info("Loading CNN detector…")
detector = CNNDetector(
    model_path=config.CNN_MODEL_PATH,
    sr=config.CNN_CONFIG['sr'],
    duration=config.CNN_CONFIG['duration'],
    n_mels=config.CNN_CONFIG['n_mels']
)
logger.info("CNN detector ready ✓")


def convert_to_wav(input_path: str, output_path: str) -> bool:
    """
    Convert any audio format to WAV using ffmpeg.
    Returns True on success.
    ffmpeg must be installed (brew install ffmpeg / apt install ffmpeg).
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-ar", str(config.CNN_CONFIG['sr']),  # resample to model sr
                "-ac", "1",                            # mono
                "-f", "wav",
                output_path
            ],
            capture_output=True,
            timeout=30
        )
        if result.returncode != 0:
            logger.error("ffmpeg stderr: %s", result.stderr.decode())
            return False
        return True
    except FileNotFoundError:
        logger.error("ffmpeg not found. Install it: brew install ffmpeg  /  apt install ffmpeg")
        return False
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timed out")
        return False


@app.route('/health', methods=['GET'])
def health():
    """Health-check used by the popup to verify the server is running."""
    return jsonify({'status': 'ok', 'model': config.CNN_MODEL_PATH})


@app.route('/detect', methods=['POST'])
def detect():
    """
    Receive a webm/ogg audio file, convert to WAV, run CNN detector, return result.

    Expected multipart form field: 'audio'

    Returns JSON:
    {
        "success": true,
        "label": "REAL" | "AI",
        "probability": 0.92,
        "raw": { ... full detector output ... }
    }
    """
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file in request'}), 400

    audio_file = request.files['audio']

    # Save the incoming file to a temp location
    suffix = Path(audio_file.filename or 'audio.webm').suffix or '.webm'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        audio_file.save(tmp_in.name)
        tmp_in_path = tmp_in.name

    tmp_wav_path = tmp_in_path.replace(suffix, '.wav')

    try:
        logger.info("Converting %s → WAV", tmp_in_path)
        ok = convert_to_wav(tmp_in_path, tmp_wav_path)
        if not ok:
            return jsonify({
                'success': False,
                'error': 'Audio conversion failed. Is ffmpeg installed?'
            }), 500

        logger.info("Running CNN detector on %s", tmp_wav_path)
        prediction = detector.predict(tmp_wav_path)

        if not prediction['success']:
            return jsonify({
                'success': False,
                'error': prediction.get('error', 'Unknown detection error')
            }), 500

        probability = prediction['probability']

        # If classified as REAL, invert probability for UI
        if prediction['label'] == 'REAL':
            probability = 1 - probability

        logger.info(
            "Result: %s  (ui_prob=%.4f)",
            prediction['label'],
            probability
        )

        return jsonify({
            'success': True,
            'label': prediction['label'],
            'probability': probability,
            'raw': prediction
        })
        

    finally:
        # Clean up temp files
        for p in [tmp_in_path, tmp_wav_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  TikTok AI Music Detector — Local Server")
    print("="*55)
    print(f"  Model : {config.CNN_MODEL_PATH}")
    print(f"  Port  : 5000")
    print("="*55 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=False)
