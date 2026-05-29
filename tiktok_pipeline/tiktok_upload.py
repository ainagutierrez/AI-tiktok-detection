import os
import json
import time
import random
from seleniumbase import Driver
from moviepy import AudioFileClip, ColorClip


AUDIO_FOLDER = r"path"
VIDEO_OUTPUT_DIR = "path"
DOWNLOAD_DIR = os.getcwd()
MAX_FILES = 100

os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)


def crear_video_base(audio_path, video_index):
    output_path = os.path.join(VIDEO_OUTPUT_DIR, f"video_{video_index:03d}.mp4")
    
    if os.path.exists(output_path):
        print(f"-> Skipping: {output_path} already exists.")
        return output_path

    print(f"\n--- Creating video: {os.path.basename(audio_path)} ---")
    try:
        audio = AudioFileClip(audio_path)

        if audio.duration > 600:  # e.g. >10 minutes
            print(f"-> Skipping abnormal audio (duration: {audio.duration}s)")
            audio.close()
            return None
        clip = ColorClip(size=(720, 1280), color=(0, 0, 0), duration=audio.duration)
        clip = clip.with_audio(audio)
        
        clip.write_videofile(
            output_path, 
            fps=24, 
            codec="libx264", 
            audio_codec="aac", 
            audio_bitrate="192k",
            logger="bar",
            temp_audiofile=f"temp_{video_index}.m4a",
            remove_temp=True,
            ffmpeg_params=["-pix_fmt", "yuv420p"]
        )
        audio.close()
        clip.close()
        return output_path
    except Exception as e:
        print(f"!!! Error creating video: {e}")
        return None


def main():
    
    audios = [os.path.join(AUDIO_FOLDER, f) for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith(".mp3")]
    audios = sorted(audios)[:MAX_FILES]
    

    for i, audio_path in enumerate(audios, start=0):
        print(f"\n{'='*40}")
        print(f"PROCESSING SONG {i} OF {len(audios)}")
        print(f"{'='*40}")
        crear_video_base(audio_path, i)


if __name__ == "__main__":
    main()