import subprocess

USERNAME = ""
url = f"https://www.tiktok.com/@{USERNAME}"

cmd = [
    "yt-dlp",
    "--extract-audio",
    "--audio-format", "mp3",
    "--audio-quality", "192K",
    "--embed-metadata",
    "--download-archive", "downloaded.txt",
    "-o", "%(uploader)s/%(upload_date)s_%(id)s.%(ext)s",
    url
]

subprocess.run(cmd)