import yt_dlp
import os
from typing import TypedDict


class VideoData(TypedDict):
    filepath: str
    caption: str
    uploader: str
    id: str
    original_url: str


def download_video(url: str) -> VideoData:
    temp_dir = "temp_reels"
    os.makedirs(temp_dir, exist_ok=True)

    # yt-dlp configuration to ensure we get an mp4
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': f'{temp_dir}/%(id)s.%(ext)s',
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        filepath = ydl.prepare_filename(info_dict)
        if not os.path.exists(filepath):
            base, _ = os.path.splitext(filepath)
            for ext in ['.mp4', '.mkv', '.webm']:
                if os.path.exists(base + ext):
                    filepath = base + ext
                    break

        return {
            "filepath": filepath,
            "caption": info_dict.get('description', ''),
            "uploader": info_dict.get('uploader', 'Unknown'),
            "id": info_dict.get('id', 'Unknown'),
            "original_url": url
        }
