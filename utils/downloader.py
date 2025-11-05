from __future__ import annotations
from pathlib import Path
from typing import Optional

import yt_dlp


def download_video(
    url: str,
    downloads_dir: Path,
    cookiesfrombrowser: Optional[str] = None,
    cookiesfile: Optional[str] = None,
) -> Path:
    """Download a YouTube video as mp4 with audio merged and return the file path.

    Uses yt-dlp best video+audio and remuxes to mp4 when possible.
    """
    downloads_dir = Path(downloads_dir)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    outtmpl = str(downloads_dir / "%(title)s-%(id)s.%(ext)s")

    ydl_opts = {
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        # Prefer mp4 to ease editing; fallback to best available
        "format": "bv*[ext=mp4]+ba[ext=m4a]/bv*+ba/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        # Improve reliability for HLS/DASH
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 1,
        "hls_prefer_native": True,
        "hls_use_mpegts": True,
        "http_headers": {"User-Agent": "Mozilla/5.0"},
        # Try different YouTube player clients to avoid edge cases
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "web"],
                "skip": ["dash"]
            }
        },
        "geo_bypass": True,
    }

    # Optional cookies
    if cookiesfrombrowser:
        ydl_opts["cookiesfrombrowser"] = (cookiesfrombrowser,)
    if cookiesfile:
        ydl_opts["cookiefile"] = cookiesfile

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # Determine output path
        if "requested_downloads" in info and info["requested_downloads"]:
            filename = info["requested_downloads"][0]["_filename"]
            return Path(filename)
        # Fallback to anticipated name
        ext = info.get("ext", "mp4")
        title = info.get("title", info.get("id", "video"))
        return downloads_dir / f"{title}-{info.get('id','id')}.{ext}"
