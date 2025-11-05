from __future__ import annotations
from typing import List, Dict, Optional
from urllib.parse import urlparse, parse_qs
import re
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound


TranscriptItem = Dict[str, float | str]


def _extract_video_id(url: str) -> Optional[str]:
    # Supports various YouTube URL formats
    # e.g., https://www.youtube.com/watch?v=VIDEO_ID or https://youtu.be/VIDEO_ID
    parsed = urlparse(url)
    if parsed.netloc in {"youtu.be"}:
        return parsed.path.strip("/") or None
    qs = parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]
    # Shorts or embed
    m = re.search(r"/(shorts|embed)/([A-Za-z0-9_-]{6,})", parsed.path)
    if m:
        return m.group(2)
    return None


def fetch_transcript(url: str, preferred_languages: Optional[list[str]] = None) -> List[TranscriptItem]:
    vid = _extract_video_id(url)
    if not vid:
        raise ValueError("Could not extract YouTube video ID from URL")

    langs = preferred_languages or ["id", "en"]

    try:
        # Try multi-language request in priority order
        for lang in langs:
            try:
                data = YouTubeTranscriptApi.get_transcript(vid, languages=[lang])
                if data:
                    return data
            except NoTranscriptFound:
                continue
        # Fallback: auto-generated transcript in any available language
        data = YouTubeTranscriptApi.get_transcript(vid)
        return data
    except (TranscriptsDisabled, NoTranscriptFound):
        return []
    except Exception:
        # Network issues, parsing errors, throttling, etc.
        return []


# ---------- Local subtitle parsing (SRT/VTT) ----------

_SRT_TIME = re.compile(r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{1,3})")
_VTT_TIME = re.compile(r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<ms>\d{1,3})")


def _parse_timecode(tc: str) -> float:
    m = _SRT_TIME.match(tc) or _VTT_TIME.match(tc)
    if not m:
        return 0.0
    h = int(m.group("h"))
    mi = int(m.group("m"))
    s = int(m.group("s"))
    ms = int(m.group("ms").ljust(3, "0")[:3])
    return h * 3600 + mi * 60 + s + ms / 1000.0


def _parse_srt_text(content: str) -> List[TranscriptItem]:
    items: List[TranscriptItem] = []
    blocks = re.split(r"\n\s*\n", content.strip(), flags=re.MULTILINE)
    for blk in blocks:
        lines = [ln.strip("\ufeff") for ln in blk.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        # optional index number at line 0
        if "-->" in lines[0]:
            timing = lines[0]
            text_lines = lines[1:]
        elif len(lines) >= 2 and "-->" in lines[1]:
            timing = lines[1]
            text_lines = lines[2:]
        else:
            continue
        try:
            start_tc, end_tc = [t.strip() for t in timing.split("-->")]
            start = _parse_timecode(start_tc)
            end = _parse_timecode(end_tc)
            duration = max(0.0, end - start)
        except Exception:
            continue
        text = " ".join(text_lines).strip()
        if not text:
            continue
        items.append({"start": start, "duration": duration, "text": text})
    return items


def _parse_vtt_text(content: str) -> List[TranscriptItem]:
    # Remove WEBVTT header if present
    content = re.sub(r"^\s*WEBVTT.*?\n+", "", content, flags=re.IGNORECASE | re.DOTALL)
    # Reuse SRT parser since format is similar with '.' ms
    return _parse_srt_text(content)


def load_local_subtitle_transcript(video_path: Path) -> List[TranscriptItem]:
    """Look for a sidecar subtitle file next to the video and parse it.

    Checks common patterns: same stem with .srt/.vtt or language suffixes.
    """
    video_path = Path(video_path)
    stem = video_path.with_suffix("").name
    parent = video_path.parent
    candidates = []
    langs = ["", ".id", ".en", ".ID", ".EN"]
    for lang in langs:
        candidates.append(parent / f"{stem}{lang}.srt")
        candidates.append(parent / f"{stem}{lang}.vtt")
    for sub_path in candidates:
        if not sub_path.exists():
            continue
        try:
            text = sub_path.read_text(encoding="utf-8", errors="ignore")
            if sub_path.suffix.lower() == ".srt":
                return _parse_srt_text(text)
            else:
                return _parse_vtt_text(text)
        except Exception:
            continue
    return []
