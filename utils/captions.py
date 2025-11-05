from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import subprocess
import shlex
import math
import tempfile
import json

from imageio_ffmpeg import get_ffmpeg_exe
import os


def _fmt_ass_time(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    # ASS uses centiseconds
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _sanitize_text(text: str) -> str:
    # Escape ASS reserved braces; keep our explicit line breaks as \N
    t = text.replace("{", "(").replace("}", ")")
    t = t.replace("\r", "")
    t = t.replace("\n", " ")
    return t.strip()


def _wrap_text(text: str, max_chars: int) -> str:
    """Greedy wrap at spaces into lines no longer than max_chars. Returns text with \\N breaks."""
    if max_chars <= 8:
        max_chars = 8
    words = text.split()
    lines = []
    cur = []
    cur_len = 0
    for w in words:
        if cur_len + (1 if cur else 0) + len(w) <= max_chars:
            cur.append(w)
            cur_len += (1 if cur_len else 0) + len(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
    if cur:
        lines.append(" ".join(cur))
    return "\\N".join(lines)


def _to_ass_color(color: str) -> str:
    """Convert CSS-like color to ASS &HAA BB GG RR (A=00 opaque). Accepts names or #RRGGBB."""
    named = {
        "yellow": (255, 255, 0),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
    }
    c = color.strip().lower()
    r = g = b = None
    if c in named:
        r, g, b = named[c]
    elif c.startswith('#') and len(c) == 7:
        try:
            r = int(c[1:3], 16)
            g = int(c[3:5], 16)
            b = int(c[5:7], 16)
        except Exception:
            r, g, b = 255, 255, 0
    else:
        r, g, b = 255, 255, 0
    # ASS format &HAA BB GG RR, AA=00 (opaque)
    return f"&H00{b:02X}{g:02X}{r:02X}"


def _make_ass_content(lines: List[Dict[str, float | str]], out_w: int, out_h: int, caption_color: str = "yellow") -> str:
    # Place about 25% from bottom: bottom-center alignment with MarginV
    margin_v = int(0.25 * out_h)
    # Left/right padding ~7% of width
    margin_l = int(0.07 * out_w)
    margin_r = int(0.07 * out_w)
    # Font size tuned for 1080x1920; scale relative to output
    base_font = 72  # bumped up a bit
    font_size = int(base_font * (out_w / 1080))
    # Estimate characters per line for manual wrap (approx 0.56 * font_size per char)
    est_char_w = 0.56 * font_size
    max_chars = max(12, int((out_w - (margin_l + margin_r)) / est_char_w))
    primary = _to_ass_color(caption_color)
    secondary = primary
    outline_color = "&H00000000"  # black
    outline_w = 4  # bumped outline thickness

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {out_w}
PlayResY: {out_h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ClipCaption, Arial, {font_size}, {primary}, {secondary}, {outline_color}, &H7F000000, -1, 0, 0, 0, 100, 100, 0, 0, 1, {outline_w}, 0, 2, {margin_l}, {margin_r}, {margin_v}, 1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for it in lines:
        start = _fmt_ass_time(float(it["start"]))
        end = _fmt_ass_time(float(it["end"]))
        txt = _sanitize_text(str(it["text"]))
        txt = _wrap_text(txt, max_chars)
        events.append(f"Dialogue: 0,{start},{end},ClipCaption,,0,0,{margin_v},,{txt}")
    return header + "\n".join(events) + "\n"


def _filter_window_transcript(transcript_items: List[Dict], window: Dict[str, float | str]) -> List[Dict]:
    if not transcript_items:
        return []
    a = float(window["start"])
    b = float(window["end"])
    lines = []
    for it in transcript_items:
        ts = float(it.get("start", 0.0))
        te = float(it.get("end", ts + max(0.5, float(it.get("duration", 0.5)))))
        if te < a or ts > b:
            continue
        # clamp + make relative to window
        rs = max(0.0, ts - a)
        re = min(b - a, te - a)
        if re - rs <= 0.05:
            continue
        lines.append({"start": rs, "end": re, "text": it.get("text", "")})
    return lines


def _try_whisper_transcribe(video_path: Path) -> List[Dict]:
    """Attempt to transcribe using openai-whisper if installed. Returns list of {start,end,text}.
    This is a best-effort optional path.
    """
    try:
        import whisper  # type: ignore
    except Exception:
        return []
    try:
        # Ensure ffmpeg from imageio-ffmpeg is visible to whisper
        ffmpeg_bin = get_ffmpeg_exe()
        ffmpeg_dir = str(Path(ffmpeg_bin).parent)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        model = whisper.load_model("base")
        res = model.transcribe(str(video_path), language="id", fp16=False)
        segments = res.get("segments", [])
        out = []
        for s in segments:
            out.append({"start": float(s["start"]), "end": float(s["end"]), "text": s.get("text", "").strip()})
        return out
    except Exception:
        return []


def generate_and_burn_captions(
    input_video: Path | str,
    output_video: Path | str,
    window: Dict[str, float | str],
    transcript_items: List[Dict] | None,
    out_w: int = 1080,
    out_h: int = 1920,
    caption_color: str = "yellow",
) -> Path:
    input_video = Path(input_video)
    output_video = Path(output_video)

    # 1) try transcript lines from input; else 2) ASR on the rendered clip file
    lines = _filter_window_transcript(transcript_items or [], window)
    if not lines:
        # ASR fallback on the already-rendered clip
        lines = _try_whisper_transcribe(input_video)
        # No need to shift times since it's the final clip (starts at 0)

    if not lines:
        # Nothing to do
        return input_video

    ass_text = _make_ass_content(lines, out_w=out_w, out_h=out_h, caption_color=caption_color)

    with tempfile.TemporaryDirectory() as td:
        ass_path = Path(td) / "subs.ass"
        ass_path.write_text(ass_text, encoding="utf-8")
        ffmpeg = get_ffmpeg_exe()
        # Burn ASS onto the video
        cmd = [
            ffmpeg,
            "-y",
            "-i", str(input_video),
            "-vf", f"subtitles={str(ass_path).replace('\\','/').replace(':','\\:')}",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(output_video),
        ]
        subprocess.run(cmd, check=True)

    return output_video
