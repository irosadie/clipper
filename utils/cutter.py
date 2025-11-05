from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Union
import subprocess
from imageio_ffmpeg import get_ffmpeg_exe


def _run(cmd: Union[str, List[str]]) -> None:
    # Prefer list form to avoid shell-like parsing issues (commas in -vf expressions)
    if isinstance(cmd, str):
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}):\n{proc.stderr.decode('utf-8', 'ignore')}")


def cut_segments(video_path: Path, windows: List[Dict[str, float | str]], outdir: Path) -> List[Path]:
    """Cut segments using ffmpeg re-encode to avoid decoder issues (mmco errors)."""
    video_path = Path(video_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []
    ffmpeg = get_ffmpeg_exe()

    for i, w in enumerate(windows, start=1):
        a = max(0.0, float(w["start"]))
        b = float(w["end"]) if "end" in w else a + float(w.get("duration", 0))
        if b <= a:
            continue
        out = outdir / f"short_{i:02d}_{int(a)}-{int(b)}.mp4"
        # Use input seeking with re-encode; set sane defaults
        # Build argv list; escape comma in scale() so ffmpeg doesn't split filters
        cmd = [
            ffmpeg,
            "-y",
            "-ss", f"{a:.3f}",
            "-to", f"{b:.3f}",
            "-i", str(video_path),
            "-vf", "scale=min(1280\\,iw):-2",
            "-r", "30",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(out),
        ]
        _run(cmd)
        outputs.append(out)

    return outputs
