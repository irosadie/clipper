# Clipper: YouTube Podcast → Shorts

A simple Python CLI to:

1. Read a YouTube URL and download the video (yt-dlp)
2. Detect highlight/punchline moments from the transcript (no API keys)
3. Cut them into short MP4 clips (moviepy + ffmpeg)

Optimized for talk/podcast-style videos. Transcript-based heuristics (exclamation, laughter tokens, excitement words, speaking-rate spikes) are used to propose highlights.

## Requirements

- Python 3.10+
- Internet access (to fetch transcript and download video)
- ffmpeg: moviepy uses imageio-ffmpeg to auto-download a binary on first use. If that fails, install ffmpeg manually.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage (quick start)

```bash
python app.py --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --duration 45 --top-k 3 --dry-run

# Remove --dry-run to download and cut
python app.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --duration 45 --top-k 3

# Use a local file directly (no download step)
python app.py --url "/path/to/local_video.mp4" --duration 45 --top-k 2
```

Outputs go to `./outputs`. Downloads go to `./downloads` (for YouTube URLs).

### Vertical 9:16 mode with face tiles

You can output vertical shorts with up to 3 stacked tiles (faces). Works for YouTube URLs (after download) or local files.

```bash
# Analyze and cut in 9:16 with up to 3 tiles
python app.py --url "/path/to/local_video.mp4" --duration 45 --top-k 3 \
  --aspect 9:16 --max-tiles 3

# Analyze only (no cutting)
python app.py --url "/path/to/local_video.mp4" --duration 45 --top-k 3 \
  --aspect 9:16 --max-tiles 3 --dry-run
```

Heuristics:
- If 2–3 distinct faces are found in a highlight, tiles will stack vertically (top→bottom) centered around faces.
- If fewer faces are found, remaining tiles repeat the main crop.
- If no faces are detected, falls back to a centered 9:16 crop.

### 9:16 speaker-focused camera (auto-framing)

Prefer a single view that follows the active speaker, like a smart PTZ camera? Use:

```bash
python app.py --url "/path/to/local_video.mp4" --duration 45 --top-k 3 \
  --aspect 9:16 --focus-speaker
```

How it works:
- MediaPipe Face Mesh estimates per-face mouth openness as a proxy for speaking.
- Faces are tracked across time; the dominant speaker drives the crop.
- A smooth pan/zoom path keeps the person centered with proportional framing (no jitters).
 - Will temporarily switch to 2–3 tiles when multiple faces are active, or show full view with blurred bars when appropriate.

### Auto captions (burned-in)

Captions are burned-in by default:

- Color: yellow by default (customizable)
- Outline: black stroke (thicker for readability)
- Placement: ~25% from bottom, bottom-center aligned
- Padding: left/right margins (~7% width) and automatic word-wrapping to avoid edge overlap

Change caption color (named or hex):

```bash
python app.py --url "/path/to/local_video.mp4" --aspect 9:16 --focus-speaker \
  --caption-color yellow         # or white/red/green/blue/cyan/magenta/black or #RRGGBB
```

Notes:
- If a transcript is available, it’s used. If not, the tool attempts automatic transcription (Whisper) for the rendered clip.
- To disable captions entirely, pass `--no-captions` (to be added if desired) or I can wire this flag on request.

### CLI options

- `--duration` target short length in seconds (default: 45)
- `--top-k` number of clips to export (default: 3)
- `--pre-roll` seconds before the highlight window (default: 2)
- `--post-roll` seconds after the highlight window (default: 2)
- `--language-priority` transcript language order (default: `id,en`)
- `--dry-run` only print detected segments without downloading/cutting
 - `--aspect` output aspect: `original` or `9:16` (default: `9:16`)
 - `--max-tiles` cap stacked tiles (faces) in 9:16 mode, 1–3 (default: 3)
 - `--focus-speaker` enable speaker-focused auto-framing (smooth pan/zoom, adaptive layout)
 - `--caption-color` caption color (named: yellow/white/red/green/blue/cyan/magenta/black or hex `#RRGGBB`)
 - `--downloads` downloads directory (default: `./downloads`)
 - `--outdir` outputs directory (default: `./outputs`)
 - `--cookies-from-browser` use cookies from a local browser for restricted YouTube videos (e.g., `safari`, `chrome`)
 - `--cookies-file` path to a cookies.txt (Netscape) file for yt-dlp

## How it works

- Transcript is fetched via `youtube-transcript-api` (no API key). If unavailable, the tool falls back to an audio-based highlight detector; captions then rely on Whisper ASR.
- Each transcript line is scored via:
  - Laughter tokens (e.g., "[laughter]", "haha", "wkwk", "ngakak")
  - Emphasis/punctuation (! and ?)
  - Excitement keywords (Indo + EN)
  - Words-per-second spikes (delivery bursts)
  - Minor preference for concise lines
- The top entries form windows around their midpoints, expanded to match `--duration`, then de-duplicated to avoid overlaps.
 - If transcript is missing, an audio envelope peak detector proposes windows, then for captions the tool attempts Whisper ASR on the final rendered clip.

Output filenames:
- `short_{idx}_{start}-{end}.mp4` for original aspect
- `short_{idx}_{start}-{end}_v916.mp4` for 9:16 tiles mode
- `short_{idx}_{start}-{end}_v916_focus.mp4` for 9:16 speaker-focused mode
- When captions are burned, a `_cap` suffix is added (e.g., `_v916_focus_cap.mp4`).

## Limitations & Next steps

- Restricted/DRM YouTube videos may fail to download even with cookies; local files are recommended.
- Audio-only fallback is used when transcripts are missing; results vary by content.
- Optional burn-in subtitles could be added using transcript + `ffmpeg drawtext`.
- Further refinements possible: active speaker via audio-visual sync, crossfades between layout changes.

## License

MIT
