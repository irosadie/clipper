#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from utils.scorer import select_highlight_windows
from utils.transcript import fetch_transcript, load_local_subtitle_transcript
from utils.downloader import download_video
from utils.cutter import cut_segments
from utils.renderer import render_vertical_tiles
from utils.audio_scorer import select_audio_highlight_windows


def main():
    parser = argparse.ArgumentParser(description="YouTube podcast clipper: detect highlights and export shorts")
    parser.add_argument("--url", required=True, help="YouTube URL or local video file path")
    parser.add_argument("--duration", type=int, default=45, help="Target duration of each short in seconds (default: 45)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of highlights to export (default: 3)")
    parser.add_argument("--downloads", default=str(Path("downloads").absolute()), help="Downloads directory")
    parser.add_argument("--outdir", default=str(Path("outputs").absolute()), help="Output directory for shorts")
    parser.add_argument("--pre-roll", type=float, default=2.0, help="Seconds before highlight center to start window")
    parser.add_argument("--post-roll", type=float, default=2.0, help="Seconds after window end to include")
    parser.add_argument("--language-priority", default="id,en", help="Comma list of transcript language codes priority order (default: id,en)")
    parser.add_argument("--dry-run", action="store_true", help="Only print detected segments; do not download/cut")
    parser.add_argument("--aspect", choices=["original", "9:16"], default="9:16", help="Output aspect mode (default: 9:16)")
    parser.add_argument("--max-tiles", type=int, default=3, help="Max stacked tiles (faces) in 9:16 mode (default: 3)")
    parser.add_argument("--focus-speaker", action="store_true", help="In 9:16 mode, follow the dominant speaker with a single centered crop that moves like a camera")
    parser.add_argument("--cookies-from-browser", help="Use cookies from local browser (e.g., 'safari', 'chrome') for restricted videos")
    parser.add_argument("--cookies-file", help="Path to a cookies.txt/Netscape file for yt-dlp")
    parser.add_argument("--captions", action="store_true", default=True, help="Burn auto captions (yellow with black border) ~25% from bottom")
    parser.add_argument("--caption-color", default="yellow", help="Caption color: named (yellow, white, red, green, blue, cyan, magenta, black) or hex like #RRGGBB")

    args = parser.parse_args()

    downloads = Path(args.downloads)
    outdir = Path(args.outdir)
    downloads.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    lang_order = [code.strip() for code in args.language_priority.split(",") if code.strip()]

    input_str = args.url
    input_path = Path(input_str)
    windows = []
    video_path: Path | None = None

    transcript_items = []
    if input_path.exists():
        print("Local file detected. Looking for sidecar subtitles…")
        transcript_items = load_local_subtitle_transcript(input_path) or []
        if transcript_items:
            print(f"Subtitle entries: {len(transcript_items)}. Scoring highlights…")
            windows = select_highlight_windows(
                transcript_items,
                target_duration=args.duration,
                top_k=args.top_k,
                pre_roll=args.pre_roll,
                post_roll=args.post_roll,
            )
        else:
            print("No subtitles found. Will analyze audio for highlights.")
        video_path = input_path
    else:
        print("Fetching transcript…")
        transcript_items = fetch_transcript(input_str, preferred_languages=lang_order) or []
        if transcript_items:
            print(f"Transcript entries: {len(transcript_items)}. Scoring highlights…")
            windows = select_highlight_windows(
                transcript_items,
                target_duration=args.duration,
                top_k=args.top_k,
                pre_roll=args.pre_roll,
                post_roll=args.post_roll,
            )
        else:
            print("No transcript available. Will use audio-based fallback after downloading the video.")

    print("Selected windows (seconds):")
    print(json.dumps(windows, indent=2))

    if args.dry_run and windows:
        return 0

    if video_path is None:
        print("Downloading source video…")
        video_path = download_video(
            args.url,
            downloads,
            cookiesfrombrowser=args.cookies_from_browser,
            cookiesfile=args.cookies_file,
        )

    if args.dry_run and not windows:
        print("Analyzing audio to detect highlight peaks (dry-run)…")
        windows = select_audio_highlight_windows(
            video_path,
            target_duration=args.duration,
            top_k=args.top_k,
        )
        print("Selected windows (seconds):")
        print(json.dumps(windows, indent=2))
        return 0

    if not windows:
        print("Analyzing audio to detect highlight peaks…")
        windows = select_audio_highlight_windows(
            video_path,
            target_duration=args.duration,
            top_k=args.top_k,
        )

    print(f"Cutting {len(windows)} segment(s) → {outdir}")
    outputs = []
    if args.aspect == "original":
        outputs = cut_segments(video_path, windows, outdir)
    else:
        # 9:16 vertical outputs
        if args.focus_speaker:
            for i, w in enumerate(windows, start=1):
                out = outdir / f"short_{i:02d}_{int(w['start'])}-{int(w['end'])}_v916_focus.mp4"
                from utils.renderer import render_speaker_focus
                render_speaker_focus(video_path, w, out)
                # optionally burn captions
                if args.captions:
                    try:
                        from utils.captions import generate_and_burn_captions
                        out_cap = out.with_name(out.stem + "_cap" + out.suffix)
                        out = generate_and_burn_captions(out, out_cap, w, transcript_items, out_w=1080, out_h=1920, caption_color=args.caption_color)
                    except Exception as e:
                        print("Captioning skipped due to error:", e)
                outputs.append(out)
        else:
            for i, w in enumerate(windows, start=1):
                out = outdir / f"short_{i:02d}_{int(w['start'])}-{int(w['end'])}_v916.mp4"
                from utils.renderer import render_vertical_tiles
                render_vertical_tiles(video_path, w, out, max_tiles=max(1, min(args.max_tiles, 3)))
                if args.captions:
                    try:
                        from utils.captions import generate_and_burn_captions
                        out_cap = out.with_name(out.stem + "_cap" + out.suffix)
                        out = generate_and_burn_captions(out, out_cap, w, transcript_items, out_w=1080, out_h=1920, caption_color=args.caption_color)
                    except Exception as e:
                        print("Captioning skipped due to error:", e)
                outputs.append(out)

    print("Exported files:")
    for p in outputs:
        print(" -", p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
