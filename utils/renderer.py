from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips, vfx
import mediapipe as mp


@dataclass
class FaceStat:
    cx: float
    cy: float
    w: float
    h: float
    count: int


def _detect_faces_in_frame(gray: np.ndarray, face_cascade) -> List[Tuple[int, int, int, int]]:
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def _sample_faces_mesh(video: VideoFileClip, start: float, end: float, sample_fps: int = 2):
    """Yield samples with timestamp and per-face bbox + expression metric using MediaPipe FaceMesh.

    Returns list of dicts: {"t": float, "faces": [{"bbox": (x1,y1,x2,y2), "expr": float}], "count": int}
    """
    mp_mesh = mp.solutions.face_mesh
    results = []
    step = 1.0 / max(1, sample_fps)
    with mp_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as mesh:
        t = start
        while t <= end:
            frame = video.get_frame(t)
            h, w, _ = frame.shape
            rgb = frame  # already RGB from moviepy
            res = mesh.process(rgb)
            faces_info = []
            if res.multi_face_landmarks:
                for lm in res.multi_face_landmarks:
                    xs = [int(pt.x * w) for pt in lm.landmark]
                    ys = [int(pt.y * h) for pt in lm.landmark]
                    x1, x2 = max(0, min(xs)), min(w - 1, max(xs))
                    y1, y2 = max(0, min(ys)), min(h - 1, max(ys))
                    # Mouth openness: indices 13 (upper) and 14 (lower)
                    try:
                        y_upper = int(lm.landmark[13].y * h)
                        y_lower = int(lm.landmark[14].y * h)
                        mouth_open = abs(y_lower - y_upper) / max(1, (y2 - y1))
                    except Exception:
                        mouth_open = 0.0
                    faces_info.append({"bbox": (x1, y1, x2, y2), "expr": float(mouth_open)})
            results.append({"t": float(t), "faces": faces_info, "count": len(faces_info)})
            t += step
    return results


def _choose_tiles(faces: List[Tuple[int, int, int, int]], frame_w: int, frame_h: int, max_tiles: int) -> List[FaceStat]:
    if not faces:
        # fallback: center crops
        return [FaceStat(frame_w / 2, frame_h / 2, frame_w * 0.4, frame_h * 0.6, 1)]

    # Cluster by x-center bins to find up to max_tiles distinct speakers
    xs = [x + w / 2 for x, y, w, h in faces]
    ys = [y + h / 2 for x, y, w, h in faces]
    ws = [w for x, y, w, h in faces]
    hs = [h for x, y, w, h in faces]

    # K-means 1D on x centers with up to max_tiles clusters (simple greedy)
    centers = []
    for _ in range(max_tiles):
        if not xs:
            break
        median_x = float(np.median(xs))
        # collect cluster near median
        idxs = [i for i, val in enumerate(xs) if abs(val - median_x) < 0.2 * frame_w]
        if not idxs:
            break
        cx = float(np.mean([xs[i] for i in idxs]))
        cy = float(np.mean([ys[i] for i in idxs]))
        w_mean = float(np.mean([ws[i] for i in idxs]))
        h_mean = float(np.mean([hs[i] for i in idxs]))
        centers.append(FaceStat(cx, cy, w_mean, h_mean, len(idxs)))
        # remove clustered points
        xs = [v for i, v in enumerate(xs) if i not in idxs]
        ys = [v for i, v in enumerate(ys) if i not in idxs]
        ws = [v for i, v in enumerate(ws) if i not in idxs]
        hs = [v for i, v in enumerate(hs) if i not in idxs]

    if not centers:
        centers = [FaceStat(frame_w / 2, frame_h / 2, frame_w * 0.4, frame_h * 0.6, 1)]

    return centers[:max_tiles]


def _compute_crop_for_tile(face: FaceStat, src_w: int, src_h: int, tile_w: int, tile_h: int) -> Tuple[int, int, int, int]:
    # want 9:16 overall; each tile is stacked vertically
    # compute crop width that matches aspect of tile
    target_aspect = tile_w / tile_h
    crop_w = min(src_w, int(target_aspect * src_h))
    crop_h = int(crop_w / target_aspect)

    # center around face.cx keeping within bounds
    x1 = int(max(0, min(src_w - crop_w, face.cx - crop_w / 2)))
    y1 = int(max(0, min(src_h - crop_h, face.cy - crop_h / 2)))
    return x1, y1, x1 + crop_w, y1 + crop_h


def render_vertical_tiles(
    video_path: Path,
    window: Dict[str, float | str],
    out_path: Path,
    max_tiles: int = 2,
    out_w: int = 1080,
    out_h: int = 1920,
) -> Path:
    video_path = Path(video_path)
    out_path = Path(out_path)

    with VideoFileClip(str(video_path)) as clip:
        a = float(window["start"])  # seconds
        b = float(window["end"])    # seconds
        sub = clip.subclip(a, b)
        src_w, src_h = sub.size

        # Sample faces across the whole subclip using MediaPipe
        samples = _sample_faces_mesh(sub, 0.0, sub.duration, sample_fps=2)

        # Decide tile count per sample
        expr_thr = 0.06  # normalized mouth openness threshold
        def tiles_for_sample(s):
            cnt = s["count"]
            max_tiles_local = min(max_tiles, 3)
            # If strong expression from a primary face, prefer 1 tile focus
            max_expr = max([f["expr"] for f in s["faces"]], default=0.0)
            if max_expr >= expr_thr:
                return 1
            if cnt >= 3:
                return min(3, max_tiles_local)
            if cnt == 2:
                return min(2, max_tiles_local)
            return 1

        # Build segments where tile count is stable
        segs = []
        if samples:
            cur_tiles = tiles_for_sample(samples[0])
            seg_start = 0.0
            for s in samples[1:]:
                t_rel = max(0.0, min(sub.duration, float(s["t"]) - float(samples[0]["t"])) )
                t_tiles = tiles_for_sample(s)
                if t_tiles != cur_tiles:
                    segs.append((seg_start, t_rel, cur_tiles))
                    seg_start = t_rel
                    cur_tiles = t_tiles
            # close last
            segs.append((seg_start, sub.duration, cur_tiles))
        else:
            segs = [(0.0, sub.duration, 1)]

        # Merge very short segments (<1.5s)
        merged = []
        for (sa, sb, nt) in segs:
            if merged and (sb - sa) < 1.5:
                # extend previous
                pa, pb, pt = merged[-1]
                merged[-1] = (pa, sb, pt)
            else:
                merged.append((sa, sb, nt))

        # Build clips per segment with appropriate layout
        seg_clips = []
        for (sa, sb, nt) in merged:
            part = sub.subclip(sa, sb)
            # choose tiles based on nt and detected faces roughly at segment mid
            mid_t = (sa + sb) / 2.0
            # find sample closest to mid
            mid_sample = min(samples, key=lambda s: abs((s["t"] - samples[0]["t"]) - mid_t)) if samples else {"faces": []}
            # compute crops
            tile_h = out_h // nt
            tile_w = out_w
            crops = []
            if mid_sample.get("faces"):
                faces_sorted = sorted(mid_sample["faces"], key=lambda f: (f["bbox"][2]-f["bbox"][0])*(f["bbox"][3]-f["bbox"][1]), reverse=True)
                for f in faces_sorted[:nt]:
                    x1,y1,x2,y2 = f["bbox"]
                    # expand around face to match aspect tile_w:tile_h
                    face_w = x2-x1
                    face_h = y2-y1
                    target_ar = tile_w / tile_h
                    crop_w = int(max(face_w*1.6, min(part.w, target_ar*part.h)))
                    crop_h = int(crop_w / target_ar)
                    cx = (x1+x2)/2
                    cy = (y1+y2)/2
                    cx = max(crop_w/2, min(part.w - crop_w/2, cx))
                    cy = max(crop_h/2, min(part.h - crop_h/2, cy))
                    cx, cy = int(cx), int(cy)
                    x1c = int(cx - crop_w/2)
                    y1c = int(cy - crop_h/2)
                    x2c = x1c + crop_w
                    y2c = y1c + crop_h
                    crops.append((x1c,y1c,x2c,y2c))
            # fallback crops if not enough
            while len(crops) < nt:
                # center crop
                target_ar = tile_w / tile_h
                cw = min(part.w, int(part.h * target_ar))
                ch = int(cw / target_ar)
                x1c = (part.w - cw)//2
                y1c = (part.h - ch)//2
                crops.append((x1c,y1c,x1c+cw,y1c+ch))

            # compose vertically
            tiles = []
            y_acc = 0
            for (x1c,y1c,x2c,y2c) in crops:
                tile = part.crop(x1=x1c, y1=y1c, x2=x2c, y2=y2c).resize((tile_w, tile_h)).set_position((0, y_acc))
                tiles.append(tile)
                y_acc += tile_h
            comp = CompositeVideoClip(tiles, size=(out_w, out_h))
            seg_clips.append(comp)

        final = concatenate_videoclips(seg_clips, method="compose")
        final.write_videofile(
            str(out_path),
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=4,
            fps=30,
            temp_audiofile=str(out_path.with_suffix(".m4a")),
            remove_temp=True,
            verbose=False,
            logger=None,
        )

    return out_path


def _track_faces_over_time(samples: List[dict], dist_thr: float = 80.0) -> List[dict]:
    """Assign a stable track id to faces across samples using nearest-center matching.

    Mutates samples in-place: each face dict gets a 'track_id'. Returns the same samples.
    """
    next_id = 1
    tracks = {}  # id -> (cx, cy)
    for s in samples:
        for f in s.get("faces", []):
            x1,y1,x2,y2 = f["bbox"]
            cx = (x1+x2)/2.0
            cy = (y1+y2)/2.0
            # find nearest track
            best = None
            best_d = 1e9
            for tid, (tx,ty) in tracks.items():
                d = ((cx-tx)**2 + (cy-ty)**2) ** 0.5
                if d < best_d:
                    best_d = d
                    best = tid
            if best is None or best_d > dist_thr:
                tid = next_id; next_id += 1
            else:
                tid = best
            f["track_id"] = tid
            tracks[tid] = (cx, cy)
    return samples


def _segment_by_speaker(samples: List[dict], min_seg_s: float = 1.5) -> List[tuple]:
    """Return segments (start_t_rel, end_t_rel, speaker_track_id) where dominant speaker stays constant.
    Uses mouth openness as expression proxy to choose dominant speaker per sample.
    """
    if not samples:
        return []
    t0 = samples[0]["t"]
    def dominant_track(s):
        if not s.get("faces"):
            return None, 0.0
        f = max(s["faces"], key=lambda f: f.get("expr", 0.0))
        return f.get("track_id"), f.get("expr", 0.0)

    cur_id, _ = dominant_track(samples[0])
    seg_start = samples[0]["t"]
    segs = []
    for s in samples[1:]:
        tid, _ = dominant_track(s)
        if tid != cur_id:
            a = max(0.0, seg_start - t0)
            b = max(a, s["t"] - t0)
            if segs and (b - a) < min_seg_s:
                # merge short segment into previous
                pa, pb, pid = segs[-1]
                segs[-1] = (pa, b, pid)
            else:
                segs.append((a, b, cur_id))
            seg_start = s["t"]
            cur_id = tid
    # close
    a = max(0.0, seg_start - t0)
    b = max(a, samples[-1]["t"] - t0)
    if segs and (b - a) < min_seg_s:
        pa, pb, pid = segs[-1]
        segs[-1] = (pa, b, pid)
    else:
        segs.append((a, b, cur_id))
    # remove None speaker segments by merging into neighbors
    cleaned = []
    for (sa,sb,tid) in segs:
        if tid is None:
            if cleaned:
                pa,pb,pt = cleaned[-1]
                cleaned[-1] = (pa, sb, pt)
            else:
                # keep as single speakerless segment
                cleaned.append((sa,sb,tid))
        else:
            cleaned.append((sa,sb,tid))
    return cleaned


def render_speaker_focus(
    video_path: Path,
    window: Dict[str, float | str],
    out_path: Path,
    out_w: int = 1080,
    out_h: int = 1920,
) -> Path:
    """Render a single-tile 9:16 clip that follows the dominant speaker with smooth camera-like motion.

    Implementation details:
    - Uses MediaPipe face mesh to estimate mouth-openness per face as a proxy for speaking.
    - Tracks faces over time to assign a stable track_id.
    - Builds segments where the dominant speaker is stable.
    - Within each segment, computes a smoothed pan/zoom path and applies a time-varying crop.
    """
    video_path = Path(video_path)
    out_path = Path(out_path)

    a = float(window["start"])  # seconds
    b = float(window["end"])    # seconds

    with VideoFileClip(str(video_path)) as clip:
        sub = clip.subclip(a, b)
        # Sample faces across subclip
        samples = _sample_faces_mesh(sub, 0.0, sub.duration, sample_fps=5)
        samples = _track_faces_over_time(samples)
        segs = _segment_by_speaker(samples, min_seg_s=1.5)

        seg_clips = []
        t0 = samples[0]["t"] if samples else 0.0
        for (sa, sb, tid) in segs:
            part = sub.subclip(sa, sb)
            H, W = part.h, part.w
            target_ar = out_w / out_h

            # Decide layout for this segment based on faces near mid and expressions
            def _layout_for_segment():
                if not samples:
                    return ("focus", 1)
                mid = (sa + sb) / 2.0
                mid_sample = min(samples, key=lambda s: abs((s["t"] - t0) - mid))
                cnt = mid_sample.get("count", 0)
                exprs = [f.get("expr", 0.0) for f in mid_sample.get("faces", [])]
                max_expr = max(exprs, default=0.0)
                if cnt >= 3 and max_expr >= 0.06:
                    return ("tiles", 3)
                if cnt == 2 and max_expr >= 0.06:
                    return ("tiles", 2)
                if cnt >= 2 and max_expr < 0.04:
                    return ("full", 1)
                return ("focus", 1)

            layout, ntiles = _layout_for_segment()

            # Full original with blurred background bars
            if layout == "full":
                # Background: cover then blur & darken
                scale_bg = max(out_w / W, out_h / H)
                bg = part.resize(scale_bg)
                x1b = max(0, (bg.w - out_w) // 2)
                y1b = max(0, (bg.h - out_h) // 2)
                bg = bg.crop(x1=x1b, y1=y1b, x2=x1b + out_w, y2=y1b + out_h)
                bg = bg.fx(vfx.gaussian_blur, sigma=25).fx(vfx.colorx, 0.6)
                # Foreground: contain and center
                scale_fg = min(out_w / W, out_h / H)
                fg = part.resize(scale_fg)
                fg = fg.set_position(((out_w - fg.w)//2, (out_h - fg.h)//2))
                comp = CompositeVideoClip([bg, fg], size=(out_w, out_h))
                seg_clips.append(comp)
                continue

            # Tiles layout for 2–3 faces
            if layout == "tiles":
                tile_h = out_h // ntiles
                tile_w = out_w
                tile_ar = tile_w / tile_h
                mid = (sa + sb) / 2.0
                mid_sample = min(samples, key=lambda s: abs((s["t"] - t0) - mid)) if samples else {"faces": []}
                crops = []
                faces_sorted = sorted(mid_sample.get("faces", []), key=lambda f: (f["bbox"][2]-f["bbox"][0])*(f["bbox"][3]-f["bbox"][1]), reverse=True)
                for f in faces_sorted[:ntiles]:
                    x1,y1,x2,y2 = f["bbox"]
                    fw, fh = (x2-x1), (y2-y1)
                    cw = int(max(fw * 1.6, min(W, tile_ar * H)))
                    ch = int(cw / tile_ar)
                    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
                    cx = max(cw/2, min(W - cw/2, cx))
                    cy = max(ch/2, min(H - ch/2, cy))
                    x1c, y1c = int(cx - cw/2), int(cy - ch/2)
                    crops.append((x1c, y1c, x1c + cw, y1c + ch))
                while len(crops) < ntiles:
                    cw = min(W, int(H * tile_ar))
                    ch = int(cw / tile_ar)
                    x1c = (W - cw)//2
                    y1c = (H - ch)//2
                    crops.append((x1c, y1c, x1c + cw, y1c + ch))
                tiles = []
                y_acc = 0
                for (x1c,y1c,x2c,y2c) in crops:
                    tile = part.crop(x1=x1c, y1=y1c, x2=x2c, y2=y2c).resize((tile_w, tile_h)).set_position((0, y_acc))
                    tiles.append(tile)
                    y_acc += tile_h
                comp = CompositeVideoClip(tiles, size=(out_w, out_h))
                seg_clips.append(comp)
                continue

            # Build keyframes for this segment: list of (t_rel, cx, cy, cw, ch)
            keyframes = []
            for s in samples:
                t_rel = s["t"] - t0
                if t_rel < sa or t_rel > sb:
                    continue
                # Choose face for this segment
                face = None
                if tid is not None:
                    for f in s.get("faces", []):
                        if f.get("track_id") == tid:
                            face = f
                            break
                if face is None and s.get("faces"):
                    face = max(s["faces"], key=lambda f: f.get("expr", 0.0))
                if face is None:
                    continue
                x1,y1,x2,y2 = face["bbox"]
                fw, fh = (x2-x1), (y2-y1)
                cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
                # Desired crop width from face size (keeps some headroom)
                base_w = max(fw * 2.0, min(W, target_ar * H))
                base_h = base_w / target_ar
                base_w = min(base_w, W)
                base_h = min(base_h, H)
                # Clamp center to keep crop within bounds
                cx = max(base_w/2, min(W - base_w/2, cx))
                cy = max(base_h/2, min(H - base_h/2, cy))
                keyframes.append((t_rel - sa, cx, cy, base_w, base_h))

            # Fallback keyframes if none were detected
            if not keyframes:
                cw = min(W, int(H * target_ar))
                ch = int(cw / target_ar)
                cx = W / 2.0
                cy = H / 2.0
                keyframes = [(0.0, cx, cy, float(cw), float(ch)), (part.duration, cx, cy, float(cw), float(ch))]

            # Sort and smooth keyframes with EMA to reduce jitter
            keyframes.sort(key=lambda k: k[0])
            alpha = 0.2
            smoothed = []
            sx = sy = sw = sh = None
            for (tr, cx, cy, cw, ch) in keyframes:
                if sx is None:
                    sx, sy, sw, sh = cx, cy, cw, ch
                else:
                    sx = alpha * cx + (1 - alpha) * sx
                    sy = alpha * cy + (1 - alpha) * sy
                    sw = alpha * cw + (1 - alpha) * sw
                    sh = alpha * ch + (1 - alpha) * sh
                smoothed.append((tr, sx, sy, sw, sh))

            # Deadband and velocity/zoom clamp to avoid "lari-lari"
            if len(smoothed) >= 2:
                max_speed = 240.0  # px per second
                max_zoom_per_s = 0.18 * W  # allowed width change per second
                deadband = 12.0  # pixels
                clamped = [smoothed[0]]
                for i in range(1, len(smoothed)):
                    t_prev, cx0, cy0, cw0, ch0 = clamped[-1]
                    t_cur, cx1, cy1, cw1, ch1 = smoothed[i]
                    dt = max(1e-3, t_cur - t_prev)
                    # deadband
                    if abs(cx1 - cx0) < deadband:
                        cx1 = cx0
                    if abs(cy1 - cy0) < deadband:
                        cy1 = cy0
                    if abs(cw1 - cw0) < deadband:
                        cw1 = cw0
                    if abs(ch1 - ch0) < deadband:
                        ch1 = ch0
                    # clamp speeds
                    dx = cx1 - cx0
                    dy = cy1 - cy0
                    dw = cw1 - cw0
                    maxd = max_speed * dt
                    if abs(dx) > maxd:
                        cx1 = cx0 + np.sign(dx) * maxd
                    if abs(dy) > maxd:
                        cy1 = cy0 + np.sign(dy) * maxd
                    maxzw = max_zoom_per_s * dt
                    if abs(dw) > maxzw:
                        cw1 = cw0 + np.sign(dw) * maxzw
                        ch1 = cw1 / target_ar
                    clamped.append((t_cur, cx1, cy1, cw1, ch1))
                smoothed = clamped

            # Helper: interpolate between smoothed keyframes
            def interp(tr: float):
                if tr <= smoothed[0][0]:
                    _, cx, cy, cw, ch = smoothed[0]
                    return cx, cy, cw, ch
                if tr >= smoothed[-1][0]:
                    _, cx, cy, cw, ch = smoothed[-1]
                    return cx, cy, cw, ch
                # find two neighbors
                for i in range(1, len(smoothed)):
                    if tr <= smoothed[i][0]:
                        t0k, cx0, cy0, cw0, ch0 = smoothed[i-1]
                        t1k, cx1, cy1, cw1, ch1 = smoothed[i]
                        u = (tr - t0k) / max(1e-6, (t1k - t0k))
                        cx = cx0 + u * (cx1 - cx0)
                        cy = cy0 + u * (cy1 - cy0)
                        cw = cw0 + u * (cw1 - cw0)
                        ch = ch0 + u * (ch1 - ch0)
                        return cx, cy, cw, ch
                # fallback last
                _, cx, cy, cw, ch = smoothed[-1]
                return cx, cy, cw, ch

            # Define dynamic crop via frame-level function (MoviePy crop doesn't accept callables in this version)
            def _dyn_frame(gf, t):
                cx, cy, cw, ch = interp(t)
                x1 = int(max(0, min(W - cw, cx - cw/2)))
                y1 = int(max(0, min(H - ch, cy - ch/2)))
                x2 = int(min(W, x1 + cw))
                y2 = int(min(H, y1 + ch))
                frame = gf(t)
                return frame[y1:y2, x1:x2]

            dynamic_crop = part.fl(_dyn_frame, apply_to=["mask"]).resize((out_w, out_h))
            seg_clips.append(dynamic_crop)

        final = concatenate_videoclips(seg_clips, method="compose")
        final.write_videofile(
            str(out_path),
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=4,
            fps=30,
            temp_audiofile=str(out_path.with_suffix(".m4a")),
            remove_temp=True,
            verbose=False,
            logger=None,
        )
    return out_path
