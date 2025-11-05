from __future__ import annotations
from pathlib import Path
from typing import List, Dict

import numpy as np
from moviepy.editor import VideoFileClip


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    win = min(win, len(x))
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(x, kernel, mode="same")


def _find_peaks(signal: np.ndarray, min_distance: int, top_k: int) -> List[int]:
    # Threshold at mean + 1 std
    mu = float(signal.mean())
    sigma = float(signal.std())
    thr = mu + 1.0 * sigma
    candidates = np.where(signal >= thr)[0]
    # Non-maximum suppression by min_distance
    peaks: List[int] = []
    for idx in candidates:
        if peaks and idx - peaks[-1] < min_distance:
            # keep the higher peak within the neighborhood
            if signal[idx] > signal[peaks[-1]]:
                peaks[-1] = idx
            continue
        peaks.append(idx)
    # If not enough, take global top
    if len(peaks) < top_k:
        order = np.argsort(signal)[::-1]
        for idx in order:
            if all(abs(int(idx) - p) >= min_distance for p in peaks):
                peaks.append(int(idx))
            if len(peaks) >= top_k:
                break
    # Sort by signal height desc, then trim
    peaks = sorted(peaks, key=lambda i: -signal[i])[:top_k]
    # Sort by index for time order
    peaks.sort()
    return peaks


def select_audio_highlight_windows(
    video_path: Path,
    target_duration: int = 45,
    top_k: int = 3,
    envelope_hz: int = 25,
) -> List[Dict[str, float | str]]:
    """Audio-based highlight detector.

    Builds an amplitude envelope at envelope_hz, smooths it, and finds salient peaks.
    Returns non-overlapping time windows centered around peaks.
    """
    video_path = Path(video_path)
    assert video_path.exists(), f"Video not found: {video_path}"

    with VideoFileClip(str(video_path)) as clip:
        duration = float(clip.duration)
        audio = clip.audio
        if audio is None:
            return []

        # Iterate frames to build amplitude envelope at low rate
        step = 1.0 / envelope_hz
        t = 0.0
        env_vals: List[float] = []
        # iter_frames returns chunks of shape (n, channels). Set fps to sampling frequency; we want envelope_hz samples per second
        for frame in audio.iter_frames(fps=envelope_hz, dtype="float32"):
            # frame is a small array of samples at this fps; compute mean absolute amplitude
            if frame.ndim == 1:
                amp = float(np.mean(np.abs(frame)))
            else:
                amp = float(np.mean(np.abs(frame), axis=None))
            env_vals.append(amp)
            t += step

    env = np.asarray(env_vals, dtype=float)
    if env.size == 0:
        return []

    # Normalize and smooth
    env = (env - env.min()) / (np.ptp(env) + 1e-9)
    smoothed = _smooth(env, win=max(3, envelope_hz // 2))

    # Peak picking with min distance roughly target_duration seconds
    min_dist_frames = max(1, int(envelope_hz * (target_duration * 0.8)))
    peaks = _find_peaks(smoothed, min_distance=min_dist_frames, top_k=top_k)

    # Convert peak indices to windows
    windows: List[Dict[str, float | str]] = []
    for i in peaks:
        center_time = i / float(envelope_hz)
        a = max(0.0, center_time - target_duration / 2)
        b = min(duration, center_time + target_duration / 2)
        if b - a < target_duration:
            # expand as possible
            diff = target_duration - (b - a)
            a = max(0.0, a - diff)
            b = min(duration, b + diff)
        windows.append(
            {
                "start": round(a, 2),
                "end": round(b, 2),
                "duration": round(b - a, 2),
                "score": round(float(smoothed[i]), 3),
                "label": "audio-peak",
            }
        )

    # Deduplicate overlaps by score
    windows = sorted(windows, key=lambda w: -w["score"])[: top_k * 2]
    kept: List[Dict[str, float | str]] = []
    for w in windows:
        if any(not (w["end"] <= k["start"] or w["start"] >= k["end"]) for k in kept):
            continue
        kept.append(w)
        if len(kept) >= top_k:
            break

    kept.sort(key=lambda w: w["start"])  # time order
    return kept
