from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import math
import re

TranscriptItem = Dict[str, float | str]


@dataclass
class Window:
    start: float
    end: float
    score: float
    label: str


LAUGHTER_TOKENS = [
    "[laughter]",
    "(laughter)",
    "[laughs]",
    "lol",
    "haha",
    "hehe",
    "wkwk",
    "ngakak",
    "tertawa",
    "ketawa",
]

EMPHASIS_REGEX = re.compile(r"[!?]+")

POSITIVE_EXCITEMENT = {
    "lucu",
    "kocak",
    "epic",
    "gokil",
    "anjay",
    "anjir",
    "gila",
    "wow",
    "wtf",
    "omg",
    "punchline",
    "roasting",
    "jokes",
    "lucunya",
}

NEGATIVE_NOISE = {"[music]", "[applause]", "[aplaus]", "[musik]"}


def score_transcript_items(items: List[TranscriptItem]) -> List[float]:
    """Heuristic scoring per transcript item.

    Signals:
    - laughter tokens
    - emphasis via punctuation ! or ?
    - excitement keywords
    - words-per-second spikes (z-score)
    - uppercase token presence
    - downweight obvious noise markers
    """
    texts = [str(x.get("text", "")) for x in items]
    durations = [float(x.get("duration", 0.0)) or 0.001 for x in items]
    starts = [float(x.get("start", 0.0)) for x in items]

    # Words per second
    wps = []
    for t, d in zip(texts, durations):
        words = max(1, len(t.split()))
        wps.append(words / max(d, 0.001))

    mean = sum(wps) / len(wps)
    var = sum((x - mean) ** 2 for x in wps) / max(1, len(wps) - 1)
    std = math.sqrt(var) if var > 0 else 1.0

    scores: List[float] = []
    for i, (text, d, s, rate) in enumerate(zip(texts, durations, starts, wps)):
        base = 0.0
        lower = text.lower()

        # Laughter and excitement
        if any(tok in lower for tok in LAUGHTER_TOKENS):
            base += 4.0
        base += 0.8 * sum(1 for _ in EMPHASIS_REGEX.finditer(text))
        base += 1.0 * sum(1 for w in lower.split() if w.strip(".,!?") in POSITIVE_EXCITEMENT)
        if any(noise in lower for noise in NEGATIVE_NOISE):
            base -= 1.0

        # Words-per-second z-score
        z = (rate - mean) / std
        if z > 0.5:
            base += z  # boost faster bursts (punchline delivery)

        # Uppercase emphasis
        if any(tok.isupper() and len(tok) > 3 for tok in text.split()):
            base += 0.5

        # Slight preference for mid-length lines
        if 2.0 <= d <= 8.0:
            base += 0.2

        scores.append(base)

    return scores


def _window_from_index(items: List[TranscriptItem], idx: int, target_duration: int) -> Tuple[float, float]:
    # center around the selected item
    start_times = [float(x["start"]) for x in items]
    end_times = [float(x["start"]) + float(x["duration"]) for x in items]
    total_end = end_times[-1]

    mid = (start_times[idx] + end_times[idx]) / 2.0
    half = target_duration / 2.0
    a = max(0.0, mid - half)
    b = min(total_end, mid + half)
    # If short at edges, expand the other side
    if b - a < target_duration:
        diff = target_duration - (b - a)
        a = max(0.0, a - diff)
        b = min(total_end, b + diff)
    return (round(a, 2), round(b, 2))


def _dedupe_overlaps(windows: List[Window]) -> List[Window]:
    windows = sorted(windows, key=lambda w: (-w.score, w.start))
    kept: List[Window] = []
    for w in windows:
        if any(not (w.end <= k.start or w.start >= k.end) for k in kept):
            continue
        kept.append(w)
    # sort by start for output
    return sorted(kept, key=lambda w: w.start)


def select_highlight_windows(
    items: List[TranscriptItem],
    target_duration: int = 45,
    top_k: int = 3,
    pre_roll: float = 2.0,
    post_roll: float = 2.0,
) -> List[Dict[str, float | str]]:
    scores = score_transcript_items(items)

    candidates: List[Window] = []
    for idx, sc in enumerate(scores):
        if sc <= 0:
            continue
        a, b = _window_from_index(items, idx, target_duration)
        a = max(0.0, a - pre_roll)
        b = b + post_roll
        label = items[idx].get("text", "").strip().replace("\n", " ")[:80]
        candidates.append(Window(a, b, sc, label))

    if not candidates:
        # fallback: take evenly spaced windows
        start_times = [float(x["start"]) for x in items]
        end_times = [float(x["start"]) + float(x["duration"]) for x in items]
        total_end = end_times[-1] if end_times else 0.0
        step = max(1, top_k)
        approx_positions = [total_end * (i + 1) / (step + 1) for i in range(top_k)]
        candidates = [
            Window(max(0.0, p - target_duration / 2), min(total_end, p + target_duration / 2), 0.0, "highlight")
            for p in approx_positions
        ]

    selected = _dedupe_overlaps(candidates)[:top_k]
    return [
        {
            "start": round(w.start, 2),
            "end": round(w.end, 2),
            "duration": round(w.end - w.start, 2),
            "score": round(w.score, 3),
            "label": w.label,
        }
        for w in selected
    ]
