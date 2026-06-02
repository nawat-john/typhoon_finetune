"""Phase 1b — fetch Thai transcripts for the listed videos.

Reads data/video_list.json (newest first), pulls the Thai transcript for each
via youtube-transcript-api, and accumulates into data/raw_transcripts.json until
a target volume is reached. Resumable: skips videos already saved.

Note: slapper's captions are auto-generated Thai (no punctuation) — that is
expected and handled later in Phase 2 cleaning.

Run:
    D:\\Code\\.venv\\Scripts\\python.exe scripts\\fetch_transcripts.py
    ... --target-hours 8 --max-clips 25
"""
import argparse
import json
import time
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

DATA = Path(__file__).resolve().parent.parent / "data"
LIST_IN = DATA / "video_list.json"
RAW_OUT = DATA / "raw_transcripts.json"

# Auto-captions run ~650 Thai chars per minute of speech; used to estimate hours.
CHARS_PER_HOUR = 650 * 60


def load_existing() -> dict:
    if RAW_OUT.exists():
        items = json.loads(RAW_OUT.read_text(encoding="utf-8"))
        return {it["video_id"]: it for it in items}
    return {}


def save(items: dict) -> None:
    # keep newest-first order, list form
    data = list(items.values())
    RAW_OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_one(ytt: YouTubeTranscriptApi, vid: str):
    """Return (text, n_snippets) or raise. Prefers manual th, falls back to auto th/en->th."""
    fetched = ytt.fetch(vid, languages=["th"])
    snips = list(fetched)
    text = " ".join(s.text for s in snips if s.text and s.text.strip())
    return text, len(snips)


def main(target_hours: float, max_clips: int, sleep_s: float) -> None:
    videos = json.loads(LIST_IN.read_text(encoding="utf-8"))
    existing = load_existing()
    ytt = YouTubeTranscriptApi()

    target_chars = int(target_hours * CHARS_PER_HOUR)
    total_chars = sum(it["n_chars"] for it in existing.values())
    n_have = len(existing)

    print(f"loaded {len(videos)} videos from list; already have {n_have} transcripts "
          f"({total_chars/CHARS_PER_HOUR:.1f} h)")
    print(f"target: ~{target_hours} h (~{target_chars} chars) or {max_clips} clips\n")

    skipped = []
    for v in videos:
        vid = v["video_id"]
        if n_have >= max_clips or total_chars >= target_chars:
            break
        if vid in existing:
            continue

        try:
            text, n_snips = fetch_one(ytt, vid)
        except (TranscriptsDisabled, NoTranscriptFound):
            skipped.append((vid, "no-th-transcript"))
            print(f"  skip {vid}: no Thai transcript")
            continue
        except VideoUnavailable:
            skipped.append((vid, "unavailable"))
            print(f"  skip {vid}: unavailable")
            continue
        except Exception as e:  # rate limit / network / parse — log and move on
            skipped.append((vid, f"error:{type(e).__name__}"))
            print(f"  skip {vid}: {type(e).__name__}: {e}")
            time.sleep(sleep_s * 3)
            continue

        n_chars = len(text)
        if n_chars < 500:  # near-empty / muted clip
            skipped.append((vid, "too-short"))
            print(f"  skip {vid}: too short ({n_chars} chars)")
            continue

        existing[vid] = {
            "video_id": vid,
            "title": v.get("title"),
            "duration": v.get("duration"),
            "n_snippets": n_snips,
            "n_chars": n_chars,
            "text": text,
        }
        total_chars += n_chars
        n_have += 1
        save(existing)  # incremental — safe to interrupt
        print(f"  ok   {vid}: {n_chars:>6} chars | running {total_chars/CHARS_PER_HOUR:5.1f} h "
              f"| {n_have} clips")
        time.sleep(sleep_s)

    print(f"\ndone. {n_have} transcripts, ~{total_chars/CHARS_PER_HOUR:.1f} h, "
          f"{total_chars} chars -> {RAW_OUT}")
    if skipped:
        print(f"skipped {len(skipped)}: " + ", ".join(f"{v}({r})" for v, r in skipped[:15])
              + (" ..." if len(skipped) > 15 else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-hours", type=float, default=8.0,
                    help="stop once accumulated speech reaches ~this many hours")
    ap.add_argument("--max-clips", type=int, default=30,
                    help="hard cap on number of clips")
    ap.add_argument("--sleep", type=float, default=1.5,
                    help="seconds between requests (politeness / anti-rate-limit)")
    a = ap.parse_args()
    main(a.target_hours, a.max_clips, a.sleep)
