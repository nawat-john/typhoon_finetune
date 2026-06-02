"""Phase 1a — enumerate newest videos from the slapper channel.

Uses yt-dlp (flat extraction, no download) to list the channel's uploads,
newest first, and writes data/video_list.json.

Run:
    D:\\Code\\.venv\\Scripts\\python.exe scripts\\list_videos.py [--limit N]
"""
import argparse
import json
from pathlib import Path

import yt_dlp

CHANNEL_URL = "https://www.youtube.com/@slapperch/videos"
OUT = Path(__file__).resolve().parent.parent / "data" / "video_list.json"


def main(limit: int | None) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,   # don't resolve each video page — fast
        "skip_download": True,
    }
    if limit:
        opts["playlistend"] = limit

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(CHANNEL_URL, download=False)

    entries = info.get("entries") or []
    videos = []
    for e in entries:
        if not e or not e.get("id"):
            continue
        videos.append({
            "video_id": e["id"],
            "title": e.get("title"),
            "duration": e.get("duration"),   # seconds (may be None in flat mode)
        })

    OUT.write_text(json.dumps(videos, ensure_ascii=False, indent=2), encoding="utf-8")

    total_sec = sum(v["duration"] or 0 for v in videos)
    print(f"channel: {info.get('title')} ({info.get('channel_id')})")
    print(f"videos listed: {len(videos)}")
    print(f"total duration (where known): {total_sec/3600:.1f} h")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only list the newest N videos (default: all)")
    main(ap.parse_args().limit)
