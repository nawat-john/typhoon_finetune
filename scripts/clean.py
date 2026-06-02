"""Phase 2a — local regex cleaning of raw transcripts (no LLM, free).

Reads data/raw_transcripts.json, strips auto-caption noise, writes
data/cleaned_transcripts.json (same structure, cleaned `text`).

Run:
    D:\\Code\\.venv\\Scripts\\python.exe scripts\\clean.py
"""
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "raw_transcripts.json"
OUT = DATA / "cleaned_transcripts.json"


def clean_text(text: str) -> str:
    text = re.sub(r"\[.*?\]", " ", text)        # [Music], [ __ ] (censored swear), [เสียง...]
    text = re.sub(r"\(.*?\)", " ", text)        # parentheticals
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)   # collapse runaway repeated chars (มากกกกก -> มากก)
    text = re.sub(r"\s+", " ", text).strip()    # normalize whitespace
    return text


def main() -> None:
    data = json.loads(RAW.read_text(encoding="utf-8"))
    out = []
    for d in data:
        cleaned = clean_text(d["text"])
        out.append({
            **{k: d[k] for k in ("video_id", "title", "duration")},
            "n_chars_raw": d["n_chars"],
            "n_chars": len(cleaned),
            "text": cleaned,
        })

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    raw_total = sum(d["n_chars_raw"] for d in out)
    new_total = sum(d["n_chars"] for d in out)
    print(f"cleaned {len(out)} clips")
    print(f"chars: {raw_total} -> {new_total} ({100*(raw_total-new_total)/raw_total:.1f}% removed)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
