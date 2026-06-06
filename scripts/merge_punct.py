"""Phase 2b (manual) — merge hand-punctuated chunks into the cache, reassemble.

Companion to restore_punct.py for when we punctuate chunks ourselves instead of
calling Gemini. Reads data/punct_work/{manifest.json, in/NN.txt, out/NN.txt},
validates every out/ against its in/ with the SAME verbatim+spacing guard the
Gemini path uses, injects the valid ones into data/punct_cache.json (keyed by the
identical content hash), then reassembles every fully-cached clip into
data/punctuated_transcripts.json.

Safe + idempotent: invalid or missing out/ files are skipped (left pending);
nothing already cached is overwritten.

    D:\\Code\\.venv\\Scripts\\python.exe scripts\\merge_punct.py
"""
import hashlib
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
IN = DATA / "cleaned_transcripts.json"
OUT = DATA / "punctuated_transcripts.json"
CACHE = DATA / "punct_cache.json"
WORK = DATA / "punct_work"
CHUNK_CHARS = 3000


def chunk_on_spaces(text: str, size: int) -> list[str]:
    words = text.split(" ")
    chunks, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > size:
            chunks.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        chunks.append(cur)
    return chunks


def _bare(s: str) -> str:
    return re.sub(r"[\s\.\?\!\,…]", "", s)


def hsh(chunk: str) -> str:
    return hashlib.sha1(chunk.encode("utf-8")).hexdigest()


def valid(chunk: str, r: str) -> tuple[bool, str]:
    if not r:
        return False, "empty"
    a, b = _bare(chunk), _bare(r)
    ratio = len(b) / max(len(a), 1)
    if not (0.85 <= ratio <= 1.15):
        return False, f"verbatim ratio {ratio:.3f} (need 0.85-1.15)"
    space = r.count(" ") / max(len(r), 1)
    if space > 0.12:
        return False, f"space ratio {space:.3f} (need <=0.12)"
    return True, "ok"


def main() -> None:
    cache: dict[str, str] = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    manifest = json.loads((WORK / "manifest.json").read_text(encoding="utf-8"))

    merged = skipped = 0
    for item in manifest:
        n, h = item["n"], item["hash"]
        if h in cache:
            continue
        out_f = WORK / "out" / f"{n:02}.txt"
        in_f = WORK / "in" / f"{n:02}.txt"
        if not out_f.exists():
            skipped += 1
            continue
        raw = in_f.read_text(encoding="utf-8")
        if hsh(raw) != h:
            print(f"  [{n:02}] in/ hash mismatch — skip"); skipped += 1; continue
        res = out_f.read_text(encoding="utf-8").strip()
        ok, why = valid(raw, res)
        if not ok:
            print(f"  [{n:02}] INVALID: {why} — left pending"); skipped += 1; continue
        cache[h] = res
        merged += 1
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"merged {merged} chunk(s), skipped {skipped}.")

    # reassemble every fully-cached clip
    clips = json.loads(IN.read_text(encoding="utf-8"))
    out, remaining = [], 0
    for clip in clips:
        chs = chunk_on_spaces(clip["text"], CHUNK_CHARS)
        if all(hsh(c) in cache for c in chs):
            joined = "\n".join(cache[hsh(c)] for c in chs)
            out.append({
                **{k: clip[k] for k in ("video_id", "title", "duration")},
                "n_chars": len(joined),
                "text": joined,
            })
        else:
            remaining += sum(1 for c in chs if hsh(c) not in cache)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(out)}/{len(clips)} clips fully punctuated -> {OUT}")
    if remaining:
        print(f"~{remaining} chunks still pending.")
    else:
        print("ALL clips done.")


if __name__ == "__main__":
    main()
