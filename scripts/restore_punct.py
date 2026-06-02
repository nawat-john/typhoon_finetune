"""Phase 2b — restore Thai punctuation / sentence segmentation with Gemini.

Auto-captions arrive as one long run of word-boundary-spaced Thai with no
punctuation. This re-spaces it into natural written Thai, segments into
utterances (one per line), and adds minimal punctuation (? for questions) —
WITHOUT changing the words.

- Chunks each clip on space boundaries (~CHUNK_CHARS).
- Caches every chunk by content hash in data/punct_cache.json → fully resumable
  and idempotent; safe to Ctrl-C and re-run.
- Rate-limit safe: min interval between calls + exponential backoff on 429/5xx.
- Verbatim guard: if the model's output drops/adds too many characters vs input,
  keep the original chunk instead.

Quota: free tier = 20 requests/day/model. We multiply available quota two ways:
  1. cycle several models (each its own 20/day),
  2. support several API keys (each its own per-model quota).
Total daily capacity ~= n_keys * n_models * 20. Plus big chunks to cut call count.

Keys come from .env: GEMINI_API_KEY (may be comma-separated for several),
and/or GEMINI_API_KEY_2, GEMINI_API_KEY_3, ...

Run:
    D:\\Code\\.venv\\Scripts\\python.exe scripts\\restore_punct.py
"""
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

DATA = Path(__file__).resolve().parent.parent / "data"
IN = DATA / "cleaned_transcripts.json"
OUT = DATA / "punctuated_transcripts.json"
CACHE = DATA / "punct_cache.json"

# Kept at 3000 on purpose: it preserves the existing hash cache (clips already
# done stay done) AND keeps output under the 8192-token cap the 2.0 models
# enforce. Extra daily capacity now comes from multiple keys x models, not from
# bigger chunks — changing this value would invalidate the cache and re-spend
# quota re-doing finished clips.
CHUNK_CHARS = 3000
MAX_RETRIES = 6

SYSTEM = (
    "คุณคือเครื่องมือจัดรูปแบบข้อความถอดเสียงภาษาไทย (auto-caption) ให้เป็นข้อความเขียนที่อ่านง่าย "
    "หน้าที่ของคุณคือ:\n"
    "1) เว้นวรรคให้เป็นธรรมชาติแบบภาษาไทยที่เขียนถูกต้อง (ลบช่องว่างที่คั่นทีละคำแบบผิดธรรมชาติออก)\n"
    "2) ตัดเป็นประโยค/ช่วงพูด บรรทัดละ 1 ประโยค\n"
    "3) เติมเครื่องหมายเท่าที่จำเป็น: ? สำหรับคำถาม, ! สำหรับตกใจ/เน้น, ... สำหรับพูดค้าง\n"
    "ข้อห้ามเด็ดขาด: ห้ามเปลี่ยน/เพิ่ม/ตัดคำพูดเดิม ห้ามแปล ห้ามสรุป ห้ามแต่งเติมเนื้อหา "
    "ใช้คำเดิมทุกคำ เพียงแค่จัดวรรคตอนและขึ้นบรรทัดใหม่ "
    "ตอบกลับเฉพาะข้อความที่จัดรูปแบบแล้วเท่านั้น ห้ามมีคำอธิบายหรือคำนำใดๆ"
)


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
    """Strip whitespace + punctuation for verbatim comparison."""
    return re.sub(r"[\s\.\?\!\,…]", "", s)


# Free tier = 20 requests/day/MODEL. We cycle through several models (each has
# its own quota) and stop gracefully when all are exhausted, emitting whatever
# clips are fully done. Re-run another day (cache resumes) to finish the rest.
DEFAULT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-pro",
]


class DailyQuotaHit(Exception):
    """The per-day free-tier request quota for a given model is used up."""


class ModelUnavailable(Exception):
    """Model not found / not accessible on this key."""


def _is_daily_quota(e) -> bool:
    details = getattr(e, "details", None)
    if isinstance(details, dict):
        for d in details.get("error", {}).get("details", []):
            for v in d.get("violations", []):
                if "PerDay" in v.get("quotaId", ""):
                    return True
    return "PerDay" in str(e) or "per day" in str(e).lower()


def call_gemini(client, model: str, chunk: str) -> str:
    cfg = types.GenerateContentConfig(
        temperature=0.1, system_instruction=SYSTEM, max_output_tokens=8192,
    )
    delay = 4.0
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(model=model, contents=chunk, config=cfg)
            return (resp.text or "").strip()
        except genai_errors.ClientError as e:
            code = getattr(e, "code", None)
            if code == 429 and _is_daily_quota(e):
                raise DailyQuotaHit(model)
            if code in (404,) or "NOT_FOUND" in str(e):
                raise ModelUnavailable(model)
            if code == 429:                       # per-minute / transient
                wait = delay * (2 ** attempt)
                print(f"    rpm/err 429 on {model}, backoff {wait:.0f}s")
                time.sleep(wait)
                continue
            raise
        except genai_errors.ServerError:
            wait = delay * (2 ** attempt)
            print(f"    5xx on {model}, backoff {wait:.0f}s")
            time.sleep(wait)
            continue
    raise RuntimeError("max retries exceeded")


def collect_keys() -> list[str]:
    """Gather API keys: GEMINI_API_KEY (comma-sep ok) + GEMINI_API_KEY_2/3/..."""
    keys, seen = [], set()
    def add(raw: str):
        for k in (raw or "").split(","):
            k = k.strip()
            if k and k not in seen:
                seen.add(k)
                keys.append(k)
    add(os.getenv("GEMINI_API_KEY", ""))
    for i in range(2, 12):
        for name in (f"GEMINI_API_KEY_{i}", f"GEMINI_API_KEY{i}"):
            add(os.getenv(name, ""))
    return keys


def main(models: list[str], interval: float, limit_clips: int) -> None:
    load_dotenv(DATA.parent / ".env")
    keys = collect_keys()
    if not keys:
        raise SystemExit("no GEMINI_API_KEY found in .env")
    clients = [genai.Client(api_key=k) for k in keys]
    # work "slots" = every (key, model) pair; each pair has its own 20/day quota
    slots = [(ci, m) for ci in range(len(clients)) for m in models]
    print(f"{len(keys)} key(s) x {len(models)} models = {len(slots)} slots "
          f"(~{len(slots)*20} requests/day capacity)")

    clips = json.loads(IN.read_text(encoding="utf-8"))
    if limit_clips:
        clips = clips[:limit_clips]
    cache: dict[str, str] = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

    def hsh(chunk: str) -> str:
        return hashlib.sha1(chunk.encode("utf-8")).hexdigest()

    # ---- Phase A: fill the cache as far as today's quota allows ----
    exhausted: set = set()      # slot indices whose daily quota is gone / unavailable
    stats = {"calls": 0}

    def valid(chunk: str, r: str) -> bool:
        """Non-empty, words preserved (verbatim ratio), and not a raw echo."""
        if not r:
            return False
        a, b = _bare(chunk), _bare(r)
        if not (0.85 <= len(b) / max(len(a), 1) <= 1.15):
            return False                                   # words drifted
        return r.count(" ") / max(len(r), 1) <= 0.12       # natural Thai, not raw caption

    def punctuate(chunk: str) -> str | None:
        """Try slots until one returns a valid result; None if quota exhausted.
        Never caches/returns a poisoned fallback."""
        tried: set = set()
        while True:
            pick = next((j for j in range(len(slots))
                         if j not in exhausted and j not in tried), None)
            if pick is None:
                return None
            ci, model = slots[pick]
            try:
                r = call_gemini(clients[ci], model, chunk)
            except DailyQuotaHit:
                print(f"  daily quota used up on key#{ci+1}/{model} -> next slot")
                exhausted.add(pick)
                continue
            except ModelUnavailable:
                print(f"  key#{ci+1}/{model} unavailable -> skip")
                exhausted.add(pick)
                continue
            stats["calls"] += 1
            time.sleep(interval)
            if valid(chunk, r):
                return r
            print(f"    bad result on key#{ci+1}/{model}, retrying elsewhere")
            tried.add(pick)

    stop = False
    for clip in clips:
        if stop:
            break
        for chunk in chunk_on_spaces(clip["text"], CHUNK_CHARS):
            h = hsh(chunk)
            if h in cache:
                continue
            result = punctuate(chunk)
            if result is None:        # all slots exhausted for today
                stop = True
                break
            cache[h] = result
            CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    n_calls = stats["calls"]

    # ---- Phase B: emit every clip whose chunks are now ALL cached ----
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

    if not out and OUT.exists():
        print("\nno clips completed this run; keeping existing output untouched.")
    else:
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(out)}/{len(clips)} clips fully punctuated -> {OUT}")
    print(f"new API calls this run: {n_calls}")
    if remaining:
        print(f"~{remaining} chunks still pending (quota). Re-run another day to continue "
              f"(cache resumes).")
    else:
        print("ALL clips done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated model fallback order (each has its own 20/day quota)")
    ap.add_argument("--interval", type=float, default=4.0,
                    help="seconds between API calls (~15 RPM free tier)")
    ap.add_argument("--limit-clips", type=int, default=0,
                    help="process only first N clips (0 = all); use for a test run")
    a = ap.parse_args()
    main([m.strip() for m in a.models.split(",") if m.strip()], a.interval, a.limit_clips)
