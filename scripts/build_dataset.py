"""Phase 3 — build the QLoRA training set from punctuated transcripts.

Groups slapper's punctuated monologue into chat turns: each assistant reply is a
coherent ~target-sized span of his actual words; the user turn is a prompt.

Prompt source:
  --prompts template  (default, quota-free) random generic Thai conversational
                      prompt — good enough for STYLE cloning, runs offline.
  --prompts gemini    tailored question per reply via Gemini (better, but uses
                      the 20/day/project quota; not needed for a first run).

Output: data/dataset.jsonl, one {"messages":[user,assistant]} per line (Llama-3
chat template is applied by Unsloth at train time — don't pre-format here).

Run:
    D:\\Code\\.venv\\Scripts\\python.exe scripts\\build_dataset.py
    ... --target 450 --min 80 --val 0.05
"""
import argparse
import json
import random
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
IN = DATA / "punctuated_transcripts.json"
OUT = DATA / "dataset.jsonl"
VAL = DATA / "dataset.val.jsonl"

# Generic Thai prompts that plausibly precede a game-caster's monologue turn.
# Style transfer doesn't need a perfectly-matched question; variety is enough.
TEMPLATE_PROMPTS = [
    "เล่าให้ฟังหน่อยว่าเป็นยังไงบ้าง",
    "ตอนนี้ทำอะไรอยู่",
    "เกิดอะไรขึ้นบ้าง",
    "คิดยังไงกับเรื่องนี้",
    "เล่าต่อหน่อยสิ",
    "ว่าไงต่อ",
    "อธิบายให้ฟังหน่อย",
    "เป็นไงบ้างเกมนี้",
    "แล้วยังไงต่อ",
    "เล่าหน่อยว่าเล่นอะไรอยู่",
    "มีอะไรอัปเดตบ้าง",
    "เอาเลย เล่าให้ฟัง",
    "ตอนนี้สถานการณ์เป็นยังไง",
    "ทำไมถึงเป็นแบบนั้น",
    "เล่าให้ฟังหน่อยว่าวันนี้เป็นไง",
]

# split a long line on Thai/latin sentence enders, keeping the ender attached
_SENT = re.compile(r"[^.!?…]+[.!?…]*")


def split_long(line: str, hard: int) -> list[str]:
    if len(line) <= hard:
        return [line]
    pieces, buf = [], ""
    for m in _SENT.findall(line):
        if buf and len(buf) + len(m) > hard:
            pieces.append(buf.strip())
            buf = m
        else:
            buf += m
    if buf.strip():
        pieces.append(buf.strip())
    # still-too-long pieces (no punctuation): fall back to space / hard cut
    out = []
    for p in pieces:
        while len(p) > hard:
            cut = p.rfind(" ", 0, hard)
            cut = cut if cut > hard // 2 else hard
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out


def segments(text: str, target: int, hard: int, min_len: int) -> list[str]:
    pieces = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if ln:
            pieces.extend(split_long(ln, hard))

    segs, cur = [], ""
    for p in pieces:
        if cur and len(cur) + 1 + len(p) > target:
            segs.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}" if cur else p
    if cur:
        segs.append(cur)
    return [s for s in segs if len(s) >= min_len]


def main(target: int, hard: int, min_len: int, val_frac: float, seed: int) -> None:
    random.seed(seed)
    clips = json.loads(IN.read_text(encoding="utf-8"))

    rows = []
    for clip in clips:
        for seg in segments(clip["text"], target, hard, min_len):
            rows.append({"messages": [
                {"role": "user", "content": random.choice(TEMPLATE_PROMPTS)},
                {"role": "assistant", "content": seg},
            ]})

    random.shuffle(rows)
    n_val = int(len(rows) * val_frac)
    val, train = rows[:n_val], rows[n_val:]

    with OUT.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with VAL.open("w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lens = [len(r["messages"][1]["content"]) for r in rows]
    print(f"examples: {len(rows)} (train {len(train)}, val {len(val)})")
    print(f"assistant reply chars: min {min(lens)} / avg {sum(lens)//len(lens)} / max {max(lens)}")
    print(f"wrote {OUT} and {VAL}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=450, help="aim per assistant reply (chars)")
    ap.add_argument("--hard", type=int, default=600, help="hard max before force-splitting a line")
    ap.add_argument("--min", type=int, default=80, help="drop replies shorter than this")
    ap.add_argument("--val", type=float, default=0.05, help="validation fraction")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.target, a.hard, a.min, a.val, a.seed)
