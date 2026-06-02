"""Phase 5 — quick inference test of the fine-tuned slapper adapter.

Loads base Typhoon2-3B (4-bit) + the trained LoRA and generates replies to a few
prompts, so we can eyeball: (a) did it pick up slapper's style/catchphrases, and
(b) is it still coherent (not looping/garbled = not overfit).

    D:\\Code\\.venv\\Scripts\\python.exe scripts\\test_infer.py
    ... --base   # compare against the untuned base model
"""
import argparse
import os
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import unsloth  # noqa: E402
from unsloth import FastLanguageModel  # noqa: E402
from unsloth.chat_templates import get_chat_template  # noqa: E402
from pathlib import Path  # noqa: E402

BASE = "scb10x/llama3.2-typhoon2-3b-instruct"
ADAPTER = Path(__file__).resolve().parent.parent / "typhoon2_slapper_lora"

PROMPTS = [
    "เล่าให้ฟังหน่อยว่าวันนี้เล่นอะไรมา",
    "เป็นไงบ้างเกมนี้ สนุกไหม",
    "เจอบอสยากๆ ทำยังไง",
    "ขายของได้กำไรเท่าไหร่",
]


def main(use_base: bool) -> None:
    model_name = BASE if use_base else str(ADAPTER)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=512, load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(model)

    print(f"\n=== {'BASE (untuned)' if use_base else 'SLAPPER LoRA'} ===\n")
    for q in PROMPTS:
        msgs = [{"role": "user", "content": q}]
        inputs = tokenizer.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True).to("cuda")
        out = model.generate(input_ids=inputs, max_new_tokens=200,
                             temperature=0.8, top_p=0.9, do_sample=True,
                             repetition_penalty=1.3, no_repeat_ngram_size=3)
        reply = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        print(f"Q: {q}\nA: {reply.strip()}\n{'-'*60}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true", help="run the untuned base model instead")
    main(ap.parse_args().base)
