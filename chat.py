"""Simple chat with the fine-tuned slapper model (input -> output).

    # interactive: type a message, get a reply; blank line or 'q' to quit
    D:\\Code\\.venv\\Scripts\\python.exe chat.py

    # one-shot
    D:\\Code\\.venv\\Scripts\\python.exe chat.py -p "เล่าหน่อยวันนี้เล่นอะไร"

    # use the untuned base model instead (to compare)
    D:\\Code\\.venv\\Scripts\\python.exe chat.py --base
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
ADAPTER = Path(__file__).resolve().parent / "typhoon2_slapper_lora"


def load(use_base: bool):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE if use_base else str(ADAPTER),
        max_seq_length=512, load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def reply(model, tokenizer, text: str) -> str:
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        return_tensors="pt", add_generation_prompt=True).to("cuda")
    out = model.generate(input_ids=inputs, max_new_tokens=200,
                         temperature=0.8, top_p=0.9, do_sample=True,
                         repetition_penalty=1.3, no_repeat_ngram_size=3)
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prompt", help="one-shot prompt; omit for interactive mode")
    ap.add_argument("--base", action="store_true", help="use untuned base model")
    a = ap.parse_args()

    model, tokenizer = load(a.base)
    tag = "base" if a.base else "slapper"

    if a.prompt:
        print(reply(model, tokenizer, a.prompt))
        return

    print(f"\n[{tag}] พิมพ์ข้อความแล้ว Enter (บรรทัดว่างหรือ q เพื่อออก)\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() == "q":
            break
        print(f"slapper> {reply(model, tokenizer, text)}\n")


if __name__ == "__main__":
    main()
