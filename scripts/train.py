"""Phase 4 — QLoRA fine-tune Typhoon2-3B on slapper's style (RTX 3050, 4GB).

Config follows the 4GB plan: 4-bit base, LoRA r=8, gradient checkpointing,
paged 8-bit optimizer (spills optimizer state to system RAM).

    # smoke test: load + 2 optimizer steps, prove it fits in 4GB (no save)
    D:\\Code\\.venv\\Scripts\\python.exe scripts\\train.py --smoke
    # real run
    D:\\Code\\.venv\\Scripts\\python.exe scripts\\train.py --epochs 2
"""
import argparse
import os
import sys

# Unsloth prints emoji; the Thai Windows console (cp874) can't encode them.
# Force UTF-8 stdout/stderr so those prints don't crash training.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# must be set before torch/cuda init — lets CUDA spill to system RAM instead of OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import unsloth  # noqa: E402  (import first; it patches transformers/trl)
from unsloth import FastLanguageModel  # noqa: E402
from unsloth.chat_templates import get_chat_template  # noqa: E402
from datasets import load_dataset  # noqa: E402
from trl import SFTTrainer, SFTConfig  # noqa: E402

from pathlib import Path  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
MODEL = "scb10x/llama3.2-typhoon2-3b-instruct"
MAX_SEQ = 512
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
ADAPTER = Path(__file__).resolve().parent.parent / "typhoon2_slapper_lora"


def main(smoke: bool, epochs: float) -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_SEQ,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    train_ds = load_dataset("json", data_files=str(DATA / "dataset.jsonl"), split="train")

    # Unsloth's SFTTrainer needs a plain text field: render each chat with the
    # Llama-3 template (special tokens included as text).
    def to_text(ex):
        return {"text": tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False)}
    train_ds = train_ds.map(to_text, remove_columns=train_ds.column_names)

    cfg = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,      # effective batch 16
        warmup_steps=5,
        learning_rate=2e-4,
        bf16=True,
        optim="paged_adamw_8bit",            # spills optimizer state to RAM
        logging_steps=1 if smoke else 5,
        output_dir=str(OUT_DIR),
        save_strategy="no" if smoke else "epoch",
        max_length=MAX_SEQ,
        dataset_text_field="text",
        dataset_num_proc=1,
        report_to="none",
        seed=42,
        **({"max_steps": 2} if smoke else {"num_train_epochs": epochs}),
    )

    trainer = SFTTrainer(model=model, train_dataset=train_ds, args=cfg)

    import torch
    trainer.train()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\npeak VRAM allocated: {peak:.2f} GB")

    if smoke:
        print("SMOKE OK — load + 2 steps ran without OOM.")
        return

    model.save_pretrained(str(ADAPTER))
    tokenizer.save_pretrained(str(ADAPTER))
    print(f"saved LoRA adapter -> {ADAPTER}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="load + 2 steps, no save")
    ap.add_argument("--epochs", type=float, default=2.0)
    a = ap.parse_args()
    main(a.smoke, a.epochs)
