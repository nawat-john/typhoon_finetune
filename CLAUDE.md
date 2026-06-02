# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This is a **planning-stage repository**. It currently holds one design document and no code, tests, or build tooling yet. The single source of truth is `typhoon2_finetune_plan.md` (written in Thai). When implementing, the code does not exist yet — you are building it from that plan.

## What this project is

A pipeline to fine-tune `scb10x/llama3.2-typhoon2-3b-instruct` so it mimics the speaking style / tone / catchphrases of a target Thai YouTuber, using cleaned YouTube transcripts as training data. Output is a QLoRA adapter, optionally merged and exported to GGUF for local inference via Ollama / LM Studio.

## Hard constraints (drive nearly every technical decision)

- **GPU: RTX 3050, 4GB VRAM.** Total memory budget (VRAM + spilled system RAM) must stay **under 10GB**.
- Training fits only via the combination of **QLoRA 4-bit** + **gradient checkpointing** + **paged optimizer (`paged_adamw_8bit`)**, which spills optimizer state to system RAM over PCIe. This is deliberately slow (hours to overnight per run). Google Colab T4 (free 16GB) is the recommended primary trainer; the local machine is the fallback.
- On Windows, set NVIDIA Control Panel → *CUDA - Sysmem Fallback Policy* → **Prefer Sysmem Fallback**, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, to avoid hard OOM crashes.

## Pipeline architecture (6 phases)

```
[select clips] → [pull transcript] → [clean] → [convert to instruction format]
   → [QLoRA fine-tune] → [test] → [merge + export GGUF] → [run via Ollama]
```

1. **Phase 1 — Collect:** `youtube-transcript-api` (Thai `languages=['th']`) → `raw_transcripts.json`. Quality over quantity: ~5–10 hours of the most characteristic speech; avoid clips where game audio drowns out the voice.
2. **Phase 2 — Clean (most important step):** strip `[...]`/`(...)` audio tags, collapse repeated chars from mis-transcription, drop short/noisy segments. Then **restore punctuation** (auto-captions have none) via an LLM pass or a Thai punctuation-restoration model — this is what teaches speaking rhythm.
3. **Phase 3 — Format:** transcripts are monologues with no questions, but the instruct model needs user→assistant pairs. Synthesize plausible user prompts via an LLM; the YouTuber's real words become the assistant reply. Target is `dataset.jsonl`, one `{"messages":[...]}` per line, Llama 3 chat template. Start small (~500–1000 examples) to debug the pipeline before scaling. (Alt path: completion-only training on the `scb10x/llama3.2-typhoon2-3b` base for "pure" style.)
4. **Phase 4 — Fine-tune:** Unsloth `FastLanguageModel` + TRL `SFTTrainer`. Baseline config: `max_seq_length=512`, LoRA `r=8`/`alpha=16`, all 7 target modules, `per_device_train_batch_size=1`, `gradient_accumulation_steps=16`, `num_train_epochs=2`, `lr=2e-4`, `optim="paged_adamw_8bit"`. Save the adapter only.
5. **Phase 5 — Test:** check both that style transferred (catchphrases/tone) **and** that the model stayed coherent (looping/garbled = overfit → lower epochs/lr).
6. **Phase 6 — Deploy:** `save_pretrained_gguf(..., quantization_method="q4_k_m")`, run under Ollama / LM Studio (~2.5GB, fits 4GB VRAM).

## OOM tuning ladder (apply one at a time, in order)

1. `max_seq_length` → 256
2. raise `gradient_accumulation_steps` to compensate
3. LoRA `r` → 4
4. reduce `target_modules` to attention only: `["q_proj","k_proj","v_proj","o_proj"]`

## Environment setup

```bash
# Python 3.10–3.11, dedicated venv
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
pip install youtube-transcript-api
pip install transformers>=4.45.0 datasets trl peft bitsandbytes accelerate
pip install pythainlp   # optional, for Thai punctuation during cleaning
```

## Conventions / gotchas

- Typhoon2 uses the **Llama 3 chat template**; Unsloth applies it automatically. Don't hand-roll it.
- Keep training data quality high and epochs low (2–3). Overfitting shows up as repetition/incoherence, not better style.
- Licensing/ethics matter here: transcripts are copyrighted; the model is licensed under Llama 3.2 Community License; output must be labeled as an AI style imitation, not the real person. See the plan's final sections before any public release.
