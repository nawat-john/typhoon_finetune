# Progress — Fine-tune Typhoon2 3B on "slapper" (Thai game-casting YouTuber) style

Tracking the build from `typhoon2_finetune_plan.md`. Goal: QLoRA fine-tune `scb10x/llama3.2-typhoon2-3b-instruct` to imitate the speaking style of the Thai game caster **slapper**, then export GGUF for local use.

## Environment (DONE — Phase 0 complete ✅)

- **Python env:** `D:\Code\.venv` (Python 3.12.13, self-contained on D:). Run scripts with the full path, via **PowerShell** (not the Bash tool — torch + native libs segfault under git-bash here):
  ```powershell
  D:\Code\.venv\Scripts\python.exe <script>.py
  D:\Code\.venv\Scripts\python.exe -m pip install <pkg>
  ```
- **GPU verified:** RTX 3050 Laptop GPU, 4.0 GB VRAM. `torch 2.6.0+cu124`, CUDA available.
- **Toolchain verified:** `unsloth` imports; `bitsandbytes` 4-bit forward runs on the GPU.
- **Training to run locally** on this GPU (user decision). Expect slow runs (overnight-ish) — that's accepted.
- **Exact working versions are pinned** in `requirements.lock.md`. Do NOT `pip install -U` — newer pyarrow/torchao break this torch-2.6 setup. See that file for the full list and the gotchas behind each pin.

---

## Checklist

### Phase 0 — Environment & deps ✅ DONE
- [x] Locate/confirm Python env (`D:\Code\.venv`, Py 3.12.13)
- [x] Verify GPU + CUDA (RTX 3050, 4GB, torch cu124 OK)
- [x] Install deps into the venv (unsloth + full QLoRA stack) — versions pinned in `requirements.lock.md`
- [x] Smoke-test `unsloth` import — OK
- [x] Smoke-test `bitsandbytes` 4-bit forward on GPU — OK (kernels run)
- [ ] Set Windows OOM safety: NVIDIA Control Panel → CUDA Sysmem Fallback → *Prefer Sysmem Fallback* (manual GUI step — **user to do**). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` will be set inside the training script via `os.environ`.

### Phase 1 — Collect transcripts ✅ DONE
- [x] `scripts/list_videos.py` — enumerate channel via yt-dlp → `data/video_list.json` (**3066 videos, 1542 h available**, newest first)
- [x] `scripts/fetch_transcripts.py` — pull Thai transcripts via `youtube-transcript-api` (resumable, incremental save) → `data/raw_transcripts.json`
- [x] Collected newest-first: **11 clips, ~8.0 h, 313,719 chars** (target was ~8 h)
- [x] **Milestone 1:** raw transcripts in hand. (Read the plan's "50–100 clips" as the hours target — 30–60 min clips means ~8 h = 11 clips. Easy to pull more: re-run `fetch_transcripts.py --target-hours N --max-clips M`, it resumes.)
- [x] Quality spot-check: clean first-person Thai monologue ("ผม…ครับ"), conversational — good for style. No punctuation (auto-caption) → Phase 2. `[ __ ]` = YouTube profanity censor, stripped by Phase 2's `[...]` rule.

### Phase 2 — Clean (most important)
- [x] `scripts/clean.py` — regex strip `[...]`/`(...)`, collapse repeated chars, normalize spaces → `data/cleaned_transcripts.json` (1.9% removed; source was clean)
- [x] `scripts/restore_punct.py` — Thai punctuation + sentence segmentation via **Gemini**, with chunking, hash-cache (resumable), **multi-key × multi-model cycling**, graceful stop on quota, and a validity guard → `data/punctuated_transcripts.json`. Quality verified excellent (natural spacing, verbatim words, `?`/`ๆ` correct).
- [~] **Milestone 2:** **6/11 clips** fully punctuated (`149,085` chars, ~1,171 sentences, only ~2.5% lines still raw). Enough to drive Phase 3–4 "start small". Remaining ~72 chunks finish on the next fresh-quota day.
- [x] Rejected local `pythainlp` crfcut fallback — needs space-stripped input which mangles mixed Thai/English (คอนเทนต์→คทent) and gives run-on segments. Gemini quality much better.

> **Gemini reality — free tier = 20 requests/day per model, PER GOOGLE PROJECT** (`GenerateRequestsPerDayPerProject**PerModel**-FreeTier`, quotaValue 20). ⚠️ The pool is **per project, NOT per key** — confirmed empirically (two keys from the same account gave 33 then only 13 more calls = shared pool). So **making new keys in the same account does NOT add quota.** ~5 models ⇒ ~**100 free calls/day per project**.
> To actually multiply capacity you need keys from **different Google projects/accounts** (each its own pool). `restore_punct.py` cycles `N keys × 5 models`; put extra keys in `.env` as `GEMINI_API_KEY` (comma-sep) / `GEMINI_API_KEY_2`, `_3`, … — but only different-project keys help. On quota-out it stops gracefully and writes every fully-cached clip.
> **To continue:** just re-run `restore_punct.py` on a fresh-quota day (cache resumes). ~72 chunks left → finishes in ~1 day with one key, fewer with more keys. It self-heals: regenerates all 11 clips cleanly.
> **Guard / cache lessons (baked into the script):** (1) never cache an **empty** API result (transient 5xx/safety) — retry on another slot; (2) validity = non-empty + verbatim char-ratio 0.85–1.15 + **space-ratio ≤ 0.12** (raw auto-caption has spaces between every word ~0.15+; punctuated natural Thai ~0.04). Newline count is NOT a reliable signal (some good outputs are long single-line sentences) — don't purge by it.
> Same 20/day applies to Phase 3 prompt-gen — budget accordingly.

### Phase 3 — Build dataset
- [x] `scripts/build_dataset.py` — splits punctuated monologue into ~450-char chat turns (splits over-long run-on lines on punctuation/space), pairs each with a prompt, shuffles, train/val split → `data/dataset.jsonl` + `dataset.val.jsonl`.
- [x] Prompt source = **template mode** (quota-free): 15 generic Thai conversational prompts, random. (Gemini `--prompts gemini` mode = tailored questions, deferred — same 20/day quota battle, not needed for first run.)
- [x] **Milestone 3:** **302 examples** (287 train / 15 val), assistant replies 81–600 chars (avg 420), from the 5 punctuated clips. Format `{"messages":[user,assistant]}` (Unsloth applies Llama-3 template at train time — not pre-formatted).
- [ ] Later: regenerate with more clips (once 11/11 punctuated) and/or `--prompts gemini` for tailored questions → larger, higher-quality set before the real training run.

### Phase 4 — QLoRA fine-tune
- [x] `scripts/train.py` — 4GB baseline (seq 512, r=8/alpha16, bs1, grad_accum16, 2 epochs, lr2e-4, `paged_adamw_8bit`, grad-checkpointing="unsloth").
- [x] **Smoke test PASSED** (`--smoke`, 2 steps): loads in 4-bit, trains, **peak VRAM 2.96/4.0 GB — fits with headroom, no OOM.** Loss ~2.7, ~13–21 s/step, 12.2M LoRA params (0.38%).
- [x] **OOM ladder NOT needed** (2.96 GB << 4 GB). Room to grow seq/rank later if wanted.
- [x] Full 2-epoch run (36 steps, ~18.5 min): loss **2.68 → 2.11** (clean decline, no divergence), peak VRAM **2.99 GB**. Adapter saved → `typhoon2_slapper_lora/`.
- [x] **Milestone 4:** training completes + adapter saved. ✅

> Script gotchas baked in: (1) Unsloth SFTTrainer needs a pre-rendered `text` field — we map `messages`→`apply_chat_template(tokenize=False)` and set `dataset_text_field="text"`; (2) force UTF-8 stdout (`sys.stdout.reconfigure`) or Unsloth's 🦥 emoji prints crash on the Thai cp874 console; (3) trl 0.23 uses `SFTConfig` with `max_length` (not `max_seq_length`).
- [ ] Scale to full dataset
- [ ] Save LoRA adapter (`typhoon2_youtuber_lora`)
- [ ] **Milestone 4:** full training run completes

### Phase 5 — Evaluate
- [x] `scripts/test_infer.py` — loads base+adapter (4-bit), generates on slapper-style prompts (`--base` to compare).
- [x] **Style transfer = clear success.** Outputs are full of his verbal tics — "เออ", "เอ้ย", "ชิบหายเลยมึง", "อ่ะ...นะครับ", "กู", "อืม" — first-person rambling gameplay narration (selling cars/items, coins, ปลดล็อก). Sounds like him.
- [~] **Coherence:** 3/4 replies coherent & on-style; **1/4 degenerated into a char-repeat loop** ("ยยยยย…"). Mild instability — expected with only 302 examples + sampling.
- [ ] Mitigations (next): add `repetition_penalty`/`no_repeat_ngram_size` at generation (added to `test_infer.py`); and the real fix = **more/cleaner data** (finish 11-clip punctuation + more clips). Loss declined smoothly so it's not classic overfit — don't slash epochs yet.

### Phase 6 — Deploy — SKIPPED (user: no deploy)
- [x] Instead: `chat.py` at repo root — simple input→output REPL / one-shot (`-p`) / `--base` toggle.
- [x] **Base-vs-tuned comparison done:** base = generic polite assistant (formal, bullet-point advice, 😊, refuses/asks for clarification); tuned = slapper's first-person rambling gameplay voice with his fillers/profanity. Fine-tune clearly changed the persona. ~~GGUF/Ollama~~ not needed.

---

## Target channel — slapper (resolved)
- **Channel:** `@slapperch` — "Slapper", ID `UC8fWyZQfxhakER1VrBnMlEQ`. 3066 videos, ~1542 h total.
- Format: long clips, **30–60 min each, single speaker** (monologue), minor game audio bleed → good fit for style cloning.
- Clip selection: **newest clips first**, pull as many as needed until data volume is sufficient.

## Open questions / TODO to confirm with user
- [x] LLM for punctuation/prompts → **Gemini** (`gemini-2.5-flash-lite`, key in `.env`)
- [ ] Is ~8 h enough, or pull more before training? (start small per plan; can always add via resumable fetch)
- [ ] Watch Gemini free-tier RPD across Phase 2 + Phase 3 — if flash-lite quota runs low, spread work across days or pull a Thai punctuation model locally

## Log
- 2026-06-02 — Set up tracking. Confirmed env `D:\Code\.venv` (Py3.12.13, torch2.6.0+cu124) and RTX 3050 4GB.
- 2026-06-02 — **Phase 0 complete.** Installed full unsloth QLoRA stack into the venv; pinned versions in `requirements.lock.md`. Verified unsloth import + bitsandbytes 4-bit GPU forward. Decided: train locally on the 3050. Hit & resolved 3 dependency traps (pyarrow 24 segfault → 19.0.1; unsloth needs older transformers/trl/datasets; torchao 0.17→0.9 for torch 2.6).
- 2026-06-02 — **Phase 1 complete.** Resolved channel = `@slapperch`. Added `scripts/list_videos.py` + `scripts/fetch_transcripts.py` (installed `yt-dlp`). Listed 3066 videos; fetched newest 11 → `data/raw_transcripts.json` (~8 h, 314k chars). youtube-transcript-api 1.2.4 uses the new `ytt.list()/ytt.fetch()` API. No rate-limiting at sleep=1.5s. Next: Phase 2 cleaning.
- 2026-06-02 — **Done (per user scope).** Added `chat.py` (simple input→output, `-p` one-shot, `--base`). Ran base-vs-tuned comparison — base is a generic polite assistant, tuned speaks as slapper. Deploy/GGUF skipped on request. Open future work: finish 11-clip punctuation + more clips → bigger dataset → retrain for better coherence.
- 2026-06-02 — **Phase 4 + 5 done (pipeline proven end-to-end).** `train.py` QLoRA full run: 36 steps/2ep, loss 2.68→2.11, **peak 2.99/4 GB**, adapter saved `typhoon2_slapper_lora/`. `test_infer.py`: **style clearly transferred** (slapper's catchphrases/tone), 3/4 coherent, 1/4 looped (added repetition_penalty + no_repeat_ngram). Whole pipeline Phase 0→5 works on the 3050. Remaining: Phase 6 GGUF/Ollama; scale data for quality.
- 2026-06-02 — **Phase 3 (start-small) done.** `build_dataset.py` → 302 examples (287/15) from 5 clips, template Thai prompts (quota-free). Ready for a Phase 4 pipeline-debug training run. Will regrow once more clips punctuate / with Gemini-tailored prompts.
- 2026-06-02 — **Phase 2 mostly done.** `clean.py` (regex) + `restore_punct.py` (Gemini, `google-genai`+`python-dotenv`). Cleaned all 11 clips. Punctuation hit Gemini free-tier **20 req/day/model** — built multi-key × multi-model cycling + graceful-stop + hash-cache + validity guard. **6/11 clips punctuated** (149k chars, ~2.5% residual raw); rest finish next fresh-quota day (self-heals to clean 11/11). Rejected local pythainlp crfcut. Lessons: don't cache empty results; validate by space-ratio not newline count (over-purged cache once, output file survived via safety guard). Ready for Phase 3 "start small" with the 6 clips.
