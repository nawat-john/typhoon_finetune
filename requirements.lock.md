# Working dependency set — Phase 0 verified (2026-06-02)

Env: `D:\Code\.venv` (Python 3.12.13). GPU: RTX 3050 Laptop, 4GB. Verified: unsloth imports, bitsandbytes 4-bit forward runs on CUDA.

These versions are **pinned deliberately** — newer ones break on this torch 2.6 / Windows / Py3.12 combo (see "Gotchas" below). Don't `pip install -U` blindly.

```
torch==2.6.0+cu124          # pre-installed in the venv; the anchor everything else matches
torchvision==0.21.0+cu124
torchao==0.9.0              # NOT >=0.13 — newer torchao needs torch 2.7+ (uses _pytree.register_constant)
transformers==4.56.2
trl==0.23.0
datasets==3.6.0
accelerate==1.13.0
peft==0.19.1
bitsandbytes==0.49.2        # Windows-native wheel, 4-bit + paged optimizer OK
pyarrow==19.0.1             # NOT 24.x — newer pyarrow segfaults when imported after torch on Windows
unsloth==2026.5.10
unsloth_zoo==2026.5.5
triton-windows==3.2.0.post21
sentencepiece==0.2.1
tyro==1.0.13
youtube-transcript-api==1.2.4
pythainlp==5.3.4
numpy==2.4.4
# xformers: intentionally NOT installed — no compatible Windows wheel for torch 2.6.
#           unsloth falls back to PyTorch SDPA, which is fine.
```

## Gotchas discovered while setting up (so we don't repeat them)

1. **pyarrow 24.0.0 segfaults** (access violation) when imported after `torch` on Windows — crashes the whole `datasets`/`unsloth` import. Fix: pin `pyarrow==19.0.1` (has the `json_()` API datasets needs; imports cleanly after torch).
2. **unsloth 2026.5.10 needs an older ecosystem** than pip's latest: `transformers<=5.5.0`, `trl<=0.24.0`, `datasets<4.4.0`. Installing unsloth with `--no-deps` then hand-pinning avoids it dragging in/keeping incompatible versions.
3. **torchao must be 0.9.0**, not the 0.13+ that unsloth-zoo's metadata requests. torchao 0.17 calls `torch.utils._pytree.register_constant`, which doesn't exist in torch 2.6 → import crash. transformers imports torchao when present, so a broken torchao breaks transformers too. 0.9.0 imports fine and unsloth still loads.
4. **xformers** can be skipped on this setup; unsloth runs without it.
5. Run Python via **PowerShell / direct `python.exe`**, not the Bash tool — torch+native libs segfaulted under git-bash here.

## Reproduce from scratch

```powershell
$py = "D:\Code\.venv\Scripts\python.exe"
& $py -m pip install "transformers==4.56.2" "trl==0.23.0" "datasets==3.6.0" "pyarrow==19.0.1" `
    accelerate peft bitsandbytes youtube-transcript-api pythainlp
& $py -m pip install --no-deps unsloth unsloth_zoo "torchao==0.9.0"
& $py -m pip install sentencepiece tyro nest-asyncio hf_transfer msgspec "wheel>=0.42.0" protobuf `
    "triton-windows<3.3" cut_cross_entropy diffusers
# data pipeline (Phase 1–3)
& $py -m pip install yt-dlp youtube-transcript-api pythainlp python-crfsuite google-genai python-dotenv
```

## Data-pipeline deps (Phase 1–3)
- `yt-dlp` — enumerate channel videos (youtube-transcript-api can't list a channel).
- `youtube-transcript-api==1.2.4` — **new API**: `YouTubeTranscriptApi().list(id)` / `.fetch(id, languages=[...])`; old static `.get_transcript()` is gone.
- `google-genai` + `python-dotenv` — Gemini for punctuation/prompts; key in `.env` as `GEMINI_API_KEY`.
  **Free tier = 20 requests/day/model** — scripts cycle models + cache to cope.
- `pythainlp` + `python-crfsuite` — installed for Thai NLP; crfcut sentence-split tried but not used (worse than Gemini).
