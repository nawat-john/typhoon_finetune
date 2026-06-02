# slappy — Fine-tune Typhoon2-3B ให้พูดสไตล์ YouTuber เกม "slapper"

Fine-tune `scb10x/llama3.2-typhoon2-3b-instruct` ด้วย **QLoRA** ให้เลียนแบบน้ำเสียง/คำติดปาก/จังหวะการพูดของช่องแคสต์เกมไทย **[Slapper](https://www.youtube.com/@slapperch)** โดยเรียนจาก YouTube transcript — ทั้ง pipeline ออกแบบให้รันได้บน GPU **VRAM แค่ 4GB**

> ผลลัพธ์: หลังเทรน โมเดลเปลี่ยนจาก "ผู้ช่วย AI สุภาพ" → "นักแคสต์เกมพูดคนเดียว" (คำติดปาก เออ/เอ้ย/อ่ะ...นะครับ, เล่าเกมมุมมองบุคคลที่ 1). แผนเต็ม: [`typhoon2_finetune_plan.md`](typhoon2_finetune_plan.md) · สถานะ: [`progress.md`](progress.md)

## Pipeline

```
list_videos → fetch_transcripts → clean → restore_punct → build_dataset → train → test_infer / chat
   (yt-dlp)   (transcript-api)   (regex)   (Gemini)       (จับคู่ prompt)  (QLoRA)
```

ผลลัพธ์แต่ละขั้นเก็บใน `data/` (ถูก gitignore — ไฟล์ใหญ่/มีลิขสิทธิ์ สร้างใหม่ได้)

## เทคนิคหลัก

**ยัด 3B QLoRA ลง VRAM 4GB** — ใช้ 3 อย่างพร้อมกัน: โหลด base แบบ **4-bit** (bitsandbytes, ~2GB), **gradient checkpointing** (`"unsloth"`), และ **paged 8-bit optimizer** (`paged_adamw_8bit`) ที่ spill optimizer state ไป system RAM อัตโนมัติเมื่อ VRAM เต็ม. จริง ๆ ใช้ peak แค่ **~3.0GB** (LoRA r=8, seq 512, batch 1 × grad-accum 16) เหลือที่ขยายได้

**เติมวรรคตอนคือขั้นสำคัญที่สุด** — auto-caption ภาษาไทยไม่มีเครื่องหมายวรรคตอนและเว้นวรรคทุกคำแบบผิดธรรมชาติ. `restore_punct.py` ส่งให้ LLM (Gemini) จัดวรรคตอน + ตัดประโยคใหม่ **โดยคงคำเดิมทุกคำ** เพื่อให้โมเดลเรียน "จังหวะการพูด" — มี guard ตรวจว่าไม่เพี้ยน (เทียบ char-ratio + space-ratio กันการ echo ข้อความดิบ)

**รับมือโควต้า Gemini free tier (20 req/วัน/โมเดล ต่อ project)** — สคริปต์ออกแบบให้:
- วน **หลายโมเดล × หลาย key** (แต่ละคู่มีโควต้าแยก), หยุดอย่างนุ่มนวลเมื่อหมด
- **hash-cache ต่อ chunk** → resume ได้ รันซ้ำวันถัดไปทำต่อจากเดิม ไม่เสีย API ซ้ำ
- ⚠️ โควต้าผูกกับ **project ไม่ใช่ key** — key หลายอันในบัญชีเดียวแชร์โควต้าเดียวกัน

**Monologue → บทสนทนา** — transcript เป็นพูดคนเดียว แต่ instruct model ต้องการคู่ user→assistant. `build_dataset.py` ตัดเป็นช่วง ~450 ตัวอักษร (คำพูดจริง = คำตอบ) แล้วจับคู่กับ prompt ภาษาไทยแบบสุ่มจากเทมเพลต (ไม่กิน API) — สำหรับ clone สไตล์ แค่นี้พอ

**กันโมเดลพูดวนซ้ำ** — generation ใส่ `repetition_penalty` + `no_repeat_ngram_size`

## ติดตั้ง

ต้องมี GPU ที่ใช้ CUDA ได้ + Python 3.10–3.12

```bash
python -m venv .venv
# เปิดใช้งาน venv (Linux/macOS: source .venv/bin/activate · Windows: .venv\Scripts\activate)

# ติดตั้ง PyTorch ที่ตรงกับ CUDA ของเครื่อง (ดู https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install unsloth transformers datasets trl peft bitsandbytes accelerate
pip install yt-dlp youtube-transcript-api google-genai python-dotenv pythainlp
```

> ⚠️ แพ็กเกจรุ่นใหม่สุดบางตัวเข้ากันไม่ได้กับ torch รุ่นเก่า/Windows (เช่น pyarrow, torchao, transformers/trl/unsloth ต้อง match กัน). ชุดเวอร์ชันที่ทดสอบแล้วว่ารันได้ + เหตุผลของแต่ละ pin อยู่ใน [`requirements.lock.md`](requirements.lock.md)

ตั้งค่า Gemini key สำหรับ Phase 2: คัดลอก `.env.example` → `.env` แล้วใส่ `GEMINI_API_KEY`

## วิธีรัน

```bash
# Phase 1 — ดึงรายการคลิป + transcript ภาษาไทย (ใหม่สุดก่อน)
python scripts/list_videos.py
python scripts/fetch_transcripts.py --target-hours 8 --max-clips 30

# Phase 2 — ทำความสะอาด + เติมวรรคตอน (resume ได้ถ้าโควต้าหมด)
python scripts/clean.py
python scripts/restore_punct.py

# Phase 3 — สร้าง dataset
python scripts/build_dataset.py

# Phase 4 — เทรน QLoRA (ลอง --smoke ก่อนเพื่อเช็คว่าไม่ OOM)
python scripts/train.py --smoke
python scripts/train.py --epochs 2          # ได้ adapter ที่ typhoon2_slapper_lora/

# Phase 5 — ทดสอบ / เทียบ base
python scripts/test_infer.py
python scripts/test_infer.py --base
```

### คุยกับโมเดล

```bash
python chat.py                       # โหมดคุยโต้ตอบ (q เพื่อออก)
python chat.py -p "วันนี้เล่นอะไรมา"   # ถามทีเดียว
python chat.py --base                # เทียบโมเดลเดิม
```

> หมายเหตุ Windows: ถ้า console เจอ error encoding จาก emoji ของ unsloth ให้ตั้ง `set PYTHONUTF8=1` ก่อนรัน

## ลิขสิทธิ์ / จริยธรรม

transcript เป็นของผู้สร้างคลิป — ใช้เพื่อการศึกษา/ส่วนตัว. อย่านำโมเดลไปแอบอ้างเป็นตัวจริงเพื่อหลอกลวง ควรระบุชัดว่าเป็น **AI เลียนแบบสไตล์**. Typhoon2 อยู่ภายใต้ Llama 3.2 Community License
