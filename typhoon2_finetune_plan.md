# แผนพัฒนา: Fine-tune Typhoon2 3B เลียนแบบสไตล์การพูด YouTuber

> เป้าหมาย: ปรับ `scb10x/llama3.2-typhoon2-3b-instruct` ให้พูดในสไตล์/น้ำเสียง/คำติดปากของ YouTuber เป้าหมาย โดยใช้ข้อมูลจาก YouTube transcript
> ฮาร์ดแวร์: RTX 3050 (VRAM 4GB) + ยืม system RAM ได้ แต่คุมงบรวม **ไม่เกิน 10GB**

---

## 0. สรุปข้อจำกัดฮาร์ดแวร์ (อ่านก่อน)

VRAM 4GB เทรน 3B ได้ "แบบเฉียดฉิว" ต้องพึ่ง 3 เทคนิคพร้อมกัน:

1. **QLoRA (4-bit)** — โหลดโมเดลแบบบีบอัด เหลือ ~2GB
2. **Gradient checkpointing** — แลกความเร็วเพื่อประหยัด VRAM ตอน backward
3. **Paged optimizer (`paged_adamw_8bit`)** — ย้าย optimizer state ไปไว้ใน RAM อัตโนมัติเมื่อ VRAM เต็ม (นี่คือกลไก "ยืม RAM" ที่คุณพูดถึง)

### งบหน่วยความจำที่วางไว้ (โดยประมาณ)

| รายการ | อยู่ที่ | ขนาด (ประมาณ) |
|---|---|---|
| โมเดล 4-bit | VRAM | ~2.0 GB |
| LoRA adapters + activations (seq 512, batch 1) | VRAM | ~1.5 GB |
| **รวม VRAM** | **GPU** | **~3.5 GB / 4 GB** |
| Optimizer state (paged → spill) | RAM | ~1–2 GB |
| Dataset + Python/CUDA overhead | RAM | ~3–5 GB |
| **รวมทั้งระบบ** | | **~7–9 GB** ✅ ต่ำกว่า 10GB |

> ⚠️ **ความจริงที่ต้องยอมรับ:** การ spill ไป RAM ทำให้ข้อมูลวิ่งผ่าน PCIe ซึ่งช้ากว่า VRAM มาก เทรนบนเครื่องนี้จะ **ช้า** (อาจหลักหลายชั่วโมง–ข้ามคืนต่อรอบ) ถ้าอยากเร็วกว่านี้ → **แนะนำ Google Colab (T4 ฟรี 16GB)** เป็นแผนหลัก แล้วใช้เครื่องตัวเองเป็นแผนสำรอง/ทดลอง

### ตั้งค่า Windows ให้ spill RAM ได้ (ถ้าเทรนในเครื่อง)
- NVIDIA Control Panel → Manage 3D Settings → **CUDA - Sysmem Fallback Policy** → ตั้งเป็น **Prefer Sysmem Fallback** (กัน CUDA out-of-memory crash โดยให้ล้นไป RAM แทน)
- ตั้ง env var ลดปัญหา memory fragmentation:
  ```
  set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```

---

## 1. ภาพรวม Pipeline

```
[เลือกคลิป] → [ดึง transcript] → [ทำความสะอาด] → [แปลงเป็น instruction format]
   → [QLoRA fine-tune] → [ทดสอบ] → [merge + แปลง GGUF] → [รันด้วย Ollama]
```

---

## 2. เตรียม Environment

```bash
# แนะนำ Python 3.10–3.11, สร้าง venv แยก
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
pip install youtube-transcript-api
pip install transformers>=4.45.0 datasets trl peft bitsandbytes accelerate
# สำหรับเติมเครื่องหมายวรรคตอนภาษาไทย (เลือกอย่างใดอย่างหนึ่งในขั้น cleaning)
pip install pythainlp
```

---

## Phase 1 — เก็บและคัดเลือกข้อมูล

**หลักการ: คุณภาพ > ปริมาณ** อย่าใช้หลายพันชั่วโมง เริ่มจาก **5–10 ชั่วโมงที่เป็นตัวเขาที่สุด** (พูดคุย เล่าเรื่อง รีแอค) เลี่ยงคลิปที่เสียงเกมกลบเสียงพูด

```python
from youtube_transcript_api import YouTubeTranscriptApi
import json

video_ids = ["xxxx", "yyyy"]  # ใส่ id คลิปที่คัดแล้ว
raw = []
for vid in video_ids:
    try:
        t = YouTubeTranscriptApi.get_transcript(vid, languages=['th'])
        text = " ".join(seg['text'] for seg in t)
        raw.append({"video_id": vid, "text": text})
    except Exception as e:
        print(f"skip {vid}: {e}")

json.dump(raw, open("raw_transcripts.json", "w"), ensure_ascii=False, indent=2)
```

**Milestone 1:** ได้ raw transcript 50–100 คลิป

---

## Phase 2 — ทำความสะอาด (ขั้นสำคัญที่สุด)

ปัญหาเฉพาะของ auto-caption YouTube:
- ไม่มีเครื่องหมายวรรคตอน / ไม่ตัดประโยค
- เสียงเกมหรือเอฟเฟกต์ปนมาเป็น `[เสียงดนตรี]`, `[Music]`, หรือข้อความมั่ว
- คำซ้ำรัวๆ จากการถอดเสียงผิด

```python
import re

def clean_text(text):
    text = re.sub(r'\[.*?\]', ' ', text)          # ตัด tag เสียง/เอฟเฟกต์
    text = re.sub(r'\(.*?\)', ' ', text)          # ตัดวงเล็บ
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)     # ลดอักษรซ้ำยาวผิดปกติ
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# กรองชิ้นที่สั้น/มั่วเกินไป (มักเป็น noise จากเสียงเกม)
def is_good(segment):
    return len(segment) > 30 and len(set(segment.split())) > 5
```

**เติมเครื่องหมายวรรคตอน** (เลือก 1 วิธี):
- **วิธีง่าย:** ส่งข้อความเป็นก้อนๆ ให้ LLM (Typhoon เอง หรือ API) ช่วยเติม `.` `?` และตัดประโยค → ทำให้โมเดลเรียน "จังหวะการพูด" ได้
- **วิธี local:** ใช้โมเดล punctuation restoration ภาษาไทย

**Milestone 2:** ได้ข้อความสะอาด แบ่งเป็นย่อหน้า/ประโยคแล้ว

---

## Phase 3 — แปลงเป็น Training Format

ปัญหา: transcript เป็น **monologue** ไม่มี "คำถาม" แต่ instruct model ต้องการคู่ user → assistant

### วิธีแนะนำ: สร้างคำถามสังเคราะห์ (synthetic prompts)
แบ่ง monologue เป็นชิ้นตามหัวข้อ (ไม่ตัดตามจำนวนคำตายตัว) แล้วใช้ LLM เดา "คำถาม/หัวข้อที่น่าจะนำไปสู่คำพูดนี้" ส่วนคำพูดจริงของ YouTuber = คำตอบ

```python
# โครงสร้างเป้าหมาย (jsonl, 1 บรรทัด = 1 ตัวอย่าง)
{"messages": [
    {"role": "user", "content": "คิดยังไงกับบอสตัวนี้"},
    {"role": "assistant", "content": "<คำพูดจริงของ youtuber สไตล์เขา>"}
]}
```

ใช้ chat template ของ Llama 3 (Typhoon2 ใช้ template เดียวกัน) — Unsloth จัดการให้อัตโนมัติ

### ทางเลือกสำรอง: completion ล้วนบน base model
ถ้าอยากได้สไตล์ "บริสุทธิ์" และไม่ต้องการความสามารถสนทนา → เทรน `scb10x/llama3.2-typhoon2-3b` (ตัว base) ด้วย monologue ดิบเป็น text completion

**Milestone 3:** ได้ `dataset.jsonl` พร้อมเทรน (เริ่มจากชุดเล็ก ~500–1000 ตัวอย่างเพื่อทดสอบ pipeline ก่อน)

---

## Phase 4 — Fine-tune ด้วย QLoRA (config สำหรับ 4GB)

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

max_seq_length = 512   # คุม VRAM — เริ่มที่ 512 ก่อน

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "scb10x/llama3.2-typhoon2-3b-instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8,                       # rank ต่ำเพื่อประหยัด VRAM (8 พอสำหรับงานสไตล์)
    lora_alpha = 16,
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing = "unsloth",   # ประหยัด VRAM ตอน backward
    random_state = 42,
)

dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 16,   # effective batch = 16
        warmup_steps = 10,
        num_train_epochs = 2,               # 2–3 พอ อย่าเยอะจน overfit
        learning_rate = 2e-4,
        bf16 = True,
        optim = "paged_adamw_8bit",          # ★ spill optimizer ไป RAM
        logging_steps = 5,
        output_dir = "outputs",
        save_strategy = "epoch",
    ),
)
trainer.train()

model.save_pretrained("typhoon2_youtuber_lora")   # เก็บแค่ LoRA adapter
```

### ปุ่มปรับเมื่อ VRAM เต็ม (OOM) — ปรับทีละข้อ
1. ลด `max_seq_length` → 256
2. เพิ่ม `gradient_accumulation_steps` (ชดเชย batch ที่เล็กลง)
3. ลด LoRA `r` → 4
4. ลด target_modules เหลือเฉพาะ attention: `["q_proj","k_proj","v_proj","o_proj"]`

**Milestone 4:** เทรนชุดเล็กผ่าน → scale ขึ้นข้อมูลเต็ม

---

## Phase 5 — ทดสอบคุณภาพ

```python
FastLanguageModel.for_inference(model)
messages = [{"role": "user", "content": "เล่าให้ฟังหน่อยว่าวันนี้เล่นอะไรมา"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt",
                                       add_generation_prompt=True).to("cuda")
out = model.generate(inputs, max_new_tokens=256, temperature=0.8)
print(tokenizer.decode(out[0]))
```

เช็ค 2 ด้าน:
- ✅ **ติดสไตล์ไหม** — คำติดปาก น้ำเสียง จังหวะ เหมือนเขาไหม
- ✅ **ยังฉลาดอยู่ไหม** — ตอบเป็นเหตุเป็นผล ไม่วนซ้ำ ไม่เพี้ยน (ถ้าเพี้ยน = overfit → ลด epoch/lr)

---

## Phase 6 — Deploy สำหรับรันบนเครื่องเล็ก

```bash
# merge LoRA เข้า base แล้วแปลงเป็น GGUF Q4 (รัน inference ใน VRAM 4GB ได้สบาย ~2.5GB)
# Unsloth ช่วย export ได้:
model.save_pretrained_gguf("typhoon2_youtuber_gguf", tokenizer, quantization_method="q4_k_m")
```
จากนั้นรันด้วย **Ollama** หรือ **LM Studio** เป็นแชตบอตในเครื่องได้เลย

---

## Timeline แนะนำ

| สัปดาห์ | งาน |
|---|---|
| 1 | ตั้ง environment + เก็บ/คัดคลิป + ดึง transcript (Phase 1) |
| 2 | ทำความสะอาด + เติมวรรคตอน (Phase 2) |
| 2–3 | แปลง format + สร้าง dataset เล็กทดสอบ (Phase 3) |
| 3 | เทรนชุดเล็ก debug pipeline (Phase 4 รอบแรก) |
| 4 | เทรนชุดเต็ม + ทดสอบ + ปรับ (Phase 4–5) |
| 5 | Export GGUF + deploy (Phase 6) |

---

## ข้อควรระวังด้านสิทธิ์/จริยธรรม

- Transcript และเนื้อหาคลิปมี **ลิขสิทธิ์ของผู้สร้าง** — ใช้ในขอบเขตส่วนตัว/ศึกษา/สร้างสรรค์
- อย่านำโมเดลไปแอบอ้างเป็นตัว YouTuber จริงเพื่อหลอกลวงผู้อื่น
- หากจะเผยแพร่ ควรขออนุญาตเจ้าตัว หรือระบุชัดเจนว่าเป็น **AI ที่เลียนแบบสไตล์**
- โมเดล Typhoon2 อยู่ภายใต้ **Llama 3.2 Community License** — ตรวจเงื่อนไขก่อนใช้เชิงพาณิชย์

---

## Troubleshooting

| ปัญหา | วิธีแก้ |
|---|---|
| `CUDA out of memory` | ลด seq_length → 256, เปิด Sysmem Fallback, ลด LoRA r |
| เทรนช้ามาก | ปกติบน 4GB+spill — พิจารณา Colab T4 |
| โมเดลพูดวนซ้ำ/เพี้ยน | overfit → ลด epoch เหลือ 1–2, ลด lr |
| ไม่ติดสไตล์ | ข้อมูลน้อย/ไม่สะอาด → เพิ่มข้อมูลคุณภาพ, เช็คขั้น cleaning |
| ลืมความสามารถเดิม | ผสมข้อมูล general เข้าไปบ้าง หรือลด epoch |
