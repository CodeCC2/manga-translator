# Local Manga Translator (EN → TH)

เว็บแอปสำหรับแปลมังงะจากภาษาอังกฤษเป็นภาษาไทยแบบ local - ไม่ต้องพึ่ง API ภายนอก

## ✨ ผลลัพธ์

| ก่อนแปล (Original) | หลังแปล (Translated) |
|:------------------:|:--------------------:|
| ![Before](assets/images/before.jpg) | ![After](assets/images/after.png) |

## 🚀 Quick Start

1. **ติดตั้ง dependencies:**
```bash
pip install -r requirements.txt
```

2. **รัน server:**
```bash
python -m uvicorn app.main:app --reload --port 8000
```

3. **เปิดเบราว์เซอร์:** http://localhost:8000

> ⚠️ ครั้งแรกจะดาวน์โหลดโมเดล (~2GB) อาจใช้เวลา 1-3 นาที

## 🔧 สถาปัตยกรรม

| Component | Technology |
|-----------|------------|
| **OCR** | EasyOCR (EN/JA) |
| **Translation** | NLLB-200 (`facebook/nllb-200-distilled-1.3B`) |
| **Text Removal** | OpenCV Inpainting (Telea) |
| **Text Redraw** | Pillow + Sarabun Font |

## 📡 API

### POST /api/process
อัพโหลดรูปภาพเพื่อแปล

```bash
curl -X POST -F "file=@manga.jpg" http://localhost:8000/api/process
```

**Query params:**
- `source_lang` - ภาษาต้นฉบับ: `en` (default) หรือ `ja`

**Response:**
```json
{
  "image_base64": "...",
  "regions": [...],
  "meta": {"model": "...", "device": "cuda/cpu"}
}
```

### GET /health
ตรวจสอบสถานะ server

## ⚙️ การปรับแต่ง

แก้ไขใน `app/pipeline.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_font_size` | ขนาด font สูงสุด | 52 |
| `min_font_size` | ขนาด font ต่ำสุด | 18 |
| `stroke_width` | ความหนาเส้นขอบ | 2 |

## 💻 GPU Support

ถ้ามี NVIDIA GPU:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 📝 หมายเหตุ

- **Inpainting:** เหมาะกับพื้นหลังเรียบ/ไล่โทน พื้นหลังซับซ้อนอาจต้องใช้ LaMa
- **OCR:** อาจพลาดข้อความเอียงมากหรือ SFX
- **Font:** ใช้ Sarabun รองรับไทย + อังกฤษ + เครื่องหมายวรรคตอน

---

Made with ❤️ for manga lovers
