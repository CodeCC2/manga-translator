from __future__ import annotations

import base64
import os
import re
import string
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import easyocr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import fitz  # PyMuPDF
import img2pdf


DEFAULT_MODEL = os.getenv("NMT_MODEL", "facebook/nllb-200-3.3B")
DEFAULT_FONT = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "Sarabun-Regular.ttf"
FALLBACK_FONT = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "NotoSans-Regular.ttf"

@dataclass
class TextRegion:
    box: List[Tuple[int, int]]
    text: str
    confidence: float
    translation: str | None = None

    def as_dict(self) -> dict:
        return {
            "box": self.box,
            "text": self.text,
            "confidence": round(float(self.confidence), 3),
            "translation": self.translation,
        }


class MangaPipeline:
    """
    Small end-to-end pipeline: OCR -> translate -> remove old text -> redraw.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        target_lang: str = "th",
        font_path: Path = DEFAULT_FONT,
        max_font_size: int = 52,
        min_font_size: int = 18,
        stroke_width: int = 2,
    ) -> None:
        self.model_name = model_name
        self.target_lang = target_lang
        self.font_path = Path(font_path)
        self.fallback_font_path = FALLBACK_FONT
        self.max_font_size = max_font_size
        self.min_font_size = min_font_size
        self.stroke_width = stroke_width
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # OCR languages: English + Japanese
        self.reader = easyocr.Reader(["en", "ja"], gpu=torch.cuda.is_available())

        # Initialize translation model once, kept in memory
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Language codes for NLLB/M2M-style tokenizers
        if hasattr(self.tokenizer, "lang_code_to_id"):
            self.src_lang = "eng_Latn"
            self.tgt_lang = "tha_Thai"
            self.tokenizer.src_lang = self.src_lang
        else:
            self.src_lang = "en"
            self.tgt_lang = "th"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16
        ).to(self.device)

    def process_image(self, image_bytes: bytes, source_lang: str = "en") -> dict:
        """
        Process a single image and return a dict with base64 image and region metadata.
        """
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        regions = self._run_ocr(image)
        if not regions:
            return {
                "image_base64": self._encode_image(image),
                "regions": [],
                "meta": {"translations": 0, "ocr_regions": 0},
            }

        translations = self._translate_batch([r.text for r in regions], source_lang=source_lang)
        for region, translated in zip(regions, translations):
            region.translation = translated

        inpainted = self._inpaint_regions(image, regions)
        redrawn = self._draw_translations(inpainted, regions)

        return {
            "image_base64": self._encode_image(redrawn),
            "regions": [r.as_dict() for r in regions],
            "meta": {
                "translations": len(translations),
                "ocr_regions": len(regions),
                "device": self.device,
                "model": self.model_name,
            },
        }

    def process_pdf(self, pdf_bytes: bytes, source_lang: str = "en", progress_callback=None) -> dict:
        """
        Process a PDF file: convert pages to images, translate each, combine back to PDF.
        """
        # Open PDF from bytes
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(pdf_doc)
        
        processed_images: List[bytes] = []
        all_regions: List[dict] = []
        
        for page_num in range(total_pages):
            if progress_callback:
                progress_callback(page_num + 1, total_pages)
            
            # Convert PDF page to image (300 DPI for good quality)
            page = pdf_doc[page_num]
            mat = fitz.Matrix(300 / 72, 300 / 72)  # 300 DPI
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            
            # Process the image
            result = self.process_image(img_bytes, source_lang=source_lang)
            
            # Store processed image
            processed_images.append(base64.b64decode(result["image_base64"]))
            all_regions.extend(result.get("regions", []))
        
        pdf_doc.close()
        
        # Combine images back to PDF
        pdf_output = img2pdf.convert(processed_images)
        pdf_base64 = base64.b64encode(pdf_output).decode("utf-8")
        
        return {
            "pdf_base64": pdf_base64,
            "total_pages": total_pages,
            "regions": all_regions,
            "meta": {
                "translations": len(all_regions),
                "pages_processed": total_pages,
                "device": self.device,
                "model": self.model_name,
            },
        }

    def _run_ocr(self, image: Image.Image) -> List[TextRegion]:
        np_image = np.array(image)
        results = self.reader.readtext(np_image, detail=1, paragraph=False)
        regions: List[TextRegion] = []
        for item in results:
            # easyocr returns [bbox, text, confidence]
            bbox, text, confidence = item
            text = text.strip()
            if not text:
                continue
            points = [(int(x), int(y)) for x, y in bbox]
            regions.append(TextRegion(box=points, text=text, confidence=float(confidence)))
        return regions

    def _translate_batch(self, texts: Sequence[str], source_lang: str = "en") -> List[str]:
        if not texts:
            return []
        # Map user input to NLLB codes
        lang_map = {
            "en": "eng_Latn",
            "ja": "jpn_Jpan",
            "jp": "jpn_Jpan",
            "eng_Latn": "eng_Latn",
            "jpn_Jpan": "jpn_Jpan",
        }
        src_code = lang_map.get(source_lang.lower(), lang_map["en"])
        if hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.device)
        generation_kwargs = {
            "max_length": 256,
            "num_beams": 5,
            "repetition_penalty": 1.05,
        }
        if hasattr(self.tokenizer, "lang_code_to_id"):
            # NLLB / M2M style forced BOS
            tgt_code = getattr(self, "tgt_lang", self.target_lang)
            generation_kwargs["forced_bos_token_id"] = self.tokenizer.lang_code_to_id[tgt_code]
        elif hasattr(self.tokenizer, "get_lang_id"):
            generation_kwargs["forced_bos_token_id"] = self.tokenizer.get_lang_id(self.target_lang)
        with torch.no_grad():
            generated_tokens = self.model.generate(**inputs, **generation_kwargs)
        outputs = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return [self._normalize_text(o) for o in outputs]

    def _inpaint_regions(self, image: Image.Image, regions: Sequence[TextRegion]) -> Image.Image:
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        for region in regions:
            poly = np.array(region.box, dtype=np.int32)
            cv2.fillPoly(mask, [poly], 255)
        cleaned = cv2.inpaint(bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))

    def _draw_translations(self, image: Image.Image, regions: Sequence[TextRegion]) -> Image.Image:
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        for region in regions:
            translated = region.translation or region.text
            xs, ys = zip(*region.box)
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            box_w = max(8, x_max - x_min)
            box_h = max(8, y_max - y_min)

            font, wrapped_text = self._fit_text(draw, translated, box_w, box_h)
            text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, stroke_width=self.stroke_width)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            x = x_min + (box_w - text_w) / 2
            y = y_min + (box_h - text_h) / 2

            # Draw white text with dark stroke for readability
            draw.multiline_text(
                (x, y),
                wrapped_text,
                font=font,
                fill=(255, 255, 255),
                stroke_width=self.stroke_width,
                stroke_fill=(20, 20, 20),
                align="center",
            )
        return canvas
    
    @staticmethod
    def _is_thai_char(ch: str) -> bool:
        """Check if character is in Thai Unicode range."""
        return "\u0e00" <= ch <= "\u0e7f"

    def _fit_text_size(self, draw: ImageDraw.ImageDraw, text: str, max_width: int, max_height: int) -> Tuple[int, str]:
        """Find the best font size and wrapped text that fits in the box. Returns (font_size, wrapped_text)."""
        text = text.strip()
        for size in range(self.max_font_size, self.min_font_size - 1, -2):
            font = ImageFont.truetype(str(self.font_path), size=size)
            wrapped = self._wrap_text(draw, text, font, max_width)
            bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, stroke_width=self.stroke_width)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width <= max_width and height <= max_height:
                return size, wrapped
        # Fallback to smallest size
        font = ImageFont.truetype(str(self.font_path), size=self.min_font_size)
        wrapped = self._wrap_text(draw, text, font, max_width)
        return self.min_font_size, wrapped


    def _fit_text(self, draw: ImageDraw.ImageDraw, text: str, max_width: int, max_height: int) -> Tuple[ImageFont.FreeTypeFont, str]:
        text = text.strip()
        for size in range(self.max_font_size, self.min_font_size - 1, -2):
            font = ImageFont.truetype(str(self.font_path), size=size)
            wrapped = self._wrap_text(draw, text, font, max_width)
            bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, stroke_width=self.stroke_width)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width <= max_width and height <= max_height:
                return font, wrapped
        # Fallback to smallest size even if it overflows
        font = ImageFont.truetype(str(self.font_path), size=self.min_font_size)
        wrapped = self._wrap_text(draw, text, font, max_width)
        return font, wrapped

    def _wrap_text(self, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
        # Simple greedy wrap that still works for Thai (no spaces) by falling back to character-based splits.
        words = text.split()
        if not words:
            words = [text]

        lines: List[str] = []
        current = ""
        for word in words:
            proposal = word if not current else f"{current} {word}"
            if draw.textlength(proposal, font=font) <= max_width or not current:
                current = proposal
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)

        # If still too wide (common for Thai text without spaces), fallback to character wrapping
        adjusted_lines: List[str] = []
        for line in lines:
            if draw.textlength(line, font=font) <= max_width:
                adjusted_lines.append(line)
                continue
            accumulator = ""
            for ch in line:
                proposal = accumulator + ch
                if draw.textlength(proposal, font=font) <= max_width or not accumulator:
                    accumulator = proposal
                else:
                    adjusted_lines.append(accumulator)
                    accumulator = ch
            if accumulator:
                adjusted_lines.append(accumulator)
        return "\n".join(adjusted_lines)

    def _encode_image(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Clean decoder artifacts and strip unsupported glyphs."""
        cleaned = unicodedata.normalize("NFKC", text)

        bad_chars = {
            "▁": " ",
            "▯": "",
            "□": "",
            "■": "",
            "◻": "",
            "◽": "",
            "�": "",
            "​": "",
            "‌": "",
            "‍": "",
            " ": " ",
            " ": " ",
            " ": " ",
            " ": " ",
            " ": " ",
            " ": " ",
            " ": " ",
            " ": " ",
            " ": " ",
            " ": " ",
            " ": " ",
            " ": " ",
            " ": " ",
            " ": " ",
            " ": " ",
            " ": " ",
            " ": " ",
            "　": " ",
            "<unk>": "",
            "[UNK]": "",
            "<s>": "",
            "</s>": "",
        }

        punct_map = str.maketrans({
            "…": "...",
            "。": ".",
            "、": ",",
            "！": "!",
            "？": "?",
            "：": ":",
            "；": ";",
            "“": """,
            "”": """,
            "‘": "'",
            "’": "'",
            "【": "[",
            "】": "]",
            "「": """,
            "」": """,
            "『": """,
            "』": """,
            "（": "(",
            "）": ")",
            "－": "-",
            "—": "-",
            "–": "-",
            "‥": "..",
            "⁇": "??",
            "⁈": "?!",
            "⁉": "!?",
            "﹖": "?",
            "﹗": "!",
            "﹑": ",",
            "﹒": ".",
            "･": ".",
        })

        cleaned = cleaned.translate(punct_map)
        for src, repl in bad_chars.items():
            cleaned = cleaned.replace(src, repl)

        cleaned = unicodedata.normalize("NFC", cleaned)
        cleaned = " ".join(cleaned.split())

        # Sarabun font supports Thai, Latin, digits, and common punctuation
        allowed_punct = set("!?,.:;\"'()-...[]")
        allowed_chars = []
        for ch in cleaned:
            # Thai characters
            if "\u0e00" <= ch <= "\u0e7f":
                allowed_chars.append(ch)
            # Latin letters (A-Z, a-z)
            elif "A" <= ch <= "Z" or "a" <= ch <= "z":
                allowed_chars.append(ch)
            # Digits
            elif "0" <= ch <= "9":
                allowed_chars.append(ch)
            # Whitespace
            elif ch in (" ", "\n", "\t"):
                allowed_chars.append(ch)
            # Allowed punctuation
            elif ch in allowed_punct:
                allowed_chars.append(ch)
            # Skip unsupported characters (Japanese, etc.)

        result_chars = []
        for ch in allowed_chars:
            cat = unicodedata.category(ch)
            if cat.startswith("M") and not result_chars:
                continue
            result_chars.append(ch)
        result = "".join(result_chars)
        # Always return filtered result, never fall back to cleaned which may have unsupported chars
        return result.strip() if result.strip() else ""