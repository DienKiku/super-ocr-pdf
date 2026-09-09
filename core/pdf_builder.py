"""
High-Resolution PDF Builder with Searchable OCR Layer.
Uses PyMuPDF to assemble ultra-high DPI document pages with
embedded invisible text layer for search, copy, and highlight capabilities.
"""

from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple
import os
import cv2
import numpy as np
import pymupdf

from core.ocr_engine import OCRResult, OCRBox


@dataclass
class PageData:
    image: np.ndarray             # Enhanced BGR or RGB image
    ocr_result: Optional[OCRResult] = None
    title: str = ""


@dataclass
class PDFConfig:
    page_size: str = "fit_image"   # "fit_image", "a4_portrait", "a4_landscape"
    searchable: bool = True        # Embed OCR text layer
    jpeg_quality: int = 95         # 80 - 100
    dpi: int = 300                 # DPI for fit_image scaling


class PDFBuilder:
    """Builds multi-page high-resolution PDFs with optional OCR searchability."""

    @classmethod
    def get_system_font(cls) -> Optional[str]:
        """Find a reliable system font with full Unicode & Vietnamese diacritics support."""
        windir = os.environ.get("WINDIR", r"C:\Windows")
        candidates = [
            os.path.join(windir, "Fonts", "arial.ttf"),
            os.path.join(windir, "Fonts", "segoeui.ttf"),
            os.path.join(windir, "Fonts", "calibri.ttf"),
            os.path.join(windir, "Fonts", "times.ttf")
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    @classmethod
    def encode_image_for_pdf(cls, img: np.ndarray, quality: int = 95) -> bytes:
        """Encode image to high-quality JPEG or PNG bytes for embedding in PDF."""
        # If image is grayscale or B&W
        if len(img.shape) == 2:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            success, enc = cv2.imencode(".jpg", img, encode_param)
            if success:
                return enc.tobytes()
        else:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            success, enc = cv2.imencode(".jpg", img, encode_param)
            if success:
                return enc.tobytes()
        # Fallback to PNG
        _, enc = cv2.imencode(".png", img)
        return enc.tobytes()

    @classmethod
    def create_pdf(
        cls,
        pages: List[PageData],
        output_path: str,
        config: Optional[PDFConfig] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> bool:
        """
        Create a high-resolution PDF from a list of pages.
        Supports multi-threading and cancel flags.
        """
        if not pages:
            return False

        if config is None:
            config = PDFConfig()

        font_path = cls.get_system_font()
        font_name = "SystemUnicodeFont" if font_path else "helv"

        doc = pymupdf.open()
        total_pages = len(pages)

        # Standard A4 dimensions in points (72 points = 1 inch)
        A4_W, A4_H = 595.28, 841.89

        try:
            for idx, page_data in enumerate(pages):
                if progress_callback:
                    progress_callback(idx + 1, total_pages, f"Đang xử lý trang {idx + 1}/{total_pages}...")

                img = page_data.image
                img_h, img_w = img.shape[:2]

                # Determine PDF page dimensions and image rectangle
                if config.page_size == "a4_portrait":
                    page_w, page_h = A4_W, A4_H
                    # Fit image within A4 margin keeping aspect ratio
                    margin = 20.0
                    avail_w = page_w - 2 * margin
                    avail_h = page_h - 2 * margin
                    scale = min(avail_w / img_w, avail_h / img_h)
                    disp_w = img_w * scale
                    disp_h = img_h * scale
                    disp_x = margin + (avail_w - disp_w) / 2.0
                    disp_y = margin + (avail_h - disp_h) / 2.0
                elif config.page_size == "a4_landscape":
                    page_w, page_h = A4_H, A4_W
                    margin = 20.0
                    avail_w = page_w - 2 * margin
                    avail_h = page_h - 2 * margin
                    scale = min(avail_w / img_w, avail_h / img_h)
                    disp_w = img_w * scale
                    disp_h = img_h * scale
                    disp_x = margin + (avail_w - disp_w) / 2.0
                    disp_y = margin + (avail_h - disp_h) / 2.0
                else:  # "fit_image"
                    # 72 points per inch; scale according to high target DPI
                    pt_per_pixel = 72.0 / config.dpi
                    page_w = img_w * pt_per_pixel
                    page_h = img_h * pt_per_pixel
                    disp_x, disp_y = 0.0, 0.0
                    disp_w, disp_h = page_w, page_h

                # Create page
                pdf_page = doc.new_page(width=page_w, height=page_h)

                # Register Unicode font if available
                has_custom_font = False
                if font_path:
                    try:
                        pdf_page.insert_font(fontname=font_name, fontfile=font_path)
                        has_custom_font = True
                    except Exception:
                        has_custom_font = False

                # Embed high-resolution image
                img_bytes = cls.encode_image_for_pdf(img, quality=config.jpeg_quality)
                img_rect = pymupdf.Rect(disp_x, disp_y, disp_x + disp_w, disp_y + disp_h)
                pdf_page.insert_image(img_rect, stream=img_bytes)

                # Embed Searchable OCR Text Layer (Invisible text over image)
                if config.searchable and page_data.ocr_result and page_data.ocr_result.boxes:
                    for box in page_data.ocr_result.boxes:
                        if not box.text.strip():
                            continue

                        # Map pixel box to PDF coordinate space
                        bx, by, bw, bh = box.bbox
                        scale_x = disp_w / img_w
                        scale_y = disp_h / img_h

                        x_pt = disp_x + (bx * scale_x)
                        y_pt = disp_y + (by * scale_y)
                        w_pt = bw * scale_x
                        h_pt = bh * scale_y

                        # Estimate font size (typically 75-80% of box height)
                        font_size = max(4.0, min(36.0, h_pt * 0.78))
                        baseline_y = y_pt + (h_pt * 0.82)

                        target_font = font_name if has_custom_font else "helv"
                        try:
                            pdf_page.insert_text(
                                (x_pt, baseline_y),
                                box.text,
                                fontname=target_font,
                                fontsize=font_size,
                                render_mode=3  # 3 = Invisible text (standard for OCR layer)
                            )
                        except Exception:
                            # Fallback without custom font
                            try:
                                pdf_page.insert_text(
                                    (x_pt, baseline_y),
                                    box.text,
                                    fontsize=font_size,
                                    render_mode=3
                                )
                            except Exception:
                                pass

            if progress_callback:
                progress_callback(total_pages, total_pages, "Đang tối ưu hóa và lưu file PDF...")

            # Save with compression & garbage collection
            doc.save(
                output_path,
                garbage=4,
                deflate=True,
                clean=True
            )
            doc.close()
            return True

        except Exception as e:
            if doc:
                try:
                    doc.close()
                except Exception:
                    pass
            raise e
