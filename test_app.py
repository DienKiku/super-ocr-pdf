"""
Automated Comprehensive Test Suite for Super OCR & High-Res PDF Studio.
Tests enhancement, super-resolution, OCR extraction, Searchable PDF creation,
and UI initialization.
"""

import sys
import os
import unittest
import numpy as np
import cv2
import pymupdf

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.enhancer import DocumentEnhancer, EnhanceParams, PRESETS
from core.ocr_engine import OCREngine, OCRResult
from core.pdf_builder import PDFBuilder, PageData, PDFConfig


class TestSuperOCRPDF(unittest.TestCase):

    def setUp(self):
        # Create a synthetic blurry/shadowed document image
        self.img_h, self.img_w = 400, 700
        self.test_img = np.full((self.img_h, self.img_w, 3), 235, dtype=np.uint8)

        # Add uneven lighting / shadow gradient across image
        for c in range(self.img_w):
            factor = 1.0 - 0.35 * (1.0 - c / self.img_w)
            self.test_img[:, c] = np.clip(self.test_img[:, c] * factor, 0, 255).astype(np.uint8)

        # Add text
        cv2.putText(
            self.test_img,
            "CONG NGHE SIEU PHAN GIAI 4K",
            (40, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (30, 30, 30),
            2
        )
        cv2.putText(
            self.test_img,
            "Document Text Sharpening Test 2026",
            (40, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (40, 40, 40),
            2
        )

        # Add synthetic motion blur
        kernel_size = 5
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))
        kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel_motion_blur = kernel_motion_blur / kernel_size
        self.blurry_img = cv2.filter2D(self.test_img, -1, kernel_motion_blur)

    def test_01_enhancement_presets(self):
        """Test all enhancement presets and verify resolution upscale."""
        for preset_name, params in PRESETS.items():
            result = DocumentEnhancer.process(self.blurry_img, params)
            self.assertIsNotNone(result)
            expected_w = int(self.img_w * params.upscale_factor)
            expected_h = int(self.img_h * params.upscale_factor)
            self.assertEqual(result.shape[1], expected_w, f"Width mismatch for preset {preset_name}")
            self.assertEqual(result.shape[0], expected_h, f"Height mismatch for preset {preset_name}")

    def test_02_sharpness_improvement(self):
        """Verify that sharpness (Laplacian variance) increases after processing."""
        params = EnhanceParams(sharpness=1.0, contrast=0.4, whitening=0.6, upscale_factor=1.0)
        enhanced = DocumentEnhancer.process(self.blurry_img, params)

        gray_orig = cv2.cvtColor(self.blurry_img, cv2.COLOR_BGR2GRAY)
        gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        var_orig = cv2.Laplacian(gray_orig, cv2.CV_64F).var()
        var_enh = cv2.Laplacian(gray_enh, cv2.CV_64F).var()

        print(f"\n[Test] Laplacian Variance - Blurry: {var_orig:.1f} -> Enhanced: {var_enh:.1f}")
        self.assertGreater(var_enh, var_orig, "Enhanced image should have significantly higher edge contrast")

    def test_03_ocr_recognition(self):
        """Test RapidOCR text recognition and bounding box extraction."""
        ocr = OCREngine.get_instance()
        res = ocr.recognize(self.test_img)

        self.assertIsNone(res.error)
        self.assertGreater(len(res.boxes), 0, "Should detect at least one text box")
        self.assertIn("PHAN GIAI", res.full_text.upper())
        print(f"[Test] OCR detected {len(res.boxes)} boxes, total text:\n{res.full_text}")

    def test_04_searchable_pdf_generation(self):
        """Test generating a Searchable PDF and verify extracted text layer."""
        params = EnhanceParams(sharpness=0.8, upscale_factor=2.0)
        enhanced = DocumentEnhancer.process(self.test_img, params)

        ocr = OCREngine.get_instance()
        ocr_res = ocr.recognize(enhanced)

        page1 = PageData(image=enhanced, ocr_result=ocr_res)
        out_pdf = os.path.join(os.path.dirname(__file__), "test_sample_output.pdf")

        cfg = PDFConfig(searchable=True, page_size="fit_image", jpeg_quality=95)
        success = PDFBuilder.create_pdf([page1], out_pdf, cfg)

        self.assertTrue(success)
        self.assertTrue(os.path.exists(out_pdf))
        self.assertGreater(os.path.getsize(out_pdf), 1000)

        # Open with PyMuPDF and check text layer
        doc = pymupdf.open(out_pdf)
        self.assertEqual(len(doc), 1)
        pdf_text = doc[0].get_text()
        print(f"[Test] PDF embedded text layer length: {len(pdf_text)}")
        self.assertIn("PHAN", pdf_text.upper())
        doc.close()

        # Clean up
        if os.path.exists(out_pdf):
            os.remove(out_pdf)

    def test_05_gui_initialization(self):
        """Test that PySide6 MainWindow initializes without errors."""
        from PySide6.QtWidgets import QApplication
        from ui.main_window import MainWindow

        app = QApplication.instance() or QApplication(sys.argv)
        win = MainWindow()
        self.assertIsNotNone(win)
        self.assertEqual(len(win.image_list.items), 0)

        # Test adding an image via synthetic save
        temp_img_path = os.path.join(os.path.dirname(__file__), "temp_test_page.png")
        cv2.imwrite(temp_img_path, self.test_img)

        win.image_list.add_images([temp_img_path])
        self.assertEqual(len(win.image_list.items), 1)

        # Rotate
        win.image_list.rotate_current(90)
        self.assertEqual(win.image_list.items[0].rotation, 90)

        # Wait for any active enhance worker
        if win.enhance_worker and win.enhance_worker.isRunning():
            win.enhance_worker.wait(2000)
        app.processEvents()

        # Clean up
        win.close()
        if os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
