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

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.transformer')

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.enhancer import DocumentEnhancer, EnhanceParams, PRESETS
from core.ocr_engine import OCREngine, OCRResult, preprocess_order_image, PaddleOCRDetector, VietOCREngine, crop_text_box
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
        """Test the 3-step offline OCR: Step 1 (Preprocessing/Deskew), Step 2 (PaddleOCR Detection & Crop), Step 3 (VietOCR Recognition)."""
        from core.ocr_engine import preprocess_order_image, PaddleOCRDetector, VietOCREngine, crop_text_box

        # --- Bước 1: Tiền xử lý ảnh (Pre-processing) ---
        deskewed_color, thresh_img, angle = preprocess_order_image(self.test_img, deskew=True, apply_adaptive_thresh=False)
        self.assertIsNotNone(deskewed_color)
        self.assertEqual(deskewed_color.shape[:2], self.test_img.shape[:2])

        # --- Bước 2: Định vị vùng chữ với PaddleOCR DBNet (Detection Only) ---
        det = PaddleOCRDetector.get_instance()
        boxes = det.detect(deskewed_color)
        self.assertGreater(len(boxes), 0, "PaddleOCR DBNet should detect at least one text box")

        # Cắt ảnh (Crop)
        cropped_box = crop_text_box(deskewed_color, boxes[0], padding=2)
        self.assertIsNotNone(cropped_box)
        self.assertGreater(cropped_box.size, 0)

        # --- Bước 3: Nhận diện chữ tiếng Việt bằng VietOCR ---
        vietocr = VietOCREngine.get_instance()
        sample_crop_text = vietocr.predict_image(cropped_box)
        self.assertIsInstance(sample_crop_text, str)

        # Toàn bộ pipeline nhận diện Offline OCREngine (trả về văn bản thuần sạch)
        ocr = OCREngine.get_instance()
        ocr.engine_mode = "offline"
        res = ocr.recognize(self.test_img, mode="offline")

        self.assertIsNone(res.error)
        self.assertGreater(len(res.boxes), 0, "Should detect at least one text box")
        self.assertTrue("4K" in res.full_text or "DOCUMENT" in res.full_text or len(res.full_text) > 5)
        self.assertNotIn("📋 KẾT QUẢ BÓC TÁCH CẤU TRÚC ĐƠN HÀNG", res.full_text)

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
        self.assertTrue("4K" in pdf_text or "DOCUMENT" in pdf_text)
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
        for w in win._active_enhance_workers:
            w.wait(2000)
        app.processEvents()

        # Clean up
        win.close()
        if os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except Exception:
                pass

    def test_06_multi_image_rapid_switch(self):
        """Stress test: upload 3 images at once and rapidly switch between them."""
        from PySide6.QtWidgets import QApplication
        from ui.main_window import MainWindow

        app = QApplication.instance() or QApplication(sys.argv)
        win = MainWindow()

        # Create 3 distinct images
        temp_paths = []
        for i in range(3):
            path = os.path.join(os.path.dirname(__file__), f"temp_multi_test_{i}.png")
            img = self.test_img.copy()
            cv2.putText(img, f"Page {i + 1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 2)
            cv2.imwrite(path, img)
            temp_paths.append(path)

        # Upload 3 images at once
        win.image_list.add_images(temp_paths)
        self.assertEqual(len(win.image_list.items), 3)

        # Rapidly switch between pages back and forth 15 times
        sequence = [0, 1, 2, 0, 2, 1, 0, 1, 2, 2, 1, 0, 1, 2, 0]
        for target_page in sequence:
            win.image_list.list_widget.setCurrentRow(target_page)
            app.processEvents()

        # Let any remaining background worker finish
        for w in list(win._active_enhance_workers):
            w.wait(3000)
        app.processEvents()

        # Now test switching to cached pages (should be instant, no new workers)
        active_count_before = len(win._active_enhance_workers)
        win.image_list.list_widget.setCurrentRow(0)
        app.processEvents()
        self.assertEqual(len(win._active_enhance_workers), active_count_before)

        # Clean up
        win.close()
        for p in temp_paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


    def test_07_auto_deskew_straight_and_tilted(self):
        """Test that auto-deskew does not distort straight documents and straightens tilted ones."""
        # 1. Straight image must return 0.0 angle
        gray_straight = cv2.cvtColor(self.test_img, cv2.COLOR_BGR2GRAY)
        ang_straight = DocumentEnhancer.detect_skew_angle(gray_straight)
        self.assertEqual(ang_straight, 0.0, "Straight document must have 0.0 detected skew")

        # 2. Tilted image (+5.0 degrees)
        h, w = self.test_img.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, 5.0, 1.0)
        tilted = cv2.warpAffine(self.test_img, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        p = EnhanceParams(auto_deskew=True, upscale_factor=1.0)
        deskewed = DocumentEnhancer.process(tilted, p)
        gray_deskewed = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
        ang_after = DocumentEnhancer.detect_skew_angle(gray_deskewed)
        self.assertEqual(ang_after, 0.0, "After deskew, image must be straight (0.0 skew)")


    def test_08_auto_crop_and_flatten(self):
        """Test smart document boundary detection, background crop, and perspective rectification."""
        from core.flattener import DocumentFlattener

        # 1. Test full-page scan: should NOT be cropped (safety guard)
        full_res, full_cropped = DocumentFlattener.crop_and_flatten(self.test_img)
        self.assertFalse(full_cropped, "Full-page scan should not be erroneously cropped")

        # 2. Test document sheet lying on a textured wooden table
        table_h, table_w = 900, 1100
        # Dark brown table
        table = np.zeros((table_h, table_w, 3), dtype=np.uint8)
        table[:, :] = [45, 75, 115]  # BGR brown wood tone

        # Place a white tilted document sheet on the table
        doc_pts = np.array([
            [120, 100],   # Top-left
            [950, 80],    # Top-right
            [980, 800],   # Bottom-right
            [80, 820]     # Bottom-left
        ], dtype=np.int32)
        cv2.fillPoly(table, [doc_pts], (245, 245, 245))

        # Add text on the document sheet
        cv2.putText(table, "HOA DON BAN HANG", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2)
        cv2.putText(table, "TONG TIEN: 500,000 VND", (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)

        # Run Auto-Crop & Flatten
        flattened, was_flattened = DocumentFlattener.crop_and_flatten(table)
        self.assertTrue(was_flattened, "Should detect document boundaries and crop the table")
        self.assertIsNotNone(flattened)

        # Verify the output is flattened to document dimensions (not table dimensions)
        self.assertLess(flattened.shape[0], table_h, "Output height must be cropped")
        self.assertLess(flattened.shape[1], table_w, "Output width must be cropped")
        self.assertGreater(flattened.shape[0], 650, "Document height should be preserved")
        self.assertGreater(flattened.shape[1], 750, "Document width should be preserved")

        # Check integrated pipeline in DocumentEnhancer
        params = EnhanceParams(auto_flatten=True, sharpness=0.5, upscale_factor=1.0)
        pipeline_out = DocumentEnhancer.process(table, params)
        self.assertIsNotNone(pipeline_out)
        self.assertLess(pipeline_out.shape[0], table_h)
        self.assertLess(pipeline_out.shape[1], table_w)
        print(f"\n[Test] Auto-Crop & Flatten: Table ({table_w}x{table_h}) -> Document ({pipeline_out.shape[1]}x{pipeline_out.shape[0]})")

    def test_09_folder_and_multipage_loader(self):
        """Test natural sorting and multi-page loader (images + PDF + JFIF)."""
        from ui.image_list_widget import load_document_pages, natural_sort_key, SUPPORTED_ALL_EXTS, ImageListWidget
        import tempfile
        import shutil
        from PIL import Image

        # Test natural sort key
        unsorted = ["doc_10.png", "doc_1.png", "doc_2.png", "doc_20.png"]
        sorted_res = sorted(unsorted, key=natural_sort_key)
        self.assertEqual(sorted_res, ["doc_1.png", "doc_2.png", "doc_10.png", "doc_20.png"])

        # Test temporary folder loading
        temp_dir = tempfile.mkdtemp(prefix="test_ocr_folder_")
        try:
            # 1. Write sample images
            cv2.imwrite(os.path.join(temp_dir, "doc_1.png"), self.test_img)
            cv2.imwrite(os.path.join(temp_dir, "doc_2.png"), self.test_img)
            cv2.imwrite(os.path.join(temp_dir, "doc_10.png"), self.test_img)

            # 2. Write a 2-page PDF
            pdf_doc = pymupdf.open()
            pdf_doc.new_page(width=200, height=200)
            pdf_doc.new_page(width=200, height=200)
            pdf_path = os.path.join(temp_dir, "multi_page.pdf")
            pdf_doc.save(pdf_path)
            pdf_doc.close()

            # 3. Write a JFIF image
            jfif_path = os.path.join(temp_dir, "test_scan.jfif")
            pil_img = Image.fromarray(cv2.cvtColor(self.test_img, cv2.COLOR_BGR2RGB))
            pil_img.save(jfif_path, format="JPEG")

            # Collect paths as prompt_add_folder would
            found_paths = []
            for root, dirs, files in os.walk(temp_dir):
                dirs.sort(key=natural_sort_key)
                for f in sorted(files, key=natural_sort_key):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in SUPPORTED_ALL_EXTS:
                        found_paths.append(os.path.join(root, f))

            self.assertEqual(len(found_paths), 5)

            # Test ImageListWidget
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance() or QApplication(sys.argv)
            widget = ImageListWidget()
            widget.add_images(found_paths)

            # 3 PNGs + 1 JFIF + 2 pages from PDF = 6 total items
            self.assertEqual(len(widget.items), 6, f"Expected 6 items, got {len(widget.items)}")
            
            # Check filename formatting
            filenames = [item.filename for item in widget.items]
            self.assertTrue(any("Trang 1/2" in fn for fn in filenames))
            self.assertTrue(any("Trang 2/2" in fn for fn in filenames))
            self.assertTrue(any("doc_1.png" in fn for fn in filenames))
            self.assertTrue(any("test_scan.jfif" in fn for fn in filenames))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_10_app_icon_transparency(self):
        """Verify that assets/logo.png and assets/logo.ico exist with true alpha transparency."""
        from PIL import Image

        logo_png = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
        logo_ico = os.path.join(os.path.dirname(__file__), "assets", "logo.ico")

        self.assertTrue(os.path.exists(logo_png), "assets/logo.png must exist")
        self.assertTrue(os.path.exists(logo_ico), "assets/logo.ico must exist")

        with Image.open(logo_png) as im:
            self.assertEqual(im.mode, "RGBA", "Logo PNG must have RGBA format")
            alpha_band = im.split()[-1]
            min_alpha = min(alpha_band.getchannel(0).tobytes() if hasattr(alpha_band, 'getchannel') else alpha_band.tobytes())
            self.assertEqual(min_alpha, 0, "Logo PNG must have 100% transparent pixels (alpha = 0)")

    def test_11_jxl_and_system_tray(self):
        """Test JXL format loading and SystemTrayIcon synchronization."""
        from ui.image_list_widget import load_document_pages, generate_fast_thumbnail, SUPPORTED_ALL_EXTS
        from ui.main_window import MainWindow
        from PySide6.QtWidgets import QApplication

        # 1. Test JXL format loading if tuning_data is present
        tuning_dir = os.path.join(os.path.dirname(__file__), "tuning_data")
        if os.path.exists(tuning_dir):
            jxl_files = [f for f in os.listdir(tuning_dir) if f.endswith(".jxl")]
            if jxl_files:
                sample_jxl = os.path.join(tuning_dir, jxl_files[0])
                self.assertTrue(".jxl" in SUPPORTED_ALL_EXTS, ".jxl must be supported")
                pages = load_document_pages(sample_jxl)
                self.assertGreater(len(pages), 0, "Must extract at least one page from JXL")
                self.assertIsNotNone(pages[0][1], "Decoded image must not be None")
                self.assertEqual(len(pages[0][1].shape), 3, "Decoded image must be 3-channel BGR")

                # Test fast thumbnail
                thumb = generate_fast_thumbnail(sample_jxl, 0)
                self.assertIsNotNone(thumb, "Fast thumbnail should generate for JXL")
                self.assertFalse(thumb.isNull(), "Thumbnail must not be null")

        # 2. Test MainWindow System Tray and Screen Adaptation
        app = QApplication.instance() or QApplication(sys.argv)
        win = MainWindow()
        self.assertIsNotNone(win.tray, "MainWindow must initialize SystemTrayIcon")
        self.assertFalse(win.tray.icon().isNull(), "Tray icon must not be null")
        self.assertGreaterEqual(win.width(), 850, "Window width must satisfy minimum size")
        self.assertGreaterEqual(win.height(), 520, "Window height must satisfy minimum size")

        win.tray.hide()
        win.close()

    def test_12_dynamic_adaptive_padding(self):
        """Test Dynamic Adaptive Padding in crop_text_box."""
        from core.ocr_engine import crop_text_box
        sample_img = np.full((300, 500, 3), 240, dtype=np.uint8)
        poly = [[50.0, 50.0], [250.0, 50.0], [250.0, 90.0], [50.0, 90.0]]  # w=200, h=40

        crop_fixed = crop_text_box(sample_img, poly, padding=2, adaptive_padding=False)
        crop_adapt = crop_text_box(sample_img, poly, adaptive_padding=True)

        self.assertIsNotNone(crop_fixed)
        self.assertIsNotNone(crop_adapt)
        # Adaptive padding uses 16% height (~6px vertical padding), which is greater than 2px
        self.assertGreater(crop_adapt.shape[0], crop_fixed.shape[0], "Adaptive padding must allocate more vertical space for diacritics")
        self.assertGreater(crop_adapt.shape[1], crop_fixed.shape[1])

    def test_13_bigram_context_disambiguation(self):
        """Test Vietnamese Bi-Gram Language Model disambiguation for homophones/unaccented words."""
        from core.ocr_engine import VietnameseLanguageModel
        lm = VietnameseLanguageModel.get_instance()

        t1 = lm.process("kiem tra toan bo")
        t2 = lm.process("chuc mung nam moi")
        t3 = lm.process("thanh toan tien hang")

        self.assertEqual(t1, "kiểm tra toàn bộ")
        self.assertEqual(t2, "chúc mừng năm mới")
        self.assertEqual(t3, "thanh toán tiền hàng")

    def test_14_column_segmentation_xy_cut(self):
        """Test Recursive XY-Cut multi-column layout preservation."""
        from core.ocr_engine import OCREngine, OCRBox
        ocr = OCREngine.get_instance()

        boxes = [
            OCRBox(polygon=[], bbox=(50, 100, 300, 30), text="Cột 1 Dòng 1", confidence=0.9),
            OCRBox(polygon=[], bbox=(50, 150, 300, 30), text="Cột 1 Dòng 2", confidence=0.9),
            OCRBox(polygon=[], bbox=(50, 200, 300, 30), text="Cột 1 Dòng 3", confidence=0.9),
            OCRBox(polygon=[], bbox=(620, 100, 300, 30), text="Cột 2 Dòng 1", confidence=0.9),
            OCRBox(polygon=[], bbox=(620, 150, 300, 30), text="Cột 2 Dòng 2", confidence=0.9),
            OCRBox(polygon=[], bbox=(620, 200, 300, 30), text="Cột 2 Dòng 3", confidence=0.9),
        ]

        cols = ocr._detect_columns(boxes, img_w=1000, img_h=1200)
        self.assertEqual(len(cols), 2, "Must identify 2 separate columns")

        lines = ocr._group_into_lines(boxes, img_w=1000, img_h=1200)
        self.assertEqual(len(lines), 6, "Must form 6 distinct reading lines")
        # Column 1 lines must come before Column 2 lines in natural reading order
        self.assertTrue("Cột 1" in lines[0][0].text)
        self.assertTrue("Cột 1" in lines[2][0].text)
        self.assertTrue("Cột 2" in lines[3][0].text)

    def test_15_dual_engine_arbitration(self):
        """Test DualEngineArbitrator voting logic."""
        from core.ocr_engine import DualEngineArbitrator

        # 1. MST / Numeric codes -> PaddleOCR priority
        t1, c1, w1 = DualEngineArbitrator.arbitrate("0106034111-001", 0.95, "0106034111-00l", 0.75)
        self.assertEqual(t1, "0106034111-001")
        self.assertEqual(w1, "paddleocr")

        # 2. Cross-validation agreement -> VietOCR with max confidence
        t2, c2, w2 = DualEngineArbitrator.arbitrate(
            "CONG TY TNHH MAI SONG NGUYEN", 0.91, "CÔNG TY TNHH MAI SONG NGUYÊN", 0.88
        )
        self.assertEqual(t2, "CÔNG TY TNHH MAI SONG NGUYÊN")
        self.assertEqual(w2, "cross_verified")
        self.assertGreaterEqual(c2, 0.98)

        # 3. Vietnamese Diacritics -> VietOCR priority
        t3, c3, w3 = DualEngineArbitrator.arbitrate("Hop dong kinh te", 0.80, "Hợp đồng kinh tế", 0.92)
        self.assertEqual(t3, "Hợp đồng kinh tế")


if __name__ == "__main__":
    unittest.main(warnings='ignore')
