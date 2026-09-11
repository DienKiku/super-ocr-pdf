"""
Prepare PaddleOCR Text Recognition Fine-tuning Dataset.
1. Reads documents from tuning_data/ (supports JPEG XL .jxl and .jpg).
2. Performs preprocessing (deskew, contrast enhancement).
3. Uses PP-OCRv4 DBNet to detect text bounding boxes.
4. Crops each line chip and predicts ground-truth text with VietOCR transformer.
5. Saves cropped chips to train_data/rec_images/ and writes train.txt / val.txt.
6. Builds comprehensive viet_dict.txt character dictionary.
"""

import os
import sys
import random
import unicodedata
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.ocr_engine import (
    PaddleOCRDetector,
    VietOCREngine,
    SmartVietnameseRestorer,
    crop_text_box,
    preprocess_order_image,
    get_base_dir
)

try:
    import pillow_jxl
    HAS_JXL = True
except ImportError:
    HAS_JXL = False


def load_image(filepath: str) -> np.ndarray:
    """Load image from path supporting .jxl and standard formats."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".jxl" or HAS_JXL:
        try:
            pil_img = Image.open(filepath)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            # Convert RGB (PIL) to BGR (OpenCV)
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    # Fallback to OpenCV
    img = cv2.imread(filepath)
    if img is not None:
        return img

    # Fallback to PIL
    try:
        pil_img = Image.open(filepath)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def generate_dataset(
    source_dir: str = "tuning_data",
    output_dir: str = "train_data",
    max_documents: int = 100,
    val_split: float = 0.15,
    random_seed: int = 42
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    images_out_dir = os.path.join(output_dir, "rec_images")
    os.makedirs(images_out_dir, exist_ok=True)

    print("Initializing PP-OCRv4 Detector and VietOCR...")
    detector = PaddleOCRDetector.get_instance()
    vietocr = VietOCREngine.get_instance()
    base_dir = get_base_dir()
    dict_path = os.path.join(base_dir, "core", "models", "viet_words.txt")
    restorer = SmartVietnameseRestorer(dict_path)

    # Collect files, excluding duplicates with '- Copy'
    all_files = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith((".jxl", ".jpg", ".png", ".jpeg"))
        and " - copy" not in f.lower()
    ]
    random.shuffle(all_files)
    selected_files = all_files[:max_documents]
    print(f"Found {len(all_files)} unique documents. Processing up to {len(selected_files)} documents...")

    dataset_entries: List[Tuple[str, str]] = []
    char_set = set()
    total_crops = 0

    for doc_idx, filename in enumerate(selected_files):
        filepath = os.path.join(source_dir, filename)
        try:
            img = load_image(filepath)
            if img is None or img.size == 0:
                continue

            # Resize if too large to speed up detection while maintaining high quality
            h, w = img.shape[:2]
            max_side = 2000
            if max(h, w) > max_side:
                scale = max_side / float(max(h, w))
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            # Deskew and preprocess
            prep, deskewed_color, angle = preprocess_order_image(img, deskew=True, apply_adaptive_thresh=True)

            polygons = detector.detect(prep)
            if not polygons and deskewed_color is not prep:
                polygons = detector.detect(deskewed_color)

            if not polygons:
                continue

            # Sort boxes top-to-bottom
            boxes_info = []
            for poly in polygons:
                ys = [p[1] for p in poly]
                xs = [p[0] for p in poly]
                boxes_info.append((min(ys), min(xs), poly))
            boxes_info.sort(key=lambda item: (item[0], item[1]))

            for _, _, poly in boxes_info:
                crop = crop_text_box(deskewed_color, poly, padding=2)
                if crop is None or crop.shape[0] < 10 or crop.shape[1] < 12:
                    continue

                text = vietocr.predict_image(crop)
                if not text:
                    continue

                text = unicodedata.normalize("NFC", text.strip())
                text = restorer.restore_line(text)

                # Quality checks: must have alphanumeric, length >= 2
                has_alphanumeric = any(c.isalnum() for c in text)
                if not has_alphanumeric or len(text) < 2:
                    continue

                # Save crop
                total_crops += 1
                crop_filename = f"crop_{total_crops:06d}.jpg"
                crop_rel_path = f"rec_images/{crop_filename}"
                crop_full_path = os.path.join(images_out_dir, crop_filename)
                cv2.imwrite(crop_full_path, crop)

                dataset_entries.append((crop_rel_path, text))
                for char in text:
                    char_set.add(char)

            if (doc_idx + 1) % 10 == 0 or (doc_idx + 1) == len(selected_files):
                print(f"[{doc_idx + 1}/{len(selected_files)}] Generated {total_crops} cropped lines so far...")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nCompleted crop extraction! Total crops: {len(dataset_entries)}")

    # Shuffle and split into train / val
    random.shuffle(dataset_entries)
    val_count = max(10, int(len(dataset_entries) * val_split))
    val_entries = dataset_entries[:val_count]
    train_entries = dataset_entries[val_count:]

    train_txt_path = os.path.join(output_dir, "train.txt")
    val_txt_path = os.path.join(output_dir, "val.txt")

    with open(train_txt_path, "w", encoding="utf-8") as f:
        for rel_path, text in train_entries:
            f.write(f"{rel_path}\t{text}\n")

    with open(val_txt_path, "w", encoding="utf-8") as f:
        for rel_path, text in val_entries:
            f.write(f"{rel_path}\t{text}\n")

    print(f"Wrote {len(train_entries)} samples to {train_txt_path}")
    print(f"Wrote {len(val_entries)} samples to {val_txt_path}")

    # Build comprehensive Vietnamese dictionary
    base_chars = (
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "aàáảãạăằắẳẵặâầấẩẫậ"
        "bcdđeèéẻẽẹêềếểễệ"
        "ghiklmnoòóỏõọôồốổỗộơờớởỡợ"
        "pqrstuùúủũụưừứửữựvxyỳýỷỹỵ"
        "AÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ"
        "BCDĐEÈÉẺẼẸÊỀẾỂỄỆ"
        "GHIKLMNOÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
        "PQRSTUÙÚỦŨỤƯỪỨỬỮỰVXYỲÝỶỸỴ"
        " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        "£€¥©®™°±×÷≠≤≥…–—“”‘’"
    )
    for c in base_chars:
        char_set.add(c)

    # Sort characters deterministically (exclude space since use_space_char: true appends it)
    sorted_chars = sorted([c for c in char_set if c != " " and c != "\n" and c != "\r" and c != "\t"])
    dict_path = os.path.join(output_dir, "viet_dict.txt")
    with open(dict_path, "w", encoding="utf-8") as f:
        for c in sorted_chars:
            f.write(f"{c}\n")

    print(f"Wrote dictionary with {len(sorted_chars)} characters to {dict_path}")


if __name__ == "__main__":
    max_docs = 100
    if len(sys.argv) > 1:
        max_docs = int(sys.argv[1])
    generate_dataset(max_documents=max_docs)
