"""
Prepare and format dataset for VietOCR fine-tuning.
Combines invoice/order text crops with Vietnamese handwriting crops.
Builds LMDB datasets compatible with NumPy 2.x and VietOCR.
"""

import os
import sys
import random
import unicodedata
import cv2
import numpy as np

# Apply NumPy 2.x compatibility shims
np.sctypes = {
    'int': [np.int8, np.int16, np.int32, np.int64],
    'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
    'float': [np.float16, np.float32, np.float64],
    'complex': [np.complex64, np.complex128],
    'others': [bool, object, bytes, str, np.void]
}

_orig_fromstring = np.fromstring
def _safe_fromstring(string, dtype=float, count=-1, sep=''):
    if sep == '' and isinstance(string, (bytes, bytearray, memoryview)):
        return np.frombuffer(string, dtype=dtype, count=count)
    return _orig_fromstring(string, dtype=dtype, count=count, sep=sep)
np.fromstring = _safe_fromstring

from vietocr.tool.create_dataset import createDataset


def prepare_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "vietocr_data")
    os.makedirs(out_dir, exist_ok=True)

    samples = []

    # 1. Load order / invoice crops from train_data/
    train_data_txt = os.path.join(base_dir, "train_data", "train.txt")
    if os.path.exists(train_data_txt):
        with open(train_data_txt, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rel_img, text = parts[0], parts[1]
                    img_path = os.path.join(base_dir, "train_data", rel_img)
                    text_clean = unicodedata.normalize("NFC", text.strip())
                    if os.path.exists(img_path) and len(text_clean) > 0:
                        samples.append((img_path, text_clean))

    val_data_txt = os.path.join(base_dir, "train_data", "val.txt")
    if os.path.exists(val_data_txt):
        with open(val_data_txt, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rel_img, text = parts[0], parts[1]
                    img_path = os.path.join(base_dir, "train_data", rel_img)
                    text_clean = unicodedata.normalize("NFC", text.strip())
                    if os.path.exists(img_path) and len(text_clean) > 0:
                        samples.append((img_path, text_clean))

    print(f"Loaded {len(samples)} order/invoice samples.")

    # 2. Load Vietnamese handwriting samples from cinnamon_dataset/
    cinnamon_txt = os.path.join(base_dir, "train_data_cinnamon", "train.txt")
    cinnamon_img_dir = os.path.join(base_dir, "cinnamon_dataset", "vn_handwritten_images", "data")
    cinnamon_count = 0
    if os.path.exists(cinnamon_txt) and os.path.exists(cinnamon_img_dir):
        with open(cinnamon_txt, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rel_img, text = parts[0], parts[1]
                    img_path = os.path.join(cinnamon_img_dir, rel_img)
                    text_clean = unicodedata.normalize("NFC", text.strip())
                    if os.path.exists(img_path) and len(text_clean) > 0:
                        samples.append((img_path, text_clean))
                        cinnamon_count += 1

    print(f"Loaded {cinnamon_count} handwriting samples.")
    print(f"Total raw samples: {len(samples)}")

    # Filter invalid images
    valid_samples = []
    for img_path, label in samples:
        try:
            # Quick shape check
            h, w = cv2.imread(img_path).shape[:2]
            if h >= 8 and w >= 8 and len(label) <= 120:
                valid_samples.append((img_path, label))
        except Exception:
            continue

    print(f"Total validated samples: {len(valid_samples)}")

    # Shuffle with fixed seed
    random.seed(42)
    random.shuffle(valid_samples)

    # Split: 90% train, 10% val
    split_idx = int(len(valid_samples) * 0.9)
    train_samples = valid_samples[:split_idx]
    val_samples = valid_samples[split_idx:]

    train_txt_path = os.path.join(out_dir, "train_annotation.txt")
    val_txt_path = os.path.join(out_dir, "val_annotation.txt")

    with open(train_txt_path, "w", encoding="utf-8") as f:
        for img_path, label in train_samples:
            f.write(f"{img_path}\t{label}\n")

    with open(val_txt_path, "w", encoding="utf-8") as f:
        for img_path, label in val_samples:
            f.write(f"{img_path}\t{label}\n")

    print(f"Train samples: {len(train_samples)} -> {train_txt_path}")
    print(f"Validation samples: {len(val_samples)} -> {val_txt_path}")

    # Build LMDB databases
    train_lmdb = os.path.join(out_dir, "train_lmdb")
    val_lmdb = os.path.join(out_dir, "val_lmdb")

    if not os.path.exists(train_lmdb):
        print("Creating train LMDB...")
        createDataset(train_lmdb, "", train_txt_path)
    else:
        print(f"train LMDB already exists: {train_lmdb}")

    if not os.path.exists(val_lmdb):
        print("Creating val LMDB...")
        createDataset(val_lmdb, "", val_txt_path)
    else:
        print(f"val LMDB already exists: {val_lmdb}")

    print("Dataset preparation complete!")


if __name__ == "__main__":
    prepare_dataset()
