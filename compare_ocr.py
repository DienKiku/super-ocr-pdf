"""
Compare OCR recognition results between:
1. Original baseline model (weights/vgg_transformer_backup.pth)
2. Fine-tuned model (weights/vgg_transformer_finetuned.pth)
"""

import os
import sys
import glob
from PIL import Image

if sys.stdout is not None:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
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

from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

base_dir = os.path.dirname(os.path.abspath(__file__))
weights_dir = os.path.join(base_dir, "weights")
backup_weights = os.path.join(weights_dir, "vgg_transformer_backup.pth")
finetuned_weights = os.path.join(weights_dir, "vgg_transformer_finetuned.pth")

config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained'] = False
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False

print("1. Loading BASELINE model...")
config['weights'] = backup_weights
baseline_detector = Predictor(config)

print("2. Loading FINE-TUNED model...")
config['weights'] = finetuned_weights
finetuned_detector = Predictor(config)

# Find test images from cinnamon_dataset and train_data
test_images = []
cinnamon_imgs = glob.glob(os.path.join(base_dir, "cinnamon_dataset", "**", "*.png"), recursive=True)[:3]
train_imgs = glob.glob(os.path.join(base_dir, "train_data", "**", "*.jpg"), recursive=True)[:3]
test_images = cinnamon_imgs + train_imgs

print("\n" + "=" * 75)
print(f"{'SO SANH KET QUA NHAN DIEN TRUOC & SAU KHI TINH CHINH':^75}")
print("=" * 75)

for idx, img_path in enumerate(test_images, 1):
    fname = os.path.basename(img_path)
    img = Image.open(img_path)
    
    pred_orig = baseline_detector.predict(img)
    pred_fine = finetuned_detector.predict(img)
    
    is_changed = ">>> [THAY DOI]" if pred_orig != pred_fine else "    [GIONG NHAU]"
    
    print(f"\n--- Mau {idx}: {fname} {is_changed} ---")
    print(f"  Goc (Baseline):   {pred_orig}")
    print(f"  Sau tinh chinh:   {pred_fine}")

print("\n" + "=" * 75)
