"""
Fine-tuning VietOCR vgg_transformer on Vietnamese handwriting and invoice documents.
Uses unified dataset from Cinnamon AI and invoice/order crops.
Optimized for CPU training with batch_size=4 and CosineAnnealingLR.
"""

import os
import sys
import time
import shutil
import numpy as np

# Force UTF-8 console output on Windows
if sys.stdout is not None:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

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

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
import vietocr.tool.logger as logger_mod


class SafeLogger:
    def __init__(self, fname):
        path, _ = os.path.split(fname)
        os.makedirs(path, exist_ok=True)
        self.logger = open(fname, 'w', encoding='utf-8')

    def log(self, string):
        self.logger.write(string + '\n')
        self.logger.flush()

    def close(self):
        self.logger.close()


logger_mod.Logger = SafeLogger


def main():
    print("=" * 65)
    print("BAT DAU FINE-TUNING VIETOCR CHO TIENG VIET & CHU VIET TAY")
    print("=" * 65)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(base_dir, "weights")
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    pretrained_path = os.path.join(weights_dir, "vgg_transformer.pth")
    backup_path = os.path.join(weights_dir, "vgg_transformer_backup.pth")
    finetuned_path = os.path.join(weights_dir, "vgg_transformer_finetuned.pth")
    log_file = os.path.join(logs_dir, "train_vietocr.log")
    logger = SafeLogger(log_file)

    if not os.path.exists(pretrained_path):
        print(f"[LOI] Khong tim thay file trong so goc tai: {pretrained_path}")
        sys.exit(1)

    # 1. Backup file trọng số gốc nếu chưa có
    if not os.path.exists(backup_path):
        print(f"[BACKUP] Dang sao luu trong so goc sang: {backup_path}")
        shutil.copyfile(pretrained_path, backup_path)
    else:
        print(f"[BACKUP] Ban sao luu goc da ton tai: {backup_path}")

    # 2. Cấu hình VietOCR vgg_transformer
    total_iters = 400
    batch_size = 4
    print_every = 20
    valid_every = 100
    val_sample_size = 30
    lr = 5e-5

    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained'] = False
    config['device'] = 'cpu'

    config['dataset']['data_root'] = 'vietocr_data'
    config['dataset']['train_annotation'] = 'train_annotation.txt'
    config['dataset']['valid_annotation'] = 'val_annotation.txt'
    config['dataset']['name'] = 'lmdb'

    config['trainer']['batch_size'] = batch_size
    config['trainer']['print_every'] = print_every
    config['trainer']['valid_every'] = valid_every
    config['trainer']['iters'] = total_iters
    config['trainer']['export'] = finetuned_path
    config['trainer']['checkpoint'] = os.path.join(weights_dir, "checkpoint_vietocr.pth")
    config['trainer']['log'] = None
    config['trainer']['metrics'] = val_sample_size
    config['dataloader']['num_workers'] = 0
    config['dataloader']['pin_memory'] = False
    config['aug']['image_aug'] = False

    print(f"Thiet bi: CPU (batch_size={batch_size})")
    print(f"Tong so buoc (Iterations): {total_iters}")
    print(f"Learning rate: {lr} (CosineAnnealing to {lr/10})")
    print(f"Luu trong so tot nhat vao: {finetuned_path}")

    # 3. Khởi tạo Trainer và nạp trọng số gốc
    print("\nDang khoi tao Trainer va nap trong so goc...")
    trainer = Trainer(config, pretrained=False)
    trainer.load_weights(pretrained_path)

    # Tùy biến optimizer và scheduler để fine-tuning ổn định trên CPU
    trainer.optimizer = AdamW(trainer.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4)
    trainer.scheduler = CosineAnnealingLR(trainer.optimizer, T_max=total_iters, eta_min=lr/10)

    # 4. Đánh giá ban đầu (Baseline)
    print("\nDang danh gia Baseline truoc khi fine-tuning...")
    base_val_loss = trainer.validate()
    base_seq_acc, base_char_acc = trainer.precision(sample=val_sample_size)
    init_msg = f"[BASELINE] val_loss: {base_val_loss:.4f} | seq_acc: {base_seq_acc*100:.2f}% | char_acc: {base_char_acc*100:.2f}%"
    print(init_msg)
    logger.log(init_msg)

    best_char_acc = base_char_acc
    best_loss = base_val_loss
    saved_best = False

    # 5. Huấn luyện (Training Loop)
    print("\n" + "=" * 65)
    print("DANG FINE-TUNING... THEO DOI TIEN TRINH:")
    print("=" * 65)

    data_iter = iter(trainer.train_gen)
    running_loss = 0.0
    start_total_time = time.time()
    t_chunk = time.time()

    for step in range(1, total_iters + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(trainer.train_gen)
            batch = next(data_iter)

        loss = trainer.step(batch)
        running_loss += loss

        if step % print_every == 0:
            avg_loss = running_loss / print_every
            cur_lr = trainer.optimizer.param_groups[0]['lr']
            dt = time.time() - t_chunk
            step_msg = f"Step {step:04d}/{total_iters} | Loss: {avg_loss:.4f} | LR: {cur_lr:.2e} | Time: {dt:.1f}s ({dt/print_every:.2f}s/it)"
            print(step_msg)
            logger.log(step_msg)
            running_loss = 0.0
            t_chunk = time.time()

        if step % valid_every == 0:
            print(f"--> Dang danh gia tai buoc {step}...")
            val_loss = trainer.validate()
            acc_seq, acc_char = trainer.precision(sample=val_sample_size)
            val_msg = f"[VALIDATION Step {step}] val_loss: {val_loss:.4f} | seq_acc: {acc_seq*100:.2f}% | char_acc: {acc_char*100:.2f}%"
            print(val_msg)
            logger.log(val_msg)

            # Lưu nếu char_acc cải thiện hoặc val_loss giảm đáng kể
            if acc_char >= best_char_acc or val_loss < best_loss:
                best_char_acc = max(best_char_acc, acc_char)
                best_loss = min(best_loss, val_loss)
                trainer.save_weights(finetuned_path)
                saved_best = True
                save_msg = f"===> DA LUU MO HINH TOT NHAT (char_acc={acc_char*100:.2f}%, val_loss={val_loss:.4f}) vao {finetuned_path}"
                print(save_msg)
                logger.log(save_msg)

    # 6. Đảm bảo file finetuned_path tồn tại
    if not saved_best or not os.path.exists(finetuned_path):
        print("Luu trong so cuoi cung lam mo hinh tinh chinh...")
        trainer.save_weights(finetuned_path)

    total_time = time.time() - start_total_time
    done_msg = f"\nHUAN LUYEN HOAN TAT TRONG {total_time/60:.1f} PHUT! File da san sang tai: {finetuned_path}"
    print(done_msg)
    logger.log(done_msg)
    logger.close()
    print("=" * 65)


if __name__ == "__main__":
    main()
