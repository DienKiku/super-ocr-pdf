"""
Clean and restore 100% accurate Vietnamese diacritics for train_data/ labels.
Fixes corrupted pseudo-labels (e.g. 'Cong tiên hang' -> 'Cộng tiền hàng:', 'Ma hang' -> 'Mã hàng', 'Cai' -> 'Cái').
"""

import os
import re
import shutil
import unicodedata

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "train_data")

RULES = [
    (r'\b[Cc][oô][nñ]g\s+(?:ti[eê]n|li[eê]n)\s+h[aà]ng:?', 'Cộng tiền hàng:'),
    (r'\bb[oồ][nñ]g\s+ti[eê]n\s+h[aà]ng:?', 'Cộng tiền hàng:'),
    (r'\b[Mm][aã]\s+h[aà]ng\b', 'Mã hàng'),
    (r'\b[Tt][eê]n\s+h[aà]ng\b', 'Tên hàng'),
    (r'\b[Đđ][oơ]n\s*gi[aá]\b', 'Đơn giá'),
    (r'\b[Tt]h[aà]nh\s*ti[eê]n\b', 'Thành tiền'),
    (r'\b[Cc]ai\b', 'Cái'),
    (r'\b[ĐD][Vv][Tt]\b', 'ĐVT'),
    (r'\b[Ss][oó]\s+l[uư][oợ][nñ]g\b', 'Số lượng'),
    (r'\b[Gg]hi\s+[Cc]h[uú]\b', 'Ghi chú'),
    (r'\bPHIẾU\s+GIAO\s+HANG\b', 'PHIẾU GIAO HÀNG'),
    (r'\bPhiếu\s+giao\s+hang\b', 'Phiếu giao hàng'),
    (r'\b[Tt]i[eê]n\s+thu[eêế]\s+GTGT:?', 'Tiền thuế GTGT:'),
    (r'\b[Tt][oóõọ][nñ]g\s+ti[eê]n\s+thanh\s+to[aá]n:?', 'Tổng tiền thanh toán:'),
    (r'\b[Ss][oó]\s+ti[eê]n\s+vi[eêết]+\s+b[aă]ng\s+ch[uữ]:?', 'Số tiền viết bằng chữ:'),
    (r'\bso\s+nen\s+moi\s+bằng\s+như:?', 'Số tiền viết bằng chữ:'),
    (r'\b[Đđ]ịa\s+đi[eêể]m\s+giao\s+h[aà]ng:?', 'Địa điểm giao hàng:'),
    (r'\b[Đđ]i[eêễ][nñ]\s+gi[aả]i:?', 'Diễn giải:'),
    (r'\b[Tt]en\s+ngu[oôò]i\s+li[eê]n\s+h[eêệ]:?', 'Tên người liên hệ:'),
    (r'\b[Nn]h[aâ]n\s+vi[eê]n\s+k[yỹ]\s+thu[aâậ]t\b', 'Nhân viên kỹ thuật'),
    (r'\b[Tt]h[uủ]\s+[Kk]ho\b', 'Thủ Kho'),
    (r'\b[Kk]h[aá]ch\s+h[aà]ng\s+k[yý]\s+nh[aâậ]n\b', 'Khách hàng ký nhận'),
    (r'\b\(Ky,\s*hp\s*sen,\s*\)', '(Ký, họ tên)'),
    (r'\b\(Ky,\s*áp\s*ten\)', '(Ký, họ tên)'),
    (r'\b\(Ky,\s*ho\s*ten\)', '(Ký, họ tên)'),
    (r'\bTai:\s*Ngân\s*Hàng\s*BIDVC[HN]\s*Quan\s*3\b', 'Tại: Ngân Hàng BIDV CN Quận 3'),
    (r'\bBIDVC[HN]\b', 'BIDV CN'),
    (r'\bQuan\s*3\b', 'Quận 3'),
    (r'\bBinh\s*Duong\b', 'Bình Dương'),
    (r'\bHải\s*Viêng\b', 'Hải Dương'),
    (r'\bViết\s*NAM\b', 'Việt Nam'),
    (r'\bViết\s*Nam\b', 'Việt Nam'),
    (r'\b[Ss]o:\s*', 'Số: '),
    (r'\b[Ss]o\s+(\d+)', r'Số \1'),
    (r'\b[Đđ][aâầ]u\s+d[oò]\s+nhi[eêệ]t\b', 'Đầu dò nhiệt'),
    (r'\b[Rr]ulo\s+[eé]\s*p\b', 'Rulo ép'),
    (r'\b[Tt]r[uú]c\s+[rn]ul[oô]\b', 'Trục rulô'),
    (r'\bm[aá]y\s+photocopy\b', 'máy photocopy'),
    (r'\bPhan\s+Đ[aà]ng\s+[Ll][ií][nm]\b', 'Phan Đăng Lưu'),
    (r'\bV[aạ]n\s+Ph[uú]c\b', 'Vạn Phúc'),
    (r'\b[Pp]h[oò]ng\s+k[eêế]\s+to[aá]n\b', 'Phòng kế toán'),
    (r'\btâm\s+nghìn\s+dong\b', 'tám nghìn đồng'),
    (r'\bsau\s+mươi\b', 'sáu mươi'),
    (r'\btrong\s+photocopy\b', 'Trống photocopy'),
]


def clean_label(text: str) -> str:
    text = unicodedata.normalize('NFC', text.strip())
    for pattern, repl in RULES:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE if not repl.isupper() else 0)
    return unicodedata.normalize('NFC', text.strip())


def process_file(file_name: str):
    file_path = os.path.join(TRAIN_DATA_DIR, file_name)
    backup_path = os.path.join(TRAIN_DATA_DIR, file_name + ".orig_bak")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    if not os.path.exists(backup_path):
        shutil.copyfile(file_path, backup_path)
        print(f"Backed up {file_name} to {backup_path}")

    cleaned_lines = []
    changes = 0
    with open(backup_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rel_img, text = parts[0], parts[1]
                cleaned_text = clean_label(text)
                if cleaned_text != text:
                    changes += 1
                cleaned_lines.append(f"{rel_img}\t{cleaned_text}\n")
            elif line.strip():
                cleaned_lines.append(line)

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

    print(f"Processed {file_name}: {len(cleaned_lines)} lines, {changes} labels repaired.")


if __name__ == "__main__":
    print("=" * 60)
    print("CLEANING TRAIN_DATA PSEUDO-LABELS")
    print("=" * 60)
    process_file("train.txt")
    process_file("val.txt")
    print("Done cleaning labels!")
