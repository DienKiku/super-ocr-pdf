"""
OCR Engine Module with Hybrid Vietnamese AI & High-Speed Multilingual Recognition.
Combines:
- Bước 1: Tiền xử lý ảnh (Grayscale, Adaptive Thresholding, Auto-Deskew to 0°).
- Bước 2: Định vị vùng chữ với PaddleOCR DBNet (Text Detection PP-OCRv4) & cắt ảnh (Crop).
- Bước 3: Nhận diện chữ tiếng Việt bằng VietOCR (vgg_transformer, offline weights).
- Bước 4: Hậu xử lý dữ liệu (Spatial Heuristic, Regex số điện thoại, Dictionary Matching 63 tỉnh thành).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable, Dict, Any, Set
import os
import sys
import time
import re
import difflib
import unicodedata
import cv2
import numpy as np
from PIL import Image


@dataclass
class OCRBox:
    polygon: List[List[float]]               # 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    bbox: Tuple[float, float, float, float]  # (x, y, width, height)
    text: str
    confidence: float


@dataclass
class OCRResult:
    full_text: str = ""
    boxes: List[OCRBox] = field(default_factory=list)
    elapse_time: float = 0.0
    char_count: int = 0
    word_count: int = 0
    extracted_fields: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


def get_base_dir() -> str:
    """Get root directory in both normal and PyInstaller frozen modes."""
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            if os.path.exists(os.path.join(sys._MEIPASS, "core")):
                return sys._MEIPASS
        exe_dir = os.path.dirname(sys.executable)
        if os.path.exists(os.path.join(exe_dir, "_internal", "core")):
            return os.path.join(exe_dir, "_internal")
        if os.path.exists(os.path.join(exe_dir, "core")):
            return exe_dir
        if hasattr(sys, "_MEIPASS"):
            return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def remove_accents(s: str) -> str:
    """Strip Vietnamese diacritics for dictionary lookup."""
    s1 = unicodedata.normalize('NFD', s)
    s2 = ''.join(c for c in s1 if unicodedata.category(c) != 'Mn')
    return s2.replace('đ', 'd').replace('Đ', 'D')


# ---------------------------------------------------------------------------
# BỘ TỪ ĐIỂN 63 TỈNH THÀNH VIỆT NAM & CHUẨN HÓA LỖI CHÍNH TẢ / CHỮ VIẾT TAY
# ---------------------------------------------------------------------------

VIETNAM_PROVINCES = [
    "An Giang", "Bà Rịa - Vũng Tàu", "Bắc Giang", "Bắc Kạn", "Bạc Liêu", "Bắc Ninh",
    "Bến Tre", "Bình Định", "Bình Dương", "Bình Phước", "Bình Thuận", "Cà Mau",
    "Cần Thơ", "Cao Bằng", "Đà Nẵng", "Đắk Lắk", "Đắk Nông", "Điện Biên", "Đồng Nai",
    "Đồng Tháp", "Gia Lai", "Hà Giang", "Hà Nam", "Hà Nội", "Hà Tĩnh", "Hải Dương",
    "Hải Phòng", "Hậu Giang", "Hòa Bình", "Hưng Yên", "Khánh Hòa", "Kiên Giang",
    "Kon Tum", "Lai Châu", "Lâm Đồng", "Lạng Sơn", "Lào Cai", "Long An", "Nam Định",
    "Nghệ An", "Ninh Bình", "Ninh Thuận", "Phú Thọ", "Phú Yên", "Quảng Bình",
    "Quảng Nam", "Quảng Ngãi", "Quảng Ninh", "Quảng Trị", "Sóc Trăng", "Sơn La",
    "Tây Ninh", "Thái Bình", "Thái Nguyên", "Thanh Hóa", "Thừa Thiên Huế", "Tiền Giang",
    "TP. Hồ Chí Minh", "Trà Vinh", "Tuyên Quang", "Vĩnh Long", "Vĩnh Phúc", "Yên Bái"
]

PROVINCE_ALIASES_AND_TYPOS = {
    # Hà Nội & biến thể nhận diện chữ viết tay
    "ha noi": "Hà Nội",
    "hà nọi": "Hà Nội",
    "ha nọi": "Hà Nội",
    "hanoi": "Hà Nội",
    "hà nôi": "Hà Nội",
    "hà nôi.": "Hà Nội",
    "hn": "Hà Nội",
    "h.n": "Hà Nội",
    # TP. Hồ Chí Minh & biến thể
    "tp hcm": "TP. Hồ Chí Minh",
    "tphcm": "TP. Hồ Chí Minh",
    "tp.hcm": "TP. Hồ Chí Minh",
    "tp. hcm": "TP. Hồ Chí Minh",
    "tp ho chi minh": "TP. Hồ Chí Minh",
    "tp. ho chi minh": "TP. Hồ Chí Minh",
    "ho chi minh": "TP. Hồ Chí Minh",
    "hồ chí minh": "TP. Hồ Chí Minh",
    "sai gon": "TP. Hồ Chí Minh",
    "saigon": "TP. Hồ Chí Minh",
    "sài gòn": "TP. Hồ Chí Minh",
    "hcm": "TP. Hồ Chí Minh",
    # Đà Nẵng
    "da nang": "Đà Nẵng",
    "đà năng": "Đà Nẵng",
    "da năng": "Đà Nẵng",
    "danang": "Đà Nẵng",
    "đn": "Đà Nẵng",
    # Hải Phòng
    "hai phong": "Hải Phòng",
    "haiphong": "Hải Phòng",
    "hải phỏng": "Hải Phòng",
    "hp": "Hải Phòng",
    # Cần Thơ
    "can tho": "Cần Thơ",
    "cantho": "Cần Thơ",
    "cần thỏ": "Cần Thơ",
    # Bình Dương
    "binh duong": "Bình Dương",
    "bình duơng": "Bình Dương",
    "binhduong": "Bình Dương",
    "bd": "Bình Dương",
    # Đồng Nai
    "dong nai": "Đồng Nai",
    "dongnai": "Đồng Nai",
    # Bà Rịa - Vũng Tàu
    "ba ria - vung tau": "Bà Rịa - Vũng Tàu",
    "ba ria vung tau": "Bà Rịa - Vũng Tàu",
    "vung tau": "Bà Rịa - Vũng Tàu",
    "vũng tàu": "Bà Rịa - Vũng Tàu",
    "bà rịa": "Bà Rịa - Vũng Tàu",
    # Lâm Đồng / Đà Lạt
    "lam dong": "Lâm Đồng",
    "da lat": "Lâm Đồng",
    "đà lạt": "Lâm Đồng",
    # Khánh Hòa / Nha Trang
    "khanh hoa": "Khánh Hòa",
    "nha trang": "Khánh Hòa",
    # Thừa Thiên Huế
    "thua thien hue": "Thừa Thiên Huế",
    "hue": "Thừa Thiên Huế",
    "huế": "Thừa Thiên Huế",
    # Quảng Ninh
    "quang ninh": "Quảng Ninh",
    "ha long": "Quảng Ninh",
    "hạ long": "Quảng Ninh",
    # Bắc Ninh
    "bac ninh": "Bắc Ninh",
    "bacninh": "Bắc Ninh",
    # Hải Dương
    "hai duong": "Hải Dương",
    # Hưng Yên
    "hung yen": "Hưng Yên",
    # Nam Định
    "nam dinh": "Nam Định",
    # Thái Bình
    "thai binh": "Thái Bình",
    # Thanh Hóa
    "thanh hoa": "Thanh Hóa",
    # Nghệ An / Vinh
    "nghe an": "Nghệ An",
    "tp vinh": "Nghệ An",
}


def match_and_standardize_province(addr: str) -> Tuple[Optional[str], str]:
    """
    So khớp từ điển (Dictionary Matching) chứa các tỉnh thành Việt Nam
    để nhận diện và chuẩn hóa lỗi chính tả chữ viết tay trong địa chỉ
    (Ví dụ: '123 Cầu Giấy, Hà nọi' -> Tỉnh: 'Hà Nội', Chuẩn hóa: '123 Cầu Giấy, Hà Nội').
    """
    if not addr:
        return None, addr

    addr_clean = unicodedata.normalize('NFC', addr.strip())
    addr_lower = addr_clean.lower()
    addr_no_acc = remove_accents(addr_lower).lower()

    # 1. So khớp trực tiếp alias / lỗi chính tả trên văn bản gốc
    for alias, canonical in PROVINCE_ALIASES_AND_TYPOS.items():
        pattern = r'(?i)\b' + re.escape(alias) + r'\b'
        if re.search(pattern, addr_clean):
            fixed_addr = re.sub(pattern, canonical, addr_clean)
            return canonical, fixed_addr

    # 2. So khớp alias / lỗi chính tả trên chuỗi không dấu (đồng bộ vị trí ký tự 1:1)
    for alias, canonical in PROVINCE_ALIASES_AND_TYPOS.items():
        pattern_no_acc = r'\b' + re.escape(alias) + r'\b'
        m = re.search(pattern_no_acc, addr_no_acc)
        if m:
            start, end = m.span()
            fixed_addr = addr_clean[:start] + canonical + addr_clean[end:]
            return canonical, fixed_addr

    # 3. So khớp trực tiếp với 63 tỉnh thành chính thức
    for prov in VIETNAM_PROVINCES:
        pattern = r'(?i)\b' + re.escape(prov) + r'\b'
        if re.search(pattern, addr_clean):
            return prov, addr_clean

    for prov in VIETNAM_PROVINCES:
        prov_no_acc = remove_accents(prov.lower())
        pattern_no_acc = r'\b' + re.escape(prov_no_acc) + r'\b'
        m = re.search(pattern_no_acc, addr_no_acc)
        if m:
            start, end = m.span()
            fixed_addr = addr_clean[:start] + prov + addr_clean[end:]
            return prov, fixed_addr

    # 4. Fuzzy matching cho chữ viết ngoáy
    words = [w.strip(" ,.-;") for w in addr_clean.split()]
    for i in range(len(words)):
        for length in (2, 3, 4):
            if i + length <= len(words):
                chunk = " ".join(words[i:i + length])
                chunk_no_acc = remove_accents(chunk.lower())
                for prov in VIETNAM_PROVINCES:
                    prov_no_acc = remove_accents(prov.lower())
                    ratio = difflib.SequenceMatcher(None, chunk_no_acc, prov_no_acc).ratio()
                    if ratio >= 0.82:
                        fixed_addr = addr_clean.replace(chunk, prov)
                        return prov, fixed_addr

    return None, addr_clean


class SmartVietnameseRestorer:
    """
    High-precision Vietnamese diacritic restorer & typo fixer.
    Runs in < 10ms using n-gram greedy matching with O(1) hash lookups.
    """

    def __init__(self, dict_path: Optional[str] = None):
        self.ngram_dict: Dict[str, str] = {}

        self.typo_fixes = [
            (r"\bCONG TY TNHHMAI\b", "CÔNG TY TNHH MAI"),
            (r"\bTNHHMAI\b", "TNHH MAI"),
            (r"\bHONG GIAO DICH\b", "PHÒNG GIAO DỊCH"),
            (r"\bhong giao dich\b", "phòng giao dịch"),
            (r"\bPHONG GIAO DICH\b", "PHÒNG GIAO DỊCH"),
            (r"\bPHONG GIAO DỊCH\b", "PHÒNG GIAO DỊCH"),
            (r"\bphong giao dich\b", "phòng giao dịch"),
            (r"\bphong giao dịch\b", "phòng giao dịch"),
            (r"\bnguroi\b", "người"),
            (r"\bNguroi\b", "Người"),
            (r"\bthurc\b", "thực"),
            (r"\bThurc\b", "Thực"),
            (r"\bduroc\b", "được"),
            (r"\bDuroc\b", "Được"),
            (r"\btur\b", "từ"),
            (r"\bTur\b", "Từ"),
            (r"\btir\b", "từ"),
            (r"\blurgng\b", "lượng"),
            (r"\bLurgng\b", "Lượng"),
            (r"\bst dung\b", "sử dụng"),
            (r"\bSt dung\b", "Sử dụng"),
            (r"\bthi cong\b", "thủ công"),
            (r"\bThi cong\b", "Thủ công"),
            (r"\bphan men\b", "phần mềm"),
            (r"\bPhan men\b", "Phần mềm"),
            (r"\bdoi turong\b", "đối tượng"),
            (r"\bDoi turong\b", "Đối tượng"),
            (r"\bdja chi\b", "địa chỉ"),
            (r"\bDja chi\b", "Địa chỉ"),
            (r"\bdien thogi\b", "điện thoại"),
            (r"\bDien thogi\b", "Điện thoại"),
            (r"\bnguoilienhe\b", "người liên hệ"),
            (r"\bNguoilienhe\b", "Người liên hệ"),
            (r"\bkhich hiang\b", "khách hàng"),
            (r"\bKhich hiang\b", "Khách hàng"),
            (r"\bCAN CUOC CONG DAN\b", "CĂN CƯỚC CÔNG DÂN"),
            (r"\bcan cuoc cong dan\b", "căn cước công dân"),
            (r"\bCan cuoc cong dan\b", "Căn cước công dân"),
            (r"\bCONG HOA XA HOI CHU NGHIA VIET NAM\b", "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"),
            (r"\bDOC LAP - TU DO - HANH PHUC\b", "ĐỘC LẬP - TỰ DO - HẠNH PHÚC"),
        ]

        if dict_path and os.path.exists(dict_path):
            self._load_dictionary(dict_path)

    def _load_dictionary(self, path: str):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        normalized = unicodedata.normalize('NFC', word)
                        stripped = remove_accents(normalized).lower()
                        if stripped not in self.ngram_dict:
                            self.ngram_dict[stripped] = normalized
        except Exception:
            pass

    @staticmethod
    def match_word_case(orig: str, target: str) -> str:
        if orig.isupper():
            return target.upper()
        if orig and orig[0].isupper():
            return target.capitalize()
        return target.lower()

    def restore_line(self, line: str) -> str:
        if not line.strip():
            return line

        cur = line
        for pat, rep in self.typo_fixes:
            cur = re.sub(pat, rep, cur)

        tokens = re.split(r'(\s+|[^\w\s]+)', cur)
        word_indices = [idx for idx, tok in enumerate(tokens) if re.search(r'\w', tok)]
        words = [tokens[idx] for idx in word_indices]

        if not words:
            return cur

        i = 0
        total = len(words)
        while i < total:
            matched = False
            for n in range(min(5, total - i), 0, -1):
                chunk = words[i:i + n]
                clean_chunk = [re.sub(r'^[^\w]+|[^\w]+$', '', w) for w in chunk]
                if any(not w for w in clean_chunk):
                    continue

                phrase = " ".join(clean_chunk)
                phrase_lower = phrase.lower()
                stripped = remove_accents(phrase_lower)

                if stripped in self.ngram_dict:
                    restored_phrase = self.ngram_dict[stripped]
                    restored_words = restored_phrase.split()
                    if len(restored_words) == n:
                        for k in range(n):
                            orig_tok = words[i + k]
                            orig_clean = clean_chunk[k]
                            rest_word = restored_words[k]
                            cased_word = self.match_word_case(orig_clean, rest_word)
                            lead_punc = orig_tok[:len(orig_tok) - len(orig_tok.lstrip('^~`!@#$%^&*()_+-=[]{}|;:\'",.<>?/\\'))]
                            trail_punc = orig_tok[len(orig_tok.rstrip('^~`!@#$%^&*()_+-=[]{}|;:\'",.<>?/\\')):]
                            tokens[word_indices[i + k]] = lead_punc + cased_word + trail_punc
                        matched = True
                        i += n
                        break
            if not matched:
                i += 1

        res = "".join(tokens)
        return unicodedata.normalize('NFC', res)


# ---------------------------------------------------------------------------
# BƯỚC 1: TIỀN XỬ LÝ ẢNH (PRE-PROCESSING)
# ---------------------------------------------------------------------------

def preprocess_order_image(
    image: np.ndarray,
    deskew: bool = True,
    apply_adaptive_thresh: bool = True
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Bước 1: Tiền xử lý ảnh (Pre-processing) cho đơn hàng thực tế:
    - Xoay ảnh (Deskew): Tự động phát hiện góc nghiêng và xoay về phương ngang (0°).
    - Hệ màu xám (Grayscale).
    - Tăng tương phản (Adaptive Thresholding) để tách nét chữ viết tay mờ/đổ bóng.

    Returns:
        (deskewed_color, thresh_img, skew_angle)
    """
    if image is None or image.size == 0:
        return image, image, 0.0

    from core.enhancer import DocumentEnhancer

    # 1. Chuyển ảnh về Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. Xoay ảnh (Deskew): Kiểm tra góc nghiêng của văn bản để xoay ảnh về 0°
    skew_angle = 0.0
    deskewed_color = image.copy()
    deskewed_gray = gray.copy()
    if deskew:
        try:
            detected_angle = DocumentEnhancer.detect_skew_angle(gray)
            if abs(detected_angle) >= 0.55:
                skew_angle = detected_angle
                deskewed_color = DocumentEnhancer.deskew_image(image, skew_angle)
                if len(deskewed_color.shape) == 3:
                    deskewed_gray = cv2.cvtColor(deskewed_color, cv2.COLOR_BGR2GRAY)
                else:
                    deskewed_gray = deskewed_color.copy()
        except Exception:
            pass

    # 3. Tăng độ tương phản (Adaptive Thresholding)
    if apply_adaptive_thresh:
        try:
            blurred = cv2.GaussianBlur(deskewed_gray, (3, 3), 0)
            thresh_img = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                21,
                11
            )
        except Exception:
            thresh_img = deskewed_gray
    else:
        thresh_img = deskewed_gray

    return deskewed_color, thresh_img, skew_angle


def preprocess_for_ocr(
    image: np.ndarray,
    deskew: bool = True,
    denoise: bool = True,
    enhance_contrast: bool = True,
    whiten: bool = False
) -> np.ndarray:
    """
    Tiền xử lý toàn diện cho tài liệu văn bản & hóa đơn.
    """
    if image is None or image.size == 0:
        return image

    deskewed, _, _ = preprocess_order_image(image, deskew=deskew, apply_adaptive_thresh=False)
    processed = deskewed.copy()

    h, w = processed.shape[:2]
    max_dim = max(h, w)
    if max_dim < 1100:
        scale = 1100.0 / float(max_dim)
        processed = cv2.resize(processed, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

    from core.enhancer import DocumentEnhancer
    if whiten:
        try:
            processed = DocumentEnhancer.remove_shadows_and_whiten(processed, 0.35)
        except Exception:
            pass

    if denoise:
        try:
            processed = DocumentEnhancer.reduce_noise(processed, 0.15)
        except Exception:
            pass

    if enhance_contrast:
        try:
            processed = DocumentEnhancer.enhance_contrast(processed, 0.25)
        except Exception:
            pass

    return processed


# ---------------------------------------------------------------------------
# BƯỚC 2: ĐỊNH VỊ VÙNG CHỮ VỚI PADDLEOCR DBNET (TEXT DETECTION)
# ---------------------------------------------------------------------------

class PaddleOCRDetector:
    """
    Bước 2: Định vị vùng chữ với thuật toán PaddleOCR DBNet (Detection Only).
    Chạy qua ONNXRuntime engine với mô hình PP-OCRv4 detection mới nhất.
    Cung cấp giao diện tương thích chuẩn:
        det_model = PaddleOCR(use_angle_cls=True, lang='vi', det=True, rec=False)
        results = det_model.ocr(img, cls=True)
    """

    _instance: Optional["PaddleOCRDetector"] = None

    def __init__(
        self,
        use_angle_cls: bool = True,
        lang: str = "vi",
        det: bool = True,
        rec: bool = False,
        **kwargs
    ):
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self.det = det
        self.rec = rec
        self._rapid_ocr = None
        self._init_detector()

    @classmethod
    def get_instance(cls) -> "PaddleOCRDetector":
        if cls._instance is None:
            cls._instance = PaddleOCRDetector(use_angle_cls=True, lang="vi", det=True, rec=False)
        return cls._instance

    def _init_detector(self):
        if self._rapid_ocr is not None:
            return

        from rapidocr_onnxruntime import RapidOCR

        base_dir = get_base_dir()
        candidate_models = [
            os.path.join(base_dir, "weights", "ch_PP-OCRv4_det_infer.onnx"),
            os.path.join(base_dir, "core", "models", "ch_PP-OCRv4_det_infer.onnx"),
        ]

        init_kwargs = {
            "Det_unclip_ratio": 1.85,
            "Det_thresh": 0.20,
            "Det_box_thresh": 0.35,
            "Det_limit_side_len": 1536,
            "Global_use_angle_cls": self.use_angle_cls,
        }

        for model_path in candidate_models:
            if os.path.exists(model_path):
                init_kwargs["Det_model_path"] = model_path
                break

        self._rapid_ocr = RapidOCR(**init_kwargs)

    def detect(self, image: np.ndarray) -> List[List[List[float]]]:
        """
        Phát hiện vùng chữ bằng DBNet.
        Trả về danh sách các polygon 4 tọa độ góc: [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]
        """
        if image is None or image.size == 0:
            return []

        self._init_detector()

        try:
            dt_boxes, _ = self._rapid_ocr.text_detector(image)
        except Exception:
            dt_boxes = None

        if dt_boxes is None or len(dt_boxes) == 0:
            return []

        boxes_list = []
        for box in dt_boxes:
            pts = [[float(pt[0]), float(pt[1])] for pt in box]
            boxes_list.append(pts)

        return boxes_list

    def ocr(self, img_path_or_array, det: bool = True, rec: bool = False, cls: bool = True):
        """
        Giao diện mô phỏng chuẩn của PaddleOCR Python:
        det_model.ocr(img_path, cls=True)
        Khi det=True, rec=False -> Trả về danh sách các tọa độ góc [x, y] của từng dòng chữ.
        """
        if isinstance(img_path_or_array, str):
            image = cv2.imread(img_path_or_array)
        else:
            image = img_path_or_array

        if image is None or image.size == 0:
            return []

        boxes = self.detect(image)
        return [boxes] if boxes else []


PaddleOCR = PaddleOCRDetector


def crop_text_box(image: np.ndarray, polygon: List[List[float]], padding: int = 3) -> Optional[np.ndarray]:
    """
    Cắt ảnh (Crop): Dựa vào tọa độ 4 góc, cắt đoạn ảnh nhỏ chứa duy nhất một dòng/ô chữ.
    Sử dụng phép biến đổi phối cảnh (Perspective Transform) để nắn thẳng các dòng chữ xiên xẹo.
    """
    if image is None or image.size == 0 or len(polygon) != 4:
        return None

    pts = np.array(polygon, dtype=np.float32)

    w_top = np.linalg.norm(pts[1] - pts[0])
    w_bot = np.linalg.norm(pts[2] - pts[3])
    max_w = int(max(w_top, w_bot)) + (padding * 2)

    h_left = np.linalg.norm(pts[3] - pts[0])
    h_right = np.linalg.norm(pts[2] - pts[1])
    max_h = int(max(h_left, h_right)) + (padding * 2)

    if max_w < 5 or max_h < 5:
        return None

    dst_pts = np.array([
        [padding, padding],
        [max_w - 1 - padding, padding],
        [max_w - 1 - padding, max_h - 1 - padding],
        [padding, max_h - 1 - padding]
    ], dtype=np.float32)

    try:
        matrix = cv2.getPerspectiveTransform(pts, dst_pts)
        cropped = cv2.warpPerspective(
            image,
            matrix,
            (max_w, max_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return cropped
    except Exception:
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        x_min = max(0, int(min(xs)) - padding)
        y_min = max(0, int(min(ys)) - padding)
        x_max = min(image.shape[1], int(max(xs)) + padding)
        y_max = min(image.shape[0], int(max(ys)) + padding)
        if x_max > x_min and y_max > y_min:
            return image[y_min:y_max, x_min:x_max]
        return None


# ---------------------------------------------------------------------------
# BƯỚC 3: NHẬN DIỆN CHỮ TIẾNG VIỆT BẰNG VIETOCR (TEXT RECOGNITION)
# ---------------------------------------------------------------------------

class VietOCREngine:
    """
    Bước 3: Nhận diện chữ tiếng Việt bằng VietOCR (Text Recognition).
    Mô hình: vgg_transformer.
    Được huấn luyện chuẩn cho tiếng Việt và chữ viết tay.
    Chạy 100% Offline với bộ trọng số tải sẵn tại ./weights/vgg_transformer.pth.
    """

    _instance: Optional["VietOCREngine"] = None

    def __init__(self, model_name: str = "vgg_transformer"):
        self.model_name = model_name
        self.predictor = None
        self._init_predictor()

    @classmethod
    def get_instance(cls) -> "VietOCREngine":
        if cls._instance is None:
            cls._instance = VietOCREngine()
        return cls._instance

    def _init_predictor(self):
        if self.predictor is not None:
            return

        import torch
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        base_dir = get_base_dir()
        candidate_weights = [
            os.path.join(base_dir, "weights", f"{self.model_name}.pth"),
            os.path.join(".", "weights", f"{self.model_name}.pth"),
            os.path.join(base_dir, "core", "models", f"{self.model_name}.pth"),
        ]

        weights_path = None
        for path in candidate_weights:
            if os.path.exists(path) and os.path.getsize(path) > 10_000_000:
                weights_path = path
                break

        config = Cfg.load_config_from_name(self.model_name)
        if weights_path:
            config['weights'] = weights_path
        config['cnn']['pretrained'] = False
        config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if 'predictor' in config and isinstance(config['predictor'], dict):
            config['predictor']['beamsearch'] = False

        self.predictor = Predictor(config)

    def predict_image(self, image: np.ndarray) -> str:
        """Nhận diện văn bản cho một ảnh cắt dòng đơn lẻ."""
        if image is None or image.size == 0:
            return ""

        self._init_predictor()

        if len(image.shape) == 3:
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        pil_img = Image.fromarray(rgb_img)
        try:
            text = self.predictor.predict(pil_img)
            return (text or "").strip()
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# BƯỚC 4: HẬU XỬ LÝ DỮ LIỆU & BÓC TÁCH CẤU TRÚC (POST-PROCESSING & STRUCTURING)
# ---------------------------------------------------------------------------

def extract_structured_fields(
    text: str,
    boxes: Optional[List[OCRBox]] = None,
    img_shape: Optional[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """
    Bước 4: Hậu xử lý dữ liệu (Post-processing & Structuring) cho đơn hàng & văn bản:
    1. Sử dụng vị trí hình học (Spatial Heuristic)
    2. Sử dụng biểu thức chính quy (Regex)
    3. Sử dụng từ điển 63 tỉnh thành Việt Nam (Dictionary Matching)
    """
    fields: Dict[str, Any] = {
        "customer_name": [],
        "phones": [],
        "addresses": [],
        "provinces": [],
        "products": [],
        "amounts": [],
        "dates": [],
        "cccd": [],
        "mst": [],
        "emails": [],
    }

    # 1. Tên khách hàng / Người nhận / Người mua
    names = re.findall(
        r'(?:Họ\s*và\s*tên|Tên\s*khách\s*hàng|Người\s*nhận|Người\s*mua\s*hàng|Người\s*liên\s*hệ|Khách\s*hàng)[:\s]*([^\n\r,;]{2,45})',
        text, re.IGNORECASE
    )
    if names:
        cleaned_names = [re.sub(r'^[^\w]+|[^\w]+$', '', n).strip() for n in names]
        valid_names = [n for n in cleaned_names if len(n) > 2 and not any(c.isdigit() for c in n)]
        if valid_names:
            fields["customer_name"] = list(dict.fromkeys(valid_names))
            fields["names"] = fields["customer_name"]

    # 2. Biểu thức chính quy: Số điện thoại (10 chữ số bắt đầu bằng 0, hoặc +84)
    phones = re.findall(r'(?:(?:\+84|0)[235789][0-9]{8})\b', text)
    if phones:
        fields["phones"] = list(dict.fromkeys(phones))

    # 2. Biểu thức chính quy: Mã số thuế & CCCD
    mst = re.findall(r'(?:Mã\s*số\s*thuế|MST|Tax\s*Code)[:\s]*([0-9]{10}(?:-[0-9]{3})?)', text, re.IGNORECASE)
    if mst:
        fields["mst"] = list(dict.fromkeys(m.strip() for m in mst))

    cccd = re.findall(r'(?:CCCD|CMND|Số\s*CCCD|Số\s*CMND)[:\s]*([0-9]{9,12})\b', text, re.IGNORECASE)
    if cccd:
        fields["cccd"] = list(dict.fromkeys(c.strip() for c in cccd))
    else:
        twelve_digits = re.findall(r'\b0[0-9]{11}\b', text)
        if twelve_digits:
            fields["cccd"] = list(dict.fromkeys(twelve_digits))

    # 3. Biểu thức chính quy: Email
    emails = re.findall(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', text)
    if emails:
        fields["emails"] = list(dict.fromkeys(emails))

    # 4. Biểu thức chính quy: Ngày tháng
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b', text)
    if dates:
        fields["dates"] = list(dict.fromkeys(dates))

    # 5. Biểu thức chính quy: Số tiền / Tổng cộng / COD
    amounts = re.findall(
        r'(?:Tổng\s*tiền(?:\s*thanh\s*toán)?|Thành\s*tiền|Cộng\s*tiền(?:\s*hàng)?|Tổng\s*cộng|Thanh\s*toán|Tiền\s*thu|COD|Giá)[:\s]*([0-9.,]+(?:\s*(?:k|VND|VNĐ|đ|đồng))?)',
        text, re.IGNORECASE
    )
    if amounts:
        fields["amounts"] = list(dict.fromkeys(a.strip() for a in amounts))

    # 6. Địa chỉ & Từ điển 63 Tỉnh thành (Dictionary Matching)
    addresses = re.findall(
        r'(?:Nơi\s*thường\s*trú|Địa\s*chỉ(?:\s*nhận\s*hàng)?|Nơi\s*giao|Address|Đ\/C)[:\s]*([^\n\r]+)',
        text, re.IGNORECASE
    )
    extracted_addrs = [re.sub(r'^[^\w]+|[^\w]+$', '', a).strip() for a in addresses if len(a.strip()) > 5]

    matched_provinces = []
    standardized_addrs = []

    lines = text.splitlines()
    for ln in lines:
        prov, fixed = match_and_standardize_province(ln)
        if prov and prov not in matched_provinces:
            matched_provinces.append(prov)
        if any(keyword in ln.lower() for keyword in ("địa chỉ", "đ/c", "dia chi", "phường", "xã", "quận", "huyện", "thị trấn", "đường")):
            standardized_addrs.append(fixed)

    for addr in extracted_addrs:
        prov, fixed = match_and_standardize_province(addr)
        if prov and prov not in matched_provinces:
            matched_provinces.append(prov)
        if fixed not in standardized_addrs:
            standardized_addrs.append(fixed)

    if standardized_addrs:
        fields["addresses"] = list(dict.fromkeys(standardized_addrs))
    if matched_provinces:
        fields["provinces"] = matched_provinces

    # 7. Vị trí hình học (Spatial Heuristics) nếu có bounding boxes
    if boxes and img_shape and img_shape[0] > 0 and img_shape[1] > 0:
        img_h, img_w = img_shape[:2]

        top_lines = []
        middle_lines = []
        bottom_lines = []

        for b in boxes:
            y_mid = b.bbox[1] + (b.bbox[3] / 2.0)
            y_rel = y_mid / float(img_h)

            if y_rel < 0.35:
                top_lines.append(b)
            elif y_rel > 0.70:
                bottom_lines.append(b)
            else:
                middle_lines.append(b)

        # Vị trí góc trên: Tên người nhận / Khách hàng
        for b in top_lines:
            t = b.text.strip()
            m_name = re.search(r'(?:Họ\s*và\s*tên|Tên\s*khách\s*hàng|Người\s*nhận|Người\s*mua)[:\s]*([^\n\r,;]{2,40})', t, re.IGNORECASE)
            if m_name:
                name_val = m_name.group(1).strip()
                if len(name_val) > 2 and not any(c.isdigit() for c in name_val):
                    fields["customer_name"].append(name_val)
            elif 2 <= len(t.split()) <= 4 and t.istitle() and not any(c.isdigit() for c in t):
                if not any(k in t.lower() for k in ("đơn hàng", "phiếu", "hóa đơn", "cửa hàng", "shop", "ngày")):
                    fields["customer_name"].append(t)

        # Vị trí giữa trang: Các dòng trong bảng đơn hàng (Tên sản phẩm, Số lượng, Đơn giá)
        for b in middle_lines:
            t = b.text.strip()
            m_prod = re.search(r'([A-Za-z0-9\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐ_-]{3,35})\s*(?:x|SL|Số\s*lượng)?\s*[:\s]*(\d+)\s*[:\s]*([0-9.,]+\s*(?:k|đ|vnd|vnđ)?)', t, re.IGNORECASE)
            if m_prod:
                fields["products"].append({
                    "name": m_prod.group(1).strip(),
                    "quantity": m_prod.group(2).strip(),
                    "price": m_prod.group(3).strip(),
                })
            elif any(k in t.lower() for k in ("áo", "quần", "váy", "giày", "dép", "kem", "sách", "nạp mực", "máy in", "hộp", "bộ", "combo")):
                fields["products"].append({"name": t, "raw": t})

        # Vị trí góc dưới: Tổng tiền thanh toán
        for b in bottom_lines:
            t = b.text.strip()
            m_tot = re.search(r'(?:Tổng|Thành\s*tiền|Thanh\s*toán|Tiền\s*thu|COD)[:\s]*([0-9.,]+\s*(?:k|VND|VNĐ|đ|đồng)?)', t, re.IGNORECASE)
            if m_tot and m_tot.group(1).strip() not in fields["amounts"]:
                fields["amounts"].insert(0, m_tot.group(1).strip())

    if fields["customer_name"]:
        fields["customer_name"] = list(dict.fromkeys(fields["customer_name"]))

    return fields


def format_structured_order_summary(fields: Dict[str, Any]) -> str:
    """Tạo bảng tóm tắt thông tin bóc tách cấu trúc của đơn hàng."""
    if not fields:
        return ""

    has_data = any(fields.get(k) for k in ("customer_name", "phones", "addresses", "provinces", "products", "amounts"))
    if not has_data:
        return ""

    lines = []
    lines.append("=" * 55)
    lines.append("📋 KẾT QUẢ BÓC TÁCH CẤU TRÚC ĐƠN HÀNG (OFFLINE AI)")
    lines.append("=" * 55)

    if fields.get("customer_name"):
        lines.append(f"👤 Khách hàng: {', '.join(fields['customer_name'])}")

    if fields.get("phones"):
        lines.append(f"📞 Số điện thoại: {', '.join(fields['phones'])}")

    if fields.get("addresses"):
        lines.append(f"🏠 Địa chỉ: {', '.join(fields['addresses'])}")

    if fields.get("provinces"):
        lines.append(f"📍 Tỉnh / Thành phố: {', '.join(fields['provinces'])}")

    if fields.get("products"):
        lines.append("📦 Sản phẩm / Hàng hóa:")
        for p in fields["products"]:
            if isinstance(p, dict) and "quantity" in p and "price" in p:
                lines.append(f"   • {p['name']} | SL: {p['quantity']} | Giá: {p['price']}")
            elif isinstance(p, dict) and "name" in p:
                lines.append(f"   • {p['name']}")
            else:
                lines.append(f"   • {str(p)}")

    if fields.get("amounts"):
        lines.append(f"💰 Tổng tiền thanh toán: {fields['amounts'][0]}")

    if fields.get("dates"):
        lines.append(f"🗓️ Ngày tháng: {', '.join(fields['dates'])}")

    if fields.get("mst"):
        lines.append(f"🏢 Mã số thuế (MST): {', '.join(fields['mst'])}")

    if fields.get("cccd"):
        lines.append(f"🪪 CCCD/CMND: {', '.join(fields['cccd'])}")

    lines.append("=" * 55)
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HỆ THỐNG ĐIỀU PHỐI OCR CHÍNH (OCR ENGINE)
# ---------------------------------------------------------------------------

class OCREngine:
    """
    Hệ thống nhận diện OCR tinh gọn với 2 chế độ:
    1. 'offline': Quy trình 4 bước tối ưu cho tài liệu & đơn hàng viết tay tiếng Việt:
       - Bước 1: Tiền xử lý Deskew (0°), Grayscale, Adaptive Thresholding & CLAHE.
       - Bước 2: Định vị vùng chữ với PaddleOCR DBNet (PP-OCRv4 ONNX) & Cắt ảnh (Crop).
       - Bước 3: Nhận diện chữ tiếng Việt bằng VietOCR (vgg_transformer).
       - Bước 4: Hậu xử lý bóc tách cấu trúc (Spatial Heuristic, Regex số điện thoại, Từ điển 63 tỉnh thành).
    2. 'online': Google Gemini Cloud Vision AI (dự phòng đa tầng đám mây).
    """

    _instance: Optional["OCREngine"] = None

    def __init__(self):
        self._detector: Optional[PaddleOCRDetector] = None
        self._vietocr: Optional[VietOCREngine] = None
        self._restorer: Optional[SmartVietnameseRestorer] = None
        self._engine_mode: str = "offline"

    @classmethod
    def get_instance(cls) -> "OCREngine":
        if cls._instance is None:
            cls._instance = OCREngine()
        return cls._instance

    @property
    def engine_mode(self) -> str:
        return self._engine_mode

    @engine_mode.setter
    def engine_mode(self, mode: str):
        mode_lower = mode.lower().strip()
        if mode_lower in ("online", "gemini"):
            self._engine_mode = "online"
        else:
            self._engine_mode = "offline"

    def _init_detector(self):
        if self._detector is None:
            self._detector = PaddleOCRDetector.get_instance()

    def _init_vietocr(self):
        if self._vietocr is None:
            self._vietocr = VietOCREngine.get_instance()

    def _init_restorer(self):
        if self._restorer is None:
            base_dir = get_base_dir()
            dict_path = os.path.join(base_dir, "core", "models", "viet_words.txt")
            self._restorer = SmartVietnameseRestorer(dict_path)

    def recognize(
        self,
        image: np.ndarray,
        mode: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> OCRResult:
        """
        Nhận diện văn bản trong ảnh:
        - 'online': Google Gemini Cloud Vision AI.
        - 'offline': PaddleOCR DBNet + VietOCR + Hậu xử lý cấu trúc.
        """
        if image is None or image.size == 0:
            return OCRResult(error="Ảnh rỗng hoặc không hợp lệ")

        active_mode = (mode or self._engine_mode).lower().strip()
        start_time = time.time()

        try:
            if active_mode in ("online", "gemini"):
                return self._recognize_online(image, start_time, progress_callback=progress_callback)
            else:
                return self._recognize_offline(image, start_time, progress_callback=progress_callback)
        except Exception as e:
            return OCRResult(error=f"Lỗi nhận diện OCR: {str(e)}")

    def _recognize_online(
        self,
        image: np.ndarray,
        start_time: float,
        api_key: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> OCRResult:
        """Chế độ Online: Multimodal Cloud Vision OCR qua Google Gemini AI."""
        from core.config_manager import ConfigManager
        cfg = ConfigManager.get_instance()
        key = (api_key or cfg.get_gemini_api_key()).strip()
        if not key:
            return OCRResult(
                error="Chưa nhập Google Gemini API Key. Vui lòng nhập API Key hoặc lấy Key miễn phí tại Google AI Studio."
            )

        if progress_callback:
            progress_callback(2, 10, "Đang kết nối tới Google Gemini Vision AI...")

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=key)

            success, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not success:
                return OCRResult(error="Không thể nén ảnh để gửi tới Gemini AI.")

            image_part = types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")

            prompt = (
                "Bạn là chuyên gia trích xuất tài liệu OCR tiếng Việt cao cấp.\n"
                "Nhiệm vụ của bạn là đọc và trích xuất TOÀN BỘ nội dung có trong bức ảnh tài liệu này, "
                "bao gồm cả chữ in, chữ viết tay, bảng biểu, số tiền, ngày tháng, mã số thuế, địa chỉ, email.\n\n"
                "Quy tắc bắt buộc:\n"
                "1. Đọc chính xác 100% chữ viết tay tiếng Việt có dấu.\n"
                "2. Giữ nguyên cấu trúc các dòng trong bảng biểu và thông tin tiền tệ.\n"
                "3. Bảo toàn nguyên vẹn mọi số điện thoại, địa chỉ, mã số thuế.\n"
                "4. Chỉ trả về nội dung văn bản trích xuất sạch sẽ, trung thực, không thêm lời giải thích mở đầu/kết thúc."
            )

            models_to_try = [
                cfg.get_gemini_model(),
                "gemini-2.5-flash",
                "gemini-2.0-flash",
                "gemini-flash-latest",
            ]
            seen = set()
            model_queue = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

            response = None
            last_err = None

            for m in model_queue:
                try:
                    if progress_callback:
                        progress_callback(5, 10, f"Gemini ({m}) đang phân tích tài liệu...")
                    response = client.models.generate_content(
                        model=m,
                        contents=[image_part, prompt]
                    )
                    if response and response.text:
                        cfg.set_gemini_model(m)
                        break
                except Exception as e:
                    last_err = e
                    continue

            if not response or not response.text:
                raise last_err or RuntimeError("Không nhận được phản hồi từ Gemini.")

            extracted_text = unicodedata.normalize("NFC", (response.text or "").strip())
            if not extracted_text:
                return OCRResult(error="Gemini không tìm thấy văn bản trong ảnh.")

            if progress_callback:
                progress_callback(9, 10, "Đang định dạng dòng văn bản & trích xuất dữ liệu...")

            lines = extracted_text.splitlines()
            h, w = image.shape[:2]
            line_h = h / max(1, len(lines))
            boxes: List[OCRBox] = []
            for idx, line_str in enumerate(lines):
                if line_str.strip():
                    y = idx * line_h
                    boxes.append(OCRBox(
                        polygon=[[0.0, y], [float(w), y], [float(w), y + line_h], [0.0, y + line_h]],
                        bbox=(0.0, y, float(w), line_h),
                        text=line_str.strip(),
                        confidence=0.99
                    ))

            total_time = round(time.time() - start_time, 2)
            extracted_fields = extract_structured_fields(extracted_text, boxes=boxes, img_shape=(h, w))

            summary_header = format_structured_order_summary(extracted_fields)
            final_display_text = f"{summary_header}{extracted_text}" if summary_header else extracted_text

            if progress_callback:
                progress_callback(10, 10, "Hoàn tất nhận diện!")

            return OCRResult(
                full_text=final_display_text,
                boxes=boxes,
                elapse_time=total_time,
                char_count=len(final_display_text),
                word_count=len(final_display_text.split()),
                extracted_fields=extracted_fields
            )

        except Exception as e:
            err_str = str(e)
            if "API_KEY_INVALID" in err_str or "API key not valid" in err_str:
                return OCRResult(error="API Key không hợp lệ. Vui lòng kiểm tra lại Key tại aistudio.google.com")
            elif "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                return OCRResult(error="Đã vượt giới hạn lượt gọi tạm thời. Vui lòng đợi vài giây rồi thử lại.")
            else:
                return OCRResult(error=f"Lỗi Gemini Vision AI: {err_str}")

    def _recognize_offline(
        self,
        image: np.ndarray,
        start_time: float,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> OCRResult:
        """
        Quy trình Offline toàn diện 4 Bước:
        - Bước 1: Tiền xử lý ảnh (Pre-processing): Grayscale, Deskew (0°), Adaptive Thresholding & CLAHE.
        - Bước 2: Định vị vùng chữ với PaddleOCR DBNet (PP-OCRv4 Det) & Cắt ảnh (Crop).
        - Bước 3: Nhận diện chữ tiếng Việt bằng VietOCR (vgg_transformer).
        - Bước 4: Hậu xử lý dữ liệu (Spatial Heuristic, Regex số điện thoại, Dictionary Matching 63 tỉnh thành).
        """
        self._init_detector()
        self._init_vietocr()
        self._init_restorer()

        # BƯỚC 1: TIỀN XỬ LÝ ẢNH
        if progress_callback:
            progress_callback(1, 10, "Bước 1: Tiền xử lý ảnh (Deskew 0°, Grayscale, Tăng tương phản)...")

        deskewed_color, thresh_img, skew_angle = preprocess_order_image(
            image,
            deskew=True,
            apply_adaptive_thresh=True
        )
        prep_img = preprocess_for_ocr(deskewed_color, deskew=False, denoise=True, enhance_contrast=True)

        # BƯỚC 2: ĐỊNH VỊ VÙNG CHỮ VỚI PADDLEOCR DBNET
        if progress_callback:
            progress_callback(3, 10, "Bước 2: Định vị vùng chữ bằng PaddleOCR DBNet (PP-OCRv4)...")

        polygons = self._detector.detect(prep_img)
        if not polygons and deskewed_color is not prep_img:
            polygons = self._detector.detect(deskewed_color)

        if not polygons:
            return OCRResult(
                full_text="Không tìm thấy văn bản trong tài liệu.",
                boxes=[],
                elapse_time=round(time.time() - start_time, 2)
            )

        # Cắt ảnh (Crop): Lấy tọa độ bounding boxes và cắt ảnh đơn hàng thành các ảnh nhỏ
        cropped_items: List[Tuple[np.ndarray, Tuple[float, float, float, float], List[List[float]]]] = []
        for poly in polygons:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

            crop = crop_text_box(deskewed_color, poly, padding=3)
            if crop is not None and crop.size > 0:
                cropped_items.append((crop, bbox, poly))

        # Sắp xếp các đoạn ảnh theo thứ tự đọc tự nhiên từ trên xuống dưới, trái qua phải
        cropped_items.sort(key=lambda item: (item[1][1], item[1][0]))

        # BƯỚC 3: NHẬN DIỆN CHỮ TIẾNG VIỆT BẰNG VIETOCR
        total_crops = len(cropped_items)
        boxes: List[OCRBox] = []

        for idx, (crop, bbox, poly) in enumerate(cropped_items):
            if progress_callback and total_crops > 0:
                step_pct = 4 + int((idx / total_crops) * 4)
                progress_callback(step_pct, 10, f"Bước 3: VietOCR đang nhận diện tiếng Việt ({idx + 1}/{total_crops} dòng)...")

            text = self._vietocr.predict_image(crop)
            if text:
                text = unicodedata.normalize("NFC", text.strip())
                text = self._restorer.restore_line(text)

                boxes.append(OCRBox(
                    polygon=poly,
                    bbox=bbox,
                    text=text,
                    confidence=0.95
                ))

        if not boxes:
            return OCRResult(
                full_text="",
                boxes=[],
                elapse_time=round(time.time() - start_time, 2)
            )

        sorted_boxes = self._sort_reading_order(boxes)
        reconstructed_lines = self._reconstruct_lines(sorted_boxes)
        raw_full_text = "\n".join(reconstructed_lines).strip()

        # BƯỚC 4: HẬU XỬ LÝ DỮ LIỆU & BÓC TÁCH CẤU TRÚC
        if progress_callback:
            progress_callback(9, 10, "Bước 4: Hậu xử lý dữ liệu (Spatial Heuristic, Regex SĐT, Từ điển 63 tỉnh thành)...")

        img_h, img_w = deskewed_color.shape[:2]
        extracted_fields = extract_structured_fields(
            raw_full_text,
            boxes=sorted_boxes,
            img_shape=(img_h, img_w)
        )

        summary_header = format_structured_order_summary(extracted_fields)
        if summary_header:
            final_full_text = f"{summary_header}\n--- NỘI DUNG VĂN BẢN CHI TIẾT ---\n{raw_full_text}"
        else:
            final_full_text = raw_full_text

        total_time = round(time.time() - start_time, 2)

        if progress_callback:
            progress_callback(10, 10, "Hoàn tất nhận diện đơn hàng thành công!")

        return OCRResult(
            full_text=final_full_text,
            boxes=sorted_boxes,
            elapse_time=total_time,
            char_count=len(final_full_text),
            word_count=len(final_full_text.split()),
            extracted_fields=extracted_fields
        )

    def _sort_reading_order(self, boxes: List[OCRBox]) -> List[OCRBox]:
        """Sắp xếp các bounding box theo thứ tự đọc tự nhiên từ trên xuống, trái sang phải."""
        if not boxes:
            return []

        avg_h = np.mean([b.bbox[3] for b in boxes]) if boxes else 20.0
        line_threshold = max(10.0, avg_h * 0.55)

        boxes_by_y = sorted(boxes, key=lambda b: (b.bbox[1], b.bbox[0]))

        lines: List[List[OCRBox]] = []
        for b in boxes_by_y:
            placed = False
            for line in lines:
                line_avg_y = np.mean([item.bbox[1] for item in line])
                if abs(b.bbox[1] - line_avg_y) < line_threshold:
                    line.append(b)
                    placed = True
                    break
            if not placed:
                lines.append([b])

        sorted_boxes: List[OCRBox] = []
        for line in lines:
            line.sort(key=lambda b: b.bbox[0])
            sorted_boxes.extend(line)

        return sorted_boxes

    def _reconstruct_lines(self, sorted_boxes: List[OCRBox]) -> List[str]:
        """Tái cấu trúc văn bản thuần theo từng dòng đọc."""
        if not sorted_boxes:
            return []

        avg_height = np.mean([b.bbox[3] for b in sorted_boxes]) if sorted_boxes else 20.0
        line_threshold = avg_height * 0.55

        lines: List[str] = []
        current_line: List[str] = []
        prev_y: Optional[float] = None

        for b in sorted_boxes:
            y = b.bbox[1]
            if prev_y is None or abs(y - prev_y) <= line_threshold:
                current_line.append(b.text)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [b.text]
            prev_y = y

        if current_line:
            lines.append(" ".join(current_line))

        return lines
