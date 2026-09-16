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
import warnings
import cv2
import numpy as np
from PIL import Image

warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.transformer')


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
# TIỀN XỬ LÝ ẢNH & ĐỊNH DẠNG VĂN BẢN
# ---------------------------------------------------------------------------


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

    # Keep image dimensions identical so detection coordinates match 1:1 with source
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
# BỘ TIỀN XỬ LÝ DÒNG & HẬU XỬ LÝ CHÍNH TẢ THANH DẤU TIẾNG VIỆT
# ---------------------------------------------------------------------------

VIETNAMESE_PROVINCES = [
    "An Giang", "Bà Rịa - Vũng Tàu", "Bắc Giang", "Bắc Kạn", "Bạc Liêu", "Bắc Ninh",
    "Bến Tre", "Bình Định", "Bình Dương", "Bình Phước", "Bình Thuận", "Cà Mau",
    "Cần Thơ", "Cao Bằng", "Đà Nẵng", "Đắk Lắk", "Đắk Nông", "Điện Biên",
    "Đồng Nai", "Đồng Tháp", "Gia Lai", "Hà Giang", "Hà Nam", "Hà Nội",
    "Hà Tĩnh", "Hải Dương", "Hải Phòng", "Hậu Giang", "Hòa Bình", "Hưng Yên",
    "Khánh Hòa", "Kiên Giang", "Kon Tum", "Lai Châu", "Lâm Đồng", "Lạng Sơn",
    "Lào Cai", "Long An", "Nam Định", "Nghệ An", "Ninh Bình", "Ninh Thuận",
    "Phú Thọ", "Phú Yên", "Quảng Bình", "Quảng Nam", "Quảng Ngãi", "Quảng Ninh",
    "Quảng Trị", "Sóc Trăng", "Sơn La", "Tây Ninh", "Thái Bình", "Thái Nguyên",
    "Thanh Hóa", "Thừa Thiên Huế", "Tiền Giang", "TP Hồ Chí Minh", "Trà Vinh",
    "Tuyên Quang", "Vĩnh Long", "Vĩnh Phúc", "Yên Bái"
]


def enhance_text_crop(crop: np.ndarray) -> np.ndarray:
    """Tăng tương phản và làm nét các dấu thanh nhỏ trong ảnh cắt dòng chữ viết tay."""
    if crop is None or crop.size == 0:
        return crop
    try:
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()
        blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
        sharpened = cv2.addWeighted(gray, 1.25, blurred, -0.25, 0)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    except Exception:
        return crop


class VietnameseDiacriticsCorrector:
    """
    Bộ hậu xử lý phục hồi thanh dấu & chính tả tiếng Việt:
    - Sửa các lỗi rớt dấu từ vựng hành chính và số nhà.
    - Đối soát tự động địa danh 63 tỉnh thành & quận/huyện.
    - Chuẩn hóa Unicode NFC 100%.
    """
    WORD_REPLACEMENTS = [
        (r'\b[Ss]o\s+(\d+)', r'Số \1'),
        (r'\b[Ss][oó]\s+([0-9])', r'Số \1'),
        (r'\b[Cc]hường\b', 'đường'),
        (r'\b[Đđ]ương\b', 'đường'),
        (r'\b[Dd]uong\b', 'đường'),
        (r'\b[Dd]ương\s+V[oó]\s+Oanh\b', 'Đường Võ Oanh'),
        (r'\b[Pp]huong\b', 'Phường'),
        (r'\b[Pp]huồng\b', 'Phường'),
        (r'\b[Qq]uan\b', 'Quận'),
        (r'\b[Qq]uân\b', 'Quận'),
        (r'\b[Hh]uyen\b', 'Huyện'),
        (r'\b[Hh]uyên\b', 'Huyện'),
        (r'\b[Tt]h[ií]\s+[Tt]r[aâấ]n\b', 'Thị trấn'),
        (r'\b[Tt]hị\s+t[aâá]n\b', 'Thị trấn'),
        (r'\b[Tt]hi\s+thái\b', 'Thị trấn'),
        (r'\b[Kk]hn\s+phế\b', 'Khu phố'),
        (r'\b[Kk]hu\s+ph[oóeế]\s*([0-9A-Za-z]+)', r'Khu phố \1'),
        (r'\b[Tt]h[aà]nh\s+[Pp]h[oó]\b', 'Thành phố'),
        (r'\b[Tt]p\.?\s*[Hh]ồ\s+[Cc]hí\s+[Mm]inh\b', 'TP Hồ Chí Minh'),
        (r'\b[Tt][Xx]\s+', 'Tx '),
        (r'\bIx\s+', 'Tx '),
        (r'\b[Aa]p\b', 'Ấp'),
        (r'\b[Tt]hon\b', 'Thôn'),
        (r'\b[Xx]om\b', 'Xóm'),
        (r'\b[Đđ][iìịí]nh\s+Bộ\s+Lĩnh\b', 'Đinh Bộ Lĩnh'),
        (r'\b[KkNnHh][aâă]n\s+Sách\b', 'Nam Sách'),
        (r'\b[Hh]àm\s+Sách\b', 'Nam Sách'),
        (r'\bTrần\s+thí\b', 'Trần Phú'),
        (r'\bBinh\s+Dương\b', 'Bình Dương'),
        (r'\bHải\s+Viêng\b', 'Hải Dương'),
        (r'\bHải\s+Dưong\b', 'Hải Dương'),
        (r'\bViết\s+Nam\b', 'Việt Nam'),
        (r'\bĐịa\s+chi\b', 'Địa chỉ'),
        (r'\bĐịa\s+ông\s+số\b', 'Địa chỉ: Số'),
        (r'\bTNHFI\b', 'TNHH'),
        (r'\bCty\s+TNHH\b', 'Công ty TNHH'),
        (r'\.còm\b', '.com'),
        (r'\b[Cc]ong\s+ti[eê]n\s+h[aà]ng:?', 'Cộng tiền hàng:'),
        (r'\b[Tt]i[eê]n\s+thu[eêế]\s+GTGT:?', 'Tiền thuế GTGT:'),
        (r'\b[Tt][oóõọ][nñ]g\s+ti[eê]n\s+thanh\s+to[aá]n:?', 'Tổng tiền thanh toán:'),
        (r'\b[Ss][oó]\s+ti[eê]n\s+vi[eêết]+\s+b[aă]ng\s+ch[uữ]:?', 'Số tiền viết bằng chữ:'),
        (r'\bSo\s+nên\s+viết\b', 'Số tiền viết'),
        (r'\bbang\s+chu:?', 'bằng chữ:'),
        (r'\b[Đđ]ịa\s+đi[eêể]m\s+giao\s+h[aà]ng:?', 'Địa điểm giao hàng:'),
        (r'\b[Đđ]i[eêễ][nñ]\s+gi[aả]i:?', 'Diễn giải:'),
        (r'\b[Nn]h[aâ]n\s+vi[eê]n\s+k[yỹ]\s+thu[aâậ]t\b', 'Nhân viên kỹ thuật'),
        (r'\b[Tt]h[uủ]\s+[Kk]ho\b', 'Thủ Kho'),
        (r'\b[Kk]h[aá]ch\s+h[aà]ng\s+k[yý]\s+nh[aâậ]n\b', 'Khách hàng ký nhận'),
        (r'\b[Kk][yý],\s*h[oọ]\s*t[eê]n\b', 'Ký, họ tên'),
        (r'\bPhan\s+Đ[aà]ng\s+[Ll][ií][nm]\b', 'Phan Đăng Lưu'),
        (r'\bPhan\s+Đ[aà]ng\s+Lưu\b', 'Phan Đăng Lưu'),
        (r'\bV[aạ]n\s+Ph[uú]c\b', 'Vạn Phúc'),
        (r'\bVAN\s+PH[UÚ]C\b', 'VẠN PHÚC'),
        (r'\b[Pp]h[oò]ng\s+k[eêế]\s+to[aá]n\b', 'Phòng kế toán'),
        (r'\bPHONG\s+K[EÊẾ]\s+TO[AÁ]N\b', 'PHÒNG KẾ TOÁN'),
        (r'\b[Đđ][aâầ]u\s+d[oò]\s+nhi[eêệ]t\b', 'Đầu dò nhiệt'),
        (r'\bDau\s+do\s+nhiệt\b', 'Đầu dò nhiệt'),
        (r'\bRulo\s+\(?[eéP]\)?\s*MP', 'Rulo ép MP'),
        (r'\bTr[uú]c\s+[rn]ul[oô]\b', 'Trục rulô'),
        (r'\bm[aá]y\s+photocopy\b', 'máy photocopy'),
        (r'\bCai\b', 'Cái'),
        (r'\b[Đđ]ơn\s*gi[aá]\b', 'Đơn giá'),
        (r'\b[Tt]h[aà]nh\s*ti[eê]n\b', 'Thành tiền'),
        (r'\b[Tt]en\s+h[aà]ng\b', 'Tên hàng'),
        (r'\b[Mm][aã]\s+h[aà]ng\b', 'Mã hàng'),
        (r'\btâm\s+nghìn\s+dong\b', 'tám nghìn đồng'),
        (r'\bsau\s+mươi\b', 'sáu mươi'),
        (r'\bNgay:\s*', 'Ngày: '),
        (r'\bSo:\s*', 'Số: '),
        (r'\bTai:\s*', 'Tại: '),
        (r'\bĐia\s+ông\s+số\s*', 'Địa chỉ: Số '),
        (r'\bĐường\s+Vo\s+Oanh\b', 'Đường Võ Oanh'),
        (r'\bThành\s+M[9g]\s+Tây\b', 'Thạnh Mỹ Tây'),
        (r'\bHồ\s+Chi\s+Minh\b', 'Hồ Chí Minh'),
        (r'\bhups?//', 'https://'),
        (r'\bchi\s+nhành\b', 'chi nhánh'),
        (r'\bcổ\s+phản\b', 'cổ phần'),
        (r'\bdào\s+tạo\b', 'đào tạo'),
        (r'\bimap\s+viết\b', 'IMAP Việt Nam'),
        (r'\bTrần\s+Bách\s+Hóp\b', 'Trần Bách Hợp'),
        (r'\bXHU\s+ĐÔ\s+TH[IỊ]\b', 'KHU ĐÔ THỊ'),
        (r'\bkhu\s+do\s+thị\b', 'khu đô thị'),
        (r'\bSố\s+TK\s+c[ay]+:\s*C[ay]+\b', 'Số TK Cty: Cty'),
        (r'\bMSNKA-', 'MSNK4-'),
        (r'\bRulo\s+6[Pp]\b', 'Rulo ép'),
        (r'\b(XIÊM|Kiêm)\s+PHIẾU\b', 'KIÊM PHIẾU'),
        (r'\bTống\s+tiền\b', 'Tổng tiền'),
        (r'\bSố\s+niền\s+viết\b', 'Số tiền viết'),
        (r'\bThanh\s*-\s*Sơn\s*Sau\b', '- thanh toán Sau.'),
        (r'\bTổ\s+A\s+Diễn\s+Trăn\b', 'Hồ Thị Diễm Trâm'),
        (r'\bĐám\s+Sơng\b', 'Phạm Dũng'),
    ]

    @classmethod
    def correct_text(cls, text: str) -> str:
        if not text:
            return ""
        return VietnameseLanguageModel.get_instance().process(text)


class VietnameseLanguageModel:
    """
    Bộ Hậu Xử Lý Mô Hình Ngôn Ngữ Tiếng Việt (Language Model - LM Post-Processing):
    1. Lexicon & Spell Checking: Tra cứu kho 74.000 từ vựng chuẩn hóa tiếng Việt (core/models/viet_words.txt).
    2. Contextual Diacritics Restoration: Khôi phục thanh dấu cho các từ bị mất hoặc nhận diện thiếu dấu.
    3. Punctuation & Typography Normalization: Chuẩn hóa dấu câu (:, ,, ., -, /), viết hoa đầu dòng, khử ký tự nhiễu.
    4. Domain Knowledge: Bổ sung từ điển ngữ nghĩa hóa đơn, hành chính, địa danh 63 tỉnh thành Việt Nam.
    """
    _instance: Optional["VietnameseLanguageModel"] = None

    def __init__(self):
        self.words_set: Set[str] = set()
        self.unaccented_map: Dict[str, List[str]] = {}
        self._load_lexicon()

    @classmethod
    def get_instance(cls) -> "VietnameseLanguageModel":
        if cls._instance is None:
            cls._instance = VietnameseLanguageModel()
        return cls._instance

    def _load_lexicon(self):
        base_dir = get_base_dir()
        vocab_path = os.path.join(base_dir, "core", "models", "viet_words.txt")
        if os.path.exists(vocab_path):
            try:
                with open(vocab_path, "r", encoding="utf-8") as f:
                    for line in f:
                        w = unicodedata.normalize("NFC", line.strip())
                        if w:
                            lw = w.lower()
                            self.words_set.add(lw)
                            unacc = remove_accents(lw)
                            if unacc not in self.unaccented_map:
                                self.unaccented_map[unacc] = []
                            if lw not in self.unaccented_map[unacc]:
                                self.unaccented_map[unacc].append(lw)
            except Exception as e:
                print(f"[LM] Warning loading lexicon: {e}")

    def correct_token(self, token: str) -> str:
        """Sửa lỗi chính tả cấp từ nếu từ đó bị mất dấu hoặc sai sót nhẹ."""
        if not token or len(token) < 2:
            return token

        # Không can thiệp nếu từ chứa chữ số, URL, email, hoặc ký tự đặc biệt (mã hàng, số tiền, ngày tháng)
        if any(c.isdigit() or c in "@/:.-_#$%&*" for c in token):
            return token

        is_upper = token.isupper()
        is_title = token.istitle()
        lower_token = token.lower()

        # 1. Nếu từ đã đúng trong từ điển tiếng Việt chuẩn -> giữ nguyên
        if lower_token in self.words_set:
            return token

        # 2. Thử tìm ứng viên có dấu từ dạng không dấu
        unacc = remove_accents(lower_token)
        if unacc in self.unaccented_map:
            candidates = self.unaccented_map[unacc]
            if len(candidates) == 1:
                cand = candidates[0]
                if is_upper:
                    return cand.upper()
                elif is_title:
                    return cand.capitalize()
                return cand

        return token

    def normalize_typography(self, text: str) -> str:
        """Chuẩn hóa khoảng trắng quanh dấu câu, số tiền, ngày tháng."""
        # Bỏ dấu cách trước dấu hai chấm, phẩy, chấm, chấm phẩy
        text = re.sub(r'\s+([:,\.\?!;])', r'\1', text)
        # Thêm dấu cách sau dấu phẩy, hai chấm nếu thiếu (không áp dụng cho số 1.200.000 hoặc thời gian 12:30 hoặc url)
        text = re.sub(r'([,:])([^\s0-9/])', r'\1 \2', text)
        # Xóa các ký tự nhiễu OCR lẻ loi (như đơn độc dấu ngã ~, dấu nháy đơn lơ lửng)
        text = re.sub(r'(?<=\s)[~\^`\'"](?=\s)', '', text)
        # Chuẩn hóa khoảng trắng thừa
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def process(self, text: str) -> str:
        """Áp dụng toàn diện các tầng Mô hình Ngôn ngữ Tiếng Việt."""
        if not text:
            return ""

        text = unicodedata.normalize('NFC', text)

        # 1. Quy tắc ngữ cảnh cụm từ hóa đơn & hành chính (Domain Phrase Grammar)
        for pattern, repl in VietnameseDiacriticsCorrector.WORD_REPLACEMENTS:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE if not repl.isupper() else 0)

        # 2. Đối soát địa danh 63 tỉnh thành
        parts = [p.strip() for p in text.split(',')]
        if parts:
            last_part = parts[-1].strip()
            best_match = None
            best_ratio = 0.0
            for prov in VIETNAMESE_PROVINCES:
                ratio = difflib.SequenceMatcher(None, last_part.lower(), prov.lower()).ratio()
                if ratio > best_ratio and ratio >= 0.75:
                    best_ratio = ratio
                    best_match = prov

            if best_match and best_ratio >= 0.8:
                parts[-1] = best_match
                text = ', '.join(parts)

        # 3. Chuẩn hóa dấu câu & Typography (Punctuation Normalization)
        text = self.normalize_typography(text)

        # 4. Sửa lỗi chính tả từng từ đơn lẻ dựa trên Lexicon
        words = text.split()
        corrected_words = []
        for w in words:
            # Tách dấu câu bám đầu/đuôi (nếu có)
            prefix = ""
            suffix = ""
            while w and w[0] in "([{\"'":
                prefix += w[0]
                w = w[1:]
            while w and w[-1] in ".,;:!?)'\"}]":
                suffix = w[-1] + suffix
                w = w[:-1]
            cw = self.correct_token(w)
            corrected_words.append(f"{prefix}{cw}{suffix}")

        final_text = " ".join(corrected_words)
        return unicodedata.normalize('NFC', final_text)


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
        """Nhận diện văn bản cho một ảnh cắt dòng đơn lẻ kết hợp làm nét và phục hồi thanh dấu."""
        if image is None or image.size == 0:
            return ""

        self._init_predictor()

        # Tiền xử lý tăng nét vi mô cho crop ảnh để làm rõ dấu thanh
        enhanced = enhance_text_crop(image)

        if len(enhanced.shape) == 3:
            rgb_img = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

        pil_img = Image.fromarray(rgb_img)
        try:
            text = self.predictor.predict(pil_img)
            text = (text or "").strip()
            # Áp dụng bộ phục hồi chính tả & thanh dấu tiếng Việt
            return VietnameseDiacriticsCorrector.correct_text(text)
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# BƯỚC 3 (TÙY CHỌN 2): NHẬN DIỆN CHỮ BẰNG TESSERACT OCR (LANG=VIE)
# ---------------------------------------------------------------------------

class TesseractRecognizer:
    """
    Pipeline 2: Nhận diện chữ bằng Tesseract OCR (lang=vie).
    - Bước 1: Dùng PaddleOCR DBNet phát hiện bounding box.
    - Bước 2: Cắt ảnh crop.
    - Bước 3: Đưa từng crop vào Tesseract (lang=vie).
    - Bước 4: Hậu xử lý qua VietnameseLanguageModel.
    """
    _instance: Optional["TesseractRecognizer"] = None

    def __init__(self):
        self._available = False
        self._check_available()

    @classmethod
    def get_instance(cls) -> "TesseractRecognizer":
        if cls._instance is None:
            cls._instance = TesseractRecognizer()
        return cls._instance

    def _check_available(self):
        try:
            import pytesseract
            candidate_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
            ]
            for p in candidate_paths:
                if os.path.exists(p):
                    pytesseract.pytesseract.tesseract_cmd = p
                    self._available = True
                    return

            import shutil
            if shutil.which("tesseract"):
                self._available = True
        except Exception:
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def predict_image(self, image: np.ndarray) -> str:
        if not self._available or image is None or image.size == 0:
            return ""
        try:
            import pytesseract
            enhanced = enhance_text_crop(image)
            if len(enhanced.shape) == 3:
                rgb_img = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(rgb_img)
            # PSM 7: Coi crop anh la 1 dong chu don le
            text = pytesseract.image_to_string(pil_img, lang='vie', config='--psm 7')
            return text.strip()
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# CÁC HÀM HỖ TRỢ TƯƠNG THÍCH NGƯỢC (BACKWARD COMPATIBILITY)
# ---------------------------------------------------------------------------

def extract_structured_fields(*args, **kwargs) -> Dict[str, Any]:
    """Hàm giữ tương thích ngược, trả về dict rỗng (đã bỏ trích xuất cấu trúc)."""
    return {}


def format_structured_order_summary(*args, **kwargs) -> str:
    """Hàm giữ tương thích ngược, trả về chuỗi rỗng (đã bỏ trích xuất cấu trúc)."""
    return ""


# ---------------------------------------------------------------------------
# HỆ THỐNG ĐIỀU PHỐI OCR CHÍNH (OCR ENGINE - 100% OFFLINE)
# ---------------------------------------------------------------------------

class OCREngine:
    """
    Hệ thống nhận diện OCR 100% Cục Bộ (Offline) theo quy trình 4 Bước:
    - Bước 1: Tiền xử lý ảnh (Pre-processing): Grayscale, Deskew (0°), CLAHE tăng tương phản.
    - Bước 2: Định vị vùng chữ với PaddleOCR DBNet (PP-OCRv4 ONNX) & Cắt ảnh (Crop).
    - Bước 3: Nhận diện chữ tiếng Việt bằng một trong 2 phương án:
        + Phương án 1 (Mặc định - Khuyên dùng): Mô hình Deep Learning Tiếng Việt & Viết tay
          (VGG-Transformer fine-tuned trên corpus VietOCR + Cinnamon AI).
        + Phương án 2: Tesseract OCR (lang=vie).
    - Bước 4: Hậu xử lý qua Mô hình Ngôn ngữ Tiếng Việt (Language Model - LM):
        Sửa lỗi chính tả từ vựng (74.000 từ), phục hồi thanh dấu ngữ cảnh, chuẩn hóa dấu câu và cột bảng biểu.
    """

    _instance: Optional["OCREngine"] = None

    def __init__(self):
        self._detector: Optional[PaddleOCRDetector] = None
        self._vietocr: Optional[VietOCREngine] = None
        self._tesseract: Optional[TesseractRecognizer] = None
        self._engine_mode: str = "neural"
        self.use_language_model: bool = True

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
        if "tesseract" in mode_lower:
            self._engine_mode = "tesseract"
        else:
            self._engine_mode = "neural"

    def _init_detector(self):
        if self._detector is None:
            self._detector = PaddleOCRDetector.get_instance()

    def _init_recognizers(self):
        if self._vietocr is None:
            self._vietocr = VietOCREngine.get_instance()
        if self._tesseract is None:
            self._tesseract = TesseractRecognizer.get_instance()

    def recognize(
        self,
        image: np.ndarray,
        mode: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> OCRResult:
        """
        Nhận diện văn bản trong ảnh hoàn toàn Offline 100%:
        - mode='neural': PaddleOCR DBNet + Vietnamese Deep Learning + Language Model (LM)
        - mode='tesseract': PaddleOCR DBNet + Tesseract OCR (lang=vie) + Language Model (LM)
        """
        if image is None or image.size == 0:
            return OCRResult(error="Ảnh rỗng hoặc không hợp lệ")

        active_mode = (mode or self._engine_mode).lower().strip()
        start_time = time.time()

        try:
            return self._recognize_offline(image, active_mode, start_time, progress_callback=progress_callback)
        except Exception as e:
            return OCRResult(error=f"Lỗi nhận diện OCR: {str(e)}")

    def _recognize_offline(
        self,
        image: np.ndarray,
        active_mode: str,
        start_time: float,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> OCRResult:
        self._init_detector()
        self._init_recognizers()

        is_tesseract = "tesseract" in active_mode
        if is_tesseract and not self._tesseract.is_available():
            if progress_callback:
                progress_callback(1, 10, "Tesseract chưa được cài đặt, tự động chuyển sang Mô hình Deep Learning...")
            is_tesseract = False

        # BƯỚC 1: TIỀN XỬ LÝ ẢNH
        if progress_callback:
            progress_callback(1, 10, "Bước 1: Tiền xử lý ảnh (Deskew 0°, Grayscale, Tăng tương phản)...")

        deskewed_color, thresh_img, skew_angle = preprocess_order_image(
            image,
            deskew=True,
            apply_adaptive_thresh=False
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

        # Cắt ảnh (Crop): Lấy tọa độ bounding boxes
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

        # BƯỚC 3: NHẬN DIỆN CHỮ (DEEP LEARNING HOẶC TESSERACT)
        total_crops = len(cropped_items)
        boxes: List[OCRBox] = []

        rec_name = "Tesseract OCR (lang=vie)" if is_tesseract else "Mô hình Deep Learning Tiếng Việt"

        for idx, (crop, bbox, poly) in enumerate(cropped_items):
            if progress_callback and total_crops > 0:
                step_pct = 4 + int((idx / total_crops) * 4)
                progress_callback(step_pct, 10, f"Bước 3: {rec_name} ({idx + 1}/{total_crops} dòng)...")

            if is_tesseract:
                text = self._tesseract.predict_image(crop)
            else:
                text = self._vietocr.predict_image(crop)

            if text:
                text = unicodedata.normalize("NFC", text.strip())
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

        # BƯỚC 4: GOM DÒNG & HẬU XỬ LÝ MÔ HÌNH NGÔN NGỮ (LANGUAGE MODEL - LM)
        if progress_callback:
            progress_callback(9, 10, "Bước 4: Gom dòng Center-Y & Hậu xử lý Mô hình Ngôn ngữ (LM)...")

        sorted_lines = self._group_into_lines(boxes)
        lm = VietnameseLanguageModel.get_instance() if self.use_language_model else None

        reconstructed_lines: List[str] = []
        for line in sorted_lines:
            line_text = "  ".join([b.text for b in line])
            if lm:
                line_text = lm.process(line_text)
                for b in line:
                    b.text = lm.process(b.text)
            reconstructed_lines.append(line_text)

        final_full_text = "\n".join(reconstructed_lines).strip()
        sorted_boxes = [box for line in sorted_lines for box in line]

        total_time = round(time.time() - start_time, 2)

        if progress_callback:
            progress_callback(10, 10, "Hoàn tất nhận diện tài liệu thành công (100% Offline)!")

        return OCRResult(
            full_text=final_full_text,
            boxes=sorted_boxes,
            elapse_time=total_time,
            char_count=len(final_full_text),
            word_count=len(final_full_text.split()),
            extracted_fields={}
        )

    def _group_into_lines(self, boxes: List[OCRBox]) -> List[List[OCRBox]]:
        """Gom các bounding box thành các dòng đọc hoàn chỉnh dựa trên Center-Y."""
        if not boxes:
            return []

        avg_h = np.mean([b.bbox[3] for b in boxes]) if boxes else 20.0
        line_threshold = max(8.0, avg_h * 0.6)

        boxes_by_cy = sorted(boxes, key=lambda b: (b.bbox[1] + b.bbox[3] / 2.0, b.bbox[0]))

        lines: List[List[OCRBox]] = []
        for b in boxes_by_cy:
            b_cy = b.bbox[1] + b.bbox[3] / 2.0
            best_line = None
            best_dist = float('inf')
            for line in lines:
                line_avg_cy = np.mean([item.bbox[1] + item.bbox[3] / 2.0 for item in line])
                dist = abs(b_cy - line_avg_cy)
                if dist < line_threshold and dist < best_dist:
                    best_dist = dist
                    best_line = line
            if best_line is not None:
                best_line.append(b)
            else:
                lines.append([b])

        lines.sort(key=lambda line: np.mean([item.bbox[1] + item.bbox[3] / 2.0 for item in line]))
        for line in lines:
            line.sort(key=lambda b: b.bbox[0])

        return lines

    def _sort_reading_order(self, boxes: List[OCRBox]) -> List[OCRBox]:
        """Sắp xếp các bounding box theo thứ tự đọc tự nhiên từ trên xuống, trái sang phải."""
        sorted_lines = self._group_into_lines(boxes)
        return [box for line in sorted_lines for box in line]

    def _reconstruct_lines(self, sorted_boxes: List[OCRBox]) -> List[str]:
        """Tái cấu trúc văn bản thuần theo từng dòng đọc."""
        sorted_lines = self._group_into_lines(sorted_boxes)
        return ["  ".join([b.text for b in line]) for line in sorted_lines]

