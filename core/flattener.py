"""
Smart Document Boundary Detection, Perspective Correction & Auto-Crop Module.
Emulates professional document scanners (CamScanner, Adobe Scan, Microsoft Lens):
- Automatically detects document boundaries on desks, floors, beds, and tables.
- Cuts off excess background, shadows, and objects outside the document.
- Warps perspective to produce a flat, rectangular, scan-quality document page.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


class DocumentFlattener:
    """Intelligent document edge detection and perspective rectification."""

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """Order 4 corner points: [top-left, top-right, bottom-right, bottom-left]."""
        pts = np.array(pts, dtype=np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]

        return np.array([tl, tr, br, bl], dtype=np.float32)

    @classmethod
    def find_document_corners(cls, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects the 4 corners of a document sheet on a contrasting or textured surface.
        Returns ordered (4, 2) corner coordinates in original image pixel space,
        or None if no distinct document boundary is found.
        """
        if img is None or img.size == 0:
            return None

        h, w = img.shape[:2]
        pad = 20

        # Add black border padding to prevent paper edges touching frame borders from being clipped
        padded = cv2.copyMakeBorder(
            img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0] if len(img.shape) == 3 else 0
        )
        ph, pw = padded.shape[:2]

        # Work at a normalized resolution for robust multi-scale edge detection
        target_dim = 800
        if max(ph, pw) > target_dim:
            scale = target_dim / float(max(ph, pw))
            small = cv2.resize(padded, (int(pw * scale), int(ph * scale)), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            small = padded

        sh, sw = small.shape[:2]
        small_area = sh * sw

        # Convert to grayscale
        if len(small.shape) == 3:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        else:
            gray = small

        # Bilateral filter to smooth background grain/wood textures while keeping page boundaries sharp
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)

        # Multi-strategy binary edge generation:
        masks = []

        # Strategy A: Otsu thresholding on grayscale
        _, thresh_otsu = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        masks.append(cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, k_close, borderType=cv2.BORDER_CONSTANT, borderValue=0))

        # Strategy B: Adaptive thresholding on lightness
        thresh_adapt = cv2.adaptiveThreshold(
            smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 6
        )
        masks.append(cv2.morphologyEx(thresh_adapt, cv2.MORPH_CLOSE, k_close, borderType=cv2.BORDER_CONSTANT, borderValue=0))

        # Strategy C: Inverted Otsu if document is darker than background
        _, thresh_inv = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        masks.append(cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE, k_close, borderType=cv2.BORDER_CONSTANT, borderValue=0))

        candidates = []

        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                frac = area / small_area
                # Document must cover a substantial part of the frame (15% to 95%)
                if frac < 0.15 or frac > 0.95:
                    continue

                peri = cv2.arcLength(cnt, True)
                found_approx = False

                # Try polygon approximation
                for eps in [0.015, 0.02, 0.03, 0.04, 0.05, 0.06]:
                    approx = cv2.approxPolyDP(cnt, eps * peri, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        pts = approx.reshape(4, 2) / scale - pad
                        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

                        ordered = cls.order_points(pts)
                        # Verify valid aspect ratio (between 0.25 and 4.0)
                        top_w = np.linalg.norm(ordered[1] - ordered[0])
                        bot_w = np.linalg.norm(ordered[2] - ordered[3])
                        left_h = np.linalg.norm(ordered[3] - ordered[0])
                        right_h = np.linalg.norm(ordered[2] - ordered[1])
                        avg_w = (top_w + bot_w) / 2.0
                        avg_h = (left_h + right_h) / 2.0
                        if avg_w > 0 and avg_h > 0:
                            aspect = avg_w / avg_h
                            if 0.25 <= aspect <= 4.0:
                                candidates.append((area * 2.0, ordered))
                                found_approx = True
                                break

                if not found_approx:
                    # Fallback to minAreaRect bounding box
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect) / scale - pad
                    box[:, 0] = np.clip(box[:, 0], 0, w - 1)
                    box[:, 1] = np.clip(box[:, 1], 0, h - 1)
                    ordered_box = cls.order_points(box)
                    top_w = np.linalg.norm(ordered_box[1] - ordered_box[0])
                    bot_w = np.linalg.norm(ordered_box[2] - ordered_box[3])
                    left_h = np.linalg.norm(ordered_box[3] - ordered_box[0])
                    right_h = np.linalg.norm(ordered_box[2] - ordered_box[1])
                    avg_w = (top_w + bot_w) / 2.0
                    avg_h = (left_h + right_h) / 2.0
                    if avg_w > 0 and avg_h > 0:
                        aspect = avg_w / avg_h
                        if 0.25 <= aspect <= 4.0:
                            candidates.append((area * 1.0, ordered_box))

        if not candidates:
            return None

        # Sort by candidate score (area * confidence)
        candidates.sort(key=lambda c: c[0], reverse=True)
        best_quad = candidates[0][1]

        # Check if detected quad is full frame (already scanned/cropped page)
        xs = best_quad[:, 0]
        ys = best_quad[:, 1]
        border_tol_x = 0.025 * w
        border_tol_y = 0.025 * h
        if (xs.min() <= border_tol_x and
            ys.min() <= border_tol_y and
            xs.max() >= (w - 1 - border_tol_x) and
            ys.max() >= (h - 1 - border_tol_y)):
            return None

        return best_quad

    @classmethod
    def crop_and_flatten(
        cls, img: np.ndarray, safety_margin: float = 0.015
    ) -> Tuple[np.ndarray, bool]:
        """
        Detects document corners, cuts off excess background, and warps to a flat rectangle.
        Returns:
            (processed_image, was_cropped_and_flattened)
        """
        if img is None or img.size == 0:
            return img, False

        h, w = img.shape[:2]
        quad = cls.find_document_corners(img)
        if quad is None:
            return img, False

        tl, tr, br, bl = quad

        # Expand slightly by safety_margin (1.5%) to guarantee edge text, stamps, and signatures are preserved
        if safety_margin > 0:
            center = np.mean([tl, tr, br, bl], axis=0)
            tl = center + (tl - center) * (1.0 + safety_margin)
            tr = center + (tr - center) * (1.0 + safety_margin)
            br = center + (br - center) * (1.0 + safety_margin)
            bl = center + (bl - center) * (1.0 + safety_margin)

            # Clamp coordinates to image boundaries
            tl = np.clip(tl, 0, [w - 1, h - 1])
            tr = np.clip(tr, 0, [w - 1, h - 1])
            br = np.clip(br, 0, [w - 1, h - 1])
            bl = np.clip(bl, 0, [w - 1, h - 1])

        # Compute output dimensions (maximum width and height of opposite edges)
        w_a = np.linalg.norm(br - bl)
        w_b = np.linalg.norm(tr - tl)
        max_w = max(int(w_a), int(w_b))

        h_a = np.linalg.norm(tr - br)
        h_b = np.linalg.norm(tl - bl)
        max_h = max(int(h_a), int(h_b))

        # Sanity check: must be at least 150x150
        if max_w < 150 or max_h < 150:
            return img, False

        # Destination coordinates for a flat, upright rectangle
        dst = np.array([
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1]
        ], dtype=np.float32)

        src = np.array([tl, tr, br, bl], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            img, M, (max_w, max_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
        )

        return warped, True
