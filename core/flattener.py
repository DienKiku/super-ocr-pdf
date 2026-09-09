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
    def order_points(pts: np.ndarray) -> Optional[np.ndarray]:
        """
        Order 4 corner points: [top-left, top-right, bottom-right, bottom-left].
        Uses centroid angle sorting to guarantee all 4 vertices are unique and sequential.
        """
        pts = np.array(pts, dtype=np.float32)
        if len(pts) != 4:
            return None

        # Guarantee 4 distinct vertices (no collapsed/coincident points)
        for i in range(4):
            for j in range(i + 1, 4):
                if np.linalg.norm(pts[i] - pts[j]) < 20:
                    return None

        center = np.mean(pts, axis=0)
        diff = pts - center
        angles = np.arctan2(diff[:, 1], diff[:, 0])
        pts_sorted = pts[np.argsort(angles)]

        # Top-left is closest to origin (min x + y)
        s = pts_sorted[:, 0] + pts_sorted[:, 1]
        tl_idx = np.argmin(s)
        pts_ordered = np.roll(pts_sorted, -tl_idx, axis=0)

        # Ensure clockwise ordering: cross product of (P1 - P0) and (P3 - P0)
        v1 = pts_ordered[1] - pts_ordered[0]
        v2 = pts_ordered[3] - pts_ordered[0]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross < 0:
            pts_ordered = np.array([pts_ordered[0], pts_ordered[3], pts_ordered[2], pts_ordered[1]])

        return pts_ordered

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
        img_area = h * w
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

        smooth = cv2.bilateralFilter(small, 9, 75, 75)
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY) if len(smooth.shape) == 3 else smooth
        hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV) if len(smooth.shape) == 3 else None

        masks = []
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        # Strategy 1: Saturation mask for white/cream paper on colored backgrounds (chairs, desks, wood)
        if hsv is not None:
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            _, s_thresh = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            v_mask = (val > 50).astype(np.uint8) * 255
            sat_paper = cv2.bitwise_and(s_thresh, v_mask)
            masks.append(('sat_otsu', cv2.morphologyEx(sat_paper, cv2.MORPH_CLOSE, k_close, borderType=cv2.BORDER_CONSTANT, borderValue=0)))

        # Strategy 2: Grayscale Otsu thresholding
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(('gray_otsu', cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, k_close, borderType=cv2.BORDER_CONSTANT, borderValue=0)))

        # Strategy 3: Adaptive thresholding on lightness
        thresh_adapt = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 6
        )
        masks.append(('adapt', cv2.morphologyEx(thresh_adapt, cv2.MORPH_CLOSE, k_close, borderType=cv2.BORDER_CONSTANT, borderValue=0)))

        # Strategy 4: Inverted Otsu if document is darker than background
        _, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        masks.append(('inv_otsu', cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE, k_close, borderType=cv2.BORDER_CONSTANT, borderValue=0)))

        candidates = []

        for name, mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                frac = area / small_area
                # Skip if too small or too large
                if frac < 0.10 or frac > 0.99:
                    continue

                peri = cv2.arcLength(cnt, True)
                found_approx = False

                # Try multi-epsilon polygon approximation
                for eps in [0.015, 0.02, 0.03, 0.04, 0.05]:
                    approx = cv2.approxPolyDP(cnt, eps * peri, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        pts = approx.reshape(4, 2) / scale - pad
                        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

                        ordered = cls.order_points(pts)
                        if ordered is not None:
                            top_w = np.linalg.norm(ordered[1] - ordered[0])
                            bot_w = np.linalg.norm(ordered[2] - ordered[3])
                            left_h = np.linalg.norm(ordered[3] - ordered[0])
                            right_h = np.linalg.norm(ordered[2] - ordered[1])
                            avg_w = (top_w + bot_w) / 2.0
                            avg_h = (left_h + right_h) / 2.0
                            if avg_w > 0 and avg_h > 0 and 0.25 <= (avg_w / avg_h) <= 4.0:
                                p_ratio = cv2.contourArea(ordered) / img_area
                                if 0.10 <= p_ratio <= 0.99:
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
                    if ordered_box is not None:
                        top_w = np.linalg.norm(ordered_box[1] - ordered_box[0])
                        bot_w = np.linalg.norm(ordered_box[2] - ordered_box[3])
                        left_h = np.linalg.norm(ordered_box[3] - ordered_box[0])
                        right_h = np.linalg.norm(ordered_box[2] - ordered_box[1])
                        avg_w = (top_w + bot_w) / 2.0
                        avg_h = (left_h + right_h) / 2.0
                        if avg_w > 0 and avg_h > 0 and 0.25 <= (avg_w / avg_h) <= 4.0:
                            p_ratio = cv2.contourArea(ordered_box) / img_area
                            if 0.10 <= p_ratio <= 0.99:
                                candidates.append((area * 1.0, ordered_box))

        if not candidates:
            return None

        # Sort by candidate score (area * confidence)
        candidates.sort(key=lambda c: c[0], reverse=True)
        best_quad = candidates[0][1]
        return best_quad

    @classmethod
    def crop_and_flatten(
        cls, img: np.ndarray, safety_margin: float = 0.01
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

        # Expand slightly by safety_margin (1%) to guarantee edge text, stamps, and signatures are preserved
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

        # Sanity check: must be at least 100x100
        if max_w < 100 or max_h < 100:
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
        if not np.all(np.isfinite(M)):
            return img, False

        warped = cv2.warpPerspective(
            img, M, (max_w, max_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
        )

        # Variance guard: Warped image must NOT be a degenerate blank/solid tone
        gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
        if gray_w.var() < 15.0:
            return img, False

        return warped, True
