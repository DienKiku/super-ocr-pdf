"""
Smart Document Boundary Detection, Perspective Correction & Auto-Crop Module.
Emulates professional document scanners (CamScanner, Adobe Scan, Microsoft Lens):
- Automatically detects document boundaries on desks, floors, beds, and tables.
- Cuts off excess background, shadows, and objects outside the document.
- Warps perspective to produce a flat, rectangular, scan-quality document page.
"""

from typing import Optional, Tuple
from itertools import combinations
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
    def is_valid_quad(
        cls, quad: Optional[np.ndarray], w: int, h: int, content_angle: Optional[float] = None
    ) -> bool:
        """
        Validates whether a 4-point quadrilateral represents a real perspective document page:
        1. Opposite edges must be roughly parallel (top vs bot <= 10 deg, left vs right <= 14 deg).
        2. Quad must not touch all 4 outer canvas boundaries (which indicates the entire image frame).
        3. Aspect ratio must be physically valid for documents (between 0.25 and 4.0).
        4. If text content angle is detected, top/bottom edges should be roughly consistent.
        """
        if quad is None or len(quad) != 4:
            return False

        tl, tr, br, bl = quad
        top_vec = tr - tl
        bot_vec = br - bl
        left_vec = bl - tl
        right_vec = br - tr

        ang_top = np.degrees(np.arctan2(top_vec[1], top_vec[0]))
        ang_bot = np.degrees(np.arctan2(bot_vec[1], bot_vec[0]))
        ang_left = np.degrees(np.arctan2(left_vec[1], left_vec[0]))
        ang_right = np.degrees(np.arctan2(right_vec[1], right_vec[0]))

        diff_tb = abs(ang_top - ang_bot)
        while diff_tb > 180:
            diff_tb = abs(diff_tb - 360)
        diff_lr = abs(ang_left - ang_right)
        while diff_lr > 180:
            diff_lr = abs(diff_lr - 360)

        # Opposite edges must be roughly parallel
        if diff_tb > 10.0 or diff_lr > 14.0:
            return False

        # If content angle is available, top/bot edges must be roughly aligned with text
        if content_angle is not None and abs(content_angle) > 0.5:
            d_top_content = abs(ang_top - content_angle)
            while d_top_content > 180:
                d_top_content = abs(d_top_content - 360)
            if d_top_content > 12.0:
                return False

        # Reject if quad touches all 4 outer borders of the camera frame
        xs, ys = quad[:, 0], quad[:, 1]
        if xs.min() <= 8 and ys.min() <= 8 and xs.max() >= (w - 10) and ys.max() >= (h - 10):
            return False

        # Aspect ratio check
        avg_w = (np.linalg.norm(top_vec) + np.linalg.norm(bot_vec)) / 2.0
        avg_h = (np.linalg.norm(left_vec) + np.linalg.norm(right_vec)) / 2.0
        if avg_w < 100 or avg_h < 100:
            return False
        if not (0.25 <= (avg_w / avg_h) <= 4.0):
            return False

        return True

    @classmethod
    def detect_content_angle(cls, gray: np.ndarray) -> float:
        """Fast detection of document text/printed line orientation angle (in degrees)."""
        h, w = gray.shape
        scale = 600.0 / w
        small = cv2.resize(gray, (600, int(h * scale)))
        edges = cv2.Canny(small, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=70, minLineLength=50, maxLineGap=10)
        if lines is None or len(lines) == 0:
            return 0.0
        angles = []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            while ang > 45:
                ang -= 90
            while ang < -45:
                ang += 90
            if abs(ang) < 35:
                angles.append(ang)
        if len(angles) >= 5:
            return float(np.median(angles))
        return 0.0

    @classmethod
    def find_content_quad(cls, img: np.ndarray, content_angle: float = 0.0) -> Optional[np.ndarray]:
        """
        Fallback for tilted documents or cluttered backgrounds:
        Detects the printed document content (text lines, tables, titles) using local background
        subtraction, computes its oriented bounding box, and safely expands to cover paper margins.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        bg = cv2.medianBlur(gray, 25)
        diff = cv2.subtract(bg, gray)
        _, text_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        # Exclude outer image boundaries
        text_mask[:35, :] = 0
        text_mask[-35:, :] = 0
        text_mask[:, :35] = 0
        text_mask[:, -35:] = 0

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        closed = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, k)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)

        pts = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 120:
                x, y, bw, bh = stats[i, :4]
                if y < 140 and area < 500:
                    continue
                pts.extend([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]])

        if len(pts) < 8:
            return None

        pts = np.array(pts, dtype=np.float32)
        center = np.mean(pts, axis=0)

        use_ang = content_angle if abs(content_angle) > 0.4 else 0.0
        if use_ang != 0.0:
            rad = -np.radians(use_ang)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_pts = np.dot(pts - center, R.T)

            min_u, min_v = np.min(rotated_pts, axis=0)
            max_u, max_v = np.max(rotated_pts, axis=0)

            pad_u = max(20.0, (max_u - min_u) * 0.04)
            pad_v = max(20.0, (max_v - min_v) * 0.03)

            corners_rot = np.array([
                [min_u - pad_u, min_v - pad_v],
                [max_u + pad_u, min_v - pad_v],
                [max_u + pad_u, max_v + pad_v],
                [min_u - pad_u, max_v + pad_v]
            ], dtype=np.float32)

            rad_b = np.radians(use_ang)
            cos_b, sin_b = np.cos(rad_b), np.sin(rad_b)
            R_b = np.array([[cos_b, -sin_b], [sin_b, cos_b]])
            quad = np.dot(corners_rot, R_b.T) + center
            quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
            quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
            return cls.order_points(quad)
        else:
            rect = cv2.minAreaRect(pts)
            (cx, cy), (rw, rh), ang = rect

            # Angle-aware dimension expansion
            if abs(ang) > 45:
                dim_x, dim_y = rh, rw
                new_dim_x = min(float(w - 4), dim_x * 1.05)
                new_dim_y = min(float(h - 4), dim_y * 1.05)
                exp_rect = ((cx, cy), (new_dim_y, new_dim_x), ang)
            else:
                dim_x, dim_y = rw, rh
                new_dim_x = min(float(w - 4), dim_x * 1.05)
                new_dim_y = min(float(h - 4), dim_y * 1.05)
                exp_rect = ((cx, cy), (new_dim_x, new_dim_y), ang)

            box = cv2.boxPoints(exp_rect)
            box[:, 0] = np.clip(box[:, 0], 0, w - 1)
            box[:, 1] = np.clip(box[:, 1], 0, h - 1)
            return cls.order_points(box)

    @classmethod
    def is_already_full_page(cls, img: np.ndarray) -> bool:
        """Checks if the image is already a clean full-frame document (no external table/floor background)."""
        h, w = img.shape[:2]
        cw = max(5, int(w * 0.04))
        ch = max(5, int(h * 0.04))
        corners = [
            img[:ch, :cw],
            img[:ch, -cw:],
            img[-ch:, -cw:],
            img[-ch:, :cw]
        ]
        for c in corners:
            if len(c.shape) == 3:
                hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
                if hsv[:, :, 1].mean() > 50 or hsv[:, :, 2].mean() < 120:
                    return False
            else:
                if c.mean() < 120:
                    return False
        return True

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
        img_area = float(h * w)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        content_ang = cls.detect_content_angle(gray)

        # If already a clean full-page document without external background and straight, preserve as is
        if cls.is_already_full_page(img) and abs(content_ang) < 0.8:
            return None

        pad = 20
        # Add border padding using BORDER_REPLICATE to prevent artificial neutral gray border
        # from corrupting saturation/contrast segmentation of real paper
        padded = cv2.copyMakeBorder(
            img, pad, pad, pad, pad, cv2.BORDER_REPLICATE
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
        p_gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY) if len(smooth.shape) == 3 else smooth
        hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV) if len(smooth.shape) == 3 else None

        masks = []
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

        # Strategy 1: Saturation mask for white/cream paper on colored backgrounds (chairs, desks, wood)
        if hsv is not None:
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            _, s_thresh = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            v_mask = (val > 50).astype(np.uint8) * 255
            sat_paper = cv2.bitwise_and(s_thresh, v_mask)
            masks.append(('sat_otsu', cv2.morphologyEx(sat_paper, cv2.MORPH_CLOSE, k_close, borderType=cv2.BORDER_CONSTANT, borderValue=0)))

        # Strategy 2: Grayscale Otsu thresholding
        _, thresh_otsu = cv2.threshold(p_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(('gray_otsu', cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, k_close, borderType=cv2.BORDER_CONSTANT, borderValue=0)))

        # Strategy 3: Adaptive thresholding on lightness
        thresh_adapt = cv2.adaptiveThreshold(
            p_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 6
        )
        masks.append(('adapt', cv2.morphologyEx(thresh_adapt, cv2.MORPH_CLOSE, k_close, borderType=cv2.BORDER_CONSTANT, borderValue=0)))

        valid_candidates = []

        for name, mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                frac = area / small_area
                if frac < 0.10 or frac > 0.99:
                    continue

                peri = cv2.arcLength(cnt, True)
                for eps in [0.015, 0.02, 0.025, 0.03, 0.035, 0.04]:
                    approx = cv2.approxPolyDP(cnt, eps * peri, True)
                    n = len(approx)
                    raw_pts = approx.reshape(-1, 2) / scale - pad
                    raw_pts[:, 0] = np.clip(raw_pts[:, 0], 0, w - 1)
                    raw_pts[:, 1] = np.clip(raw_pts[:, 1], 0, h - 1)

                    if n == 4 and cv2.isContourConvex(approx):
                        ordered = cls.order_points(raw_pts)
                        if cls.is_valid_quad(ordered, w, h, content_ang):
                            top_v = ordered[1] - ordered[0]
                            bot_v = ordered[2] - ordered[3]
                            l_v = ordered[3] - ordered[0]
                            r_v = ordered[2] - ordered[1]
                            at = np.degrees(np.arctan2(top_v[1], top_v[0]))
                            ab = np.degrees(np.arctan2(bot_v[1], bot_v[0]))
                            al = np.degrees(np.arctan2(l_v[1], l_v[0]))
                            ar = np.degrees(np.arctan2(r_v[1], r_v[0]))
                            dtb = abs(at - ab)
                            while dtb > 180: dtb = abs(dtb - 360)
                            dlr = abs(al - ar)
                            while dlr > 180: dlr = abs(dlr - 360)
                            q_area = cv2.contourArea(ordered)
                            score = q_area - (dtb + dlr) * 10000
                            valid_candidates.append((score, ordered))
                    elif 5 <= n <= 7:
                        for combo in combinations(raw_pts, 4):
                            q = np.array(combo, dtype=np.float32)
                            ordered = cls.order_points(q)
                            if cls.is_valid_quad(ordered, w, h, content_ang):
                                top_v = ordered[1] - ordered[0]
                                bot_v = ordered[2] - ordered[3]
                                l_v = ordered[3] - ordered[0]
                                r_v = ordered[2] - ordered[1]
                                at = np.degrees(np.arctan2(top_v[1], top_v[0]))
                                ab = np.degrees(np.arctan2(bot_v[1], bot_v[0]))
                                al = np.degrees(np.arctan2(l_v[1], l_v[0]))
                                ar = np.degrees(np.arctan2(r_v[1], r_v[0]))
                                dtb = abs(at - ab)
                                while dtb > 180: dtb = abs(dtb - 360)
                                dlr = abs(al - ar)
                                while dlr > 180: dlr = abs(dlr - 360)
                                q_area = cv2.contourArea(ordered)
                                score = q_area - (dtb + dlr) * 10000
                                valid_candidates.append((score, ordered))

        if valid_candidates:
            valid_candidates.sort(key=lambda c: c[0], reverse=True)
            return valid_candidates[0][1]

        # If pure boundary contours failed (e.g. tilted document with background clutter), fallback to content quad!
        c_quad = cls.find_content_quad(img, content_ang)
        if c_quad is not None and cls.is_valid_quad(c_quad, w, h):
            return c_quad

        return None

    @classmethod
    def crop_and_flatten(
        cls, img: np.ndarray, safety_margin: float = 0.0
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
