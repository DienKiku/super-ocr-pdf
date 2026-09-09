"""
Smart Document Boundary Detection, Perspective Correction & Auto-Crop Module.
Pipeline: Canny Edge Detection → Contour & Missing-Corner Reconstruction → Hough Lines Fallback → Homography → Perspective Warp.
Emulates professional document scanners (CamScanner, Adobe Scan, Microsoft Lens):
- Automatically detects document boundaries on desks, floors, beds, and tables.
- Reconstructs missing/cut-off corners from the intersection of adjacent edges.
- Cuts off excess background, shadows, and objects outside the document.
- Warps perspective to produce a flat, rectangular, scan-quality document page.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


class DocumentFlattener:
    """Intelligent document edge detection and perspective rectification."""

    @staticmethod
    def line_from_points(p1, p2) -> Tuple[float, float, float]:
        """Line equation ax + by + c = 0 from two points."""
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        return a, b, c

    @staticmethod
    def line_intersection(l1: Tuple[float, float, float], l2: Tuple[float, float, float]) -> Optional[np.ndarray]:
        """Computes intersection (x, y) of two lines a1*x + b1*y + c1 = 0 and a2*x + b2*y + c2 = 0."""
        a1, b1, c1 = l1
        a2, b2, c2 = l2
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-6:
            return None
        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det
        return np.array([x, y], dtype=np.float32)

    @staticmethod
    def order_points(pts: np.ndarray) -> Optional[np.ndarray]:
        """
        Order 4 corner points: [top-left, top-right, bottom-right, bottom-left].
        Uses sum/diff method for robust ordering regardless of rotation.
        """
        pts = np.array(pts, dtype=np.float32)
        if len(pts) != 4:
            return None

        # Guarantee 4 distinct vertices (no collapsed/coincident points)
        for i in range(4):
            for j in range(i + 1, 4):
                if np.linalg.norm(pts[i] - pts[j]) < 15:
                    return None

        ordered = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)                    # x + y
        diff = np.diff(pts, axis=1).flatten()  # y - x

        ordered[0] = pts[np.argmin(s)]         # top-left: smallest sum
        ordered[2] = pts[np.argmax(s)]         # bottom-right: largest sum
        ordered[1] = pts[np.argmin(diff)]      # top-right: smallest diff (largest x - y)
        ordered[3] = pts[np.argmax(diff)]      # bottom-left: largest diff (smallest x - y)

        # Validate: ordered points must form a valid convex polygon with positive area
        area = 0.5 * abs(
            (ordered[1][0] - ordered[0][0]) * (ordered[2][1] - ordered[0][1])
            - (ordered[2][0] - ordered[0][0]) * (ordered[1][1] - ordered[0][1])
            + (ordered[2][0] - ordered[0][0]) * (ordered[3][1] - ordered[0][1])
            - (ordered[3][0] - ordered[0][0]) * (ordered[2][1] - ordered[0][1])
        )
        if area < 100:
            return None

        return ordered

    @classmethod
    def is_already_full_page(cls, img: np.ndarray) -> bool:
        """Checks if the image is already a clean full-frame document (no external background)."""
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
    def is_valid_document_quad(cls, quad: Optional[np.ndarray], w: int, h: int) -> bool:
        """
        Validates whether a 4-point quadrilateral represents a real perspective document page:
        1. Must span a substantial portion of image width and height (> 35%).
        2. Aspect ratio must be physically valid for documents (between 0.35 and 2.8).
        3. Opposite edges must be roughly parallel (top vs bot <= 14 deg, left vs right <= 14 deg).
        4. Area must be > 20% of image area.
        """
        if quad is None or len(quad) != 4:
            return False
        tl, tr, br, bl = quad

        top = tr - tl
        bot = br - bl
        left = bl - tl
        right = br - tr

        w_top = np.linalg.norm(top)
        w_bot = np.linalg.norm(bot)
        h_left = np.linalg.norm(left)
        h_right = np.linalg.norm(right)

        avg_w = (w_top + w_bot) / 2.0
        avg_h = (h_left + h_right) / 2.0

        if avg_w < w * 0.35 or avg_h < h * 0.35:
            return False

        ratio = avg_w / avg_h
        if not (0.35 <= ratio <= 2.8):
            return False

        ang_top = np.degrees(np.arctan2(top[1], top[0]))
        ang_bot = np.degrees(np.arctan2(bot[1], bot[0]))
        diff_tb = abs(ang_top - ang_bot)
        while diff_tb > 180:
            diff_tb = abs(diff_tb - 360)

        ang_left = np.degrees(np.arctan2(left[1], left[0]))
        ang_right = np.degrees(np.arctan2(right[1], right[0]))
        diff_lr = abs(ang_left - ang_right)
        while diff_lr > 180:
            diff_lr = abs(diff_lr - 360)

        if diff_tb > 14.0 or diff_lr > 14.0:
            return False

        area = cv2.contourArea(quad)
        if area < (w * h * 0.20):
            return False

        return True

    @classmethod
    def _detect_corners_contour(cls, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Step 1 & 2: Canny Edge Detection + Contour filtering to find 4 corners.
        If a corner is clipped / missing (5-point polygon), estimates the missing
        corner from the intersection of adjacent edge lines.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        canny = cv2.Canny(blurred, 30, 100)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(canny, k, iterations=2)

        cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # Check for 4-point convex hull first
        for c in cnts[:5]:
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            for eps in np.linspace(0.02, 0.045, 12):
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    quad = approx.reshape(4, 2).astype(np.float32)
                    ordered = cls.order_points(quad)
                    if cls.is_valid_document_quad(ordered, w, h):
                        return ordered

        # Check for 5-point convex hull (one corner cut into a chamfer)
        for c in cnts[:5]:
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            for eps in np.linspace(0.015, 0.035, 10):
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 5 and cv2.isContourConvex(approx):
                    pts = approx.reshape(5, 2).astype(np.float32)
                    lengths = [np.hypot(pts[(i + 1) % 5, 0] - pts[i, 0], pts[(i + 1) % 5, 1] - pts[i, 1]) for i in range(5)]
                    min_idx = np.argmin(lengths)

                    p_prev = pts[(min_idx - 1) % 5]
                    p_c1 = pts[min_idx]
                    p_c2 = pts[(min_idx + 1) % 5]
                    p_next = pts[(min_idx + 2) % 5]

                    l1 = cls.line_from_points(p_prev, p_c1)
                    l2 = cls.line_from_points(p_c2, p_next)
                    est_corner = cls.line_intersection(l1, l2)
                    if est_corner is not None:
                        reconstructed = []
                        for i in range(5):
                            if i == min_idx:
                                reconstructed.append(est_corner)
                            elif i == (min_idx + 1) % 5:
                                continue
                            else:
                                reconstructed.append(pts[i])
                        reconstructed = np.array(reconstructed, dtype=np.float32)
                        ordered = cls.order_points(reconstructed)
                        if cls.is_valid_document_quad(ordered, w, h):
                            return ordered

        return None

    @classmethod
    def _detect_corners_lines(cls, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Fallback Method: Line Intersection.
        Detects document edge lines using Hough transform, filters lines matching text orientation,
        identifies Top, Bottom, Left, Right boundaries, and computes pairwise intersections.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        canny = cv2.Canny(blurred, 30, 100)

        lines = cv2.HoughLinesP(
            canny, 1, np.pi / 180,
            threshold=45,
            minLineLength=int(min(w, h) * 0.10),
            maxLineGap=20
        )
        if lines is None:
            return None
        lines = lines.reshape(-1, 4)

        # Detect text orientation angle from Hough lines
        angles = []
        for x1, y1, x2, y2 in lines:
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            while a > 45: a -= 90
            while a < -45: a += 90
            if abs(a) < 25:
                angles.append(a)
        median_angle = float(np.median(angles)) if angles else 0.0

        # Filter horizontal lines matching text tilt
        horiz = []
        for x1, y1, x2, y2 in lines:
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            while a > 90: a -= 180
            while a < -90: a += 180
            if abs(a - median_angle) < 6.0:
                l = np.hypot(x2 - x1, y2 - y1)
                y_mid = (y1 + y2) / 2.0
                horiz.append((x1, y1, x2, y2, l, y_mid, a))

        # Filter vertical lines roughly perpendicular (80-100 deg)
        vert = []
        for x1, y1, x2, y2 in lines:
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            while a < 0: a += 180
            if abs(a - 90.0) < 15.0:
                l = np.hypot(x2 - x1, y2 - y1)
                x_mid = (x1 + x2) / 2.0
                vert.append((x1, y1, x2, y2, l, x_mid, a))

        if not horiz:
            return None

        horiz.sort(key=lambda x: x[5])
        top_line_data = horiz[0]
        bot_line_data = horiz[-1]

        # Must span at least 40% of image height
        if (bot_line_data[5] - top_line_data[5]) < h * 0.4:
            return None

        l_top = cls.line_from_points((top_line_data[0], top_line_data[1]), (top_line_data[2], top_line_data[3]))
        l_bot = cls.line_from_points((bot_line_data[0], bot_line_data[1]), (bot_line_data[2], bot_line_data[3]))

        if vert:
            vert.sort(key=lambda x: x[5])
            left_line_data = vert[0]
            right_line_data = vert[-1]

            if left_line_data[5] < w * 0.25:
                l_left = cls.line_from_points((left_line_data[0], left_line_data[1]), (left_line_data[2], left_line_data[3]))
            else:
                l_left = cls.line_from_points((0, 0), (0, h))

            if right_line_data[5] > w * 0.75:
                l_right = cls.line_from_points((right_line_data[0], right_line_data[1]), (right_line_data[2], right_line_data[3]))
            else:
                l_right = cls.line_from_points((w - 1, 0), (w - 1, h))
        else:
            l_left = cls.line_from_points((0, 0), (0, h))
            l_right = cls.line_from_points((w - 1, 0), (w - 1, h))

        tl = cls.line_intersection(l_top, l_left)
        tr = cls.line_intersection(l_top, l_right)
        br = cls.line_intersection(l_bot, l_right)
        bl = cls.line_intersection(l_bot, l_left)

        if any(p is None for p in [tl, tr, br, bl]):
            return None

        # Extend top-right if bottom-right reaches right border but top-right was cut early
        if br[0] >= w - 15 and tr[0] < w - 25:
            slope_top = (top_line_data[3] - top_line_data[1]) / max(1e-5, (top_line_data[2] - top_line_data[0]))
            tr[0] = float(w - 1)
            tr[1] = top_line_data[1] + slope_top * (tr[0] - top_line_data[0])

        quad = np.array([tl, tr, br, bl], dtype=np.float32)
        quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)

        ordered = cls.order_points(quad)
        if cls.is_valid_document_quad(ordered, w, h):
            return ordered

        return None

    @classmethod
    def find_document_corners(cls, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Full Corner Detection Pipeline:
        Step 1: Contour-based 4-corner detection + Missing corner estimation.
        Step 2: Line intersection fallback.
        Returns ordered (4, 2) array: [TL, TR, BR, BL] or None.
        """
        if img is None or img.size == 0:
            return None

        # Check if already a clean full-page scan
        if cls.is_already_full_page(img):
            return None

        # Try contour detection first
        corners = cls._detect_corners_contour(img)
        if corners is not None:
            return corners

        # Try line intersection fallback
        corners = cls._detect_corners_lines(img)
        if corners is not None:
            return corners

        return None

    @classmethod
    def crop_and_flatten(
        cls, img: np.ndarray, safety_margin: float = 0.0
    ) -> Tuple[np.ndarray, bool]:
        """
        Step 3 & 4: Computes Homography / Perspective Matrix and warps perspective to flatten.
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

        # Expand slightly by safety_margin if requested
        if safety_margin > 0:
            center = np.mean([tl, tr, br, bl], axis=0)
            tl = center + (tl - center) * (1.0 + safety_margin)
            tr = center + (tr - center) * (1.0 + safety_margin)
            br = center + (br - center) * (1.0 + safety_margin)
            bl = center + (bl - center) * (1.0 + safety_margin)

            for pt in [tl, tr, br, bl]:
                pt[0] = np.clip(pt[0], 0, w - 1)
                pt[1] = np.clip(pt[1], 0, h - 1)

        # Compute output dimensions (maximum width and height of opposite edges)
        w_top = np.linalg.norm(tr - tl)
        w_bot = np.linalg.norm(br - bl)
        max_w = max(int(w_top), int(w_bot))

        h_left = np.linalg.norm(bl - tl)
        h_right = np.linalg.norm(br - tr)
        max_h = max(int(h_left), int(h_right))

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
            img, M, (max_w, max_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE
        )

        # Variance guard: Warped image must NOT be a degenerate blank/solid tone
        gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
        if gray_w.var() < 15.0:
            return img, False

        return warped, True
