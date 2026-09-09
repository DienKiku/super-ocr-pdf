"""
Image Enhancement & Super-Resolution Module for Text & Documents.
Provides advanced algorithms for sharpening text, background whitening,
super-resolution upscaling, noise reduction, and document binarization.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import cv2
import numpy as np


@dataclass
class EnhanceParams:
    sharpness: float = 0.8       # 0.0 to 2.0 (0 = off, 0.8 = strong crisp text)
    contrast: float = 0.3        # 0.0 to 1.5 (0 = normal, 0.3 = clear readable)
    whitening: float = 0.4       # 0.0 to 1.0 (remove shadows & whiten paper)
    denoise: float = 0.15        # 0.0 to 1.0 (remove camera noise/grain)
    upscale_factor: float = 2.0  # 1.0, 1.5, 2.0, 3.0, 4.0
    color_mode: str = "color"    # "color", "grayscale", "clean_bw"
    auto_deskew: bool = False    # Auto-straighten tilted pages
    rotation: int = 0            # 0, 90, 180, 270 degrees


PRESETS = {
    "ultra_sharp": EnhanceParams(
        sharpness=0.9,
        contrast=0.35,
        whitening=0.5,
        denoise=0.15,
        upscale_factor=2.0,
        color_mode="color",
        auto_deskew=False,
    ),
    "clean_scan": EnhanceParams(
        sharpness=0.7,
        contrast=0.4,
        whitening=0.75,
        denoise=0.2,
        upscale_factor=2.0,
        color_mode="grayscale",
        auto_deskew=False,
    ),
    "crisp_bw": EnhanceParams(
        sharpness=0.6,
        contrast=0.4,
        whitening=0.85,
        denoise=0.15,
        upscale_factor=2.0,
        color_mode="clean_bw",
        auto_deskew=False,
    ),
    "super_res": EnhanceParams(
        sharpness=0.6,
        contrast=0.25,
        whitening=0.3,
        denoise=0.15,
        upscale_factor=3.0,
        color_mode="color",
        auto_deskew=False,
    ),
    "natural": EnhanceParams(
        sharpness=0.2,
        contrast=0.1,
        whitening=0.1,
        denoise=0.05,
        upscale_factor=1.0,
        color_mode="color",
        auto_deskew=False,
    ),
}


class DocumentEnhancer:
    """High-performance document and text image enhancer."""

    @staticmethod
    def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by 90, 180, or 270 degrees clockwise."""
        angle = angle % 360
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    @staticmethod
    def detect_skew_angle(gray: np.ndarray) -> float:
        """
        Robustly estimate document skew angle (-30 to 30 degrees)
        using multi-strategy line and text analysis:
        1. Primary: Probabilistic Hough Line Transform on Canny edges (detects
           horizontal table grid lines, underlines, and text baseline strokes).
        2. Vertical line cross-validation (90-degree orthogonal lines).
        3. Statistical IQR outlier rejection + weighted median.
        4. Fallback: Morphological text strips for dense text documents.
        5. Deadband threshold: angles < 0.55 degrees are clamped to 0.0 to prevent
           degrading already-straight documents with interpolation blur.
        """
        if gray is None or gray.size == 0:
            return 0.0

        h, w = gray.shape[:2]
        target_dim = 900
        if max(h, w) > target_dim:
            scale = target_dim / float(max(h, w))
            small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = gray

        sh, sw = small.shape[:2]

        # Strategy 1: Hough Lines on Canny edges (table lines, underlines, printed text lines)
        edges = cv2.Canny(small, 50, 150, apertureSize=3)
        min_len = max(20, int(sw * 0.06))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45, minLineLength=min_len, maxLineGap=12)

        angles = []
        weights = []
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx * dx + dy * dy)
                ang = np.degrees(np.arctan2(dy, dx))

                # Near-horizontal lines (text rows, table borders, underlines)
                if abs(ang) <= 30.0:
                    angles.append(ang)
                    weights.append(length)
                # Near-vertical lines (table columns, page margins)
                elif abs(ang) >= 60.0:
                    vert_ang = ang - 90.0 if ang > 0 else ang + 90.0
                    if abs(vert_ang) <= 30.0:
                        angles.append(vert_ang)
                        weights.append(length)

        if len(angles) >= 6:
            angles = np.array(angles)
            weights = np.array(weights)

            # Filter extreme outliers using IQR
            q25, q75 = np.percentile(angles, [25, 75])
            iqr = q75 - q25
            if iqr > 0:
                valid_mask = (angles >= q25 - 1.5 * iqr) & (angles <= q75 + 1.5 * iqr)
                if np.any(valid_mask):
                    angles = angles[valid_mask]
                    weights = weights[valid_mask]

            # Weighted median
            sorted_indices = np.argsort(angles)
            sorted_angles = angles[sorted_indices]
            sorted_weights = weights[sorted_indices]
            cum_weights = np.cumsum(sorted_weights)
            cutoff = cum_weights[-1] / 2.0
            median_idx = np.searchsorted(cum_weights, cutoff)
            final_angle = float(sorted_angles[min(median_idx, len(sorted_angles) - 1)])

            # Deadband threshold: if within 0.55 degrees, document is straight!
            # Do NOT rotate to avoid interpolation blur, aliasing, and corner cutouts.
            if abs(final_angle) < 0.55:
                return 0.0
            return final_angle

        # Strategy 2: Fallback for documents without clear lines (morphological text strips)
        _, thresh = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        k_w = max(20, sw // 25)
        k_h = max(2, sh // 250)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fallback_angles = []
        min_area = sw * sh * 0.0008
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (bw, bh), angle = rect
            if bw < bh:
                bw, bh = bh, bw
                angle = angle + 90.0 if angle < 0 else angle - 90.0
            while angle > 45.0:
                angle -= 90.0
            while angle < -45.0:
                angle += 90.0
            if bh > 0 and (bw / bh) > 3.0 and bw > (sw * 0.12):
                if abs(angle) <= 25.0:
                    fallback_angles.append(angle)

        if len(fallback_angles) >= 4:
            med = float(np.median(fallback_angles))
            if abs(med) < 0.55:
                return 0.0
            return med

        return 0.0

    @staticmethod
    def deskew_image(img: np.ndarray, angle: float) -> np.ndarray:
        """
        Deskew image by rotation angle with smart border padding.
        Samples the edge/corner background color to eliminate ugly white triangular wedges.
        """
        if abs(angle) < 0.55:
            return img

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Sample border color from image corners to seamlessly match background
        is_color = len(img.shape) == 3 and img.shape[2] == 3
        if is_color:
            c1 = img[0:5, 0:5].reshape(-1, 3)
            c2 = img[0:5, -5:].reshape(-1, 3)
            c3 = img[-5:, 0:5].reshape(-1, 3)
            c4 = img[-5:, -5:].reshape(-1, 3)
            corners = np.concatenate([c1, c2, c3, c4], axis=0)
            border_val = tuple(int(v) for v in np.median(corners, axis=0))
        else:
            c1 = img[0:5, 0:5].flatten()
            c2 = img[0:5, -5:].flatten()
            c3 = img[-5:, 0:5].flatten()
            c4 = img[-5:, -5:].flatten()
            corners = np.concatenate([c1, c2, c3, c4], axis=0)
            border_val = int(np.median(corners))

        rotated = cv2.warpAffine(
            img,
            rot_mat,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val
        )
        return rotated

    @staticmethod
    def remove_shadows_and_whiten(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Removes uneven page shadows, phone shadows, and whitens background paper
        using division normalization and selective highlight paper desaturation.
        Preserves vibrant colors such as red stamps, logos, and blue pen ink.
        """
        if strength <= 0.01:
            return img

        is_color = len(img.shape) == 3 and img.shape[2] == 3

        if is_color:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
        else:
            l_channel = img.copy()

        # Large-scale background illumination estimation
        ksize = max(31, min(img.shape[0], img.shape[1]) // 20)
        if ksize % 2 == 0:
            ksize += 1
        bg = cv2.GaussianBlur(l_channel, (ksize, ksize), 0)

        # Division normalization: l_channel / bg * 255
        norm_l = cv2.divide(l_channel, bg, scale=255.0)

        # Blend normalized with original L based on strength
        blended_l = cv2.addWeighted(norm_l, strength, l_channel, 1.0 - strength, 0)

        # Curve stretching to make near-white paper pure white #FFFFFF
        cutoff = int(255 - 45 * strength)
        lookup = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            if i >= cutoff:
                lookup[i] = 255
            else:
                ratio = i / cutoff
                lookup[i] = int(np.clip(255 * (ratio ** 1.15), 0, 255))

        stretched_l = cv2.LUT(blended_l, lookup)

        if is_color:
            # Paper desaturation: neutralize color casts on paper while preserving colored stamps and signatures
            paper_mask = (stretched_l.astype(np.float32) - 220.0) / 35.0
            paper_mask = np.clip(paper_mask, 0.0, 1.0) * (strength * 0.85)

            # Preserve strong color saturation (e.g. red stamps or blue ink)
            color_dist = np.sqrt((a_channel.astype(np.float32) - 128.0) ** 2 + (b_channel.astype(np.float32) - 128.0) ** 2)
            color_preservation = np.clip(color_dist / 18.0, 0.0, 1.0)
            paper_mask = paper_mask * (1.0 - color_preservation)

            new_a = (a_channel.astype(np.float32) * (1.0 - paper_mask) + 128.0 * paper_mask).astype(np.uint8)
            new_b = (b_channel.astype(np.float32) * (1.0 - paper_mask) + 128.0 * paper_mask).astype(np.uint8)

            merged_lab = cv2.merge([stretched_l, new_a, new_b])
            return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        return stretched_l

    @staticmethod
    def enhance_contrast(img: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """
        Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
        to make faint text stand out without overexposing paper.
        """
        if strength <= 0.01:
            return img

        clip_limit = 1.0 + (strength * 3.0)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

        if len(img.shape) == 3 and img.shape[2] == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            enhanced_l = clahe.apply(l)
            merged = cv2.merge([enhanced_l, a, b])
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        else:
            return clahe.apply(img)

    @staticmethod
    def sharpen_text(img: np.ndarray, strength: float = 0.8) -> np.ndarray:
        """
        High-frequency edge and stroke sharpener using Unsharp Masking.
        Makes text edges crisp, eliminates blur without ringing halos.
        """
        if strength <= 0.01:
            return img

        # Unsharp Masking with Gaussian Blur
        sigma = 1.2
        gaussian = cv2.GaussianBlur(img, (0, 0), sigma)
        unsharp_weight = 1.0 + (strength * 0.75)
        blurred_weight = -(strength * 0.75)
        unsharp = cv2.addWeighted(img, unsharp_weight, gaussian, blurred_weight, 0)
        return unsharp

    @staticmethod
    def reduce_noise(img: np.ndarray, strength: float = 0.15) -> np.ndarray:
        """Removes digital camera sensor grain while preserving text edge sharpness."""
        if strength <= 0.01:
            return img

        d = int(5 + strength * 4)
        sigma_color = int(20 + strength * 35)
        sigma_space = int(20 + strength * 35)

        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    @staticmethod
    def super_resolve_upscale(img: np.ndarray, factor: float = 2.0) -> np.ndarray:
        """
        Upscales image using high-quality Lanczos-4 interpolation.
        Produces razor-sharp text scaling without pixelation or ringing artifacts.
        """
        if abs(factor - 1.0) < 0.05:
            return img

        h, w = img.shape[:2]
        new_w = int(w * factor)
        new_h = int(h * factor)

        upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return upscaled

    @staticmethod
    def convert_clean_bw(img: np.ndarray) -> np.ndarray:
        """
        Converts document to ultra-clean high-contrast scanned black and white
        using background illumination division and smooth contrast S-curve.
        Retains anti-aliased font edges, smooth curves, and bold readable text.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Large-scale illumination estimation
        ksize = max(31, min(gray.shape) // 20)
        if ksize % 2 == 0:
            ksize += 1
        bg = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        # Division normalization: gray / bg * 255
        norm = cv2.divide(gray, bg, scale=255.0)

        # Smooth contrast S-curve: clean white background, dark rich text, anti-aliased edges
        norm_f = norm.astype(np.float32)
        scaled = np.clip((norm_f - 60.0) * (255.0 / (225.0 - 60.0)), 0.0, 255.0)
        gamma_corrected = np.clip(255.0 * ((scaled / 255.0) ** 1.3), 0.0, 255.0).astype(np.uint8)

        return gamma_corrected

    @classmethod
    def process(cls, img: np.ndarray, params: EnhanceParams, cancel_check: Optional[Callable[[], bool]] = None) -> np.ndarray:
        """
        Full enhancement pipeline:
        1. Manual Rotation
        2. Auto Deskew (morphological angle detection + clean white padding)
        3. Denoise (bilateral filtering)
        4. Whitening & Shadow Removal (division normalization)
        5. Contrast Enhancement (CLAHE)
        6. Color mode conversion (Color, Grayscale, or Clean B&W)
        7. Text Sharpening (Unsharp Masking)
        8. Super-Resolution Upscaling (Lanczos-4)
        """
        if img is None or img.size == 0:
            return img

        processed = img.copy()

        # Step 1: Manual Rotation
        if params.rotation != 0:
            processed = cls.rotate_image(processed, params.rotation)

        if cancel_check and cancel_check():
            return processed

        # Step 2: Auto Deskew
        if params.auto_deskew:
            if len(processed.shape) == 3:
                gray_for_skew = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                gray_for_skew = processed
            angle = cls.detect_skew_angle(gray_for_skew)
            if abs(angle) >= 0.55:
                processed = cls.deskew_image(processed, angle)

        if cancel_check and cancel_check():
            return processed

        # Step 3: Denoise
        if params.denoise > 0:
            processed = cls.reduce_noise(processed, params.denoise)

        if cancel_check and cancel_check():
            return processed

        # Step 4: Whitening and Shadow Removal
        if params.whitening > 0:
            processed = cls.remove_shadows_and_whiten(processed, params.whitening)

        if cancel_check and cancel_check():
            return processed

        # Step 5: Contrast Enhancement
        if params.contrast > 0:
            processed = cls.enhance_contrast(processed, params.contrast)

        if cancel_check and cancel_check():
            return processed

        # Step 6: Color mode
        if params.color_mode == "grayscale":
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        elif params.color_mode == "clean_bw":
            bw = cls.convert_clean_bw(processed)
            processed = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

        if cancel_check and cancel_check():
            return processed

        # Step 7: Text Sharpening
        if params.sharpness > 0:
            processed = cls.sharpen_text(processed, params.sharpness)

        if cancel_check and cancel_check():
            return processed

        # Step 8: Super-Resolution Upscaling (Lanczos-4)
        if params.upscale_factor > 1.0:
            processed = cls.super_resolve_upscale(processed, params.upscale_factor)

        return processed
