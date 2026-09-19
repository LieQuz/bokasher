from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from bokasher.types import FaceBox

_MASK_CANVAS = 96


def _kernel_size(box: FaceBox, strength: float) -> int:
    base = max(box.w, box.h)
    size = int(base * (0.18 + 0.55 * strength))
    size = max(size, 7)
    if size % 2 == 0:
        size += 1
    return size


@lru_cache(maxsize=1)
def _base_mask() -> np.ndarray:
    # 楕円マスクは小さなキャンバスで一度だけ作り、必要なサイズへ拡大して使う
    canvas = np.zeros((_MASK_CANVAS, _MASK_CANVAS), dtype=np.float32)
    center = (_MASK_CANVAS // 2, _MASK_CANVAS // 2)
    axes = (_MASK_CANVAS // 2 - 1, _MASK_CANVAS // 2 - 1)
    cv2.ellipse(canvas, center, axes, 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(canvas, (0, 0), sigmaX=_MASK_CANVAS * 0.08)


def _ellipse_mask(w: int, h: int) -> np.ndarray:
    return cv2.resize(_base_mask(), (w, h), interpolation=cv2.INTER_LINEAR)


def apply_face_blur(
    bgr: np.ndarray,
    boxes: list[FaceBox],
    strength: float = 0.7,
) -> np.ndarray:
    if not boxes:
        return bgr
    strength = min(max(strength, 0.0), 1.0)
    height, width = bgr.shape[:2]
    expand = 0.18 + 0.22 * strength
    out = bgr.copy()

    for box in boxes:
        region = box.expanded(expand, width, height)
        x, y, w, h = region.x, region.y, region.w, region.h
        if min(w, h) < 4:
            continue
        roi = out[y : y + h, x : x + w]
        if roi.size == 0 or min(roi.shape[:2]) < 4:
            continue
        ksize = _kernel_size(region, strength)
        blurred = _fast_blur(roi, ksize)
        mask = _ellipse_mask(w, h)
        out[y : y + h, x : x + w] = cv2.blendLinear(blurred, roi, mask, 1.0 - mask)
    return out


def _fast_blur(roi: np.ndarray, ksize: int) -> np.ndarray:
    height, width = roi.shape[:2]
    shortest = min(height, width)
    if shortest < 2:
        return roi
    # 大きい顔は縮小してからボカし、見た目を保ったまま計算量を落とす
    longest = max(height, width)
    if longest >= 96 and ksize >= 15 and shortest >= 8:
        if longest >= 480:
            scale = 0.125
        elif longest >= 240:
            scale = 0.25
        else:
            scale = 0.5
        # 端に張り付いた細い箱は 0 画素になって OpenCV が落ちる
        scale = max(scale, 2.0 / shortest)
        if scale < 1.0:
            small_w = max(int(width * scale), 2)
            small_h = max(int(height * scale), 2)
            small = cv2.resize(
                roi, (small_w, small_h), interpolation=cv2.INTER_AREA
            )
            small_k = max(int(ksize * scale) | 1, 3)
            small = cv2.GaussianBlur(small, (small_k, small_k), 0)
            return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    odd = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(roi, (odd, odd), 0)
