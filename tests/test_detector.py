from __future__ import annotations

import numpy as np

from bokasher.blur import apply_face_blur
from bokasher.detector import FaceDetector


def test_detects_face_and_blurs_it(portrait_bgr: np.ndarray) -> None:
    with FaceDetector() as detector:
        boxes = detector.detect(portrait_bgr)
    assert boxes, "顔が検出されませんでした"
    blurred = apply_face_blur(portrait_bgr, boxes, strength=0.9)
    box = boxes[0]
    cx, cy = box.x + box.w // 2, box.y + box.h // 2
    assert not np.array_equal(
        portrait_bgr[cy - 4 : cy + 4, cx - 4 : cx + 4],
        blurred[cy - 4 : cy + 4, cx - 4 : cx + 4],
    )
