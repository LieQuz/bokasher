from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from bokasher.tracker import merge_boxes
from bokasher.types import FaceBox

MEDIAPIPE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MEDIAPIPE_NAME = "blaze_face_short_range.tflite"
YUNET_NAME = "face_detection_yunet_2023mar.onnx"


def cache_dir() -> Path:
    return Path.home() / ".cache" / "bokasher"


def default_model_path() -> Path:
    return cache_dir() / MEDIAPIPE_NAME


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return dest


def ensure_model(model_path: Path | None = None, timeout: float = 60.0) -> Path:
    del timeout
    return _download(MEDIAPIPE_URL, model_path or default_model_path())


def ensure_yunet_model() -> Path:
    return _download(YUNET_URL, cache_dir() / YUNET_NAME)


class FaceDetector:
    def __init__(
        self,
        min_confidence: float = 0.35,
        model_path: Path | None = None,
        max_side: int = 1280,
        use_dual: bool = True,
    ) -> None:
        self.max_side = max(int(max_side), 64)
        self.min_confidence = min_confidence
        self.use_dual = use_dual
        self._mp = None
        self._yunet = None

        yunet_path = ensure_yunet_model()
        self._yunet = cv2.FaceDetectorYN.create(
            str(yunet_path),
            "",
            (320, 320),
            float(min_confidence),
            0.3,
            50,
        )

        if use_dual:
            resolved = ensure_model(model_path)
            options = vision.FaceDetectorOptions(
                base_options=python.BaseOptions(model_asset_path=str(resolved)),
                min_detection_confidence=min_confidence,
            )
            self._mp = vision.FaceDetector.create_from_options(options)

    def detect(self, bgr: np.ndarray) -> list[FaceBox]:
        height, width = bgr.shape[:2]
        work, scale = _downscale(bgr, self.max_side)
        boxes = self._detect_yunet(work, scale, width, height)
        # MediaPipe は毎フレーム回さず、YuNet が見逃したときだけ使う
        if not boxes and self._mp is not None:
            boxes = self._detect_mediapipe(work, scale, width, height)
        return merge_boxes(boxes)

    def _detect_yunet(
        self, work: np.ndarray, scale: float, width: int, height: int
    ) -> list[FaceBox]:
        if self._yunet is None:
            return []
        h, w = work.shape[:2]
        self._yunet.setInputSize((w, h))
        _ok, faces = self._yunet.detect(work)
        if faces is None:
            return []
        inv = 1.0 / scale
        boxes: list[FaceBox] = []
        for face in faces:
            score = float(face[-1])
            if score < self.min_confidence:
                continue
            boxes.append(
                FaceBox(
                    x=int(round(float(face[0]) * inv)),
                    y=int(round(float(face[1]) * inv)),
                    w=int(round(float(face[2]) * inv)),
                    h=int(round(float(face[3]) * inv)),
                    score=score,
                ).clamp(width, height)
            )
        return boxes

    def _detect_mediapipe(
        self, work: np.ndarray, scale: float, width: int, height: int
    ) -> list[FaceBox]:
        if self._mp is None:
            return []
        rgb = np.ascontiguousarray(cv2.cvtColor(work, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._mp.detect(image)
        if not result.detections:
            return []
        inv = 1.0 / scale
        boxes: list[FaceBox] = []
        for detection in result.detections:
            bbox = detection.bounding_box
            score = 0.0
            if detection.categories:
                score = float(detection.categories[0].score or 0.0)
            boxes.append(
                FaceBox(
                    x=int(round(bbox.origin_x * inv)),
                    y=int(round(bbox.origin_y * inv)),
                    w=int(round(bbox.width * inv)),
                    h=int(round(bbox.height * inv)),
                    score=score,
                ).clamp(width, height)
            )
        return boxes

    def close(self) -> None:
        if self._mp is not None:
            closer = getattr(self._mp, "close", None)
            if closer is not None:
                closer()
            self._mp = None

    def __enter__(self) -> FaceDetector:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _downscale(bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = bgr.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return bgr, 1.0
    scale = max_side / longest
    work = cv2.resize(
        bgr,
        (max(int(width * scale), 1), max(int(height * scale), 1)),
        interpolation=cv2.INTER_AREA,
    )
    return work, scale
