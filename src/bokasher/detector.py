from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from bokasher.tracker import merge_boxes
from bokasher.types import FaceBox

SCRFD_NAME = "scrfd_2.5g_kps.onnx"
SCRFD_SHA256 = "3f1ac54e769cb5fd76eda11ac3c088eed78d1f51a935a839d04d49b0e770219e"
SCRFD_URLS = (
    "https://huggingface.co/RuteNL/SCRFD-face-detection-ONNX/resolve/main/2.5g_bnkps.onnx",
    "https://huggingface.co/RuteNL/SCRFD-face-detection-ONNX/resolve/main/2.5g_bnkps.onnx?download=true",
)
_INPUT_MEAN = 127.5
_INPUT_STD = 128.0
_STRIDES = (8, 16, 32)
_NUM_ANCHORS = 2
_NMS_THRESHOLD = 0.4


def cache_dir() -> Path:
    return Path.home() / ".cache" / "bokasher"


def default_model_path() -> Path:
    return cache_dir() / SCRFD_NAME


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "bokasher"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != SCRFD_SHA256:
            raise RuntimeError("SCRFD モデルのチェックサムが一致しません。")
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return dest


def ensure_model(model_path: Path | None = None, timeout: float = 60.0) -> Path:
    del timeout
    dest = Path(model_path) if model_path is not None else default_model_path()
    if dest.exists() and dest.stat().st_size > 0:
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest == SCRFD_SHA256:
            return dest
        dest.unlink()
    last_error: Exception | None = None
    for url in SCRFD_URLS:
        try:
            return _download(url, dest)
        except Exception as exc:
            last_error = exc
    raise RuntimeError("SCRFD-2.5G_KPS をダウンロードできませんでした。") from last_error


def choose_onnx_providers() -> list[str]:
    available = set(ort.get_available_providers())
    providers: list[str] = []
    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def detector_display_name(providers: list[str] | None = None) -> str:
    names = providers if providers is not None else choose_onnx_providers()
    if names and names[0] == "CoreMLExecutionProvider":
        return "SCRFD-2.5G_KPS (CoreML)"
    return "SCRFD-2.5G_KPS (CPU)"


class FaceDetector:
    def __init__(
        self,
        min_confidence: float = 0.35,
        model_path: Path | None = None,
        max_side: int = 640,
        use_dual: bool = False,
    ) -> None:
        del use_dual
        self.max_side = max(int(max_side), 64)
        self.min_confidence = min_confidence
        # 公式 ONNX は 640x640 固定。CoreML もこのサイズだけ通る
        self.input_size = (640, 640)
        self._centers: dict[tuple[int, int, int], np.ndarray] = {}
        resolved = ensure_model(model_path)
        self.providers = choose_onnx_providers()
        self._session = ort.InferenceSession(str(resolved), providers=self.providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [item.name for item in self._session.get_outputs()]

    def detect(self, bgr: np.ndarray) -> list[FaceBox]:
        height, width = bgr.shape[:2]
        blob, scale = _letterbox_blob(bgr, self.input_size)
        outputs = self._session.run(self._output_names, {self._input_name: blob})
        boxes = self._decode(outputs, scale, width, height)
        return merge_boxes(boxes)

    def _decode(
        self,
        outputs: list[np.ndarray],
        scale: float,
        width: int,
        height: int,
    ) -> list[FaceBox]:
        if len(outputs) not in (6, 9):
            raise RuntimeError(f"想定外の SCRFD 出力数です: {len(outputs)}")
        levels = 3
        scores_all: list[np.ndarray] = []
        boxes_all: list[np.ndarray] = []
        for index, stride in enumerate(_STRIDES):
            scores = _first_batch(outputs[index])
            distances = _first_batch(outputs[index + levels]) * stride
            fmap_h = self.input_size[1] // stride
            fmap_w = self.input_size[0] // stride
            centers = self._anchor_centers(fmap_h, fmap_w, stride)
            keep = np.where(scores.ravel() >= self.min_confidence)[0]
            if keep.size == 0:
                continue
            decoded = _distance2bbox(centers, distances)[keep]
            scores_all.append(scores.ravel()[keep])
            boxes_all.append(decoded)

        if not boxes_all:
            return []

        boxes = np.vstack(boxes_all) / scale
        scores = np.concatenate(scores_all)
        order = np.argsort(-scores, kind="stable")
        boxes = boxes[order]
        scores = scores[order]
        keep = _nms(boxes, scores, _NMS_THRESHOLD)
        result: list[FaceBox] = []
        for index in keep:
            x1, y1, x2, y2 = boxes[index]
            result.append(
                FaceBox(
                    x=int(round(float(x1))),
                    y=int(round(float(y1))),
                    w=int(round(float(x2 - x1))),
                    h=int(round(float(y2 - y1))),
                    score=float(scores[index]),
                ).clamp(width, height)
            )
        return result

    def _anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        cached = self._centers.get(key)
        if cached is not None:
            return cached
        grid = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        centers = (grid * stride).reshape(-1, 2)
        if _NUM_ANCHORS > 1:
            centers = np.stack([centers] * _NUM_ANCHORS, axis=1).reshape(-1, 2)
        self._centers[key] = centers
        return centers

    def close(self) -> None:
        self._session = None

    def __enter__(self) -> FaceDetector:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _letterbox_blob(bgr: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    height, width = bgr.shape[:2]
    target_w, target_h = input_size
    image_ratio = height / max(width, 1)
    model_ratio = target_h / target_w
    if image_ratio > model_ratio:
        new_h = target_h
        new_w = max(int(new_h / image_ratio), 1)
    else:
        new_w = target_w
        new_h = max(int(new_w * image_ratio), 1)
    scale = new_h / max(height, 1)
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    blob = cv2.dnn.blobFromImage(
        canvas,
        1.0 / _INPUT_STD,
        input_size,
        (_INPUT_MEAN, _INPUT_MEAN, _INPUT_MEAN),
        swapRB=True,
    )
    return np.ascontiguousarray(blob, dtype=np.float32), scale


def _first_batch(output: np.ndarray) -> np.ndarray:
    if output.ndim == 3:
        return output[0]
    return output


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[current], x1[order[1:]])
        yy1 = np.maximum(y1[current], y1[order[1:]])
        xx2 = np.minimum(x2[current], x2[order[1:]])
        yy2 = np.minimum(y2[current], y2[order[1:]])
        width = np.maximum(0.0, xx2 - xx1 + 1.0)
        height = np.maximum(0.0, yy2 - yy1 + 1.0)
        overlap = width * height
        iou = overlap / (areas[current] + areas[order[1:]] - overlap + 1e-6)
        order = order[1:][iou <= threshold]
    return keep
