from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from bokasher.blur import apply_face_blur
from bokasher.detector import FaceDetector
from bokasher.ffmpeg import (
    FFmpegReader,
    FFmpegWriter,
    apply_display_rotation,
    extract_frame,
    probe_video,
)
from bokasher.tracker import FaceTracker
from bokasher.types import ProcessSettings, VideoInfo

ProgressCallback = Callable[[int, int | None, float, str], None]


class ProcessingCancelled(Exception):
    pass


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_blurred.mp4")


def even_crop(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    return frame[: height - (height % 2), : width - (width % 2)]


def fit_even_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    if max_side > 0 and max(width, height) > max_side:
        scale = max_side / max(width, height)
        width = int(round(width * scale))
        height = int(round(height * scale))
    width -= width % 2
    height -= height % 2
    return max(width, 2), max(height, 2)


def process_frame_size(info: VideoInfo, settings: ProcessSettings) -> tuple[int, int]:
    width = info.width - (info.width % 2)
    height = info.height - (info.height % 2)
    return fit_even_size(width, height, settings.output_max_side)


def render_preview(input_path: Path, settings: ProcessSettings) -> np.ndarray:
    frame = _pick_preview_frame(Path(input_path), settings)
    with FaceDetector(
        max_side=settings.detect_max_side,
        use_dual=settings.use_dual_detector,
    ) as detector:
        boxes = detector.detect(frame)
    return apply_face_blur(frame, boxes, settings.blur_strength)


def process_video(
    input_path: Path,
    output_path: Path,
    settings: ProcessSettings,
    on_progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    source = Path(input_path)
    dest = Path(output_path)
    info = probe_video(source)
    width, height = process_frame_size(info, settings)

    reader: FFmpegReader | None = None
    cap: cv2.VideoCapture | None = None
    writer: FFmpegWriter | None = None
    started = time.monotonic()
    try:
        use_ffmpeg = settings.prefer_gpu and (
            info.rotation != 0 or width * height >= 1280 * 720
        )
        first, reader, cap = _open_frames(
            source,
            width,
            height,
            rotation=info.rotation,
            prefer_ffmpeg=use_ffmpeg or info.rotation != 0,
        )
        height, width = first.shape[:2]
        writer = FFmpegWriter(
            dest,
            width=width,
            height=height,
            fps_text=info.fps_text,
            fps=info.fps,
            source=source,
            has_audio=info.has_audio,
            prefer_gpu=settings.prefer_gpu,
        )
        encoder_label = writer.encoder
        with FaceDetector(
            max_side=settings.detect_max_side,
            use_dual=settings.use_dual_detector,
        ) as detector:
            tracker = FaceTracker()
            frame_index = 0
            current = first
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(
                    _read_next, reader, cap, info.rotation, width, height
                )
                while True:
                    _raise_if_cancelled(cancel_event)
                    if frame_index % settings.detect_every == 0:
                        boxes = tracker.update(detector.detect(current), width, height)
                    else:
                        boxes = tracker.update(None, width, height)
                    blurred = apply_face_blur(current, boxes, settings.blur_strength)
                    writer.write(blurred)
                    frame_index += 1
                    if on_progress is not None:
                        on_progress(
                            frame_index,
                            info.frame_count,
                            time.monotonic() - started,
                            encoder_label,
                        )
                    nxt = pending.result()
                    if nxt is None:
                        break
                    current = nxt
                    pending = pool.submit(
                        _read_next, reader, cap, info.rotation, width, height
                    )
        writer.close()
        writer = None
    except ProcessingCancelled:
        if writer is not None:
            writer.abort()
        raise
    except Exception:
        if writer is not None:
            writer.abort()
        raise
    finally:
        if reader is not None:
            reader.close()
        if cap is not None:
            cap.release()


def _open_frames(
    source: Path,
    width: int,
    height: int,
    rotation: int,
    prefer_ffmpeg: bool,
) -> tuple[np.ndarray, FFmpegReader | None, cv2.VideoCapture | None]:
    if prefer_ffmpeg:
        try:
            reader = FFmpegReader(source, width, height, hwaccel=True)
            first = reader.read()
            if first is not None:
                return first, reader, None
            reader.close()
        except Exception:
            pass
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {source}")
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("先頭フレームを読めませんでした。")
    return _normalize_frame(first, rotation, width, height), None, cap


def _read_next(
    reader: FFmpegReader | None,
    cap: cv2.VideoCapture | None,
    rotation: int,
    width: int,
    height: int,
) -> np.ndarray | None:
    if reader is not None:
        return reader.read()
    if cap is None:
        return None
    ok, frame = cap.read()
    if not ok:
        return None
    return _normalize_frame(frame, rotation, width, height)


def _normalize_frame(
    frame: np.ndarray,
    rotation: int,
    width: int,
    height: int,
) -> np.ndarray:
    cropped = even_crop(frame)
    if cropped.shape[1] == width and cropped.shape[0] == height:
        return cropped
    rotated = even_crop(apply_display_rotation(frame, rotation))
    if rotated.shape[1] == width and rotated.shape[0] == height:
        return rotated
    return cv2.resize(rotated, (width, height), interpolation=cv2.INTER_AREA)


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ProcessingCancelled()


def _pick_preview_frame(path: Path, settings: ProcessSettings) -> np.ndarray:
    info = probe_video(path)
    width, height = process_frame_size(info, settings)
    times = [0.0]
    if info.duration and info.duration > 0:
        times.extend([min(0.5, info.duration * 0.1), min(1.0, info.duration * 0.25)])
    elif info.fps:
        times.extend([0.5, 1.0])

    best: np.ndarray | None = None
    best_faces = -1
    detector: FaceDetector | None = None
    try:
        for time_s in times:
            frame = extract_frame(path, time_s, width, height)
            if frame is None:
                continue
            if detector is None:
                detector = FaceDetector(
                    max_side=settings.detect_max_side,
                    use_dual=settings.use_dual_detector,
                )
            faces = detector.detect(frame)
            if len(faces) > best_faces:
                best = frame
                best_faces = len(faces)
            if best_faces > 0:
                break
    finally:
        if detector is not None:
            detector.close()
    if best is not None:
        return best

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {path}")
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("プレビュー用のフレームを読めませんでした。")
        return _normalize_frame(frame, info.rotation, width, height)
    finally:
        cap.release()
