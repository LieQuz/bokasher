from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import pytest

import cv2
import numpy as np

from bokasher.blur import apply_face_blur
from bokasher.detector import FaceDetector
from bokasher.ffmpeg import probe_video
from bokasher.pipeline import (
    ProcessingCancelled,
    default_output_path,
    fit_even_size,
    process_video,
    render_preview,
)
from bokasher.types import ProcessSettings


def _make_video(path: Path, seconds: float = 0.4, size: str = "160x120") -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:s={size}:d={seconds}:r=10",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )


def test_default_output_path() -> None:
    assert default_output_path(Path("/tmp/demo.mov")) == Path("/tmp/demo_blurred.mp4")


def test_fit_even_size_downscales_4k_portrait() -> None:
    assert fit_even_size(2160, 3840, 1920) == (1080, 1920)
    assert fit_even_size(180, 320, 1920) == (180, 320)
    assert fit_even_size(640, 480, 0) == (640, 480)


def test_process_video_keeps_audio(tmp_path: Path) -> None:
    source = tmp_path / "plain.mp4"
    dest = tmp_path / "plain_blurred.mp4"
    _make_video(source)
    process_video(source, dest, ProcessSettings(blur_strength=0.5, detect_every=2))
    assert dest.exists()
    info = probe_video(dest)
    assert info.has_audio
    assert info.width == 160
    assert info.height == 120


def test_cancel_removes_incomplete_output(tmp_path: Path) -> None:
    source = tmp_path / "long.mp4"
    dest = tmp_path / "long_blurred.mp4"
    _make_video(source, seconds=2.0)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(ProcessingCancelled):
        process_video(
            source,
            dest,
            ProcessSettings(),
            cancel_event=cancel,
        )
    time.sleep(0.05)
    assert not dest.exists()


def test_output_max_side_downscales(tmp_path: Path) -> None:
    source = tmp_path / "large.mp4"
    dest = tmp_path / "large_blurred.mp4"
    _make_video(source, seconds=0.4, size="640x480")
    process_video(
        source,
        dest,
        ProcessSettings(blur_strength=0.5, detect_every=2, output_max_side=320),
    )
    info = probe_video(dest)
    assert info.width == 320
    assert info.height == 240


def test_portrait_output_keeps_vertical_size(tmp_path: Path) -> None:
    source = tmp_path / "portrait.mp4"
    dest = tmp_path / "portrait_blurred.mp4"
    _make_video(source, seconds=0.4, size="180x320")
    process_video(source, dest, ProcessSettings(blur_strength=0.5, detect_every=2))
    info = probe_video(dest)
    assert info.width == 180
    assert info.height == 320
    assert info.height > info.width


def test_rotated_landscape_becomes_portrait(tmp_path: Path) -> None:
    source = tmp_path / "rotated.mp4"
    dest = tmp_path / "rotated_blurred.mp4"
    raw = tmp_path / "rotated_raw.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=320x180:d=0.4:r=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(raw),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-display_rotation",
            "90",
            "-i",
            str(raw),
            "-c",
            "copy",
            str(source),
        ],
        check=True,
    )
    probed = probe_video(source)
    assert probed.rotation % 180 == 90
    assert probed.width == 180
    assert probed.height == 320
    process_video(source, dest, ProcessSettings(prefer_gpu=False))
    info = probe_video(dest)
    assert info.width == 180
    assert info.height == 320


def _make_still_video(image_path: Path, dest: Path, seconds: float = 0.4) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-i",
            str(image_path),
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-t",
            str(seconds),
            "-r",
            "10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(dest),
        ],
        check=True,
    )


def test_preview_matches_process_settings(tmp_path: Path, portrait_bgr: np.ndarray) -> None:
    image_path = tmp_path / "portrait.jpg"
    cv2.imwrite(str(image_path), portrait_bgr)
    source = tmp_path / "face.mp4"
    dest = tmp_path / "face_blurred.mp4"
    _make_still_video(image_path, source)
    settings = ProcessSettings(blur_strength=0.85, detect_every=1)

    preview = render_preview(source, settings)
    cap = cv2.VideoCapture(str(source))
    ok, frame = cap.read()
    cap.release()
    assert ok
    frame = frame[: frame.shape[0] - frame.shape[0] % 2, : frame.shape[1] - frame.shape[1] % 2]
    with FaceDetector() as detector:
        boxes = detector.detect(frame)
    assert boxes
    expected = apply_face_blur(frame, boxes, settings.blur_strength)
    assert preview.shape == expected.shape
    assert np.mean(np.abs(preview.astype(np.int16) - expected.astype(np.int16))) < 2

    process_video(source, dest, settings)
    info = probe_video(dest)
    assert info.has_audio
    out_cap = cv2.VideoCapture(str(dest))
    ok, out_frame = out_cap.read()
    out_cap.release()
    assert ok
    box = boxes[0]
    cx, cy = box.x + box.w // 2, box.y + box.h // 2
    assert not np.array_equal(
        frame[cy - 4 : cy + 4, cx - 4 : cx + 4],
        out_frame[cy - 4 : cy + 4, cx - 4 : cx + 4],
    )
