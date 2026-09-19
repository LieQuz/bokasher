from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from bokasher.types import VideoInfo


class FFmpegError(RuntimeError):
    pass


def require_ffmpeg() -> tuple[str, str]:
    bundled = _bundled_ffmpeg()
    if bundled is not None:
        return bundled
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise FFmpegError(
            "FFmpeg が見つかりません。macOS では `brew install ffmpeg` で導入してください。"
        )
    return ffmpeg, ffprobe


def _bundled_ffmpeg() -> tuple[str, str] | None:
    ffmpeg = os.environ.get("BOKASHER_FFMPEG")
    ffprobe = os.environ.get("BOKASHER_FFPROBE")
    if ffmpeg and ffprobe and Path(ffmpeg).exists() and Path(ffprobe).exists():
        return ffmpeg, ffprobe
    return None


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def probe_video(path: Path) -> VideoInfo:
    _ffmpeg, ffprobe = require_ffmpeg()
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    result = _run(cmd)
    if result.returncode != 0:
        raise FFmpegError(result.stderr.strip() or f"ffprobe に失敗しました: {path}")

    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise FFmpegError("動画ストリームが見つかりません。")

    stored_width = int(video.get("width") or 0)
    stored_height = int(video.get("height") or 0)
    if stored_width <= 0 or stored_height <= 0:
        raise FFmpegError("動画の解像度を取得できませんでした。")
    rotation = _rotation_degrees(video)
    width, height = display_size(stored_width, stored_height, rotation)

    rate = video.get("r_frame_rate") or video.get("avg_frame_rate") or "30/1"
    fps = _fps_from_ratio(rate)
    nb_frames = video.get("nb_frames")
    frame_count = int(nb_frames) if nb_frames and str(nb_frames).isdigit() else None

    duration = None
    raw_duration = video.get("duration") or (payload.get("format") or {}).get("duration")
    if raw_duration is not None:
        try:
            duration = float(raw_duration)
        except (TypeError, ValueError):
            duration = None
    if frame_count is None and duration is not None:
        frame_count = max(int(round(duration * fps)), 1)

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        fps_text=str(rate),
        frame_count=frame_count,
        duration=duration,
        has_audio=audio is not None,
        rotation=rotation,
    )


def _rotation_degrees(stream: dict) -> int:
    tags = stream.get("tags") or {}
    for key in ("rotate", "ROTATE"):
        raw = tags.get(key)
        if raw is None:
            continue
        try:
            return int(round(float(raw))) % 360
        except (TypeError, ValueError):
            continue
    for side in stream.get("side_data_list") or []:
        raw = side.get("rotation")
        if raw is None:
            continue
        try:
            return int(round(float(raw))) % 360
        except (TypeError, ValueError):
            continue
    return 0


def display_size(width: int, height: int, rotation: int) -> tuple[int, int]:
    if abs(rotation) % 180 == 90:
        return height, width
    return width, height


def apply_display_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    rot = rotation % 360
    if rot == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def extract_frame(path: Path, time_s: float, width: int, height: int) -> np.ndarray | None:
    ffmpeg, _ = require_ffmpeg()
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(time_s, 0.0):.3f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-vf",
            f"scale={width}:{height}:flags=fast_bilinear",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True)
    expected = width * height * 3
    if result.returncode != 0 or len(result.stdout) < expected:
        return None
    return np.frombuffer(result.stdout[:expected], dtype=np.uint8).reshape(
        (height, width, 3)
    ).copy()


def _fps_from_ratio(rate: str) -> float:
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            return float(num) / float(den)
        except (TypeError, ValueError, ZeroDivisionError):
            return 30.0
    try:
        return float(rate)
    except (TypeError, ValueError):
        return 30.0


@lru_cache(maxsize=1)
def choose_video_encoder(prefer_gpu: bool = True) -> str:
    if prefer_gpu and sys.platform == "darwin":
        ffmpeg, _ = require_ffmpeg()
        result = _run([ffmpeg, "-hide_banner", "-encoders"])
        if result.returncode == 0 and "h264_videotoolbox" in (result.stdout or ""):
            return "h264_videotoolbox"
    return "libx264"


def encoder_display_name(encoder: str) -> str:
    if encoder == "h264_videotoolbox":
        return "VideoToolbox (GPU)"
    return "libx264 (CPU)"


def _encoder_args(encoder: str, width: int, height: int, fps: float) -> list[str]:
    if encoder == "h264_videotoolbox":
        bitrate = int(width * height * max(fps, 24.0) * 0.10)
        bitrate = min(max(bitrate, 2_000_000), 40_000_000)
        return [
            "-c:v",
            "h264_videotoolbox",
            "-b:v",
            str(bitrate),
            "-allow_sw",
            "1",
            "-realtime",
            "0",
            "-prio_speed",
            "1",
            "-pix_fmt",
            "yuv420p",
        ]
    return [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
    ]


class FFmpegReader:
    def __init__(
        self,
        source: Path,
        width: int,
        height: int,
        hwaccel: bool = True,
    ) -> None:
        ffmpeg, _ = require_ffmpeg()
        self.width = width
        self.height = height
        self._nbytes = width * height * 3
        self.backend = "videotoolbox" if hwaccel and sys.platform == "darwin" else "cpu"
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error"]
        if self.backend == "videotoolbox":
            cmd += ["-hwaccel", "videotoolbox"]
        cmd += [
            "-i",
            str(source),
            "-vf",
            f"scale={width}:{height}:flags=fast_bilinear",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def read(self) -> np.ndarray | None:
        if self._proc.stdout is None:
            return None
        raw = self._proc.stdout.read(self._nbytes)
        if not raw or len(raw) < self._nbytes:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()

    def close(self) -> None:
        if self._proc.stdout is not None:
            try:
                self._proc.stdout.close()
            except OSError:
                pass
        if self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()


class FFmpegWriter:
    def __init__(
        self,
        output: Path,
        width: int,
        height: int,
        fps_text: str,
        fps: float = 30.0,
        source: Path | None = None,
        has_audio: bool = False,
        prefer_gpu: bool = True,
    ) -> None:
        ffmpeg, _ = require_ffmpeg()
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.encoder = choose_video_encoder(prefer_gpu)
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-thread_queue_size",
            "8",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            fps_text,
            "-i",
            "pipe:0",
        ]
        if has_audio and source is not None:
            cmd += [
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-c:a",
                "copy",
                "-shortest",
            ]
        else:
            cmd += ["-map", "0:v:0"]
        cmd += _encoder_args(self.encoder, width, height, fps)
        cmd += [
            "-metadata:s:v:0",
            "rotate=0",
            "-movflags",
            "+faststart",
            str(self.output),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=width * height * 3 * 2,
        )

    def write(self, frame: np.ndarray | bytes) -> None:
        if self._proc.stdin is None:
            raise FFmpegError("FFmpeg の入力パイプが閉じています。")
        payload: bytes | memoryview
        if isinstance(frame, np.ndarray):
            payload = memoryview(np.ascontiguousarray(frame))
        else:
            payload = frame
        try:
            self._proc.stdin.write(payload)
        except BrokenPipeError as exc:
            raise FFmpegError("FFmpeg への書き込みに失敗しました。") from exc

    def close(self) -> None:
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
        returncode = self._proc.wait()
        if returncode != 0:
            raise FFmpegError(f"FFmpeg が終了コード {returncode} で失敗しました。")

    def abort(self) -> None:
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
        if self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()
        if self.output.exists():
            self.output.unlink()
