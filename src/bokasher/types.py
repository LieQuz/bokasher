from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def clamp(self, width: int, height: int) -> FaceBox:
        x = min(max(self.x, 0), max(width - 1, 0))
        y = min(max(self.y, 0), max(height - 1, 0))
        w = min(max(self.w, 1), max(width - x, 1))
        h = min(max(self.h, 1), max(height - y, 1))
        return FaceBox(x=x, y=y, w=w, h=h, score=self.score)

    def expanded(self, ratio: float, width: int, height: int) -> FaceBox:
        pad_x = int(self.w * ratio)
        pad_top = int(self.h * (ratio + 0.18))
        pad_bottom = int(self.h * (ratio + 0.08))
        return FaceBox(
            x=self.x - pad_x,
            y=self.y - pad_top,
            w=self.w + pad_x * 2,
            h=self.h + pad_top + pad_bottom,
            score=self.score,
        ).clamp(width, height)


@dataclass(frozen=True)
class ProcessSettings:
    blur_strength: float = 0.7
    detect_every: int = 1
    detect_max_side: int = 640
    prefer_gpu: bool = True
    use_dual_detector: bool = True
    output_max_side: int = 1920

    def __post_init__(self) -> None:
        strength = min(max(self.blur_strength, 0.0), 1.0)
        every = max(int(self.detect_every), 1)
        max_side = max(int(self.detect_max_side), 64)
        # 0 は元解像度のまま。4K はデコード時点で縮小してメモリとエンコードを軽くする
        max_out = max(int(self.output_max_side), 0)
        object.__setattr__(self, "blur_strength", strength)
        object.__setattr__(self, "detect_every", every)
        object.__setattr__(self, "detect_max_side", max_side)
        object.__setattr__(self, "output_max_side", max_out)


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    fps_text: str
    frame_count: int | None
    duration: float | None
    has_audio: bool
    rotation: int = 0
