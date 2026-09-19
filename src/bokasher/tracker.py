from __future__ import annotations

from dataclasses import dataclass

from bokasher.types import FaceBox

# SORT / Roboflow Detections Stabilizer に合わせた定数。
# 位置は低めの EMA（ジッタ抑制）、サイズは速く広げて遅く縮める。
_POS_ALPHA = 0.22
_SIZE_EXPAND = 0.45
_SIZE_SHRINK = 0.08
_VEL_ALPHA = 0.25
_VEL_DAMP = 0.92
_ENV_DECAY = 0.985


def iou(a: FaceBox, b: FaceBox) -> float:
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = inter_w * inter_h
    union = a.w * a.h + b.w * b.h - inter
    if union <= 0:
        return 0.0
    return inter / union


def merge_boxes(boxes: list[FaceBox], iou_threshold: float = 0.4) -> list[FaceBox]:
    remaining = sorted(boxes, key=lambda box: box.score, reverse=True)
    kept: list[FaceBox] = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [box for box in remaining if iou(current, box) < iou_threshold]
    return kept


def _center(box: FaceBox) -> tuple[float, float]:
    return box.x + box.w / 2.0, box.y + box.h / 2.0


@dataclass
class _Track:
    id: int
    cx: float
    cy: float
    w: float
    h: float
    vx: float = 0.0
    vy: float = 0.0
    env_w: float = 0.0
    env_h: float = 0.0
    missed: int = 0
    hits: int = 1
    score: float = 1.0

    def predict(self) -> None:
        self.cx += self.vx
        self.cy += self.vy
        self.vx *= _VEL_DAMP
        self.vy *= _VEL_DAMP
        self.env_w *= _ENV_DECAY
        self.env_h *= _ENV_DECAY

    def correct(self, box: FaceBox) -> None:
        ncx, ncy = _center(box)
        dx = ncx - self.cx
        dy = ncy - self.cy
        self.vx = (1.0 - _VEL_ALPHA) * self.vx + _VEL_ALPHA * dx
        self.vy = (1.0 - _VEL_ALPHA) * self.vy + _VEL_ALPHA * dy
        self.cx = (1.0 - _POS_ALPHA) * self.cx + _POS_ALPHA * ncx
        self.cy = (1.0 - _POS_ALPHA) * self.cy + _POS_ALPHA * ncy
        if box.w >= self.w:
            self.w = (1.0 - _SIZE_EXPAND) * self.w + _SIZE_EXPAND * box.w
        else:
            self.w = (1.0 - _SIZE_SHRINK) * self.w + _SIZE_SHRINK * box.w
        if box.h >= self.h:
            self.h = (1.0 - _SIZE_EXPAND) * self.h + _SIZE_EXPAND * box.h
        else:
            self.h = (1.0 - _SIZE_SHRINK) * self.h + _SIZE_SHRINK * box.h
        self.env_w = max(self.w, self.env_w)
        self.env_h = max(self.h, self.env_h)
        self.score = box.score
        self.missed = 0
        self.hits += 1

    def as_box(self, width: int | None, height: int | None) -> FaceBox:
        out_w = max(self.w, self.env_w)
        out_h = max(self.h, self.env_h)
        x = int(round(self.cx - out_w / 2.0))
        y = int(round(self.cy - out_h / 2.0))
        box = FaceBox(
            x=x,
            y=y,
            w=max(int(round(out_w)), 1),
            h=max(int(round(out_h)), 1),
            score=self.score,
        )
        if width is not None and height is not None:
            return box.clamp(width, height)
        return box


def _from_box(track_id: int, box: FaceBox) -> _Track:
    cx, cy = _center(box)
    return _Track(
        id=track_id,
        cx=cx,
        cy=cy,
        w=float(box.w),
        h=float(box.h),
        env_w=float(box.w),
        env_h=float(box.h),
        score=box.score,
    )


class FaceTracker:
    def __init__(
        self,
        iou_threshold: float = 0.2,
        alpha: float = 0.22,
        max_missed: int = 18,
        hold_grow: float = 0.0,
        max_tracks: int = 12,
    ) -> None:
        del alpha, hold_grow
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.max_tracks = max_tracks
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(
        self,
        detections: list[FaceBox] | None,
        width: int | None = None,
        height: int | None = None,
    ) -> list[FaceBox]:
        # None = 検出スキップ（寿命は減らさない）。[] = 検出したが顔なし。
        skipped = detections is None
        detections = [] if detections is None else detections

        for track in self._tracks:
            track.predict()

        if not skipped:
            used: set[int] = set()
            for track in self._tracks:
                best_idx = -1
                best_iou = self.iou_threshold
                predicted = track.as_box(width, height)
                for idx, det in enumerate(detections):
                    if idx in used:
                        continue
                    score = iou(predicted, det)
                    if score > best_iou:
                        best_iou = score
                        best_idx = idx
                if best_idx >= 0:
                    used.add(best_idx)
                    track.correct(detections[best_idx])
                else:
                    track.missed += 1

            self._tracks = [
                track for track in self._tracks if track.missed <= self.max_missed
            ]

            for idx, det in enumerate(detections):
                if idx in used:
                    continue
                if any(
                    iou(track.as_box(width, height), det) >= self.iou_threshold
                    for track in self._tracks
                ):
                    continue
                self._tracks.append(_from_box(self._next_id, det))
                self._next_id += 1

            if len(self._tracks) > self.max_tracks:
                self._tracks.sort(
                    key=lambda track: (track.missed, -track.hits, -track.score)
                )
                self._tracks = self._tracks[: self.max_tracks]

        return [track.as_box(width, height) for track in self._tracks]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
