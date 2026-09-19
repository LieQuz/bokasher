from __future__ import annotations

import argparse
import json
import queue
import threading
from collections.abc import Iterator
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from bokasher.detector import detector_display_name
from bokasher.ffmpeg import (
    FFmpegError,
    choose_video_encoder,
    encoder_display_name,
    probe_video,
    require_ffmpeg,
)
from bokasher.pipeline import ProcessingCancelled, process_video, render_preview
from bokasher.types import ProcessSettings

APP_DISPLAY_NAME = "bokasher / ぼかっしゃー"


class SettingsPayload(BaseModel):
    blur_strength: float = Field(default=70, ge=10, le=100)
    mode: str = "速度優先"


class PathPayload(SettingsPayload):
    path: str


class ProcessPayload(SettingsPayload):
    input_path: str
    output_path: str


def settings_from_payload(payload: SettingsPayload) -> ProcessSettings:
    accurate = payload.mode == "精度優先"
    return ProcessSettings(
        blur_strength=payload.blur_strength / 100.0,
        detect_every=2 if accurate else 4,
        detect_max_side=960 if accurate else 640,
        prefer_gpu=True,
        output_max_side=2560 if accurate else 1920,
    )


class JobHub:
    def __init__(self) -> None:
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.busy = False

    def begin(self) -> None:
        with self.lock:
            if self.busy:
                raise HTTPException(status_code=409, detail="別の処理が実行中です。")
            self.busy = True
            self.cancel.clear()

    def end(self) -> None:
        with self.lock:
            self.busy = False


hub = JobHub()
app = FastAPI(title=APP_DISPLAY_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"ok": True, "name": APP_DISPLAY_NAME}


@app.get("/backends")
def backends() -> dict[str, str]:
    try:
        require_ffmpeg()
        encoder = choose_video_encoder(True)
        encode = encoder_display_name(encoder)
    except FFmpegError as exc:
        encode = str(exc)
    return {
        "encode": encode,
        "detect": detector_display_name(),
    }


@app.post("/probe")
def probe(payload: PathPayload) -> dict[str, object]:
    path = Path(payload.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="ファイルが見つかりません。")
    try:
        info = probe_video(path)
    except FFmpegError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "name": path.name,
        "path": str(path),
        "width": info.width,
        "height": info.height,
        "fps": info.fps,
        "duration": info.duration,
        "has_audio": info.has_audio,
        "frame_count": info.frame_count,
        "rotation": info.rotation,
    }


@app.post("/preview")
def preview(payload: PathPayload) -> Response:
    path = Path(payload.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="ファイルが見つかりません。")
    try:
        frame = render_preview(path, settings_from_payload(payload))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc) or exc.__class__.__name__) from exc
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise HTTPException(status_code=500, detail="プレビュー画像を作れませんでした。")
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


@app.post("/process")
def process(payload: ProcessPayload) -> StreamingResponse:
    source = Path(payload.input_path)
    dest = Path(payload.output_path)
    if not source.exists():
        raise HTTPException(status_code=404, detail="入力ファイルが見つかりません。")
    if dest.resolve() == source.resolve():
        raise HTTPException(status_code=400, detail="入力と同じファイルには書き出せません。")
    settings = settings_from_payload(payload)
    hub.begin()
    updates: queue.Queue[dict | None] = queue.Queue()

    def worker() -> None:
        try:
            def on_progress(
                current: int, total: int | None, elapsed: float, encoder: str = ""
            ) -> None:
                updates.put(
                    {
                        "current": current,
                        "total": total,
                        "elapsed": elapsed,
                        "encoder": encoder,
                    }
                )

            process_video(
                source,
                dest,
                settings,
                on_progress=on_progress,
                cancel_event=hub.cancel,
            )
            updates.put({"done": True, "output": str(dest)})
        except ProcessingCancelled:
            updates.put({"cancelled": True})
        except Exception as exc:
            updates.put({"error": str(exc) or exc.__class__.__name__})
        finally:
            updates.put(None)
            hub.end()

    threading.Thread(target=worker, daemon=True).start()

    def events() -> Iterator[bytes]:
        while True:
            item = updates.get()
            if item is None:
                break
            yield _sse(item)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/cancel")
def cancel() -> dict[str, bool]:
    hub.cancel.set()
    return {"ok": True}


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=APP_DISPLAY_NAME)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
