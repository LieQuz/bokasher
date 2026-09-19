from __future__ import annotations

import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import customtkinter as ctk
import cv2
from PIL import Image
from tkinter import filedialog, messagebox

from bokasher.ffmpeg import (
    FFmpegError,
    choose_video_encoder,
    encoder_display_name,
    require_ffmpeg,
)
from bokasher.pipeline import (
    ProcessingCancelled,
    default_output_path,
    process_video,
    render_preview,
)
from bokasher.types import ProcessSettings

PREVIEW_SIZE = (640, 360)


class BokasherApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Bokasher")
        self.geometry("920x780")
        self.minsize(780, 680)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.input_path: Path | None = None
        self.output_path: Path | None = None
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self._preview_image: ctk.CTkImage | None = None
        self._busy = False
        self._last_progress_ui = 0.0

        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(200, self._check_ffmpeg)

    def _build(self) -> None:
        root = ctk.CTkFrame(self, fg_color="transparent")
        root.pack(fill="both", expand=True, padx=20, pady=20)
        root.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(root, text="Bokasher", font=ctk.CTkFont(size=26, weight="bold"))
        title.grid(row=0, column=0, columnspan=3, sticky="w")
        subtitle = ctk.CTkLabel(
            root,
            text="動画の顔を、ローカルで自動的にボカします。",
            text_color=("gray30", "gray70"),
        )
        subtitle.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 16))

        ctk.CTkLabel(root, text="入力").grid(row=2, column=0, sticky="w", pady=6)
        self.input_var = ctk.StringVar(value="動画ファイルを選択してください")
        ctk.CTkEntry(root, textvariable=self.input_var, state="readonly").grid(
            row=2, column=1, sticky="ew", padx=8, pady=6
        )
        ctk.CTkButton(root, text="選択", width=90, command=self._choose_input).grid(
            row=2, column=2, pady=6
        )

        ctk.CTkLabel(root, text="出力").grid(row=3, column=0, sticky="w", pady=6)
        self.output_var = ctk.StringVar(value="")
        ctk.CTkEntry(root, textvariable=self.output_var).grid(
            row=3, column=1, sticky="ew", padx=8, pady=6
        )
        ctk.CTkButton(root, text="変更", width=90, command=self._choose_output).grid(
            row=3, column=2, pady=6
        )

        controls = ctk.CTkFrame(root, fg_color="transparent")
        controls.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(12, 8))
        controls.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(controls, text="ブラー強度").grid(row=0, column=0, sticky="w")
        self.strength = ctk.DoubleVar(value=70)
        self.strength_label = ctk.CTkLabel(controls, text="70")
        self.strength_label.grid(row=0, column=2, padx=(8, 0))
        slider = ctk.CTkSlider(
            controls,
            from_=10,
            to=100,
            variable=self.strength,
            command=self._on_strength,
        )
        slider.grid(row=0, column=1, sticky="ew", padx=12)

        ctk.CTkLabel(controls, text="検出").grid(row=1, column=0, sticky="w", pady=(12, 0))
        self.mode = ctk.StringVar(value="速度優先")
        ctk.CTkSegmentedButton(
            controls,
            values=["精度優先", "速度優先"],
            variable=self.mode,
        ).grid(row=1, column=1, sticky="w", padx=12, pady=(12, 0))

        self.preview_label = ctk.CTkLabel(
            root,
            text="プレビューはここに表示されます",
            width=PREVIEW_SIZE[0],
            height=PREVIEW_SIZE[1],
            fg_color=("gray85", "gray17"),
            corner_radius=8,
        )
        self.preview_label.grid(row=5, column=0, columnspan=3, pady=16)

        buttons = ctk.CTkFrame(root, fg_color="transparent")
        buttons.grid(row=6, column=0, columnspan=3, sticky="ew")
        self.start_button = ctk.CTkButton(
            buttons, text="処理開始", command=self._start, height=36
        )
        self.start_button.pack(side="left")
        self.cancel_button = ctk.CTkButton(
            buttons,
            text="キャンセル",
            command=self._cancel,
            height=36,
            fg_color="gray30",
            state="disabled",
        )
        self.cancel_button.pack(side="left", padx=8)
        self.preview_button = ctk.CTkButton(
            buttons, text="プレビュー更新", command=self._refresh_preview, height=36
        )
        self.preview_button.pack(side="left")

        self.progress = ctk.CTkProgressBar(root)
        self.progress.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(16, 6))
        self.progress.set(0)
        self.status = ctk.CTkLabel(root, text="待機中", anchor="w")
        self.status.grid(row=8, column=0, columnspan=3, sticky="ew")
        self.backend = ctk.CTkLabel(
            root, text="", anchor="w", text_color=("gray30", "gray70")
        )
        self.backend.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(4, 0))

    def _settings(self) -> ProcessSettings:
        accurate = self.mode.get() == "精度優先"
        return ProcessSettings(
            blur_strength=self.strength.get() / 100.0,
            detect_every=2 if accurate else 4,
            detect_max_side=960 if accurate else 640,
            prefer_gpu=True,
            use_dual_detector=accurate,
            output_max_side=2560 if accurate else 1920,
        )

    def _on_strength(self, value: float) -> None:
        self.strength_label.configure(text=str(int(round(float(value)))))

    def _check_ffmpeg(self) -> None:
        try:
            require_ffmpeg()
            encoder = choose_video_encoder(True)
            self.backend.configure(
                text=f"書き出し: {encoder_display_name(encoder)}  /  検出: YuNet（見逃し時のみ MediaPipe）"
            )
        except FFmpegError as exc:
            messagebox.showwarning("FFmpeg が必要です", str(exc))
            self.status.configure(text=str(exc))

    def _choose_input(self) -> None:
        if self._busy:
            return
        path = filedialog.askopenfilename(
            title="動画を選択",
            filetypes=[
                ("動画", "*.mp4 *.mov *.m4v *.mkv *.avi *.webm"),
                ("すべて", "*.*"),
            ],
        )
        if not path:
            return
        self.input_path = Path(path)
        self.input_var.set(str(self.input_path))
        self.output_path = default_output_path(self.input_path)
        self.output_var.set(str(self.output_path))
        self._refresh_preview()

    def _choose_output(self) -> None:
        if self._busy:
            return
        initial = self.output_var.get() or str(Path.home() / "untitled_blurred.mp4")
        path = filedialog.asksaveasfilename(
            title="出力先",
            defaultextension=".mp4",
            initialfile=Path(initial).name,
            initialdir=str(Path(initial).parent),
            filetypes=[("MP4", "*.mp4")],
        )
        if not path:
            return
        self.output_path = Path(path)
        self.output_var.set(str(self.output_path))

    def _refresh_preview(self) -> None:
        if self.input_path is None:
            return
        if self._busy:
            return
        self.status.configure(text="プレビューを準備しています…")
        path = self.input_path
        settings = self._settings()

        def work() -> None:
            try:
                frame = render_preview(path, settings)
                self.after(0, lambda: self._show_preview(frame, "プレビューを更新しました"))
            except Exception as exc:
                self.after(0, lambda e=exc: self._show_error(e, "プレビューに失敗しました"))

        threading.Thread(target=work, daemon=True).start()

    def _show_preview(self, bgr, status: str) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        self._preview_image = ctk.CTkImage(
            light_image=image, dark_image=image, size=image.size
        )
        self.preview_label.configure(image=self._preview_image, text="")
        self.status.configure(text=status)

    def _start(self) -> None:
        if self._busy:
            return
        if self.input_path is None:
            messagebox.showinfo("入力がありません", "先に動画ファイルを選択してください。")
            return
        output_text = self.output_var.get().strip()
        if not output_text:
            messagebox.showinfo("出力先がありません", "出力ファイルのパスを指定してください。")
            return
        self.output_path = Path(output_text)
        if self.output_path.resolve() == self.input_path.resolve():
            messagebox.showerror("出力先が不正です", "入力と同じファイルには書き出せません。")
            return

        self._busy = True
        self.cancel_event.clear()
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.preview_button.configure(state="disabled")
        self.progress.set(0)
        self._last_progress_ui = 0.0
        encoder = choose_video_encoder(True)
        self.status.configure(
            text=f"{encoder_display_name(encoder)} で処理を開始しています…"
        )

        settings = self._settings()
        source = self.input_path
        dest = self.output_path

        def work() -> None:
            try:
                process_video(
                    source,
                    dest,
                    settings,
                    on_progress=self._on_progress,
                    cancel_event=self.cancel_event,
                )
                self.after(0, lambda: self._on_finished(dest))
            except ProcessingCancelled:
                self.after(0, self._on_cancelled)
            except Exception as exc:
                self.after(0, lambda e=exc: self._show_error(e, "処理に失敗しました"))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _on_progress(
        self, current: int, total: int | None, elapsed: float, encoder: str = ""
    ) -> None:
        now = time.monotonic()
        is_last = total is not None and current >= total
        if not is_last and now - self._last_progress_ui < 0.12:
            return
        self._last_progress_ui = now

        def update() -> None:
            speed = f"{current / elapsed:.1f} fps" if elapsed > 0 else ""
            label = encoder_display_name(encoder) if encoder else ""
            extra = "  ".join(part for part in (f"{elapsed:.1f} 秒", speed, label) if part)
            if total and total > 0:
                self.progress.set(min(current / total, 1.0))
                self.status.configure(text=f"{current} / {total} フレーム  ({extra})")
            else:
                self.progress.set(0)
                self.status.configure(text=f"{current} フレーム処理済み  ({extra})")

        self.after(0, update)

    def _cancel(self) -> None:
        if not self._busy:
            return
        self.cancel_event.set()
        self.status.configure(text="キャンセルしています…")

    def _on_finished(self, dest: Path) -> None:
        self._set_idle()
        self.progress.set(1)
        self.status.configure(text=f"完了: {dest}")
        if dest.exists():
            if messagebox.askyesno("完了", f"書き出しました。\n{dest}\n\nフォルダを開きますか？"):
                _reveal(dest)

    def _on_cancelled(self) -> None:
        self._set_idle()
        self.progress.set(0)
        self.status.configure(text="キャンセルしました。未完成の出力は削除しています。")

    def _show_error(self, exc: Exception, title: str) -> None:
        self._set_idle()
        detail = str(exc) or exc.__class__.__name__
        self.status.configure(text=f"{title}: {detail}")
        messagebox.showerror(title, detail)

    def _set_idle(self) -> None:
        self._busy = False
        self.start_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.preview_button.configure(state="normal")

    def _on_close(self) -> None:
        if self._busy:
            self.cancel_event.set()
        self.destroy()


def _reveal(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(path)], check=False)
    elif sys.platform == "win32":
        subprocess.run(["explorer", "/select,", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path.parent)], check=False)


def main() -> None:
    try:
        app = BokasherApp()
        app.mainloop()
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
