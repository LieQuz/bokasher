from pathlib import Path

from bokasher.ffmpeg import (
    _bundled_ffmpeg,
    choose_video_encoder,
    display_size,
    encoder_display_name,
    require_ffmpeg,
)


def test_choose_video_encoder_prefers_videotoolbox_on_mac() -> None:
    encoder = choose_video_encoder(True)
    assert encoder in {"h264_videotoolbox", "libx264"}
    if encoder == "h264_videotoolbox":
        assert "GPU" in encoder_display_name(encoder)


def test_cpu_fallback_encoder() -> None:
    assert choose_video_encoder(False) == "libx264"


def test_display_size_swaps_for_portrait_rotation() -> None:
    assert display_size(1920, 1080, 90) == (1080, 1920)
    assert display_size(1920, 1080, 270) == (1080, 1920)
    assert display_size(1920, 1080, 0) == (1920, 1080)
    assert display_size(1080, 1920, 0) == (1080, 1920)


def test_bundled_ffmpeg_uses_env(tmp_path: Path, monkeypatch) -> None:
    ffmpeg = tmp_path / "ffmpeg"
    ffprobe = tmp_path / "ffprobe"
    ffmpeg.write_text("")
    ffprobe.write_text("")
    monkeypatch.setenv("BOKASHER_FFMPEG", str(ffmpeg))
    monkeypatch.setenv("BOKASHER_FFPROBE", str(ffprobe))
    assert _bundled_ffmpeg() == (str(ffmpeg), str(ffprobe))
    assert require_ffmpeg() == (str(ffmpeg), str(ffprobe))
    monkeypatch.delenv("BOKASHER_FFMPEG")
    assert _bundled_ffmpeg() is None
