from bokasher.ffmpeg import choose_video_encoder, display_size, encoder_display_name


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
