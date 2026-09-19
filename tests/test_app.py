from __future__ import annotations

from bokasher.app import APP_DISPLAY_NAME, desktop_dir
from bokasher.server import settings_from_payload
from bokasher.server import SettingsPayload


def test_display_name() -> None:
    assert APP_DISPLAY_NAME == "bokasher / ぼかっしゃー"


def test_desktop_app_exists() -> None:
    desktop = desktop_dir()
    assert (desktop / "package.json").exists()
    assert (desktop / "src" / "App.jsx").exists()


def test_settings_from_payload() -> None:
    fast = settings_from_payload(SettingsPayload(blur_strength=70, mode="速度優先"))
    assert fast.detect_every == 4
    assert fast.output_max_side == 1920
    accurate = settings_from_payload(SettingsPayload(blur_strength=80, mode="精度優先"))
    assert accurate.detect_every == 2
    assert accurate.output_max_side == 2560
    assert abs(accurate.blur_strength - 0.8) < 1e-6
