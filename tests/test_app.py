from __future__ import annotations

import pytest


def test_app_starts() -> None:
    try:
        from bokasher.app import BokasherApp
    except ModuleNotFoundError as exc:
        pytest.skip(f"GUI 依存がありません: {exc}")

    app = BokasherApp()
    try:
        app.update_idletasks()
        assert app.title() == "Bokasher"
        assert app.start_button.cget("text") == "処理開始"
        assert app.mode.get() == "速度優先"
    finally:
        app.destroy()
