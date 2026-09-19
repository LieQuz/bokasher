from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

APP_DISPLAY_NAME = "bokasher / ぼかっしゃー"


def desktop_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "desktop"


def main() -> None:
    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit("Node.js / npm が見つかりません。Electron 版の起動に必要です。")
    desktop = desktop_dir()
    if not (desktop / "package.json").exists():
        raise SystemExit(f"desktop アプリが見つかりません: {desktop}")
    if not (desktop / "node_modules").exists():
        subprocess.run([npm, "install"], cwd=desktop, check=True)
    raise SystemExit(subprocess.call([npm, "start"], cwd=desktop))


if __name__ == "__main__":
    main()
