#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DESKTOP="$ROOT/desktop"
VENV_PY="$ROOT/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "先に .venv を作り、pip install -e \".[dev]\" してください。" >&2
  exit 1
fi

cd "$DESKTOP"
if [[ ! -d node_modules ]]; then
  npm install
fi
npm install
npm run build

"$VENV_PY" -m pip install -q "pyinstaller>=6.10"
rm -rf "$DESKTOP/resources/bokasher-server" "$DESKTOP/build/pyinstaller"
"$VENV_PY" -m PyInstaller \
  --noconfirm \
  --clean \
  --distpath "$DESKTOP/resources" \
  --workpath "$DESKTOP/build/pyinstaller" \
  "$ROOT/packaging/bokasher-server.spec"

chmod +x "$DESKTOP/resources/bokasher-server/bokasher-server"

mkdir -p "$DESKTOP/resources/ffmpeg"
FFMPEG_BIN="$(node -p "require('ffmpeg-static')")"
FFPROBE_BIN="$(node -p "require('ffprobe-static').path")"
cp "$FFMPEG_BIN" "$DESKTOP/resources/ffmpeg/ffmpeg"
cp "$FFPROBE_BIN" "$DESKTOP/resources/ffmpeg/ffprobe"
chmod +x "$DESKTOP/resources/ffmpeg/ffmpeg" "$DESKTOP/resources/ffmpeg/ffprobe"

"$VENV_PY" "$ROOT/scripts/generate_dmg_assets.py"

npm run dist:mac

echo
echo "作成しました:"
ls -lh "$DESKTOP/release/"*.dmg
