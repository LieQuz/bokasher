# bokasher / ぼかっしゃー

動画ファイルに写っている人の顔を、ローカルで自動検出してボカすデスクトップアプリです。画面は Electron + React、処理は Python です。動画を外部に送信しません。

## インストール（リリースから）

Apple Silicon（M シリーズ）向けの `.dmg` を [Releases](https://github.com/LieQuz/bokasher/releases) からダウンロードします。最新は [v0.1.0](https://github.com/LieQuz/bokasher/releases/tag/v0.1.0) です。

1. `bokasher-0.1.0-mac-arm64.dmg` をダウンロードする
2. コード署名していないので、隔離属性を外す

```bash
xattr -cr ~/Downloads/bokasher-0.1.0-mac-arm64.dmg
```

3. Finder で DMG を開き、`bokasher` を Applications にドラッグする
4. Applications から起動する。「壊れている」「開かない」と出る場合は次を実行する

```bash
xattr -cr /Applications/bokasher.app
```

それでも開かないときは、アプリを右クリック → 開くを選んでください。FFmpeg は同梱しています。

Windows / Intel Mac 向けのインストーラはまだありません。その場合は下の手動ビルドを使ってください。

## 使い方

1. 動画をドラッグ＆ドロップ（またはクリックして選択）する
2. 出力パスを確認する（初期値は `元ファイル名_blurred.mp4`）
3. ブラー強度と「精度優先 / 速度優先」を選ぶ
4. プレビューで結果を確認する
5. 「処理開始」で書き出す

精度優先は 2 フレームごと・検出長辺 960、速度優先は 4 フレームごと・検出長辺 640 です。検出は InsightFace の SCRFD-2.5G_KPS を使います（Apple Silicon では CoreML）。iPhone などの縦撮りは回転メタデータを見て正立してから処理します。書き出しは macOS の VideoToolbox、それ以外は libx264 です。音声がある動画は、元の音声をコピーして出力に残します。

## 手動ビルド

開発中の起動と、macOS インストーラの再作成です。

### 必要なもの

- macOS / Windows / Linux（いまの実機確認は macOS 中心）
- Python 3.10〜3.13（3.12 を推奨）
- Node.js 20+
- [FFmpeg](https://ffmpeg.org/)（macOS なら `brew install ffmpeg`）

### セットアップ

```bash
brew install ffmpeg
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cd desktop && npm install && cd ..
```

初回の検出時に、InsightFace の SCRFD-2.5G_KPS を `~/.cache/bokasher/` へダウンロードします。この重みは上流の利用条件（研究・非商用）に従います。

```bash
pytest
```

### 起動

```bash
source .venv/bin/activate
python -m bokasher.app
```

または:

```bash
cd desktop && npm start
```

Electron が Python のローカル API（`127.0.0.1:8765`）と Vite（`127.0.0.1:5178`）を起動します。

### macOS インストーラを作る

Apple Silicon 向けの `.dmg` を手元で作る場合:

```bash
source .venv/bin/activate
pip install -e ".[dev]"
bash scripts/build_mac.sh
```

成果物は `desktop/release/bokasher-0.1.0-mac-arm64.dmg` です。Finder で開いて Applications にドラッグしてください。配布する場合も、上と同じ `xattr` が必要です。

## 処理の流れ

1. Electron が UI を表示し、Python API にファイルパスを渡す
2. 回転メタデータを見て表示向き（縦 / 横）を決める
3. 可能なら VideoToolbox でデコード
4. SCRFD-2.5G_KPS で顔検出
5. フレーム間の箱の平滑化
6. 楕円マスクつきガウシアンブラー
7. VideoToolbox（なければ libx264）で H.264 書き出し + 元音声の mux

## いまできないこと

- カメラや画面共有のリアルタイム処理
- 「この人だけ残す」といった選択的ブラー
- Windows / Intel Mac 向けインストーラ
