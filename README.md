# Bokasher

動画ファイルに写っている人の顔を、ローカルで自動検出してボカす macOS 向けデスクトップアプリです。処理はすべて手元のマシンで行い、動画を外部に送信しません。

## 必要なもの

- macOS
- Python 3.10〜3.13（3.12 を推奨。MediaPipe が 3.14 以降に未対応なことがあります。1.x は macOS で初期化に失敗することがあるため 0.10 系を使います）
- Tk 対応（Homebrew の Python なら `brew install python-tk@3.12`）
- [FFmpeg](https://ffmpeg.org/)（`brew install ffmpeg`）

## セットアップ

```bash
brew install ffmpeg python-tk@3.12
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

初回の検出時に、顔検出モデルを `~/.cache/bokasher/` へダウンロードします。

開発用テストを走らせる場合:

```bash
pip install -e ".[dev]"
pytest
```

## 起動

```bash
source .venv/bin/activate
python -m bokasher.app
```

または:

```bash
bokasher
```

## 使い方

1. 入力の動画ファイルを選ぶ
2. 出力パスを確認する（初期値は `元ファイル名_blurred.mp4`）
3. ブラー強度と「精度優先 / 速度優先」を選ぶ
4. プレビューで結果を確認する
5. 「処理開始」で書き出す

精度優先は毎フレーム、YuNet と MediaPipe の両方で検出します。速度優先は YuNet のみ、3 フレームごとです。iPhone などの縦撮りは回転メタデータを見て正立してから処理します。書き出しは macOS の VideoToolbox（Apple Silicon / AMD GPU）を使います。音声がある動画は、元の音声をコピーして出力に残します。

## 処理の流れ

1. 回転メタデータを見て表示向き（縦 / 横）を決める
2. 可能なら VideoToolbox でデコード
3. YuNet（必要なら MediaPipe も）で顔検出
4. フレーム間の箱の平滑化
5. 楕円マスクつきガウシアンブラー
6. VideoToolbox（なければ libx264）で H.264 書き出し + 元音声の mux

## いまできないこと

- カメラや画面共有のリアルタイム処理
- 「この人だけ残す」といった選択的ブラー
- インストーラや `.app` 配布
