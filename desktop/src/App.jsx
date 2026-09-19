import { useEffect, useMemo, useRef, useState } from "react";

const DEFAULT_MODE = "速度優先";

function defaultOutput(inputPath) {
  const slash = Math.max(inputPath.lastIndexOf("/"), inputPath.lastIndexOf("\\"));
  const dir = slash >= 0 ? inputPath.slice(0, slash + 1) : "";
  const name = slash >= 0 ? inputPath.slice(slash + 1) : inputPath;
  const dot = name.lastIndexOf(".");
  const stem = dot > 0 ? name.slice(0, dot) : name;
  return `${dir}${stem}_blurred.mp4`;
}

function nativePath(file) {
  if (!file) return "";
  if (window.bokasher?.pathForFile) {
    try {
      return window.bokasher.pathForFile(file) || "";
    } catch {
      return "";
    }
  }
  return file.path || "";
}

export default function App() {
  const api = window.bokasher;
  const [apiBase, setApiBase] = useState("http://127.0.0.1:8765");
  const [inputPath, setInputPath] = useState("");
  const [outputPath, setOutputPath] = useState("");
  const [meta, setMeta] = useState("");
  const [strength, setStrength] = useState(70);
  const [mode, setMode] = useState(DEFAULT_MODE);
  const [previewUrl, setPreviewUrl] = useState("");
  const [status, setStatus] = useState("待機中");
  const [backend, setBackend] = useState("");
  const [busy, setBusy] = useState(false);
  const [percent, setPercent] = useState("");
  const [progress, setProgress] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const previewRef = useRef("");

  const settings = useMemo(
    () => ({ blur_strength: strength, mode }),
    [strength, mode]
  );

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const base = api?.apiBase ? await api.apiBase() : apiBase;
      if (!cancelled) setApiBase(base);
      try {
        const res = await fetch(`${base}/backends`);
        const data = await res.json();
        if (!cancelled) {
          setBackend(`書き出し: ${data.encode}  /  検出: ${data.detect}`);
        }
      } catch {
        if (!cancelled) setBackend("バックエンドに接続できません");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [api]);

  useEffect(() => {
    return () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    };
  }, []);

  async function applyInput(filePath) {
    if (!filePath) return;
    setInputPath(filePath);
    setOutputPath(defaultOutput(filePath));
    setError("");
    setStatus("読み込み中…");
    try {
      const probe = await fetch(`${apiBase}/probe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: filePath, ...settings }),
      });
      const data = await probe.json();
      if (!probe.ok) throw new Error(data.detail || "probe failed");
      const parts = [`${data.width}×${data.height}`, `${Number(data.fps).toFixed(2)} fps`];
      if (data.duration) parts.push(`${Number(data.duration).toFixed(1)} 秒`);
      parts.push(data.has_audio ? "音声あり" : "音声なし");
      setMeta(parts.join("・"));
      await refreshPreview(filePath);
    } catch (exc) {
      setMeta("クリックまたはドロップで別の動画に変更できます");
      setError(String(exc.message || exc));
      setStatus("読み込みに失敗しました");
    }
  }

  async function refreshPreview(path = inputPath) {
    if (!path) return;
    setStatus("プレビューを準備しています…");
    const res = await fetch(`${apiBase}/preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, ...settings }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({ detail: "preview failed" }));
      throw new Error(data.detail || "preview failed");
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    previewRef.current = url;
    setPreviewUrl(url);
    setStatus("プレビューを更新しました");
  }

  async function chooseInput() {
    if (busy) return;
    if (api?.pickInput) {
      const selected = await api.pickInput();
      await applyInput(selected);
      return;
    }
  }

  async function chooseOutput() {
    if (busy) return;
    if (api?.pickOutput) {
      const selected = await api.pickOutput(outputPath);
      if (selected) setOutputPath(selected);
    }
  }

  async function start() {
    if (busy) return;
    if (!inputPath) {
      setError("先に動画ファイルを選択してください。");
      return;
    }
    if (!outputPath.trim()) {
      setError("出力ファイルのパスを指定してください。");
      return;
    }
    setBusy(true);
    setError("");
    setProgress(0);
    setPercent("");
    setStatus("処理を開始しています…");
    try {
      const res = await fetch(`${apiBase}/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input_path: inputPath,
          output_path: outputPath.trim(),
          ...settings,
        }),
      });
      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => ({ detail: "process failed" }));
        throw new Error(data.detail || "process failed");
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() || "";
        for (const chunk of chunks) {
          const line = chunk
            .split("\n")
            .find((item) => item.startsWith("data: "));
          if (!line) continue;
          const payload = JSON.parse(line.slice(6));
          if (payload.error) throw new Error(payload.error);
          if (payload.cancelled) {
            setStatus("キャンセルしました。未完成の出力は削除しています。");
            setProgress(0);
            setPercent("");
            return;
          }
          if (payload.done) {
            setProgress(1);
            setPercent("100%");
            setStatus(`完了: ${payload.output}`);
            if (api?.reveal && window.confirm(`書き出しました。\n${payload.output}\n\nフォルダを開きますか？`)) {
              api.reveal(payload.output);
            }
            return;
          }
          const current = payload.current || 0;
          const total = payload.total;
          const speed = payload.elapsed > 0 ? `${(current / payload.elapsed).toFixed(1)} fps` : "";
          const extra = [`${Number(payload.elapsed || 0).toFixed(1)} 秒`, speed]
            .filter(Boolean)
            .join("  ");
          if (total) {
            const ratio = Math.min(current / total, 1);
            setProgress(ratio);
            setPercent(`${Math.round(ratio * 100)}%`);
            setStatus(`${current} / ${total} フレーム  (${extra})`);
          } else {
            setStatus(`${current} フレーム処理済み  (${extra})`);
          }
        }
      }
    } catch (exc) {
      setError(String(exc.message || exc));
      setStatus(`処理に失敗しました: ${exc.message || exc}`);
    } finally {
      setBusy(false);
    }
  }

  async function cancel() {
    if (!busy) return;
    setStatus("キャンセルしています…");
    await fetch(`${apiBase}/cancel`, { method: "POST" });
  }

  return (
    <div className="app">
      <div className="titlebar" aria-hidden="true" />
      <header className="header">
        <h1>bokasher / ぼかっしゃー</h1>
        <p>動画の顔を、ローカルで自動的にボカします。</p>
      </header>

      <section className="card">
        <h2>動画ファイル</h2>
        <div
          className={`drop${dragging ? " active" : ""}`}
          onClick={chooseInput}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            const file = event.dataTransfer.files?.[0];
            const filePath = nativePath(file);
            if (filePath) applyInput(filePath);
          }}
        >
          <strong>{inputPath ? inputPath.split(/[\\/]/).pop() : "動画をドラッグ＆ドロップ"}</strong>
          <span>
            {meta || "またはクリックして選択（MP4 / MOV / MKV など）"}
          </span>
        </div>
        <div className="output-row">
          <label>出力先</label>
          <input
            value={outputPath}
            placeholder="元ファイル名_blurred.mp4"
            onChange={(event) => setOutputPath(event.target.value)}
            disabled={busy}
          />
          <button className="ghost" onClick={chooseOutput} disabled={busy} type="button">
            変更
          </button>
        </div>
      </section>

      <section className="card">
        <h2>設定</h2>
        <div className="settings">
          <div className="setting">
            <label>ブラー強度</label>
            <input
              type="range"
              min="10"
              max="100"
              value={strength}
              onChange={(event) => setStrength(Number(event.target.value))}
              disabled={busy}
            />
            <strong>{strength}</strong>
          </div>
          <div className="setting modes-row">
            <label>検出モード</label>
            <div className="modes">
              {["速度優先", "精度優先"].map((value) => (
                <button
                  key={value}
                  type="button"
                  className={mode === value ? "active" : ""}
                  onClick={() => setMode(value)}
                  disabled={busy}
                >
                  {value}
                </button>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="card preview-card">
        <h2>プレビュー</h2>
        <div className="preview">
          {previewUrl ? (
            <img src={previewUrl} alt="ぼかしプレビュー" />
          ) : (
            "動画を選ぶと、ぼかし結果をここで確認できます"
          )}
        </div>
      </section>

      <div className="actions">
        <button className="primary" type="button" onClick={start} disabled={busy}>
          処理開始
        </button>
        <button className="ghost" type="button" onClick={cancel} disabled={!busy}>
          キャンセル
        </button>
        <button
          className="textish"
          type="button"
          disabled={busy || !inputPath}
          onClick={() => refreshPreview().catch((exc) => setError(String(exc.message || exc)))}
        >
          プレビュー更新
        </button>
      </div>

      <footer className="footer">
        <div className="progress-row">
          <progress value={progress} max={1} />
          <span>{percent}</span>
        </div>
        <p className={`status${error ? " error" : ""}`}>{error || status}</p>
        <p className="backend">{backend}</p>
      </footer>
    </div>
  );
}
