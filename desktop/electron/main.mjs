import { spawn } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { app, BrowserWindow, dialog, ipcMain, shell } from "electron";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..", "..");
const API_PORT = 8765;
const API_BASE = `http://127.0.0.1:${API_PORT}`;

let serverProcess = null;

function pythonBin() {
  const mac = path.join(repoRoot, ".venv", "bin", "python");
  const win = path.join(repoRoot, ".venv", "Scripts", "python.exe");
  if (fs.existsSync(mac)) return mac;
  if (fs.existsSync(win)) return win;
  return process.platform === "win32" ? "python" : "python3";
}

function serverCommand() {
  if (!app.isPackaged) {
    return {
      command: pythonBin(),
      args: ["-m", "bokasher.server", "--host", "127.0.0.1", "--port", String(API_PORT)],
      cwd: repoRoot,
      env: process.env,
    };
  }
  const resources = process.resourcesPath;
  const command = path.join(resources, "server", "bokasher-server");
  const ffmpegDir = path.join(resources, "ffmpeg");
  return {
    command,
    args: ["--host", "127.0.0.1", "--port", String(API_PORT)],
    cwd: path.dirname(command),
    env: {
      ...process.env,
      PATH: `${ffmpegDir}${path.delimiter}${process.env.PATH || ""}`,
      BOKASHER_FFMPEG: path.join(ffmpegDir, "ffmpeg"),
      BOKASHER_FFPROBE: path.join(ffmpegDir, "ffprobe"),
    },
  };
}

function startServer() {
  const { command, args, cwd, env } = serverCommand();
  serverProcess = spawn(command, args, {
    cwd,
    env,
    stdio: "inherit",
  });
  serverProcess.on("exit", (code) => {
    if (code && code !== 0) {
      console.error(`bokasher server exited with ${code}`);
    }
  });
}

function waitForHealth(timeoutMs = 45000) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const ping = () => {
      const req = http.get(`${API_BASE}/health`, (res) => {
        res.resume();
        if (res.statusCode === 200) {
          resolve();
          return;
        }
        retry();
      });
      req.on("error", retry);
    };
    const retry = () => {
      if (Date.now() - started > timeoutMs) {
        reject(new Error("Python サーバーの起動に失敗しました。"));
        return;
      }
      setTimeout(ping, 250);
    };
    ping();
  });
}

async function createWindow() {
  const win = new BrowserWindow({
    width: 760,
    height: 740,
    resizable: false,
    maximizable: false,
    fullscreenable: false,
    title: "bokasher / ぼかっしゃー",
    backgroundColor: "#000000",
    titleBarStyle: "hiddenInset",
    trafficLightPosition: { x: 16, y: 18 },
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });
  win.removeMenu();
  if (app.isPackaged) {
    await win.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  } else {
    await win.loadURL("http://127.0.0.1:5178");
  }
}

ipcMain.handle("api-base", () => API_BASE);

ipcMain.handle("pick-input", async () => {
  const result = await dialog.showOpenDialog({
    title: "動画を選択",
    properties: ["openFile"],
    filters: [
      { name: "動画", extensions: ["mp4", "mov", "m4v", "mkv", "avi", "webm"] },
      { name: "すべて", extensions: ["*"] },
    ],
  });
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle("pick-output", async (_event, initial) => {
  const result = await dialog.showSaveDialog({
    title: "出力先",
    defaultPath: initial || "untitled_blurred.mp4",
    filters: [{ name: "MP4", extensions: ["mp4"] }],
  });
  return result.canceled ? null : result.filePath;
});

ipcMain.handle("reveal", async (_event, filePath) => {
  if (filePath) shell.showItemInFolder(filePath);
});

app.whenReady().then(async () => {
  startServer();
  await waitForHealth();
  await createWindow();
});

app.on("window-all-closed", () => {
  if (serverProcess && !serverProcess.killed) {
    serverProcess.kill();
  }
  app.quit();
});
