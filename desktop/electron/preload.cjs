const { contextBridge, ipcRenderer, webUtils } = require("electron");

contextBridge.exposeInMainWorld("bokasher", {
  apiBase: () => ipcRenderer.invoke("api-base"),
  pickInput: () => ipcRenderer.invoke("pick-input"),
  pickOutput: (initial) => ipcRenderer.invoke("pick-output", initial),
  reveal: (filePath) => ipcRenderer.invoke("reveal", filePath),
  pathForFile: (file) => (file ? webUtils.getPathForFile(file) : ""),
});
