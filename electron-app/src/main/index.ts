import { app, shell, BrowserWindow, ipcMain, dialog } from 'electron'
import { join } from 'path'
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'fs'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'
import { spawn, ChildProcess } from 'child_process'
import { platform } from 'os'
import * as http from 'http'

// ---------------------------------------------------------------------------
// Proje kök dizinini bul / hatırla
// ---------------------------------------------------------------------------

const CONFIG_PATH = join(app.getPath('userData'), 'config.json')

function loadConfig(): { projectRoot?: string } {
  try {
    return JSON.parse(readFileSync(CONFIG_PATH, 'utf-8'))
  } catch {
    return {}
  }
}

function saveConfig(config: { projectRoot?: string }): void {
  mkdirSync(app.getPath('userData'), { recursive: true })
  writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2))
}

async function getProjectRoot(): Promise<string | null> {
  const config = loadConfig()

  // Daha önce kaydedilmiş ve hâlâ geçerli bir yol var mı?
  if (config.projectRoot && existsSync(join(config.projectRoot, 'src', 'api_server.py'))) {
    return config.projectRoot
  }

  // Dev modunda otomatik bul (electron-app/out/main → 3 üst = proje kökü)
  const devCandidate = join(__dirname, '../../..')
  if (existsSync(join(devCandidate, 'src', 'api_server.py'))) {
    saveConfig({ projectRoot: devCandidate })
    return devCandidate
  }

  // Bulunamadı — kullanıcıya sor (sadece bir kez)
  const result = await dialog.showOpenDialog({
    title: 'NBA Pred proje klasörünü seçin',
    properties: ['openDirectory'],
    buttonLabel: 'Seç',
  })

  if (result.canceled || result.filePaths.length === 0) return null

  const selected = result.filePaths[0]
  if (!existsSync(join(selected, 'src', 'api_server.py'))) {
    dialog.showErrorBox(
      'Geçersiz klasör',
      'Seçilen klasörde src/api_server.py bulunamadı.\nLütfen doğru proje kökünü seçin.'
    )
    return null
  }

  saveConfig({ projectRoot: selected })
  return selected
}

// ---------------------------------------------------------------------------
// Python / FastAPI server yönetimi
// ---------------------------------------------------------------------------

let pythonProcess: ChildProcess | null = null
const API_PORT = 8765

function startPythonServer(projectRoot: string): void {
  const pythonCmd = platform() === 'win32' ? 'python' : 'python3'
  console.log(`[main] Python başlatılıyor @ ${projectRoot}`)

  pythonProcess = spawn(
    pythonCmd,
    ['-m', 'uvicorn', 'src.api_server:app', '--host', '127.0.0.1', '--port', String(API_PORT)],
    { cwd: projectRoot, stdio: ['ignore', 'pipe', 'pipe'] }
  )

  pythonProcess.stdout?.on('data', (d) => console.log('[uvicorn]', String(d).trimEnd()))
  pythonProcess.stderr?.on('data', (d) => console.log('[uvicorn]', String(d).trimEnd()))
  pythonProcess.on('exit', (code) => {
    console.log(`[uvicorn] çıktı, kod: ${code}`)
    pythonProcess = null
  })
}

function checkHealth(): Promise<boolean> {
  return new Promise((resolve) => {
    const req = http.get(`http://127.0.0.1:${API_PORT}/api/health`, (res) => {
      resolve(res.statusCode === 200)
    })
    req.on('error', () => resolve(false))
    req.setTimeout(1000, () => {
      req.destroy()
      resolve(false)
    })
  })
}

async function waitForApi(maxWaitMs = 60000): Promise<void> {
  const start = Date.now()
  while (Date.now() - start < maxWaitMs) {
    if (await checkHealth()) {
      console.log('[main] API hazır.')
      return
    }
    await new Promise((r) => setTimeout(r, 800))
  }
  console.warn('[main] API zaman aşımı — pencere yine de açılıyor.')
}

function stopPythonServer(): void {
  if (pythonProcess) {
    console.log('[main] Python sunucusu kapatılıyor...')
    pythonProcess.kill()
    pythonProcess = null
  }
}

// ---------------------------------------------------------------------------
// Electron penceresi
// ---------------------------------------------------------------------------

function createWindow(): void {
  const mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    show: false,
    autoHideMenuBar: true,
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

// ---------------------------------------------------------------------------
// App lifecycle
// ---------------------------------------------------------------------------

app.whenReady().then(async () => {
  electronApp.setAppUserModelId('com.electron')

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  ipcMain.on('ping', () => console.log('pong'))

  // Proje kökünü bul
  const projectRoot = await getProjectRoot()

  if (projectRoot) {
    // API zaten çalışıyorsa tekrar spawn etme
    const alreadyRunning = await checkHealth()
    if (!alreadyRunning) {
      startPythonServer(projectRoot)
      await waitForApi(60000)
    } else {
      console.log('[main] API zaten çalışıyor, spawn atlandı.')
    }
  } else {
    console.warn('[main] Proje kökü bulunamadı, server başlatılmıyor.')
  }

  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  stopPythonServer()
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('will-quit', () => {
  stopPythonServer()
})
