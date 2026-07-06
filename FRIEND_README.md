# campeditor — for the friend who received this

A web app that turns landscape video clips into 9:16 vertical shorts with auto-captions,
AI-selected viral clips, red word highlights, and a color grade. Runs entirely on your
machine — only ffmpeg and a few cloud APIs (you bring your own keys).

## What you need

- **Windows 10 or 11**
- **Internet access** (the installer pulls Python packages + FFmpeg if missing)
- About **5 GB free disk** (Python venv + FFmpeg + cache)
- Free API keys — see "API keys" below

## Install + run (one command)

1. **Extract** `campeditor.zip` somewhere you'll remember (e.g. `C:\campeditor`).
2. Open **File Explorer**, go into the `campeditor` folder.
3. Open **PowerShell** in that folder:
   - In File Explorer, click the address bar, type `powershell`, press Enter.
4. Run this single command:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\install_and_run.ps1
   ```

   The script will:
   - Check for Python 3.11+ (install via winget if missing)
   - Check for FFmpeg (install via winget if missing)
   - Create a `.venv` and install all dependencies
   - Copy `.env.example` → `.env` (you'll fill in keys next)
   - **If `.env` is missing keys, it will pause and tell you** — fill them in, then
     re-run the same command to start the server.

5. Once the server is up, open **http://127.0.0.1:8000** in your browser.
6. Press **Ctrl+C** in the PowerShell window to stop the server.

## API keys

campeditor calls a few cloud services. You need at minimum:

| Key             | What for                                  | Get one                                               |
| --------------- | ----------------------------------------- | ----------------------------------------------------- |
| `GROQ_API_KEY`  | Whisper transcription (required)          | https://console.groq.com/keys                         |
| `LLM_API_KEY`   | Title generation + clip selection (required) | https://router.bynara.id/settings *(Telegram-gated)* |

Optional (the app falls back gracefully without them):

| Key                       | Purpose                                          | Sign up                                          |
| ------------------------- | ------------------------------------------------ | ------------------------------------------------ |
| `NVIDIA_API_KEY`          | Better vision provider                           | https://build.nvidia.com/explore/discover        |
| `YOUTUBE_DATA_API_KEY`    | B-roll search quota                              | https://console.cloud.google.com/apis/credentials|
| `PEXELS_API_KEY`          | Stock-footage fallback                           | https://www.pexels.com/api/                      |
| `OLLAMA_*`                | Fully-local fallback (no cloud)                  | Install https://ollama.com/, then `ollama pull gemma3:4b` |

Open the `.env` file in any text editor (Notepad is fine), paste your keys, save, restart.

## After it's running

- Upload a landscape video, trim or let AI pick, hit render.
- Renders land under `data/jobs/<job_id>/output.mp4`.
- Browser console (F12) shows progress.

## Manual install (if the auto-install fails)

```powershell
# Python — pick ONE
winget install -e --id Python.Python.3.12
# or download from https://www.python.org/downloads/ (check "Add to PATH")

# FFmpeg — pick ONE
winget install -e --id Gyan.FFmpeg
# or download from https://www.gyan.dev/ffmpeg/builds/ and add bin\ to PATH

# Then re-run
powershell -ExecutionPolicy Bypass -File .\install_and_run.ps1
```

## Troubleshooting

| Symptom                                  | Fix                                                                       |
| ---------------------------------------- | ------------------------------------------------------------------------- |
| `python` not recognized                  | Re-open PowerShell after installing Python, or use the full path.         |
| `ffmpeg` not recognized                  | Re-open PowerShell after installing ffmpeg, or use `Refresh-Path`.        |
| Winget not found                         | Update Windows, or use the manual links above.                           |
| Server starts but uploads fail           | Check `.env` — most failures are missing/expired API keys.                |
| `telegram_required` 403                  | Link Telegram at https://router.bynara.id/settings.                      |
| `audio-separator` install fails          | Run `pip install -e .` again; on some machines the first run needs 2 tries. |

## Files in this folder

```
campeditor/
├── app/                  backend (FastAPI)
├── static/               web UI (single-page)
├── scripts/              helper scripts
├── install_and_run.ps1   ← run this once
├── package_for_friend.ps1 (only relevant to the sender)
├── pyproject.toml        dependency list
├── .env.example          key template (copy → .env, fill in)
└── FRIEND_README.md      this file
```