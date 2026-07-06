# Deploying campeditor

campeditor is a long-running FastAPI server that shells out to **ffmpeg**, downloads
multi-GB video with yt-dlp, and takes 5–10 minutes per render. That shape decides
everything below:

- **Render** (or any container/VM host) runs the whole app well — it is the recommended target.
- **Netlify** only runs short-lived serverless functions (10 s free / 26 s Pro) with no
  ffmpeg and no persistent disk, so it **cannot run the backend**. Use Netlify only for the
  static frontend, pointed at the Render backend.

The recommended setup is therefore a **split**: backend on Render, frontend on Netlify.

---

## 1. Backend on Render (recommended)

The repo ships a `Dockerfile` that installs ffmpeg — use it (Render's native Python
runtime has no ffmpeg).

1. Push this repo to GitHub.
2. Render → **New → Web Service** → connect the repo.
3. **Runtime: Docker** (Render auto-detects the `Dockerfile`). Leave build/start commands blank —
   the Dockerfile's `CMD` binds `0.0.0.0:$PORT`.
4. Instance type: at least **Standard** (renders are CPU/RAM heavy; Free/Starter will OOM or
   time out on `audio-separator` and ffmpeg).
5. **Environment variables** — copy every key from your local `.env` into Render's
   *Environment* tab. At minimum:
   - `NVIDIA_API_KEY`, `NVIDIA_FALLBACK_API_KEY`, `NVIDIA_FALLBACK_API_KEY_2`, `NVIDIA_VISION_MODEL`
   - `GROQ_API_KEY`
   - `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`
   - `PEXELS_API_KEY`
   - `BROLL_*` knobs you have customized, `VARIATION_COUNT`, `MAX_BULK_PAIRS`, `BROLL_LEARNING_ENABLED`
   - `CORS_ALLOW_ORIGINS=https://<your-site>.netlify.app` (see the split section below)
6. **yt-dlp cookies:** add `data/youtube-cookies.txt` as a Render **Secret File**, then set
   `YTDLP_COOKIES_FILE=/etc/secrets/youtube-cookies.txt`. (`YTDLP_COOKIES_FROM_BROWSER`
   only works on a machine with Chrome logged in — it does nothing on a server.)
7. **Persistent storage:** the container filesystem is wiped on every deploy. Job outputs
   under `data/` therefore do **not** survive a redeploy. For real use, attach a Render
   **Disk** mounted at `/app/data` (set `CAMPEDITOR_DATA_DIR=/app/data`), or move outputs
   to external object storage (S3/R2). Without one, treat renders as ephemeral — download
   them before the next deploy.

> ⚠️ **API keys are no longer required.** As of the all-local rewrite, campeditor
> runs entirely against a bundled Ollama daemon + faster-whisper for
> transcription. No Groq / NVIDIA / Gemini / OpenAI keys are read at
> runtime. If you previously pasted any of these in chat, rotate them
> out of an abundance of caution — they don't grant access to this
> deployment anymore, but they're still in the chat history.

### Render without Docker (fallback)

If you must use Render's native Python env instead of Docker, ffmpeg is not preinstalled.
Add a build step that vendors a static ffmpeg (e.g. `imageio-ffmpeg`) and set `FFMPEG_PATH`
to its path. The Docker route above avoids this entirely — prefer it.

---

## 2. Frontend on Netlify (static, points at Render)

The frontend is plain static files in `static/` (`index.html`, `app.js`, `styles.css`) and
already honors a global `window.API_BASE` (defaults to `""` = same origin, so local dev is
unchanged).

1. Netlify → **Add new site → Deploy manually** (or connect the repo).
2. **Publish directory:** `static`. **Build command:** none.
3. Tell the frontend where the backend lives — add this line to `static/index.html`
   **before** `<script src="/app.js"></script>`:
   ```html
   <script>window.API_BASE = "https://<your-service>.onrender.com";</script>
   ```
4. On the backend (Render), set `CORS_ALLOW_ORIGINS` to your Netlify origin, e.g.
   `CORS_ALLOW_ORIGINS=https://<your-site>.netlify.app`. Multiple origins are
   comma-separated. The FastAPI app reads this and configures `CORSMiddleware`.

Local / all-in-one deploys need none of this: leave `window.API_BASE` unset and
`CORS_ALLOW_ORIGINS=*` (the default), and the backend serves `static/` itself at `/`.

---

## 3. Why not all-on-Netlify

Splitting each endpoint into a Netlify Function would require bundling an ffmpeg binary into
every function, adding external storage (functions are stateless), and — fatally — the
render pipeline runs for minutes while Netlify Functions cap at 10 s (free) / 26 s (Pro).
It is not viable. Run the backend on Render.

---

## Quick reference

| Concern            | Local / all-in-one         | Split (Render + Netlify)                    |
| ------------------ | -------------------------- | ------------------------------------------- |
| Backend            | `uvicorn app.main:app`     | Render Web Service (Docker)                 |
| Frontend           | served by backend at `/`   | Netlify static (`static/`)                  |
| `window.API_BASE`  | unset (`""`)               | `https://<service>.onrender.com`            |
| `CORS_ALLOW_ORIGINS` | `*`                      | `https://<site>.netlify.app`                |
| ffmpeg             | on PATH                    | installed by `Dockerfile`                   |
| Renders persist?   | yes (local disk)           | only with a Render Disk at `/app/data`      |

---

## Deploying to Render — step-by-step click guide

Use this section if you want a single Render Web Service running both the
backend and the frontend (the simpler setup), and you don't want Netlify
in the picture. The repo already contains `render.yaml` + `Dockerfile`
+ `start.sh` that wire everything together.

### 1. Push to GitHub

```powershell
cd C:\campeditor
git init 2>$null
git add -A
git commit -m "deploy to render" 2>$null
# Create an empty repo on github.com/new (no README, no .gitignore)
git remote add origin https://github.com/<your-username>/campeditor.git
git branch -M main
git push -u origin main
```

### 2. Sign up to Render

Go to https://render.com → **Get Started for Free** → sign up with GitHub.
Authorize Render to read your repos (scope to `campeditor` only if you
want to be conservative).

### 3. Create the Blueprint

1. Dashboard → **New +** → **Blueprint**.
2. Pick your `campeditor` repo from the list.
3. Render reads `render.yaml` and previews the service. Confirm:
   - Service name: `campeditor`
   - Plan: **Free**
   - Branch: `main`
   - Dockerfile: `./Dockerfile`
   - Health check path: `/api/health`
4. Click **Apply**.

The first build takes ~3–8 minutes. Watch **Logs** for:

```
[start.sh] launching campeditor on 0.0.0.0:10000 with 1 worker(s)
INFO:     Uvicorn running on http://0.0.0.0:10000
INFO:     Application startup complete.
```

### 4. Set environment variables

In the Render dashboard for the new service → **Environment**:

| Key                                  | Value                   | Notes                            |
|--------------------------------------|-------------------------|----------------------------------|
| `GROQ_API_KEY`                       | your Groq key           | Primary vision provider          |
| `NVIDIA_API_KEY`                     | your NVIDIA key         | First NVIDIA fallback            |
| `NVIDIA_FALLBACK_API_KEY`            | another NVIDIA key      | Optional                         |
| `NVIDIA_FALLBACK_API_KEY_2`          | another NVIDIA key      | Optional                         |
| `NVIDIA_FALLBACK_API_KEY_3`          | another NVIDIA key      | Optional                         |
| `GEMINI_API_KEY`                     | your Gemini key         | Optional                         |
| `OLLAMA_VISION_MODEL`                | leave blank             | No local Ollama on Render        |

Defaults already set in `render.yaml` (override if you want):

- `BROLL_INTELLIGENCE_DAILY_QUOTA_COOLDOWN_SECONDS = 1800` (30 min)
- `DAILY_QUOTA_COOLDOWN_SECONDS = 1800`
- `PYTHONUNBUFFERED = 1`

Render's environment storage is encrypted at rest; these keys never
appear in your repo or in deploy logs.

### 5. Share the URL

The URL is at the top of your service page, e.g.
`https://campeditor.onrender.com`. Send that link to anyone; they open
it in a browser, upload a video, get a short back. No login required.

### What the free tier actually gives you

- 750 CPU-hours/month
- Sleeps after 15 min idle; next request wakes in ~30 s
- 512 MB RAM, 0.1 vCPU — a 60 s render takes 5–15 min wall-clock
- 100 GB outbound bandwidth/month
- Auto-renewing HTTPS + custom-domain support
- No persistent disk — every redeploy wipes `data/cache/`. The B-roll
  library cache rebuilds on first render (~5 min one-time vision cost
  per redeploy for your 290 clips).

### Pushing updates

```powershell
cd C:\campeditor
git add -A
git commit -m "what changed"
git push
```

Render auto-deploys on push. New build → new container → ~30 s downtime
on free tier.

### Upgrading

- **Starter ($7/mo):** zero cold starts, 2 GB RAM.
- **Standard ($25/mo):** multiple concurrent renders.
- **Persistent disk:** add a Render Disk mounted at `/app/data` and the
  B-roll cache survives redeploys.
