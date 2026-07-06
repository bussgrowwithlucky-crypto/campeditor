import json
import logging
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.jobs import Pipeline
from app.models import BrollPackDownload, ClipMode, ColorGrade, Job, JobSummary, TitleMode
from app.store import JobStore
from app.youtube_cookies import save_youtube_cookies

logger = logging.getLogger(__name__)


def _warm_frameio_library(settings: Settings) -> None:
    """Background startup warm-up: sync the configured Frame.io share(s) and
    vision-tag their clips so the first Frame.io-sourced job doesn't pay the
    whole download + index cost inside its own watchdog budget. Failures
    are logged, never fatal — the job path re-attempts the sync itself."""
    try:
        from app.broll import build_library_index
        from app.frameio_source import ensure_frameio_library

        library_dirs = []
        for share_url in (
            settings.broll_frameio_share_url.strip(),
            settings.broll_frameio_share_url_2.strip(),
        ):
            if not share_url:
                continue
            try:
                library_dirs.append(ensure_frameio_library(share_url, settings))
            except Exception:
                logger.exception("Frame.io share warm-up failed for %s (jobs will retry)", share_url)
        if library_dirs:
            build_library_index(settings, library_dirs)
            logger.info("Frame.io B-roll library warm-up complete: %s", library_dirs)
    except Exception:
        logger.exception("Frame.io B-roll library warm-up failed (jobs will retry)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not shutil.which(settings.ffmpeg_path):
        raise RuntimeError(
            f"FFmpeg not found at '{settings.ffmpeg_path}'. Install FFmpeg and add it to PATH, "
            "or set FFMPEG_PATH in .env"
        )
    store = JobStore(settings.data_dir)
    app.state.settings = settings
    app.state.store = store
    app.state.pipeline = Pipeline(settings, store)
    if settings.broll_frameio_share_url.strip():
        threading.Thread(
            target=_warm_frameio_library, args=(settings,),
            name="frameio-warmup", daemon=True,
        ).start()
    yield
    app.state.pipeline.shutdown()


app = FastAPI(title="campeditor", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict:
    """Liveness probe for Render / any container host.

    Kept deliberately cheap — no disk checks, no pipeline queries — so it
    responds in microseconds and doesn't trigger ffmpeg work. The host's
    HTTP-level health check uses this to distinguish "process up, just
    slow" from "process dead". Detailed readiness lives at /api/jobs/<id>
    on a per-job basis.
    """
    return {"status": "ok"}

_cors_origins = [o.strip() for o in get_settings().cors_allow_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline


def get_store(request: Request) -> JobStore:
    return request.app.state.store


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


class YouTubeCookiesPayload(BaseModel):
    cookies: str


def _summary(job: Job) -> JobSummary:
    variation_urls = (
        [f"/api/renders/{job.id}/v{i}.mp4" for i in range(len(job.variations))]
        if job.variations
        else []
    )
    # Expose pack clip URLs in index order (rank-1 of span 0 comes first,
    # then rank-2 of span 0, then rank-1 of span 1, etc.) — the same order
    # gather_broll_pack emits them in, which keeps the UI "download pack"
    # affordance deterministic without needing client-side sorting.
    broll_pack_urls = [
        BrollPackDownload(
            span_index=item.span_index,
            rank=item.rank,
            start=item.start,
            end=item.end,
            query=item.query,
            url=f"/api/renders/{job.id}/broll/{i}.mp4",
        )
        for i, item in enumerate(job.broll_pack_items)
    ]
    return JobSummary(
        id=job.id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        output_url=f"/api/renders/{job.id}.mp4" if job.output_path else None,
        variation_urls=variation_urls,
        broll_pack_urls=broll_pack_urls,
        error=job.error,
        warning=getattr(job, "warning", "") or "",
        eta_seconds=job.eta_seconds,
        stage_timings=job.stage_timings,
    )


@app.post("/api/jobs/upload", response_model=JobSummary)
def upload_job(
    file: UploadFile = File(...),
    clip_mode: ClipMode = Form(ClipMode.MANUAL),
    start: float = Form(0.0),
    end: float = Form(15.0),
    title_mode: TitleMode = Form(TitleMode.AUTO),
    manual_title: str = Form(""),
    color_grade: ColorGrade = Form(ColorGrade.NONE),
    replicate: bool = Form(False),
    reference: UploadFile | None = File(None),
    reference_url: str = Form(""),
    music: UploadFile | None = File(None),
    logo: UploadFile | None = File(None),
    broll_pack: bool = Form(False),
    enable_learned_broll: bool = Form(True),
    use_intelligent_selector: bool = Form(True),
    add_caption: bool = Form(True),
    broll_source: str = Form("both"),
    use_broll_frameio_2: bool = Form(False),
    pipeline: Pipeline = Depends(get_pipeline),
) -> JobSummary:
    try:
        job = pipeline.create_job(
            file,
            clip_mode,
            start,
            end,
            title_mode,
            manual_title,
            color_grade,
            replicate=replicate,
            reference_upload=reference,
            reference_url=reference_url,
            music_upload=music,
            logo_upload=logo,
            broll_pack=broll_pack,
            enable_learned_broll=enable_learned_broll,
            use_intelligent_selector=use_intelligent_selector,
            add_caption=add_caption,
            broll_source=broll_source,
            use_broll_frameio_2=use_broll_frameio_2,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _summary(job)


@app.post("/api/jobs/bulk", response_model=list[JobSummary])
def upload_bulk(
    files: list[UploadFile] = File(...),
    references: list[UploadFile] = File(...),
    clip_mode: ClipMode = Form(ClipMode.AUTO),
    title_mode: TitleMode = Form(TitleMode.AUTO),
    color_grade: ColorGrade = Form(ColorGrade.NONE),
    music: UploadFile | None = File(None),
    logo: UploadFile | None = File(None),
    pipeline: Pipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_app_settings),
) -> list[JobSummary]:
    if len(files) != len(references):
        raise HTTPException(
            status_code=400,
            detail="Each raw video needs one matching reference (counts differ)",
        )
    if len(files) < 1:
        raise HTTPException(status_code=400, detail="Bulk upload requires at least one pair")
    if len(files) > settings.max_bulk_pairs:
        raise HTTPException(
            status_code=400,
            detail=f"Bulk upload is capped at {settings.max_bulk_pairs} pairs",
        )
    summaries: list[JobSummary] = []
    try:
        for raw, reference in zip(files, references):
            if music is not None and music.filename:
                music.file.seek(0)
            if logo is not None and logo.filename:
                logo.file.seek(0)
            job = pipeline.create_bulk_job(
                raw,
                reference,
                clip_mode=clip_mode,
                title_mode=title_mode,
                color_grade=color_grade,
                music_upload=music,
                logo_upload=logo,
            )
            summaries.append(_summary(job))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return summaries


@app.post("/api/jobs/upload-from-folder", response_model=JobSummary)
def upload_from_folder(
    frameio_url: str = Form(...),
    clip_mode: ClipMode = Form(ClipMode.AUTO),
    title_mode: TitleMode = Form(TitleMode.AUTO),
    manual_title: str = Form(""),
    color_grade: ColorGrade = Form(ColorGrade.NONE),
    reference: UploadFile | None = File(None),
    reference_url: str = Form(""),
    music: UploadFile | None = File(None),
    logo: UploadFile | None = File(None),
    pipeline: Pipeline = Depends(get_pipeline),
) -> JobSummary:
    """Create a job that auto-finds its raw source video from a Frame.io folder.

    Provide a Frame.io folder URL (containing raw/long-form source footage) and
    a reference short video.  The system syncs the folder, visually matches the
    reference against every video to find which one it was cut from, identifies
    the exact timestamp range, and then proceeds with replicate-mode editing
    (B-roll, title, music, rendering).
    """
    try:
        job = pipeline.create_folder_job(
            frameio_url=frameio_url,
            clip_mode=clip_mode,
            title_mode=title_mode,
            manual_title_text=manual_title,
            color_grade=color_grade,
            reference_upload=reference,
            reference_url=reference_url,
            music_upload=music,
            logo_upload=logo,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _summary(job)


@app.get("/api/jobs/{job_id}", response_model=JobSummary)
def get_job(job_id: str, store: JobStore = Depends(get_store)) -> JobSummary:
    try:
        return _summary(store.get(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@app.post("/api/jobs/{job_id}/cancel", response_model=JobSummary)
def cancel_job(
    job_id: str,
    store: JobStore = Depends(get_store),
    pipeline: Pipeline = Depends(get_pipeline),
) -> JobSummary:
    try:
        store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    pipeline.cancel(job_id)
    return _summary(store.get(job_id))


@app.post("/api/youtube-cookies")
def save_cookies(
    payload: YouTubeCookiesPayload,
    settings: Settings = Depends(get_app_settings),
):
    try:
        return save_youtube_cookies(payload.cookies, settings)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/renders/{job_id}.mp4")
def download_render(job_id: str, store: JobStore = Depends(get_store)) -> FileResponse:
    try:
        job = store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    if not job.output_path:
        raise HTTPException(status_code=404, detail="Render not ready")
    renders_root = store.jobs_dir.resolve()
    resolved = Path(job.output_path).resolve()
    if renders_root not in resolved.parents or not resolved.exists():
        raise HTTPException(status_code=404, detail="Rendered file is missing")
    return FileResponse(resolved, media_type="video/mp4", filename=f"campeditor-{job_id}.mp4")


@app.get("/api/renders/{job_id}/v{index}.mp4")
def download_variation(
    job_id: str, index: int, store: JobStore = Depends(get_store)
) -> FileResponse:
    try:
        job = store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    if index < 0 or index >= len(job.variations):
        raise HTTPException(status_code=404, detail="Variation not found")
    output_path = job.variations[index].output_path
    if not output_path:
        raise HTTPException(status_code=404, detail="Render not ready")
    renders_root = store.jobs_dir.resolve()
    resolved = Path(output_path).resolve()
    if renders_root not in resolved.parents or not resolved.exists():
        raise HTTPException(status_code=404, detail="Rendered file is missing")
    return FileResponse(
        resolved, media_type="video/mp4", filename=f"campeditor-{job_id}-v{index}.mp4"
    )


@app.get("/api/renders/{job_id}/broll/{index}.mp4")
def download_broll_pack_clip(
    job_id: str, index: int, store: JobStore = Depends(get_store)
) -> FileResponse:
    """Download one trimmed clip from a job's B-roll pack.

    Mirror of `download_variation` — same path-containment check against
    store.jobs_dir works because pack files live under data/jobs/<id>/broll/.
    The downloadable filename surfaces the human-meaningful span + rank so a
    user unzipping their downloads can see at a glance which cut each file
    is meant for.
    """
    try:
        job = store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    if index < 0 or index >= len(job.broll_pack_items):
        raise HTTPException(status_code=404, detail="B-roll pack clip not found")
    item = job.broll_pack_items[index]
    renders_root = store.jobs_dir.resolve()
    resolved = Path(item.clip_path).resolve()
    if renders_root not in resolved.parents or not resolved.exists():
        raise HTTPException(status_code=404, detail="Pack clip file is missing")
    return FileResponse(
        resolved,
        media_type="video/mp4",
        filename=f"campeditor-{job_id}-broll-span{item.span_index + 1}-option{item.rank}.mp4",
    )


app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent.parent / "static", html=True), name="static")
