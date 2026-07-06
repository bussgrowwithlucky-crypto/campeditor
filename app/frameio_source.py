"""Sync video assets from a public Frame.io share into the local B-roll library.

Auth mechanism ported from the user's own `raw file finder` tool
(frameio-public-share.ts): a public share doesn't need a login — the GraphQL
API accepts requests carrying `x-frameio-share-authentication: base64(shareId)`
plus the web app's Apollo client identity headers. No browser/session needed.

Traversal: GetShareCollectionAssets(folderId, assetType=FOLDER) recursively
discovers subfolders; assetType=FILE lists each folder's files; HydrateAssets
resolves names + downloadable video transcodes for a batch of ids.
"""

import base64
import json
import logging
import re
import shutil
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.frame.io/graphql"
PAGE_SIZE = 200
# Never let a sync run the host drive dry: stop starting new downloads once
# free space would drop below this floor.
MIN_FREE_BYTES = 5 * 1024**3

_LIST_ASSETS_QUERY = """
query GetShareCollectionAssets($shareId: ID!, $folderId: ID, $assetType: ChildAssetTypeInput, $page: PageInput!) {
  share(shareId: $shareId) {
    id
    ... on Share {
      collectionAssets(page: $page, assetType: $assetType, folderId: $folderId) {
        pageInfo { endCursor hasNextPage }
        nodes { id }
        totalCount
      }
    }
  }
}
"""

_HYDRATE_ASSETS_QUERY = """
query HydrateAssets($ids: [ID!]!) {
  assets(assetIds: $ids) {
    id
    __typename
    filetype
    name
    insertedAt
    ... on FolderAsset { itemCount filesize }
    ... on VideoAsset {
      media {
        filesize
        duration
        videoTranscodes { key downloadUrl filesizeInBytes width height encodeStatus }
      }
    }
  }
}
"""


def extract_share_id(url: str) -> str | None:
    match = re.search(r"/share/([0-9a-f-]+)", url, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _headers(share_id: str) -> dict[str, str]:
    return {
        "accept": "*/*",
        "content-type": "application/json",
        "referer": "https://next.frame.io/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) campeditor/0.1",
        "apollographql-client-name": "web-app",
        "apollographql-client-version": "@frameio/next-web-app@510.0",
        "x-frameio-share-authentication": base64.b64encode(share_id.encode()).decode(),
    }


def _post_graphql(share_id: str, operation_name: str, query: str, variables: dict) -> dict:
    body = json.dumps(
        {"operationName": operation_name, "query": query, "variables": variables}
    ).encode()
    headers = {**_headers(share_id), "x-gql-op": operation_name}
    request = urllib.request.Request(GRAPHQL_URL, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def _list_asset_ids(share_id: str, folder_id: str | None, asset_type: str) -> list[str]:
    ids: list[str] = []
    after: str | None = None
    while True:
        page: dict = {"first": PAGE_SIZE}
        if after:
            page["after"] = after
        result = _post_graphql(
            share_id,
            "GetShareCollectionAssets",
            _LIST_ASSETS_QUERY,
            {"shareId": share_id, "folderId": folder_id, "assetType": asset_type, "page": page},
        )
        collection = (result.get("data") or {}).get("share", {}).get("collectionAssets") or {}
        ids.extend(node["id"] for node in collection.get("nodes") or [])
        page_info = collection.get("pageInfo") or {}
        if not page_info.get("hasNextPage") or not page_info.get("endCursor"):
            break
        after = page_info["endCursor"]
    return ids


def _hydrate_assets(share_id: str, ids: list[str]) -> list[dict]:
    if not ids:
        return []
    assets: list[dict] = []
    for start in range(0, len(ids), 50):
        result = _post_graphql(
            share_id, "HydrateAssets", _HYDRATE_ASSETS_QUERY, {"ids": ids[start : start + 50]}
        )
        assets.extend((result.get("data") or {}).get("assets") or [])
    return assets


def _best_download(asset: dict) -> dict | None:
    transcodes = ((asset.get("media") or {}).get("videoTranscodes")) or []
    candidates = [t for t in transcodes if t.get("downloadUrl") and t.get("encodeStatus") == "SUCCESS"]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t.get("filesizeInBytes") or 0)


def _safe_name(name: str, fallback: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", (name or "").strip())
    return cleaned or fallback


def sync_frameio_share(url: str, target_dir: Path) -> dict:
    """Recursively download every video asset from a public Frame.io share
    into target_dir, mirroring the share's folder structure. Idempotent:
    skips files that already exist with the expected size.
    """
    share_id = extract_share_id(url)
    if not share_id:
        raise ValueError(f"Could not extract a Frame.io share ID from: {url}")

    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    stopped_low_disk = False

    def low_disk() -> bool:
        return shutil.disk_usage(target_dir.anchor or str(target_dir)).free < MIN_FREE_BYTES

    def walk(folder_id: str | None, relative_dir: Path) -> None:
        nonlocal stopped_low_disk
        if stopped_low_disk:
            return
        subfolder_ids = _list_asset_ids(share_id, folder_id, "FOLDER")
        for folder in _hydrate_assets(share_id, subfolder_ids):
            if stopped_low_disk:
                return
            if folder.get("__typename") != "FolderAsset":
                continue
            child_dir = relative_dir / _safe_name(folder.get("name"), folder["id"][:8])
            walk(folder["id"], child_dir)

        file_ids = _list_asset_ids(share_id, folder_id, "FILE")
        for asset in _hydrate_assets(share_id, file_ids):
            if asset.get("__typename") != "VideoAsset":
                continue
            best = _best_download(asset)
            if best is None:
                logger.info("No downloadable transcode for %s", asset.get("name"))
                continue
            file_name = _safe_name(asset.get("name"), asset["id"][:8])
            if not file_name.lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".m4v")):
                file_name += ".mp4"
            output_dir = target_dir / relative_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / file_name
            expected_size = best.get("filesizeInBytes") or 0
            if output_path.exists() and (
                expected_size == 0 or abs(output_path.stat().st_size - expected_size) < 1024
            ):
                skipped.append(str(output_path))
                continue
            if low_disk():
                stopped_low_disk = True
                logger.warning("Stopping Frame.io sync: free disk space below %d bytes", MIN_FREE_BYTES)
                return
            try:
                download_request = urllib.request.Request(
                    best["downloadUrl"],
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) campeditor/0.1"},
                )
                with urllib.request.urlopen(download_request, timeout=180) as source, output_path.open(
                    "wb"
                ) as out:
                    while chunk := source.read(1024 * 256):
                        out.write(chunk)
                        if shutil.disk_usage(target_dir.anchor or str(target_dir)).free < MIN_FREE_BYTES:
                            raise IOError("Free disk space dropped below safety floor mid-download")
                downloaded.append(str(output_path))
                logger.info("Downloaded B-roll asset: %s", output_path)
            except Exception:
                logger.exception("Failed to download %s", file_name)
                output_path.unlink(missing_ok=True)
                failed.append(file_name)

    walk(None, Path("."))
    return {
        "share_id": share_id,
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "stopped_low_disk": stopped_low_disk,
    }


# Skip re-walking the share (many GraphQL round-trips even when every file is
# already on disk) if the last successful sync is younger than this. Job-time
# B-roll recovery should prefer the local mirror; a full Frame.io tree walk can
# take minutes on large shares and makes rendering look stuck.
_SYNC_TTL_SECONDS = 12 * 60 * 60


def ensure_frameio_library(share_url: str, settings: "Settings") -> Path:
    """Mirror the Frame.io share into a dedicated B-roll library dir and
    return that dir. Deliberately NOT settings.broll_library_dir: when a job
    selects the Frame.io source, its library must be exactly the share's
    contents — never the machine-local clip folder.

    Idempotent and cheap after the first run: sync_frameio_share skips files
    already on disk, and a `.last_sync` marker skips the share walk entirely
    when it ran successfully within _SYNC_TTL_SECONDS.

    Raises RuntimeError with a user-facing message when the share can't be
    reached or yields no videos.
    """
    share_id = extract_share_id(share_url)
    if not share_id:
        raise RuntimeError(f"Not a valid Frame.io share URL: {share_url}")
    library_dir = settings.data_dir / "broll_frameio" / share_id
    marker = library_dir / ".last_sync"

    def has_videos() -> bool:
        return any(
            p.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm", ".m4v")
            for p in library_dir.rglob("*") if p.is_file()
        )

    if marker.exists() and has_videos():
        try:
            if time.time() - float(marker.read_text(encoding="utf-8").strip()) < _SYNC_TTL_SECONDS:
                return library_dir
        except (ValueError, OSError):
            pass

    try:
        result = sync_frameio_share(share_url, library_dir)
    except Exception as exc:
        # A previously synced copy is still a usable library even when the
        # share is briefly unreachable (network blip, Frame.io hiccup).
        if has_videos():
            logger.warning("Frame.io re-sync failed (%s); using existing local mirror", exc)
            return library_dir
        raise RuntimeError(
            f"Frame.io share is not reachable ({type(exc).__name__}: {exc}). "
            "Check the share URL / your network."
        ) from exc

    if not has_videos():
        raise RuntimeError(
            "Frame.io share sync finished but no videos were found "
            f"(downloaded={len(result['downloaded'])}, failed={len(result['failed'])}). "
            "Check that the share contains video files."
        )
    try:
        marker.write_text(str(time.time()), encoding="utf-8")
    except OSError:
        logger.debug("Could not write Frame.io sync marker", exc_info=True)
    logger.info(
        "Frame.io B-roll library ready at %s (downloaded=%d skipped=%d failed=%d)",
        library_dir, len(result["downloaded"]), len(result["skipped"]), len(result["failed"]),
    )
    return library_dir
