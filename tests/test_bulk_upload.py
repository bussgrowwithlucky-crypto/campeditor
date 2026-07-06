from fastapi.testclient import TestClient


def _pair_files(count: int):
    files = [("files", (f"raw{i}.mp4", b"\x00" * 2048, "video/mp4")) for i in range(count)]
    refs = [("references", (f"ref{i}.mp4", b"\x00" * 2048, "video/mp4")) for i in range(count)]
    return files + refs


def test_bulk_upload_creates_one_job_per_pair(monkeypatch):
    monkeypatch.setattr("app.jobs.Pipeline._run", lambda self, job_id: None)
    from app.main import app

    with TestClient(app) as client:
        response = client.post("/api/jobs/bulk", files=_pair_files(3))
        assert response.status_code == 200, response.text
        summaries = response.json()
        assert len(summaries) == 3
        store = app.state.store
        for summary in summaries:
            job = store.get(summary["id"])
            assert job.bulk is True
            assert job.variation_count == 1
            assert job.replicate is True


def test_bulk_upload_rejects_mismatched_lengths(monkeypatch):
    monkeypatch.setattr("app.jobs.Pipeline._run", lambda self, job_id: None)
    from app.main import app

    with TestClient(app) as client:
        files = [("files", (f"raw{i}.mp4", b"\x00" * 512, "video/mp4")) for i in range(2)]
        refs = [("references", (f"ref{i}.mp4", b"\x00" * 512, "video/mp4")) for i in range(3)]
        response = client.post("/api/jobs/bulk", files=files + refs)
        assert response.status_code == 400


def test_bulk_upload_rejects_too_many_pairs(monkeypatch):
    monkeypatch.setattr("app.jobs.Pipeline._run", lambda self, job_id: None)
    from app.main import app

    with TestClient(app) as client:
        response = client.post("/api/jobs/bulk", files=_pair_files(101))
        assert response.status_code == 400
