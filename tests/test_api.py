import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(tmp_path, run_jobs=False))


def _video(path, frames=5, fps=10, size=(64, 48)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(frames):
        w.write(np.zeros((size[1], size[0], 3), np.uint8))
    w.release()
    return path.read_bytes()


def test_upload_creates_pending_job(client, tmp_path):
    data = _video(tmp_path / "v.mp4")
    r = client.post("/api/jobs", files={"file": ("v.mp4", data, "video/mp4")})
    assert r.status_code == 201
    job = r.json()
    assert job["status"] == "pending" and job["players"] == []
    assert client.get(f"/api/jobs/{job['id']}").json()["id"] == job["id"]
    assert len(client.get("/api/jobs").json()) == 1


def test_rejects_wrong_extension_and_non_video(client):
    assert client.post("/api/jobs", files={"file": ("a.txt", b"x")}).status_code == 400
    assert client.post("/api/jobs", files={"file": ("a.mp4", b"not a video")}).status_code == 400


def test_unknown_job_and_unsafe_file_names(client, tmp_path):
    assert client.get("/api/jobs/nope").status_code == 404
    data = _video(tmp_path / "v.mp4")
    job = client.post("/api/jobs", files={"file": ("v.mp4", data)}).json()
    assert client.get(f"/api/jobs/{job['id']}/files/input.mp4").status_code == 404
    assert client.get(f"/api/jobs/{job['id']}/files/overview.png").status_code == 404


def test_health_and_metrics(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    body = client.get("/metrics").text
    assert "playervision_jobs_total" in body and "playervision_jobs_queued" in body


def test_rejects_video_longer_than_limit(client, tmp_path):
    # 400 tiny frames at 1 fps = 400 s of video in a very small file
    data = _video(tmp_path / "long.mp4", frames=400, fps=1, size=(16, 16))
    r = client.post("/api/jobs", files={"file": ("long.mp4", data)})
    assert r.status_code == 400 and "limit" in r.json()["detail"]
    assert client.get("/api/jobs").json() == []


def test_accepts_video_just_under_limit(client, tmp_path):
    data = _video(tmp_path / "ok.mp4", frames=250, fps=1, size=(16, 16))
    assert client.post("/api/jobs", files={"file": ("ok.mp4", data)}).status_code == 201
