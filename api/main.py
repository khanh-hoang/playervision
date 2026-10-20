"""FastAPI app: upload a clip, poll the job, fetch per-player heatmaps.

Run:  uvicorn api.main:app --port 8000
"""
import logging
import os
import re
import shutil
import uuid
from pathlib import Path

import cv2
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from sqlalchemy import text

from api.db import Job, make_session_factory
from api.jobs import JobRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent
ALLOWED = {".mp4", ".mov", ".avi", ".mkv"}
MAX_UPLOAD = int(os.environ.get("PLAYERVISION_MAX_UPLOAD_MB", "500")) * 1024 * 1024
MAX_SECONDS = int(os.environ.get("PLAYERVISION_MAX_SECONDS", "300"))
FILE_RE = re.compile(r"^(overview|player_\d+)\.png$")


class PlayerOut(BaseModel):
    track_id: int
    frames: int
    coverage_pct: float
    reliability: str
    approx_distance_m: float


class JobOut(BaseModel):
    id: str
    filename: str
    status: str
    progress: float
    error: str | None = None
    players: list[PlayerOut] = []


def create_app(data_dir=None, run_jobs=True, calib=None):
    data = Path(data_dir or os.environ.get("PLAYERVISION_DATA", ROOT / "data"))
    jobs_dir = data / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    sessions = make_session_factory(data / "playervision.db")
    runner = JobRunner(sessions, jobs_dir, calib or str(ROOT / "calibration.json"))

    with sessions() as s:  # jobs interrupted by a restart can never finish
        for j in s.query(Job).filter(Job.status.in_(["pending", "processing"])):
            j.status, j.error = "failed", "Interrupted by server restart"
        s.commit()

    app = FastAPI(title="PlayerVision")

    @app.get("/healthz")
    def healthz():
        with sessions() as s:
            s.execute(text("SELECT 1"))
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    def to_out(job):
        players = []
        summary = jobs_dir / job.id / "output" / "summary.csv"
        if job.status == "completed" and summary.exists():
            players = pd.read_csv(summary).to_dict("records")
        return JobOut(id=job.id, filename=job.filename, status=job.status, progress=job.progress,
                      error=job.error, players=players)

    def get_job(job_id: str):
        with sessions() as s:
            job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    @app.post("/api/jobs", response_model=JobOut, status_code=201)
    async def create_job(file: UploadFile):
        name = Path(file.filename or "").name
        if Path(name).suffix.lower() not in ALLOWED:
            raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED))}")
        job_id = uuid.uuid4().hex
        folder = jobs_dir / job_id
        folder.mkdir()
        path = folder / "input.mp4"
        size = 0
        with open(path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    f.close()
                    shutil.rmtree(folder)
                    raise HTTPException(413, "File too large")
                f.write(chunk)
        cap = cv2.VideoCapture(str(path))
        readable = cap.isOpened() and cap.read()[0]
        fps = cap.get(cv2.CAP_PROP_FPS)
        seconds = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
        cap.release()
        if not readable:
            shutil.rmtree(folder)
            raise HTTPException(400, "Could not read this file as a video")
        if seconds > MAX_SECONDS:
            shutil.rmtree(folder)
            raise HTTPException(400, f"Video is {seconds:.0f}s long; the limit is {MAX_SECONDS}s")
        with sessions() as s:
            job = Job(id=job_id, filename=name)
            s.add(job)
            s.commit()
        if run_jobs:
            runner.submit(job_id)
        return to_out(job)

    @app.get("/api/jobs", response_model=list[JobOut])
    def list_jobs():
        with sessions() as s:
            jobs = s.query(Job).order_by(Job.created_at.desc()).all()
        return [to_out(j) for j in jobs]

    @app.get("/api/jobs/{job_id}", response_model=JobOut)
    def job_detail(job_id: str, job: Job = Depends(get_job)):
        return to_out(job)

    @app.get("/api/jobs/{job_id}/files/{name}")
    def job_file(job_id: str, name: str, job: Job = Depends(get_job)):
        path = jobs_dir / job.id / "output" / name
        if not FILE_RE.match(name) or not path.exists():
            raise HTTPException(404, "File not found")
        return FileResponse(path, media_type="image/png")

    dist = ROOT / "frontend" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app


app = create_app()
