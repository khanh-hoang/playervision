"""Background processing: detect, track, export. One job at a time (CPU bound)."""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from analytics.export import export_players
from api.db import Job
from api.metrics import DURATION, JOBS, QUEUE
from vision.track import track_video

log = logging.getLogger("playervision.jobs")
STRIDE = 2
TRACK_SHARE = 0.9  # tracking is most of the work; export takes the rest of the progress bar


class JobRunner:
    def __init__(self, sessions, jobs_dir, calib):
        self.sessions, self.jobs_dir, self.calib = sessions, Path(jobs_dir), calib
        self.pool = ThreadPoolExecutor(max_workers=1)

    def submit(self, job_id):
        QUEUE.inc()
        self.pool.submit(self.run, job_id)

    def _update(self, job_id, **fields):
        with self.sessions() as s:
            job = s.get(Job, job_id)
            for k, v in fields.items():
                setattr(job, k, v)
            s.commit()

    def run(self, job_id):
        folder = self.jobs_dir / job_id
        video = str(folder / "input.mp4")
        QUEUE.dec()
        started = time.monotonic()
        try:
            self._update(job_id, status="processing", progress=0.0)
            cap = cv2.VideoCapture(video)
            total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // STRIDE, 1)
            cap.release()

            def on_progress(i):
                if i % 5 == 0:
                    self._update(job_id, progress=min(i / total, 1.0) * TRACK_SHARE)

            tracks = track_video(video, stride=STRIDE, on_progress=on_progress)
            if tracks.empty:
                raise ValueError("No players detected in the video")
            self._update(job_id, progress=TRACK_SHARE)
            export_players(video, tracks, self.calib, str(folder / "output"))
            self._update(job_id, status="completed", progress=1.0)
            JOBS.labels("completed").inc()
            log.info("job %s completed in %.0fs", job_id, time.monotonic() - started)
        except Exception as e:  # keep the worker alive; surface the reason to the user
            log.exception("job %s failed", job_id)
            self._update(job_id, status="failed", error=str(e)[:500])
            JOBS.labels("failed").inc()
        finally:
            DURATION.observe(time.monotonic() - started)
