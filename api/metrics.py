from prometheus_client import Counter, Gauge, Histogram

JOBS = Counter("playervision_jobs_total", "Finished jobs by outcome", ["status"])
DURATION = Histogram("playervision_job_duration_seconds", "Time to process one job",
                     buckets=(30, 60, 120, 300, 600, 1200, 3600))
QUEUE = Gauge("playervision_jobs_queued", "Jobs waiting for the worker")
