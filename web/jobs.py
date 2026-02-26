"""In-memory background job manager for long-running grading tasks."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Job:
    id: str
    status: str = "running"  # running | completed | failed
    message: str = "Starting..."
    progress: int = 0
    total: int = 0
    result_html: Optional[str] = None
    started_at: float = field(default_factory=time.time)


class JobManager:
    """Thread-safe job store with automatic cleanup."""

    def __init__(self, ttl_seconds: int = 1800):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def start(self, message: str = "Starting...", total: int = 0) -> str:
        job_id = uuid.uuid4().hex[:8]
        job = Job(id=job_id, message=message, total=total)
        with self._lock:
            self._jobs[job_id] = job
        # Schedule cleanup
        timer = threading.Timer(self._ttl, self._remove, args=[job_id])
        timer.daemon = True
        timer.start()
        return job_id

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_progress(self, job_id: str, progress: int, message: str = "") -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.progress = progress
                if message:
                    job.message = message

    def complete(self, job_id: str, result_html: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "completed"
                job.result_html = result_html

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "failed"
                job.message = error

    def _remove(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)


job_manager = JobManager()
