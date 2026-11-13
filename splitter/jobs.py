"""The work queue: one heavy job at a time, in an order you can change.

Jobs wait in `order`; a single worker thread takes the first one. A running job
reports progress through `report(step, frac)`, which is also where a cancel
lands: the next report after Cancel raises core.Cancelled inside the job.
Time estimates come from how long earlier jobs took per MB of file, kept
separately for audio and video files (video MBs are mostly picture).
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from splitter.core import Cancelled

ACTIVE = ("queued", "running")
PUBLIC = ("id", "kind", "name", "project", "path", "state", "step", "pct", "error",
          "added", "started", "finished", "eta")


class JobQueue:
    def __init__(self, run_fn, *, stats_path: Path | None = None, on_failed=None, keep_recent: int = 12,
                 start: bool = True):
        """run_fn(job, report) does the work; on_failed(job) cleans up after an error or cancel."""
        self.run_fn, self.on_failed, self.keep_recent = run_fn, on_failed, keep_recent
        self.stats_path = stats_path
        self.jobs: dict[str, dict] = {}
        self.order: list[str] = []      # waiting, first runs next
        self.recent: list[str] = []     # finished, failed or cancelled, newest first
        self.current: str | None = None
        self.cv = threading.Condition()
        self.rate = self._load_rate()   # {"audio"|"video": seconds of work per MB of file}
        if start:
            threading.Thread(target=self._worker, daemon=True, name="splitter-jobs").start()

    # ---- adding and changing --------------------------------------------
    def add(self, **job) -> dict:
        with self.cv:
            jid = uuid.uuid4().hex[:10]
            job = {"kind": "split", "name": "", "project": None, "path": "", "size": 0, **job,
                   "id": jid, "state": "queued", "step": "Waiting", "pct": 0.0, "error": None,
                   "added": time.time(), "started": None, "finished": None, "cancel": False}
            self.jobs[jid] = job
            self.order.append(jid)
            self.cv.notify_all()
            return job

    def active_for(self, project: str) -> dict | None:
        with self.cv:
            return next((j for j in self.jobs.values() if j["project"] == project and j["state"] in ACTIVE), None)

    def cancel(self, jid: str) -> dict:
        """A waiting job leaves the queue; a running one stops at its next progress report."""
        with self.cv:
            j = self.jobs.get(jid)
            if not j or j["state"] not in ACTIVE:
                raise KeyError(jid)
            if j["state"] == "queued":
                self.order.remove(jid)
                self._finish(j, "cancelled")
            else:
                j["cancel"], j["step"] = True, "Stopping…"
            return j

    def move(self, jid: str, to: str) -> list[str]:
        """to: "top", "up" or "down" within the waiting list."""
        with self.cv:
            if jid not in self.order:
                raise KeyError(jid)
            i = self.order.index(jid)
            self.order.pop(i)
            j = {"top": 0, "up": max(0, i - 1), "down": min(len(self.order), i + 1)}.get(to)
            if j is None:
                self.order.insert(i, jid)
                raise ValueError(to)
            self.order.insert(j, jid)
            return list(self.order)

    def retry(self, jid: str) -> dict:
        with self.cv:
            old = self.jobs.get(jid)
            if not old or old["state"] not in ("error", "cancelled"):
                raise KeyError(jid)
            if self.active_for(old["project"]):
                raise ValueError("This file is already in the queue.")
            self.dismiss(jid)
            keep = {k: v for k, v in old.items() if k not in PUBLIC and k not in ("cancel", "log")}
            return self.add(kind=old["kind"], name=old["name"], project=old["project"], path=old["path"], **keep)

    def dismiss(self, jid: str) -> None:
        with self.cv:
            if jid in self.recent:
                self.recent.remove(jid)
                self.jobs.pop(jid, None)

    def clear_recent(self) -> None:
        with self.cv:
            for jid in self.recent:
                self.jobs.pop(jid, None)
            self.recent.clear()

    # ---- reading ----------------------------------------------------------
    def public(self, j: dict) -> dict:
        return {k: j.get(k) for k in PUBLIC}

    def snapshot(self) -> dict:
        with self.cv:
            now, ahead = time.time(), 0.0
            running = self.jobs.get(self.current) if self.current else None
            if running:
                running["eta"] = self._eta_running(running, now)
                ahead = running["eta"] or 0.0
            queued = []
            for jid in self.order:
                j = self.jobs[jid]
                own = self._estimate(j)
                ahead = ahead + own if own is not None and ahead is not None else None
                j["eta"] = ahead  # time until this one is done
                queued.append(self.public(j))
            return {"running": self.public(running) if running else None, "queued": queued,
                    "recent": [self.public(self.jobs[jid]) for jid in self.recent if jid in self.jobs],
                    "eta": ahead if (running or queued) else 0}

    # ---- the worker -------------------------------------------------------
    def _worker(self):
        while True:
            with self.cv:
                while not self.order:
                    self.cv.wait()
                jid = self.order.pop(0)
                j = self.jobs[jid]
                j.update(state="running", started=time.time(), step="Starting")
                self.current = jid
            self.run_one(j)

    def run_one(self, j: dict) -> None:
        def report(step=None, frac=None):
            if j["cancel"]:
                raise Cancelled()
            if step:
                j["step"] = step
            if frac is not None:
                j["pct"] = round(max(j["pct"], min(1.0, frac)), 3)
        try:
            self.run_fn(j, report)
            with self.cv:
                self._learn(j)
                self._finish(j, "done")
        except Cancelled:
            self._cleanup(j)
            with self.cv:
                self._finish(j, "cancelled")
        except Exception as e:  # noqa: BLE001 - shown in the queue
            from splitter.server import friendly_error
            import traceback
            traceback.print_exc()
            self._cleanup(j)
            with self.cv:
                j["error"] = friendly_error(e)
                self._finish(j, "error")

    def _cleanup(self, j):
        if self.on_failed:
            try:
                self.on_failed(j)
            except Exception:
                pass

    def _finish(self, j: dict, state: str) -> None:
        j.update(state=state, finished=time.time(), eta=None,
                 step={"done": "Done", "cancelled": "Cancelled", "error": "Failed"}[state])
        if self.current == j["id"]:
            self.current = None
        self.recent.insert(0, j["id"])
        for old in self.recent[self.keep_recent:]:
            self.jobs.pop(old, None)
        del self.recent[self.keep_recent:]

    # ---- estimates ---------------------------------------------------------
    def _estimate(self, j: dict) -> float | None:
        r = self.rate.get(j.get("media") or "audio")
        if not r or not j.get("size"):
            return None
        return r * j["size"] / 1e6

    def _eta_running(self, j: dict, now: float) -> float | None:
        elapsed = now - (j["started"] or now)
        if j["pct"] >= 0.08 and elapsed > 5:
            return max(0.0, elapsed / j["pct"] * (1 - j["pct"]))
        own = self._estimate(j)
        return None if own is None else max(5.0, own - elapsed)

    def _learn(self, j: dict) -> None:
        if not j.get("size") or not j.get("started") or j["kind"] != "split":
            return
        took = time.time() - j["started"]
        r, media = took / (j["size"] / 1e6), j.get("media") or "audio"
        old = self.rate.get(media)
        self.rate[media] = r if not old else 0.7 * old + 0.3 * r
        if self.stats_path:
            try:
                self.stats_path.write_text(json.dumps({"seconds_per_mb": self.rate}), encoding="utf-8")
            except OSError:
                pass

    def _load_rate(self) -> dict:
        try:
            saved = json.loads(self.stats_path.read_text(encoding="utf-8"))["seconds_per_mb"]
            return {k: float(v) for k, v in saved.items() if k in ("audio", "video")}
        except Exception:
            return {}
