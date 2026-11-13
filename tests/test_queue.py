"""The work queue: order, moving, cancelling, retrying, one job at a time, and labels."""
import sys
import threading

import pytest
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splitter import core  # noqa: E402
from splitter.jobs import JobQueue  # noqa: E402
from splitter.server import create_app  # noqa: E402


def _q(run=lambda j, report: None, **kw):
    return JobQueue(run, start=False, **kw)


def test_waiting_order_can_be_changed():
    q = _q()
    a, b, c = (q.add(name=n, project=n) for n in "abc")
    assert q.move(c["id"], "top") == [c["id"], a["id"], b["id"]]
    assert q.move(a["id"], "down") == [c["id"], b["id"], a["id"]]
    assert q.move(a["id"], "up") == [c["id"], a["id"], b["id"]]
    assert [j["name"] for j in q.snapshot()["queued"]] == ["c", "a", "b"]


def test_cancel_a_waiting_job_takes_it_out():
    q = _q()
    a, b = q.add(name="a", project="a"), q.add(name="b", project="b")
    q.cancel(a["id"])
    snap = q.snapshot()
    assert [j["name"] for j in snap["queued"]] == ["b"]
    assert snap["recent"][0]["state"] == "cancelled"


def test_cancel_a_running_job_stops_it_at_the_next_step_and_cleans_up():
    cleaned, reached = [], []

    def run(job, report):
        report("Finding who speaks when", 0.2)
        q.cancel(job["id"])          # someone presses Cancel mid-job
        report(None, 0.5)            # the next progress report stops it
        reached.append("too far")

    q = _q(run, on_failed=lambda j: cleaned.append(j["id"]))
    j = q.add(name="a", project="a")
    q.order.remove(j["id"]); q.current = j["id"]; j["state"] = "running"
    q.run_one(j)
    assert j["state"] == "cancelled" and cleaned == [j["id"]] and not reached
    assert q.snapshot()["running"] is None


def test_a_failed_job_can_be_retried_with_the_same_settings():
    def run(job, report):
        raise RuntimeError("401 gated")
    q = _q(run)
    j = q.add(name="a", project="a", speakers=2, language="ja")
    q.order.remove(j["id"]); q.run_one(j)
    assert j["state"] == "error" and "Setup" in j["error"]
    again = q.retry(j["id"])
    assert again["state"] == "queued" and again["speakers"] == 2 and again["language"] == "ja"
    assert [r["id"] for r in q.snapshot()["recent"]] == []  # the failed entry made way for the retry


def test_estimates_learn_from_finished_jobs(tmp_path):
    q = _q(stats_path=tmp_path / "speed.json")
    j = q.add(name="a", project="a", size=10_000_000)
    assert q.snapshot()["queued"][0]["eta"] is None      # nothing to go on yet
    q.order.remove(j["id"]); j["started"] = time.time() - 20; q._learn(j)
    nxt = q.add(name="b", project="b", size=5_000_000)
    assert 9 < q.snapshot()["queued"][0]["eta"] < 11       # 2 s per MB, 5 MB
    assert JobQueue(lambda j, r: None, stats_path=tmp_path / "speed.json", start=False).rate["audio"] == pytest.approx(2.0, rel=1e-3)
    assert nxt["id"]


def test_only_one_job_runs_at_a_time():
    running, peak, gate = [0], [0], threading.Event()

    def run(job, report):
        running[0] += 1
        peak[0] = max(peak[0], running[0])
        gate.wait(2)
        running[0] -= 1
    q = JobQueue(run)
    for n in "abc":
        q.add(name=n, project=n)
    time.sleep(0.2)
    snap = q.snapshot()
    assert snap["running"]["name"] == "a" and [j["name"] for j in snap["queued"]] == ["b", "c"]
    gate.set()
    for _ in range(50):
        if len(q.snapshot()["recent"]) == 3:
            break
        time.sleep(0.05)
    assert peak[0] == 1 and [r["state"] for r in q.snapshot()["recent"]] == ["done"] * 3


def _files(tmp_path, n=3):
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True)
    out = []
    for i in range(n):
        f = inbox / f"talk{i}.wav"
        sf.write(str(f), np.zeros(core.SR * 2, dtype="float32"), core.SR)
        out.append(str(f))
    (tmp_path / "secrets").mkdir()
    (tmp_path / "secrets" / "hf_token.txt").write_text("hf_x")
    return out


def test_queue_api_lists_reorders_and_cancels(tmp_path):
    files = _files(tmp_path)
    c = create_app(root=tmp_path, start_worker=False).test_client()
    ids = [c.post("/api/split", json={"path": f}).get_json()["job"] for f in files]
    q = c.get("/api/queue").get_json()
    assert [j["id"] for j in q["queued"]] == ids and q["running"] is None
    assert c.post(f"/api/jobs/{ids[2]}/move", json={"to": "top"}).get_json()["order"][0] == ids[2]
    c.post(f"/api/jobs/{ids[0]}/cancel")
    q = c.get("/api/queue").get_json()
    assert [j["id"] for j in q["queued"]] == [ids[2], ids[1]] and q["recent"][0]["state"] == "cancelled"
    lib = c.get("/api/library").get_json()
    assert len(lib["running"]) == 2                       # still waiting, so not "Split speakers" again
    assert c.post(f"/api/jobs/{ids[0]}/move", json={"to": "up"}).status_code == 409


def test_labels_are_saved_per_file_and_cleaned(tmp_path):
    files = _files(tmp_path, 1)
    c = create_app(root=tmp_path, start_worker=False).test_client()
    r = c.put("/api/labels", json={"path": files[0], "labels": [" Podcast ", "podcast", "  Ep  12 ", ""]})
    assert r.get_json()["labels"] == ["Podcast", "Ep 12"]
    row = next(f for f in c.get("/api/library").get_json()["files"] if f["path"] == files[0])
    assert row["labels"] == ["Podcast", "Ep 12"]
    assert c.put("/api/labels", json={"path": "/etc/passwd", "labels": ["x"]}).status_code == 400
