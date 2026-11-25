"""Deleting a file or its split from the library, and downloading the whole transcript."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splitter import core  # noqa: E402
from splitter.server import create_app  # noqa: E402
from test_queue import _files  # noqa: E402

TURNS = [
    {"id": "t1", "start": 0.0, "end": 2.5, "speaker": "A", "text": "Hi there.", "excluded": False},
    {"id": "t2", "start": 2.5, "end": 4.0, "speaker": "B", "text": "Hello!", "excluded": False},
    {"id": "t3", "start": 4.0, "end": 5.0, "speaker": "B", "text": "um", "excluded": True},
    {"id": "t4", "start": 5.0, "end": 3725.25, "speaker": "A", "text": "Bye.", "excluded": False},
]


def _split(tmp_path, client, path):
    pid = next(f for f in client.get("/api/library").get_json()["files"] if f["path"] == path)["pid"]
    d = tmp_path / "projects" / pid
    (d / "export").mkdir(parents=True)
    core.save_project(d, {"name": Path(path).name, "turns": TURNS,
                          "speakers": [{"id": "A", "name": "Ana"}, {"id": "B", "name": "Ben"}]})
    return pid, d


def test_transcript_downloads_as_text_and_subtitles(tmp_path):
    f = _files(tmp_path, 1)[0]
    c = create_app(root=tmp_path, start_worker=False).test_client()
    pid, _ = _split(tmp_path, c, f)
    r = c.get(f"/api/projects/{pid}/transcript.txt")
    assert r.status_code == 200 and "attachment" in r.headers["Content-Disposition"]
    assert "talk0%20transcript.txt" in r.headers["Content-Disposition"]
    assert r.get_data(as_text=True) == "[0:00] Ana:\nHi there.\n\n[0:02] Ben:\nHello!\n\n[0:05] Ana:\nBye.\n"
    srt = c.get(f"/api/projects/{pid}/transcript.srt").get_data(as_text=True)
    assert "3\n00:00:05,000 --> 01:02:05,250\nAna: Bye." in srt and "um" not in srt
    assert "um" in c.get(f"/api/projects/{pid}/transcript.txt?all=1").get_data(as_text=True)
    assert c.get(f"/api/projects/{pid}/transcript.pdf").status_code == 404
    assert c.get("/api/projects/nope/transcript.txt").status_code == 404


def test_remove_split_keeps_the_file(tmp_path):
    f = _files(tmp_path, 1)[0]
    c = create_app(root=tmp_path, start_worker=False).test_client()
    c.put("/api/labels", json={"path": f, "labels": ["Keep"]})
    pid, d = _split(tmp_path, c, f)
    assert c.post("/api/files/delete", json={"path": f}).status_code == 200
    assert not d.exists() and Path(f).exists()
    row = c.get("/api/library").get_json()["files"][0]
    assert row["project"] is None and row["labels"] == ["Keep"]


def test_delete_file_recycles_it_and_forgets_its_labels(tmp_path, monkeypatch):
    f = _files(tmp_path, 2)[0]
    binned = []
    monkeypatch.setattr(core, "recycle", lambda p: (binned.append(p), Path(p).unlink()))
    c = create_app(root=tmp_path, start_worker=False).test_client()
    c.put("/api/labels", json={"path": f, "labels": ["Gone"]})
    _, d = _split(tmp_path, c, f)
    assert c.post("/api/files/delete", json={"path": f, "file": True}).status_code == 200
    assert binned == [Path(f).resolve()] and not d.exists()
    assert [x["path"] for x in c.get("/api/library").get_json()["files"]] != [f]
    assert "Gone" not in (tmp_path / "labels.json").read_text()


def test_delete_cancels_a_waiting_split_but_not_a_running_one(tmp_path):
    a, b = _files(tmp_path, 2)
    app = create_app(root=tmp_path, start_worker=False)
    c = app.test_client()
    ja = c.post("/api/split", json={"path": a}).get_json()["job"]
    c.post("/api/split", json={"path": b})
    assert c.post("/api/files/delete", json={"path": a}).status_code == 200
    q = c.get("/api/queue").get_json()
    assert ja not in [j["id"] for j in q["queued"]] and len(q["queued"]) == 1
    app.config["JOBS"].jobs[q["queued"][0]["id"]]["state"] = "running"   # b is being split now
    r = c.post("/api/files/delete", json={"path": b})
    assert r.status_code == 409 and "Cancel" in r.get_data(as_text=True)


def test_delete_refuses_files_outside_the_library(tmp_path):
    _files(tmp_path, 1)
    c = create_app(root=tmp_path, start_worker=False).test_client()
    outside = tmp_path / "elsewhere.wav"
    outside.write_bytes(b"x" * 4096)
    assert c.post("/api/files/delete", json={"path": str(outside), "file": True}).status_code == 404
    assert outside.exists()
    assert c.delete("/api/projects/..%2Fsecrets").status_code in (400, 404)
