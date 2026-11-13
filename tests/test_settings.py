"""Settings page API tests.  python tests/test_settings.py"""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_splitter as T  # noqa: E402
from splitter import core  # noqa: E402
from splitter.server import create_app  # noqa: E402


def main():
    tmp = Path(tempfile.mkdtemp())
    root = tmp / "voice_splitter"; root.mkdir(); (tmp / "downloads").mkdir()
    T.make_audio(tmp / "downloads" / "talk.wav")
    app = create_app(root=root, diarize_fn=T.fake_diarize, transcribe_fn=T.fake_words)
    c = app.test_client()

    r = c.get("/api/settings").get_json()
    assert r["settings"]["max_clip"] == 12.0 and r["token"]["source"] in ("none", "env")
    kinds = [f["kind"] for f in r["system"]["folders"]]
    assert kinds[:2] == ["inbox", "ytdip"], kinds
    assert c.get("/settings").status_code == 200
    print("ok  settings page and defaults")

    r = c.put("/api/settings", json={"whisper_model": "turbo", "device": "cpu", "max_clip": 999,
                                      "min_clip": 3, "vp_margin": "0.15", "window": "browser",
                                      "speakers": "auto", "language": "ja", "whisper_model_x": 1})
    s = r.get_json()["settings"]
    assert s["whisper_model"] == "turbo" and s["device"] == "cpu" and s["max_clip"] == 60
    assert s["min_clip"] == 3 and s["vp_margin"] == 0.15 and s["window"] == "browser" and s["language"] == "ja"
    assert core.RUNTIME == {"whisper_model": "turbo", "device": "cpu"}
    assert json.loads((root / "settings.json").read_text())["whisper_model"] == "turbo"
    assert c.put("/api/settings", json={"whisper_model": "evil"}).get_json()["settings"]["whisper_model"] == "turbo"
    assert c.put("/api/settings", json={"min_clip": "abc"}).status_code == 400
    print("ok  settings validated, clamped, saved and applied to the models")

    # token: save, mask, remove
    import os
    os.environ.pop("HF_TOKEN", None)
    assert c.post("/api/token", json={"token": "nope"}).status_code == 400
    c.post("/api/token", json={"token": "hf_abcdefghijklmnop1234"})
    t = c.get("/api/settings").get_json()["token"]
    assert t["source"] == "file" and t["masked"] == "hf_ab…1234", t
    c.delete("/api/token")
    assert c.get("/api/settings").get_json()["token"]["source"] == "none"
    print("ok  token saved, masked (never sent back in full), removable")

    # folders
    extra = tmp / "Music"; extra.mkdir()
    c.post("/api/folders", json={"path": str(extra)})
    rows = c.get("/api/settings").get_json()["system"]["folders"]
    assert rows[-1]["kind"] == "custom" and rows[-1]["remove"] == str(extra)
    c.delete("/api/folders", json={"path": rows[-1]["remove"]})
    assert len(c.get("/api/settings").get_json()["system"]["folders"]) == 2
    print("ok  folders add/remove")

    # export uses settings; storage lists & deletes projects
    src = c.get("/api/library").get_json()["files"][0]["path"]
    j = c.post("/api/split", json={"path": src, "speakers": 2}).get_json()
    while c.get(f"/api/jobs/{j['job']}").get_json()["state"] not in ("done", "error"):
        time.sleep(0.05)
    c.put("/api/settings", json={"min_clip": 2, "max_clip": 5, "language": "ja"})
    c.post(f"/api/projects/{j['project']}/export", json={"open": False})
    out = root / "projects" / j["project"] / "export"
    import soundfile as sf
    durs = [sf.info(str(p)).duration for p in out.rglob("clip_*.wav")]
    assert durs and max(durs) <= 5.05, durs
    line = next(out.rglob("dataset.list")).read_text().splitlines()[0]
    assert line.split("|")[2] == "ja", line
    st = c.get("/api/settings").get_json()["projects"]
    assert len(st) == 1 and st[0]["exported"] and st[0]["size"] > 0
    c.delete(f"/api/projects/{j['project']}")
    assert c.get("/api/settings").get_json()["projects"] == []
    print("ok  export follows clip-length + language settings; storage list and delete")
    print("\nALL PASSED")


if __name__ == "__main__":
    main()


def test_voice_toggle():
    tmp = Path(tempfile.mkdtemp())
    root = tmp / "voice_splitter"; root.mkdir(); (tmp / "downloads").mkdir()
    T.make_audio(tmp / "downloads" / "talk.wav")
    app = create_app(root=root, diarize_fn=T.fake_diarize, transcribe_fn=T.fake_words)
    c = app.test_client()
    lib = c.get("/api/library").get_json()
    j = c.post("/api/split", json={"path": lib["files"][0]["path"], "speakers": 2}).get_json()
    while c.get(f"/api/jobs/{j['job']}").get_json()["state"] not in ("done", "error"):
        time.sleep(0.05)
    proj = c.get(f"/api/projects/{j['project']}").get_json()
    assert [s["name"] for s in proj["speakers"]] == ["Speaker 1", "Speaker 2"]
    assert all(s["export"] for s in proj["speakers"])
    proj["speakers"][1]["export"] = False
    proj["speakers"].append({"id": "VOICE_02", "name": "Third", "color": "#b89be8", "export": True})
    proj["turns"][-1]["speaker"] = "VOICE_02"
    c.put(f"/api/projects/{j['project']}", json=proj)
    saved = c.get(f"/api/projects/{j['project']}").get_json()
    assert saved["speakers"][1]["export"] is False and saved["speakers"][2]["name"] == "Third"
    assert saved["turns"][-1]["speaker"] == "VOICE_02"
    s = c.post(f"/api/projects/{j['project']}/export", json={"open": False}).get_json()["summary"]
    assert "Speaker_2" not in s and "Speaker_1" in s and "Third" in s, s
    out = root / "projects" / j["project"] / "export"
    assert not (out / "Speaker_2").exists()
    assert c.get("/api/library").get_json()["files"][0]["exported"] is True
    print("ok  voice switched off is skipped on export; new voices can be added; library shows Exported")


if __name__ == "__main__":
    test_voice_toggle()
