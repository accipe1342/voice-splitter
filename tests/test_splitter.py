"""End-to-end test with fake models (no downloads).  python tests/test_splitter.py"""
import io
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from splitter import core  # noqa: E402
from splitter.server import create_app  # noqa: E402

SR = 44100
# Speaker A = 220 Hz tone, speaker B = 440 Hz tone.
SCRIPT = [(0.0, 6.0, "A"), (6.0, 9.0, "B"), (9.0, 30.0, "A"), (30.0, 31.0, "B"), (31.0, 40.0, "B")]


def make_audio(path: Path):
    t = np.arange(int(SR * 40)) / SR
    freq = np.zeros_like(t)
    for s, e, spk in SCRIPT:
        freq[(t >= s) & (t < e)] = 220 if spk == "A" else 440
    sf.write(path, (0.3 * np.sin(2 * np.pi * freq * t)).astype("float32"), SR)


def fake_diarize(wav, n, token):
    # one deliberate mistake: 30-31s is really B but labeled A
    return [(0.0, 5.9, "SPEAKER_00"), (6.1, 8.9, "SPEAKER_01"), (9.1, 29.9, "SPEAKER_00"),
            (30.4, 30.95, "SPEAKER_00"), (31.1, 39.9, "SPEAKER_01")]


def fake_words(wav, lang):
    words = []
    for s, e, spk in SCRIPT:
        x = s + 0.1
        while x + 0.4 < e:
            words.append({"w": "alpha" if spk == "A" else "bravo", "s": round(x, 2), "e": round(x + 0.35, 2)})
            x += 0.5
    return words


def fake_embed_for(wav):
    data, sr = sf.read(str(wav), dtype="float32")

    def embed(s, e):
        seg = data[int(s * sr):int(e * sr)]
        f = np.abs(np.fft.rfft(seg)).argmax() * sr / len(seg)
        return np.array([1.0, 0.0]) if f < 330 else np.array([0.0, 1.0])
    return embed


def dominant(path):
    d, r = sf.read(str(path))
    return round(np.abs(np.fft.rfft(d)).argmax() * r / len(d))


def main():
    tmp = Path(tempfile.mkdtemp())
    root = tmp / "voice_splitter"
    root.mkdir()
    (tmp / "downloads").mkdir()           # pretend we live inside YTdip
    make_audio(tmp / "downloads" / "talk.wav")

    embed_holder = {}
    app = create_app(root=root, diarize_fn=fake_diarize, transcribe_fn=fake_words,
                     embed_fn=lambda s, e: embed_holder["f"](s, e))
    c = app.test_client()

    lib = c.get("/api/library").get_json()
    names = [f["name"] for f in lib["files"]]
    assert "talk.wav" in names and any(f.endswith("downloads") for f in lib["folders"]), lib
    print("ok  YTdip downloads folder picked up automatically")

    # upload another file through the page
    buf = io.BytesIO(); sf.write(buf, np.zeros(SR * 3, dtype="float32") + 0.01, SR, format="WAV"); buf.seek(0)
    r = c.post("/api/upload", data={"files": (buf, "my voice memo.wav")}, content_type="multipart/form-data")
    assert r.status_code == 200, r.data
    assert "my_voice_memo.wav" in [f["name"] for f in c.get("/api/library").get_json()["files"]]
    r = c.post("/api/upload", data={"files": (io.BytesIO(b"x"), "notes.txt")}, content_type="multipart/form-data")
    assert r.status_code == 400
    print("ok  uploads land in the library; non-audio rejected")

    # security: can't split files outside the library
    assert c.post("/api/split", json={"path": str(Path(__file__))}).status_code == 400
    assert c.get("/api/projects/..%2f..%2fetc").status_code in (400, 404)
    print("ok  only library files can be processed")

    # split
    src = next(f["path"] for f in lib["files"] if f["name"] == "talk.wav")
    j = c.post("/api/split", json={"path": src, "speakers": 2}).get_json()
    for _ in range(100):
        st = c.get(f"/api/jobs/{j['job']}").get_json()
        if st["state"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert st["state"] == "done", st
    pid = j["project"]
    proj = c.get(f"/api/projects/{pid}").get_json()
    assert len(proj["speakers"]) == 2 and proj["turns"][0]["text"].startswith("alpha")
    assert c.get(f"/api/projects/{pid}/peaks").get_json()["duration"] > 39
    assert c.get(f"/api/projects/{pid}/playback").status_code == 200
    assert c.post("/api/split", json={"path": src}).get_json()["existing"] is True
    print("ok  split job: audio, waveform, transcript per line, playback")

    # voiceprint: confirm one line per speaker, then it should flag the 30-31s mistake
    embed_holder["f"] = fake_embed_for(root / "projects" / pid / "audio.wav")
    for t in proj["turns"]:
        if t["start"] < 1 or 6 < t["start"] < 7:
            t["confirmed"] = True
    c.put(f"/api/projects/{pid}", json=proj)
    r = c.post(f"/api/projects/{pid}/voiceprint").get_json()
    flagged = [t for t in r["project"]["turns"] if t["suggest"]]
    assert r["flagged"] == 1 and 30.3 < flagged[0]["start"] < 30.5 and flagged[0]["suggest"] == "SPEAKER_01", r["flagged"]
    print("ok  voiceprint check flags the mislabeled line")

    # edits: accept suggestion, rename speakers, exclude, resize, split, edit text
    proj = r["project"]
    for t in proj["turns"]:
        if t["suggest"]:
            t["speaker"], t["suggest"] = t["suggest"], None
    proj["speakers"][0]["name"] = "Me"
    proj["speakers"][1]["name"] = "Other person"
    long_a = next(t for t in proj["turns"] if t["start"] > 9 and t["start"] < 10)
    new_id = max(t["id"] for t in proj["turns"]) + 1
    proj["turns"].append({**long_a, "id": new_id, "start": 20.0, "excluded": True})   # split, exclude 2nd half
    long_a["end"] = 20.0
    first = proj["turns"][0]
    first["text"], first["text_edited"] = "this is my corrected sentence", True
    assert c.put(f"/api/projects/{pid}", json=proj).status_code == 200
    saved = c.get(f"/api/projects/{pid}").get_json()
    ln = next(t for t in saved["turns"] if t["id"] == long_a["id"])
    assert ln["end"] == 20.0 and all(w["s"] < 20 for w in ln["words"]), "words re-derived after resize"
    assert next(t for t in saved["turns"] if t["id"] == new_id)["excluded"]
    print("ok  edits saved: relabel, rename, split, exclude, trim, text correction")

    # export
    r = c.post(f"/api/projects/{pid}/export", json={"open": False}).get_json()
    s = r["summary"]
    assert set(s) == {"Me", "Other_person"}, s
    out = root / "projects" / pid / "export"
    me = sorted((out / "Me").glob("clip_*.wav"))
    other = sorted((out / "Other_person").glob("clip_*.wav"))
    assert me and other
    assert all(dominant(p) == 220 for p in me), [dominant(p) for p in me]
    assert all(dominant(p) == 440 for p in other), [dominant(p) for p in other]
    for p in me + other:
        assert 2.0 <= sf.info(str(p)).duration <= 12.05, (p, sf.info(str(p)).duration)
    me_total = sum(sf.info(str(p)).duration for p in me)
    assert me_total < 17, me_total          # excluded 20-30s is gone
    lines = (out / "Me" / "dataset.list").read_text().strip().splitlines()
    assert lines[0].split("|")[1:3] == ["Me", "en"] and "corrected sentence" in lines[0]
    assert all("bravo" not in l for l in lines)
    assert (out / "Me_preview.wav").exists()
    print(f"ok  export: Me {s['Me']['clips']} clips / Other {s['Other_person']['clips']} clips, every clip only its own voice,"
          " 2-12s, excluded audio dropped, dataset.list uses your text")

    # library shows where each file is in review
    st = next(f for f in c.get("/api/library").get_json()["files"] if f["name"] == "talk.wav")["review"]
    assert st["stage"] == "exported", st
    print("ok  library knows the file is exported")

    # re-split the whole recording: speakers keep their names, checked lines stay put
    before = c.get(f"/api/projects/{pid}").get_json()
    checked = [t for t in before["turns"] if t["confirmed"]]
    j = c.post(f"/api/projects/{pid}/resplit", json={"start": 0, "end": None}).get_json()
    for _ in range(100):
        st = c.get(f"/api/jobs/{j['job']}").get_json()
        if st["state"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert st["state"] == "done", st
    after = c.get(f"/api/projects/{pid}").get_json()
    assert [s["name"] for s in after["speakers"]] == ["Me", "Other person"], after["speakers"]
    for t in checked:
        assert t in [{**x, "words": x["words"]} for x in after["turns"]], ("checked line changed", t)
    assert after["can_restore"] is True
    assert core.review_stage(root / "projects" / pid)["stage"] != "exported"  # edits since the export
    restored = c.post(f"/api/projects/{pid}/restore").get_json()
    assert len(restored["turns"]) == len(before["turns"])
    assert c.post(f"/api/projects/{pid}/resplit", json={"start": 5, "end": 5.5}).status_code == 400
    print("ok  re-split keeps names and checked lines; previous split restorable")

    # plan_clips splits at word boundaries
    p = core.load_project(root / "projects" / pid)
    for cl in core.plan_clips(p):
        for w in p["words"]:
            assert not (w["s"] < cl["start"] - 0.09 < w["e"] - 0.1 and cl["start"] > 1), ("cut through a word", cl, w)
    print("ok  long lines split between words, not through them")
    print("\nALL PASSED")


def test_end_to_end():
    main()


if __name__ == "__main__":
    main()


def test_save_survives_a_briefly_locked_file(tmp_path, monkeypatch):
    """Windows: a reader or antivirus can block the rename for a moment (WinError 5)."""
    from splitter import core as C
    real, calls = C.os.replace, {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(5, "Access is denied")
        return real(a, b)

    monkeypatch.setattr(C.os, "replace", flaky)
    C.save_project(tmp_path, {"name": "x", "turns": []})
    assert calls["n"] == 3 and C.load_project(tmp_path)["name"] == "x"
