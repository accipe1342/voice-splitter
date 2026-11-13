"""Things found in the pre-release pass: export naming and transcripts, local-only
requests, friendly errors, and a library that survives odd files."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splitter import core  # noqa: E402
from splitter.server import create_app, friendly_error  # noqa: E402


def _project(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    sf.write(str(d / "audio.wav"), np.zeros(core.SR * 12, dtype="float32"), core.SR)
    words = lambda text, s: [{"w": w, "s": s + i * 0.5, "e": s + i * 0.5 + 0.4} for i, w in enumerate(text.split())]
    turns = [
        {"id": 0, "start": 0.0, "end": 4.0, "speaker": "A", "text": "hello there | friend", "words": words("hello there | friend", 0.2),
         "excluded": False, "confirmed": False, "text_edited": False, "suggest": None},
        {"id": 1, "start": 5.0, "end": 9.0, "speaker": "B", "text": "hi back", "words": words("hi back", 5.2),
         "excluded": False, "confirmed": False, "text_edited": False, "suggest": None},
        {"id": 2, "start": 9.5, "end": 11.5, "speaker": "B", "text": "left out", "words": words("left out", 9.6),
         "excluded": True, "confirmed": False, "text_edited": False, "suggest": None},
    ]
    p = {"duration": 12.0, "language": "ja", "speakers": [
        {"id": "A", "name": "Host", "color": "#fff", "export": True},
        {"id": "B", "name": "host", "color": "#000", "export": True}], "turns": turns}
    return d, p


def test_same_names_get_their_own_folders_and_lists_stay_one_line(tmp_path):
    d, p = _project(tmp_path)
    summary = core.export_project(d, p, language="auto", min_len=1.0)
    assert sorted(summary) == ["Host", "host_2"]
    line = (d / "export" / "Host" / "dataset.list").read_text(encoding="utf-8").strip()
    assert line.count("|") == 3 and line.endswith("|Host|ja|hello there friend")  # detected language, no stray |


def test_export_writes_the_whole_transcript(tmp_path):
    d, p = _project(tmp_path)
    core.export_project(d, p, min_len=1.0)
    txt = (d / "export" / "transcript.txt").read_text(encoding="utf-8")
    assert "[0:00] Host:" in txt and "[0:05] host:" in txt and "left out" not in txt
    srt = (d / "export" / "transcript.srt").read_text(encoding="utf-8")
    assert "00:00:05,000 --> 00:00:09,000" in srt and "host: hi back" in srt


def test_only_answers_requests_for_this_pc(tmp_path):
    c = create_app(root=tmp_path).test_client()
    assert c.get("/api/version").status_code == 200
    assert c.get("/api/version", base_url="http://127.0.0.1:7861").status_code == 200
    assert c.get("/api/library", base_url="http://evil.example:7861").status_code == 403


def test_friendly_errors():
    assert "Setup" in friendly_error(RuntimeError("401 Client Error: gated repo"))
    assert "smaller transcription model" in friendly_error(RuntimeError("CUDA out of memory"))
    assert friendly_error(ValueError("plain")) == "plain"


def test_bad_speaker_count_is_a_400_not_a_crash(tmp_path):
    app = create_app(root=tmp_path)
    f = tmp_path / "inbox" / "a.wav"
    sf.write(str(f), np.zeros(core.SR * 2, dtype="float32"), core.SR)
    r = app.test_client().post("/api/split", json={"path": str(f), "speakers": "two"})
    assert r.status_code == 400


def test_unreadable_audio_says_so(tmp_path):
    bad = tmp_path / "broken.mp3"
    bad.write_bytes(b"not audio" * 500)
    (tmp_path / "proj").mkdir()
    try:
        core.prepare_audio(bad, tmp_path / "proj", core.find_ffmpeg())
    except RuntimeError as e:
        assert "Couldn't read the audio in broken.mp3" in str(e)
    else:
        raise AssertionError("expected a RuntimeError")
