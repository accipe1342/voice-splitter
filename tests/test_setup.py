"""First-run setup: the PC check, token, model downloads (all faked)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from splitter import setup as S  # noqa: E402
from splitter.server import create_app  # noqa: E402

GPU = {"torch": "2.8", "cuda_build": "12.8", "available": True, "name": "RTX 5070", "vram_gb": 12.0, "error": None}


def fake_checks():
    return [S._check("python", "Python", True, "3.11.9"), S._check("gpu", "Graphics card", True, "RTX 5070")], GPU


def app(tmp_path, **kw):
    got = []
    a = create_app(root=tmp_path, download_fn=lambda repo, token: (time.sleep(0.2), got.append((repo, token))),
                   checks_fn=fake_checks, **kw)
    return a.test_client(), got


def wait(c, key):
    for _ in range(50):
        j = c.get(f"/api/setup/download/{key}").get_json()
        if j["state"] != "running":
            return j
        time.sleep(0.05)
    raise AssertionError("download never finished")


def test_fresh_install_opens_setup_and_recommends_for_the_gpu(tmp_path):
    c, _ = app(tmp_path)
    r = c.get("/")
    assert r.status_code == 302 and r.headers["Location"].endswith("/setup")
    assert c.get("/?skip").status_code == 200
    s = c.get("/api/setup").get_json()
    assert s["whisper"]["recommended"] == "turbo" and not s["done"]
    assert s["speaker"]["token"] == "none" and set(s["whisper"]["models"]) == set(S.WHISPER)


def test_speaker_needs_a_token_then_downloads_with_it(tmp_path):
    c, got = app(tmp_path)
    assert c.post("/api/setup/download", json={"model": "speaker"}).status_code == 400
    c.post("/api/token", json={"token": "hf_test"})
    c.post("/api/setup/download", json={"model": "speaker"})
    assert wait(c, "speaker")["state"] == "done"
    assert got == [(S.SPEAKER_REPO, "hf_test")]


def test_choosing_a_transcription_model_downloads_and_saves_it(tmp_path):
    c, got = app(tmp_path)
    c.post("/api/setup/download", json={"model": "whisper:small"})
    assert wait(c, "whisper:small")["state"] == "done"
    assert got[0][0] == S.WHISPER["small"]["repo"]
    assert c.get("/api/settings").get_json()["settings"]["whisper_model"] == "small"
    assert c.post("/api/setup/download", json={"model": "whisper:huge"}).status_code == 400
    c.post("/api/setup/complete")
    assert c.get("/").status_code == 200  # setup done: straight to the library


def test_existing_installs_skip_setup(tmp_path):
    (tmp_path / "settings.json").write_text('{"speakers": 2}')
    c, _ = app(tmp_path)
    assert c.get("/").status_code == 200


def test_recommendation_follows_memory():
    assert S.recommend({"available": True, "vram_gb": 12}) == "turbo"
    assert S.recommend({"available": True, "vram_gb": 4}) == "medium"
    assert S.recommend({"available": False}) == "small"
