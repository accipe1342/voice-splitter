"""The Voice Splitter web app: library, background jobs, review and export."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
from pathlib import Path

from flask import redirect, Flask, abort, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from . import core

ROOT = Path(__file__).resolve().parent.parent          # voice_splitter/
STATIC = Path(__file__).resolve().parent / "static"


from splitter import setup as setup_mod  # noqa: E402
WHISPER_MODELS = list(setup_mod.WHISPER)
# Whisper handles ~100 languages; these are the common ones (codes are what Whisper and the export use).
LANGUAGES = {"auto": "Auto-detect", "en": "English", "es": "Spanish", "fr": "French", "de": "German",
             "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish", "ru": "Russian",
             "uk": "Ukrainian", "tr": "Turkish", "ar": "Arabic", "hi": "Hindi", "vi": "Vietnamese",
             "th": "Thai", "id": "Indonesian", "tl": "Tagalog", "ja": "Japanese", "ko": "Korean",
             "zh": "Chinese (Mandarin)", "yue": "Cantonese"}


def default_settings() -> dict:
    return {
        "folders": [], "speakers": 2, "language": "en",
        "whisper_model": os.environ.get("WHISPER_MODEL", "turbo"),
        "device": "auto",
        "min_clip": 2.0, "max_clip": 12.0, "edge_pad": 0.08, "preview_seconds": 25,
        "vp_margin": 0.08, "vp_max_len": 4.0,
        "window": "app",
        "setup_done": False,
    }


class Settings:
    def __init__(self, path: Path):
        self.path = path
        self.data = default_settings()
        if path.exists():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                saved.setdefault("setup_done", True)  # set up before the setup page existed
                self.data.update(saved)
            except Exception:
                pass

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def update(self, incoming: dict) -> list[str]:
        """Validate and apply edits from the settings page. Returns problems, if any."""
        d, errs = self.data, []

        def num(key, lo, hi, cast=float):
            if key not in incoming:
                return
            try:
                v = cast(incoming[key])
            except (TypeError, ValueError):
                errs.append(f"{key}: not a number")
                return
            d[key] = max(lo, min(hi, v))

        if "speakers" in incoming:
            v = incoming["speakers"]
            d["speakers"] = "auto" if v in ("auto", "", None) else max(1, min(9, int(v)))
        if incoming.get("language") in LANGUAGES:
            d["language"] = incoming["language"]
        if incoming.get("whisper_model") in WHISPER_MODELS:
            d["whisper_model"] = incoming["whisper_model"]
        if incoming.get("device") in ("auto", "cuda", "cpu"):
            d["device"] = incoming["device"]
        if incoming.get("window") in ("app", "browser"):
            d["window"] = incoming["window"]
        num("min_clip", 0.5, 30)
        num("max_clip", 2, 60)
        num("edge_pad", 0, 0.5)
        num("preview_seconds", 5, 120, int)
        num("vp_margin", 0.01, 0.5)
        num("vp_max_len", 1, 30)
        if d["max_clip"] < d["min_clip"] + 1:
            d["max_clip"] = d["min_clip"] + 1
        self.save()
        return errs


def create_app(*, root: Path = ROOT, extra_folders=(), diarize_fn=None, transcribe_fn=None,
               embed_fn=None, ffmpeg=None, download_fn=None, checks_fn=None) -> Flask:
    projects_dir = root / "projects"
    inbox = root / "inbox"
    secrets = root / "secrets"
    for d in (projects_dir, inbox, secrets):
        d.mkdir(parents=True, exist_ok=True)
    settings = Settings(root / "settings.json")
    token_file = secrets / "hf_token.txt"

    def token_source() -> str:
        if os.environ.get("HF_TOKEN"):
            return "env"
        return "file" if token_file.exists() and token_file.read_text(encoding="utf-8").strip() else "none"

    def token():
        if os.environ.get("HF_TOKEN"):
            return os.environ["HF_TOKEN"]
        return token_file.read_text(encoding="utf-8").strip() if token_file.exists() else None

    def apply_runtime():
        core.RUNTIME["whisper_model"] = settings.data["whisper_model"]
        core.RUNTIME["device"] = settings.data["device"]
    apply_runtime()

    gpu_info: dict = {"checked": False}

    def check_gpu():
        try:
            import torch
            ok = torch.cuda.is_available()
            gpu_info.update(available=ok, name=torch.cuda.get_device_name(0) if ok else None,
                            torch=torch.__version__)
        except Exception as e:
            gpu_info.update(available=False, name=None, error=str(e)[:200])
        gpu_info["checked"] = True

    def folders() -> list[Path]:
        out = [inbox]
        ytdip = root.parent / "downloads"                 # YTdip's downloads, if we live inside it
        if ytdip.is_dir():
            out.append(ytdip)
        for f in list(extra_folders) + settings.data["folders"]:
            p = Path(f).expanduser()
            if p.is_dir() and p not in out:
                out.append(p)
        return [p.resolve() for p in out]

    def folder_rows() -> list[dict]:
        rows = [{"path": str(inbox.resolve()), "kind": "inbox", "remove": None}]
        ytdip = root.parent / "downloads"
        if ytdip.is_dir():
            rows.append({"path": str(ytdip.resolve()), "kind": "ytdip", "remove": None})
        for f in settings.data["folders"]:
            p = Path(f).expanduser()
            rows.append({"path": str(p.resolve()) if p.is_dir() else f, "kind": "custom",
                         "missing": not p.is_dir(), "remove": f})
        return rows

    def allowed(path: Path) -> bool:
        path = path.resolve()
        return path.is_file() and path.suffix.lower() in core.MEDIA_EXT and \
            any(path.is_relative_to(f) for f in folders())

    def project_id(path: Path) -> str:
        st = path.stat()
        h = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:8]
        stem = "".join(c if c.isalnum() else "-" for c in path.stem)[:40].strip("-").lower() or "audio"
        return f"{stem}-{h}"

    def proj_dir(pid: str) -> Path:
        if not pid or "/" in pid or "\\" in pid or ".." in pid:
            abort(400)
        d = projects_dir / pid
        if not (d / "project.json").exists():
            abort(404)
        return d

    # ---- jobs: one at a time, since the models are heavy -----------------
    jobs: dict[str, dict] = {}
    work: "queue.Queue[str]" = queue.Queue()

    def runner():
        while True:
            jid = work.get()
            job = jobs[jid]
            job["state"] = "running"
            try:
                apply_runtime()

                def progress(msg):
                    job["log"].append(msg)
                    job["step"] = msg
                if job.get("kind") == "resplit":
                    core.resplit(projects_dir / job["project"], start=job["start"], end=job["end"],
                                 num_speakers=job["speakers"], token=token(), language=job["language"],
                                 retranscribe=job["retranscribe"], keep_checked=job["keep_checked"],
                                 progress=progress, diarize_fn=diarize_fn, transcribe_fn=transcribe_fn)
                else:
                    core.analyze(Path(job["path"]), projects_dir / job["project"],
                                 num_speakers=job["speakers"], token=token(), language=job["language"],
                                 ffmpeg=ffmpeg, progress=progress,
                                 diarize_fn=diarize_fn, transcribe_fn=transcribe_fn)
                job["state"] = "done"
            except Exception as e:
                traceback.print_exc()
                job["state"], job["error"] = "error", friendly_error(e)
                if job.get("kind") != "resplit":  # a failed re-split leaves your project as it was
                    shutil.rmtree(projects_dir / job["project"], ignore_errors=True)
    threading.Thread(target=runner, daemon=True, name="splitter-jobs").start()

    app = Flask(__name__, static_folder=None)

    @app.before_request
    def local_only():
        # Only answer pages addressed to this PC. A website can't use a DNS trick
        # to point its own name at 127.0.0.1 and then read or change your files.
        host = (request.host or "").rsplit(":", 1)[0].strip("[]").lower()
        if host not in ("127.0.0.1", "localhost", "::1"):
            abort(403)
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024
    app.config["JOBS"] = jobs

    @app.get("/")
    def home():
        if not settings.data.get("setup_done") and request.args.get("skip") is None:
            return redirect("/setup")
        return send_from_directory(STATIC, "library.html")

    # ---- first-run setup ------------------------------------------------
    downloads = setup_mod.Downloads(download_fn)

    @app.get("/api/version")
    def version():
        from splitter import __version__
        return jsonify({"version": __version__})

    @app.get("/setup")
    def setup_page():
        return send_from_directory(STATIC, "setup.html")

    @app.get("/api/setup")
    def setup_status():
        checks, gpu = (checks_fn or (lambda: setup_mod.system_checks(lambda: core.find_ffmpeg(ffmpeg))))()
        has = download_fn is None  # tests fake the downloads, so nothing is "already there"
        models = {k: {**v, "downloaded": has and setup_mod.is_downloaded(v["repo"]),
                      "job": downloads.status(f"whisper:{k}")}
                  for k, v in setup_mod.WHISPER.items()}
        return jsonify({
            "checks": checks, "gpu": gpu,
            "speaker": {"repo": setup_mod.SPEAKER_REPO, "page": setup_mod.SPEAKER_PAGE,
                        "token_page": setup_mod.TOKEN_PAGE, "token": token_source(),
                        "downloaded": has and setup_mod.is_downloaded(setup_mod.SPEAKER_REPO),
                        "job": downloads.status("speaker")},
            "whisper": {"models": models, "chosen": settings.data["whisper_model"],
                        "recommended": setup_mod.recommend(gpu)},
            "done": bool(settings.data.get("setup_done")),
        })

    @app.post("/api/setup/download")
    def setup_download():
        body = request.get_json(silent=True) or {}
        what = body.get("model", "")
        if what == "speaker":
            if not token():
                abort(400, "Add your Hugging Face token first.")
            return jsonify(downloads.start("speaker", setup_mod.SPEAKER_REPO, token(), 0.04))
        name = what.removeprefix("whisper:")
        m = setup_mod.WHISPER.get(name) or abort(400, "Unknown model.")
        settings.data["whisper_model"] = name
        settings.save()
        apply_runtime()
        return jsonify(downloads.start(f"whisper:{name}", m["repo"], None, m["gb"]))

    @app.get("/api/setup/download/<path:key>")
    def setup_download_status(key):
        return jsonify(downloads.status(key) or {"key": key, "state": "idle"})

    @app.post("/api/setup/complete")
    def setup_complete():
        settings.data["setup_done"] = True
        settings.save()
        return jsonify({"ok": True})

    @app.get("/review/<pid>")
    def review(pid):
        proj_dir(pid)
        return send_from_directory(STATIC, "review.html")

    @app.get("/settings")
    def settings_page():
        return send_from_directory(STATIC, "settings.html")

    @app.get("/static/<path:name>")
    def static_file(name):
        return send_from_directory(STATIC, name)

    # ---- library ------------------------------------------------------
    def review_of(pid):
        d = projects_dir / pid
        if not (d / "project.json").exists():
            return None
        try:
            return core.review_stage(d)
        except Exception:
            return None

    @app.get("/api/library")
    def library():
        files = []
        for f in folders():
            entries = []
            for p in f.iterdir():
                try:  # a file that's locked, still downloading or vanishing shouldn't break the list
                    if p.is_file() and p.suffix.lower() in core.MEDIA_EXT:
                        entries.append((p, p.stat()))
                except OSError:
                    continue
            for p, st in sorted(entries, key=lambda e: e[1].st_mtime, reverse=True):
                if st.st_size > 2048:
                    pid = project_id(p)
                    files.append({"path": str(p), "name": p.name, "folder": str(f), "pid": pid,
                                  "size": st.st_size, "mtime": st.st_mtime,
                                  "project": pid if (projects_dir / pid / "project.json").exists() else None,
                                  "exported": (projects_dir / pid / "export").exists(),
                                  "review": review_of(pid),
                                  "video": p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".avi"}})
        running = {j["project"]: {"job": jid, "state": j["state"], "step": j["step"]}
                   for jid, j in jobs.items() if j["state"] in ("queued", "running")}
        try:
            core.find_ffmpeg(ffmpeg)
            ff = True
        except RuntimeError:
            ff = False
        return jsonify({"folders": [str(f) for f in folders()], "files": files, "running": running,
                        "has_token": bool(token()), "ffmpeg": ff, "inbox": str(inbox),
                        "defaults": {"speakers": settings.data["speakers"],
                                     "language": settings.data["language"]}})

    @app.post("/api/upload")
    def upload():
        saved = []
        for f in request.files.getlist("files"):
            name = secure_filename(f.filename or "") or f"upload-{uuid.uuid4().hex[:6]}"
            if Path(name).suffix.lower() not in core.MEDIA_EXT:
                continue
            dest = inbox / name
            i = 1
            while dest.exists():
                dest = inbox / f"{Path(name).stem}-{i}{Path(name).suffix}"
                i += 1
            f.save(dest)
            saved.append(str(dest))
        if not saved:
            abort(400, "No audio or video files in that upload.")
        return jsonify({"saved": saved})

    @app.post("/api/folders")
    def add_folder():
        p = Path((request.get_json(silent=True) or {}).get("path", "")).expanduser()
        if not p.is_dir():
            abort(400, "That folder doesn't exist.")
        if str(p) not in settings.data["folders"]:
            settings.data["folders"].append(str(p))
            settings.save()
        return jsonify({"ok": True})

    @app.delete("/api/folders")
    def remove_folder():
        p = (request.get_json(silent=True) or {}).get("path", "")
        settings.data["folders"] = [f for f in settings.data["folders"] if Path(f) != Path(p)]
        settings.save()
        return jsonify({"ok": True})

    @app.post("/api/token")
    def set_token():
        t = ((request.get_json(silent=True) or {}).get("token") or "").strip()
        if not t.startswith("hf_"):
            abort(400, "Hugging Face tokens start with hf_")
        token_file.write_text(t, encoding="utf-8")
        return jsonify({"ok": True})

    @app.delete("/api/token")
    def clear_token():
        if token_file.exists():
            token_file.unlink()
        return jsonify({"ok": True, "source": token_source()})

    @app.post("/api/token/test")
    def test_token():
        """Check the token can reach each gated model (the 403 you'd otherwise hit mid-split)."""
        t = token()
        if not t:
            return jsonify({"ok": False, "error": "No token set."})
        repos = [os.environ.get("PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1"),
                 os.environ.get("PYANNOTE_EMBEDDING", "pyannote/wespeaker-voxceleb-resnet34-LM")]
        results = []
        try:
            import huggingface_hub as hf
        except ImportError:
            return jsonify({"ok": False, "error": "huggingface_hub isn't installed yet."})
        try:
            who = hf.whoami(token=t).get("name")
        except Exception as e:
            return jsonify({"ok": False, "error": f"Token rejected: {str(e)[:200]}"})
        for repo in repos:
            try:
                if hasattr(hf, "auth_check"):
                    hf.auth_check(repo, token=t)
                else:
                    hf.get_hf_file_metadata(hf.hf_hub_url(repo, "config.yaml"), token=t)
                results.append({"repo": repo, "ok": True})
            except Exception as e:
                msg = str(e)
                hint = "Accept the terms on the model page with this account." if ("403" in msg or "gated" in msg.lower() or "restricted" in msg.lower()) else msg[:200]
                results.append({"repo": repo, "ok": False, "error": hint,
                                "url": f"https://huggingface.co/{repo}"})
        return jsonify({"ok": all(r["ok"] for r in results), "user": who, "models": results})

    # ---- settings -----------------------------------------------------
    @app.get("/api/settings")
    def get_settings():
        if not gpu_info["checked"] and not gpu_info.get("running"):
            gpu_info["running"] = True
            threading.Thread(target=check_gpu, daemon=True).start()
        try:
            ff = core.find_ffmpeg(ffmpeg)
        except RuntimeError:
            ff = None
        t = token()
        projects = []
        for d in sorted(projects_dir.iterdir()) if projects_dir.exists() else []:
            pj = d / "project.json"
            if not pj.exists():
                continue
            try:
                meta = json.loads(pj.read_text(encoding="utf-8"))
                name, created = meta.get("name", d.name), meta.get("created")
            except Exception:
                name, created = d.name, None
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            projects.append({"id": d.name, "name": name, "size": size, "created": created,
                             "exported": (d / "export").exists()})
        return jsonify({
            "settings": settings.data,
            "options": {"whisper_models": [{"id": k, "label": f"{v['label']} ({v['gb']:g} GB)"} for k, v in setup_mod.WHISPER.items()],
                        "languages": LANGUAGES},
            "token": {"source": token_source(),
                      "masked": (t[:5] + "…" + t[-4:]) if t and len(t) > 12 else None},
            "system": {"ffmpeg": ff, "gpu": {k: v for k, v in gpu_info.items() if k != "running"},
                       "python": sys.version.split()[0], "root": str(root),
                       "folders": folder_rows()},
            "projects": projects,
        })

    @app.put("/api/settings")
    def put_settings():
        errs = settings.update(request.get_json(silent=True) or {})
        apply_runtime()
        if errs:
            abort(400, "; ".join(errs))
        return jsonify({"ok": True, "settings": settings.data})

    @app.post("/api/open/<which>")
    def open_dir(which):
        target = {"projects": projects_dir, "inbox": inbox, "root": root}.get(which) or abort(404)
        open_folder(target)
        return jsonify({"ok": True})

    @app.post("/api/split")
    def split():
        body = request.get_json(silent=True) or {}
        path = Path(body.get("path", ""))
        if not allowed(path):
            abort(400, "File isn't in the library.")
        speakers = _speakers(body.get("speakers"))
        language = (body.get("language") or settings.data["language"])[:8]
        settings.data.update({"speakers": speakers or "auto", "language": language})
        settings.save()
        pid = project_id(path)
        for jid, j in jobs.items():
            if j["project"] == pid and j["state"] in ("queued", "running"):
                return jsonify({"job": jid, "project": pid})
        if body.get("redo"):
            shutil.rmtree(projects_dir / pid, ignore_errors=True)
        elif (projects_dir / pid / "project.json").exists():
            return jsonify({"job": None, "project": pid, "existing": True})
        jid = uuid.uuid4().hex[:10]
        jobs[jid] = {"path": str(path), "project": pid, "speakers": speakers, "language": language,
                     "state": "queued", "step": "Waiting", "log": [], "error": None}
        work.put(jid)
        return jsonify({"job": jid, "project": pid})

    @app.get("/api/jobs/<jid>")
    def job(jid):
        j = jobs.get(jid) or abort(404)
        return jsonify({k: j[k] for k in ("project", "state", "step", "log", "error")})

    # ---- projects -----------------------------------------------------
    @app.get("/api/projects/<pid>")
    def get_project(pid):
        d = proj_dir(pid)
        return jsonify({**core.load_project(d), "can_restore": (d / "project.prev.json").exists()})

    @app.post("/api/projects/<pid>/resplit")
    def resplit(pid):
        """Re-run the speaker split on part or all of a project (queued like a split)."""
        d = proj_dir(pid)
        body = request.get_json(silent=True) or {}
        for jid, j in jobs.items():
            if j["project"] == pid and j["state"] in ("queued", "running"):
                abort(409, "This file is already being worked on.")
        speakers = _speakers(body.get("speakers"))
        duration = core.load_project(d)["duration"]
        start = max(0.0, float(body.get("start") or 0))
        end = body.get("end")
        end = duration if end in (None, "") else min(duration, float(end))
        if end - start < 1.0:
            abort(400, "Mark at least one second to re-split.")
        jid = uuid.uuid4().hex[:10]
        jobs[jid] = {"kind": "resplit", "path": "", "project": pid, "speakers": speakers,
                     "language": settings.data["language"], "start": start, "end": end,
                     "retranscribe": bool(body.get("retranscribe")), "keep_checked": body.get("keep_checked", True) is not False,
                     "state": "queued", "step": "Waiting", "log": [], "error": None}
        work.put(jid)
        return jsonify({"job": jid, "project": pid})

    @app.post("/api/projects/<pid>/restore")
    def restore(pid):
        d = proj_dir(pid)
        if not core.restore_previous(d):
            abort(404, "Nothing to restore.")
        return jsonify({**core.load_project(d), "can_restore": True})

    @app.put("/api/projects/<pid>")
    def put_project(pid):
        d = proj_dir(pid)
        incoming = request.get_json(silent=True) or abort(400)
        project = core.sanitize_project(incoming, core.load_project(d))
        core.save_project(d, project)
        return jsonify({"ok": True})

    @app.get("/api/projects/<pid>/peaks")
    def peaks(pid):
        return send_file(proj_dir(pid) / "peaks.json", mimetype="application/json")

    @app.get("/api/projects/<pid>/playback")
    def playback(pid):
        return send_file(proj_dir(pid) / "playback.mp3", mimetype="audio/mpeg", conditional=True)

    @app.post("/api/projects/<pid>/voiceprint")
    def voiceprint(pid):
        d = proj_dir(pid)
        project = core.load_project(d)
        try:
            n = core.voiceprint_check(d, project, token=token(), embed_fn=embed_fn,
                                      margin=float(settings.data["vp_margin"]),
                                      max_len=float(settings.data["vp_max_len"]))
        except ValueError as e:
            abort(400, str(e))
        except Exception as e:
            traceback.print_exc()
            abort(500, f"Voiceprint check failed: {e}")
        core.save_project(d, project)
        return jsonify({"flagged": n, "project": project})

    @app.post("/api/projects/<pid>/export")
    def export(pid):
        d = proj_dir(pid)
        body = request.get_json(silent=True) or {}
        project = core.load_project(d)
        sd = settings.data
        try:
            summary = core.export_project(d, project, language=sd["language"],
                                          min_len=float(body.get("min_seconds", sd["min_clip"])),
                                          max_len=float(body.get("max_seconds", sd["max_clip"])),
                                          pad=float(sd["edge_pad"]),
                                          preview_seconds=float(sd["preview_seconds"]))
        except PermissionError:
            abort(409, "Windows wouldn't let the old export be replaced. Close any clips or the export "
                       "folder open in another app, then export again.")
        if body.get("open", True):
            open_folder(d / "export")
        return jsonify({"summary": summary, "folder": str((d / "export").resolve())})

    @app.post("/api/projects/<pid>/open")
    def open_export(pid):
        d = proj_dir(pid)
        open_folder(d / "export" if (d / "export").exists() else d)
        return jsonify({"ok": True})

    @app.delete("/api/projects/<pid>")
    def delete_project(pid):
        shutil.rmtree(proj_dir(pid), ignore_errors=True)
        return jsonify({"ok": True})

    return app


def friendly_error(e: Exception) -> str:
    """Turn the errors people actually hit into something they can act on."""
    msg = str(e) or e.__class__.__name__
    low = msg.lower()
    if "401" in msg or "403" in msg or "gated" in low or "restricted" in low:
        return ("Hugging Face refused the speaker model. Open Setup: check your token and accept the "
                "model's terms with the same account.")
    if "out of memory" in low or "cuda oom" in low:
        return ("The graphics card ran out of memory. Pick a smaller transcription model in Settings, "
                "or set Run on to CPU, then try again.")
    if isinstance(e, (ConnectionError, TimeoutError)) or "connection" in low and "huggingface" in low:
        return "Couldn't reach Hugging Face to download a model. Check your internet connection and try again."
    return msg[:500]


def _speakers(v):
    """Speaker count from a request: a number from 1 to 9, or None for auto."""
    if v in (None, "", "auto"):
        return None
    try:
        return max(1, min(9, int(v)))
    except (TypeError, ValueError):
        abort(400, "Speaker count must be a number or auto.")


def open_folder(path: Path):
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


def open_app_window(url: str):
    """Open as its own window (Edge/Chrome app mode); fall back to a browser tab."""
    candidates = [shutil.which("msedge"), shutil.which("chrome"),
                  r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    for c in candidates:
        if c and Path(c).exists():
            try:
                subprocess.Popen([c, f"--app={url}", "--window-size=1400,900"])
                return
            except Exception:
                pass
    import webbrowser
    webbrowser.open(url)
