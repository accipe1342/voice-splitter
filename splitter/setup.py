"""First-run setup: check the PC, connect the speaker model, pick and download
the transcription model. Everything here is read-only except the downloads,
which go to the normal Hugging Face cache (shared with other apps)."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

SPEAKER_REPO = os.environ.get("PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1")
SPEAKER_PAGE = f"https://huggingface.co/{SPEAKER_REPO}"
TOKEN_PAGE = "https://huggingface.co/settings/tokens"

# faster-whisper names -> repo, rough size (GB), and the memory they want.
WHISPER = {
    "base": {"repo": "Systran/faster-whisper-base", "gb": 0.15, "vram": 1,
             "label": "Base", "note": "Fastest, least accurate. Fine for clear speech on a slow PC."},
    "small": {"repo": "Systran/faster-whisper-small", "gb": 0.48, "vram": 2,
              "label": "Small", "note": "Good on CPU-only PCs."},
    "medium": {"repo": "Systran/faster-whisper-medium", "gb": 1.5, "vram": 5,
               "label": "Medium", "note": "More accurate; slow without a graphics card."},
    "turbo": {"repo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo", "gb": 1.6, "vram": 6,
              "label": "Large v3 Turbo", "note": "Nearly as accurate as Large v3 and much faster. Best for most GPUs."},
    "large-v3": {"repo": "Systran/faster-whisper-large-v3", "gb": 3.1, "vram": 10,
                 "label": "Large v3", "note": "Most accurate, slowest. Worth it for hard audio."},
}


def hf_cache() -> Path:
    try:
        from huggingface_hub import constants
        return Path(constants.HF_HUB_CACHE)
    except Exception:
        return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def _check(id_, label, ok, detail, fix=None):
    """ok: True (good), False (must fix), "warn" (works, but could be better)."""
    return {"id": id_, "label": label, "ok": ok, "detail": detail, "fix": fix}


def gpu_info() -> dict:
    """What PyTorch can see. Never raises."""
    info = {"torch": None, "cuda_build": None, "available": False, "name": None, "vram_gb": None, "error": None}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_build"] = torch.version.cuda
        info["available"] = bool(torch.cuda.is_available())
        if info["available"]:
            info["name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
    except Exception as e:
        info["error"] = str(e)[:200]
    return info


def nvidia_present() -> str | None:
    """Name of an NVIDIA card from the driver, even if PyTorch can't use it."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run([smi, "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


def reachable(url="https://huggingface.co", timeout=5) -> bool:
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout)
        return True
    except Exception as e:
        # an HTTP error still means we got there
        return hasattr(e, "code")


def system_checks(ffmpeg_finder) -> tuple[list[dict], dict]:
    checks = []
    v = sys.version_info
    checks.append(_check("python", "Python", v >= (3, 10), f"{v.major}.{v.minor}.{v.micro}",
                         None if v >= (3, 10) else "Install Python 3.11 from python.org, then run Start Voice Splitter.bat again."))
    checks.append(_check("os", "System", True, f"{platform.system()} {platform.release()}"))

    g = gpu_info()
    card = nvidia_present()
    if g["available"]:
        checks.append(_check("gpu", "Graphics card", True, f"{g['name']}, {g['vram_gb']} GB (CUDA {g['cuda_build']})"))
    elif card and g["torch"] and not g["cuda_build"]:
        checks.append(_check("gpu", "Graphics card", "warn", f"{card} found, but the CPU-only PyTorch is installed",
                             "Delete the .venv folder and run Start Voice Splitter.bat again to install the GPU version."))
    elif card:
        checks.append(_check("gpu", "Graphics card", "warn", f"{card} found, but PyTorch can't use it ({g['error'] or 'driver too old?'})",
                             "Update the NVIDIA driver (nvidia.com/drivers), then restart the app."))
    else:
        checks.append(_check("gpu", "Graphics card", "warn", "No NVIDIA card. It will run on the processor (slower).", None))
    if not g["torch"]:
        checks.append(_check("torch", "PyTorch", False, g["error"] or "not installed",
                             "Run Start Voice Splitter.bat; it installs PyTorch."))

    try:
        ff = ffmpeg_finder()
        checks.append(_check("ffmpeg", "ffmpeg (reads audio and video)", True, ff))
    except RuntimeError:
        checks.append(_check("ffmpeg", "ffmpeg (reads audio and video)", False, "not found",
                             "Run Start Voice Splitter.bat again; it installs a copy."))

    cache = hf_cache()
    probe = cache if cache.exists() else Path.home()
    free = shutil.disk_usage(probe).free / 1024 ** 3
    checks.append(_check("disk", "Free space for models", True if free > 8 else "warn" if free > 4 else False,
                         f"{free:.0f} GB free on the drive with {probe}",
                         None if free > 8 else "Models need up to 4 GB. Free up space, or set HF_HOME to a bigger drive."))
    online = reachable()
    checks.append(_check("net", "Hugging Face (model downloads)", True if online else "warn",
                         "reachable" if online else "can't reach huggingface.co",
                         None if online else "Check your internet connection. Models you already have still work offline."))
    return checks, g


def recommend(gpu: dict) -> str:
    vram = gpu.get("vram_gb") or 0
    if gpu.get("available") and vram >= 6:
        return "turbo"
    if gpu.get("available") and vram >= 4:
        return "medium"
    return "small"


def is_downloaded(repo: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:
        return False


def _repo_bytes(repo: str) -> int:
    d = hf_cache() / ("models--" + repo.replace("/", "--"))
    if not d.exists():
        return 0
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file() and not f.is_symlink())


class Downloads:
    """One background download per model, with progress read from the cache."""

    def __init__(self, download_fn=None):
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.download_fn = download_fn

    def _total(self, repo, token, fallback_gb):
        try:
            import huggingface_hub as hf
            info = hf.model_info(repo, files_metadata=True, token=token)
            return sum((s.size or 0) for s in info.siblings) or int(fallback_gb * 1e9)
        except Exception:
            return int(fallback_gb * 1e9)

    def start(self, key: str, repo: str, token: str | None, fallback_gb: float) -> dict:
        with self.lock:
            j = self.jobs.get(key)
            if j and j["state"] == "running":
                return j
            j = self.jobs[key] = {"key": key, "repo": repo, "state": "running", "error": None,
                                  "done": 0, "total": int(fallback_gb * 1e9), "started": time.time()}

        def run():
            try:
                j["total"] = self._total(repo, token, fallback_gb)
                if self.download_fn:
                    self.download_fn(repo, token)
                else:
                    from huggingface_hub import snapshot_download
                    snapshot_download(repo, token=token)
                j["state"], j["done"] = "done", j["total"]
            except Exception as e:
                msg = str(e)
                if "403" in msg or "gated" in msg.lower() or "401" in msg:
                    msg = "Hugging Face refused the download. Accept the model's terms on its page and check your token."
                j["state"], j["error"] = "error", msg[:300]
        threading.Thread(target=run, daemon=True, name=f"download-{key}").start()
        return j

    def status(self, key: str) -> dict | None:
        j = self.jobs.get(key)
        if j and j["state"] == "running":
            j["done"] = min(j["total"], _repo_bytes(j["repo"]))  # includes *.incomplete blobs in flight
        return j
