"""Analysis and export: who speaks when, what they said, and the clips.

A *project* is one analyzed recording, stored as  projects/<id>/ :
    audio.wav       44.1 kHz mono, used for cutting clips
    playback.mp3    small copy the review page streams
    peaks.json      waveform outline for the review page
    project.json    turns (start, end, speaker, text, words, flags) -- what you edit
    export/         the clips, written by export_project()

Edits never touch your original file or the audio; they only change
project.json. Export re-cuts everything from it.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
MEDIA_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
             ".mp4", ".mkv", ".webm", ".mov", ".avi"}
COLORS = ["#e8b04f", "#6cb4e4", "#b89be8", "#84cf8f", "#e5776d", "#d9d36f", "#e08fbf", "#8f9be0"]


# ------------------------------------------------------------------ ffmpeg
def find_ffmpeg(explicit: str | None = None) -> str:
    here = Path(__file__).resolve().parent.parent          # voice_splitter/
    candidates = [explicit, os.environ.get("FFMPEG"), shutil.which("ffmpeg"),
                  here / "ffmpeg.exe", here.parent / "ffmpeg.exe",   # YTdip's copy
                  here / "ffmpeg", here.parent / "ffmpeg"]
    for c in candidates:
        if c and Path(c).is_file():
            return str(c)
        if c and shutil.which(str(c)):
            return shutil.which(str(c))
    try:  # the copy that ships with the imageio-ffmpeg package (installed by the setup)
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise RuntimeError("ffmpeg not found. Run Start Voice Splitter.bat again (it installs a copy), "
                       "pass --ffmpeg, or install it:  winget install Gyan.FFmpeg")


def prepare_audio(src: Path, proj_dir: Path, ffmpeg: str) -> float:
    """Make audio.wav (for cutting) and playback.mp3 (for the page). Returns duration."""
    wav, mp3 = proj_dir / "audio.wav", proj_dir / "playback.mp3"
    for cmd in ([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR), str(wav)],
                [ffmpeg, "-y", "-loglevel", "error", "-i", str(wav), "-b:a", "64k", str(mp3)]):
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode != 0:
            why = (r.stderr or "").strip().splitlines()[-1:] or ["no details"]
            raise RuntimeError(f"Couldn't read the audio in {src.name}. Is it a complete audio or video file? "
                               f"(ffmpeg: {why[0][:200]})")
    duration = sf.info(str(wav)).duration
    if duration < 1:
        raise RuntimeError(f"{src.name} has no audio, or less than a second of it.")
    return duration


def write_peaks(wav: Path, out: Path, per_second: int = 50) -> None:
    data, sr = sf.read(str(wav), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    step = max(1, sr // per_second)
    n = len(data) // step
    peaks = np.abs(data[:n * step]).reshape(n, step).max(axis=1) if n else np.zeros(0)
    top = float(peaks.max()) if len(peaks) else 1.0
    if top > 0:
        peaks = peaks / top
    out.write_text(json.dumps({"duration": len(data) / sr,
                               "peaks": [round(float(p), 3) for p in peaks]}))


# ------------------------------------------------------------------ models
def load_audio_tensor(wav: Path):
    import torch
    audio, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    return {"waveform": torch.from_numpy(audio.T.copy()), "sample_rate": sr}


# Set from the settings page before each job: whisper model and device ("auto"/"cuda"/"cpu").
RUNTIME = {"whisper_model": os.environ.get("WHISPER_MODEL", "large-v3"), "device": "auto"}


def _device():
    try:
        import torch
        pref = RUNTIME.get("device", "auto")
        if pref == "cpu":
            return torch.device("cpu")
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    except Exception:
        return None


_pipelines = {}


def diarize(wav: Path, num_speakers: int | None, token: str | None) -> list[tuple[float, float, str]]:
    from pyannote.audio import Pipeline
    model = os.environ.get("PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1")
    dev = _device()
    pipe = _pipelines.get((model, str(dev)))
    if pipe is None:
        pipe = Pipeline.from_pretrained(model, token=token) if token else Pipeline.from_pretrained(model)
        if pipe is None:
            raise RuntimeError("Couldn't load the speaker model. Open Setup, accept the model's terms on "
                               "Hugging Face and add your token.")
        if dev is not None:
            pipe.to(dev)
        _pipelines[(model, str(dev))] = pipe
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    out = pipe(load_audio_tensor(wav), **kwargs)
    ann = getattr(out, "exclusive_speaker_diarization", None) or getattr(out, "speaker_diarization", out)
    turns = []
    if hasattr(ann, "itertracks"):
        for seg, _t, spk in ann.itertracks(yield_label=True):
            turns.append((float(seg.start), float(seg.end), str(spk)))
    else:
        for seg, spk in ann:
            turns.append((float(seg.start), float(seg.end), str(spk)))
    return sorted(turns)


_whisper = {}
LAST_LANGUAGE = {"code": None}  # what Whisper heard, when the language is set to auto-detect


def transcribe_words(wav: Path, language: str | None) -> list[dict]:
    """Whole-file transcription with word timestamps: [{"w", "s", "e"}]."""
    from faster_whisper import WhisperModel
    dev = _device()
    cuda = dev is not None and dev.type == "cuda"
    key = (RUNTIME.get("whisper_model") or "large-v3", cuda)
    if key not in _whisper:
        _whisper[key] = WhisperModel(key[0], device="cuda" if cuda else "cpu",
                                     compute_type="float16" if cuda else "int8")
    segs, info = _whisper[key].transcribe(str(wav), language=None if language in (None, "", "auto") else language,
                                       word_timestamps=True, vad_filter=True)
    LAST_LANGUAGE["code"] = getattr(info, "language", None)
    words = []
    for s in segs:
        for w in s.words or []:
            words.append({"w": w.word.strip(), "s": round(float(w.start), 3), "e": round(float(w.end), 3)})
    return words


# ------------------------------------------------------------------ building turns
def merge_turns(turns, gap: float = 0.35):
    out: list[list] = []
    for s, e, spk in sorted(turns):
        if out and out[-1][2] == spk and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e, spk])
    return [tuple(t) for t in out]


def assign_words(turns, words):
    """Give each word to the turn containing its midpoint (or the nearest one)."""
    buckets = [[] for _ in turns]
    if not turns:
        return buckets
    j = 0
    for w in words:
        mid = (w["s"] + w["e"]) / 2
        while j + 1 < len(turns) and turns[j][1] < mid and turns[j + 1][0] <= mid:
            j += 1
        best = j
        if not (turns[j][0] <= mid <= turns[j][1]):  # in a gap: nearest edge
            cands = [k for k in (j - 1, j, j + 1) if 0 <= k < len(turns)]
            best = min(cands, key=lambda k: min(abs(mid - turns[k][0]), abs(mid - turns[k][1])))
        buckets[best].append(w)
    return buckets


def build_project(source: Path, duration: float, turns, words, num_speakers) -> dict:
    turns = merge_turns(turns)
    buckets = assign_words(turns, words)
    labels = sorted({t[2] for t in turns})
    speakers = [{"id": l, "name": f"Speaker {i + 1}", "color": COLORS[i % len(COLORS)], "export": True}
                for i, l in enumerate(labels)]
    rows = []
    for i, ((s, e, spk), ws) in enumerate(zip(turns, buckets)):
        rows.append({"id": i, "start": round(s, 3), "end": round(e, 3), "speaker": spk,
                     "text": " ".join(w["w"] for w in ws).strip(), "words": ws, "text_edited": False,
                     "excluded": False, "confirmed": False, "suggest": None})
    return {"version": 1, "source": str(source), "name": source.name, "duration": duration,
            "num_speakers": num_speakers, "created": time.strftime("%Y-%m-%d %H:%M"),
            "speakers": speakers, "turns": rows, "words": list(words)}


def analyze(source: Path, proj_dir: Path, *, num_speakers=None, token=None, language="en",
            ffmpeg=None, progress=lambda msg: None, diarize_fn=None, transcribe_fn=None) -> dict:
    proj_dir.mkdir(parents=True, exist_ok=True)
    progress("Extracting audio")
    duration = prepare_audio(source, proj_dir, find_ffmpeg(ffmpeg))
    write_peaks(proj_dir / "audio.wav", proj_dir / "peaks.json")
    progress("Finding who speaks when (first run downloads the model)")
    turns = (diarize_fn or diarize)(proj_dir / "audio.wav", num_speakers, token)
    progress("Transcribing")
    try:
        words = (transcribe_fn or transcribe_words)(proj_dir / "audio.wav", language)
    except ImportError:
        progress("faster-whisper not installed; continuing without a transcript")
        words = []
    project = build_project(source, duration, turns, words, num_speakers)
    project["language"] = language if language not in (None, "", "auto") else (LAST_LANGUAGE["code"] or "en")
    save_project(proj_dir, project)
    progress("Done")
    return project


# Windows refuses to replace a file another thread has open for reading, and
# antivirus or the search indexer can hold it for a moment too. So reads and
# writes take turns, and the final rename retries briefly before giving up.
_io_lock = threading.RLock()


def load_project(proj_dir: Path) -> dict:
    with _io_lock:
        return json.loads((proj_dir / "project.json").read_text(encoding="utf-8"))


def save_project(proj_dir: Path, project: dict) -> None:
    with _io_lock:
        tmp = proj_dir / "project.json.tmp"
        tmp.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")
        for attempt in range(20):
            try:
                os.replace(tmp, proj_dir / "project.json")
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05 * (attempt + 1))


def _even_words(text: str, s: float, e: float) -> list[dict]:
    toks = text.split()
    if not toks:
        return []
    step = (e - s) / len(toks)
    return [{"w": w, "s": round(s + i * step, 3), "e": round(s + (i + 1) * step, 3)}
            for i, w in enumerate(toks)]


def clean_words(words: list, duration: float, limit: int = 200_000) -> list[dict] | None:
    """A transcript you corrected in the script view, checked before it's kept:
    every word has text and times inside the recording. Returns None if it
    doesn't hold up, so the stored transcript stays."""
    if len(words) > limit:
        return None
    out = []
    for w in words:
        try:
            text = str(w["w"]).strip()[:100]
            s, e = float(w["s"]), float(w["e"])
        except (KeyError, TypeError, ValueError):
            return None
        if not text or not (0 <= s <= e <= duration + 0.5):
            return None
        out.append({"w": text, "s": round(s, 3), "e": round(min(e, duration), 3)})
    return sorted(out, key=lambda w: w["s"])


def sanitize_project(incoming: dict, current: dict) -> dict:
    """Accept edits from the page, but only to the fields that are editable.

    Turns may be resized, relabeled, excluded, confirmed, split (new ids),
    merged or removed. Each turn's words are re-derived from the original
    transcript by time, unless you typed a correction (text_edited)."""
    spk_ids = {s["id"] for s in incoming.get("speakers", current["speakers"])}
    speakers = []
    for s in incoming.get("speakers", current["speakers"]):
        speakers.append({"id": str(s["id"])[:40], "name": str(s.get("name") or s["id"])[:60],
                         "color": str(s.get("color") or COLORS[len(speakers) % len(COLORS)])[:20],
                         "export": bool(s.get("export", True))})
    dur = float(current["duration"])
    by_id = {t["id"]: t for t in current["turns"]}
    all_words = clean_words(incoming.get("words"), dur) if isinstance(incoming.get("words"), list) else None
    if all_words is None:  # no transcript sent (or it didn't check out): keep the stored one
        all_words = current.get("words") or [w for t in current["turns"] for w in t["words"]]
    default_spk = next(iter(spk_ids), None)
    turns, seen = [], set()
    for t in incoming.get("turns", []):
        tid = t.get("id")
        if not isinstance(tid, int) or tid in seen:
            continue
        seen.add(tid)
        base = by_id.get(tid) or {"id": tid, "speaker": t.get("speaker", default_spk), "text": "",
                                  "excluded": False, "confirmed": False, "text_edited": False}
        s = max(0.0, min(float(t.get("start", base.get("start", 0))), dur))
        e = max(s, min(float(t.get("end", base.get("end", s))), dur))
        spk = t.get("speaker", base["speaker"])
        edited = bool(t.get("text_edited", base.get("text_edited", False)))
        if edited:
            text = str(t.get("text", base.get("text", "")))[:5000].strip()
            words = _even_words(text, s, e)
        else:
            words = [w for w in all_words if s <= (w["s"] + w["e"]) / 2 < e]
            text = " ".join(w["w"] for w in words).strip()
        turns.append({"id": tid, "start": round(s, 3), "end": round(e, 3),
                      "speaker": spk if spk in spk_ids else base["speaker"],
                      "text": text, "words": words, "text_edited": edited,
                      "excluded": bool(t.get("excluded", base["excluded"])),
                      "confirmed": bool(t.get("confirmed", base["confirmed"])),
                      "suggest": t.get("suggest") if t.get("suggest") in spk_ids else None})
    turns.sort(key=lambda t: t["start"])
    return {**current, "speakers": speakers, "turns": turns, "words": all_words}


# ------------------------------------------------------------------ voiceprint check
def voiceprint_check(proj_dir: Path, project: dict, *, token=None, margin=0.08, max_len=4.0,
                     embed_fn=None) -> int:
    """Use the turns you've confirmed as each speaker's voiceprint, then flag
    unconfirmed short turns that sound more like someone else. Returns count."""
    confirmed = defaultdict(list)
    for t in project["turns"]:
        if t["confirmed"] and not t["excluded"] and t["end"] - t["start"] >= 1.0:
            confirmed[t["speaker"]].append(t)
    if len(confirmed) < 2:
        raise ValueError("Confirm at least one line for two different speakers first (press C).")
    embed = embed_fn or _make_embedder(proj_dir / "audio.wav", token)
    cents = {}
    for spk, ts in confirmed.items():
        vs = np.stack([embed(t["start"], t["end"]) for t in ts[:20]])
        c = vs.mean(axis=0)
        cents[spk] = c / (np.linalg.norm(c) + 1e-9)
    n = 0
    for t in project["turns"]:
        t["suggest"] = None
        if t["confirmed"] or t["excluded"] or t["end"] - t["start"] > max_len or t["end"] - t["start"] < 0.4:
            continue
        v = embed(t["start"], t["end"])
        v = v / (np.linalg.norm(v) + 1e-9)
        sims = {spk: float(v @ c) for spk, c in cents.items()}
        best = max(sims, key=sims.get)
        if best != t["speaker"] and sims[best] - sims.get(t["speaker"], -1) >= margin:
            t["suggest"] = best
            n += 1
    return n


def _make_embedder(wav: Path, token):
    from pyannote.audio import Inference, Model
    from pyannote.core import Segment
    name = os.environ.get("PYANNOTE_EMBEDDING", "pyannote/wespeaker-voxceleb-resnet34-LM")
    model = Model.from_pretrained(name, token=token) if token else Model.from_pretrained(name)
    inf = Inference(model, window="whole")
    dev = _device()
    if dev is not None:
        inf.to(dev)
    audio = load_audio_tensor(wav)

    def embed(s, e):
        return np.asarray(inf.crop(audio, Segment(s, e))).reshape(-1)
    return embed


# ------------------------------------------------------------------ export
def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\- ]+", "", name).strip().replace(" ", "_") or "speaker"


def plan_clips(project: dict, *, min_len=2.0, max_len=12.0, pad=0.08, gap=0.35):
    """From the edited turns to clips: merge neighbours, trim edges, and split
    long stretches at word boundaries so no word is cut in half."""
    turns = sorted((t for t in project["turns"] if not t["excluded"]), key=lambda t: t["start"])
    groups: list[dict] = []
    for t in turns:
        if groups and groups[-1]["speaker"] == t["speaker"] and t["start"] - groups[-1]["end"] <= gap:
            g = groups[-1]
            g["end"] = max(g["end"], t["end"])
            g["words"] += t["words"]
        else:
            groups.append({"speaker": t["speaker"], "start": t["start"], "end": t["end"],
                           "words": list(t["words"])})
    clips = []
    for g in groups:
        s, e = g["start"] + pad, g["end"] - pad
        words = [w for w in g["words"] if s - 0.05 <= (w["s"] + w["e"]) / 2 <= e + 0.05]
        cuts = [s]
        if e - s > max_len and words:
            seg_start = s
            for i, w in enumerate(words[:-1]):
                nxt = words[i + 1]
                if nxt["e"] - seg_start > max_len:
                    cut = round((w["e"] + nxt["s"]) / 2, 3)
                    if cut - seg_start >= min_len:
                        cuts.append(cut)
                        seg_start = cut
        elif e - s > max_len:  # no words: even split
            k = int(np.ceil((e - s) / max_len))
            cuts += [s + i * (e - s) / k for i in range(1, k)]
        cuts.append(e)
        for a, b in zip(cuts, cuts[1:]):
            if b - a < min_len:
                continue
            text = " ".join(w["w"] for w in words if a <= (w["s"] + w["e"]) / 2 < b).strip()
            clips.append({"speaker": g["speaker"], "start": round(a, 3), "end": round(b, 3), "text": text})
    return clips


def export_project(proj_dir: Path, project: dict, *, language="en", min_len=2.0, max_len=12.0,
                   preview_seconds=25.0, pad=0.08) -> dict:
    if language in (None, "", "auto"):
        language = project.get("language") or "en"
    audio, sr = sf.read(str(proj_dir / "audio.wav"), dtype="float32")
    out = proj_dir / "export"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    names, used = {}, set()
    for s in project["speakers"]:  # two voices with the same name get their own folders
        base = n = _safe(s["name"])
        i = 2
        while n.lower() in used:
            n, i = f"{base}_{i}", i + 1
        used.add(n.lower())
        names[s["id"]] = n
    skipped = {s["id"] for s in project["speakers"] if not s.get("export", True)}
    clips = [c for c in plan_clips(project, min_len=min_len, max_len=max_len, pad=pad)
             if c["speaker"] not in skipped]
    summary = {}
    by = defaultdict(list)
    for c in clips:
        by[c["speaker"]].append(c)
    for spk, cs in by.items():
        folder = out / names.get(spk, _safe(spk))
        folder.mkdir(parents=True, exist_ok=True)
        lines, preview, total = [], [], 0.0
        gap = np.zeros(int(0.4 * sr), dtype=audio.dtype)
        for i, c in enumerate(cs, 1):
            chunk = audio[int(c["start"] * sr):int(c["end"] * sr)]
            path = folder / f"clip_{i:04d}.wav"
            sf.write(str(path), chunk, sr)
            total += c["end"] - c["start"]
            text = " ".join(c["text"].replace("|", " ").split())  # one line per clip, | separates fields
            if text:
                lines.append(f"{path.resolve()}|{names.get(spk, spk)}|{language}|{text}")
            if sum(len(p) for p in preview) / sr < preview_seconds:
                preview += [chunk, gap]
        (folder / "dataset.list").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if preview:
            sf.write(str(out / f"{names.get(spk, spk)}_preview.wav"), np.concatenate(preview), sr)
        summary[names.get(spk, spk)] = {"clips": len(cs), "minutes": round(total / 60, 2),
                                        "folder": str(folder.resolve())}
    write_transcripts(out, project, {s["id"]: s["name"] for s in project["speakers"]})
    return summary


def _clock(t: float, srt: bool = False) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1000)
    if srt:
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def write_transcripts(out: Path, project: dict, names: dict) -> None:
    """The whole conversation (every kept line, every voice) as transcript.txt and transcript.srt."""
    turns = [t for t in sorted(project["turns"], key=lambda t: t["start"]) if not t["excluded"] and t["text"].strip()]
    txt, srt, last = [], [], None
    for t in turns:
        who = names.get(t["speaker"], t["speaker"])
        if who != last:
            txt.append(f"\n[{_clock(t['start'])}] {who}:" if txt else f"[{_clock(t['start'])}] {who}:")
            last = who
        txt.append(t["text"].strip())
    for i, t in enumerate(turns, 1):
        srt += [str(i), f"{_clock(t['start'], True)} --> {_clock(t['end'], True)}",
                f"{names.get(t['speaker'], t['speaker'])}: {t['text'].strip()}", ""]
    (out / "transcript.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    (out / "transcript.srt").write_text("\n".join(srt), encoding="utf-8")


# ------------------------------------------------------------------ re-split
def _subtract(spans: list[tuple[float, float]], holes: list[tuple[float, float]], min_len: float = 0.2):
    """Cut ``holes`` out of ``spans``; drop slivers under ``min_len``."""
    out = []
    for s, e in spans:
        parts = [(s, e)]
        for hs, he in holes:
            nxt = []
            for a, b in parts:
                if he <= a or hs >= b:
                    nxt.append((a, b))
                    continue
                if hs - a >= min_len:
                    nxt.append((a, hs))
                if b - he >= min_len:
                    nxt.append((he, b))
            parts = nxt
        out += [p for p in parts if p[1] - p[0] >= min_len]
    return out


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def map_labels(new_turns, old_turns) -> dict[str, str | None]:
    """Match the model's fresh labels to the speakers you already have, by how
    much their time overlaps (most overlap first, one-to-one). Labels with no
    match come back as None and become new speakers."""
    labels = sorted({t[2] for t in new_turns})
    olds = sorted({t["speaker"] for t in old_turns})
    score = {(n, o): 0.0 for n in labels for o in olds}
    for s, e, n in new_turns:
        for t in old_turns:
            score[(n, t["speaker"])] += _overlap((s, e), (t["start"], t["end"]))
    mapping: dict[str, str | None] = {n: None for n in labels}
    used: set[str] = set()
    for (n, o), v in sorted(score.items(), key=lambda kv: -kv[1]):
        if v > 0 and mapping[n] is None and o not in used:
            mapping[n] = o
            used.add(o)
    return mapping


def apply_resplit(project: dict, new_turns, start: float, end: float, *, new_words: list[dict] | None = None,
                  keep_checked: bool = True) -> dict:
    """Replace the split between ``start`` and ``end`` with ``new_turns``.

    - Speakers carry over: fresh labels are matched to your existing speakers
      (names, colors and export choices stay); only truly new voices are added.
    - Lines you checked (and, with ``keep_checked``, ones whose words you
      corrected) are left exactly as they are; the new split fills around them.
    - Lines that only partly overlap the range are trimmed, not lost.
    - Words: ``new_words`` (a fresh transcript of the range) replaces the old
      words there; otherwise the existing transcript is reused.
    """
    s0, e0 = max(0.0, start), min(float(project["duration"]), end)
    words = [w for w in project.get("words") or [] if not (s0 <= (w["s"] + w["e"]) / 2 < e0)] \
        if new_words is not None else list(project.get("words") or [])
    if new_words is not None:
        words = sorted(words + [w for w in new_words if s0 <= (w["s"] + w["e"]) / 2 < e0], key=lambda w: w["s"])

    inside = [t for t in project["turns"] if _overlap((t["start"], t["end"]), (s0, e0)) > 0]
    protect = [t for t in inside if keep_checked and (t["confirmed"] or t.get("text_edited"))]
    replace = [t for t in inside if t not in protect]
    holes = [(t["start"], t["end"]) for t in protect]

    # Old lines: keep everything outside the range, trim the ones that straddle it.
    next_id = max([t["id"] for t in project["turns"]] + [-1]) + 1
    kept = [t for t in project["turns"] if t not in replace]
    for t in replace:
        for i, (a, b) in enumerate(_subtract([(t["start"], t["end"])], [(s0, e0)])):
            ws = [w for w in words if a <= (w["s"] + w["e"]) / 2 < b]
            kept.append({**t, "id": t["id"] if i == 0 else next_id, "start": round(a, 3), "end": round(b, 3),
                         "words": ws, "text": t["text"] if t.get("text_edited") else " ".join(w["w"] for w in ws),
                         "suggest": None})
            if i:
                next_id += 1

    # New lines: clipped to the range, around the lines you protected.
    clipped = []
    for s, e, lab in merge_turns(new_turns):
        for a, b in _subtract([(max(s, s0), min(e, e0))], holes):
            clipped.append((a, b, lab))
    mapping = map_labels(clipped, replace or inside)
    speakers = [dict(sp) for sp in project["speakers"]]
    for lab, target in mapping.items():
        if target is None:
            n = len(speakers)
            new_id = f"RESPLIT_{n + 1}"
            while any(sp["id"] == new_id for sp in speakers):
                n += 1
                new_id = f"RESPLIT_{n + 1}"
            speakers.append({"id": new_id, "name": f"Speaker {len(speakers) + 1}",
                             "color": COLORS[len(speakers) % len(COLORS)], "export": True})
            mapping[lab] = new_id
    range_words = [w for w in words if s0 <= (w["s"] + w["e"]) / 2 < e0
                   and not any(a <= (w["s"] + w["e"]) / 2 < b for a, b in holes)]
    turns3 = [(a, b, mapping[lab]) for a, b, lab in clipped]
    for (a, b, spk), ws in zip(turns3, assign_words(turns3, range_words) if turns3 else []):
        kept.append({"id": next_id, "start": round(a, 3), "end": round(b, 3), "speaker": spk,
                     "text": " ".join(w["w"] for w in ws).strip(), "words": ws, "text_edited": False,
                     "excluded": False, "confirmed": False, "suggest": None})
        next_id += 1
    kept.sort(key=lambda t: t["start"])
    return {**project, "speakers": speakers, "turns": kept, "words": words}


def cut_range(proj_dir: Path, start: float, end: float, pad: float = 0.3) -> tuple[Path, float]:
    """Write ``start``–``end`` (plus a little context) to its own wav. Returns
    (path, offset) so the model's times can be shifted back."""
    info = sf.info(str(proj_dir / "audio.wav"))
    a = max(0.0, start - pad)
    b = min(info.duration, end + pad)
    data, sr = sf.read(str(proj_dir / "audio.wav"), start=int(a * info.samplerate), stop=int(b * info.samplerate),
                       dtype="float32")
    out = proj_dir / "resplit.wav"
    sf.write(str(out), data, sr)
    return out, a


def save_previous(proj_dir: Path) -> None:
    """Keep the split as it was, so one click brings it back."""
    with _io_lock:
        shutil.copy2(proj_dir / "project.json", proj_dir / "project.prev.json")


def restore_previous(proj_dir: Path) -> bool:
    prev = proj_dir / "project.prev.json"
    if not prev.exists():
        return False
    with _io_lock:
        current = load_project(proj_dir)
        save_project(proj_dir, json.loads(prev.read_text(encoding="utf-8")))
        prev.write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")  # restore is itself undoable
    return True


def resplit(proj_dir: Path, *, start: float, end: float | None, num_speakers=None, token=None, language="en",
            retranscribe=False, keep_checked=True, progress=lambda msg: None, diarize_fn=None,
            transcribe_fn=None) -> dict:
    """Run the speaker split again on part (or all) of a project. See apply_resplit."""
    project = load_project(proj_dir)
    end = min(project["duration"], end if end is not None else project["duration"])
    start = max(0.0, start)
    if end - start < 1.0:
        raise ValueError("Mark at least one second to re-split.")
    progress("Cutting the part to re-split")
    wav, offset = cut_range(proj_dir, start, end)
    try:
        progress("Finding who speaks when")
        raw = (diarize_fn or diarize)(wav, num_speakers, token)
        turns = [(s + offset, e + offset, lab) for s, e, lab in raw]
        words = None
        if retranscribe:
            progress("Transcribing")
            words = [{**w, "s": round(w["s"] + offset, 3), "e": round(w["e"] + offset, 3)}
                     for w in (transcribe_fn or transcribe_words)(wav, language)]
    finally:
        wav.unlink(missing_ok=True)
    save_previous(proj_dir)
    updated = apply_resplit(project, turns, start, end, new_words=words, keep_checked=keep_checked)
    save_project(proj_dir, updated)
    progress("Done")
    return updated


def review_stage(proj_dir: Path) -> dict:
    """Where a project is in review, for the library: counts plus a stage."""
    p = load_project(proj_dir)
    on = {s["id"] for s in p["speakers"] if s.get("export", True)}
    kept = [t for t in p["turns"] if not t["excluded"] and t["speaker"] in on]
    checked = sum(1 for t in kept if t["confirmed"])
    listing = list((proj_dir / "export").glob("*/dataset.list"))
    fresh_export = bool(listing) and min(f.stat().st_mtime for f in listing) >= (proj_dir / "project.json").stat().st_mtime
    stage = ("exported" if fresh_export else "to_review" if checked == 0
             else "reviewed" if checked >= len(kept) else "reviewing")
    return {"stage": stage, "kept": len(kept), "checked": checked}
