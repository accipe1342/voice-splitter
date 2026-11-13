"""Voice Splitter -- separate the people in a recording, review, export clips.

    python voice_splitter.py                 start the app (opens its window)
    python voice_splitter.py split FILE      analyze one file without the UI
    python voice_splitter.py export ID       re-cut clips from a project's edits
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def already_running(port: int):
    """None if nothing is on the port; otherwise the running app's version ("" if it's too old to say)."""
    import json
    import socket
    import urllib.request
    with socket.socket() as sock:
        sock.settimeout(0.5)
        if sock.connect_ex(("127.0.0.1", port)) != 0:
            return None
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/version", timeout=2) as r:
            return json.load(r).get("version", "")
    except Exception:
        return ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ffmpeg", help="path to ffmpeg(.exe) if it's not found automatically")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="start the app (default)")
    s.add_argument("--port", type=int, default=7861)
    s.add_argument("--no-window", action="store_true", help="don't open the app window")
    s.add_argument("--library", action="append", default=[], help="extra folder to list")
    sp = sub.add_parser("split", help="analyze a file without the UI")
    sp.add_argument("file", type=Path)
    sp.add_argument("--speakers", type=int)
    sp.add_argument("--language", default="en")
    ex = sub.add_parser("export", help="cut clips from a project's edits")
    ex.add_argument("project", help="project id (a folder name under projects/)")
    args = ap.parse_args(argv)

    from splitter import core

    if args.cmd == "split":
        import os
        token = os.environ.get("HF_TOKEN")
        tf = ROOT / "secrets" / "hf_token.txt"
        if not token and tf.exists():
            token = tf.read_text().strip()
        from splitter.server import create_app  # reuse id scheme
        pid = args.file.stem.lower()[:40] + "-cli"
        d = ROOT / "projects" / pid
        core.analyze(args.file, d, num_speakers=args.speakers, token=token, language=args.language,
                     ffmpeg=args.ffmpeg, progress=lambda m: print(f"[split] {m}"))
        print(f"Project: {d}\nReview it in the app, or export now:  python voice_splitter.py export {pid}")
        return

    if args.cmd == "export":
        d = ROOT / "projects" / args.project
        summary = core.export_project(d, core.load_project(d))
        for name, info in summary.items():
            print(f"{name}: {info['clips']} clips, {info['minutes']} min -> {info['folder']}")
        return

    import webbrowser
    from splitter import __version__
    from splitter.server import Settings, create_app, open_app_window
    port = getattr(args, "port", 7861)
    running = already_running(port)
    if running == __version__:  # this version is already open: just show it
        print(f"Voice Splitter is already running at http://127.0.0.1:{port}/")
        if not getattr(args, "no_window", False):
            open_app_window(f"http://127.0.0.1:{port}/")
        return
    if running is not None:
        print("\n  An older Voice Splitter is still running, so this one can't start.\n"
              "  Close its black command window (or its app window), then start Voice Splitter again.\n")
        sys.exit(1)
    app = create_app(extra_folders=getattr(args, "library", []), ffmpeg=args.ffmpeg)
    url = f"http://127.0.0.1:{port}/"
    if not getattr(args, "no_window", False):
        window = Settings(ROOT / "settings.json").data.get("window", "app")
        opener = open_app_window if window == "app" else webbrowser.open
        threading.Thread(target=lambda: (time.sleep(1.2), opener(url)), daemon=True).start()
    print(f"Voice Splitter running at {url}  (close this window to quit)")
    app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
