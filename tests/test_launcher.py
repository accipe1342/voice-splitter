"""The launcher must not silently hand the window to an older copy that's still running."""
import socket
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import voice_splitter  # noqa: E402
from splitter import __version__  # noqa: E402
from splitter.server import create_app  # noqa: E402


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(app, port):
    from werkzeug.serving import make_server
    srv = make_server("127.0.0.1", port, app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_nothing_running():
    assert voice_splitter.already_running(_free_port()) is None


def test_same_version_running(tmp_path):
    port = _free_port()
    srv = _serve(create_app(root=tmp_path), port)
    try:
        assert voice_splitter.already_running(port) == __version__
    finally:
        srv.shutdown()


def test_old_copy_running():
    from flask import Flask
    port = _free_port()
    srv = _serve(Flask("old"), port)  # an old build: no /api/version
    try:
        assert voice_splitter.already_running(port) == ""
    finally:
        srv.shutdown()
