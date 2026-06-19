#!/usr/bin/env python3
"""
run.py — one command to build + serve + open the synchronized video dashboard.

    python "dashboard 2/run.py"

It rebuilds index.html from the current JSONLs, starts a local HTTP server on
http://127.0.0.1:8765 (with HTTP Range support so the video scrub bar works),
and opens the dashboard in your default browser. Press Ctrl-C to stop.
"""

from __future__ import annotations

import functools
import http.server
import os
import re
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Make sure we can import the sibling build.py regardless of CWD.
DASH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DASH_DIR))

import build  # noqa: E402

HOST = "127.0.0.1"
PORT = build.PORT


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler that honours single-range `Range:` requests.

    The stdlib handler ignores Range, which breaks <video> seeking/scrubbing in
    most browsers. This adds minimal 206 Partial Content support.
    """

    def log_message(self, fmt, *args):  # quieter console
        pass

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()

        ctype = self.guess_type(path)
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        try:
            fs = os.fstat(f.fileno())
            size = fs.st_size
            rng = self.headers.get("Range")

            if rng is None:
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(size))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
                self.end_headers()
                return f

            m = re.match(r"bytes=(\d*)-(\d*)\s*$", rng)
            if not m or (m.group(1) == "" and m.group(2) == ""):
                self.send_error(400, "Invalid Range header")
                f.close()
                return None

            start_s, end_s = m.group(1), m.group(2)
            if start_s == "":  # suffix range: last N bytes
                length = int(end_s)
                start = max(0, size - length)
                end = size - 1
            else:
                start = int(start_s)
                end = int(end_s) if end_s else size - 1

            if start >= size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", "bytes */%d" % size)
                self.end_headers()
                f.close()
                return None

            end = min(end, size - 1)
            length = end - start + 1

            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.end_headers()

            # Stream the requested slice ourselves, then signal do_GET to stop.
            f.seek(start)
            remaining = length
            chunk = 64 * 1024
            try:
                while remaining > 0:
                    buf = f.read(min(chunk, remaining))
                    if not buf:
                        break
                    self.wfile.write(buf)
                    remaining -= len(buf)
            except (BrokenPipeError, ConnectionResetError):
                pass
            f.close()
            return None
        except Exception:
            f.close()
            raise


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    # 1) build
    build.main()

    # 2) serve
    handler = functools.partial(RangeRequestHandler, directory=str(DASH_DIR))
    try:
        httpd = Server((HOST, PORT), handler)
    except OSError as e:
        sys.exit(
            f"\nCould not bind {HOST}:{PORT} ({e}).\n"
            f"Another process may already be using port {PORT}. "
            "Stop it or change PORT in build.py."
        )

    url = f"http://{HOST}:{PORT}/index.html"
    print("\n" + "=" * 60)
    print(f"  Dashboard running at {url}")
    print("  Press Ctrl-C to stop.")
    print("=" * 60 + "\n")

    # 3) open browser shortly after the server starts accepting
    #    (skip with --no-browser or DASH2_NO_BROWSER=1, e.g. for headless preview)
    no_browser = "--no-browser" in sys.argv or os.environ.get("DASH2_NO_BROWSER")
    if not no_browser:
        threading.Thread(target=lambda: (time.sleep(0.6), webbrowser.open(url)),
                         daemon=True).start()

    # 4) block until Ctrl-C
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down…")
    finally:
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
