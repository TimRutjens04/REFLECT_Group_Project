"""Live prompt input: a daemon stdin-reader thread.

Type a comma-separated list in the terminal (``apple, bowl, mug``) and press
Enter; the main loop polls ``current()`` each frame and re-prompts the
detector only when the version changes. Works identically on macOS/Linux
without fighting OpenCV's key handling.
"""

from __future__ import annotations

import sys
import threading


def parse_prompt_line(line: str) -> list[str]:
    return [t.strip() for t in line.split(",") if t.strip()]


class PromptInput:
    def __init__(self, initial: list[str]):
        self._prompts = list(initial)
        self._version = 0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        try:
            for line in sys.stdin:
                names = parse_prompt_line(line)
                if not names:
                    continue
                with self._lock:
                    self._prompts = names
                    self._version += 1
        except (ValueError, OSError):
            return  # stdin closed (headless/piped) — keep initial prompts

    def current(self) -> tuple[int, list[str]]:
        with self._lock:
            return self._version, list(self._prompts)
