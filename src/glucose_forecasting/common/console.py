"""Console output helpers safe on Windows (cp1251 / legacy code pages)."""
from __future__ import annotations

import sys


def configure_stdio_utf8() -> None:
    """Prefer UTF-8 stdout/stderr when the stream supports reconfiguration."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        reconfigure(encoding="utf-8", errors="replace")


def safe_echo(message: str, *, err: bool = False) -> None:
    """Print a line without raising ``UnicodeEncodeError`` on narrow encodings."""
    stream = sys.stderr if err else sys.stdout
    text = message if message.endswith("\n") else f"{message}\n"
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        stream.write(text)
        stream.flush()
    except UnicodeEncodeError:
        stream.buffer.write(text.encode(encoding, errors="replace"))
        stream.flush()


def init_cli_console() -> None:
    """Call once at CLI startup (before any user-visible output)."""
    configure_stdio_utf8()
