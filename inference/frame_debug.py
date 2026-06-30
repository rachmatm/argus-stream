"""
Frame offset debugging helpers.

Decodes between byte offsets and (x, y, channel) tuples for an RGB24 640x480 frame
(see detection_engine.py for the layout). Toggle via FRAME_DEBUG in the .env at
the project root, or in the shell environment that launches Python.

Usage from detection_engine.py:
    from frame_debug import debug_enabled, log_frame_boundaries
    if debug_enabled():
        log_frame_boundaries(raw_bytes)

Standalone CLI (useful from a shell when poking at a specific offset):
    python inference/frame_debug.py --info
    python inference/frame_debug.py 0                # decode offset 0
    python inference/frame_debug.py 100 200 1        # encode (x=100, y=200, G) -> offset
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

WIDTH, HEIGHT, CHANNELS = 640, 480, 3
FRAME_SIZE = WIDTH * HEIGHT * CHANNELS
CHANNEL_NAMES = ("R", "G", "B")

# .env lives at the project root, two levels up from this file (inference/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

_TRUTHY = frozenset(("1", "true", "yes", "on"))


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader - KEY=VALUE per line, no quoting tricks.

    Uses os.environ.setdefault so values the shell already exported always win.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ENV_PATH)


def debug_enabled() -> bool:
    """True iff FRAME_DEBUG is set to a truthy value in env or .env."""
    return os.environ.get("FRAME_DEBUG", "0").strip().lower() in _TRUTHY


def decode_offset(offset: int) -> tuple[int, int, int]:
    """Byte offset -> (x, y, channel_index). Channel 0=R, 1=G, 2=B."""
    if not 0 <= offset < FRAME_SIZE:
        raise ValueError(f"offset {offset} out of range [0, {FRAME_SIZE})")
    pixel = offset // CHANNELS
    channel = offset % CHANNELS
    return pixel % WIDTH, pixel // WIDTH, channel


def encode_xyz(x: int, y: int, channel: int) -> int:
    """(x, y, channel_index) -> byte offset."""
    if not 0 <= x < WIDTH:
        raise ValueError(f"x {x} out of range [0, {WIDTH})")
    if not 0 <= y < HEIGHT:
        raise ValueError(f"y {y} out of range [0, {HEIGHT})")
    if not 0 <= channel < CHANNELS:
        raise ValueError(f"channel {channel} out of range [0, {CHANNELS})")
    return (y * WIDTH + x) * CHANNELS + channel


def describe_offset(offset: int) -> str:
    """One-line human description of a byte offset (no byte value - caller provides)."""
    x, y, ch = decode_offset(offset)
    return f"offset {offset:>7}  ->  ({x:>3}, {y:>3})  channel={CHANNEL_NAMES[ch]}"


def log_frame_boundaries(raw_bytes: bytes, frame_index: int = 0) -> None:
    """Log decoded (x, y, channel) + byte value for sentinel offsets of a frame."""
    if not raw_bytes or len(raw_bytes) != FRAME_SIZE:
        print(
            f"[frame-debug] frame #{frame_index} skipped: expected {FRAME_SIZE} bytes, got {len(raw_bytes) if raw_bytes else 0}",
            file=sys.stderr,
            flush=True,
        )
        return

    sentinels = [
        ("top-left",     0),
        ("top-right",    encode_xyz(WIDTH - 1, 0, 0)),
        ("center",       encode_xyz(WIDTH // 2, HEIGHT // 2, 0)),
        ("bottom-left",  encode_xyz(0, HEIGHT - 1, 0)),
        ("bottom-right", encode_xyz(WIDTH - 1, HEIGHT - 1, CHANNELS - 1)),
        ("last byte",    FRAME_SIZE - 1),
    ]
    print(
        f"[frame-debug] frame #{frame_index}  bytes={len(raw_bytes)}  debug={'on' if debug_enabled() else 'off'}",
        file=sys.stderr,
        flush=True,
    )
    for label, off in sentinels:
        x, y, ch = decode_offset(off)
        b = raw_bytes[off]
        print(
            f"[frame-debug]   {label:<14} offset={off:>7}  ({x:>3}, {y:>3})  "
            f"{CHANNEL_NAMES[ch]}={b:3d}  0x{b:02X}",
            file=sys.stderr,
            flush=True,
        )


def _cli(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--info":
        print(f"WIDTH={WIDTH} HEIGHT={HEIGHT} CHANNELS={CHANNELS} FRAME_SIZE={FRAME_SIZE}")
        print(f"FRAME_DEBUG={os.environ.get('FRAME_DEBUG', '<unset>')}  enabled={debug_enabled()}")
        print(f"loaded .env from {ENV_PATH} (exists={ENV_PATH.exists()})")
        return 0
    if len(argv) == 1:
        print(describe_offset(int(argv[0])))
        return 0
    if len(argv) == 3:
        x, y, ch = int(argv[0]), int(argv[1]), int(argv[2])
        print(encode_xyz(x, y, ch))
        return 0
    print(
        "usage: frame_debug.py --info | <offset> | <x> <y> <channel>",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))