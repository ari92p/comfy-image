"""Tiny helpers for talking to the user in the terminal (standard library only)."""

import os
import pathlib
import re
import shlex
import subprocess
import sys

IS_WIN = os.name == "nt"
VERBOSE = False   # -v: show everything the setup and ComfyUI print
NAME = "easy-image"


def banner():
    say(f"{NAME} - AI images on your own computer: generate, edit, characters, styles")


def say(*a):
    print(*a, flush=True)


def die(msg):
    print(f"\nerror: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def ask(prompt, default=""):
    try:
        s = input(f"{prompt} " + (f"[{default}] " if default else "")).strip()
    except (EOFError, KeyboardInterrupt):
        say("")
        sys.exit(130)
    return s or default


def yes(prompt, default="y"):
    return ask(prompt + " (y/n)", default).lower() in ("y", "yes", "j", "ja")


def clean_path(s):
    """Accept a path as typed, quoted, or dragged into the terminal window."""
    s = s.strip()
    if len(s) > 1 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1]
    elif not IS_WIN:
        s = re.sub(r"\\(.)", r"\1", s)
    return str(pathlib.Path(os.path.expanduser(s)))


def rel(p):
    try:
        return os.path.relpath(p)
    except ValueError:
        return str(p)


def shell_join(argv):
    argv = [str(a) for a in argv]
    return subprocess.list2cmdline(argv) if IS_WIN else " ".join(shlex.quote(a) for a in argv)


def launcher():
    """The short start command of this folder (what the user types)."""
    return "easy-image.cmd" if IS_WIN else "./easy-image"


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1000
