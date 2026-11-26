"""Saved characters and styles: pictures you keep under a name, so `--character me` works without the file.
They live in ./characters and ./styles in the repository folder (not part of the repository) and are plain image files."""

import os
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXTS = (".png", ".jpg", ".jpeg", ".webp")
KINDS = ("character", "style")


def folder(kind):
    base = pathlib.Path(os.environ.get("IMAGE_LIBRARY_DIR") or ROOT)
    return base / ("characters" if kind == "character" else "styles")


def clean_name(text):
    return re.sub(r"[^a-z0-9_-]+", "-", text.lower()).strip("-")[:40]


def names(kind):
    d = folder(kind)
    return sorted({p.stem for p in d.iterdir() if p.suffix.lower() in EXTS}) if d.exists() else []


def find(kind, name):
    n = clean_name(name)
    d = folder(kind)
    for ext in EXTS:
        p = d / (n + ext)
        if n and p.exists():
            return p
    return None


def resolve(kind, text):
    """A file you named, or a saved picture of that name -> its bytes. None when neither exists."""
    p = pathlib.Path(os.path.expanduser(text))
    if p.is_file():
        return p.read_bytes()
    saved = find(kind, text)
    return saved.read_bytes() if saved else None


def save(kind, name, data, ext=".png"):
    n = clean_name(name)
    if not n:
        raise ValueError("give the saved picture a name made of letters and numbers")
    d = folder(kind)
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob(n + ".*"):
        old.unlink()
    p = d / (n + (ext if ext in EXTS else ".png"))
    p.write_bytes(data)
    return p
