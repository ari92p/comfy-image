#!/bin/sh
# One-time setup for Mac and Linux. Needs nothing installed: it fetches uv, uv fetches Python.
# Everything goes into .venvs/ in this folder. ComfyUI and the models are set up on the first start (it asks first).
cd "$(dirname "$0")/.." || exit 1
say() { echo "#easy-image: $*"; }
PY=.venvs/app/bin/python
if [ -f .venvs/app/.image-ready ]; then say "Already set up. Start it with: ./easy-image"; exit 0; fi

UV=$(command -v uv)
if [ -z "$UV" ]; then
  UV=.venvs/uv/bin/uv
  if [ ! -x "$UV" ]; then
    say "Installing uv (the Python setup tool)..."
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$PWD/.venvs/uv/bin" UV_NO_MODIFY_PATH=1 sh >/dev/null || { say "Could not install uv."; exit 1; }
  fi
fi

say "Setting up (a minute)..."
rm -rf .venvs/app
"$UV" venv --quiet --python 3.13 .venvs/app || exit 1
[ -x "$PY" ] || exit 1
touch .venvs/app/.image-ready
say "Done. Start it with: ./easy-image"
