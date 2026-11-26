#!/bin/sh
# Starts the image service on port 8192 (reachable from other computers). Run setup.sh once first.
cd "$(dirname "$0")/.." || exit 1
exec ./comfy-image --serve --host 0.0.0.0 --port 8192 -y
