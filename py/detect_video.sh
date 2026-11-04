#!/usr/bin/env bash
# Play a video in mpv with live SpikeYOLO detections drawn on top.
#   ./detect_video.sh VIDEO [--model int|bin] [--imgsz 640] [--conf 0.25] [--mpv-option=value ...]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
[ $# -ge 1 ] || { echo "usage: $0 VIDEO [--model int|bin] [--imgsz N] [--conf X] [mpv options]" >&2; exit 1; }
exec "$here/.venv/bin/python" "$here/video_detect.py" "$@"
