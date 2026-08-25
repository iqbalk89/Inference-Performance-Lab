#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIM_IMAGE="manimcommunity/manim:v0.21.0"
PROBLEM="${1:-}"
QUALITY="-qm"
VIDEO_RESOLUTION="720p30"
OPEN_VIDEO=true
SHOW_COMMAND=false

usage() {
  echo "Usage: ./render.sh PROBLEM [--low|--medium|--high] [--no-open] [--show-command]"
  echo "Example: ./render.sh 01"
}

if [[ -z "$PROBLEM" ]]; then
  usage
  exit 2
fi
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --low) QUALITY="-ql"; VIDEO_RESOLUTION="480p15" ;;
    --medium) QUALITY="-qm"; VIDEO_RESOLUTION="720p30" ;;
    --high) QUALITY="-qh"; VIDEO_RESOLUTION="1080p60" ;;
    --no-open) OPEN_VIDEO=false ;;
    --show-command) SHOW_COMMAND=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
  shift
done

case "$PROBLEM" in
  1|01)
    SCENE_FILE="scenes/problem_01.py"
    SCENE_CLASS="TransformerProjectionCost"
    MEDIA_SUBDIR="problem_01"
    ;;
  *)
    echo "No visual solution is registered for problem '$PROBLEM'."
    exit 2
    ;;
esac

DOCKER_COMMAND=(
  docker run --rm
  -v "$SCRIPT_DIR:/manim"
  "$MANIM_IMAGE"
  manim "$QUALITY" "$SCENE_FILE" "$SCENE_CLASS"
)

if [[ "$SHOW_COMMAND" == true ]]; then
  printf '%q ' "${DOCKER_COMMAND[@]}"
  printf '\n'
  exit 0
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker Desktop is not running."
  echo "Open Docker Desktop, wait for 'Engine running', then run this command again."
  exit 1
fi

echo "Rendering Problem $PROBLEM with $MANIM_IMAGE..."
"${DOCKER_COMMAND[@]}"

VIDEO_PATH="$SCRIPT_DIR/media/videos/$MEDIA_SUBDIR/$VIDEO_RESOLUTION/${SCENE_CLASS}.mp4"
if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Render completed, but the expected MP4 was not found under media/videos."
  exit 1
fi

echo "Created: $VIDEO_PATH"
if [[ "$OPEN_VIDEO" == true ]] && [[ "$(uname -s)" == "Darwin" ]]; then
  open "$VIDEO_PATH"
fi
