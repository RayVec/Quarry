#!/bin/zsh
set -u

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR/web" || exit 1

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is not installed or not on PATH."
  echo
  read -r "?Press Enter to close..."
  exit 1
fi

if [[ ! -d "$ROOT_DIR/web/node_modules" ]]; then
  echo "Missing frontend dependencies."
  echo "Run: cd \"$ROOT_DIR/web\" && npm install"
  echo
  read -r "?Press Enter to close..."
  exit 1
fi

echo "Starting QUARRY frontend..."
echo "Project: $ROOT_DIR/web"
echo "URL: http://127.0.0.1:5173"
echo

npm run dev -- --host 127.0.0.1 --port 5173

status=$?
echo
echo "Frontend exited with status $status"
read -r "?Press Enter to close..."
exit $status
