#!/usr/bin/env bash
# One-off: list file names and ids of a public Google Drive folder (embedded folder view, no login, no download).
# Usage: wsl bash temp/setup/list_drive_folder.sh <folder id>
set -euo pipefail
curl -sL "https://drive.google.com/embeddedfolderview?id=$1" \
  | grep -o 'href="https://drive.google.com/file/d/[^/"]*/[^"]*"[^>]*>.\{0,0\}\|<div class="flip-entry-title">[^<]*' \
  | sed -E 's#href="https://drive.google.com/file/d/([^/"]*)/.*#id \1#; s#<div class="flip-entry-title">#name #'
command -v gdown >/dev/null && echo "gdown: $(command -v gdown)" || echo "gdown: not installed"
~/miniconda3/envs/gcar/bin/python -c "import gdown" 2>/dev/null && echo "gdown in gcar env" || echo "gdown not in gcar env"
