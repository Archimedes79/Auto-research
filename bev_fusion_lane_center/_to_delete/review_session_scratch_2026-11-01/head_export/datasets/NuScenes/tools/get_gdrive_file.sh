#!/usr/bin/env bash
# Download ONE public Google Drive file (resumable), check it against the Content-Length of the server, test gzip archives.
# Usage: get_gdrive_file.sh <file id> <destination file>
# (Drive gives no checksum on this endpoint: size + gzip integrity are the checks; unpack with extract_blob.sh.)
set -euo pipefail
id="$1"; dest="$2"
url="https://drive.usercontent.google.com/download?id=${id}&export=download&confirm=t"
mkdir -p "$(dirname "$dest")"
want="$(curl -sIL -A 'Mozilla/5.0' "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v}')"
[ -n "$want" ] || { echo "ABORT: no Content-Length from the server"; exit 1; }
echo "$(date +%T) download $(basename "$dest") ($want bytes)"
curl -sfL --retry 10 --retry-delay 20 -C - -A 'Mozilla/5.0' -o "$dest" "$url"
have="$(stat -c %s "$dest")"
[ "$have" = "$want" ] || { echo "MISMATCH: $have bytes, expected $want"; exit 1; }
case "$dest" in *.gz) gzip -t "$dest" && echo "$(date +%T) gzip test ok";; esac
echo "$(date +%T) done $(basename "$dest"): $have bytes"
