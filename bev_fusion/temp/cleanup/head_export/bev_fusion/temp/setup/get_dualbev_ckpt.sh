#!/usr/bin/env bash
# One-off (approved by Mirko 2026-10-25): download the DualBEV single-frame checkpoint dualbev-r50-cbgs.pth
# (316.7 MB, Google Drive folder of the DualBEV authors, file id 1rUM1yQzQKRd0GH1LgOtpN5p8oqQu0L2a) into
# models/pretrained/ of this project, then record size and sha256. Run from the bev_fusion folder.
set -euo pipefail
dst=models/pretrained
mkdir -p "$dst"
f="$dst/dualbev-r50-cbgs.pth"
curl -fL --retry 5 -C - -o "$f" "https://drive.usercontent.google.com/download?id=1rUM1yQzQKRd0GH1LgOtpN5p8oqQu0L2a&export=download&confirm=t"
ls -la "$f"
file "$f" | head -1
sha256sum "$f" | tee "$dst/dualbev-r50-cbgs.pth.sha256"
