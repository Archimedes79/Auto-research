#!/bin/bash
# throw-away: the video checks of the evalsuite selftest with $FFMPEG = the Windows ffmpeg.exe of this PC (H.264 path through WSL interop)
export OMP_NUM_THREADS=2 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export FFMPEG="/mnt/c/Program Files/DownloadHelper CoApp/ffmpeg.exe"
cd /mnt/c/Auto-research/autoresearch || exit 1
/home/ml/miniconda3/envs/gcar/bin/python -m evalsuite selftest video
/home/ml/miniconda3/envs/gcar/bin/python - <<'EOF'
import os, tempfile
import numpy as np
from evalsuite.render import video
with tempfile.TemporaryDirectory() as d:
    print("with FFMPEG:", {k: v for k, v in video.encode([np.full((75, 101, 3), 40 * i + 20, np.uint8) for i in range(4)], os.path.join(d, "v.mp4")).items() if k != "path"})
EOF
