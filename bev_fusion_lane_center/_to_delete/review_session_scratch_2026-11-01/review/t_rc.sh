#!/usr/bin/env bash
mkdir -p out
git -C /mnt/c/Auto-research archive no_such_tag_xyz bev_fusion/training | tar -x -C out --strip-components=2 || echo "pipeline failed -> exit 1 path"
echo "pipeline status after: reached next line"
python3 -c "import sys; sys.exit(3)"
rc=$?
echo "x exit $rc"
