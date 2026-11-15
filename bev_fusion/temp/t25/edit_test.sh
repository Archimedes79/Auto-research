#!/usr/bin/env bash
# scratch: apply the with_gt -> stub edit to the test (sed line ranges)
set -e
f=tests/test_radar_recursive.py
sed -i '296,301d' $f
sed -i '295a\
def keyframe(batch):\
    """the fields run_clip reads from a key frame: the ego steps to the neighbouring key frames, the tokens (the GT is not needed: the loss is stubbed below)"""\
    gaps = {"prev": [(-1 / 12, 8.0, 0.05)] * 6, "next": [(1 / 12, 8.0, 0.05)] * 6}\
    return dict(batch, gap_steps=[gaps] * B, tokens=["a", "b"])' $f
