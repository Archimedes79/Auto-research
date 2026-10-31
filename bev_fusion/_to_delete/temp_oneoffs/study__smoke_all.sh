#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
for a in ref_s0 trunk_second fusion_cmx; do bash temp/train/smoke.sh temp/study/t2s_$a.json; done
