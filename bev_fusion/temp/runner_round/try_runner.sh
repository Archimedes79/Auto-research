#!/bin/bash
cd /mnt/c/Auto-research/autoresearch || exit 1
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
P=~/miniconda3/envs/gcar/bin/python
echo "=== methods"; $P -m runsuite methods 2>&1 | cut -c1-150 | head -40
echo "=== check predict job"
cat > /tmp/pred_job.json <<'EOF'
{"id": "pred.t20.small", "kind": "run", "mode": "predict", "run": "t20_base_s0", "frames": "small.val", "taps": ["fused"], "device": "cpu", "workers": 0}
EOF
$P -m runsuite Runner.check /tmp/pred_job.json 2>&1 | cut -c1-300
echo "exit $?"
echo "=== check bad frames"
cat > /tmp/pred_bad.json <<'EOF'
{"id": "pred.bad", "kind": "run", "mode": "predict", "run": "t20_base_s0", "frames": "random:x:val", "scene": "scene-0003"}
EOF
$P -m runsuite Runner.check /tmp/pred_bad.json 2>&1 | cut -c1-300
echo "=== results t20"
$P -m runsuite Runner.results runs/t20_base_s0 2>&1 | head -5
echo "=== plan (opens data; skipped)"
