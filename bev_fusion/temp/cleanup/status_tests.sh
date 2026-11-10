#!/bin/bash
# the test suites of the rebuild, once, at the state of now
cd /mnt/c/Auto-research/bev_fusion || exit 1
echo "== BEV gates (fast set)"
bash tests/run_gates.sh > /tmp/st_gates.txt 2>&1; echo "rc=$? PASS=$(grep -c '^PASS' /tmp/st_gates.txt) FAIL=$(grep -c '^FAIL' /tmp/st_gates.txt)"; grep '^FAIL' /tmp/st_gates.txt | cut -c1-200
cd /mnt/c/Auto-research
echo "== autoresearch tools selftest"
python3 autoresearch/tools/selftest.py > /tmp/st_tools.txt 2>&1; echo "rc=$? $(tail -n 1 /tmp/st_tools.txt | cut -c1-120)"
echo "== suitecli, jobrunner"
(cd autoresearch && python3 suitecli.py --selftest 2>&1 | tail -n 1; python3 -m jobrunner selftest 2>&1 | tail -n 1)
echo "== check_git summary"
python3 autoresearch/tools/check_git.py > /tmp/st_git.txt 2>&1; echo "rc=$?"; grep -E "ahead|^pending|^scratch in git|ERROR" /tmp/st_git.txt | cut -c1-160
echo "== train/eval/GT suites present"
ls -d autoresearch/trainsuite autoresearch/evalsuite autoresearch/jobrunner bev_fusion/runner GtGenerator 2>&1 | tr '\n' ' '
