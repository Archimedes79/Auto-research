set -e
S=/mnt/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/6e9638e0-a42f-4660-b7d5-cac910d9084d/scratchpad/fix3
python3 $S/patch_head.py /mnt/c/Auto-research/autoresearch/tools
rm -rf /tmp/arhead && mkdir -p /tmp/arhead && tar -xf $S/arhead.tar -C /tmp/arhead
cp $S/mine/*.py /tmp/arhead/autoresearch/tools/
cd /tmp/arhead/autoresearch/tools
diff <(sed 's/\r$//' /mnt/c/Auto-research/autoresearch/tools/decide.py) decide.py | head -60 || true
set +e
OMP_NUM_THREADS=2 nice python3 selftest.py > /tmp/ar_st_head.txt 2>&1; echo "HEAD+mine selftest exit $?"
grep -B6 -A3 Traceback /tmp/ar_st_head.txt | head -20; tail -1 /tmp/ar_st_head.txt
