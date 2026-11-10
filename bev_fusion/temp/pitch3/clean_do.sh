cd "$(git rev-parse --show-toplevel)"
cp bev_fusion/temp/pitch3/clean_do.sh /tmp/clean_do_keep.sh
for f in pitch2 pitch3; do AR_PROJECT=../bev_fusion python3 autoresearch/tools/clean_temp.py $f --move --force 2>&1 | tail -6 | cut -c1-220; echo ===; done
