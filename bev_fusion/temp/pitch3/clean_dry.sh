cd "$(git rev-parse --show-toplevel)"
AR_PROJECT=../bev_fusion python3 autoresearch/tools/clean_temp.py pitch3 2>&1 | tail -12 | cut -c1-200
echo ====; AR_PROJECT=../bev_fusion python3 autoresearch/tools/clean_temp.py pitch2 2>&1 | tail -8 | cut -c1-200
