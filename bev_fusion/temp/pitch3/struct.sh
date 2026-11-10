cd "$(git rev-parse --show-toplevel)"
echo "--- bev_fusion"; AR_PROJECT=../bev_fusion python3 autoresearch/tools/check_structure.py 2>&1 | tail -25
echo "--- default"; python3 autoresearch/tools/check_structure.py 2>&1 | tail -8
