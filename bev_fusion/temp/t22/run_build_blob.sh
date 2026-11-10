#!/bin/bash
# scratch: dumps the committed ledger files and builds the blobs with only the t22 rows
cd /mnt/c/Auto-research/bev_fusion || exit 1
git -C /mnt/c/Auto-research show HEAD:bev_fusion/records/LEDGER.csv > temp/t22/head_LEDGER.csv || exit 1
git -C /mnt/c/Auto-research show HEAD:bev_fusion/records/LEDGER.md > temp/t22/head_LEDGER.md || exit 1
python3 temp/t22/build_ledger_blob.py
echo "--- diff of the csv blob against HEAD (ids only):"
diff <(python3 -c "import csv;print('\n'.join(r['id'] for r in csv.DictReader(open('temp/t22/head_LEDGER.csv',newline=''))))") <(python3 -c "import csv;print('\n'.join(r['id'] for r in csv.DictReader(open('temp/t22/blob_LEDGER.csv',newline=''))))")
echo "--- diff of the md blob against HEAD (line count):"
diff temp/t22/head_LEDGER.md temp/t22/blob_LEDGER.md | grep -c '^>'
