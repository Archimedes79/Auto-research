#!/bin/bash
cd /mnt/c/Auto-research/autoresearch/runsuite || exit 1
sed -i '1s/`plan`, `results`, `tap`/`plan`, `list_results`, `tap`/; 6s/p\.results(/p.list_results(/' predict.py
sed -i '2s/`plan`, `results`, `tap`/`plan`, `list_results`, `tap`/' __main__.py
head -6 predict.py | cut -c1-200
sed -n '1,3p' __main__.py | cut -c1-200
