#!/bin/bash
cd "$(dirname "$0")/../.."
for e in E317 E318 E320; do echo "== $e"; python3 ../autoresearch/tools/experiment.py check $e 2>&1 | tail -n 8; done
