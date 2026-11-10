#!/bin/bash
cd /mnt/c/Auto-research
PY="$HOME/miniconda3/envs/gcar/bin/python"
P="autoresearch/evalsuite/api.py autoresearch/evalsuite/apicli.py autoresearch/evalsuite/results.py autoresearch/evalsuite/selftest_api.py autoresearch/evalsuite/__init__.py autoresearch/evalsuite/cli.py autoresearch/evalsuite/quantitative.py autoresearch/evalsuite/README.md autoresearch/evalsuite/EXTENSIONS.md autoresearch/INTERFACE.md"
$PY autoresearch/tools/check_git.py --paths $P | tail -3
$PY autoresearch/tools/commit_paths.py -F bev_fusion/temp/pitch/commit_a_msg.txt $P 2>&1 | tail -14
