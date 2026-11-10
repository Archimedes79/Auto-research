#!/bin/bash
cd /mnt/c/Auto-research || exit 1
git log --oneline -3 | cut -c1-120
echo "=== diff stat of the files I will commit"
git diff --stat -- autoresearch/runsuite autoresearch/trainsuite/api.py autoresearch/trainsuite/selftest_api.py autoresearch/trainsuite/README.md bev_fusion/project.json | cat
echo "=== trainsuite/api.py diff"
git diff -U0 -- autoresearch/trainsuite/api.py | cut -c1-200
echo "=== project.json diff"
git diff -U0 -- bev_fusion/project.json | cut -c1-260
echo "=== untracked of mine"
git status --short -- autoresearch/runsuite bev_fusion/tests bev_fusion/records/jobs | cat
echo "=== autoresearch/README.md: other hunks besides mine?"
git diff -U0 -- autoresearch/README.md | grep -E '^[-+]\|' | cut -c1-120
