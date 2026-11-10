#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion
until [ -f temp/lane_tests/DONE ]; do sleep 60; done
echo chain DONE 11/05/2026 18:45:14; tail -6 temp/lane_tests/chain.log
