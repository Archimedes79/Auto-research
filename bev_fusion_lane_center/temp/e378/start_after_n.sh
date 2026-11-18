#!/bin/bash
# E378: start the supervisor on slot 2 right after E377.n_s1's training (its GPU dumps queue first) AND after test_jtoken.py ran green at the launch commit (the card's kill)
cd "$(dirname "$0")/../.."
until grep -q "\[sup377\] n_s1 training done" temp/e377/chain377.log 2>/dev/null && grep -q "test_jtoken ok" temp/e378/test_jtoken_launch.log 2>/dev/null; do sleep 20; done
sleep 45
echo "$(date '+%m-%d %H:%M:%S') [start] E377.n_s1 training done and test_jtoken green, starting sup378" >> temp/e378/chain378.log
bash temp/e378/sup378.sh
