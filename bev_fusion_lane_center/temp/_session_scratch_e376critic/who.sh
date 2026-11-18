#!/bin/bash
for p in $(pgrep -f test_jpast_train.py); do echo "$p ppid=$(ps -o ppid= -p $p) $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-220)"; done
