#!/bin/bash
while pgrep -f "[r]un_all_tests" >/dev/null; do sleep 5; done
echo tests finished
