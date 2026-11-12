#!/bin/bash
while pgrep -f "[r]un_config.py" >/dev/null; do sleep 3; done
echo finished
