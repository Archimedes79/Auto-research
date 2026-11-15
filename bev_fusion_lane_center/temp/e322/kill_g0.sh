#!/bin/bash
# stop the G0 scoring jobs (script file: pkill -f inside wsl bash -c would kill its own shell)
pkill -f '[r]un_g0_promoted.sh'
pkill -f '[r]un_g0_e320.sh'
sleep 1
pkill -f '[j]wiring.py'
sleep 1
ps aux | grep -E "[j]wiring|[r]un_g0" | cut -c1-150
echo killed
