#!/bin/bash
# stop the waiting day13 chain (it has done nothing since the restart): script file, pattern with brackets so that pkill does not kill its own shell
pkill -f '[d]ay13.sh'
sleep 1
ps aux | grep -E "[d]ay13" | cut -c1-120
echo "day13 stopped"
