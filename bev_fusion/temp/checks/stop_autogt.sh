#!/bin/bash
# stop a running autogt report (the pattern in brackets so that pkill does not match its own shell)
pkill -f "[a]utogt report" || echo "nothing running"
sleep 3
ps aux | grep "[a]utogt report" | wc -l
free -m | sed -n 2,3p
