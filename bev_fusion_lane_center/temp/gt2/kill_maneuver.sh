pgrep -af "[m]aneuver.py e305" | cut -c1-100
kill $(pgrep -f "[m]aneuver.py e305") 2>/dev/null
sleep 3
uptime
