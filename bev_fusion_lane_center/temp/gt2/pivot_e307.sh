cd "$(dirname "$0")/../.."
echo "before:"; pgrep -af "[d]ay5/day5.sh|[j]maps_train.py --name e304" | cut -c1-90 | head -8
kill $(pgrep -f "[d]ay5/day5.sh") 2>/dev/null
sleep 1
kill -TERM $(pgrep -f "[j]maps_train.py --name e304") 2>/dev/null
sleep 8
echo "after:"; pgrep -af "[d]ay5/day5.sh|[j]maps_train.py --name e304" | cut -c1-90 | head -4 || true
tail -n 3 temp/day4/day4.log | cut -c1-100
