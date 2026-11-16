cd "$(dirname "$0")/../.."
echo "before:"; pgrep -af "[d]ay8/day8.sh|[j]ends_train.py --name e298" | cut -c1-80 | head -6
kill $(pgrep -f "[d]ay8/day8.sh") 2>/dev/null
sleep 1
kill -TERM $(pgrep -f "[j]ends_train.py --name e298") 2>/dev/null
sleep 8
echo "after:"; pgrep -af "[d]ay8/day8.sh|[j]ends_train.py --name e298" | cut -c1-80 | head -4
bash -n temp/day9/day9.sh && echo day9-ok
