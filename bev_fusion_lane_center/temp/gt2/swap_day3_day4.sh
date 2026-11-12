cd "$(dirname "$0")/../.."
bash -n temp/day4/day4.sh || exit 1
echo "before:"; pgrep -af "[d]ay3/day3.sh|[j]ends_train.py --name e305" | cut -c1-110
kill $(pgrep -f "[d]ay3/day3.sh") 2>/dev/null
sleep 2
echo "after:"; pgrep -af "[d]ay3/day3.sh" || echo "day3 stopped"; pgrep -af "[j]ends_train.py --name e305_e_s" | cut -c1-100
