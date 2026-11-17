cd "$(dirname "$0")/../.."
# the old chains only wait (no training yet): stop them, start day3
pgrep -af "[d]ay1/day.sh|[d]ay2/day2b.sh" | cut -c1-80
kill $(pgrep -f "[d]ay1/day.sh|[d]ay2/day2b.sh") 2>/dev/null
sleep 2
pgrep -af "[d]ay1/day.sh|[d]ay2/day2b.sh" || echo "old chains stopped"
pgrep -af "[j]ends_train|[j]maps_train" || echo "no training running"
