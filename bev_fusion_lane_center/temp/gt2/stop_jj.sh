cd "$(dirname "$0")/../.."
pids=$(pgrep -f "[j]maps_train.py --name e286_jj" | tr '\n' ' ')
echo "killing $pids"
kill -TERM $pids
sleep 8
pgrep -af "[j]maps_train.py --name e286_jj" | cut -c1-100 || echo "none left"
