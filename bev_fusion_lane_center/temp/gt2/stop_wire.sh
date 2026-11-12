kill $(pgrep -f "[w]ire_run.sh") $(pgrep -f "[w]ire_sweep.py") 2>/dev/null
sleep 2
pgrep -af "[w]ire_" | cut -c1-90 || echo "wire sweep stopped"
