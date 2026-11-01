for i in $(seq 1 55); do kill -0 $1 2>/dev/null || break; sleep 10; done
kill -0 $1 2>/dev/null && echo "still running" || echo "gone"; free -g | head -2; date
