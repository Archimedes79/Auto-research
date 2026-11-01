for i in $(seq 1 50); do grep -q EXIT /tmp/t11probe/train_selftest.log && break; sleep 10; done
tail -c 1500 /tmp/t11probe/train_selftest.log; free -g | head -2
