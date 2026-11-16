# prints one line when the GPU is free: no compute process and < 2 GB used, no GPU run.py job
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  jobs=$(ps aux | grep '[r]un.py' | grep -c -- '--gpu')
  if [ "$apps" -eq 0 ] && [ "$used" -lt 2000 ] && [ "$jobs" -eq 0 ]; then echo "GPU free: $used MiB, $(date +%H:%M)"; exit 0; fi
  sleep 60
done
