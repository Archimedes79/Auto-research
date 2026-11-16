nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader
ps aux | grep python | grep -v grep | cut -c1-200
uptime
