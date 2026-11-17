ps -eo pid,etime,args | grep -E "python|run.py|train|nvidia|jends|jmaps|day[0-9]" | grep -v grep | cut -c1-140 | head -30
echo ---; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo ---; uptime
