ps -eo pid,etime,cmd | grep -E "[j]ends_train|[j]maps_train|[j]ends_eval|[j]pair_pr|[j]rule_control|[j]splits|[p]air_diag|[n]ight3|[d]ay1|[d]ay2|[r]un.py" | cut -c1-140
uptime
free -g | head -2
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
