ps -eo pid,etime,cmd | grep -E "[d]ay3|[t]hr_sweep|[t]hr_run|[j]ends|[j]maps" | cut -c1-120
