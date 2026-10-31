#!/bin/bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "--- python processes"
ps -eo pid,etime,pcpu,args | grep -E "[p]ython" | cut -c1-120
