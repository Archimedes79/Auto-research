#!/bin/bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
uptime
ps -eo pid,etime,args --sort=-%cpu | grep -E "python" | grep -v grep | head -8 | cut -c1-150
