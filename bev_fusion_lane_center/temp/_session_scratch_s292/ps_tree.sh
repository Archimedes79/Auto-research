#!/bin/bash
ps -eo pid,ppid,etime,args --forest | grep -B1 -A6 "434595" | grep -v grep | cut -c1-170
echo ---
ps -eo pid,ppid,etime,args | grep -E "run.py --id E318.u_s1|E318.u_s1|start_pairs.sh" | grep -v grep | cut -c1-170
