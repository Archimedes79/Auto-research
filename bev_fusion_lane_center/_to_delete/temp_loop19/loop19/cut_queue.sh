#!/bin/bash
# cut the loop-19 GPU queue: waiting launchers first, then the E244.a0 training (its launcher 37712 sees the exit)
kill 37838 37832 37787 37741
sleep 2
kill 56248
sleep 5
ps -eo pid,etime,args | grep -E '[r]un.py --gpu-max-mb|[p]16_train' | cut -c1-110
