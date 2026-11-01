# stop the orphaned workers of the OOM-killed t10_dual_s1 training and my queue script (both mine); then show the memory users
pkill -f "[b]ev_queue1.sh"
pkill -f "[r]uns/t10_dual_s1/code/train.py"
sleep 5
pgrep -fa "[r]uns/t10_dual_s1/code/train.py" | head -3
nvidia-smi --query-gpu=memory.used --format=csv,noheader
free -g | head -2
ps -eo pid,rss,etime,args --sort=-rss | head -12 | cut -c1-170
