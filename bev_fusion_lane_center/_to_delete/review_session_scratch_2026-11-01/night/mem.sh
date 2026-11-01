cd /mnt/c/Auto-research/bev_fusion
grep -io 'mem[a-z_]*": *[0-9.]*' runs/t10_dual_s0/log.jsonl | sort | uniq -c | sort -rn | head -5
grep -i 'memory\|GiB' runs/t10_dual_s0/train.log | tail -3
tail -1 runs/t10_dual_s0/log.jsonl | cut -c1-400
free -g | head -2; nproc
