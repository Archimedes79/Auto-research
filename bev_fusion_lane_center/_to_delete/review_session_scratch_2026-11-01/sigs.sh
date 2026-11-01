cd /mnt/c/Auto-research/bev_fusion
timeout 100 ls data/cache/gt | awk -F_ '{print $NF}' | sort | uniq -c | sort -rn | head
du -sh data/cache/gt 2>/dev/null | tail -1
ls data/autogt 2>/dev/null
