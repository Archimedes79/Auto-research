cd /mnt/c/Auto-research/bev_fusion
ls -la data/cache | head
timeout 100 find -L data/cache/gt -maxdepth 1 -name "*.npz" -size -2k | head
ls _to_delete/gt_cache_partial -la
