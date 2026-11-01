for v in ~/data/nuscenes_full ~/data/nuscenes; do echo "== $v"; ls -la $v | awk '{print $9, $10, $11}'; done
ls ~/data/
ls -la ~/data/nuscenes_full/samples | head; ls ~/data/nuscenes_full/sweeps
ls /mnt/d/Auto-research/NuScenes/nuscenes_full/samples
for d in ~/data/nuscenes_full/* ~/data/nuscenes/*; do [ -e "$d" ] || echo "BROKEN $d"; done
