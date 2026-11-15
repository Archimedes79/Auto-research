ls ~/data/nuscenes_full | head -20
echo --- samples; ls ~/data/nuscenes_full/samples/ 2>/dev/null
echo --- sweeps; ls ~/data/nuscenes_full/sweeps/ 2>/dev/null
for d in CAM_FRONT CAM_BACK CAM_FRONT_LEFT RADAR_FRONT LIDAR_TOP; do echo "$d samples: $(ls ~/data/nuscenes_full/samples/$d 2>/dev/null | wc -l)  sweeps: $(ls ~/data/nuscenes_full/sweeps/$d 2>/dev/null | wc -l)"; done
ls ~/data/nuscenes_full/v1.0-trainval | head
cd ~/work/CRN; sed -n 300,330p exps/base_exp.py; sed -n 715,800p datasets/nusc_det_dataset.py
