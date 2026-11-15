cd ~/work/CRN || exit 1
sed -n 433,470p exps/base_exp.py
echo ---; sed -n 226,300p evaluators/det_evaluators.py
echo --- infos dir; ls -la --time-style=long-iso /mnt/c/Auto-research/bev_fusion_lane_center/data/nuscenes_additions/lcr_crn/ | head; ls /mnt/c/Auto-research/bev_fusion_lane_center/data/nuscenes_additions/ | head
cat /mnt/c/Auto-research/bev_fusion_lane_center/data/nuscenes_additions/lcr_crn/README* 2>/dev/null | head -20
ls ~/data/nuscenes_full/radar_pv_filter | head -3
