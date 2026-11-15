cd ~/work/CRN || exit 1
echo ---gen_radar_pv; sed -n 1,200p scripts/gen_radar_pv.py | head -120
echo ---exp_val_parts; grep -n "return_depth\|return_radar_pv\|data_root\|ann_file\|info_path\|nuscenes_infos\|batch\|key_idxes\|eval\|class_names\|TF32\|tf32\|amp\|half\|fp16\|cudnn" exps/det/CRN_r50_256x704_128x128_4key.py | head -60
