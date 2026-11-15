cd ~/work/CRN || exit 1
ls datasets datasets/*
echo; grep -n "radar\|depth\|lidar\|ann_file\|data_root\|is_train\|key_idxes\|def " datasets/nusc_det_dataset.py | head -120
