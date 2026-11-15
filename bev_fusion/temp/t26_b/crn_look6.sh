cd ~/work/CRN || exit 1
grep -n "def test_step\|def eval_step\|def predict\|def test_dataloader\|def val_dataloader\|ann_file\|nuscenes_infos\|data_root\|DATA\|evaluate\|json\|results\|NuScenesEval\|nuscenes.eval\|def \|use_cbgs\|batch_size\|num_workers" exps/base_exp.py | head -80
echo ----; sed -n 805,900p datasets/nusc_det_dataset.py
echo ---- evaluator; ls evaluators evaluators/*; grep -n "TF32\|allow_tf32\|matmul" -r --include=*.py . 2>/dev/null | grep -v "mmdetection3d\|build/" | head
