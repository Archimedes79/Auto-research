cd ~/work/CRN || exit 1
ls scripts scripts/* | head -30
echo ---exps; ls exps exps/* | head -30
echo ---outputs; ls outputs | head
echo ---env
ls ~/miniconda3/envs/crn/lib/python3*/site-packages 2>/dev/null | grep -i -E '^(torch|mmcv|mmdet|mmseg|pytorch_lightning|nuscenes|mmdet3d)' | head -20
~/miniconda3/envs/crn/bin/python -c 'import sys; print(sys.version)' 2>&1 | head -2
echo ---yaml; head -50 CRN.yaml
echo ---git; git log --oneline | head -3
ls ~/work
