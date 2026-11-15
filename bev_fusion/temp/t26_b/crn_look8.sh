cd ~/work/CRN || exit 1
ls -la ~/data/nuscenes_full/*.pkl ; ls ~/data/nuscenes_full/radar_pv_filter | wc -l; ls ~/data/nuscenes_full/radar_bev_filter | wc -l
ls -la --time-style=long-iso ~/data/nuscenes_full | head -20
echo --- evaluator; grep -n "json\|output\|class \|def \|dump\|submission\|result" evaluators/det_evaluators.py | head -60
echo --- base_cli; cat exps/base_cli.py | head -70
echo --- datasets_defaults; sed -n 166,235p exps/base_exp.py
