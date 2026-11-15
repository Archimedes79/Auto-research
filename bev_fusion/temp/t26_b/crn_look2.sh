cd ~/work/CRN || exit 1
echo ---so; find build ops -name '*.so' | head; ls ops
echo ---gen_info; cat scripts/gen_info.py | head -80
echo ---gen_radar_bev; sed -n 1,90p scripts/gen_radar_bev.py
