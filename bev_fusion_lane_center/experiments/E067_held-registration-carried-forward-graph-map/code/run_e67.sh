#!/usr/bin/env bash
# E67: Ausfall mitten in der Fahrt - Registrierung halten. Hauptlauf (Bias 1 m, kein Drift), Drift-Lauf, groesserer Bias, transienter Ausfall (mittleres Drittel), Abbildungslauf (nur Nacht und Unschaerfe, speichert die Registrierungen)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
./run_gcar.sh hold_eval.py --sigma 1.0 --workers 6 --out e67_s1 2>&1 | grep -E "^\[|Tor-|Mittel|Traceback|Error"
./run_gcar.sh hold_eval.py --sigma 1.0 --drift 0.1 --drift_deg 0.1 --workers 6 --out e67_drift 2>&1 | grep -E "^\[|Tor-|Mittel|Traceback|Error"
./run_gcar.sh hold_eval.py --sigma 2.0 --workers 6 --out e67_s2 2>&1 | grep -E "^\[|Tor-|Mittel|Traceback|Error"
./run_gcar.sh hold_eval.py --onset 0.33 --end 0.67 --sigma 1.0 --workers 6 --out e67_trans 2>&1 | grep -E "^\[|Tor-|Mittel|Traceback|Error"
./run_gcar.sh hold_eval.py --conds dark,blur --sigma 1.0 --workers 4 --out e67_viz 2>&1 | grep -E "^\[|Tor-|Mittel|Traceback|Error"
./run_gcar.sh hold_eval.py --loco 1 --sigma 1.0 --workers 8 --out e67_loco 2>&1 | grep -E "^\[|Tor-|Mittel|Traceback|Error"
./run_gcar.sh hold_eval.py --scenecv 1 --sigma 1.0 --workers 8 --out e67_scv 2>&1 | grep -E "^\[|Tor-|Mittel|Traceback|Error"
echo "== $(date +%H:%M:%S) done"
