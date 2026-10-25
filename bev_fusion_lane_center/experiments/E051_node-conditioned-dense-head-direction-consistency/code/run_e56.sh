#!/usr/bin/env bash
# E56: Richtungs-Konsistenz als zusaetzliches Annahmekriterium der Verbinder (E53): mittlerer Kosinus zwischen Kurventangente und vorhergesagter Richtungskarte laengs der Kurve >= cos_c; VAL-Durchlauf
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
W=${W:-3}
for TH in 0.10 0.50; do
  for CC in 0.3 0.6 0.8; do
    P="{\"tau\":0.5,\"par\":{\"cut\":1,\"r_s\":16.0,\"r_b\":8.0,\"cos_min\":0.5,\"d_c\":1.5,\"prune\":6.0,\"synth\":$TH,\"cos_c\":$CC}}"
    echo "== $(date +%H:%M:%S) synth tau_h $TH cos_c $CC (VAL)"
    ./run_gcar.sh node_assemble.py --graphs e43/graphs_rule_e41_A --nodes e44_nodes --heat e41_A --sets val --out e56_h${TH}_c$CC --par_json "$P" --workers $W 2>&1 | grep -E "^\[(val|test)\] \{|Traceback|Error"
  done
done
echo "== $(date +%H:%M:%S) done"
