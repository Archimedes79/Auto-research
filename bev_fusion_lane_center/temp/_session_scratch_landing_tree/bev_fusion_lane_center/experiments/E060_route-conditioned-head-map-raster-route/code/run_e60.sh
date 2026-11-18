#!/usr/bin/env bash
# E60: Routen-bedingter dichter Kopf: Hauptlauf (Route + alle Karten-Spuren) und Kontrolle (nur Route)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
V=$1
if [ "$V" = "main" ]; then
  ./run_gcar.sh route_head.py --name e60_main --steps 4000 --maps route+lanes 2>&1 | grep -E "^\[|FERTIG|Traceback|Error|ROLLBACK|Schritt (1000|2000|3000|4000)/"
else
  ./run_gcar.sh route_head.py --name e60_route_only --steps 4000 --maps route 2>&1 | grep -E "^\[|FERTIG|Traceback|Error|ROLLBACK|Schritt (1000|2000|3000|4000)/"
fi
