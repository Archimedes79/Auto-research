# A. Wiederholung

Die Zahlen des Berichts stammen aus den Experimentkarten (`bev_fusion_lane_center/experiments/E39*/experiment.md`, `bev_fusion/experiments/t25_one_frame_one_scan/`) und aus den kleinen Ergebniskarten unter `data/runs/` dieses Berichts. Bericht neu bauen und prüfen:

```
python reporter/reporter.py build bev_fusion_lane_center/results/E2E_Report_2026-11-25 --strict
python reporter/reporter.py check bev_fusion_lane_center/results/E2E_Report_2026-11-25 --strict
```

Einen Lauf wiederholen (Beispiel):

```
python3 autoresearch/tools/recipe.py show E393.n_s0
python3 autoresearch/tools/recipe.py repeat E393.n_s0
```

Die Umgebung ist `gcar` (`bev_fusion_lane_center/tools/env/environment_gcar.yml`); die abgeleiteten Daten und Caches macht `bev_fusion_lane_center/data/rebuild.json`.
