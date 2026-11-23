# A. Wie man die Zahlen erneut erzeugt

Alle Befehle laufen aus `bev_fusion_lane_center/` (WSL, conda-Umgebung gcar; `tools/env/_py14.sh` wechselt das Verzeichnis, daher bei `--out` absolute Pfade, in Git Bash `MSYS_NO_PATHCONV=1`). Das Code-Paket enthält die Quellen; Gewichte, Caches und Dumps liegen in den Run-Ordnern des Repositorys (Kapitel 10).

| Zahl | Befehl (Karte nennt die vollständige Zeile) |
|---|---|
| E390: die Sonden von Version 7 | `bash tools/env/_py14.sh sdgraph_probe.py read --arm g7 --seed S` (Encoder `runs/e390_g7_s<S>`; Daten: `lane_sdgraph_data.py build --split val`) |
| E391: Nulls, Training, Lesen | `bash tools/env/_py14.sh mapassoc_probe.py nulls --seed S [--split val\|holdout]`, `... train --arm F1 --seed S`, `... read --arm F1 --seed S --split val` |
| E392: Arme und Lesen | `jarm.py`-Ids `E392.<Schritt>_<Arm>_s<S>` (die Karte nennt die Befehle); Lesen mit `jtoken_eval.py` (Dekodierung D115) |
| die Beispielbilder von Kapitel 6 | `bash tools/env/_py14.sh map_assoc_fig.py --seed 0 --out <abs. Ordner>` (Regel zur Frameauswahl im Kopf des Skripts; `--selftest`) |
| die Diagramme der Kapitel 5 und 7 | `bash tools/env/_py14.sh report_charts.py --out <abs. Ordner>` (liest die Karten; `--selftest`) |
| dieser Bericht | `python reporter.py build results/E2E_Report_2026-11-23 --strict`, danach `check` und `pack`; `autoresearch/tools/check_numbers.py` auf das HTML mit den Quellen aus `report.json` |
| die Tests | `bash tools/env/_py14.sh training/grid/test_sdgraph.py`, `test_mapassoc.py`, `test_maperror.py` (je einzeln; die Sammlung `tools/maintenance/run_all_tests.sh` ruft die älteren Tests auf) |
