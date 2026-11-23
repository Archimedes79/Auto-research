# 10. Was in git liegt und was nicht

**Kurz.** Code und Texte der Experimente liegen im Repository, mit Tests; Daten, Gewichte, Caches, Dumps und die erzeugten Bilder und Berichte nicht. Der Bericht und das Code-Paket sind erzeugte Dateien (nicht in git); die Quellen, aus denen sie gebaut werden, sind es.

Table: wo was liegt.

| Was | Pfad | in git |
|---|---|---|
| E388, E381: Karten, Ergebnisse | `experiments/E388_*`, `E381_*` | ja |
| E389 / E389b: Kopf, Trainer, Daten, Auswertung, Tests; Karten; Bericht vom 2026-11-20 | `training/grid/jroute.py`, `training/train/jroute_train.py`, `training/data/lane_route_data.py`, `evaluation/jroute_eval.py`; `experiments/E389_*`, `E389b_*` | ja (Quellen); der erzeugte Bericht nicht |
| E390: die Kodierung, der Encoder, die Sonden, der Test, die Karte | `training/data/lane_sdgraph_data.py`, `training/grid/jsdgraph.py`, `evaluation/sdgraph_probe.py`, `training/grid/test_sdgraph.py`; `experiments/E390_*` | ja |
| E391: Labels, Köpfe, Sonde, Test, Karte | `training/data/lane_mapassoc_data.py`, `training/grid/jmapassoc.py`, `evaluation/mapassoc_probe.py`, `training/grid/test_mapassoc.py`; `experiments/E391_*` | ja |
| E392: die Runden im Token-Kopf (Optionen von `jtoken`), Trainer, Karte | `training/grid/jtoken.py`, `training/train/jtoken_train.py`, `evaluation/jtoken_eval.py`; `experiments/E392_*` | ja |
| E393: Bauteile 1 - 5, Karte (Entwurf) | `training/train/jlive_train.py` (Optionen), `training/trainer.py` (`lr_mult`); `experiments/E393_*` | ja |
| E394: Fehlerkatalog, Test, Sonde, Karte (Entwurf, pausiert) | `training/data/lane_maperror_data.py`, `training/grid/test_maperror.py`, `evaluation/maperror_probe.py`; `experiments/E394_*` | ja (Commit `8ef6e28be8`) |
| Bild- und Diagrammwerkzeuge dieses Berichts | `tools/map_assoc_fig.py`, `tools/report_charts.py` (und die Werkzeuge der Berichte vom 18. und 20.11.) | ja (mit Selbsttest) |
| Ledger-Zeilen der Läufe, Entscheidungen D129 - D138 | `experiment_execution/LEDGER.csv`, `DECISIONS.md` | ja |
| Läufe, Gewichte, Dumps, Score-Karten, Caches | `runs/e390_*`, `runs/e391_*`, `runs/e392_*`, `runs/e389b_*`, `runs/e365_cache/`, `runs/e391_assoc/`, `runs/e390_sdgraph/` | **nein** (Checkpoints und Rohergebnisse werden nie gelöscht, Caches und Dumps sind abgeleitet; die Zahlen des Berichts kommen aus ihnen) |
| Dieser Bericht und das Code-Paket | `results/E2E_Report_2026-11-23/` | Texte und `report.json` ja; **nicht** das erzeugte HTML, das Zip und die Bilder |
| Scratch (Ketten, Urteils-Skripte der Versionen) | `temp/e390/`, `temp/e391/`, `temp/e392_pre/` | nein (Scratch; von den Lauf-Tags gehalten, `lcr/E390.*`, `lcr/E391.*`, `lcr/E392.*`) |

**Wie man die Zahlen erneut erzeugt:** Anhang A. **Was der Bericht nicht liefert:** Gewichte (zu groß für das Paket; sie liegen in den Run-Ordnern) und TEST-Zahlen (keine).
