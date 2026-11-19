# A. Wie man es wiederholt

Alle Befehle laufen aus `bev_fusion_lane_center/`; die Läufe mit Einfluss auf ein Urteil gehen durch `run.py` (Karte vorher, Ledger-Zeile, Tag). Daten und Gewichte sind nicht im Paket (Kapitel 9).

**Daten (CPU).** Die gecachten Token von B' (`runs/e365_cache/...`, Seeds 0 / 1) und die gelieferte Menge des Stufe-1-Kopfes (`runs/e373_cb_s<S>/token_val.pkl`) müssen vorhanden sein. Die Routen-GT und die Routen-Token baut die Datenschicht in `runs/e389_route/{val,train}` (VAL 873 Frames, TRAIN 8414); die Daten-Prüfung und die Selbsttests: `bash tools/env/_py14.sh test_lane_route.py`, `test_jroute.py`.

**Training.** Pro Arm und Seed (etwa 5 Minuten, Karte vorher, `--require-card`):

```
bash tools/env/_pyr.sh jroute_train.py --name e389_<arm>_s<S> --seed S --arm m|z --steps 3000 --bs 8 --lr 1e-3 --workers 4 --ident e292_a [--inputs geo] [--init-seed 1]
```

Arme: z und m auf den Seeds 0 und 1, z2 (z mit `--init-seed 1`, Seed 0), explorativ zg und mg (`--inputs geo`).

**Auswertung (VAL, TEST nie).**

```
bash tools/env/_py14.sh jroute_eval.py dump  --run e389_m_s0 --seed 0 --mode error      # je Modus: exact error none wrong gross
bash tools/env/_py14.sh jroute_eval.py score --run e389_m_s0 --seed 0 --mode error --ref-run e389_z_s0 --champion e373_cb_s0
bash tools/env/_py14.sh jroute_eval.py rules --seed 0 --split val --modes exact,error      # Referenzzeilen, ohne Kopf
```

**Urteil.** Von Hand nach der Regel der Karte (Kapitel 4); die Karten der Auswertung liegen in `runs/<run>/route_card_<mode>[_vs_<z>].json`.

**Zahlen erreichen, bevor man dem Urteil traut.** {{E389_ZAHLEN_ZIEL}}

**Bericht.** `python reporter/reporter.py build results/E2E_Report_E389_2026-11-20 --strict` (scheitert, solange ein PENDING-Zeichen oder ein Platzhalter ohne Wert bleibt), `check ... --strict`, `pack`; die Zeichnung des Kopfes: `python tools/draw_route_head.py <svg>` (gcar-Python mit torch). Zusätzlich `python autoresearch/tools/check_numbers.py` auf den Kapiteln gegen die Karten und das Ledger.

**Bekannte Fallstricke.** Die Routen-Tabelle ist auf 64 Stützstellen gekappt (Kapitel 4); die Daten-Caches heißen nach der Kappung neu (v4), alte Referenzkarten dürfen nicht mit neuen Köpfen gemischt werden; `_py14.sh` verweigert ein Skript, dessen Name unter `temp/` doppelt vorkommt; ein Lauf durch `_py14.sh` kann trotz Traceback mit Exit 0 enden: das Urteil des Laufs ist die letzte „ok“-Zeile, nicht der Exit-Code.
