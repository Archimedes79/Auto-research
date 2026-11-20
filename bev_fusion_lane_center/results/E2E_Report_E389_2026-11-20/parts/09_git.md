# 9. Was in git liegt und was nicht

**Kurz.** In git (Hauptzweig, lokal; nichts wurde gepusht, D128) liegen der Code, seine Tests und die Textprotokolle. Nicht in git liegen Daten, Caches, Checkpoints und Rohergebnisse. Die Tabelle gilt für den Stand beim Bau des Berichts: Die Auswertungsschicht, die Datenschicht mit der Kappung 160 und das Modell sind committet (`d45531b0`), die Ledger-Zeilen der 18 Läufe (`7f1ddc89`), die Karte mit dem Ergebnis (`bfc79976`) und die Berichtsquellen mit der Zeichnung (`tools/draw_route_head.py`) und dem Beispielbild-Skript (`tools/route_frames_fig.py`).

Table: Was wo liegt.

| Was | Wo | in git |
|---|---|---|
| Modellschicht: `RouteHead`, `route_loss` (33,738 / 21,130 Parameter) | `training/grid/jroute.py`, Selbsttests und Mutanten in `training/grid/test_jroute.py` | ja (Commit `b0e2fa73fb`) |
| Trainer (Arme m / z, `--inputs full|geo`, `--smoke`) | `training/train/jroute_train.py` | ja (derselbe Commit) |
| Datenschicht: Routen-GT, Routen-Token, Kartenmodi, Gabelungs-Frames, Daten-Prüfung | `training/data/lane_route_data.py`, `training/grid/test_lane_route.py` | ja (Commit `9f774c54fa`) |
| Auswertungsschicht: Dump, Score, Referenzregeln, gepaarter Bootstrap | `evaluation/jroute_eval.py`, `evaluation/tests/test_jroute_eval.py` | ja (Commit `d45531b0`) |
| Karten: E389 (Version 3b, Commit `103b295c42`), E388, E381; Entscheidungen D127, D128 | `experiments/E389_*`, `E388_*`, `E381_*`, `experiment_execution/decisions/` | ja |
| Ledger-Zeilen der Läufe | `experiment_execution/LEDGER.csv` | ja (Commit `7f1ddc89`) |
| Dieser Bericht: Quellen (Kapitel, Zeichnung, Zeichenskript, `report.json`) | `results/E2E_Report_E389_2026-11-20/` | ja (Text und Skript); **nicht** das erzeugte HTML und das Code-Paket (Binärdateien, gesendet) |
| Daten-Caches (Routen-GT, Token-Caches, abgeleitete Daten) | `runs/e389_route/{val,train}`, `runs/e365_cache/` | **nein** (abgeleitet, wiederherstellbar) |
| Referenzzeilen (Regeln) | `runs/e389_rules/rules_card_val_s<S>.json` | nein (abgeleitet) |
| Läufe, Gewichte, Dumps, Karten der Auswertung | `runs/e389_*` | **nein** (Checkpoints und Rohergebnisse; die Zahlen des Berichts kommen aus ihnen) |
| Scratch (Sonden, Patch-Skripte, Notizen) | `temp/e389/`, `temp/e388b/` | nein (Scratch, nie gelöscht; ein Lauf, der es braucht, hat ein Tag) |
| Läufe durch `run.py` | Git-Tags je Lauf | lokal |

**Wie man die Zahlen erneut erzeugt:** Anhang A. **Was dieser Bericht nicht liefert:** Gewichte (zu groß für das Paket; sie liegen in den Run-Ordnern) und TEST-Zahlen (keine).
