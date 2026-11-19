# 5. Ergebnisse von E389 gegen die passenden Baselines

**Kurz.** VAL, 873 Frames, zwei Seeds (s0 / s1 = Cache-Seed von B'), Einheit: abgedeckte gelieferte GT-Knoten (Kapitel 4). Die Differenzen sind **gepaart** (derselbe Frame, derselbe B'-Seed, Position gegen Position) mit Szenen-Bootstrap, 95 % in Klammern. Jede Zahl kommt aus den Karten von `jroute_eval.py score` und aus dem Urteil der Karte, nicht aus der Hand. **Urteil nach der vor dem Lauf geschriebenen Regel (KEEP / KEEP ausstehend (Kosten) / TIE (NO GAIN) / TIE (UNSAFE) / CUT / INCONCLUSIVE): {{E389_URTEIL}}.**

**Vorhersage gegen Ergebnis.** Vorhergesagt war m - z = +0.15 (Fenster +-0.05) im Modus *error*: {{E389_VORHERSAGE_TREFFER}}

Table: Kopf mit Karte (m) gegen den passenden Kopf ohne Karte (z), Modus *error* (die Zeile des Urteils), route_F1_road, VAL.

| Arm | Seed 0 | Seed 1 |
|---|---|---|
| z (ohne Karte) | {{E389_Z_ROAD_S0}} | {{E389_Z_ROAD_S1}} |
| z2 (Seed-Zwilling von z, Seed 0) | {{E389_Z2_ROAD}} | nicht trainiert |
| m2 (Seed-Zwilling von m, Seed 0) | {{E389_M2_ROAD}} | nicht trainiert |
| m (mit Karte) | {{E389_M_ROAD_S0}} | {{E389_M_ROAD_S1}} |
| **m - z** (gepaart, 95 %) | {{E389_D_ROAD_S0}} | {{E389_D_ROAD_S1}} |
| gerader Prior (Referenz, ohne Kopf) | 0.7272 | 0.7286 |
| geometrische Regel auf der fehlerhaften Route (Referenz, ohne Kopf) | 0.9516 | 0.9505 |

**Rauschen (vor dem Urteil).** |z - z2| = {{E389_Z_Z2}}, |m - m2| = {{E389_M_M2}}; die Streuung ist das Maximum beider ({{E389_SPREAD}}), die Grenze für INCONCLUSIVE ist 0.02, und die Marges der Vertrauensklausel sind max(Marge, Streuung).

Table: Kartenmodi, route_F1_road von m und z (die Regel liest *none* und *wrong* als Vertrauen); letzte Spalte: die Referenzregel ohne Kopf auf demselben Modus (Seed 0; bei *none* der gerade Prior).

| Modus | z s0 / s1 | m s0 / s1 | m - z s0 / s1 | Regel (Referenz) |
|---|---|---|---|---|
| exact (ohne Fehler) | {{E389_MODE_EXACT_Z}} | {{E389_MODE_EXACT_M}} | {{E389_MODE_EXACT_D}} | 0.9934 |
| error (Urteil) | {{E389_MODE_ERROR_Z}} | {{E389_MODE_ERROR_M}} | {{E389_MODE_ERROR_D}} | 0.9516 |
| gross (error + 5 m) | {{E389_MODE_GROSS_Z}} | {{E389_MODE_GROSS_M}} | {{E389_MODE_GROSS_D}} | 0.6637 |
| none (keine Karte) | {{E389_MODE_NONE_Z}} | {{E389_MODE_NONE_M}} | {{E389_MODE_NONE_D}} | 0.7272 (gerader Prior) |
| wrong (Route einer anderen Szene) | {{E389_MODE_WRONG_Z}} | {{E389_MODE_WRONG_M}} | {{E389_MODE_WRONG_D}} | 0.6137 |

Table: Gabelungs-Frames (nur Frames, in denen ein Routenstück mit mehr als einem Nachfolger innerhalb 40 m voraus endet; Zahl der Frames und Szenen in der Zeile), route_F1_road.

| Modus | Frames / Szenen | z s0 / s1 | m s0 / s1 | m - z s0 / s1 |
|---|---|---|---|---|
| error | {{E389_FORK_N}} | {{E389_FORK_Z}} | {{E389_FORK_M}} | {{E389_FORK_D}} |
| wrong | {{E389_FORK_N}} | {{E389_FORKW_Z}} | {{E389_FORKW_M}} | {{E389_FORKW_D}} |

Table: Spur-Ebene und Nebenwerte (immer berichtet, nicht entscheidend außer Klausel 4).

| Größe | z s0 / s1 | m s0 / s1 |
|---|---|---|
| route_F1_lane (Modus error) | {{E389_LANE_Z}} | {{E389_LANE_M}} |
| Erfolg@30 m je Frame (Modus error) | {{E389_SUCC_Z}} | {{E389_SUCC_M}} |
| Markierungsrate auf den FALSCHEN gelieferten Token (kein GT-Knoten, ungewertet; ein Planer bekommt sie trotzdem) | {{E389_FALSE_Z}} | {{E389_FALSE_M}} |

**Kalibrierung.** Anteil der Token auf der Route pro Bin von p_route (m, Modus error): {{E389_KALIBRIERUNG}}

**Die Regel, Klausel für Klausel.**

Table: Klauseln der vor dem Lauf geschriebenen Regel und ihr Stand (Quelle: das Ergebnis-Kapitel der Karte).

| Klausel | verlangt | gemessen | erfüllt |
|---|---|---|---|
| (1) Registrierung | m - z >= +0.10 im Modus error, beide Seeds, untere Grenze über 0 | {{E389_K1_MESS}} | {{E389_K1_OK}} |
| (2) Vertrauen | none: nicht mehr als 0.01 unter z; wrong: nicht mehr als 0.03 (alle Frames) und 0.05 (Gabelungen) unter z; jede Marge mindestens so groß wie die Streuung der Zwillinge | {{E389_K2_MESS}} | {{E389_K2_OK}} |
| (3) Gabelungen | m - z >= +0.10 auf den Gabelungs-Frames, untere Grenze über 0 | {{E389_K3_MESS}} | {{E389_K3_OK}} |
| (4) Spur | F1_lane nicht mehr als 0.02 unter z | {{E389_K4_MESS}} | {{E389_K4_OK}} |
| (5) Abbruch | kein kill | {{E389_K5_MESS}} | {{E389_K5_OK}} |
| (6) Kosten | höchstens 1.0 ms zusätzlich, nur durch einen ruhigen Zeitlauf entschieden (sonst „KEEP ausstehend (Kosten)“) | {{E389_K6_MESS}} | {{E389_K6_OK}} |

**Kontrolle ohne Knoten-Token (explorativ, kein Urteil).** Die Arme zg und mg lesen nur Geometrie (ohne `tok`, 21,130 Parameter): {{E389_GEO}}

**Kosten.** Kopf: 33,738 Parameter. Zeit pro Frame: {{E389_MS_ZEILE}}

**Was das Ergebnis für den Champion bedeutet.** {{E389_BEDEUTUNG}}
