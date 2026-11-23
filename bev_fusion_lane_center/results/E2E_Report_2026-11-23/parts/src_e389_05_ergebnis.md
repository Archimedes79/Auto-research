# 5. Ergebnisse von E389 gegen die passenden Baselines

**Kurz.** VAL, 873 Frames, zwei Seeds (s0 / s1 = Cache-Seed von B'), Einheit: abgedeckte gelieferte GT-Knoten (Kapitel 4). Die Differenzen sind **gepaart** (derselbe Frame, derselbe B'-Seed) mit Szenen-Bootstrap, 95 % in eckigen Klammern. Jede Zahl wird aus den Karten von `jroute_eval.py score` gelesen, nicht getippt. **Urteil nach der vor dem Lauf geschriebenen Regel (von Hand, Karte v3b): TIE (NO GAIN).** Die Karte schlägt den Kopf ohne Karte im Modus *error* um etwa +0.095, die Hürde war +0.10.

**Das Wichtigste in drei Sätzen.** (1) Die Karte hilft wirklich: m liegt vor z um {{d_road_error_s0}} / {{d_road_error_s1}} F1 an der Fahrbahn (Untergrenzen {{lo_road_error_s0}} / {{lo_road_error_s1}}), und m liegt sogar **über** der geometrischen Regel auf der fehlerhaften Route (0.9516): eine mögliche Lesung (nicht getestet): der Kopf verbindet Karte und eigene Merkmale, was die feste Regel allein auf der Karte nicht tut; die Regel ist keine Obergrenze (D88). (2) Die Hürde von +0.10 wird auf beiden Seeds knapp verfehlt, und auch der Seed-Zwilling m2 liegt mit {{d2_road}} darunter: die drei Köpfe liegen innerhalb von 0.004 voneinander. (3) Der Hauptbefund ist der Kopf **ohne** Karte: z erreicht {{z_road_s0}} / {{z_road_s1}} gegen den geraden Prior 0.7272 / 0.7286. Die Token tragen den größten Teil der Routenzuordnung schon (laut der Kontrolle ohne `tok` vor allem Position, Richtung und Flags, `tok` selbst bringt etwa 0.02), die Karte fügt etwa 0.096 hinzu, nicht die vorhergesagten 0.15.

**Vorhersage gegen Ergebnis.** Vorhergesagt war m - z = +0.15 (Fenster +-0.05) im Modus *error*; gemessen ist der Mittelwert über die Seeds {{mean_diff}}: **verfehlt** (zu hoch geschätzt, weil z stärker ist als angenommen: die Karte nahm einen Kopf ohne Karte zwischen dem geraden Prior 0.727 und 0.78 an).

Table: Kopf mit Karte (m) gegen den passenden Kopf ohne Karte (z), Modus *error* (die Zeile des Urteils), route_F1_road, VAL, 873 Frames.

| Arm | Seed 0 | Seed 1 |
|---|---|---|
| gerader Prior (Referenz, ohne Kopf) | 0.7272 | 0.7286 |
| z (ohne Karte) | {{z_road_s0}} | {{z_road_s1}} |
| z2 (Seed-Zwilling von z, Seed 0) | {{z2_road}} | nicht trainiert |
| m (mit Karte) | {{m_road_error_s0}} | {{m_road_error_s1}} |
| m2 (Seed-Zwilling von m, Seed 0) | {{m2_road}} | nicht trainiert |
| **m - z** (gepaart, 95 %) | {{d_road_error_s0}} [{{lo_road_error_s0}}, {{hi_road_error_s0}}] | {{d_road_error_s1}} [{{lo_road_error_s1}}, {{hi_road_error_s1}}] |
| m2 - z (gepaart, 95 %) | {{d2_road}} [{{lo2_road}}, {{hi2_road}}] | nicht trainiert |
| geometrische Regel auf der fehlerhaften Route (Referenz, ohne Kopf) | 0.9516 | 0.9505 |
| geometrische Regel auf der exakten Route (Referenz, ohne Kopf) | 0.9934 | 0.9930 |

**Rauschen (vor dem Urteil).** |z - z2| = {{abs_z_z2}}, |m - m2| = {{abs_m_m2}}; die Streuung ist das Maximum beider, {{spread}} (Quelle: `runs/eval/e389_verdict.json`). Die Grenze für INCONCLUSIVE ist 0.02: die Hürden bleiben, wie geschrieben, und die Marges der Vertrauensklausel sind max(Marge, Streuung), also unverändert.

Table: Weitere Zahlen des Modus *error* (nicht entscheidend außer Klausel 4): Spur-Ebene, Gabelungs-Frames (234 Frames aus 14 Szenen, Frame-Ebene), Erfolg@30 m je Frame, Markierungsrate auf FALSCHEN gelieferten Token (kein GT-Knoten, ungewertet; ein Planer bekommt sie trotzdem).

| Größe | z s0 / s1 | m s0 / s1 | m - z s0 | m - z s1 |
|---|---|---|---|---|
| route_F1_lane | {{z_lane_s0}} / {{z_lane_s1}} | {{m_lane_error_s0}} / {{m_lane_error_s1}} | {{d_lane_error_s0}} [{{lo_lane_error_s0}}, {{hi_lane_error_s0}}] | {{d_lane_error_s1}} [{{lo_lane_error_s1}}, {{hi_lane_error_s1}}] |
| F1 Fahrbahn auf Gabelungs-Frames | {{z_fork_s0}} / {{z_fork_s1}} | {{m_fork_error_s0}} / {{m_fork_error_s1}} | {{d_fork_error_s0}} [{{lo_fork_error_s0}}, {{hi_fork_error_s0}}] | {{d_fork_error_s1}} [{{lo_fork_error_s1}}, {{hi_fork_error_s1}}] |
| Erfolg@30 m | {{z_succ_s0}} / {{z_succ_s1}} | {{m_succ_s0}} / {{m_succ_s1}} | {{d_succ_s0}} | {{d_succ_s1}} |
| Markierungsrate auf falschen gelieferten Token (Fahrbahn) | {{z_false_s0}} / {{z_false_s1}} | {{m_false_s0}} / {{m_false_s1}} | | |

Die Spur-Ebene **gewinnt** deutlich ({{d_lane_error_s0}} / {{d_lane_error_s1}}); die Karte sagte etwa +0.01 voraus, weil die SD-Route die Spurwahl verliert.

Table: Kartenmodi, route_F1_road von m, m - z (gepaart) und die Referenzregel ohne Kopf auf demselben Modus (Seed 0; bei *none* der gerade Prior). z liest die Karte nicht: ein Dump von z dient jedem Modus.

| Modus | m s0 / s1 | m - z s0 | m - z s1 | Regel (Referenz) |
|---|---|---|---|---|
| exact (ohne Fehler) | {{m_road_exact_s0}} / {{m_road_exact_s1}} | {{d_road_exact_s0}} [{{lo_road_exact_s0}}, {{hi_road_exact_s0}}] | {{d_road_exact_s1}} [{{lo_road_exact_s1}}, {{hi_road_exact_s1}}] | 0.9934 |
| error (Urteil) | {{m_road_error_s0}} / {{m_road_error_s1}} | {{d_road_error_s0}} [{{lo_road_error_s0}}, {{hi_road_error_s0}}] | {{d_road_error_s1}} [{{lo_road_error_s1}}, {{hi_road_error_s1}}] | 0.9516 |
| gross (error + 5 m) | {{m_road_gross_s0}} / {{m_road_gross_s1}} | {{d_road_gross_s0}} [{{lo_road_gross_s0}}, {{hi_road_gross_s0}}] | {{d_road_gross_s1}} [{{lo_road_gross_s1}}, {{hi_road_gross_s1}}] | 0.6637 |
| none (keine Karte) | {{m_road_none_s0}} / {{m_road_none_s1}} | {{d_road_none_s0}} [{{lo_road_none_s0}}, {{hi_road_none_s0}}] | {{d_road_none_s1}} [{{lo_road_none_s1}}, {{hi_road_none_s1}}] | 0.7272 (gerader Prior) |
| wrong (Route einer anderen Szene) | {{m_road_wrong_s0}} / {{m_road_wrong_s1}} | {{d_road_wrong_s0}} [{{lo_road_wrong_s0}}, {{hi_road_wrong_s0}}] | {{d_road_wrong_s1}} [{{lo_road_wrong_s1}}, {{hi_road_wrong_s1}}] | 0.6137 |

Der Kopf liest die Karte (Abbruchkriterium nicht ausgelöst): gegen sich selbst verliert m mit der falschen Karte {{self_wrong_s0}} [{{self_wrong_lo_s0}}, {{self_wrong_hi_s0}}] / {{self_wrong_s1}} [{{self_wrong_lo_s1}}, {{self_wrong_hi_s1}}] an F1 gegenüber der richtigen. Mit **keiner** Karte (*none*) ist m praktisch so gut wie z ({{d_road_none_s0}} / {{d_road_none_s1}}). Mit der **falschen** Karte ist m **schlechter** als der Kopf ohne Karte ({{d_road_wrong_s0}} / {{d_road_wrong_s1}}), aber deutlich besser als die Regel auf einer falschen Route (0.6137): der Kopf folgt der falschen Karte nicht blind, er verliert aber mehr, als die Regel erlaubt (Marge -0.03).

Table: Die Klauseln der vor dem Lauf geschriebenen Regel (Quelle: Ergebnisabschnitt der Karte, von Skript geschrieben) und ihr Stand.

| Klausel | verlangt | gemessen (Seed 0 / Seed 1) | erfüllt |
|---|---|---|---|
| (1) Registrierung | m - z >= +0.10 im Modus error, beide Seeds, untere Grenze über 0 | {{d_road_error_s0}} / {{d_road_error_s1}} | **nein** (unter der Hürde um 0.006 / 0.003) |
| (2) Vertrauen, *none* | m - z >= -max(0.01, Streuung) | {{d_road_none_s0}} / {{d_road_none_s1}} | ja |
| (2) Vertrauen, *wrong*, alle Frames | m - z >= -max(0.03, Streuung) | {{d_road_wrong_s0}} / {{d_road_wrong_s1}} | **nein** |
| (2) Vertrauen, *wrong*, Gabelungs-Frames | m - z >= -max(0.05, Streuung) | {{d_fork_wrong_s0}} / {{d_fork_wrong_s1}} | ja |
| (3) Gabelungs-Frames (Frame-Ebene) | m - z >= +0.10, untere Grenze über 0 | {{d_fork_error_s0}} / {{d_fork_error_s1}} | ja (234 Frames, 14 Szenen) |
| (4) Spur | m - z >= -0.02 | {{d_lane_error_s0}} / {{d_lane_error_s1}} | ja |
| (5) Abbruch | kein nicht endlicher Verlust, Verlustquotient unter 0.8, liest die Karte | Verlust Schritt 100 -> 3000: z 0.44 / 0.45, z2 0.46, m 0.23 / 0.24, m2 0.25; Rollbacks 0, übersprungene Schritte 0 in allen sechs Läufen | nicht ausgelöst |
| (6) Kosten | höchstens 1.0 ms zusätzlich, nur im ruhigen Fenster entschieden | Kopf allein, Batch 1, K 512, M 66 / 136: fp32 {{ms_full_M66_fp32}} / {{ms_full_M136_fp32}} ms, bf16 {{ms_full_M66_bf16}} / {{ms_full_M136_bf16}} ms (Median aus 300) | **nicht ruhig** (die GPU war zu 18 % belastet, Desktop): nur indikativ, etwa das Doppelte der Hürde |

Die Regel liefert **TIE (NO GAIN)**: Klausel 1 scheitert. Als zweites scheitert Klausel 2 (falsche Karte), und Klausel 6 ist ohne ruhiges Fenster nicht entschieden; die Karte selbst merkt an, dass eine Hürde von +0.08 Klausel 1 erfüllt hätte und das Urteil dann TIE (UNSAFE) gewesen wäre; das ist eine nachträgliche Überlegung (nach TIE (UNSAFE) gehört auch Klausel 6, die hier unentschieden ist). Die Hürde wurde vor dem Lauf geschrieben; sie wird nicht nachträglich herabgesetzt. Standard bleibt z: **keine Kartenstufe im Champion.**

**Kalibrierung.** Anteil der Knoten, die auf der Route liegen, je Bin von p_route (m, Modus error, Seed 0, abgedeckte Knoten). Die Ränder sind scharf: bei p unter 0.1 liegen {{cal0_s}} der Knoten auf der Route (n = {{cal0_n}}), bei p ab 0.9 sind es {{cal9_s}} (n = {{cal9_n}}).

Table: Kalibrierung der Fahrbahn-Wahrscheinlichkeit von m (Seed 0, Modus error).

| Bin von p_route | Knoten | Anteil auf der Route |
|---|---|---|
| 0.0 - 0.1 | {{cal0_n}} | {{cal0_s}} |
| 0.1 - 0.2 | {{cal1_n}} | {{cal1_s}} |
| 0.2 - 0.3 | {{cal2_n}} | {{cal2_s}} |
| 0.3 - 0.4 | {{cal3_n}} | {{cal3_s}} |
| 0.4 - 0.5 | {{cal4_n}} | {{cal4_s}} |
| 0.5 - 0.6 | {{cal5_n}} | {{cal5_s}} |
| 0.6 - 0.7 | {{cal6_n}} | {{cal6_s}} |
| 0.7 - 0.8 | {{cal7_n}} | {{cal7_s}} |
| 0.8 - 0.9 | {{cal8_n}} | {{cal8_s}} |
| 0.9 - 1.0 | {{cal9_n}} | {{cal9_s}} |

**Kontrolle ohne Knoten-Token (explorativ, kein Urteil, nur Seed 0).** Die Arme zg und mg lesen nur Geometrie (ohne `tok`, 21,130 Parameter): zg {{zg_road}}, mg {{mg_road}}, mg - zg {{gdiff_road}} [{{glo_road}}, {{ghi_road}}] (Spur {{gdiff_lane}} [{{glo_lane}}, {{ghi_lane}}], Gabelungen {{gdiff_fork}} [{{glo_fork}}, {{ghi_fork}}]); Vertrauen: mg ohne Karte gegen zg {{gnone_d}}, mit der falschen Karte {{gwrong_d}}. Ohne `tok` ist der Kopf ohne Karte etwas schwächer ({{zg_road}} gegen {{z_road_s0}}): der größte Teil von z kommt aus Position, Richtung und Flags, `tok` bringt etwa 0.02. Der Kartengewinn ist ohne `tok` nominell größer ({{gdiff_road}} gegen {{d_road_error_s0}}); der Unterschied von etwa 0.013 liegt innerhalb dessen, was ein Seed zeigen kann (die Intervalle überlappen stark, die beiden Werte sind kein gepaarter Vergleich).

**Kosten.** Kopf: 33,738 Parameter. Zeit pro Frame, Kopf allein (nicht der Zuwachs im Champion-Pfad), mit Zufallsgewichten, Batch 1: {{ms_full_M66_fp32}} ms (fp32) / {{ms_full_M66_bf16}} ms (bf16) bei 66 Stützstellen, **nicht im ruhigen Fenster** (GPU belastet): nur indikativ, Klausel 6 ist **unentschieden**. Die Messung steht in `runs/eval/e389_head_timing.json`; ein ruhiger Lauf wurde versucht und nicht erreicht (die GPU las in 87 Proben über 15 Minuten nie ruhig, 14 - 34 % Auslastung durch Desktop-Prozesse), die Klausel bleibt „ausstehend“. Hypothese (nicht profiliert): der Kopf ist durch Kernel-Starts begrenzt, nicht durch Arithmetik (66 und 136 Stützstellen brauchen gleich lang, bf16 ist langsamer); ein ruhiger Zeitlauf und eine fusionierte Formulierung (die Routen-Token sind für alle Token eines Frames gleich) stehen aus.

**Was das Ergebnis für den Champion bedeutet.** Der Champion bleibt rein wahrnehmend: **keine Kartenstufe** (TIE (NO GAIN)). Die Frage, ob ein Planer die Zuordnung braucht, bleibt offen. Der stärkste Befund ist, dass die Token des Champions die Route schon zu großen Teilen enthalten (z erreicht {{z_road_s0}}); die Karte ist mit einer falschen Route schlechter als gar keine, und die Kosten des Kopfes sind unentschieden (indikativ etwa 2.1 ms allein, nicht ruhig gemessen).
