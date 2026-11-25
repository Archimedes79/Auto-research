# 8. Stand des Systems

**Der Champion (D120) ist unverändert:** zweistufiges Token-Rezept (Kopf von E373, Knoten-Kopf von B' in Stufe 2 mittrainiert), VAL D115 live topo F1 0.6682 / 0.6813, Kreuzungs-F1 0.2538 / 0.2481 (Seed 0 / 1), ganzer Pfad 16.2 - 16.4 ms, 2.323 M Parameter. Sein Code liegt in `winner/`; der BEV-Sieger ist `bev_fusion/winner/t21_ctl` (ein Gleichstand nach Wert). Sieger sind nur gelernte Systeme; TEST wurde nicht gelesen.

**E392 (aus dem Bericht vom 2026-11-23, unverändert):** das frühe Verdrahtungs-Loss mit Nachrichtenaustausch (r2) gewinnt etwa +0.010 topo F1 gegen die Kontrolle c und repliziert mit anderen Kopf-Seeds bei etwa +0.007, verfehlt aber die Kostenhürde (+3.93 ms gegen +2.0 ms nach der Kartenregel; je nach Lesart etwa 1.8 bis 3.9 ms auf einem belasteten Rechner) und wird nicht angenommen.

**Die Karten-Linie:** Schritt 1 (die SD-Karte als Straßengraph, E390) ist SOUND, Schritt 2 (Zuordnung, E391) ist NO FUSION, auch nach der Wiederholung auf der gelieferten Population (E391b, Kapitel 4), bei Schritt 3 (Fehlermodelle und Vertrauen, E394) ist das erste Tor g auf TRAIN geschlossen, die weiteren Arme (v2) sind nicht gestartet. Die Karte ist überall simuliert; ohne Vertrauen ist ihr Gewinn nicht belastbar.
