# 1. Einleitung und Stand

**Kurz.** Dieser Bericht fasst zusammen, was seit dem Bericht vom 2026-11-20 (E389, Kartenzuordnung der Spurgraph-Token) geschah, und gibt den Stand des Systems am 2026-11-23. Er ist der Abschlussbericht vor der Pause der Experimente (Entscheidung D138). Die Zahlen mit Platzhalter stammen aus den Ergebnisdateien der Läufe (Score-Karten, Probe-Karten) und werden beim Bau eingesetzt; die Hürden, Nulls, Kosten- und Champion-Zahlen stammen aus den Karten und aus `HANDOVER.md` und werden von `check_numbers.py` gegen sie geprüft (die Kritiker haben sie von Hand gelesen). **Kein Experiment dieses Berichts hat TEST berührt.** Die Karte ist überall eine **Simulation** aus der Ground Truth, keine echte Navigationskarte.

**Der Champion (D120) ist unverändert:** zweistufiges Token-Rezept, der Kopf von E373 (None-Gewicht 0.25, Dekodierung von E375: Kurzlinks 0.3, lange Verbinder 0.2, Knoten bei p_none < 0.6) und in Stufe 2 der Kopf zusammen mit dem Knotenkopf von B' (S4, `runs/e372_s4_s0 / s1`). Er liefert aus Kamera und Radar einen Spurmitten-Graphen mit topo F1 0.6682 / 0.6813 (Seed 0 / 1), junction F1 0.2538 / 0.2481, in 16.2 - 16.4 ms auf der ruhigen GPU, mit 2.323 M Parametern (Quelle: `HANDOVER.md`, Karte E372). **Neu seit dem 20.11.** ist ein Kandidat für die Stufe 1 (r2 aus E392, Kapitel 7); ob er angenommen ist, hängt an der Kostenmessung von D138 (Kapitel 7.5; beim Bau dieses Berichts lagen die Rohzahlen vor, das Urteil des Lane-Chats noch nicht).

Table: Überblick: was seit dem 2026-11-20 gelaufen ist, mit einer Zahl je Experiment (VAL, 873 Frames, Seed 0 / 1, sofern nicht anders gesagt).

| Experiment | Frage | Ausgang | eine Zahl |
|---|---|---|---|
| E388 | die drei Dekodier-Schwellen des Champions per Gradient lernen | **NO** | topo F1 gelernt - Hand -0.025 / -0.028 (Kapitel 3) |
| E381 | Rauschen auf den Trainings-Token, passend zur TRAIN-zu-VAL-Lücke | **TIE** | topo F1 +0.0004 / +0.0041 auf der Verdict-Zelle, Intervalle enthalten 0 (Kapitel 3) |
| E389 / E389b | ordnet ein kleiner Kopf die Token der Route einer SD-Karte zu; eine Attention statt zwei | **TIE (NO GAIN)**; eine Attention genügt | Fahrbahn-F1 gegen den Kopf ohne Karte +0.094 / +0.097 gegen die Hürde +0.10 (Kapitel 4) |
| E390 | Schritt 1: die SD-Karte als logischer Straßengraph (Spuren pro Richtung, Verbindungen) | **SOUND** (Version 7, drei Blicke auf dieselben VAL-Frames) | Spuren pro Richtung bei 30 % Zählfehler {{e390_p1b_s0}} / {{e390_p1b_s1}} (Kapitel 5) |
| E391 | Schritt 2: die gelieferten Token den Karten-Segmenten zuordnen, logische Daten darüber lesen | **NO FUSION** | Zuordnung {{e391_val_assoc_s0}} / {{e391_val_assoc_s1}} unter der Hürde 0.8427 / 0.8397; Spuren pro Richtung {{e391_val_lanes_s0}} / {{e391_val_lanes_s1}} gegen {{e391_val_zlanes_s0}} / {{e391_val_zlanes_s1}} ohne Karte (Kapitel 6) |
| E392 | frühes Verdrahtungs-Loss und Nachrichtenaustausch im Token-Kopf (nur Wahrnehmung) | r2 **REPLICATES** | topo F1 gegen die Zwillingskontrollen im Mittel +0.0067 (Kapitel 7) |
| E393 | die Pyramide von B' durch das Token-Loss mittrainieren | Entwurf, **nicht gelaufen** | Kapitel 8 |
| E391b, E394 | die Zuordnung verbessern; Schritt 3: Fehlermodelle der Karte und Vertrauen | **pausiert (D138)** | Kapitel 8 und 9 |

**Wie der Bericht gebaut ist.** Kapitel 2 ist der kurze Leitfaden, wie die Ground Truth aussehen muss (jetzt mit dem, was die Kartenlinie braucht). Kapitel 3 bis 8 gehen die Experimente in der Reihenfolge der Tabelle durch, mit ehrlichen Negativen; E388 / E381 / E389 sind bewusst kurz, ihre Berichte liegen vor. Kapitel 9 sagt, was offen ist und was der Eigentümer als Nächstes entscheiden kann. Kapitel 10 nennt, was in git liegt, Anhang A, wie man die Zahlen erneut erzeugt. Die Bilder zeigen echte VAL-Frames, die nach einer vor dem Anschauen geschriebenen Regel gewählt sind.
