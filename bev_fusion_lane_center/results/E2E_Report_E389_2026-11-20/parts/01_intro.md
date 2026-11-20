# 1. Einleitung und Stand

**Kurz.** Der Token-Champion (D120, zwei Stufen, topo F1 0.6682 / 0.6813) liest nur Kamera und Radar. Er weiß nicht, welche seiner wahrgenommenen Lane-Knoten auf der **Navigationsroute einer SD-Karte** liegen (Problemgruppe A6 in `PROBLEM.md`: Zuordnung des wahrgenommenen Graphen zur SD-Route; A7: was ein Planer braucht). E389 fragt, ob **ein kleiner, differenzierbarer Kopf** diese Zuordnung lernen kann: eine Cross-Attention von den Token des Champions auf die Routen-Token der Karte, ein MLP auf den eigenen Merkmalen, zwei Ausgaben pro Token (liegt auf der Fahrbahn der Route, liegt auf der Spur, die das Fahrzeug fährt). Als Maßstab dient derselbe Kopf **ohne** Karte (der passende No-Map-Kopf).

**Ergebnis in einem Satz.** **TIE (NO GAIN).** Die Karte verbessert die Zuordnung der Fahrbahn gegenüber dem Kopf ohne Karte um {{d_road_error_s0}} / {{d_road_error_s1}} F1 (Untergrenzen {{lo_road_error_s0}} / {{lo_road_error_s1}}), die vor dem Lauf geschriebene Hürde war +0.10; mit einer falschen Karte ist der Kopf schlechter als ohne ({{d_road_wrong_s0}} / {{d_road_wrong_s1}}); der Kopf ohne Karte ist mit {{z_road_s0}} / {{z_road_s1}} viel stärker als angenommen. Der Champion bekommt keine Kartenstufe.

**Warum diese Frage.** Der Champion liefert einen Spurgraph, aber keine Zuordnung zur Karte. Der frühere Versuch mit einem Kartenblock im Token-Kopf (S3, E374) war ein TIE für die Topologie: die Karte half dem Graphen nicht. E389 stellt die Frage anders: nicht „wird der Graph besser?“, sondern „kann der Kopf sagen, welche Token auf der Route liegen, unter den Kartenfehlern des Protokolls (Versatz 1 - 2 m, Gierfehler, ausgedünnte Polylinie, Spurzahl-Fehler)?“. Die Aufgabe ist im Kern **Registrierung** unter dem Kartenfehler, plus die Entscheidung an **Gabelungen**, wo nur die Karte weiß, welcher Ast die Route ist. Mirko hat sie bestellt (D127: „ok, dann hole das bitte nach“, nachdem er erfahren hatte, dass der Champion keine Zuordnung zur Karte hat), und er wollte sie vor dem Herunterfahren fertig und gelesen haben (D128).

**Was dieser Bericht enthält.**

| Kapitel | Inhalt |
|---|---|
| 2 | Der Kopf: Zeichnung aus dem echten Modul, Parameter, Eingaben, Verlust |
| 3 | Ground Truth: ein kurzer Leitfaden, wie die GT eines Lane-Graphen aussehen muss, und die Routen-GT von E389 |
| 4 | Protokoll: Arme, Kartenmodi, die vor dem Lauf geschriebene Regel, Vorhersage, Referenzzeilen |
| 5 | Ergebnisse von E389 gegen die passenden Baselines |
| 6 | Qualitative Beispiele, nach einer vorher geschriebenen Regel gewählt |
| 7 | Ehrliche Negative und Grenzen |
| 8 | Die zwei Experimente der Nacht, kurz: E388 (NO, gelernte Ausgangsschwellen) und E381 (TIE, Rauschen auf den Trainings-Token), mit der Diagnose E388b |
| 9 | Was in git liegt und was nicht |
| 10 | Offene Punkte |
| A | Wie man es wiederholt; B Hilfsskripte; C Quellen der Zahlen |

**Stand des Projekts (unverändert durch diesen Bericht).** Champion bleibt der Zwei-Stufen-Kopf (E373 + S4, D120): 2.323 M Parameter, 16.21 ms im ruhigen Fenster. VAL (873 Frames, 22 Szenen) wurde gelesen, TEST ist unberührt. Dieser Bericht ändert den Champion nicht; E389 ist eine **zusätzliche Stufe auf den gecachten Token** (der Kopf liest die Token des eingefrorenen B', er hängt nicht in Stufe 2).
