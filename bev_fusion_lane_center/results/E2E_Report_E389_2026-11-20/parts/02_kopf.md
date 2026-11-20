# 2. Der Kopf: Architektur und Beschreibung

**Kurz.** `RouteHead` (`training/grid/jroute.py`) ist ein tokenweises Netz mit **einer** Cross-Attention von jedem Token auf die Routen-Token der SD-Karte. Es hat **33,738 Parameter** (21,130 in der Kontrolle ohne den Knoten-Token `tok`, „geo“). Es gibt keine Self-Attention zwischen den Token: die Ausgabe eines Tokens hängt nur von seiner eigenen Zeile und der Route ab. Die Zeichnung unten wird von `draw_route_head.py` aus einer Instanz des Moduls erzeugt; die Parameterzahlen jedes Blocks sind gelesen, nicht getippt, und ein umbenannter oder entfernter Block lässt das Zeichnen fehlschlagen.

**Der Weg eines Tokens.**

1. **Eingabe.** Die gecachten Token des eingefrorenen B' (K = 512 pro Frame): der 64-dimensionale Knoten-Token `tok`, Position, Richtung, Existenz-Logit `e`, acht Flag-Logits. Keine Zielgröße, kein GT-Schlüssel erreicht den Kopf (der Trainer reicht nur die Schlüssel der festen Eingabeliste durch, und der Vorwärtslauf liest nur sie; die Routen-Tabelle wird auf genau vier Schlüssel geprüft).
2. **Eigene Merkmale.** `tok` wird linear projiziert und normiert; ein kleines MLP liest die Geometrie in **Metern** (Position / 50 und Position / 5 begrenzt auf +-3, Richtung, sigmoid(e), Flags). Das gibt dem Kopf ohne Karte die Möglichkeit, den Geradeaus-Korridor zu lernen: der No-Map-Kopf ist also nicht schwach, sondern der ehrliche Vergleich.
3. **Routen-Token.** Die SD-Route wird alle 2 m neu abgetastet (höchstens 160 Stützstellen innerhalb +-50 m; die Kappung 64 der ersten Fassung schnitt 36 % der VAL-Frames ab, Kapitel 4): Position, Fahrtrichtung, Spurzahl (0 = unbekannt bis 5), gültig. Genau diese vier Schlüssel; Segment-IDs, Straßen-IDs, die Routenstückliste, ein Spurindex oder ein Knotenlabel wären ein Leck und werden verweigert. Die Kartenfehler des Protokolls (Versatz, Gier, Ausdünnung, Spurzahl-Fehler) werden auf die Route angewendet; die Labels ändern sich nie mit dem Kartenfehler.
4. **Paarmerkmale in Metern.** Für jedes Paar (Token, Routenstütze): Drehung (cos / sin), Versatz längs und quer zur Token-Richtung und zur Routenrichtung, Abstand, Spurzahl, alle in Metern / 5 und auf +-3 begrenzt. Der Quer-Versatz zur Routenlinie ist genau die Größe, die die Registrierung braucht („liegt der Token innerhalb von n x 1.75 m + 0.5 m der Route?“). Die alte Cross-Attention des S2-Blocks skalierte Versätze mit 1/20 und diente nur als Bias; für „innerhalb von 2 m“ ist das zu grob, deshalb die Meter-Merkmale.
5. **Attention A (gelernt, 4 Köpfe).** Softmax über {gültige Routenstützen} + {ein gelernter **None-Schlüssel** je Kopf}; der Logit ist q.k / 4 + Bias(Paarmerkmale). Ohne Karte (kein gültiger Schlüssel) schaut jeder Token nur auf den None-Schlüssel: so kann der Kopf eine fehlende Karte ignorieren.
6. **Attention B (scharf, fest, ohne Parameter).** Weiches Argmin über den Abstand (Temperatur 0.3 m): die nächste Stütze trägt fast das ganze Gewicht, ohne harte Auswahl. Sie liefert die Meter-Merkmale der nächsten Stütze, vor allem den Quer-Versatz.
7. **Ausgabe-MLP.** [x, gewichtete Werte, Attention-Masse auf echten Schlüsseln, Paarmerkmale gepoolt nach A und nach B, Karte vorhanden] (181 Werte) -> 64 -> zwei Logits: *road* (der Token liegt auf der Fahrbahn der Route) und *lane* (er liegt auf der Spur, die das Fahrzeug fährt).

Table: Parameter je Block, aus dem Modul gelesen (`tools/draw_route_head.py --table`); die Zeichnung zeigt die drei Wege eines Tokens: (1) eigene Merkmale, (2) gelernte Attention A, (3) scharfe Attention B, danach verketten und eine Ausgabe-MLP.

| Block | Parameter |
|---|---|
| Token-Projektion (`tok` -> 64, LayerNorm) | 4,288 |
| Geometrie-MLP (eigene Merkmale) | 1,568 |
| Anfrage q (LayerNorm + Linear) | 6,400 |
| Routen-Schlüssel (Eingabe -> 64, LayerNorm) | 832 |
| Schlüssel und Werte (64 -> 128) | 8,320 |
| Bias-MLP der Paarmerkmale (8 -> 32 -> 4) | 420 |
| None-Schlüssel (k, v, Bias; 4 Köpfe) | 132 |
| Ausgabe-MLP (181 -> 64 -> 2) | 11,778 |
| **Summe** (mit `tok`) | **33,738** |
| Summe nur Geometrie (ohne `tok`, Kontrolle zg / mg) | 21,130 |

**Verlust.** BCE beider Logits auf den Token mit Label, positive Gewichte nach der TRAIN-Häufigkeit; 3000 Schritte, Batch 8, lr 1e-3, AdamW, Cosinus. Das Training liest die gecachten Token, die Stufe-1-Köpfe des Champions sind **nicht** in der Schleife (ihr Ergebnis, die gelieferte Token-Menge, wird erst bei der Auswertung gelesen).

**Differenzierbarkeit.** Der Kopf ist durchgehend differenzierbar (weiche Aufmerksamkeit, kein harter Schritt innerhalb des Kopfes). Genannte feste Schritte außerhalb: die Auswahl der **gelieferten Token** (der Champion-Kopf E373 bei der Dekodierung D115) und die Schwelle p_route >= 0.5 bei der Ausgabe. Es gibt keinen Gradienten von der Route in B' oder in den Champion: eine gemeinsame Feinabstimmung wäre eine eigene Karte.

**Die Arme.**

| Arm | was | Rolle |
|---|---|---|
| z | derselbe Kopf, die Route jedes Samples ungültig (nur der None-Schlüssel) | der **passende No-Map-Kopf**: gleiche Daten, Labels, Schritte, Seeds; kann Token-Zustände und Geometrie (Geradeaus-Prior) nutzen |
| z2 | Seed-Zwilling von z (Cache-Seed 0, andere Initialisierung `--init-seed 1`) | misst vor dem Urteil das Rauschen zwischen zwei Köpfen ohne Karte |
| m | der Kopf mit den Routen-Token, trainiert auf der Mischung der Kartenmodi (exakt 25 %, fehlerhaft 47 %, Spurzahl unbekannt 8 %, fehlend 10 %, grob 10 %: die Hälfte davon eine starre Verschiebung um 4 - 8 m, die andere Hälfte die fehlerhafte Route einer ANDEREN Szene, so dass etwa 5 % aller Züge falsche Karten sind) | die Variante |
| m2 | Seed-Zwilling von m (Cache-Seed 0, `--init-seed 1`: andere Initialisierung, Datenreihenfolge, Kartenfehler-Züge) | misst die Streuung eines trainierten Kopfes gegen einen anderen; mit z2 legt sie das Rauschen vor dem Urteil fest |
| zg, mg | die Kontrolle nur mit Geometrie (ohne `tok`): 21,130 Parameter | explorativ, kein Urteil |

**Kosten.** Parameter: 33,738 (full) und 21,130 (geo), im Selbsttest festgeschrieben. Zeit pro Frame, Kopf allein: {{ms_full_M66_fp32}} ms (fp32) bei 66 Routen-Stützen, **nicht im ruhigen Fenster** gemessen, also nur indikativ (Kapitel 5).
