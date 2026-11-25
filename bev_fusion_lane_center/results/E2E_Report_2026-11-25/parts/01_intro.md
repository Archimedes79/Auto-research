# 1. Einleitung: was seit dem 2026-11-23 gelaufen ist

**Kurz.** Auf deine Worte (D142, D144, D145, D148) liefen die offenen Experimente beider Projekte allein weiter: die Catmull-Rom-Linie (E395, mit Zusatz A1), die verbesserte Karten-Zuordnung (E391b), die Pyramide im Token-Loss (E393), die Fehlermodelle der Karte (E394) und im BEV-Projekt das Ein-Bild-ein-Scan-Design (t25). **Nichts wurde angenommen: der Champion (D120) bleibt unverändert.** Die Zahlen stammen von VAL (873 Frames, 22 Szenen, Spurmitte) bzw. VAL mit 1000 Frames (BEV), soweit nicht anders gesagt: E391b Teil A liest den TRAIN-Holdout und E394 nur TRAIN. TEST hat kein Experiment gelesen.

Table: Überblick: Ausgang je Experiment mit einer Zahl.

| Experiment | Frage in einem Satz | Ausgang | eine Zahl |
|---|---|---|---|
| E395, Schritt 1 | Catmull-Rom statt Hermite zwischen den gelieferten Knoten? | LOSS, die Kurzlink-Variante TIE | topo F1 mit Catmull-Rom -0.0302 / -0.0372 |
| E395, Schritt 2 und A1 | verschiebt ein kleiner Positionskopf die Knoten näher an die Wahrheit? | Schritt 2 INCONCLUSIVE; Zusatz: p0n REPRODUZIERT unter der Latte | F1@0.5 +0.018 / +0.023, Latte 0.033 |
| E391b, Teil A | lässt sich die Entscheidung "falscher Token / fehlendes Segment" aus Token und Karte treffen? | CLOSED: an der Latte nicht als trennbar gezeigt | bestes Tor 0.076 / 0.090 unter der Latte (TRAIN-Holdout) |
| E391b, Teil B | war E391s NO FUSION ein Artefakt der Trainings-Population? | NO FUSION: nicht umgekehrt | assoc_acc 0.8220 / 0.8239, Latte 0.8427 / 0.8397 |
| E393 | lernt die Pyramide von B' mit dem Token-Loss mit? | kein messbarer Gewinn, kein Kandidat | e - n topo F1 +0.0012 / -0.0004 |
| E394 | label-freies Tor, das einer falschen Karte misstraut? | auf TRAIN geschlossen (nicht erreichbar) | kein Schwellwert erfüllt Vertrauen und Schaden zugleich |
| t25 (BEV) | ein Bild und ein Radar-Scan, Vergangenheit rekursiv? | R: nicht bestanden (eine Schutzregel); A1 und A2: FAILS und CUT; Speicher nach der Regel TIE | `detf` mAP R / A2 gegen ctl -0.0013 / -0.0343 (kalt) |

**Wie der Bericht gebaut ist.** Kapitel 2 ist der kurze Leitfaden, wie die Ground Truth aussehen muss (unverändert aus dem Bericht vom 2026-11-23). Kapitel 3 bis 7 sind die Experimente, Kapitel 8 der Stand des Systems, Kapitel 9 Schnitte und offene Punkte, Kapitel 10 die Reproduzierbarkeit. Die Regeln sind die der Karten: eine vor dem Lauf geschriebene Regel entscheidet; wo die Zahlen eine Handlesart brauchten, steht "von Hand".
