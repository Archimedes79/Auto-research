# 8. Die zwei Experimente der Nacht, kurz

**Kurz.** In derselben Nacht wurden zwei Fragen am Champion (S4, D120) geschlossen, beide mit negativem Ausgang: **E388** (gelernte Ausgangsschwellen: NO) und **E381** (Rauschen auf den Trainings-Token: TIE). Der Champion blieb unverändert. Beide Karten stehen in `experiments/`; die Zahlen unten sind aus ihren Ergebnisabschnitten.

## 8.1 E388: die drei Dekodier-Schwellen des Champions per Gradient lernen: NO

**Frage (D125).** Der Champion dekodiert mit drei von Hand auf TRAIN gewählten Zahlen (Kurzlinks 0.3, lange Verbinder 0.2, Knoten behalten bei p_none < 0.6; das None-Tor bleibt aus). Kann man sie stattdessen per Gradient lernen, auf einem weichen F1-Ersatz der gelieferten Entscheidung, gerechnet auf einer TRAIN-Stichprobe (jeder dritte Frame, 291), und einmal auf VAL mit dem echten Scorer gelesen?

**Ergebnis (nach der vor dem Lauf geschriebenen Regel der Karte v2.1): NO.** Beide Seeds scheitern.

Table: E388, gelernte minus handgewählte Schwellen, VAL, echter Scorer, gepaart auf demselben Dump; Seed 0 / Seed 1.

| Größe | Seed 0 | Seed 1 |
|---|---|---|
| topo F1 | -0.025 | -0.028 |
| junction F1 | -0.119 | -0.109 |
| lanes found | -0.011 | -0.012 |
| gelernte Schwellen kurz / lang / behalten (Hand: 0.3 / 0.2 / 0.6) | 0.05 / 0.19 / 0.388 | 0.05 / 0.187 / 0.376 |

Die Vorhersage der Karte (-0.002, Fenster -0.008 bis +0.003) war ein MISS; die vorhergesagten Schwellen (0.35 / 0.3 / 0.55) trafen ebenfalls nicht. Handlung: die handgewählte Dekodierung 0.3 / 0.2 / 0.6 bleibt, kein Champion-Wechsel.

**Warum (ehrlicher Befund).** Die Anpassung tat, wofür sie gebaut war: auf TRAIN schlägt ihre Zelle die Handzelle auf dem eigenen Ersatzziel. Das Ziel ist das Problem: die Kurzlink-Schwelle läuft an die Untergrenze 0.05, weil ein tokenweises F1 über die Paare jeden zusätzlichen richtigen Kurzlink belohnt, während 34 - 36 % der GT-Links gar keinen Kandidaten haben und Richtung Recall ziehen. Im Graphen werden daraus falsche Verdrahtungen an Kreuzungen (die Präzision der Kreuzungen halbiert sich, der Recall steigt).

### E388b: die Diagnose (D126, von Hand, kein Urteil; das NO bleibt)

Zwei Erklärungen standen: (H1) die TRAIN-Token von B' sind sauberer als die auf VAL, die angepassten Schwellen passen deshalb nicht; (H2) das weiche F1 ist das falsche Ziel. Zwei Lesungen trennen sie.

Table: E388b Lesung (i): dieselbe gelernte Zelle gegen die Handzelle, echter Scorer, auf der TRAIN-Stichprobe selbst (291 Frames), neben VAL.

| Seed | TRAIN topo F1 gelernt - Hand [95 %] | VAL topo F1 (E388) | TRAIN junction F1 | VAL junction F1 |
|---|---|---|---|---|
| 0 | -0.0256 [-0.0323, -0.0187] | -0.0252 | -0.1532 | -0.1189 |
| 1 | -0.0272 [-0.0337, -0.0209] | -0.0280 | -0.1579 | -0.1086 |

Table: E388b Lesung (ii): Kreuz-Anpassung auf VAL (die 22 Szenen nach der Parität des Rangs in zwei Hälften; auf einer Hälfte anpassen, auf der anderen lesen), gelernt minus Hand auf der gehaltenen Hälfte.

| Seed | angepasst auf | gelesen auf | topo F1 [95 %] | junction F1 | lanes found |
|---|---|---|---|---|---|
| 0 | A | B | -0.0116 [-0.0236, +0.0002] | -0.1037 | -0.0165 |
| 0 | B | A | -0.0093 [-0.0147, -0.0033] | -0.0585 | -0.0178 |
| 1 | A | B | -0.0066 [-0.0107, -0.0019] | -0.0634 | -0.0223 |
| 1 | B | A | -0.0032 [-0.0064, -0.0002] | -0.0361 | -0.0316 |

**Lesung (von Hand).** Die gelernte Zelle verliert auf TRAIN, auf den Frames, auf denen sie angepasst wurde, genauso stark wie auf VAL: eine TRAIN-zu-VAL-Verschiebung kann das nicht erklären. Auf einer Hälfte angepasst und auf der anderen gelesen sind die Schwellen weniger extrem, und die Zelle verliert trotzdem auf allen vier gehaltenen Hälften (alle Punktschätzungen negativ, drei von vier Intervallen unter 0). **H2 gilt: das Ziel, nicht die Verschiebung**; die Verschiebung mag ein wenig dazukommen (die auf VAL angepassten Zellen verlieren weniger als die auf TRAIN angepasste). Lehre für gelernte Dekodierung: das Ziel muss die Graph-Metrik sein (oder ein weicher Ersatz für die Kreuzungs-Präzision des gelieferten Graphen), nicht ein paarweises Token-F1. Kosten (Ledger): E388 selbst 174 s GPU für die Label-Dumps plus etwa 13 Minuten CPU; E388b etwa 439 s GPU und 1183 s CPU.

## 8.2 E381: Rauschen auf den Trainings-Token, passend zur TRAIN-zu-VAL-Lücke von B': TIE

**Frage (D126).** B' ist auf TRAIN besser als auf VAL, der Kopf sieht beim Training zu saubere Token. Hilft es, den Stufe-1-Kopf des Champions auf Token zu trainieren, die dem ähneln, was er sehen wird: Positionsjitter und Rauschen des Existenz-Logits, kalibriert an der gemessenen Lücke, zur Trainingszeit, nichts zur Laufzeit?

**Ergebnis (nach der Regel der Karte v2.1): TIE.** Auf der Verdict-Zelle 0.3 / 0.2 / 0.5 / aus: Seed 0 topo F1 +0.0004 [-0.0058, +0.0080] mit lanes found -0.0116 (die Schranke -0.010 knapp gebrochen); Seed 1 +0.0041 [-0.0020, +0.0097], unter der Hürde von +0.005, das Intervall enthält 0. Weder ADOPT (kein Seed besteht) noch NO (kein Seed bei topo F1 <= -0.005). Handlung: der Stufe-1-Kopf bleibt ohne Rauschen (D106: das einfachere Rezept); keine S4-Wiederholung; der Champion bleibt.

Daneben berichtet (nicht entscheidend): bei D115 lesen die verrauschten Köpfe +0.0028 [-0.0033, +0.0099] und +0.0056 [+0.00004, +0.0109] topo F1 bei unveränderten F1@1.0 und lanes found: eine kleine positive Tendenz auf beiden Seeds, auf Seed 0 innerhalb der Streuung der Kopf-Seeds (etwa 0.003). Die Vorhersage der Karte (+0.003, Fenster -0.005 bis +0.010) war ein HIT; die Quoten waren TIE 65 %.

**Was es sagt (ehrlicher Befund).** Eingangsrauschen, das an die Lücke zwischen TRAIN- und VAL-Token kalibriert ist, hilft dem Kopf nicht und schadet ihm nicht über das Rauschen hinaus: die Existenz-AP des Kopfes auf VAL ist unverändert (+0.0008 / -0.0001). Der Mechanismus (zu viel Vertrauen in saubere TRAIN-Positionen) zeigt sich am Ausgang des Kopfes nicht. Die Lesung der Karte (von Hand, nicht getestet): der Hebel gegen die Lücke liegt vor dem Kopf (S4 trainiert B' mit dem Kopf; ein B' außerhalb der Falte), nicht in einer Augmentierung auf der Kopfseite. Kosten: 1.4 GPU-Stunden (zwei Köpfe parallel, 43 Minuten) plus Dumps 0.1 h und etwa 1 h CPU.
