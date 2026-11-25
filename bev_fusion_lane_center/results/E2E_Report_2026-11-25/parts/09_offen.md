# 9. Was geschnitten wurde und was auf dich wartet

## 9.1 Geschnitten oder nicht fertig

- **Aus Zeitgründen geschnitten: nichts.** Die Läufe der Spurmitte dieser Nacht und alle vier Arme von t25 sind fertig (letzter Lauf: die Auswertung von R bis 03:45).
- **Nicht gelaufen (BEV):** t26, der lange Lauf von Grund auf (geschätzt 70 - 104 GPU-Stunden, seine Frage D102 ist offen); t22 Arm R und Seed-Zwilling (durch D68 zurückgestellt, etwa 8 GPU-Stunden); ältere Zeilen von `records/schedule.csv` (keine Karte mit Regel für diesen Auftrag).
- **Ungültig:** E393 Arm p, Seed 1 (Wiederaufnahme nach CUDA-Fehler, Kapitel 5).
- **Gestartet und mit eigener Regel geschlossen:** E394 auf TRAIN (Kapitel 6), E391b Teil A (CLOSED).
- **Zeitfenster:** der BEV-Lauf t25 A1 lief eine Minute über das Fensterende; die Auswertung von t25 R startete nach der 03:00-Grenze, nach der Auslegung von Lead- und BEV-Chat als Teil des vorher gestarteten Arms (beides Kapitel 7).

## 9.2 Offen für dich

- **E395:** ob der Positionskopf mit Knoten-Loss (p0n, PARTIAL: ein echter Gewinn unter der Latte) dir das wert ist, entscheidest du.
- **BEV t25, Lesart bei R:** nach dem Wortlaut der Karte besteht R nicht; zählt nur die Schnitt-Stufe, bestünde R (Kapitel 7). Du kannst das korrigieren.
- **BEV t26, Tor 7:** der Prüfwert (Hash) des Evaluators hat sich geändert; die Basis braucht eine Neuwertung oder dein ausdrückliches Erlauben.
- **Offene Fragen in den Entscheidungen:** D66 (BEV: Spur-Trenner-Labels in Kreuzungen), D102 (BEV t26, der lange Lauf), D110 (Maßstab der Spurmitten-Karten).
- **D147 (Dreischichten-Plan der Spurmitte):** der Plan braucht 2 - 2.5 Arbeitstage und kann bis zum Stichtag nicht grün werden; Wahl A / B / C (Plan in `experiment_execution/plans/`).
- **TEST-OK für E343** (der TEST wird nur auf dein aufgezeichnetes Wort gelesen).
- **LICENSE-Datei:** es gibt keine; ohne sie darf niemand den Code nutzen.
- **Alte Mail-Adresse in der bereits gepushten Historie:** nur ein History-Rewrite entfernt sie; die Regeln verbieten das ohne dich.
- **Push:** `main` wurde mit deinem Wort (D149) bis zum Commit `6674f0a9` gepusht, ohne Tags; die Commits dieser Nacht danach sind lokal.
- **Weitere Punkte aus dem Handover:** die zurückgehaltenen Experimente E317 - E319, die Platte C: ist zu 95 % voll, die alte Referenzkette (D97), die Wahl des nächsten Experiments, das Aufräumen der Scratch- und Run-Ordner nach den Läufen.

## 9.3 Nächste Schritte (Vorschläge, nichts gestartet)

- "Knoten-Anker weglassen" (Nebenbefund E393) mit weiteren Seeds gegen die Latte prüfen: pro Schritt kostet das nichts extra (Pyramide fest), ein dritter Seed braucht aber ein eigenes B' und einen eigenen Cache.
- E395: ob die gemeinsame Querabweichung Information braucht, die die Merkmale eines Einzelbilds nicht enthalten (Vergangenheit, SD-Karte, zweite BEV-Ansicht), ist offen.
- t25, Reihenfolge der Reparaturliste der Karte: zuerst die Schreibmaske des Gedächtnisses (bewegte Objekte), dann ein zweiter Radar-Scan (braucht dein Wort), dann der rekursive Radarzustand statt des rohen Rasters für das Bewegungsfeld.
