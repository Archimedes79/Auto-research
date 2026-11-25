# 10. Reproduzierbarkeit: ein Lauf aus dem frischen Klon

Seit dem 2026-11-24 gibt es `autoresearch/tools/recipe.py`. Pro Experiment liegen eine `recipe.json` und, wo nötig, ein `code/`-Ordner in Git: der **Basis-Commit** auf `main`, der **Befehl** (Ledger) und das **Overlay**, also nur die ausgeführten Dateien, die vom Basis-Commit abweichen (Versionen, die schon in der Historie stehen, nur als Blob-ID). Damit braucht ein Klon keine der lokalen Run-Tags.

- `python3 autoresearch/tools/recipe.py show <run id>`: Briefing für Mensch oder KI (Frage, Regel, Code, Befehl, Eingaben, erwartetes Ergebnis, Lücken).
- `python3 autoresearch/tools/recipe.py repeat <run id>`: baut den Arbeitsbaum und druckt den `run.py`-Befehl (startet nichts).
- `python3 autoresearch/tools/recipe.py check`: welche Läufe sich wiederholen lassen.

Nach dem Push vom 2026-11-25 waren 1124 von 1304 Lane-Läufen exakt wiederholbar und 8 näherungsweise (Ausgabe von `recipe.py check` direkt danach, nicht als Datei gespeichert). Für die Läufe dieser Nacht (Spurmitte E391b, E393, E394, E395 und BEV t25) sind die Rezepte geschrieben und lokal committet; aus einem Klon wiederholbar werden sie mit dem nächsten Push, mit diesen Ausnahmen:

- `E394.gdesign_s0`, der Lauf hinter dem Abschluss von E394: das Skript lag nicht im Snapshot des Laufs. Die zwei Minuten nach dem Lauf committete Fassung ist vermutlich die gelaufene, belegt ist das nicht.
- `E395.d115_vs_p0nt_s1` und `E395.sweep_p0ntb_s0`: der Starter wurde beendet, das Ledger hat keinen Befehl (von Hand nachgetragen), es gibt kein Rezept.

(`E391b.read_s0` und `read_s1` sind nie gelaufen.) Daten, Checkpoints und Umgebung liegen nicht in Git; `show` nennt, woher sie kommen.
