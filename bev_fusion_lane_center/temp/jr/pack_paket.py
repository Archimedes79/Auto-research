"""Pack the mail package of the Spurmitte result report: html + champion code + read-me, one zip under 2 MB.
Run (WSL): cd /mnt/c/Auto-research/bev_fusion_lane_center && python3 temp/jr/pack_paket.py --html <report.html> --out <package.zip>
The code is read from git (files tracked under training/lanegraph, sources + README only); the package is refused when the
folder has uncommitted changes or the zip is 2 MB or more."""
import argparse, os, subprocess, sys, zipfile

LIMIT = 2_000_000
CODE = "training/lanegraph"
README = """Spurmitte - Ergebnisbericht (Paket, weniger als 2 MB)

1. Spurmitte_Ergebnisbericht.html  - der Bericht, eine Datei, Bilder und Zeichnung eingebettet (in jedem Browser oeffnen; beginnt mit "Start here")
2. champion_code/                  - sauberer Code des Champions (Inferenzpfad, 11 kleine Module + Skripte + Tests + README), laedt die trainierten Checkpoints unveraendert
3. Nicht im Paket: die Gewichte (5,8 MB fp16 waeren zu gross, und fp16 ist nicht exakt); die exakten liegen im Repository unter runs/e324_s_s<seed>.

Ende-zu-Ende-Status (D109, 2026-11-16): KEINE reine E2E-Loesung (benannte harte Schritte auf der Ausgabe bleiben, Liste auf der Startseite des Berichts).
Die Ergebnis-Zahl des Systems ist das End-to-End-Read-out (angeheftete Enden, nur Snap-Verdrahtung): VAL topo F1 0.6912 / 0.6948 (Seed 0 / 1).
Das gesnappte Read-out (0.6981 / 0.7010) enthaelt einen Snap-Geometrie-Regelschritt und wird nicht als Loesung akzeptiert.
Im Code: lg_infer.py --attached-ends (Config.attached_ends=True) liefert das End-to-End-Read-out; OHNE dieses Flag ist der Standard das gesnappte Read-out.

Stand: VAL = 873 Frames / 22 Szenen, TEST = 833 Frames / 21 Szenen (einmal benutzt, nur E312; der Champion E324 s hat noch keine TEST-Zahl).
Alle Zahlen stammen aus gespeicherten Karten und Ergebnisdateien; der Bericht nennt jeweils die Quelle.
"""


def git(*a):
    return subprocess.check_output(("git",) + a, text=True, encoding="utf-8").splitlines()


ap = argparse.ArgumentParser()
ap.add_argument("--html", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

dirty = git("status", "--porcelain", "--", CODE)
if dirty:
    sys.exit("refused: uncommitted changes in %s:\n%s" % (CODE, "\n".join(dirty)))
files = [f for f in git("ls-files", CODE) if f.endswith((".py", ".md"))]
with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    z.write(a.html, "Spurmitte_Ergebnisbericht.html")
    z.writestr("LIES_MICH.txt", README)
    for f in files:
        z.write(f, "champion_code/" + f[len(CODE) + 1:])
n = os.path.getsize(a.out)
print(a.out, n, "bytes", round(n / 1e6, 3), "MB;", len(files), "code files")
if n >= LIMIT:
    sys.exit("refused: package is %d bytes, limit %d" % (n, LIMIT))
