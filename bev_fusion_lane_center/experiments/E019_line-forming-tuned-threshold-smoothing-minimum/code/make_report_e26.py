"""Bericht zur Sparsifizierung (E26 und Folgeversuche): Punkt-fuer-Punkt-Linienverfolger mit Linienpruefung.
Aufruf: make_report_e26.py <crn|gcar> [Berichtsnummer]   (liest out/crn/exp_trace_<quelle>.json, Bilder aus plot_e26.py, Text aus e26_text.py)"""
import importlib.util
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_lib import blocks_arch  # noqa: E402
from report_lib import make_report as _make_report  # noqa: E402

O = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/_loop1_crn"
FAM = {"crn": ("crn_orakel", "crn_unet64", "Kamera+Radar (CRN) U-Netz 64", "R26"), "gcar": ("crn_orakel", "gcar_unet64", "Kamera+Radar (GaussianCaR) U-Netz 64", None)}
SKS, SKT = "Skelett Standard (0,5 m)", "Skelett abgestimmt"
V1, V1D, V1M, V2, V23, V26 = "Verfolger v1 (Punkt-F1 abgestimmt)", "v1 + Duplikate entfernt", "v1 + Mittelachsenzug", "Verfolger abgestimmt", "Verfolger abgestimmt, 3-m-Knoten", "Verfolger abgestimmt, 6-m-Knoten"
SHOW = {V2: "Verfolger v2 (1:1 abgestimmt)", V23: "Verfolger v2, Knoten alle 3 m", V26: "Verfolger v2, Knoten alle 6 m", "Verfolger Standardparameter": "Verfolger, Standardparameter (nicht abgestimmt)"}
f3 = lambda x: f"{x:.3f}".replace(".", ",")
f1 = lambda x: f"{x:.1f}".replace(".", ",")
f0 = lambda x: f"{x:.0f}"


def _dec(s):
    return re.sub(r"(\d)\.(\d)( %)", r"\1,\2\3", str(s))


def make_report(exp_id, title, frage, aufbau, arch, table, images, auswertung, grenzen=(), **kw):
    head, rows = table
    return _make_report(exp_id, title, frage, [_dec(a) for a in aufbau], arch, (head, [[_dec(c) for c in r] for r in rows]), images, [_dec(a) for a in auswertung], [_dec(g) for g in grenzen], **kw)


def arch(ax):
    bl = [
        (0.1, 2.55, 2.3, 1.0, "Abstandsfeld\n200 x 200 Pixel\n(eingefroren)", "frozen"),
        (2.6, 2.55, 2.3, 1.0, "1 Saatpunkt:\nbester freier Tal-\nboden (Feldminimum)", "post"),
        (5.1, 2.55, 2.3, 1.0, "2 Schritt: Faecher\n+-45 Grad, 1 m vor,\ntiefstes Feld", "post"),
        (7.6, 2.55, 2.3, 1.0, "3 Zentrieren:\nquer zur Linie in\ndie Talmitte", "post"),
        (10.1, 2.55, 2.3, 1.0, "4 Pruefen:\nRohfeld <= t_on?\nLuecke <= gap_max", "post"),
        (12.6, 2.55, 2.3, 1.0, "5 Treffen: Winkel\nklein: verschmelzen\ngross: kreuzen", "post"),
        (0.1, 0.55, 4.8, 1.0, "Vergleichsarm: Schwelle -> Zhang-Suen-\nSkelett -> Pixelgraph (E03, E19)", "data"),
        (5.6, 0.55, 3.8, 1.0, "Polylinien (Vektor)\nKnoten 1 m, ausgeduennt 3 m / 6 m", "ok"),
        (10.1, 0.55, 4.8, 1.0, "Auswertung: Punkt-F1, 1:1-F1, Spuren,\nSchleifen, Knoten; Orakel = Soll-Feld", "ok"),
    ]
    blocks_arch(ax, bl, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 8), (0, 6)], height=3.9,
                note="Schritte 2-5 wiederholen sich, bis das Feld die Linie nicht mehr bestaetigt (Ende), sie auf eine andere Linie trifft oder der Rand erreicht ist; dann der naechste Saatpunkt.")


def build(fam, exp_id):
    ork, key, label, _ = FAM[fam]
    J = {k: json.load(open(f"{O}/exp_trace_{k}.json")) for k in (ork, key)}
    NAME = {ork: "Soll-Feld (Orakel)", key: label}
    real = [SKS, SKT, "Verfolger Standardparameter"] + [a for a in (V1, V1D, V1M) if a in J[key]["arms"]] + [V2, V23, V26]
    ARMS = {ork: [SKS, V2, V23], key: real}
    rows = []
    for k in (ork, key):
        for arm in ARMS[k]:
            u = J[k]["arms"][arm]["ungesehen"]
            rows.append([NAME[k] if arm == SKS else "", SHOW.get(arm, arm), f3(u["0.5"][2]), f3(u["1.0"][2]), f3(u["1.5"][2]), f3(u["o2o"]["1.0"][2]), f"{f3(u['1.0'][0])} / {f3(u['1.0'][1])}",
                         f3(u["lane_recall"]), f1(u["topo"][1]), f0(u["nodes"])])
        g = J[k]["gt"]
        rows.append(["", "Karte (Soll, gezeichnet)", "-", "-", "-", "-", "-", "-", f1(g["topo"][1]), f"{f0(g['nodes'])} (alle 3 m)"])
    pr = J[key]["arms"][V2]["params"]
    ptxt = ", ".join(f"{n} {pr[n]}" for n in ("sig_nav", "t_seed", "t_on", "min_len", "gap_max", "lam", "r_seed", "merge_deg", "beta", "dup_m")).replace(".", ",")
    aufbau = [
        f"Pixelbasiert (Feld) und vektorbasiert (Ausgabe): kein neues Netz, kein Training. Eingabe sind die fertigen Abstandsfelder der 1.706 Validierungs-Samples (Teile 1-3): {label} (bestes Netz dieses Fundaments); dazu das Soll-Abstandsfeld als Orakel (perfektes Feld; misst den Verlust der Linienbildung allein).",
        "Verfolger (line_tracer.py, numba): Saatpunkt = bester noch nicht abgedeckter Talboden des Feldes; Schritt = Richtung aus einem Faecher von 9 Richtungen (+-45 Grad) mit dem tiefsten Feld 1 m und 2 m voraus, 1 m vorruecken; Zentrieren = Punkt quer zur Fahrtrichtung in die Mitte des Tals schieben (Schwerpunkt des Querschnitts, +-1,5 px); Pruefen = Rohfeld am Punkt <= t_on, hoechstens gap_max Schritte ohne Linie werden ueberbrueckt, danach Ende; Treffen = stoesst der Kopf auf eine fertige Linie, verschmelzen (Winkel unter merge_deg: Abzweig / Zusammenfuehrung) oder kreuzen (Winkel darueber). Um fertige Linien werden neue Saatpunkte gesperrt (r_seed). Ausgabe: Polylinien, Knoten alle 1 m.",
        "Zwei Erweiterungen (v2, aus dem Gewirr der Fassung v1 entstanden): Mittelachsen-Zug beta (Richtungswahl und Saatpunkte nutzen Feld + beta * (12 px - halbe Talbreite), damit in breiten Flecken nur die Mitte verfolgt wird) und Duplikat-Entfernung (Linien, die zu 70 % ihrer Laenge innerhalb dup_m einer laengeren liegen, entfallen).",
        "Arme: Skelett Standard (Schwelle 0,5 m), Skelett abgestimmt (E19: 0,15 m, Mindestlaenge 30 px), Verfolger mit Standardparametern, Verfolger v1 (nur auf Punkt-F1 abgestimmt, ohne die Erweiterungen) mit je einer Erweiterung einzeln (falls gemessen), Verfolger v2 (alle 10 Parameter per Koordinatenabstieg, 2 Runden, auf den geraden Samples, Ziel: F1 bei 1 m mit 1:1-Zuordnung), v2 auf Knoten alle 3 m bzw. 6 m ausgeduennt (lineare Zwischenpunkte fuer die Auswertung; 3 m = Knotenabstand der Karte).",
        "Auswertung auf den 853 ungeraden Samples (weder Skelett in E19 noch Verfolger dort abgestimmt): Punkt-F1 gegen die dichten Karten-Mittellinienpunkte wie in E19; F1 mit 1:1-Zuordnung (gierig, naechste Paare zuerst: jeder Soll-Punkt zaehlt nur einmal, doppelte Linien werden bestraft); Spuren gefunden = Anteil der Soll-Spuren, von denen mindestens 80 % der Punkte innerhalb 1 m der geschaetzten Linien liegen; Schleifen je Sample = eingeschlossene Flaechen ab 4 m2 im duennen Raster; Knoten bzw. Skelett-Pixel je Sample.",
        f"Abgestimmte Parameter von v2: {ptxt}.",
    ]
    imgs = [
        (f"{O}/viz_trace_{fam}_beispiele.png", f"{label}: Feld, Skelett (abgestimmt), Verfolger (v1, v2) und v2 auf 3-m-Knoten (rot) gegen Karte (blau); Val-Samples 40, 500, 1100, 1500"),
        (f"{O}/viz_trace_{fam}_schritte.png", "Wie der Verfolger vorgeht (Sample 1100): Saatpunkte, Linien nach Reihenfolge mit Grund des Endes, Pruefung entlang der Linie (Feldwert gegen wahren Abstand) und Verlaesslichkeit der Pruefung"),
        (f"{O}/viz_trace_{fam}_zahlen.png", "Punkt-F1, 1:1-F1, gefundene Spuren, Schleifen, Komponenten und Knoten fuer Orakel und vorhergesagtes Feld (ungesehene Samples; blau gestrichelt: Karte)"),
    ]
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e26_text.py")
    spec = importlib.util.spec_from_file_location("e26_text", p)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    make_report(
        exp_id, "Sparsifizierung auf dem Abstandsfeld: Punkt-fuer-Punkt-Linienverfolger mit Linienpruefung" + ("" if fam == "crn" else f" ({label.split(' (')[1].split(')')[0]})"),
        "Taugt das dichte Abstandsfeld des besten Netzes als Grundlage fuer eine Sparsifizierung (Feld -> wenige Linienpunkte)? Liegt der Verlust in der Linienbildung oder im Feld, und hilft Verfolgen Punkt fuer Punkt mit Pruefung auf der Linie gegen Schwelle + Skelett?",
        aufbau, arch,
        (["Quelle", "Arm", "F1 @0,5 m", "F1 @1,0 m", "F1 @1,5 m", "F1 @1 m (1:1)", "Praez. / Ausb. @1 m", "Spuren gefunden", "Schleifen je Sample", "Knoten bzw. Pixel je Sample"], rows),
        imgs, T.auswertung(J, fam, ork, key, label), T.GRENZEN, status="abgeschlossen", arch_height=3.9, img_ratios=[11.9, 5.6, 7.3],
        col_widths=[0.12, 0.2, 0.07, 0.07, 0.07, 0.08, 0.1, 0.08, 0.09, 0.12])
    print(exp_id, "geschrieben")


if __name__ == "__main__":
    fam = sys.argv[1] if len(sys.argv) > 1 else "crn"
    exp_id = sys.argv[2] if len(sys.argv) > 2 else FAM[fam][3]
    build(fam, exp_id)
