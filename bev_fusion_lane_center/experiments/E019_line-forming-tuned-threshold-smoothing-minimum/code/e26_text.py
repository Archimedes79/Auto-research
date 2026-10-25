"""Text (Erkenntnisse, Grenzen) fuer den Bericht zur Sparsifizierung (Linienverfolger) aus den Zahlen der JSON-Dateien; wird von make_report_e26.py geladen."""
import json

f3 = lambda x: f"{x:.3f}".replace(".", ",")
f2 = lambda x: f"{x:.2f}".replace(".", ",")
f1 = lambda x: f"{x:.1f}".replace(".", ",")
f0 = lambda x: f"{x:.0f}"
sg = lambda x, n=3: (f"{x:+.{n}f}").replace(".", ",")

SKS, SKT = "Skelett Standard (0,5 m)", "Skelett abgestimmt"
V1, V1D, V1M, V2, V23, V26 = "Verfolger v1 (Punkt-F1 abgestimmt)", "v1 + Duplikate entfernt", "v1 + Mittelachsenzug", "Verfolger abgestimmt", "Verfolger abgestimmt, 3-m-Knoten", "Verfolger abgestimmt, 6-m-Knoten"
OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/_loop1_crn"


def auswertung(J, fam, ork, key, label):
    u = lambda k, a: J[k]["arms"][a]["ungesehen"]
    has_v1 = V1 in J[key]["arms"]
    o_sk, o_tr, o_t3 = u(ork, SKS)["1.0"][2], u(ork, V2)["1.0"][2], u(ork, V23)["1.0"][2]
    out = []
    out.append(
        f"Das Feld ist die Grundlage, nicht die Linienbildung: mit perfektem Feld (Soll-Abstandsfeld) erreichen Skelett und Verfolger F1 bei 1 m von {f3(o_sk)} / {f3(o_tr)}; die Sparsifizierung selbst verliert also hoechstens {f1(100 * (1 - o_tr))} % (abgestimmte Einstellung). "
        f"Mit dem vorhergesagten Feld von {label} sind es {f3(u(key, V2)['1.0'][2])}: der Rest ({f3(o_tr - u(key, V2)['1.0'][2])} F1) entsteht im Feld, nicht in der Linienbildung.")
    out.append(
        f"Sparsifizierung ohne Verlust: auf Knoten alle 3 m ausgeduennt (wie die Karte) bleibt der F1 bei 1 m gleich ({f3(u(key, V2)['1.0'][2])} -> {f3(u(key, V23)['1.0'][2])}), "
        f"mit {f0(u(key, V23)['nodes'])} Knoten je Sample statt {f0(u(key, SKT)['nodes'])} Skelett-Pixeln (Faktor {f1(u(key, SKT)['nodes'] / u(key, V23)['nodes'])}); die Karte hat {f0(J[key]['gt']['nodes'])} Knoten je Sample. "
        f"Auch 6-m-Knoten kosten nur {f3(u(key, V2)['1.0'][2] - u(key, V26)['1.0'][2])} F1.")
    out.append(
        f"Punkt fuer Punkt verfolgen gegen Schwelle + Skelett (beide abgestimmt, ungesehene Samples): F1 bei 1 m {f3(u(key, SKT)['1.0'][2])} -> {f3(u(key, V2)['1.0'][2])} ({sg(u(key, V2)['1.0'][2] - u(key, SKT)['1.0'][2])}), "
        f"gefundene Spuren {f3(u(key, SKT)['lane_recall'])} -> {f3(u(key, V2)['lane_recall'])}. Mit Standardparametern (ohne Abstimmung) liegt der Verfolger bei {f3(u(key, 'Verfolger Standardparameter')['1.0'][2])}: "
        f"besser als das Skelett mit Standard-Schwelle ({f3(u(key, SKS)['1.0'][2])}), aber nicht besser als das abgestimmte Skelett. Das Verfolgen ist also ein gleichwertiger, glatterer Sparsifizierer mit etwas mehr gefundenen Spuren, kein Sprung in der Genauigkeit.")
    t = (f"die Punkt-F1 belohnt Gewirr: mit der 1:1-Zuordnung (jeder Soll-Punkt zaehlt nur einmal) sinkt der F1 bei 1 m auf {f3(u(key, V2)['o2o']['1.0'][2])}; "
         f"die Obergrenze mit perfektem Feld liegt bei {f3(u(ork, V2)['o2o']['1.0'][2])}, weil sich Soll-Spuren teilweise ueberlappen. ")
    if has_v1:
        t += (f"Die Frueh-Fassung v1, nur auf Punkt-F1 abgestimmt, kommt auf Punkt-F1 {f3(u(key, V1)['1.0'][2])}, hat aber {f1(u(key, V1)['topo'][1])} Schleifen je Sample (Karte {f1(J[key]['gt']['topo'][1])}, Skelett {f1(u(key, SKT)['topo'][1])}), "
              f"vor allem in grossen Kreuzungen, wo das Feld ein breiter Fleck ist. Der Mittelachsen-Zug (breite Taeler zur Mitte ziehen) senkt die Schleifen ({f1(u(key, V1)['topo'][1])} -> {f1(u(key, V1M)['topo'][1])}) und kostet Ausbeute "
              f"(gefundene Spuren {f3(u(key, V1)['lane_recall'])} -> {f3(u(key, V1M)['lane_recall'])}); die Duplikat-Entfernung aendert wenig ({f1(u(key, V1D)['topo'][1])} Schleifen). ")
    t += (f"Die auf den 1:1-F1 abgestimmte Fassung v2 hat {f1(u(key, V2)['topo'][1])} Schleifen bei Punkt-F1 {f3(u(key, V2)['1.0'][2])} (Mittelachsen-Zug beta = {str(J[key]['arms'][V2]['params']['beta']).replace('.', ',')}); "
          "Gewirr und Ausbeute lassen sich mit diesem Feld nicht gleichzeitig verbessern.")
    out.append(t[0].upper() + t[1:])
    cal = json.load(open(f"{OUT}/trace_calib_{fam}.json"))

    def share(i0, i1):
        n = sum(cal["n"][i0:i1])
        return 100 * sum(cal["ok1"][i0:i1]) / max(n, 1)

    out.append(
        f"Die Pruefung am Feld ist nur so gut wie das Feld: von den Punkten mit Feldwert unter 0,05 m liegen {share(0, 1):.0f} % tatsaechlich innerhalb 1 m an der Karte; bei Feldwerten ab 1 m sind es nur {share(5, 7):.0f} % "
        "(nachsichtige Einstellung, jedes 4. Sample). Die Pruefung trennt also 'Linie' von 'keine Linie', erkennt aber Fehler im Feld nicht: eine Linie mitten in einem verschmolzenen Tal besteht die Pruefung ueberall (Feldwert um 0) "
        "und liegt trotzdem bis zu 1 m neben der Karte (Bild, dritte Spalte).")
    out.append(
        "Wo es nicht gelingt (Bilder): verschmolzene Spuren in einem breiten Tal (zwei bis vier Spuren werden eine oder zwei Linien), grosse Kreuzungs-Flecken (Linienknaeuel) und Flecken am Bildrand ohne Karte; der Verfolger bildet diese Feldfehler getreu ab.")
    tm, tk = J[key]["arms"][V2].get("zeit_ms"), J[key]["arms"][SKT].get("zeit_ms")
    out.append(f"Aufwand: der Verfolger braucht {f1(tm)} ms je Sample auf einem CPU-Kern (numba), das Skelett {f1(tk)} ms je Sample auf der GPU im Stapel; die Rechenzeit spricht gegen keines der beiden Verfahren.")
    out.append(
        "Folgerung fuer einen sparsamen Kopf: das dichte Feld taugt als Grundlage und als Pruefinstanz (Punkt auf Linie? ja/nein), die Verfolgung als Initialisierung der Linien; um verschmolzene Spuren und Kreuzungen zu loesen, muss der Kopf die Linien aus den Merkmalen lernen (nicht nur aus dem Feld), "
        "etwa mit dem Verfolger als Lehrer und dem Feld am Linienpunkt als zusaetzlichem Verlust.")
    return out


GRENZEN = [
    "Kein Training: der Verfolger ist ein festes Verfahren auf fertigen Feldern; seine 10 Parameter wurden per Koordinatenabstieg (2 Runden) auf den geraden Samples abgestimmt (lokales Optimum), das Skelett hat 2 abgestimmte Parameter (E19). Berichtet wird auf den ungeraden Samples; ein getrennter Testsatz existiert nicht.",
    "Die Topologie-Kennzahlen stammen aus dem duennen Raster: Schleifen = eingeschlossene Flaechen ab 4 m2; die Karte besteht aus einzelnen Spur-Polylinien (nicht verbunden an den Spurenden), daher sind Komponenten und Endpunkte der Karte (12 bzw. 32 je Sample) keine Referenz fuer 'richtige' Topologie, nur fuer den Vergleich der Arme. Eine echte Graph-Metrik (Verbindungen, Kreuzungen) fehlt weiter.",
    "Das Orakel nutzt das exakte Soll-Abstandsfeld im Fahrzeug-System. Die fuer gelernte Felder abgestimmte Skelett-Einstellung (Schwelle 0,15 m) ist auf dem exakten Soll-Feld nicht uebertragbar und wird dort nicht gezeigt.",
    "Ein Netz je Fundament (Seed 0); Unterschiede von wenigen Tausendstel F1 sind nicht gesichert. Die 1:1-Zuordnung ist gierig (naechste Paare zuerst), nicht optimal; ihre Obergrenze mit perfektem Feld liegt unter 1, weil sich Soll-Spuren teilweise ueberlappen.",
    "Der Verfolger sieht nur das Feld (keine Zeit, keine Karte, keine Fahrtrichtung); Spurzuordnung zur Karte, zeitliches Tracking und Navigationsziel sind nicht Teil dieses Versuchs.",
]
