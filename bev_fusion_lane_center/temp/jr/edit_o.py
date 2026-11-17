"""Scratch (run once): afternoon edits of 09_rebuild.md, 07_code.md, 02_architecture.md, and the read-me text of pack_paket.py."""
import os
R = os.environ["BFLC_ROOT"]
D = os.path.join(R, "results", "Ergebnis_Spurmitte_2026-11-15", "parts")


def edit(fn, pairs, base=D):
    p = os.path.join(base, fn)
    t = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert t.count(old) == 1, (fn, t.count(old), old[:70])
        t = t.replace(old, new)
    open(p, "w", encoding="utf-8").write(t)


edit("09_rebuild.md", [
    ("### 8.10 Where to look: the experiments of 2026-11-15 and the night to 2026-11-16 (status 2026-11-16 05:10, from the result sections of the cards)",
     "### 8.10 Where to look: the experiments of 2026-11-15, the night to 2026-11-16 and the afternoon of 2026-11-16 (status 2026-11-16 afternoon, from the result sections of the cards)"),
    ("| **E324 s, end-to-end read-out** (+ learned attachment, geometry loss, 6000 steps; attached ends, no snap geometry) |",
     "| **E324 s, end-to-end read-out with the learned wiring head W** (E358: attached ends, the head replaces the nearest-piece wiring) | **0.693 / 0.696** | - | - | the reported read-out; +545 parameters, ms not measured |\n"
     "| **E324 s, end-to-end read-out, rule wiring** (+ learned attachment, geometry loss, 6000 steps; attached ends, no snap geometry) |"),
    ("Order of reading: the question and the rule of section 0 first",
     "| **E358** (D109 / D111) | the learned wiring head W on the attached ends instead of the nearest-piece rule | KEEP+: end-to-end read-out topo F1 0.6931 / 0.6964 against 0.6912 / 0.6948 (+0.0019 / +0.0016, intervals above 0, every guard held); +545 parameters; ms not measured | `E358_wiring-head-on-attached-ends` |\n"
     "| **E359** (D111) | T0 of the token design with a node-type output and a per-source softmax, 3000 steps, oracle nodes | FAIL on both seeds: topo F1 0.7721 / 0.7626, gap share 0.267 / 0.232, junction-link F1 0.531 / 0.502 | `E359_node-type-head-per-source-softmax` |\n"
     "| **E360** (D111) | the Matrix-NMS layer instead of the hard close-node step, against the end-to-end control (CPU replay) | INVALID by the written stop rule (iii); topo F1 +0.0162 / +0.0244 and junction F1 +0.087 / +0.100 over the control, duplicate guard broken; would be TIE with a looser tolerance | `E360_matrix-nms-in-the-e2e-stack` |\n"
     "| **E361** (D111) | E359 with 10000 steps | FAIL by a hair: topo F1 0.8316 / 0.8297 (half-gap line 0.834), junction-link F1 0.687 / 0.688 (bar 0.75) | `E361_node-type-head-10000-steps` |\n"
     "| **E362** | E361 with 16000 steps | text draft only, not started | `E362_node-type-head-16000-steps` |\n"
     "Order of reading: the question and the rule of section 0 first"),
])

edit("07_code.md", [
    ("The attach-only read-out (attached ends without any wiring rule, every piece in reach, 0.6616 / 0.6646) is not implemented.",
     "The package does **not** contain the learned wiring head W of E358 (the reported end-to-end read-out 0.6931 / 0.6964 is the card's, from the evaluation code of the experiment): `--attached-ends` gives the rule-wired control 0.6912 / 0.6948. The attach-only read-out (attached ends without any wiring rule, every piece in reach, 0.6616 / 0.6646) is not implemented."),
])

edit("02_architecture.md", [
    ("VAL topo F1 of the end-to-end read-out 0.6912 / 0.6948 (the headline since D109;",
     "VAL topo F1 of the end-to-end read-out 0.6931 / 0.6964 with the learned wiring head W of E358 (rule wiring, the control: 0.6912 / 0.6948; the headline since D109;"),
])

# the read-me of the package
pp = os.path.join(R, "temp", "jr", "pack_paket.py")
t = open(pp, encoding="utf-8").read()
old = "Die Ergebnis-Zahl des Systems ist das End-to-End-Read-out (angeheftete Enden, nur Snap-Verdrahtung): VAL topo F1 0.6912 / 0.6948 (Seed 0 / 1).\n"
new = ("Die Ergebnis-Zahl des Systems ist das End-to-End-Read-out (angeheftete Enden, gelernter Verdrahtungskopf W, E358): VAL topo F1 0.6931 / 0.6964 (Seed 0 / 1);\n"
       "die Kontrolle mit Regel-Verdrahtung (naechstes Stueck) ist 0.6912 / 0.6948. Der Kopf W ist NICHT im Paket-Code; der Code liefert mit --attached-ends die Kontrolle 0.6912 / 0.6948.\n")
assert t.count(old) == 1
t = t.replace(old, new)
old2 = "Im Code: lg_infer.py --attached-ends (Config.attached_ends=True) liefert das End-to-End-Read-out; OHNE dieses Flag ist der Standard das gesnappte Read-out."
assert t.count(old2) == 1
t = t.replace(old2, "Im Code: lg_infer.py --attached-ends (Config.attached_ends=True) liefert das End-to-End-Read-out mit Regel-Verdrahtung; OHNE dieses Flag ist der Standard das gesnappte Read-out.\nDer Rechner wird heute um 17:30 auf Mirkos Anweisung heruntergefahren (D111); spaetere Ergebnisse stehen in HANDOVER.md, nicht im Bericht.")
open(pp, "w", encoding="utf-8").write(t)
print("ok")
