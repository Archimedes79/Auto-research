"""Charts of the final report 2026-11-23, drawn from the stored result files (no number is typed into a picture): `e390_owner_numbers.png` (chapter 5) and `e392_arms.png` (chapter 7).

    bash tools/env/_py14.sh report_charts.py --out <absolute dir>      # _py14.sh changes the directory: give an absolute --out; reads runs/e390_g7_s<S>/probe_card.json, runs/e390_sdgraph/nulls_val.json,
                                                                       #   runs/e392_<arm>_s<S>/token_card_rec_lt0.3_ltl0.2_kt0.6_nng_vs_<ref>.json
    bash tools/env/_py14.sh report_charts.py --selftest                # the row builders on hand-made dicts, no data files

e390_owner_numbers.png: the three numbers the owner asked for (E390 version 7, VAL, both seeds): lanes per direction under 30 % count error, the elements of missing segments left unassigned (20 % of the segments missing),
the association at a 2 m pose error of the map, each against the simplest rules (the same draws). Grey = a rule / null, blue = the encoder (seed 0 / seed 1). No red.
e392_arms.png: the difference of topo F1 to the control (the paired 95 % interval over frames) of every arm of E392, both seeds; the dashed line is the +0.010 band the written rules use. Colour-weakness-safe (Okabe-Ito).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CARD = "token_card_rec_lt0.3_ltl0.2_kt0.6_nng_vs_{ref}.json"
BLUE, BLUE2, GREY = "#0072B2", "#56B4E9", "#9a9a9a"
ARMS = [("x", "x: Austausch ohne frühes Loss", "e392_c_s{S}"), ("r1", "r1: eine Runde + frühes Loss", "e392_c_s{S}"), ("r2", "r2: zwei Runden + frühes Loss", "e392_c_s{S}"),
        ("a1", "a1: nur frühes Loss (1 Lesen)", "e392_c_s{S}"), ("a2", "a2: nur frühes Loss (2 Lesen)", "e392_c_s{S}"), ("u", "u2: r2 mit gleichen Nachrichten", "e392_c_s{S}"),
        ("n", "n: Block + frühes Loss, kein Austausch", "e392_c_s{S}"), ("m0", "m0: Block allein", "e392_c_s{S}"),
        ("ct", "ct: Zwilling von c (Kopf-Seeds A)", "e392_c_s{S}"), ("ctb", "ctb: Zwilling von c (Kopf-Seeds B)", "e392_c_s{S}"),
        ("r2t", "r2t: r2 gegen ct (Kopf-Seeds A)", "e392_ct_s{S}"), ("r2tb", "r2tb: r2 gegen ctb (Kopf-Seeds B)", "e392_ctb_s{S}")]


def owner_rows(cards, nulls):
    """cards = [probe_card of seed 0, of seed 1] (dicts with `levels`), nulls = the E390 nulls file -> [(title, [(label, value, kind)])], kind 'null' / 's0' / 's1': the three numbers of the owner"""
    L0, L1 = (c["levels"] for c in cards)
    nl = nulls["levels"]
    return [
        ("Spuren pro Richtung bei 30 % falschen Zählungen", [("Kopie der Eingabe", nl["base"]["p1_copy"][0], "null"), ("Bayes aus der Fehlerrate", nl["base"]["p1_bayes"][0], "null"),
                                                              ("Encoder, Seed 0", L0["base"]["p1"][0], "s0"), ("Encoder, Seed 1", L1["base"]["p1"][0], "s1")]),
        ("Elemente fehlender Segmente richtig nicht zugeordnet (20 % der Segmente fehlen)", [("nächste-Segment-Regel", nl["nd"]["p6_none_rule"][0], "null"), ("Encoder, Seed 0", L0["nd"]["p6_none"][0], "s0"),
                                                                                    ("Encoder, Seed 1", L1["nd"]["p6_none"][0], "s1")]),
        ("Zuordnung bei 2 m Lagefehler der Karte", [("nächste-Segment-Regel", nl["pose2"]["p6_rule"][0], "null"), ("logistisches Modell", nl["pose2"]["p6_lr"][0], "null"),
                                                    ("Encoder, Seed 0", L0["pose2"]["p6"][0], "s0"), ("Encoder, Seed 1", L1["pose2"]["p6"][0], "s1")]),
    ]


def arm_rows(read):
    """read(arm, seed) -> the paired topo F1 dict {diff, lo, hi} -> [(label, [(seed, diff, lo, hi)])] in the order of ARMS"""
    return [(lab, [(s, *[read(a, s)[k] for k in ("diff", "lo", "hi")]) for s in (0, 1)]) for a, lab, _ in ARMS]


def draw_owner(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = dict(null=GREY, s0=BLUE, s1=BLUE2)
    fig, axs = plt.subplots(len(rows), 1, figsize=(7.2, 1.25 * sum(len(r[1]) for r in rows) + 2.4), gridspec_kw=dict(height_ratios=[len(r[1]) for r in rows], hspace=0.7))
    for ax, (title, bars) in zip(axs, rows):
        n = len(bars)
        for i, (lab, v, kind) in enumerate(bars):
            y = n - 1 - i
            ax.barh(y, v, color=col[kind], height=0.7)
            ax.text(min(v + 0.012, 0.93), y, f"{v:.3f}", va="center", ha="left", fontsize=12)
        ax.set_yticks(range(n))
        ax.set_yticklabels([b[0] for b in bars][::-1], fontsize=11.5)
        ax.set_xlim(0, 1.0)
        ax.set_title(title, fontsize=12, loc="left")
        ax.tick_params(axis="x", labelsize=9.5)
        ax.grid(axis="x", alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axs:
        ax.set_xlabel("Anteil (VAL, 873 Frames; mehr = besser)", fontsize=10.5)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def draw_arms(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {0: BLUE, 1: BLUE2}
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.4, 0.62 * n + 1.5))
    for i, (lab, pts) in enumerate(rows):
        y = n - 1 - i
        for s, d, lo, hi in pts:
            yy = y + (0.17 if s == 0 else -0.17)
            ax.plot([lo, hi], [yy, yy], "-", color=col[s], lw=2.2, solid_capstyle="round")
            ax.plot([d], [yy], "o", color=col[s], ms=6.5, label=f"Seed {s}" if i == 0 else None, zorder=3)
    ax.axvline(0, color="#444", lw=1.0)
    ax.axvline(0.010, color="#444", lw=0.9, ls=(0, (4, 3)))
    ax.text(0.0103, n - 0.45, "Hürde +0.010", fontsize=10, color="#444", va="bottom")
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=11)
    ax.set_xlabel("Unterschied topo F1 zur Kontrolle (95-%-Intervall); rechts = besser", fontsize=10.5)
    ax.grid(axis="x", alpha=0.25)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=11, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def selftest():
    card = lambda a, b, c, d: dict(levels=dict(base=dict(p1=[a, 0, 0]), nd=dict(p6_none=[b, 0, 0]), pose2=dict(p6=[c, 0, 0])))   # noqa: E731
    nulls = dict(levels=dict(base=dict(p1_copy=[0.7, 0, 0], p1_bayes=[0.8, 0, 0]), nd=dict(p6_none_rule=[0.2, 0, 0]), pose2=dict(p6_rule=[0.82, 0, 0], p6_lr=[0.83, 0, 0])))
    rows = owner_rows([card(0.9, 0.7, 0.95, 0), card(0.91, 0.8, 0.94, 0)], nulls)
    assert [len(r[1]) for r in rows] == [4, 3, 4] and rows[0][1][2] == ("Encoder, Seed 0", 0.9, "s0") and rows[1][1][0][1] == 0.2 and rows[2][1][1] == ("logistisches Modell", 0.83, "null")
    ar = arm_rows(lambda a, s: dict(diff=0.01 * (s + 1), lo=-0.01, hi=0.03))
    assert len(ar) == len(ARMS) and ar[0][1][1] == (1, 0.02, -0.01, 0.03) and ar[2][0].startswith("r2:")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        draw_owner(rows, os.path.join(d, "a.png"))
        draw_arms(ar, os.path.join(d, "b.png"))
        assert os.path.getsize(os.path.join(d, "a.png")) > 5000 and os.path.getsize(os.path.join(d, "b.png")) > 5000
    for h in (BLUE, BLUE2, GREY, "#E69F00"):
        r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
        assert not (r > 150 and g < 110 and b < 110), f"a red colour {h}"
    print("report_charts selftest ok")


def main(out):
    runs = os.path.join(ROOT, "runs")
    cards = [json.load(open(os.path.join(runs, f"e390_g7_s{s}", "probe_card.json"))) for s in (0, 1)]
    nulls = json.load(open(os.path.join(runs, "e390_sdgraph", "nulls_val.json")))
    os.makedirs(out, exist_ok=True)
    draw_owner(owner_rows(cards, nulls), os.path.join(out, "e390_owner_numbers.png"))

    def read(arm, s):
        ref = dict((a, r) for a, _, r in ARMS)[arm].format(S=s)
        p = os.path.join(runs, f"e392_{arm}_s{s}", CARD.format(ref=ref))
        return json.load(open(p))["info"]["paired_vs_ref_run"]["topo_F1"]
    draw_arms(arm_rows(read), os.path.join(out, "e392_arms.png"))
    print("written", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        assert a.out and os.path.isabs(a.out), "--out <absolute dir>"
        main(a.out)
