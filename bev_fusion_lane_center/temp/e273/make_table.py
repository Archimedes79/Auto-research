"""one-off: the numbers table of the E273 result README, read from the scoring cards runs/e273/<run>/card.json (never typed by hand)

    python3 temp/e273/make_table.py e273_p_s0 e273_p_s1 > temp/e273/table.md
"""
import json
import sys

runs = sys.argv[1:]
cards = [json.load(open(f"runs/e273/{r}/card.json")) for r in runs]
d_ref = cards[0]["d_ref"] if "d_ref" in cards[0] else {}


def f(v, n=3):
    return "-" if v is None else f"{v:.{n}f}"


rows = []
hdr = "| VAL 873 frames | " + " | ".join(f"p seed {c['seed']}" for c in cards) + " | mean p | g, mean of the seeds (E272 chain + geometric assigner) | d (recall_sp0_dp) |"
rows.append(hdr)
rows.append("|" + "---|" * (len(cards) + 4))


def mean(vs):
    vs = [v for v in vs if v is not None]
    return sum(vs) / len(vs) if vs else None


def line(name, fp, g=None, d=None, n=4):
    vals = [fp(c) for c in cards]
    rows.append(f"| {name} | " + " | ".join(f(v, n) for v in vals) + f" | {f(mean(vals), n)} | {f(g, n) if g is not None else ''} | {f(d, n) if d is not None else ''} |")


gm = lambda a, k: sum(c["g"][a][k] for c in cards) / len(cards)                    # g is the E272 chain of the same seed: mean of the seeds
line("lane_acc, map error 1-2 m", lambda c: c["p"]["assign_error"]["lane_acc"], g=gm("assign_error", "lane_acc"))
line("lane_acc, exact map", lambda c: c["p"]["assign_exact"]["lane_acc"], g=gm("assign_exact", "lane_acc"))
line("road_acc (road level derived from the lanes), map error", lambda c: c["p"]["assign_error"]["road_acc"], g=gm("assign_error", "road_acc"))
line("p - g, map error", lambda c: c["diff"]["error"]["mean"])
line("p - g, exact map", lambda c: c["diff"]["exact"]["mean"])
D = {"topo_F1": d_ref.get("topo_F1"), "F1@1.0": d_ref.get("F1_1"), "junction_F1": d_ref.get("junction_F1"), "lanes_found": d_ref.get("lanes_found"), "Q": d_ref.get("Q"), "dup_frac": d_ref.get("dup_frac"), "vert100_dp": None}
for k in ("topo_F1", "F1@1.0", "junction_F1", "lanes_found", "Q", "dup_frac", "vert100_dp", "pieces_per_gt_piece", "id_switches_per_100m"):
    line(k, lambda c, k=k: c["p"]["graph"].get(k), d=D.get(k), n=3)
line("node F1 (2 m grid)", lambda c: c["p"]["nodes"].get("node_F1"))
line("node position error, median (m)", lambda c: c["p"]["nodes"].get("pos_err_med"))
print("\n".join(rows))
print()
print("p - g with the 95 % scene bootstrap (873 VAL frames, 1000 resamples of scenes, paired):")
for c in cards:
    e, x = c["diff"]["error"], c["diff"]["exact"]
    print(f"- seed {c['seed']}: map error {e['mean']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}]; exact map {x['mean']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]")
