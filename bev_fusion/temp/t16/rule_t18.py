"""scratch: the written rule of the t18 card (section 1) applied to the stored tables of the control t17_data (a) and the bundle arm t18_fix (b);
the band per metric = max(floor, the 95 % half-width of the paired scene bootstrap of compare_t18.txt).
    wsl python3 bev_fusion/temp/t16/rule_t18.py > bev_fusion/temp/t16/rule_t18.txt   (from the repository root)"""
import re


def load(p):
    d = {}
    for line in open(p, encoding="utf-8"):
        m = re.match(r"^(\S+)\s+(-?[0-9.]+(?:e-?[0-9]+)?|nan)\s*$", line.rstrip())
        if m:
            d[m.group(1)] = float(m.group(2))
    return d


def half_widths(p):
    """metric key (table spelling) -> half-width of the 95 % interval of the compare output"""
    out = {}
    for line in open(p, encoding="utf-8"):
        m = re.match(r"^(\S+)\s+a\s+(\S+)\s+b\s+(\S+)\s+b-a\s+(\S+)\s+95% \[(\S+), (\S+)\]", line)
        if m:
            lo, hi = float(m.group(5)), float(m.group(6))
            key = re.sub(r"^(seg\.fov|seg\.obj|det|detf|detfp|curve|vel|loss)\.dt", lambda x: x.group(1).replace(".", "_") + "/dt", m.group(1), count=1)
            out[m.group(1)] = (hi - lo) / 2
    return out


A, B = "t17_data", "t18_fix"
va, vb = (load(f"bev_fusion/runs/{r}/eval/by_n/table_n1000.txt") for r in (A, B))
hw = half_widths("bev_fusion/temp/t16/compare_t18.txt")


def hwk(key):
    """the compare output spells 'vel.dt+0.00/...' where the table has 'vel/dt+0.00/...'"""
    k = key.replace("seg_fov/seg/", "seg.fov/").replace("seg_obj/seg/", "seg.obj/")
    k = re.sub(r"^(vel|det|detf|detfp|curve|loss)/dt", r"\1.dt", k)
    return hw.get(k)


ST, MV, CE = "vel/dt+0.00/cells/static/epe", "vel/dt+0.00/cells/vehicle_moving/epe", "vel/dt+0.00/centre/field/moving/epe"
print("VAL 1000 frames, GT v8, one seed each; a = t17_data (control), b = t18_fix (bundle: hygiene flags + motion block lr 1e-4, warm-up 1000); b-a < 0 is better for EPE")
rows = [("static EPE (cells, m/s)", ST, 0.02), ("moving vehicle EPE (cells, m/s)", MV, 0.10), ("moving box-centre EPE (m/s)", CE, 0.10)]
res = {}
for n, k, floor in rows:
    h = hwk(k)
    band = max(floor, h if h is not None else 0)
    d = vb[k] - va[k]
    res[k] = (d, band)
    print(f"  {n:34s} a {va[k]:.4f}  b {vb[k]:.4f}  b-a {d:+.4f}  band {band:.4f} (floor {floor}, bootstrap half-width {h if h is not None else float('nan'):.4f})  ratio b/a {vb[k] / va[k]:.2f}")
d_st, band_st = res[ST]
print(f"\nprediction: static EPE <= 0.6 x control ({0.6 * va[ST]:.4f}) and <= 0.30: {vb[ST]:.4f}  -> {'HIT' if vb[ST] <= 0.6 * va[ST] and vb[ST] <= 0.30 else 'MISS'}")
GUARDS = [("front mAP2 (det mean mAP)", "det/dt+0.00/mean/all/mAP", 0.010, -1), ("vehicle IoU at +-0.5 s (seg_obj, +0.5)", "seg_obj/seg/dt+0.50/vehicle/all/all/iou@0", 0.02, -1),
          ("vehicle IoU at +-0.5 s (seg_obj, -0.5)", "seg_obj/seg/dt+-0.50/vehicle/all/all/iou@0", 0.02, -1)]
MAPS = ["seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0", "seg_fov/seg/dt+0.00/walkway/all/all/iou@0", "seg_fov/seg/dt+0.00/carpark_area/all/all/iou@0",
        "seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5", "seg_fov/seg/dt+0.00/road_divider/all/all/iou@1"]
mean_a = sum(va[k] for k in MAPS if k in va and k in vb) / len(MAPS)
mean_b = sum(vb[k] for k in MAPS if k in va and k in vb) / len(MAPS)
g_ok = True
for n, k, band, sign in GUARDS:
    if k not in va or k not in vb:
        print(f"  guard {n:40s} missing in a table")
        continue
    d = vb[k] - va[k]
    bad = d < -band
    g_ok &= not bad
    print(f"  guard {n:40s} a {va[k]:.4f}  b {vb[k]:.4f}  b-a {d:+.4f}  band {band}  {'WORSE' if bad else 'holds'}")
dm = mean_b - mean_a
bad = dm < -0.010
g_ok &= not bad
print(f"  guard {'maps (mean of the six IoUs)':40s} a {mean_a:.4f}  b {mean_b:.4f}  b-a {dm:+.4f}  band 0.010  {'WORSE' if bad else 'holds'}")

d_mv, band_mv = res[MV]
d_ce, band_ce = res[CE]
moving_worse = d_mv > band_mv or d_ce > band_ce
if -d_st > 3 * band_st and not moving_worse and g_ok:
    verdict = "ADOPT"
elif -d_st > band_st and not moving_worse:
    verdict = "CANDIDATE"
elif moving_worse:
    verdict = "CUT"
else:
    verdict = "TIE"
print(f"\nrule (card section 1): static better by > 3 x band AND moving / centre not worse than the band AND guards hold -> ADOPT; static better by > band -> CANDIDATE; moving or centre worse by > band -> CUT; static inside band -> TIE")
print(f"static b-a {d_st:+.4f} (band {band_st:.4f}); moving b-a {d_mv:+.4f} (band {band_mv:.4f}); centre b-a {d_ce:+.4f} (band {band_ce:.4f}); moving/centre worse than the band: {moving_worse}; guards hold: {g_ok}")
print(f"VERDICT (by hand: one seed per arm; the bundle is 'lr + hygiene flags'): {verdict}")
