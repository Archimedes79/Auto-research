"""One-off: add the pixel-permuted depth control (G4b) to the S1 synthetic gate after G4 (next sample's depth) failed on 2026-10-31."""
p = "tests/synthetic_s1_data.py"
s = open(p, encoding="utf-8").read()
a = '    if kind == "uniform":'
b = '''    if kind == "permuted":                    # the same histograms at random other pixel groups of the SAME sample: marginal kept, depth-pixel relation destroyed
        B, D_, h, w = prob.shape
        g = torch.Generator().manual_seed(1234)
        out = prob.clone().view(B, D_, h * w)
        for i in range(B):
            out[i] = out[i][:, torch.randperm(h * w, generator=g)]
        return out.view(B, D_, h, w)
    if kind == "uniform":'''
assert a in s
s = s.replace(a, b, 1)
s = s.replace("blurred (Gaussian in 1/z, mimics the frozen head's far-range blur), shuffled (another sample's), uniform", "blurred (Gaussian in 1/z, mimics the frozen head's far-range blur), shuffled (another sample's: keeps the scene-independent flat-ground depth, see the S1a card section 5), permuted (the sample's histograms at random other pixel groups), uniform")
s = s.replace("""    \"\"\"prob (B, D, h, w) oracle histograms -> the variant: 'oracle', 'blurred', 'shuffled' (the next sample's), 'uniform'.\"\"\"""", """    \"\"\"prob (B, D, h, w) oracle histograms -> the variant: 'oracle', 'blurred', 'shuffled' (the next sample's), 'permuted', 'uniform'.\"\"\"""")
open(p, "w", encoding="utf-8").write(s)

p = "tests/test_s1_pipeline.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''        "ours PD shuffled": ("ours", {}, "shuffled", (0,)),''', '''        "ours PD shuffled": ("ours", {}, "shuffled", (0,)),
        "ours PD permuted": ("ours", {}, "permuted", (0,)),''')
a = '''    sh_v, _ = mean_over_seeds("ours PD shuffled", veh)
    gates[f"G4 control: vehicles shuffled {sh_v:.3f} <= P {p_v:.3f} + 0.03"] = sh_v <= p_v + 0.03
'''
b = '''    sh_v, _ = mean_over_seeds("ours PD shuffled", veh)
    log(f"info: PD with the NEXT sample's depth (shuffled) reaches vehicles {sh_v:.3f} against P {p_v:.3f}: it keeps the scene-independent ground depth")
    if ("ours PD permuted", 0) in res:
        pm_v, _ = mean_over_seeds("ours PD permuted", veh)
        gates[f"G4b control: vehicles with pixel-permuted depth {pm_v:.3f} <= P {p_v:.3f} + 0.03"] = pm_v <= p_v + 0.03
'''
assert a in s
s = s.replace(a, b, 1)
s = s.replace("  G4 control    PD with shuffled depth: vehicles <= P + 0.03 (the depth CONTENT carries the gain, not the extra weight)",
              "  G4b control   PD with pixel-permuted depth (the sample's own histograms at random other pixel groups): vehicles <= P + 0.03 (the depth\n                CONTENT at its place carries the gain, not the extra weight). History: G4 as first written (PD with the NEXT sample's depth, vehicles <=\n                P + 0.03) FAILED in the first full run of 2026-10-31 (0.477 vs 0.286): that depth keeps the scene-independent flat-ground depth (89 % of\n                ground groups within 1 m of the truth), so it is no null; replaced by G4b, not removed silently")
open(p, "w", encoding="utf-8").write(s)
print("patched both")
