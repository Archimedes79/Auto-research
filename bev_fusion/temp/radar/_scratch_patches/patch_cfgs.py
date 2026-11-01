p = "temp/radar/_scratch_patches/make_configs.py"
s = open(p, encoding="utf-8").read()
s = s.replace('        c["steps"] = 2000\n        c["warmup"] = 200\n', '        c["steps"] = 1500                                  # accum 2: the same 9000 samples as the 3000 steps of t3_radar_only\n        c["accum"] = 2                                    # effective batch 6: every run with 3 had a dead velocity branch (t11)\n        c["warmup"] = 150\n        c["eval_every"] = 500\n        c["model"]["object_moving"] = True                # the t11 hurdle: moving / static logit, velocity on moving centres only\n')
open(p, "w", encoding="utf-8").write(s)
