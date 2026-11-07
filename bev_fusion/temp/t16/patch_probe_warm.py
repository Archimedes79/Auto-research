p = "bev_fusion/temp/t16/probe_modules.py"
s = open(p, encoding="utf-8").read()
a = """        refs += [fr[len(fr) // 2], fr[len(fr) // 3]]
    refs = refs[:a.frames]
"""
b = """        refs += [(s, len(fr) // 2), (s, len(fr) // 3)]
    refs = refs[:a.frames]
"""
assert s.count(a) == 1
s = s.replace(a, b)
a = """    for k, ref in enumerate(refs):
        sample = src.load(ref)
"""
b = """    for k, (scene, idx) in enumerate(refs):
        ref = src.frames(scene)[idx]
        sample = src.load(ref)
"""
assert s.count(a) == 1
s = s.replace(a, b)
a = """        modules.setdefault("features(S)", []).append(stat(res["features"]))
"""
b = """        modules.setdefault("features(S)", []).append(stat(res["features"]))
        # the warm frame: the next frame with the temporal state of this one (the memories read what the first call wrote)
        s2 = src.load(src.frames(scene)[idx + 1])
        store.clear()
        with torch.no_grad():
            _, resw, _ = T.forward_loss(model, loss, T.to_device(s2.inputs, subj.device), 1.0, subj.device, cfg.amp, cfg, train_terms=False,
                                        state=ns.detach_tree(res["state"]), step=ns.clip_step(sample.inputs, subj.device))
        for n, d in store.items():
            for p, t in d.items():
                if n in ("cam_temporal", "rad_temporal", "fusion"):
                    modules.setdefault(f"WARM {n}:{p}", []).append(stat(t))
        store.clear()
"""
assert s.count(a) == 1
s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("patched")
