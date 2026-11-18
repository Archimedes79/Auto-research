def sub(p, reps):
    t = open(p, encoding="utf-8").read()
    for a, b in reps:
        assert t.count(a) == 1, (p, a[:70])
        t = t.replace(a, b)
    open(p, "w", encoding="utf-8").write(t)

sub("training/data/lane_ptok_gpu.py", [
('''        last = torch.full((B * K * K,), -1, dtype=torch.long, device=self.dev)
        last.scatter_reduce_(0, key[ok], order[ok], "amax", include_self=True)  # the LAST pair of a cell wins (numpy's assignment order), deterministically
        win = ok & (last[key] == order)
        y = torch.zeros(B * K * K, dtype=torch.uint8, device=self.dev)
        y[key[ok]] = 1
        y = y.view(B, K, K)''','''        dump = B * K * K                                                        # one extra cell takes the padding rows: no boolean-mask indexing, no host sync
        keyd = torch.where(ok, key, torch.full_like(key, dump))
        last = torch.full((dump + 1,), -1, dtype=torch.long, device=self.dev)
        last.scatter_reduce_(0, keyd.reshape(-1), torch.where(ok, order, torch.full_like(order, -1)).reshape(-1), "amax", include_self=True)    # the LAST pair of a cell wins (numpy's assignment order), deterministically
        win = ok & (last[keyd] == order)
        y = torch.zeros(dump + 1, dtype=torch.uint8, device=self.dev)
        y[keyd.reshape(-1)] = 1                                                 # the same value for every claim of a cell
        y = y[:dump].view(B, K, K)'''),
('''            ys = torch.full((B * K * K, 2), float("nan"), device=self.dev)
            ys[key[win]] = yps[win]
            out["yscale"] = ys.view(B, K, K, 2)''','''            ys = torch.full((dump + 1, 2), float("nan"), device=self.dev)
            ys[torch.where(win, key, torch.full_like(key, dump)).reshape(-1)] = yps.reshape(-1, 2)         # unique cells for the winners; the losers write the dump cell
            out["yscale"] = ys[:dump].view(B, K, K, 2)'''),
])
sub("training/grid/jtoken_fast.py", [
('''Not covered (the plain head runs instead, `forward` delegates): prior="rule", a past or map block, live_inputs, and eval mode for the scales.''','''Not covered (the plain head runs instead, `forward` delegates): prior="rule", a past or map block, live_inputs. Eval mode computes the dense handle scales as the plain head does, unless `scale_in_eval = False` (frozen teachers).'''),
('''(the same as jtoken.source_rows of the E379 work; inlined here so that this file imports nothing that is not on main)''','''(the same as `jtoken.source_rows`)'''),
])
sub("training/train/jtoken_fast_train.py", [
('''        assert a.token_drop == 0 and a.teacher == "none" and not a.distill_from, "the GPU loader does not carry what --token-drop / --teacher rule / --distill-from read: add --cpu-loader"''','''        assert a.token_drop == 0 and a.teacher == "none" and not a.distill_from, "the GPU loader does not carry what --token-drop / --teacher rule / --distill-from read: add --cpu-loader"
    assert a.prior == "none" and not getattr(a, "live_inputs", False), "the speed path is the plain head: --prior rule runs the parent's forward, which has no scale_at for the fast loss"'''),
('''    JK.jtoken_loss = JF.fast_loss
    if not cpu_loader:''','''    JK.jtoken_loss = JF.fast_loss
    load = JT.load_teachers

    def fast_teachers(args, dev):                                            # frozen teachers (E379) read no handle scales: skip the dense scale MLP in their forward
        heads = load(args, dev)
        for h in heads:
            h.scale_in_eval = False
        return heads

    JT.load_teachers = fast_teachers
    if not cpu_loader:'''),
])
print("ok")
