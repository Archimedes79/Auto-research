"""one-off: --gate-doppler - each candidate's Doppler residual (BEFORE the Doppler update) as gate input"""
p = "tests/test_flow_explains.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''GATE = "--gate" in sys.argv                      # learned soft gate carried vs fresh start, recursive training (nuScenes)
if GATE:
    TAG = TAG + "_gate"''', '''GATE = "--gate" in sys.argv or "--gate-doppler" in sys.argv   # learned soft gate carried vs fresh start, recursive training
GATE_DOPPLER = "--gate-doppler" in sys.argv     # + each candidate's Doppler residual (before the update) as gate input
if GATE:
    TAG = TAG + ("_gatedop" if GATE_DOPPLER else "_gate")''')
rep('''            self.gate = nn.Sequential(nn.Conv2d(C + 6, 32, 3, padding=1),''',
    '''            self.gate = nn.Sequential(nn.Conv2d(C + 6 + (3 if GATE_DOPPLER else 0), 32, 3, padding=1),''')
rep('''        v0, corr = (r["v_mean"] if v_init is None else v_init), None      # v_init: recursive start (carried estimate)
        if DOPPLER and vgt is not None and v_override is None:      # the measurement corrects the start and EVERY iteration
            meas = radar_measure(f0, vgt)
''', '''        v0, corr = (r["v_mean"] if v_init is None else v_init), None      # v_init: recursive start (carried estimate)
        dres = None
        if DOPPLER and vgt is not None and v_override is None:      # the measurement corrects the start and EVERY iteration
            meas = radar_measure(f0, vgt)
            if GATE_DOPPLER and v_carried is not None:              # disagreement with the radar BEFORE the update (after
                u, d, valid = meas                                   # it both candidates agree radially by construction)
                B_ = v0.shape[0]

                def dop_res(v):
                    pred = (u[:, 0] * v.reshape(B_, 2, -1)).sum(1)
                    return ((pred - d[:, 0]).abs() * valid[:, 0]).view(B_, 1, G.nx, G.ny) / 10.0
                dres = torch.cat([dop_res(v0), dop_res(v_carried), valid.view(B_, 1, G.nx, G.ny)], 1)
''')
rep('''            w = torch.softmax(self.gate(torch.cat([e0, v0 / 10.0, v_carried / 10.0, rf_, rc_], 1)), 1)''',
    '''            gin = [e0, v0 / 10.0, v_carried / 10.0, rf_, rc_]
            if GATE_DOPPLER:
                gin.append(dres if dres is not None else e0.new_zeros(e0.shape[0], 3, G.nx, G.ny))
            w = torch.softmax(self.gate(torch.cat(gin, 1)), 1)''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
