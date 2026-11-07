p = "evaluation/bev_views.py"
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
rep('''        if gvel is not None:
            panels += self.layer_panels(grid, o0, g0, gt_prob, gt_dist, m, vel, gvel)
        veh = np.asarray''', '''        veh = np.asarray''')
rep('''            panels.append(Panel("text", "24 speed on vehicle cells: predicted against GT", txt))
        return Page(title="BEV outputs"''', '''            panels.append(Panel("text", "24 speed on vehicle cells: predicted against GT", txt))
        if gvel is not None:
            panels += self.layer_panels(grid, o0, g0, gt_prob, gt_dist, m, vel, gvel)
        return Page(title="BEV outputs"''')
open(p, "w", encoding="utf-8").write(s)
