p = "evaluation/bev_adapter.py"
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new, 1)
rep('''            "motion_l2": Hook("motion_block", pick=lambda o: o[-1]),                    # the finest level, whatever the number of levels
            "_rad_branch": Hook("rad_temporal", pick=lambda o: o[0])}                   # private (no TapSpec): the input of the radar aux head when the radar memory is on''',
    '''            "motion_l2": Hook("motion_block", pick=lambda o: o[-1])}                    # the finest level, whatever the number of levels''')
rep('''                want |= {"radar_feat", "_rad_branch"}''', '''                want |= {"radar_feat", "radar_memory"}''')
rep('''feat = arrays.get("_rad_branch", arrays.get("radar_feat"))''', '''feat = arrays.get("radar_memory", arrays.get("radar_feat"))                      # the branch's output: the memory's when on''')
open(p, "w", encoding="utf-8").write(s)
