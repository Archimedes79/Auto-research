lines = open("mutate.py", encoding="utf-8").read().split("\n")
lines[41] = ''' ("score_seed B flag unchecked", "\\"B's dump\\", n, exp(run_b))}", "\\"B's dump\\", n, exp(run_b) if i[\\"b_made\\"] else None)}"),'''
lines[42] = ''' ("score_seed ctl ends unchecked", "get_dump(i[\\"ctl_ends\\"], None, split, \\"the control's pair dump\\", n, exp(sub_seed(CONTROL_RUN, seed)))", "load_dump(i[\\"ctl_ends\\"], split, \\"the control's pair dump\\", n)"),'''
lines[43] = ''' ("score_seed ctl B unchecked", "get_dump(i[\\"ctl_b\\"], None, split, \\"the control's B dump\\", n, exp(run_b))", "load_dump(i[\\"ctl_b\\"], split, \\"the control's B dump\\", n)"),'''
open("mutate.py", "w", encoding="utf-8").write("\n".join(lines))
