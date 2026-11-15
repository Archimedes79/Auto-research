src = open("temp/t25_review/mut_run.py").read()
start = src.index('if name == "carry_block_output":')
end = src.index('elif name == "radar_no_ego_motion":')
new = '''if name == "carry_block_output":
    orig = B._recur

    def recur(block, bevs, moves, state, fused, obs, residual=None):
        if residual is None:
            return orig(block, bevs, moves, state, fused, obs, residual)
        past, res = fused, None
        for j, k in enumerate(range(bevs.shape[1] - 1, -1, -1)):
            out, state, _ = block(bevs[:, k], [] if past is None else [(past, moves[j])], state, moves[j], obs[k])
            past = out
            res = bevs[:, k] + residual(out)
        return res, state
    B._recur = staticmethod(recur)
'''
open("temp/t25_review/mut_run.py", "w").write(src[:start] + new + src[end:])
import ast
ast.parse(open("temp/t25_review/mut_run.py").read())
print("syntax ok")
