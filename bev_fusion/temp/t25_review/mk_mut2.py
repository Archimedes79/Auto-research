src = open("temp/t25_review/mut_run.py").read()
marker = 'else:\n    raise SystemExit("unknown mutant " + name)'
new = '''elif name == "radar_moves_negated":
    orig = B._recur

    def recur(block, bevs, moves, state, fused, obs, residual=None):
        if residual is not None:
            moves = [[(dt, -v, w) for (dt, v, w) in mv] for mv in moves]
        return orig(block, bevs, moves, state, fused, obs, residual)
    B._recur = staticmethod(recur)
elif name == "radar_moves_omega_negated":
    orig = B._recur

    def recur(block, bevs, moves, state, fused, obs, residual=None):
        if residual is not None:
            moves = [[(dt, v, -w) for (dt, v, w) in mv] for mv in moves]
        return orig(block, bevs, moves, state, fused, obs, residual)
    B._recur = staticmethod(recur)
'''
assert marker in src
open("temp/t25_review/mut_run2.py", "w").write(src.replace(marker, new + marker))
print("ok")
