p = "tests/test_student_window.py"
s = open(p).read()
n0 = len(s)
s = s.replace('''pb = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bc.items()}
BP_.perturb(pb, "zero_cam")''', '''pb = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bc.items()}
pb["tokens"] = ["a", "b"]
BP_.perturb(pb, "zero_cam")''')
s = s.replace('''any("lacks 1 of the config's 3 scenes" in x and "scene-9999" in x for x in''', '''any(f"lacks 1 of the config's {len(s1 | s2) + 1} scenes" in x and "scene-9999" in x for x in''')
s = s.replace('''    LB.build_cache([_subset(toks[:2])], root, 1, older=True)
    bt = LB.older_built(root)''', '''    with contextlib.redirect_stdout(_Capture()):
        LB.build_cache([_subset(toks[:2])], root, 1, older=True)
    bt = LB.older_built(root)''')
s = s.replace('''    LB.build_cache([_subset(toks[2:])], root, 1, older=True)
    ok["marker: a second build''', '''    with contextlib.redirect_stdout(_Capture()):
        LB.build_cache([_subset(toks[2:])], root, 1, older=True)
    ok["marker: a second build''')
s = s.replace('''    LB.build_cache([_subset(toks[:1])], kroot, 1)
''', '''    with contextlib.redirect_stdout(_Capture()):
        LB.build_cache([_subset(toks[:1])], kroot, 1)
''')
open(p, "w").write(s)
print(n0, len(s))
