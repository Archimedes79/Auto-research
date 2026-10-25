const fs=require("fs");let s=fs.readFileSync("mig_plan.py","utf8");
s=s.replace('    elif any(nm.startswith(t) or t.startswith(nm.split(".")[0]) for t in tok):\n        keep, rsn = True, "named by live code or state files"',
`    elif matched(nm):
        keep, rsn = True, "named by live code or state files"`);
s=s.replace('# ---------------------------------------------------------------- out/loop2 entries\nruns = {}',
`# ---------------------------------------------------------------- out/loop2 entries
def matched(nm):
    base = nm.rsplit(".", 1)[0] if nm.endswith((".done", ".log", ".out")) else nm
    for t in tok:
        if base == t or nm.startswith(t + "_") or nm.startswith(t + "."):
            return True
        if t.endswith("_") and nm.startswith(t):
            return True
    return False


def matched_by(nm):
    base = nm.rsplit(".", 1)[0] if nm.endswith((".done", ".log", ".out")) else nm
    return [t for t in tok if base == t or nm.startswith(t + "_") or nm.startswith(t + ".") or (t.endswith("_") and nm.startswith(t))][:3]


runs = {}`);
s=s.replace('big_kept = sorted(','big_kept0 = sorted(');
s+=`
big_kept = sorted(((r["mb"], nm) for nm, r in runs.items() if r["keep"] and r["why"].startswith("named")), reverse=True)[:30]
print("kept by token (largest):", [(nm, round(mb), matched_by(nm)) for mb, nm in big_kept])
`;
fs.writeFileSync("mig_plan.py",s);
