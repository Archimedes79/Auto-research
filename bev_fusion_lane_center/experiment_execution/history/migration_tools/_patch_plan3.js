const fs=require("fs");let s=fs.readFileSync("mig_plan.py","utf8");
s=s.replace(`        if base == t or nm.startswith(t + "_") or nm.startswith(t + "."):
            return True
        if t.endswith("_") and nm.startswith(t):
            return True`,`        if base == t or nm.startswith(t + "."):
            return True
        if t.endswith("_") and nm.startswith(t):
            return True`);
s=s.replace(`[t for t in tok if base == t or nm.startswith(t + "_") or nm.startswith(t + ".") or (t.endswith("_") and nm.startswith(t))][:3]`,`[t for t in tok if base == t or nm.startswith(t + ".") or (t.endswith("_") and nm.startswith(t))][:3]`);
fs.writeFileSync("mig_plan.py",s);
