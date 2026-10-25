const fs=require("fs");let s=fs.readFileSync("build_literature.py","utf8");
s=s.replace('for r in tab["register"]:\n    notes = []','REG = tab["register"]\nREG = [dict(k=k, **v) for k, v in REG.items()] if isinstance(REG, dict) else REG\nfor r in REG:\n    notes = []');
s=s.replace('{len(tab[\'register\'])} techniques','{len(tab[\'register\'])} techniques');
s=s.replace('json.dump(tab["register"], open(f"{L}/register.json"','json.dump(REG, open(f"{L}/register.json"');
fs.writeFileSync("build_literature.py",s);
