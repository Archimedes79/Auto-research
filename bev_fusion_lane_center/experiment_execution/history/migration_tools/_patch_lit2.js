const fs=require("fs");let s=fs.readFileSync("build_literature.py","utf8");
s=s.replace('re.match(r"([ABC]\d+) - (.*)", head)','re.match(r"([A-Z]+\d+) - (.*)", head)');
s=s.replace('capture_output=True, text=True).stdout.strip()','capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.strip()');
fs.writeFileSync("build_literature.py",s);
