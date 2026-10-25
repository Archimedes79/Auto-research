const fs=require("fs");let s=fs.readFileSync("mig_plan.py","utf8");
s=s.replace('print("largest kept:", [(nm, round(mb)) for mb, nm in big_kept0])\n','');
s=s.replace('print("largest kept:", [(nm, round(mb)) for mb, nm in big_kept])\n','');
fs.writeFileSync("mig_plan.py",s);
