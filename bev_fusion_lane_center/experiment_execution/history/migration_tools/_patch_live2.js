const fs=require("fs");let s=fs.readFileSync("mig_live2.py","utf8");
s=s.replace('SC = f"{R}/scripts"','SC = f"{R}/scripts"\nINV = {x["name"]: x["mtime"] for x in json.load(open(f"{RS}/inventory.json"))["scripts"]}  # mtimes BEFORE the path rewrite touched every file')
s=s.replace('roots = [m for m in mods if ROOT_PAT.match(m) or os.path.getmtime(f"{SC}/{m}.py") >= RECENT]','roots = [m for m in mods if ROOT_PAT.match(m) or INV.get(m + ".py", 0) >= RECENT]')
fs.writeFileSync("mig_live2.py",s);
