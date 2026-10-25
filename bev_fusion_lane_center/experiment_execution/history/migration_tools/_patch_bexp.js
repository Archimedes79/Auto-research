const fs=require("fs");let s=fs.readFileSync("build_experiments.py","utf8");
s=s.replace(`    if os.path.isfile(src):
        out.append((src, label))
        return out`,`    if os.path.isfile(src):
        if os.path.splitext(src)[1].lower() in OKEXT and 0 < os.path.getsize(src) <= MAXF:
            out.append((src, label))
        return out`);
s=s.replace(`                if sz <= MAXF:
                    out.append`,`                if 0 < sz <= MAXF:
                    out.append`);
s=s.replace(`        os.makedirs(d, exist_ok=True)
        open(f"{d}/README.md", "w", encoding="utf-8").write(text)`,`        os.makedirs(d, exist_ok=True)
        for sub in ("results", "figures"):
            shutil.rmtree(f"{d}/{sub}", ignore_errors=True)
        open(f"{d}/README.md", "w", encoding="utf-8").write(text)`);
s=s.replace("Folder scheme: `E<ids>_<slug>`; ids zero-padded to three digits; one folder per catalogue row (rows with several ids, e.g. E94b / E96 / E98, share a folder).","Folder scheme: `E<first id, three digits><letter>_<slug>`, one folder per catalogue row; rows that cover several ids (e.g. E94b / E96 / E98) carry all of them in the README table and sit under the first.");
s=s.replace("key=lambda t: (ids_of_row(t[1][\"id\"])[:1] or [0], t[0])","key=lambda t: (ids_of_row(t[1][\"id\"])[:1] or [999], t[0])");
fs.writeFileSync("build_experiments.py",s);
