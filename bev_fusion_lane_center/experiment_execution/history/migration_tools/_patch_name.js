const fs=require("fs");let s=fs.readFileSync("mig_common.py","utf8");
const a=s.indexOf("def folder_name(row):");const b=s.indexOf("def load_table");
const neu=`def folder_name(row):
    """E<first id, three digits><letter if any>_<slug of the description>; the other ids of a row are listed in its README"""
    m = re.search(r"E(\d+)([a-z]?)", row["id"])
    tag = f"E{int(m.group(1)):03d}{m.group(2)}" if m else (re.sub(r"[^A-Za-z0-9]+", "", row["id"]) or "X")
    return f"{tag}_{slug(row['what'])}"


`;
s=s.slice(0,a)+neu+s.slice(b);fs.writeFileSync("mig_common.py",s);
