import os
P = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "jtest_eval.py")
s = open(P, encoding="utf-8").read()
pairs = [
    ('''"val_tokens.json": os.path.join(QE.LS.B123, "feats", "val_tokens.json")}''',
     '''"val_tokens.json": os.path.join(QE.LS.B123, "feats", "val_tokens.json"), "poses_val.npz": os.path.join(QE.LS.B123, "poses_val.npz")}'''),
    ('''"""the small files and library versions the numbers depend on, for the cards (on TEST compared with the rehearsal's card): the sha256 of data/splits.json, val_tokens.json and `runs/<run>/config.json` of every run of the system (B and c2 have no
    config.json: absent is a value too), the versions of torch, numpy and shapely (the big cache files are not hashed)"""''',
     '''"""the small files and library versions the numbers depend on, for the cards (on TEST compared with the rehearsal's card): the sha256 of data/splits.json, val_tokens.json, poses_val.npz (the scene column of the bootstrap) and `runs/<run>/config.json` of
    every run of the system (B and c2 have no config.json: absent is a value too), the versions of torch, numpy and shapely (the big cache files are not hashed)"""'''),
    ('''(sha256 of data/splits.json, val_tokens.json, the `config.json` of the runs;''', '''(sha256 of data/splits.json, val_tokens.json, poses_val.npz, the `config.json` of the runs;'''),
    ('''small files hashed (data/splits.json, val_tokens.json, run configs)."]''', '''small files hashed (data/splits.json, val_tokens.json, poses_val.npz, run configs)."]'''),
]
for old, new in pairs:
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)
open(P, "w", encoding="utf-8").write(s)
print("ok")
