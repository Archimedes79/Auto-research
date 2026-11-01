p = "autogt/report.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    names = scenes or (scene_names(dataroot, version) if light else AutoGT(dataroot, version, **options).scenes())
    gen0 = AutoGT(dataroot, version, light=light, scenes=names[:1] if light else None, **options)
''', '''    if light:
        names = scenes or scene_names(dataroot, version)
        gen0 = AutoGT(dataroot, version, light=True, scenes=names[:1], **options)
    else:
        gen0 = AutoGT(dataroot, version, **options)
        names = scenes or gen0.scenes()
''')
s = s.replace('''        gen = AutoGT(dataroot, version, light=light, scenes=grp if light else None, **options) if (light or gen0 is None) else gen0
''', '''        gen = AutoGT(dataroot, version, light=True, scenes=grp, **options) if light else gen0
''')
s = s.replace('''    gen = AutoGT(dataroot, version, light=light, scenes=keep if light else None, **options)
''', '''    gen = AutoGT(dataroot, version, light=True, scenes=keep, **options) if light else gen0
''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
