"""one-off: a result package for every FINISHED run of the project in AR_PROJECT that still has its run folder (existing packages are kept)"""
import os
import sys

sys.path.insert(0, "../autoresearch/tools")
import package as K  # noqa: E402
import provenance as V  # noqa: E402

made, skipped = 0, []
for rid in K.run_ids():
    r = K.run_record(rid)
    rd = V.find_run_dir(rid)
    if r.get("status") not in K.FINAL_STATES or r.get("exit") != 0:
        skipped.append((rid, f"not final ({r.get('status')}, {r.get('exit')})"))
    elif not rd:
        skipped.append((rid, "no run folder"))
    elif os.path.isfile(K.package_path(rid)):
        skipped.append((rid, "has a package"))
    else:
        pk = K.write(rid, rd)
        made += 1
        print(f"{rid:30} tier {pk['tier']:7} GT {pk['gt']['version']:8} bundle {pk['bundle']['code_hash']} ({pk['bundle']['files']} files) ckpt {pk['model']['ckpt']} derived {len(pk['deep']['files'])} files")
print(f"made {made}; skipped {len(skipped)}:")
for rid, why in skipped:
    print("  ", rid, why)
