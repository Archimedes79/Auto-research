import os
p = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "README.md")
s = open(p, encoding="utf-8").read()
old = '''(`--max N` smoke runs to `temp/e343/<split>_max<N>/`). '''
new = '''(`--max N` smoke runs to `temp/e343/<split>_max<N>/`). A dump made by this module carries its provenance (run, device, checkpoint sha256, digest of the imported lane code) in `<name>_head.json` and is reused only under the same one (else refused with a message, nothing deleted); dumps named by a flag (E321's, the stored VAL dumps) are exempt. Cards and `results.md` are written once (refused when they exist), name the device and carry the sha256 of the imported lane code (`CODE DRIFT: <files>` when one changed during the run). `all` checks the seeds, the stack's pixel pyramid, the flagged dumps and the control card before it dumps anything; `--no-stack` drops arm k (its failed check) and w and p are then checked alone. The module docstring has the dress-rehearsal command lines for VAL. '''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
