import json, os, sys
sys.path.insert(0, os.path.join(os.environ['BFLC_ROOT'], 'evaluation'))
import jtest_eval as J
d = json.load(open(os.path.join(os.environ['BFLC_ROOT'], 'temp', 'e343', '_session_scratch_', 'closure_probe.json')))
files = set(J.code_sha256())
loaded = set(d['loaded'])
print('digest', len(files), 'loaded', len(loaded))
print('LOADED NOT IN DIGEST', sorted(x for x in loaded if x not in files and not x.startswith('GtGenerator/')))
print('IN DIGEST NOT LOADED', sorted(files - loaded))
