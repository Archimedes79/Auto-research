p = "evaluation/bev_gtcheck.py"
s = open(p, encoding="utf-8").read()
s = s.replace("IoU(bin)", "IoU(>0.5)")
s = s.replace('''def load(path):''', '''def resolve_sig(sig=None):
    """the GT signature to check: the one named (it must have cached frames), else the current v5 `15efbde2d1` when it is cached, else the one with the most frames"""
    have = signatures()
    if sig:
        assert sig in have, f"no cached frames with signature {sig} in {cache_dir()}; cached: {dict(sorted(have.items(), key=lambda x: -x[1]))}"
        return sig
    return "15efbde2d1" if "15efbde2d1" in have else max(have, key=have.get)


def load(path):''', 1)
s = s.replace('''    ap.add_argument("--sig", default="15efbde2d1")''', '''    ap.add_argument("--sig", default=None, help="GT signature (default: 15efbde2d1 = v5, else the most frequent one in the cache)")''')
s = s.replace('''    a = ap.parse_args()
    if a.list:''', '''    a = ap.parse_args()
    a.sig = resolve_sig(a.sig) if not (a.list or a.versions) else a.sig
    if a.list:''')
open(p, "w", encoding="utf-8").write(s)
p = "evaluation/bev_response.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    ap.add_argument("--sig", default="15efbde2d1")''', '''    ap.add_argument("--sig", default=None, help="GT signature (default: 15efbde2d1 = v5, else the most frequent one in the cache)")''')
s = s.replace('''    a = ap.parse_args()
    samples, rows = all_rows(a.n, a.sig, a.seed)''', '''    a = ap.parse_args()
    a.sig = G.resolve_sig(a.sig)
    samples, rows = all_rows(a.n, a.sig, a.seed)''')
open(p, "w", encoding="utf-8").write(s)
