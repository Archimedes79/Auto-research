const fs=require("fs");let s=fs.readFileSync("build_evaluation.py","utf8");
const a=s.indexOf("def vert100_dp");const b=s.indexOf("'''", a);
const neu=`def vert100_dp(preds, eps=0.3, ds=0.5, clip=48.0, unique_ends=False):
    """vertices per 100 m of polyline length when every piece is simplified with Douglas-Peucker (tolerance eps m, samples every ds m, |x|,|y| < clip, pieces with < 3 samples skipped).
    Default = the definition of scripts/oracle_vertices.py that gave the oracle value 14.3 per 100 m on the ground truth: kept vertices of every piece, end points of neighbouring pieces counted per piece.
    unique_ends=True counts coinciding piece end points once (closer to vert100 of vertex graphs). For graphs WITHOUT vertex ids (rule graphs from dense heads), where vert100 would count every polyline point."""
    nv, tot = 0, 0.0
    for g in preds:
        ends = set()
        for p in g["pts"]:
            q = GE.resample(np.asarray(p, np.float64), ds)
            if len(q) < 3:
                continue
            q = q[(np.abs(q) < clip).all(1)]
            if len(q) < 3:
                continue
            tot += float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
            k = _dp_keep(q, eps)
            if unique_ends:
                nv += len(k) - 2
                ends.add((round(float(q[0, 0]), 1), round(float(q[0, 1]), 1)))
                ends.add((round(float(q[-1, 0]), 1), round(float(q[-1, 1]), 1)))
            else:
                nv += len(k)
        nv += len(ends)
    return 100.0 * nv / max(tot, 1e-9)
`;
s=s.slice(0,a)+neu+s.slice(b);
fs.writeFileSync("build_evaluation.py",s);
