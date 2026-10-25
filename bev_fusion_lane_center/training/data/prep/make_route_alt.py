"""Schleife 5 / E59 (Erweiterung): ALTERNATIVE Navigationsziele an Verzweigungen. Je Sample, dessen gefahrene Route eine Verzweigung (Stueck mit >= 2 Nachfolgern) durchquert, wird die Route so umgebaut, dass sie am ersten solchen Stueck in den ANDEREN Zweig laeuft (der Zweig mit dem groessten Richtungsunterschied zum gefahrenen), danach folgt sie dem Nachfolger mit der kleinsten Richtungsaenderung bis 40 m hinter der Verzweigung.
Ausgabe: ~/exp_gcar_b123/route_alt_val.pkl (RT-Eintraege der alternativen Route; None fuer Samples ohne Alternative) und route_drv_sub_val.pkl (gefahrene Route NUR derselben Samples; Vergleich).   Aufruf (gcar): make_route_alt.py"""
import os
import pickle

import numpy as np

HOME = os.path.expanduser("~")
D = f"{HOME}/exp_gcar_b123"


def tan_start(p):
    p = np.asarray(p, np.float64)
    q = p[min(3, len(p) - 1)] - p[0]
    return q / (np.linalg.norm(q) + 1e-9)


def tan_end(p):
    p = np.asarray(p, np.float64)
    q = p[-1] - p[max(len(p) - 4, 0)]
    return q / (np.linalg.norm(q) + 1e-9)


def length(p):
    p = np.asarray(p, np.float64)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def main():
    GT = pickle.load(open(f"{D}/graph_gt_val.pkl", "rb"))
    RT = pickle.load(open(f"{D}/route_gt_val.pkl", "rb"))
    alt, drv = {}, {}
    n_alt = 0
    for i in RT if isinstance(RT, dict) else range(len(RT)):
        rt, gt = RT[i], GT[i]
        alt[i], drv[i] = None, None
        if rt is None or gt is None:
            continue
        r = list(rt["route"])
        succ = {}
        for a, b in gt["edges"]:
            succ.setdefault(a, []).append(b)
        done = False
        for j in range(len(r) - 1):
            cand = [s for s in succ.get(r[j], []) if s != r[j + 1] and len(gt["pts"][s]) >= 2]
            if len(succ.get(r[j], [])) >= 2 and cand and len(gt["pts"][r[j + 1]]) >= 2:
                t_drv = tan_start(gt["pts"][r[j + 1]])
                s = min(cand, key=lambda x: float(tan_start(gt["pts"][x]) @ t_drv))  # groesster Richtungsunterschied
                route = r[: j + 1] + [s]
                cur, L = s, length(gt["pts"][s])
                seen = set(route)
                while L < 40.0:
                    nxt = [x for x in succ.get(cur, []) if x not in seen and len(gt["pts"][x]) >= 2]
                    if not nxt:
                        break
                    t0 = tan_end(gt["pts"][cur])
                    cur = max(nxt, key=lambda x: float(tan_start(gt["pts"][x]) @ t0))
                    route.append(cur)
                    seen.add(cur)
                    L += length(gt["pts"][cur])
                e = dict(rt)
                e["route"] = route
                alt[i] = e
                drv[i] = rt
                n_alt += 1
                done = True
                break
        if not done:
            continue
    pickle.dump(alt, open(f"{D}/route_alt_val.pkl", "wb"))
    pickle.dump(drv, open(f"{D}/route_drv_sub_val.pkl", "wb"))
    print("Samples mit Verzweigung auf der gefahrenen Route:", n_alt, "von", sum(1 for v in RT.values() if v is not None) if isinstance(RT, dict) else len(RT))


if __name__ == "__main__":
    main()
