p = "autogt/view.py"
s = open(p, encoding="utf-8").read()
s = s.replace("4 total apparent flow (static world from the ego motion + objects) in m/s: hue = direction (wheel in the corner), brightness =",
              "4 motion RELATIVE TO THE STATIC WORLD in m/s (the ego motion removed: the static world sits at the centre of the colour wheel,\n  black; only moving things have colour): hue = direction (wheel in the corner), brightness =")
s = s.replace("def figure(gt_gen, name, k, out_dir):", "def figure(gt_gen, name, k, out_dir, total=False):")
s = s.replace('''    fl = np.where((gt["flow_weight"] > 0)[None], gt["flow"], 0)
    show(ax[3], flow_rgb(fl), "total apparent flow = static world + objects [m/s]")''',
'''    # default: relative to the static world (flow - flow_ego = the objects' own world velocity); total=True: the apparent flow
    fl = np.where((gt["flow_weight"] > 0)[None], gt["flow"] if total else gt["velocity"], 0)
    show(ax[3], flow_rgb(fl), "total apparent flow = static world + objects [m/s]" if total
         else "motion relative to the static world [m/s] (static world = black = wheel centre)")''')
s = s.replace('def run(out_dir, picks=None, dataroot="~/data/nuscenes", version="v1.0-mini"):',
              'def run(out_dir, picks=None, dataroot="~/data/nuscenes", version="v1.0-mini", total=False):')
s = s.replace("return [figure(gen, n, int(k), out_dir) for n, k in (picks or DEFAULT)]", "return [figure(gen, n, int(k), out_dir, total) for n, k in (picks or DEFAULT)]")
s = s.replace('"flow: hue = direction, brightness = speed (to 12 m/s)"]', '"motion relative to the static world: static world black (wheel centre), hue = direction, brightness = speed (to 12 m/s)"]')
open(p, "w", encoding="utf-8").write(s)
p = "autogt/cli.py"
s = open(p, encoding="utf-8").read()
s = s.replace('    v.add_argument("picks", nargs="*")', '    v.add_argument("picks", nargs="*")\n    v.add_argument("--total", action="store_true", help="4th panel: total apparent flow (ego motion included) instead of motion relative to the static world")')
s = s.replace("run(a.out, [x.split(\":\") for x in a.picks] or None)", "run(a.out, [x.split(\":\") for x in a.picks] or None, total=a.total)")
open(p, "w", encoding="utf-8").write(s)
print("ok")
