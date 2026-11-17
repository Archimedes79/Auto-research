import os
R = os.environ["BFLC_ROOT"]
p = os.path.join(R, "training/train/jmap_train.py")
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


sub('''[--no-map] [--init-seed I] [--smoke]''', '''[--no-map] [--map-mode train|other] [--init-seed I] [--smoke]''')
sub('''Arm z runs the SAME trainer and loader with `--no-map`:''', '''`--map-mode other` is the CAPACITY CONTROL of the card (arm c): the same block and loop, but every draw of the map is made on a RANDOM frame of ANOTHER scene (a fresh random choice at every read, not a fixed frame-to-map plan, which could be memorised; `lane_map_data` mode `other`): same parameters
and compute, no information about this frame; it needs the block (not with `--no-map`). Arm z runs the SAME trainer and loader with `--no-map`:''')
sub('''RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "steps", "lr", "no_map")''', '''RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "steps", "lr", "no_map", "map_mode")''')
sub('''    ap.add_argument("--smoke", action="store_true", help="20 steps on synthetic batches")''', '''    ap.add_argument("--map-mode", default="train", choices=("train", "other"), dest="map_mode", help="train = the card's mix of maps of THIS frame; other = the capacity control c: the same mix drawn on a random frame of ANOTHER scene at every read (no information)")
    ap.add_argument("--smoke", action="store_true", help="20 steps on synthetic batches")''')
sub('''    assert args["no_map"] or args["bs"] >= 2,''', '''    assert not args["no_map"] or args["map_mode"] == "train", "--map-mode other needs the block: arm z (--no-map) builds no map"
    assert args["no_map"] or args["bs"] >= 2,''')
sub('''def synthetic_batch(bs, K, seed, no_map=False):
    """the smoke batch: `jtoken.synthetic_batch` (arm z: just that, as `jtoken_train.py --smoke` makes it) + synthetic maps (arm m)"""
    b = JK.synthetic_batch(bs, K, seed=seed)
    return b if no_map else dict(b, maps=synthetic_maps(b, seed))''', '''def synthetic_batch(bs, K, seed, no_map=False, other=False):
    """the smoke batch: `jtoken.synthetic_batch` (arm z: just that, as `jtoken_train.py --smoke` makes it) + synthetic maps (arm m); other (the capacity control): every frame gets the map of ANOTHER frame of the batch (rolled by one)"""
    b = JK.synthetic_batch(bs, K, seed=seed)
    if no_map:
        return b
    maps = synthetic_maps(b, seed)
    return dict(b, maps={k: v.roll(1, 0) for k, v in maps.items()} if other else maps)''')
sub('''init * 100003 + step, a.no_map)''', '''init * 100003 + step, a.no_map, a.map_mode == "other")''')
sub('''init_seed=init, maps=not a.no_map, ident=a.ident''', '''init_seed=init, maps=not a.no_map, mode=a.map_mode, ident=a.ident''')
sub('''f"init seed {init}, " + (f"synthetic batches''', '''f"init seed {init}, map mode {a.map_mode}, " + (f"synthetic batches''')
open(p, "w", encoding="utf-8").write(s)
print("patched train")
