
# ------------------------------------------------------------------------------------------------ F
print("F: window images on nuScenes mini")
MINI = os.path.expanduser("~/data/nuscenes")
HAVE_MINI = os.path.isdir(os.path.join(MINI, "v1.0-mini"))
SWEEPS = "sweeps"                                                    # written out here, not LB.OLDER: a changed folder name must fail the test


def data_checks():
    """the window images on the mini set: builder, marker, dataset item, collate; name -> bool (the temporary folder is removed by the caller)"""
    ok = {}
    root = tempfile.mkdtemp(prefix="imgs_", dir=tmp)
    LB._DS = mini
    LB._fill((root, toks))
    LB._fill_older((root, toks))
    want = {sd["token"] for t in toks for sd in walk(t, 2, 2)}
    got = {f[:-4] for d, _, fs in os.walk(os.path.join(root, SWEEPS)) for f in fs if f.endswith(".jpg")}
    ok["builder: _fill_older writes one JPEG per older frame (-2 and -4 sweeps of each sample) under <root>/sweeps, named by the frame's sample_data token (independent walk along prev)"] = got == want and len(want) == 6
    keys = {mini.gt.cam(t)["token"] for t in toks}
    ok["builder: the key frames are in <root>, not in sweeps, and the two sets are disjoint (the next key frame's oldest frame is at -6: no gap, no overlap)"] = \
        all(os.path.exists(LB.image_path(root, k)) for k in keys) and not (keys & got) and not any(os.path.exists(LB.image_path(os.path.join(root, SWEEPS), k)) for k in keys)
    errs = [float((decode(os.path.join(root, SWEEPS), sd["token"]).float() - torch.tensor(mini._image_uint8(sd)).permute(2, 0, 1).float()).abs().mean()) for t in toks for sd in walk(t, 2, 2)]
    ok[f"builder: every older JPEG is the dataset's own image of that frame (resize + crop 256 x 704) within JPEG error (max {max(errs):.2f} of 255)"] = max(errs) < 3.0
    before = {p: os.path.getmtime(p) for p in (LB.image_path(os.path.join(root, SWEEPS), sd["token"]) for t in toks for sd in walk(t, 2, 2))}
    LB._fill_older((root, toks))
    ok["builder: resumable (a second run rewrites nothing, no temporary file is left)"] = all(os.path.getmtime(p) == t_ for p, t_ in before.items()) and not any(".tmp" in f for _, _, fs in os.walk(root) for f in fs)
    # the dataset item
    cds = FC.CachedFrontRadarDataset(MINI, "v1.0-mini", store=StubStore(), n_cam=3, cam_stride=2, with_gt=False, lane_images=root, lane_images_older=True, nusc=mini.nusc)
    plain = FC.CachedFrontRadarDataset(MINI, "v1.0-mini", store=StubStore(), n_cam=3, cam_stride=2, with_gt=False, lane_images=root, nusc=mini.nusc)
    items = [cds[cds.samples.index(t)] for t in toks[:2]]
    ok["dataset item: lane_img = the JPEG of frames[0], lane_img_old = the JPEGs of frames[1:] in frame order (independent decode of the files the walk names), uint8 (F - 1, 3, 256, 704)"] = \
        all(torch.equal(it["lane_img"], decode(root, it["frames"][0]["token"])) and it["lane_img_old"].dtype == torch.uint8 and tuple(it["lane_img_old"].shape) == (2, 3, 256, 704)
            and [f["token"] for f in it["frames"][1:]] == [sd["token"] for sd in walk(t, 2, 2)]
            and all(torch.equal(it["lane_img_old"][k], decode(os.path.join(root, SWEEPS), walk(t, 2, 2)[k]["token"])) for k in range(2)) for it, t in zip(items, toks))
    ok["dataset item: the three images of a window differ (key frame, -2, -4)"] = all(len({it["lane_img"].float().mean().item(), it["lane_img_old"][0].float().mean().item(), it["lane_img_old"][1].float().mean().item()}) == 3 for it in items)
    ok["dataset item: without lane_images_older no lane_img_old"] = "lane_img_old" not in plain[plain.samples.index(toks[0])]
    batch = FC.collate_cached(items)
    ok["collate_cached: lane_img_old is (B, F - 1, 3, H, W) uint8 in item order, lane_img (B, 3, H, W)"] = tuple(batch["lane_img_old"].shape) == (2, 2, 3, 256, 704) and batch["lane_img_old"].dtype == torch.uint8 \
        and all(torch.equal(batch["lane_img_old"][i], items[i]["lane_img_old"]) for i in range(2)) and tuple(batch["lane_img"].shape) == (2, 3, 256, 704)
    miss = tempfile.mkdtemp(prefix="imgs_missing_", dir=tmp)
    LB._fill((miss, toks[:1]))                                                      # the key frame, no sweeps
    d_miss = FC.CachedFrontRadarDataset(MINI, "v1.0-mini", store=StubStore(), n_cam=3, cam_stride=2, with_gt=False, lane_images=miss, lane_images_older=True, nusc=mini.nusc)
    ok["dataset: a missing older image is a FileNotFoundError that names build-cache-older"] = raises(FileNotFoundError, lambda: d_miss[d_miss.samples.index(toks[0])], "build-cache-older")
    ok["dataset: lane_images_older without lane_images, and with n_cam 1, is refused"] = \
        raises(ValueError, lambda: FC.CachedFrontRadarDataset(MINI, "v1.0-mini", store=StubStore(), lane_images_older=True, nusc=mini.nusc), "lane_images") \
        and raises(ValueError, lambda: FC.CachedFrontRadarDataset(MINI, "v1.0-mini", store=StubStore(), lane_images=root, lane_images_older=True, n_cam=1, nusc=mini.nusc), "no older frame")
    return ok


def marker_checks():
    """the finished-build marker (BUILT.json) of build_cache(older=True): scenes, merged by a second build; startable.older_cache_findings reads it"""
    ok = {}
    root, key = tempfile.mkdtemp(prefix="marker_", dir=tmp), LB.older_key(3, 2)
    s1, s2 = {scene_of(t) for t in toks[:2]}, {scene_of(toks[2])}
    LB.build_cache([_subset(toks[:2])], root, 1, older=True)
    bt = LB.older_built(root)
    ok["marker: build_cache(older=True) writes BUILT.json with the window and the scenes of the datasets, after every sample"] = set(bt) == {key} and bt[key]["scenes"] == sorted(s1) and "when" in bt[key]
    LB.build_cache([_subset(toks[2:])], root, 1, older=True)
    ok["marker: a second build of the same window is merged (the scenes of both)"] = LB.older_built(root)[key]["scenes"] == sorted(s1 | s2)
    ok["marker: older_missing_scenes names the scenes no finished build of this window covers (an unknown scene, another stride, all of them without a marker)"] = \
        LB.older_missing_scenes(root, 3, 2, s1 | s2) == [] and LB.older_missing_scenes(root, 3, 2, s1 | {"scene-9999"}) == ["scene-9999"] \
        and LB.older_missing_scenes(root, 3, 4, s1) == sorted(s1) and LB.older_missing_scenes(os.path.join(tmp, "nothing"), 3, 2, s1) == sorted(s1)
    kroot = tempfile.mkdtemp(prefix="marker_keys_", dir=tmp)
    LB.build_cache([_subset(toks[:1])], kroot, 1)
    ok["marker: the key-frame build writes no marker and no sweeps folder"] = LB.older_built(kroot) == {} and not os.path.exists(os.path.join(kroot, SWEEPS))
    cfg = T.TrainConfig(name="x", device="cpu", lane_images="x", n_cam=3, cam_stride=2, train_scenes=tuple(s1), val_scenes=tuple(s2))
    ok["startable: a finished build that covers the config's scenes is no problem; a scene it lacks, another stride, no sweeps folder and no scene list are named"] = \
        ST.older_cache_findings(root, cfg) == [] \
        and any("lacks 1 of the config's 3 scenes" in x and "scene-9999" in x for x in ST.older_cache_findings(root, dataclasses.replace(cfg, val_scenes=cfg.val_scenes + ("scene-9999",)))) \
        and any("no finished build" in x for x in ST.older_cache_findings(root, dataclasses.replace(cfg, cam_stride=4))) \
        and any("build-cache-older" in x for x in ST.older_cache_findings(kroot, cfg)) and any("names no scenes" in x for x in ST.older_cache_findings(root, dataclasses.replace(cfg, train_scenes=(), val_scenes=())))
    return ok


def _subset(ts):
    """the mini dataset restricted to the samples `ts` (a shallow copy)"""
    d = copy.copy(mini)
    d.samples = list(ts)
    return d


def cli_checks():
    """python training/lane_branch.py build-cache[-older] / check-cache[-older] end to end on one train and one val scene of mini, with the scene lists in the config"""
    ok = {}
    from nuscenes.utils.splits import create_splits_scenes
    sp = create_splits_scenes()
    cfg_path, root = os.path.join(tmp, "cli_config.json"), tempfile.mkdtemp(prefix="cli_", dir=tmp)
    json.dump({"_about": "t26 cli test", "name": "x", "data_root": MINI, "version": "v1.0-mini", "n_cam": 3, "cam_stride": 2, "lane_images": root, "gt_cache": "", "train_scenes": [sp["mini_train"][0]],
               "val_scenes": [sp["mini_val"][0]], "device": "cpu"}, open(cfg_path, "w"))
    say = {}
    for cmd in ("build-cache", "build-cache-older", "check-cache", "check-cache-older"):
        with contextlib.redirect_stdout(_Capture()) as cap:
            LB._cli([cmd, cfg_path, "--n", "3"])
        say[cmd] = cap.text
    n_key = len(os.listdir(root)) - 1 if os.path.isdir(os.path.join(root, SWEEPS)) else len(os.listdir(root))
    sweeps = [f for d, _, fs in os.walk(os.path.join(root, SWEEPS)) for f in fs if f.endswith(".jpg")]
    keys = [f for d, _, fs in os.walk(root) for f in fs if f.endswith(".jpg") and SWEEPS not in d]
    ok[f"cli: build-cache-older fills sweeps/ with two JPEGs per sample ({len(sweeps)} older images for {len(keys)} key frames) and BUILT.json covers both scenes of the config"] = \
        len(sweeps) == 2 * len(keys) and set(LB.older_built(root)[LB.older_key(3, 2)]["scenes"]) == {sp["mini_train"][0], sp["mini_val"][0]} and "older frames of" in say["build-cache-older"]
    ok["cli: check-cache-older compares the val frames with the dataset's own image and the marker with the config's scenes; check-cache the key frames"] = \
        "older-frame cache vs dataset image" in say["check-cache-older"] and "BUILT.json covers all 2 scenes" in say["check-cache-older"] and "cache vs dataset image" in say["check-cache"] and "older" not in say["check-cache"]
    bt = LB.older_built(root)
    bt[LB.older_key(3, 2)]["scenes"] = bt[LB.older_key(3, 2)]["scenes"][:1]
    json.dump({"built": bt}, open(os.path.join(root, SWEEPS, LB.BUILT), "w"))
    ok["cli: check-cache-older fails when the marker lacks a scene of the config"] = raises(AssertionError, lambda: LB._cli(["check-cache-older", cfg_path, "--n", "2"]), "no finished build")
    split_cfg = os.path.join(tmp, "split_config.json")
    json.dump({"_about": "x", "name": "x", "split": "small", "lane_images": root, "device": "cpu"}, open(split_cfg, "w"))
    c = LB.load_config(split_cfg)
    ok["cli: a config with a \"split\" is resolved to its scene lists (small: 60 train, 10 val), the comment key is ignored"] = len(c.train_scenes) == 60 and len(c.val_scenes) == 10
    return ok


class _Capture:
    """a stdout that keeps what was printed"""
    def __init__(self):
        self.text = ""

    def write(self, s):
        self.text += s

    def flush(self):
        pass


if not HAVE_MINI:
    print("  SKIPPED: no nuScenes mini at ~/data/nuscenes (the window-image checks need it)")
else:
    from nuscenes_dataset import FrontRadarDataset  # noqa: E402
    from torchvision.io import decode_jpeg, read_file  # noqa: E402
    mini = FrontRadarDataset(MINI, "v1.0-mini", n_cam=3, cam_stride=2, with_gt=False)
    toks = [mini.samples[10], mini.samples[200], mini.samples[300]]
    scene_of = lambda t: mini.nusc.get("scene", mini.nusc.get("sample", t)["scene_token"])["name"]  # noqa: E731

    def walk(sample_token, n, stride):
        """independent of the builder: the sample_data of the key frame, then `stride` steps along `prev` per older frame (clamped at the start of the scene)"""
        sd = mini.gt.cam(sample_token)
        out = []
        for _ in range(n):
            for _ in range(stride):
                sd = mini.nusc.get("sample_data", sd["prev"]) if sd["prev"] else sd
            out.append(sd)
        return out

    def decode(folder, tok):
        return decode_jpeg(read_file(LB.image_path(folder, tok)))

    class StubStore:
        def feature_hw(self):
            return (16, 44)

        def get(self, tok):
            return torch.zeros(118, 16, 44, dtype=torch.float16), torch.zeros(80, 16, 44, dtype=torch.float16)

    for group in (data_checks, marker_checks, cli_checks):
        for name, v in group().items():
            check(v, name)
