"""GtGenerator: ground truth for BEV perception from nuScenes (front GT: static layers, objects, masks, velocity / flow; lane GT: the lane graph with tiers and
visibility).

THE INTERFACE. A problem (`problems/`), the research core (`autoresearch/`) and `datasets/` reach the ground truth through `import GtGenerator` and the names
below (they are `GtGenerator.__all__`), across a process boundary through `python -m GtGenerator COMMAND` (each command is one method of the class, `--help` has
the map), and through nothing below them: no module or file of this folder, no folder or file name below the store root, no file format, no version table. A
name that is missing is added to the table here, not imported from below; `check_imports` is the test of the rule (interface_selftest.py). What comes back is
plain data: numpy arrays, dicts, lists, str. On the class and on a key only the members named here are the interface (`_MEMBERS`). Guide for agents: AGENTS.md;
reference: README.md.

where the GT is and which GT it is (cheap: no devkit, no folder listing)
    store_root()                               str: <data root>/groundtruth, resolved at every call (RESEARCH_DATA, else `data` of the machine file)
    current_signature(**options)               str, 10 hex: the STORE key of the front GT of this source (after `same_gt_as`): what a cache name or a record
                                               carries
    lane_signature(rig="lane_center")          str, 10 hex: the STORE key of the lane GT of a rig ("lane_center" | "bev_fusion"), likewise
    current_version(**options)                 str: the front GT version of this source, "v13" ("unreleased" without a row)
    version_of_key(key)                        str or None: the front GT version of a cache key that ends in a signature ("0.5_4_a107e0b4e8" -> "v13")
    version_names()                            list[str]: the front GT versions, oldest first ("v1" ... "v16"): what a split or a card may name as its GT
    gt_key(system, dataset="v1.0-trainval", set_name="full", out=None, dataroot=None, scenes=None, deep=True)
                                               GtKey of what is stored for "bev_fusion" | "lane_center"; nothing is generated
    GtKey                                      frozen dataclass: system, dataset, set_name, id (12 hex: the same GT has the same id), data, release, out (the
                                               store root), parts
                                               (name, signature = code, store, version, commit, dirty, complete, path) and part(name), complete, command (the
                                               shell line that makes what is missing), load(token), lane(token), mask(token, kind, border_m), as_dict()
read stored GT (no nuScenes needed; `data` = another store root)
    load(token, data=None, **options)          dict: 24 float32 layers [ix, iy] of shape (120, 104) (vectors (2, 120, 104)) + "instance" int32 + "meta" dict
                                               (AGENTS.md section 3)
    missing(tokens, data=None, **options)      list[str]: the tokens without a stored front keyframe ([] = the store covers them)
    load_lane(token, rig="lane_center", data=None)     dict: the lane graph of a keyframe (AGENTS.md section 3b)
    lane_tokens(rig="lane_center", data=None)  set[str]: the sample tokens with a stored lane graph
    scene_tokens(scene, version="v1.0-mini", dataroot=None)     list[str]: the sample tokens of a sequence in time order (json tables only)
make GT
    GtGenerator(dataroot, version, *, grid=None, ss=4, light=False, scenes=None, nusc=None)
                                               the generator object, for a loader (everything after `version` by keyword; `nusc`: an open devkit):
                                               keyframe(token) -> dict as `load`; training_sample(token, grid, ss) -> {"prev", "cur", "next"} of such dicts, or
                                               None; keyframes(scene), scenes() -> list[str]; cam(token)
                                               -> the CAM_FRONT sample_data record; camera_track(token) -> list of them; ego_steps(t0, t1, track) -> (dt, v,
                                               omega) per camera frame; nusc (the devkit); signature() = the code, store_signature() = the store (cache names,
                                               records). As static methods of the class, the same as the names of this table: generate, generate_bev_gt,
                                               generate_lane_gt, gt_key; and have(set_name=None, scenes=None, ...), check_data(dataroot, ...) -> dict {ok, text,
                                               ...}. The other static methods are the command line's, not a problem's
    generate(dataroot, out=None, ...)          dict: the front GT, the lane GT of both systems and the masks of a nuScenes folder, written below the store root;
                                               resumable
    generate_bev_gt(dataroot, set_name=None, scenes=None, ...)     GtKey: BEV Fusion (front GT + lane GT of the rig bev_fusion + area masks); resumable
    generate_lane_gt(dataroot, rig="all", ...) GtKey (a dict rig -> GtKey for "all"): the lane GT of one system or both; resumable
raw nuScenes as the GT reads it
    light_nuscenes(dataroot, version, scenes=None, cameras=("CAM_FRONT",), radars=True)
                                               a devkit-like object (get, the tables) holding only these sequences and sensors
    pose_matrix(record)                        (4, 4) float64: the transform of a nuScenes ego_pose / calibrated_sensor record
definitions the network and the tests share with the GT (numpy only)
    GridSpec                                   frozen dataclass x_min, x_max, y_min, y_max, res (default the front grid, 0.5 m): nx, ny, centres() -> (gx, gy),
                                               both (nx, ny)
    AREAS, LINES, OBJECTS                      tuple[str]: the names of the map-area, line and object layers of a front keyframe
    DIST_MAX                                   float: the truncation of the line distance layers in metres
    cell_centres(grid)                         (gx, gy): the cell-centre coordinates of a grid, both float (nx, ny)
    occlusion(gx, gy, cam_xy, footprints)      bool array of gx's shape: cells whose line of sight from the camera crosses a (4, 2) box footprint
    radar_cells(gx, gy, poses)                 bool array of gx's shape: cells inside the fan of one of the radars (poses: 4 x 4 sensor -> ego)
    coverage_for(g, kind)                      bool (nx, ny): where an output type ("map", "objects", "velocity", "camera_aux", "radar_aux") is trained and
                                               scored in the keyframe g
    shares_in(g, corners, height=None, group=None)     dict {"camera", "radar", "fusion"} of float 0 .. 1: the observed shares of a box footprint in the
                                                       keyframe g
    mask_for(project)                          dict of bool arrays "camera", "radar", "fusion": the static coverage of "bev_fusion" (120, 104) | "lane_center"
                                               (200, 200), no boxes
pictures
    lane_camera_image(gt, sample, graph, channel, T_e2g_front, scale=0.5)     (H, W, 3) uint8 BGR: the camera image of `channel` with the lane graph projected
    on the ground
the rule
    check_imports(paths)                       list[str] "file:line: the source line  (why)": the uses below this interface in the .py files of folders / files;
                                               [] = none

    import GtGenerator
    key = GtGenerator.gt_key("bev_fusion")                             # the check before a training: key.complete, else key.command makes it
    g = GtGenerator.load(GtGenerator.scene_tokens("scene-0561", "v1.0-trainval")[8])
    graph = GtGenerator.load_lane(token)                               # Spurmitte; rig="bev_fusion": the graph in BEV Fusion's frame and window
    gen = GtGenerator.GtGenerator("~/data/nuscenes", "v1.0-mini")      # in a loader: gen.training_sample(token), cache files named by gen.store_signature()
"""
# Nothing below is imported at `import GtGenerator` (PEP 562): a public name is imported from its owner when it is first asked for, so the package imports
# without numpy and a reader of the lane store never loads the front generator. public name -> (module of this folder, the name there); `__all__` is this table.
_PUBLIC = {
    "store_root": ("where", "root"), "current_signature": ("suite", "current_signature"), "lane_signature": ("gt_api", "lane_signature"),
    "current_version": ("suite", "current_version"), "version_of_key": ("versions", "version_of_key"), "version_names": ("versions", "version_names"),
    "gt_key": ("gt_api", "gt_key"), "GtKey": ("gt_api", "GtKey"),
    "load": ("suite", "load"), "missing": ("suite", "missing"), "load_lane": ("gt_api", "load_lane"), "lane_tokens": ("gt_api", "lane_tokens"), "scene_tokens": ("suite", "scene_tokens"),
    "GtGenerator": ("suite", "GtGenerator"), "generate": ("api", "generate"), "generate_bev_gt": ("gt_api", "generate_bev_gt"), "generate_lane_gt": ("gt_api", "generate_lane_gt"),
    "light_nuscenes": ("light_nusc", "light_nuscenes"), "pose_matrix": ("nuscenes_adapter", "pose_matrix"),
    "GridSpec": ("grid", "GridSpec"), "AREAS": ("layers", "AREAS"), "LINES": ("layers", "LINES"), "OBJECTS": ("layers", "OBJECTS"), "DIST_MAX": ("layers", "DIST_MAX"),
    "cell_centres": ("layers", "cell_centres"), "occlusion": ("layers", "occlusion"), "radar_cells": ("coverage", "radar_cells"), "coverage_for": ("coverage", "coverage_for"),
    "shares_in": ("coverage", "shares_in"), "mask_for": ("rig_masks", "mask_for"),
    "lane_camera_image": ("lane_graph_view", "camera_image"),
    "check_imports": ("interface_selftest", "check_imports"),
}
__all__ = sorted(_PUBLIC)
# What a problem may take FROM a public class (`GtGenerator.GtGenerator.<member>`, `GtGenerator.GtKey.<member>`): the members the contract lines above name. The
# class `GtGenerator` also carries one static method per command of the command line (`versions(update=True)` writes versions.csv, `clean(delete=True)` deletes
# reports): those are reached through `python -m GtGenerator`, not from a problem's code. check_imports enforces the list where the class is named; it cannot
# follow an object (`gen.clean`).
_MEMBERS = {
    "GtGenerator": ("keyframe", "training_sample", "keyframes", "scenes", "cam", "camera_track", "ego_steps", "signature", "store_signature",
                    "generate", "generate_bev_gt", "generate_lane_gt", "gt_key", "have", "check_data"),
    "GtKey": ("system", "dataset", "set_name", "id", "data", "release", "out", "parts", "part", "complete", "command", "load", "lane", "mask", "as_dict"),
}


def __getattr__(name):
    """a public name, imported from its owner on first use and kept as a plain attribute. A module of this folder answers too (`GtGenerator.lane_gt`): that is
    for the code and the tools of this folder, the interface is `__all__`"""
    import importlib
    import os
    if name in _PUBLIC:
        module, attribute = _PUBLIC[name]
        value = getattr(importlib.import_module(f".{module}", __name__), attribute)
        globals()[name] = value
        return value
    if name.isidentifier() and os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), name + ".py")):
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'GtGenerator' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_PUBLIC))
