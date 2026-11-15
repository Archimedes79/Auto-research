"""The jobs of GtGenerator for the job runner (autoresearch/jobrunner): `JOBS`, the table that makes the two generation commands calls a job file can name -

    {"id": "gt.bev.small", "call": "GtGenerator.generate_bev_gt", "args": {"dataroot": "~/data/nuscenes", "set_name": "small"}}
    python3 -m jobrunner JobRunner.check job.json      # looks only; JobRunner.submit starts it under run.py

The format is the one of autoresearch/jobrunner/README.md ("What is a job: the catalogue"). The data are arguments (the nuScenes folder, the set or the sequences, the rig); the job reads
`path:{dataroot}` and writes the stores it names (`gt:front@full`, `gt:lane.bev_fusion@full`, `gt:masks@full`; a job that writes what another job writes at that moment is refused by the runner).
The job runs as `how = "process"` under run.py in the interpreter of `GtGenerator/run_nice.sh` (the gcar environment, nice 15, one thread, the repository on PYTHONPATH). Only the calls that
write the stores are jobs; looking (`key`, `have`, `data`, `versions`) and the pictures stay commands.

No numpy and no devkit here: the job runner reads this table in its own interpreter (plain python3), and `import GtGenerator` is lazy for the same reason.

The class: the job runner builds `cls(project_folder)`, calls `method(**args)` and expects plain data with `ok`. The real class `GtGenerator.GtGenerator` is built from a nuScenes folder and its commands
return a `GtKey`, so a job sees this small class of the same name: the same method names and arguments (a test pins them against gt_api.py), the result is the key as plain data, `ok` = the key is complete."""
import os

RIGS = ("lane_center", "bev_fusion")                                                                  # the systems of the lane GT (gt_api.SYSTEMS; a test pins that)
NICE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_nice.sh")        # the "interpreter" of the child process: gcar python, nice 15, one thread (run_nice.sh passes the arguments on)


def _plain(key):
    """a GtKey as json: ok = complete, the key id and its parts, its line"""
    return {"ok": bool(key.complete), "text": str(key), **key.as_dict()}


class GtGenerator:
    """GtGenerator as the job runner builds it (see the module doc): the two commands that write the stores"""

    def __init__(self, project=None):
        self.project = project

    def generate_bev_gt(self, dataroot, set_name=None, scenes=None, out=None, version=None, workers=1, borders=(0.0, 0.5), verbose=True):
        """BEV Fusion: the front GT + the lane GT of the rig bev_fusion + the packed area masks for `set_name` (small | full) or `scenes`; resumable -> the GT key as plain data"""
        from . import gt_api
        return _plain(gt_api.generate_bev_gt(dataroot, set_name=set_name, scenes=scenes, out=out, version=version, workers=workers, borders=tuple(borders), verbose=verbose))

    def generate_lane_gt(self, dataroot, rig, set_name=None, scenes=None, out=None, version=None, workers=1, verbose=True):
        """the lane GT of ONE system, rig lane_center | bev_fusion, for `set_name` or `scenes`; resumable -> the GT key as plain data. A job per rig (not `all`, as the command has it), so that the store
        it writes is exactly the reference it names (`gt:lane.<rig>@<set>`) and the runner can tell two jobs that write the same store"""
        if rig not in RIGS:
            raise ValueError(f"rig must be one of {RIGS} (one job per system), not {rig!r}")
        from . import gt_api
        return _plain(gt_api.generate_lane_gt(dataroot, rig=rig, set_name=set_name, scenes=scenes, out=out, version=version, workers=workers, verbose=verbose))


JOBS = {
    "GtGenerator.generate_bev_gt": dict(suite="GtGenerator.jobs", cls="GtGenerator", method="generate_bev_gt", how="process", python=NICE, needs=dict(cpu=1, est_hours=6),
                                        reads=["path:{dataroot}", "set:{set_name}"], writes=["gt:front@{set_name}", "gt:lane.bev_fusion@{set_name}", "gt:masks@{set_name}"]),
    "GtGenerator.generate_lane_gt": dict(suite="GtGenerator.jobs", cls="GtGenerator", method="generate_lane_gt", how="process", python=NICE, needs=dict(cpu=1, est_hours=2),
                                         reads=["path:{dataroot}", "set:{set_name}"], writes=["gt:lane.{rig}@{set_name}"]),
}
