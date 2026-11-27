"""The worker of one run: its steps in order, from the run's snapshot, never from the working tree.

The code is the snapshot commit unpacked once per commit into <runs>/<problem>/_code/<commit>/. A step is a command of problem.json (its words
may name {python} {config} {run_dir} {data} {runs} {seed}) or the core's evaluation step. A GPU step waits for one of the machine's GPU slots.
Exit 0 = done; 75 = stopped (trainkit's stop protocol; `research resume` continues); anything else = failed. A kill rule of the card that fires
stops the training and marks the run killed; a step's log.jsonl and CONTROL lie in its folder (problem.json step "dir", default the run
folder). run.json (in the records) changes at step boundaries only; the heartbeat lies in the run folder."""
from __future__ import annotations

import fcntl
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from research import spec
from research.launch import SNAPSHOT_PATHS
from research.paths import (ResearchError, child_env, expand, interpreter, machine, module_path, read_json, record_dir, repo, run_dir, runs_root,
                            state_dir, write_json)
from research import gitio

EXIT_STOPPED = 75
BEAT_S = 15


@contextmanager
def flocked(path: str, wait: bool = True) -> Iterator[bool]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        yield True
    finally:
        os.close(fd)


class Stopped(Exception):
    """a stop request came while the run waited"""


@contextmanager
def gpu_slot(n_slots: int, beat, stopped=lambda: False) -> Iterator[int]:
    """one of the machine's GPU slots, waited for (the heartbeat goes on while waiting; a stop request ends the wait: Stopped)"""
    if n_slots < 1:
        raise ResearchError("this machine has no GPU slot (machine.json gpu_slots) and the step needs one")
    while True:
        for k in range(n_slots):
            path = os.path.join(state_dir(), "slots", f"gpu{k}.lock")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            try:
                yield k
            finally:
                os.close(fd)
            return
        if stopped():
            raise Stopped()
        beat("waiting for a GPU slot")
        time.sleep(5)


def ensure_code(problem: str, commit: str) -> str:
    """the snapshot unpacked (once per commit; a half-written folder is never used)"""
    base = os.path.join(runs_root(problem), "_code")
    dest = os.path.join(base, commit[:12])
    if os.path.isfile(os.path.join(dest, ".complete")):
        return dest
    with flocked(dest + ".lock"):
        if not os.path.isfile(os.path.join(dest, ".complete")):
            if os.path.exists(dest):
                os.rename(dest, f"{dest}.broken{int(time.time())}")
            root = repo()
            paths = [x for x in SNAPSHOT_PATHS if gitio.git(root, "ls-tree", "--name-only", commit, "--", x)] + [f"problems/{problem}"]
            gitio.unpack(root, commit, dest, paths)
            open(os.path.join(dest, ".complete"), "w").close()
    return dest


def step_alive(problem: str, rid: str) -> bool:
    """does the process group of the run's current step (step.pid in the run folder) still exist?"""
    try:
        with open(os.path.join(run_dir(problem, rid), "step.pid"), encoding="utf-8") as f:
            pid = int(f.read().strip())
        os.killpg(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def needed(other: str, status: Optional[str], stop: bool) -> Optional[str]:
    """the rule for a run this one needs: None = wait, '' = go on, else why this run ends; a text that starts with 'stopped' ends it as
    stopped (`resume` takes it again), any other as failed. A stop request comes first; a needed run that was stopped stops this one too"""
    if status == "ok":
        return ""
    if stop:
        return f"stopped while it waited for {other}"
    if status == "stopped":
        return f"stopped: the needed run {other} is stopped (resume it and this one)"
    if status in ("failed", "killed", "missing"):
        return f"the needed run {other} is {status}"
    return None


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class Kill:
    """the card's kill rules, checked on every row of the step's log.jsonl as it grows (from the rule's after_step on); NaN fires a rule"""

    def __init__(self, rules: List[Dict[str, Any]], log_path: str):
        self.rules, self.path, self.pos, self.done = list(rules), log_path, 0, set()

    def fired(self) -> str:
        if not self.rules or not os.path.exists(self.path):
            return ""
        with open(self.path, encoding="utf-8") as f:
            f.seek(self.pos)
            chunk = f.read()
        lines = chunk.split("\n")
        self.pos += len(chunk) - len(lines[-1])                 # a half-written last line is read again next time
        for line in lines[:-1]:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or not _number(row.get("step")):
                continue                                         # not a row of a step
            for i, r in enumerate(self.rules):
                value = row.get(r["metric"])
                if i in self.done or row["step"] < r["after_step"] or not _number(value):
                    continue                                     # not a number in this row (a nested validation block): the next row may carry it
                v = float(value)
                if math.isnan(v):
                    why = f"kill rule: {r['metric']} is not a number at step {row['step']}"
                elif "above" in r and v > r["above"]:
                    why = f"kill rule: {r['metric']} = {v:.4g} at step {row['step']} is above {r['above']}"
                elif "below" in r and v < r["below"]:
                    why = f"kill rule: {r['metric']} = {v:.4g} at step {row['step']} is below {r['below']}"
                else:
                    continue
                self.done.add(i)
                return why
        return ""


class Worker:
    def __init__(self, problem: str, rid: str, test: bool = False):
        self.problem, self.rid, self.test = problem, rid, test
        self.rec = record_dir(problem, rid)
        self.out = run_dir(problem, rid)
        self.run_path = os.path.join(self.rec, "run.json")
        self.run = read_json(self.run_path)
        self.m = machine()

    # ---------------------------------------------------------------------------------------- bookkeeping
    def save(self, **fields: Any) -> None:
        self.run.update(fields)
        write_json(self.run_path, self.run)

    def beat(self, text: str = "") -> None:
        with open(os.path.join(self.out, "heartbeat"), "w", encoding="utf-8") as f:
            f.write(f"{time.time():.0f} {text}\n")

    def stop_requested(self) -> bool:
        return os.path.exists(os.path.join(self.out, "STOP_REQUESTED"))

    # ---------------------------------------------------------------------------------------- the steps
    def command(self, step: Dict[str, Any], code: str, cfg_path: str, p: Dict[str, Any]) -> List[str]:
        py = interpreter(p["python"])
        if step["run"] == "evaluate":
            return [py, "-m", "research.evaluate", self.problem, self.rid] + (["--test"] if self.test else [])
        names = {"python": py, "config": cfg_path, "run_dir": self.out, "data": self.m["data"], "runs": runs_root(self.problem),
                 "seed": str(self.run["seed"]), "split": self.split}
        try:
            return [w.format(**names) for w in step["run"]]
        except (KeyError, IndexError, ValueError) as e:
            raise ResearchError(f"step command {step['run']}: unknown placeholder {e}") from None

    def env(self, code: str) -> Dict[str, str]:
        e = child_env(RESEARCH_REPO=repo(), RESEARCH_PROBLEM=self.problem, RESEARCH_RUN=self.rid, RESEARCH_RUN_DIR=self.out, RESEARCH_RECORD=self.rec,
                      RESEARCH_DATA=self.m["data"], RESEARCH_RUNS=runs_root(self.problem), RESEARCH_COMMIT=self.run["commit"], RESEARCH_CODE=code,
                      RESEARCH_SPLIT=self.split, PYTHONPATH=module_path(code, self.problem), PYTHONUNBUFFERED="1")
        if self.test:
            e["RESEARCH_HOLDOUT_RUN"] = self.rid
        return e

    def execute(self, name: str, cmd: List[str], code: str, kill: Optional[Kill], where: str) -> Dict[str, Any]:
        """one step; `where` is its folder (CONTROL goes there)"""
        log_path = os.path.join(self.out, f"{name}.log")
        pid_path = os.path.join(self.out, "step.pid")
        control = os.path.join(where, "CONTROL")
        t0 = time.time()
        with open(log_path, "ab") as log:
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n".encode())
            log.flush()
            proc = subprocess.Popen(cmd, cwd=os.path.join(code, "problems", self.problem), env=self.env(code), stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            with open(pid_path, "w") as f:
                f.write(str(proc.pid))
            reason, asked = "", 0.0
            try:
                while proc.poll() is None:
                    self.beat(name)
                    if not reason:
                        reason = kill.fired() if kill else ""
                        if not reason and self.stop_requested():
                            reason = "stop requested"
                    if reason and not os.path.exists(control):  # written again while it is missing: a trainer that starts moves an early one aside
                        write_json(control, {"cmd": "stop", "by": "research worker", "reason": reason})
                        asked = asked or time.time()
                    if asked and time.time() - asked > 600:      # a step that ignores the stop protocol is ended
                        os.killpg(proc.pid, signal.SIGTERM)
                        time.sleep(30)
                        if proc.poll() is None:
                            os.killpg(proc.pid, signal.SIGKILL)
                    for _ in range(BEAT_S):
                        if proc.poll() is not None:
                            break
                        time.sleep(1)
            finally:
                if proc.poll() is None:                          # the worker itself fails: its step must not run on alone
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(60)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                if os.path.exists(pid_path):
                    os.remove(pid_path)
        code_ = proc.returncode
        status = "ok" if code_ == 0 and not reason else ("killed" if reason.startswith("kill rule") else "stopped" if reason or code_ == EXIT_STOPPED else "failed")
        return {"status": status, "exit": code_, "seconds": round(time.time() - t0, 1), "log": os.path.basename(log_path), **({"why": reason} if reason else {})}

    def wait_needs(self, card_needs: Dict[str, str]) -> str:
        """'' when every run this one needs is finished, else why it cannot go on (see `needed`)"""
        for key, ref in card_needs.items():
            other = f"{ref}_s{self.run['seed']}"
            path = os.path.join(record_dir(self.problem, other), "run.json")
            while True:
                try:
                    st = read_json(path).get("status")
                except (OSError, ValueError):
                    st = "missing"
                why = needed(other, st, self.stop_requested())
                if why is None:
                    self.beat(f"waiting for {other}")
                    time.sleep(BEAT_S)
                    continue
                if why:
                    return why
                break
        return ""

    def main(self) -> int:
        os.makedirs(self.out, exist_ok=True)
        with flocked(os.path.join(self.out, "worker.lock"), wait=False) as mine:
            if not mine:
                print(f"{self.rid}: another worker runs it", flush=True)
                return 0
            try:
                return self._main()
            except Exception as e:                                # noqa: BLE001 - a worker always leaves a record
                if self.test:                                      # a TEST evaluation never changes the run's own record
                    print(f"TEST evaluation of {self.rid} failed: {type(e).__name__}: {e}", flush=True)
                else:
                    self.save(status="failed", error=f"{type(e).__name__}: {e}"[:500], ended=time.strftime("%Y-%m-%d %H:%M:%S"))
                raise

    @staticmethod
    def _rotate(folder: str, names: tuple) -> None:
        """a fresh attempt: an earlier stop is history"""
        for f in names:
            path = os.path.join(folder, f)
            if os.path.exists(path):
                os.replace(path, f"{path}.{int(time.time())}")

    def _main(self) -> int:
        self._rotate(self.out, ("CONTROL", "STOPPED", "STOP_REQUESTED"))
        code = ensure_code(self.problem, self.run["commit"])
        p = read_json(os.path.join(code, "problems", self.problem, "problem.json"))
        for d in sorted({s.get("dir", "") for s in p["steps"].values()} - {""}):
            self._rotate(os.path.join(self.out, d), ("CONTROL", "STOPPED"))
        card = spec.parse_card(open(os.path.join(self.rec, "..", "..", "experiment.md"), encoding="utf-8").read())
        frozen = spec.parse_card(gitio.show(repo(), self.run["commit"], os.path.relpath(os.path.join(self.rec, "..", "..", "experiment.md"), repo()).replace(os.sep, "/")))
        if not spec.same_rule(frozen, card):
            raise ResearchError("the card in the working tree is not the card of the snapshot: the run follows the snapshot, and so must the records")
        self.split = p["holdout"][0] if self.test else spec.split_of(frozen, p)
        cfg = expand(read_json(os.path.join(self.rec, "config.json")), self.problem)
        cfg_path = os.path.join(self.out, "config.json")
        write_json(cfg_path, cfg)
        if not self.test:
            self.save(status="running", host=socket.gethostname(), started=self.run.get("started") or time.strftime("%Y-%m-%d %H:%M:%S"))
        steps = list(p.get("test_steps", ["evaluate"])) if self.test else list(self.run["steps"])
        why = "" if self.test else self.wait_needs(frozen["arms"][self.run["arm"]].get("needs", {}))
        if why:
            stopped = why.startswith("stopped")
            self.save(status="stopped" if stopped else "failed", error=why, ended=time.strftime("%Y-%m-%d %H:%M:%S"))
            return EXIT_STOPPED if stopped else 1
        for name in steps:
            if not self.test and self.run["steps"][name].get("status") == "ok":
                continue
            if self.stop_requested():
                if not self.test:
                    self.save(status="stopped", ended=time.strftime("%Y-%m-%d %H:%M:%S"))
                return EXIT_STOPPED
            step = p["steps"][name]
            cmd = self.command(step, code, cfg_path, p)
            where = os.path.join(self.out, step.get("dir", ""))
            kill = Kill(frozen.get("kill", []), os.path.join(where, "log.jsonl")) if step["run"] != "evaluate" and not self.test else None
            if not self.test:
                self.run["steps"][name] = {"status": "running", "since": time.strftime("%Y-%m-%d %H:%M:%S")}
                self.save()
            if step.get("gpus", 0) > 0:
                try:
                    with gpu_slot(self.m["gpu_slots"], self.beat, self.stop_requested) as slot:
                        res = {**self.execute(name, cmd, code, kill, where), "gpu_slot": slot}
                except Stopped:
                    res = {"status": "stopped", "why": "stop requested while it waited for a GPU slot"}
            else:
                res = self.execute(name, cmd, code, kill, where)
            if self.test:
                if res["status"] != "ok":
                    print(f"TEST step {name} of {self.rid}: {res}", flush=True)
                    return 1
                continue
            self.run["steps"][name] = res
            if res["status"] != "ok":
                self.save(status=res["status"], ended=time.strftime("%Y-%m-%d %H:%M:%S"))
                return 1 if res["status"] in ("failed", "killed") else EXIT_STOPPED
            self.save()
        if not self.test:
            self.save(status="ok", ended=time.strftime("%Y-%m-%d %H:%M:%S"))
        return 0


def main(problem: str, rid: str, test: bool = False) -> int:
    return Worker(problem, rid, test).main()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], "--test" in sys.argv[3:]))
