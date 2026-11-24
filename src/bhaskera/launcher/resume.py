"""Continue a run that stopped half way (failed, was killed, ran out of time).

    python -m bhaskera.launcher.resume show --run <mlflow run id>          # what the run was, and which checkpoints it left
    ./submit.sh resume --run <run id> [--step N] [--nodes N] [--hours H]   # submit it again from its checkpoint

A run records its Slurm job (the ``slurm_job`` tag) and its checkpoint folder (the ``checkpoint.save_dir`` parameter); Slurm's
accounting keeps the command that submitted the job, with the recipe, the number of nodes and the time limit. Resuming submits
that recipe again with the same nodes and time (``--nodes`` and ``--hours`` change them). The training job finds the newest
complete checkpoint in the folder and continues from it: model, optimizer and learning-rate schedule, skipping the training data the
earlier steps already used. ``--step N`` resumes from an earlier checkpoint instead, for example before a loss spike; the checkpoints
after it are set aside (renamed ``rolledback-<step>``), not deleted. The recipe is read again, so a change made to it first (a lower
learning rate, say) applies to the resumed job.
"""
from __future__ import annotations

from bhaskera import paths

import argparse
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

_STEP = re.compile(r"^step_(\d+)$")
STOPPED = ("FAILED", "KILLED")


class ResumeError(Exception):
    """A reason the run cannot be resumed, in words for the person who asked."""


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def run_record(store: str | Path, run_id: str) -> dict:
    """The name, status, Slurm job, checkpoint folder and step limit of a run in an MLflow file store."""
    hits = list(Path(store).glob(f"*/{run_id}/meta.yaml"))
    if not hits:
        raise ResumeError(f"there is no run {run_id} in {store}")
    run = hits[0].parent
    meta = {}
    for line in _read(run / "meta.yaml").splitlines():
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip()
    status = {"3": "FINISHED", "4": "FAILED", "5": "KILLED", "1": "RUNNING"}.get(meta.get("status", ""), meta.get("status", ""))
    return {"run_id": run_id, "name": meta.get("run_name", ""), "status": status, "job": _read(run / "tags" / "slurm_job"),
            "save_dir": _read(run / "params" / "checkpoint.save_dir"), "max_steps": _read(run / "params" / "training.max_steps"),
            "run_dir": str(run)}


def parse_time(text: str) -> int:
    """Hours (rounded up) of a Slurm time limit such as 1:00:00, 12:00:00, 1-00:00:00 or 90."""
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [int(x) for x in text.split(":")]
    if len(parts) == 1:                    # Slurm: M, M:S or H:M:S
        h, m, s = 0, parts[0], 0
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        h, m, s = parts[:3]
    return max(1, days * 24 + math.ceil((h * 3600 + m * 60 + s) / 3600))


def slurm_submission(job_id: str, runner: Optional[Callable] = None) -> dict:
    """The recipe, nodes, time limit, folder and state of a finished job, from Slurm's accounting."""
    runner = runner or subprocess.run
    try:
        out = runner(["sacct", "-j", str(job_id), "-n", "-P", "-X", "-o", "SubmitLine,NNodes,Timelimit,WorkDir,State"],
                     capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError) as e:
        raise ResumeError(f"could not ask Slurm about job {job_id}: {e}")
    if not out:
        raise ResumeError(f"Slurm no longer knows job {job_id}, so its recipe and node count cannot be found; "
                          "submit the recipe again with ./submit.sh <recipe> <nodes> <hours>")
    line, nodes, limit, workdir, state = (out[0].split("|") + [""] * 5)[:5]
    try:
        tokens = shlex.split(line)
    except ValueError:
        tokens = line.split()
    recipe = next((t for t in tokens if t.endswith((".yaml", ".yml"))), "")
    time_arg = next((t.split("=", 1)[1] for t in tokens if t.startswith("--time=")), limit)
    return {"recipe": recipe, "nodes": int(nodes or 0), "hours": parse_time(time_arg) if time_arg else 0, "workdir": workdir,
            "state": state.split()[0] if state else ""}


def checkpoint_steps(save_dir: str) -> list[int]:
    """The steps of the complete checkpoints in a folder, oldest first."""
    root = Path(save_dir)
    if not save_dir or not root.is_dir():
        return []
    return sorted(int(m.group(1)) for p in root.iterdir() if p.is_dir() and (m := _STEP.match(p.name)) and (p / ".complete").exists())


def describe(store: str | Path, run_id: str, runner: Optional[Callable] = None) -> dict:
    """Everything needed to resume a run, and whether it can be: see the module text."""
    rec = run_record(store, run_id)
    info = {**rec, "recipe": "", "recipe_path": "", "nodes": 0, "hours": 0, "steps": [], "latest": 0, "resumable": False, "why": ""}
    if not rec["job"]:
        info["why"] = "this run does not record its Slurm job, so its recipe and node count are unknown"
        return info
    sub = slurm_submission(rec["job"], runner)
    if sub["state"] in ("RUNNING", "PENDING", "COMPLETING"):
        info["why"] = f"job {rec['job']} is still {sub['state'].lower()}"
        return info
    path = Path(sub["recipe"]) if sub["recipe"] else None
    if path is not None and not path.is_absolute():
        path = Path(sub["workdir"]) / path
    info.update(nodes=sub["nodes"], hours=sub["hours"], recipe=Path(sub["recipe"]).name if sub["recipe"] else "")
    if path is None or not path.is_file():
        info["why"] = f"the recipe {sub['recipe'] or '(unknown)'} is not there any more"
        return info
    info["recipe_path"] = str(path.resolve())
    save_dir = rec["save_dir"]
    if save_dir and not Path(save_dir).is_absolute():
        save_dir = str(Path(sub["workdir"]) / save_dir)
    info["save_dir"] = save_dir
    info["steps"] = checkpoint_steps(save_dir)
    info["latest"] = info["steps"][-1] if info["steps"] else 0
    if not info["steps"]:
        if _checkpoints_off(info["recipe_path"]):
            info["why"] = ("the recipe has checkpoint.enabled: false, so no checkpoint was saved and there is nothing to continue "
                           "from; turn it on (and keep checkpoint.save_every_steps) so that a failed job loses little, then submit it again")
        else:
            info["why"] = ("no checkpoint was saved before the job stopped, so there is nothing to continue from; lower "
                           "checkpoint.save_every_steps in the recipe so that a failed job loses little, and submit it again")
        return info
    info["resumable"] = True
    return info


def _checkpoints_off(recipe_path: str) -> bool:
    try:
        import yaml
        raw = yaml.safe_load(Path(recipe_path).read_text()) or {}
        return (raw.get("checkpoint") or {}).get("enabled") is False
    except Exception:
        return False


def _relative_save_dir(recipe_path: str) -> str:
    try:
        import yaml
        raw = yaml.safe_load(Path(recipe_path).read_text()) or {}
        value = (raw.get("checkpoint") or {}).get("save_dir") or ""
    except Exception:
        return ""
    return value if value and not Path(value).is_absolute() else ""


def plan(info: dict, step: int = 0, nodes: int = 0, hours: int = 0, submit_dir: Optional[str] = None) -> dict:
    """What `submit.sh resume` submits, or a ResumeError. ``submit_dir`` is the folder the job will be submitted from."""
    if not info["resumable"]:
        raise ResumeError(info["why"])
    relative = _relative_save_dir(info["recipe_path"])
    if submit_dir and relative and (Path(submit_dir) / relative).resolve() != Path(info["save_dir"]).resolve():
        raise ResumeError(f"the recipe's checkpoint folder '{relative}' is relative to the folder a job is submitted from, and the run "
                          f"saved its checkpoints in {info['save_dir']}. Submit from the folder the run was submitted from "
                          "(cd there, then ./submit.sh resume ...), or make checkpoint.save_dir an absolute path")
    if step and step not in info["steps"]:
        raise ResumeError(f"there is no checkpoint at step {step}; the run left checkpoints at steps {', '.join(map(str, info['steps']))}")
    return {"recipe": info["recipe_path"], "nodes": nodes or info["nodes"], "hours": hours or info["hours"],
            "step": step, "from_step": step or info["latest"], "run": info["run_id"]}


def _summary(info: dict) -> str:
    lines = [f"run {info['name']} ({info['run_id'][:8]}), {info['status'].lower()}, job {info['job'] or '-'}"]
    if info["recipe"]:
        lines.append(f"recipe {info['recipe']}, {info['nodes']} nodes, {info['hours']} h")
    if info["steps"]:
        lines.append(f"checkpoints at steps {', '.join(map(str, info['steps']))}" +
                     (f" of {info['max_steps']}" if info["max_steps"] else ""))
    lines.append("can be resumed" if info["resumable"] else f"cannot be resumed: {info['why']}")
    return "\n".join(lines)


def _store_default() -> str:
    import os
    return paths.mlflow_store()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="bhaskera-resume", description="Continue a run that stopped half way.")
    sub = p.add_subparsers(dest="action", required=True)
    for name in ("show", "plan"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True, help="MLflow run id")
        s.add_argument("--store", default=_store_default())
    pl = sub.choices["plan"]
    pl.add_argument("--step", type=int, default=0, help="resume from this checkpoint instead of the latest")
    pl.add_argument("--nodes", type=int, default=0)
    pl.add_argument("--hours", type=int, default=0)
    pl.add_argument("--submit-dir", dest="submit_dir", help="the folder the job will be submitted from (submit.sh passes its own)")
    args = p.parse_args(argv)
    try:
        info = describe(args.store, args.run)
        if args.action == "show":
            print(_summary(info))
            return 0 if info["resumable"] else 1
        todo = plan(info, args.step, args.nodes, args.hours, args.submit_dir)
        print(f"resuming {info['name']} from step {todo['from_step']} on {todo['nodes']} nodes for {todo['hours']} h", file=sys.stderr)
        print(f"{todo['recipe']} {todo['nodes']} {todo['hours']} {todo['step']} {todo['run']}")
        return 0
    except ResumeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
