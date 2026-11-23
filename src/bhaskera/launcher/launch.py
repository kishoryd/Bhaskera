"""Start a training, chat-server or benchmark job from the home page (the Start button).

The page sends a small form, never a command. Everything that reaches ``submit.sh`` is checked here first:

* a recipe is one of the files this module lists itself (the ``recipes`` folder of the working folder or the kit, and
  any folder in ``$BHASKERA_RECIPES``), picked by file name;
* nodes, hours and the example limit are whole numbers in a fixed range;
* a run id is 32 hex characters and belongs to a run that has an adapter;
* benchmark task names are letters, digits, ``_``, ``-`` and commas.

The command is a list of arguments (no shell). ``SKIP_QOS_CHECK`` and ``SKIP_PREFLIGHT`` are removed from its
environment: the page never skips a check. ``check`` runs the same script with ``DRY_RUN=1`` (limits and preflight, no
job). A runaway guard allows a few starts per hour. Set ``BHASKERA_HUB_LAUNCH=0`` to turn the buttons off.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

RUN_ID = re.compile(r"^[0-9a-f]{32}$")
TASKS = re.compile(r"^[A-Za-z0-9_,\-]{1,200}$")
RECIPE_FILE = re.compile(r"^[A-Za-z0-9_.\-]{1,100}\.ya?ml$")
SUBMITTED = re.compile(r"submitted job (\d+)")
PLACEHOLDER = re.compile(r"<project>|<username>|REPLACE_")
KINDS = ("train", "serve", "eval", "resume", "register", "cancel")
MAX_NODES, MAX_HOURS, MAX_LIMIT = 1024, 720, 100000
STARTS_PER_HOUR = 12
MIN_GAP_S = 2.0
TIMEOUT_S = 180


class LaunchError(ValueError):
    """The form is not acceptable; the text says why, in words for the person who filled it in."""


def find_kit(env: Optional[dict] = None, cwd: Optional[Path] = None) -> Optional[Path]:
    """The folder with submit.sh: $BHASKERA_WORKDIR, the current folder, $BHASKERA_KIT, then $BHASKERA_DIR/slurm."""
    env = os.environ if env is None else env
    cwd = Path.cwd() if cwd is None else cwd
    for c in (env.get("BHASKERA_WORKDIR"), cwd, env.get("BHASKERA_KIT"), Path(env["BHASKERA_DIR"]) / "slurm" if env.get("BHASKERA_DIR") else None):
        if c and (Path(c) / "submit.sh").is_file():
            return Path(c).resolve()
    return None


def recipe_dirs(kit: Path, env: Optional[dict] = None) -> list[Path]:
    env = os.environ if env is None else env
    dirs = [kit / "recipes"] + [Path(p) for p in (env.get("BHASKERA_RECIPES") or "").split(":") if p]
    seen, out = set(), []
    for d in dirs:
        d = d.expanduser()
        if d.is_dir() and d.resolve() not in seen:
            seen.add(d.resolve())
            out.append(d.resolve())
    return out


def _describe(path: Path) -> dict:
    """What the form shows about a recipe: model, method, strategy, and whether it still has placeholders."""
    import yaml
    info = {"file": path.name, "folder": str(path.parent), "model": "", "method": "", "strategy": "", "ready": True, "why": "", "tokenize": False}
    try:
        text = path.read_text()
        raw = yaml.safe_load(text) or {}
    except (OSError, yaml.YAMLError):
        return {**info, "ready": False, "why": "the file cannot be read as YAML"}
    if not isinstance(raw, dict):
        return {**info, "ready": False, "why": "not a recipe"}
    model = raw.get("model") if isinstance(raw.get("model"), dict) else {}
    info["model"] = os.path.basename(str(model.get("name", "")).rstrip("/"))
    lora = raw.get("lora") if isinstance(raw.get("lora"), dict) else {}
    quant = str(model.get("quantization") or "")
    info["method"] = "QLoRA" if quant == "qlora" else ("LoRA" if lora.get("enabled") else "full fine-tune")
    dist = raw.get("distributed") if isinstance(raw.get("distributed"), dict) else {}
    info["strategy"] = str(dist.get("strategy") or "")
    if "tokenize" in path.name or ("model" not in raw and "data" in raw):
        info["tokenize"], info["ready"], info["why"] = True, False, "a tokenizing recipe; run it with ./submit.sh tokenize"
    elif PLACEHOLDER.search(re.sub(r"(?m)^\s*#.*$", "", text)):        # comments may mention the placeholders
        info["ready"], info["why"] = False, "still has <project>/<username> or REPLACE_ placeholders: copy it, fill them in, and keep it in your own recipes folder"
    return info


def list_recipes(kit: Path, env: Optional[dict] = None) -> list[dict]:
    out = []
    for d in recipe_dirs(kit, env):
        for f in sorted(list(d.glob("*.yaml")) + list(d.glob("*.yml"))):
            if RECIPE_FILE.match(f.name):
                out.append(_describe(f))
    return out


def _int(value, name: str, low: int, high: int, required: bool = True) -> Optional[int]:
    if value in (None, ""):
        if required:
            raise LaunchError(f"{name} is missing")
        return None
    if isinstance(value, bool) or not re.fullmatch(r"\d{1,7}", str(value)):
        raise LaunchError(f"{name} must be a whole number")
    n = int(value)
    if not low <= n <= high:
        raise LaunchError(f"{name} must be between {low} and {high}")
    return n


class Launcher:
    def __init__(self, kit: Optional[Path], *, runs: Callable[[], list[dict]], env: Optional[dict] = None,
                 run: Callable[..., "subprocess.CompletedProcess"] = subprocess.run, clock: Callable[[], float] = time.time,
                 store: Optional[str] = None, describe: Optional[Callable[[str, str], dict]] = None,
                 jobs: Optional[Callable[[], list[dict]]] = None):
        self.kit, self._store, self._describe, self._jobs = kit, store, describe, jobs
        self._runs, self._env, self._run, self._clock = runs, dict(os.environ if env is None else env), run, clock
        self._lock = threading.Lock()
        self._starts: list[float] = []
        self._last = 0.0

    def disabled_reason(self) -> str:
        if self._env.get("BHASKERA_HUB_LAUNCH", "1") == "0":
            return "Starting jobs from this page is turned off (BHASKERA_HUB_LAUNCH=0)."
        if self.kit is None:
            return "submit.sh was not found. Start the hub from your working folder (the copy of the kit) or set BHASKERA_WORKDIR."
        missing = [v for v in ("BHASKERA_DIR", "SLURM_ACCOUNT") if not self._env.get(v)]
        if missing:
            return f"{' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not set in the shell that started this page; set {'it' if len(missing) == 1 else 'them'} and start it again."
        return ""

    def options(self) -> dict:
        reason = self.disabled_reason()
        from . import cluster_profile
        prof = cluster_profile.current()
        try:
            rules = [{"max_nodes": r.max_nodes, "partition": r.name, "max_hours": r.max_hours} for r in prof.partitions]
        except Exception:
            rules = []
        recipes = list_recipes(self.kit, self._env) if self.kit else []
        runs = [{"run_id": r["run_id"], "name": r["name"], "status": r["status"]} for r in self._runs() if r.get("has_adapter")]
        return {"enabled": not reason, "reason": reason, "recipes": recipes, "runs": runs, "stopped": self._stopped(), "models": self._models(), "partitions": rules,
                "gpus_per_node": prof.gpus_per_node, "default_hours": 2, "max_nodes": MAX_NODES, "max_hours": MAX_HOURS}

    def _recipe_path(self, name) -> Path:
        if not isinstance(name, str) or not RECIPE_FILE.match(name):
            raise LaunchError("pick a recipe from the list")
        folder = None
        for d in recipe_dirs(self.kit, self._env):
            if (d / name).is_file():
                folder = d
                break
        if folder is None:
            raise LaunchError("that recipe is not in the list")
        info = _describe(folder / name)
        if not info["ready"]:
            raise LaunchError(f"{name} cannot be started: {info['why']}")
        return folder / name

    def _models(self) -> list[dict]:
        """The models already in the registry (name, number of versions, aliases), to offer when registering another version."""
        try:
            from . import register
            return [{"name": m["name"], "versions": len(m["versions"]), "aliases": sorted({a for v in m["versions"] for a in v["aliases"]})}
                    for m in register.listing(register.make_client(self._store or register._store_default()))]
        except Exception:
            return []

    def _register(self, form: dict, dry: bool) -> dict:
        """Save the adapter of a trained run in the model registry under a name. Quick, so no Slurm job: the command runs here."""
        from . import register
        run = self._run_id(form.get("run"))
        name, alias = form.get("name"), form.get("alias") or None
        if not isinstance(name, str) or not register.NAME.match(name):
            raise LaunchError("a model name is letters, digits, _, - and . (at most 64), starting with a letter or digit")
        if alias is not None and (not isinstance(alias, str) or not register.ALIAS.match(alias)):
            raise LaunchError("an alias is letters, digits, _ and - (at most 32), starting with a letter")
        if dry:
            return {"ok": True, "job": None, "dry": True, "output": f"Looks fine: run {run[:8]} would be registered as {name}" + (f" with the alias {alias}" if alias else "")}
        cmd = [sys.executable, "-m", "bhaskera.launcher.register", "add", "--run", run, "--name", name] + (["--alias", alias] if alias else [])
        if self._store:
            cmd += ["--store", self._store]
        try:
            done = self._run(cmd, cwd=str(self.kit), env={k: v for k, v in self._env.items()}, capture_output=True, text=True, timeout=60,
                             stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return {"ok": False, "job": None, "dry": False, "output": "registering took too long; try again"}
        return {"ok": done.returncode == 0, "job": None, "dry": False, "output": ((done.stdout or "") + (done.stderr or "")).strip()[-2000:]}

    def _cancel(self, form: dict, dry: bool) -> dict:
        """Cancel one of the user's own Bhaskera jobs (training, chat server, benchmark): `scancel <job>`. Never another job."""
        job = form.get("job")
        if not isinstance(job, str) or not re.fullmatch(r"\d{1,12}", job):
            raise LaunchError("pick a job from the list")
        mine = {str(j["id"]): j for j in (self._jobs() if self._jobs else []) if str(j.get("comment", "")).startswith("bhaskera")}
        if job not in mine:
            raise LaunchError("that is not one of your running Bhaskera jobs, or it has already ended")
        if dry:
            return {"ok": True, "job": job, "dry": True, "output": f"Job {job} ({mine[job].get('name', '')}) would be cancelled"}
        try:
            done = self._run(["scancel", job], cwd=str(self.kit), env=dict(self._env), capture_output=True, text=True, timeout=30,
                             stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError) as e:
            return {"ok": False, "job": job, "dry": False, "output": f"could not run scancel: {e}"}
        return {"ok": done.returncode == 0, "job": job, "dry": False,
                "output": ((done.stdout or "") + (done.stderr or "")).strip()[-1000:] or f"job {job} cancelled"}

    def _stopped(self) -> list[dict]:
        """Runs that stopped half way (failed, killed, or no longer writing) and know their Slurm job."""
        return [{"run_id": r["run_id"], "name": r["name"], "status": "STALLED" if r.get("stale") else r["status"]}
                for r in self._runs() if r.get("job") and (r["status"] in ("FAILED", "KILLED") or r.get("stale"))]

    def resume_info(self, run_id: str) -> dict:
        """What a stopped run left behind (recipe, nodes, time, checkpoints) and whether it can be resumed."""
        if not isinstance(run_id, str) or not RUN_ID.match(run_id) or run_id not in {r["run_id"] for r in self._stopped()}:
            raise LaunchError("pick a stopped run from the list")
        from . import resume
        try:
            info = (self._describe or resume.describe)(self._store or resume._store_default(), run_id)
        except resume.ResumeError as e:
            return {"run_id": run_id, "resumable": False, "why": str(e), "steps": []}
        return {k: info[k] for k in ("run_id", "name", "status", "recipe", "nodes", "hours", "steps", "latest", "max_steps", "resumable", "why")}

    def _run_id(self, value) -> str:
        if not isinstance(value, str) or not RUN_ID.match(value):
            raise LaunchError("pick a trained run from the list")
        if value not in {r["run_id"] for r in self._runs() if r.get("has_adapter")}:
            raise LaunchError("that run has no saved adapter to use")
        return value

    def build(self, form: dict) -> tuple[list[str], dict]:
        """(arguments of submit.sh, extra environment) for a form, or LaunchError."""
        kind = form.get("kind")
        if kind not in KINDS:
            raise LaunchError("choose what to start: training, chat server, benchmark or resume")
        if kind == "train":
            recipe = self._recipe_path(form.get("recipe"))
            nodes = _int(form.get("nodes"), "nodes", 1, MAX_NODES)
            hours = _int(form.get("hours"), "hours", 1, MAX_HOURS, required=False)
            return [str(recipe), str(nodes)] + ([str(hours)] if hours else []), {}
        if kind == "resume":
            run = form.get("run")
            if not isinstance(run, str) or run not in {r["run_id"] for r in self._stopped()}:
                raise LaunchError("pick a stopped run from the list")
            args = ["resume", "--run", run]
            for field, flag, low, high in (("step", "--step", 1, 10_000_000), ("nodes", "--nodes", 1, MAX_NODES), ("hours", "--hours", 1, MAX_HOURS)):
                v = _int(form.get(field), field, low, high, required=False)
                if v:
                    args += [flag, str(v)]
            return args, {}
        run = self._run_id(form.get("run"))
        hours = _int(form.get("hours"), "hours", 1, MAX_HOURS, required=False) or 2
        args = [kind, "--run", run]
        if form.get("four_bit") is True:
            args.append("--4bit")
        if kind == "eval":
            if form.get("samples") is True:
                args.append("--save-samples")
            tasks = form.get("tasks") or ""
            if tasks:
                if not isinstance(tasks, str) or not TASKS.match(tasks):
                    raise LaunchError("benchmark names are letters, digits, _ and - separated by commas")
                args += ["--tasks", tasks]
            limit = _int(form.get("limit"), "examples per task", 1, MAX_LIMIT, required=False)
            if limit:
                args += ["--limit", str(limit)]
        return args, {"EVAL_HOURS" if kind == "eval" else "SERVE_HOURS": str(hours)}

    def _guard(self, now: float) -> None:
        self._starts = [t for t in self._starts if now - t < 3600]
        if now - self._last < MIN_GAP_S:
            raise LaunchError("one moment: the last start was only just now")
        if len(self._starts) >= STARTS_PER_HOUR:
            raise LaunchError(f"{STARTS_PER_HOUR} jobs were started from this page in the last hour; use the terminal for more")

    def launch(self, form: dict) -> dict:
        """{'ok', 'job', 'output'} for action 'start' (a job is submitted) or 'check' (limits and preflight only)."""
        reason = self.disabled_reason()
        if reason:
            raise LaunchError(reason)
        dry = form.get("action") == "check"
        if form.get("kind") == "register":
            return self._register(form, dry)
        if form.get("kind") == "cancel":
            return self._cancel(form, dry)
        args, extra = self.build(form)
        if not self._lock.acquire(blocking=False):
            raise LaunchError("another start is still running; wait for it to finish")
        try:
            now = self._clock()
            if not dry:
                self._guard(now)
                self._last = now
            env = {k: v for k, v in self._env.items() if k not in ("SKIP_QOS_CHECK", "SKIP_PREFLIGHT", "DRY_RUN")}
            env.update(extra)
            if dry:
                env["DRY_RUN"] = "1"
            try:
                done = self._run(["bash", str(self.kit / "submit.sh")] + args, cwd=str(self.kit), env=env, capture_output=True, text=True,
                                 timeout=TIMEOUT_S, stdin=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                return {"ok": False, "job": None, "dry": dry, "output": f"submit.sh did not finish within {TIMEOUT_S} s; check squeue before trying again."}
            out = ((done.stdout or "") + (done.stderr or "")).strip()
            found = SUBMITTED.search(out)
            ok = done.returncode == 0 and (dry or bool(found))
            if ok and not dry:
                self._starts.append(now)
            return {"ok": ok, "job": found.group(1) if found else None, "dry": dry, "output": out[-6000:]}
        finally:
            self._lock.release()
