"""Where Bhaskera keeps what a user creates: in the folder the user set up (the *run folder*, the one that holds ``submit.sh``).

The run folder is ``$BHASKERA_WORKDIR`` (``submit.sh`` exports it for every job), else the current folder when it has a ``submit.sh``. Inside it:

    recipes/  checkpoints/  cache/  logs/  tasks/     made by the commands and the home page
    mlflow-runs/                                      the MLflow store: runs, adapters, evaluation samples and the model registry
    .bhaskera/                                        small settings (saved MLflow port, faulty nodes, an optional cluster profile) and training logs
    .cache/huggingface/                               benchmark data for offline evaluation

A user who already has the same thing in their home folder (``~/mlflow-runs``, ``~/.bhaskera``, ``~/.cache/huggingface``) keeps using it until a copy or a link
exists in the run folder, so that nobody's runs disappear. Without a run folder (a command run from somewhere else) the home folder is used, as before.
``$MLFLOW_TRACKING_URI`` always wins for the MLflow store.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


DEFAULT_NAME = "bhaskera-runs"


def default_folder(env: Optional[dict] = None) -> Path:
    """Where the run folder is made when the user names none: $BHASKERA_RUNS, else ~/bhaskera-runs."""
    env = os.environ if env is None else env
    return Path(env.get("BHASKERA_RUNS") or (Path.home() / DEFAULT_NAME))


def run_folder(env: Optional[dict] = None, cwd: Optional[Path] = None) -> Optional[Path]:
    """The folder that holds ``submit.sh``: $BHASKERA_WORKDIR, else the current folder, else the default folder in the home folder
    (when it exists, see ``ensure_default``); None when there is none."""
    env = os.environ if env is None else env
    given = env.get("BHASKERA_WORKDIR")
    if given and (Path(given) / "submit.sh").is_file():
        return Path(given).resolve()
    here = Path(cwd) if cwd else Path.cwd()
    if (here / "submit.sh").is_file():
        return here.resolve()
    mine = default_folder(env)
    if (mine / "submit.sh").is_file():
        return mine.resolve()
    return None


def ensure_default(env: Optional[dict] = None) -> Optional[Path]:
    """The default run folder, made from the shared kit (``$BHASKERA_KIT``, else ``$BHASKERA_DIR/slurm``) when it does not exist yet.

    A folder that already exists without ``submit.sh`` is left alone (None): nothing of the user's is overwritten."""
    env = os.environ if env is None else env
    mine = default_folder(env)
    if (mine / "submit.sh").is_file():
        return mine.resolve()
    if mine.exists() and any(mine.iterdir()):
        return None
    for c in (env.get("BHASKERA_KIT"), Path(env["BHASKERA_DIR"]) / "slurm" if env.get("BHASKERA_DIR") else None):
        if c and (Path(c) / "submit.sh").is_file():
            try:
                import shutil
                shutil.copytree(c, mine, dirs_exist_ok=True, copy_function=shutil.copy, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "logs"))
                for f in list(mine.rglob("*")) + [mine]:
                    f.chmod(f.stat().st_mode | 0o200)                 # the kit may be read-only; the copy is the user's
            except OSError:
                return None
            return mine.resolve()
    return None


def _pick(name: str, legacy: Path, env: Optional[dict], cwd: Optional[Path]) -> Path:
    """<run folder>/name when it exists, else the legacy home copy when it exists, else <run folder>/name (new), else the legacy place."""
    run = run_folder(env, cwd)
    if run and (run / name).exists():
        return run / name
    if legacy.exists():
        return legacy
    return run / name if run else legacy


def state_dir(env: Optional[dict] = None, cwd: Optional[Path] = None) -> Path:
    """Small settings and training logs: <run folder>/.bhaskera."""
    return _pick(".bhaskera", Path.home() / ".bhaskera", env, cwd)


def mlflow_store(env: Optional[dict] = None, cwd: Optional[Path] = None) -> str:
    """The MLflow store as a folder: $MLFLOW_TRACKING_URI, else <run folder>/mlflow-runs."""
    env = os.environ if env is None else env
    given = (env.get("MLFLOW_TRACKING_URI") or "").strip()
    if given:
        return given[len("file://"):] if given.startswith("file://") else given
    return str(_pick("mlflow-runs", Path.home() / "mlflow-runs", env, cwd))


def mlflow_uri(env: Optional[dict] = None, cwd: Optional[Path] = None) -> str:
    store = mlflow_store(env, cwd)
    return store if "://" in store else f"file://{store}"


def hf_home(env: Optional[dict] = None, cwd: Optional[Path] = None) -> Optional[Path]:
    """Where benchmark data is kept when it should not be the default of Hugging Face: <run folder>/.cache/huggingface.
    None means leave Hugging Face's own default (an existing ~/.cache/huggingface, or no run folder)."""
    env = os.environ if env is None else env
    if env.get("HF_HOME") or env.get("HF_DATASETS_CACHE"):
        return None
    run = run_folder(env, cwd)
    mine = run / ".cache" / "huggingface" if run else None
    if mine and mine.exists():
        return mine
    if (Path.home() / ".cache" / "huggingface").exists() or mine is None:
        return None
    return mine


def apply_cache_env(env: Optional[dict] = None, cwd: Optional[Path] = None) -> None:
    """Point Hugging Face at the run folder's cache for this process, unless the user set it or already has one at home."""
    target = hf_home(env, cwd)
    if target is not None:
        os.environ.setdefault("HF_HOME", str(target))
