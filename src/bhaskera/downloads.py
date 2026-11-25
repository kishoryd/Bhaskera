"""Download a model or a dataset from Hugging Face into a folder you choose, and sign in with your Hugging Face key.

Compute nodes have no internet, so models and data have to be on disk before a job starts. This runs on the machine that serves
the home page (a login node), as the user, and writes into the folder the user names.

* The repository name is checked against Hugging Face's own format, and only Hugging Face is contacted: the page never takes a web address.
* The folder must be a full path the user can write to. System folders are refused, and the operating system's permissions still apply.
* A download is a separate process that can be cancelled, and starting the same download again resumes it (finished files are kept).
* **Sign in:** a key pasted on the page is checked with Hugging Face, then kept in ``<hf home>/token`` (readable by the user only), the
  place ``huggingface-cli login`` uses. It is never shown again, never logged and never put in a command line: a download gets it through
  its environment. Sign out deletes the file.
* A finished model is remembered (``<settings>/downloads.json``) so that the model lists offer it; a finished dataset is registered as a
  dataset version, with the repository and revision as its source.
"""
from __future__ import annotations

import collections
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from . import dataversions, paths

REPO = re.compile(r"^(?:[A-Za-z0-9][A-Za-z0-9._-]{0,95}/)?[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$")
TOKEN = re.compile(r"^hf_[A-Za-z0-9]{20,200}$")
KINDS = ("model", "dataset")
DENIED = ("/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64", "/boot", "/proc", "/sys", "/dev", "/var", "/opt", "/root", "/run", "/srv")
LARGE_BYTES = 50 << 30
MAX_WORKERS = "4"


class DownloadError(ValueError):
    """A problem with the request, in words for the person who made it."""


def _clean(text: str) -> str:
    return re.sub(r"hf_[A-Za-z0-9]{8,}", "hf_…", text or "")


# ---------------------------------------------------------------- sign in
def _hf_dir(env: Optional[dict] = None) -> Path:
    env = os.environ if env is None else env
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"])
    return paths.hf_home(env) or (Path.home() / ".cache" / "huggingface")


def token_file(env: Optional[dict] = None) -> Path:
    return _hf_dir(env) / "token"


def token(env: Optional[dict] = None) -> Optional[str]:
    """The key in use: $HF_TOKEN, else the saved file (this run folder's, else the usual Hugging Face place)."""
    env = os.environ if env is None else env
    given = (env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if given:
        return given
    for f in (token_file(env), Path.home() / ".cache" / "huggingface" / "token"):
        try:
            text = f.read_text().strip()
            if text:
                return text
        except OSError:
            continue
    return None


def _whoami(tok: str) -> dict:
    from huggingface_hub import HfApi
    return HfApi(token=tok).whoami()


def auth_status(env: Optional[dict] = None, whoami: Optional[Callable[[str], dict]] = None) -> dict:
    """{'signed_in', 'user', 'source'} for the key in use; the key itself is never returned."""
    tok = token(env)
    if not tok:
        return {"signed_in": False}
    source = "environment" if ((env or os.environ).get("HF_TOKEN") or (env or os.environ).get("HUGGING_FACE_HUB_TOKEN")) else "saved"
    try:
        who = (whoami or _whoami)(tok)
        return {"signed_in": True, "user": who.get("name") or who.get("fullname") or "", "source": source}
    except Exception as e:
        return {"signed_in": False, "problem": _failure(e), "source": source}


def sign_in(key, env: Optional[dict] = None, whoami: Optional[Callable[[str], dict]] = None) -> dict:
    """Check the key with Hugging Face and keep it for this user. Returns the status; raises DownloadError when the key is refused."""
    if not isinstance(key, str) or not TOKEN.match(key.strip()):
        raise DownloadError("that does not look like a Hugging Face key: it starts with hf_ and has no spaces (create one at huggingface.co/settings/tokens)")
    key = key.strip()
    try:
        who = (whoami or _whoami)(key)
    except Exception as e:
        raise DownloadError(_failure(e, signing_in=True))
    f = token_file(env)
    f.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(f), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as out:
        out.write(key)
    os.chmod(f, 0o600)
    return {"signed_in": True, "user": who.get("name") or who.get("fullname") or "", "source": "saved"}


def sign_out(env: Optional[dict] = None) -> dict:
    removed = False
    for f in (token_file(env), Path.home() / ".cache" / "huggingface" / "token"):
        try:
            f.unlink()
            removed = True
        except OSError:
            pass
    still = bool((env or os.environ).get("HF_TOKEN") or (env or os.environ).get("HUGGING_FACE_HUB_TOKEN"))
    return {"signed_in": False, "removed": removed, "note": "A key set in the environment (HF_TOKEN) is still used." if still else ""}


def _failure(e: Exception, signing_in: bool = False) -> str:
    name = type(e).__name__
    text = _clean(str(e))
    if name in ("GatedRepoError",):
        return "this repository needs access: open it on huggingface.co, accept its licence, and sign in with your key"
    if name in ("RepositoryNotFoundError", "RevisionNotFoundError", "EntryNotFoundError"):
        return "Hugging Face has no such repository or revision (a private one needs you to sign in first)"
    if name in ("HTTPError", "HfHubHTTPError") and ("401" in text or "403" in text):
        return "Hugging Face did not accept this key" if signing_in else "Hugging Face refused access: sign in with a key that can read it"
    if "401" in text and signing_in:
        return "Hugging Face did not accept this key"
    if any(w in name for w in ("Connection", "Timeout", "Offline", "SSL")) or any(w in text.lower() for w in ("connection", "timed out", "resolve", "network", "offline", "unreachable")):
        return "could not reach Hugging Face from this machine: compute nodes have no internet, so start the page on a login node"
    return text[:300] or name


# ---------------------------------------------------------------- checks
def check_repo(kind, repo, revision) -> tuple[str, str, str]:
    if kind not in KINDS:
        raise DownloadError("choose a model or a dataset")
    if not isinstance(repo, str) or not REPO.match(repo.strip()):
        raise DownloadError("a repository name looks like owner/name, for example Qwen/Qwen3-14B (letters, digits, . _ -)")
    rev = (revision or "main")
    if not isinstance(rev, str) or not REVISION.match(rev.strip()):
        raise DownloadError("a revision is a branch, tag or commit name (letters, digits, . _ - /)")
    return kind, repo.strip(), rev.strip()


def check_target(path) -> Path:
    """The folder to download into: a full path, not a system folder, that the user can write to (it is made when it does not exist)."""
    if not isinstance(path, str) or not path.strip():
        raise DownloadError("the folder to download into is missing")
    p = Path(path.strip()).expanduser()
    if not p.is_absolute():
        raise DownloadError("the folder must be a full path (starting with /)")
    p = Path(os.path.realpath(p))
    for bad in DENIED:
        if str(p) == bad or str(p).startswith(bad + os.sep):
            raise DownloadError(f"{bad} is a system folder; choose a folder of your own")
    if p.exists() and not p.is_dir():
        raise DownloadError(f"{p} is a file, not a folder")
    anc = p
    while not anc.exists():
        anc = anc.parent
    if not os.access(anc, os.W_OK | os.X_OK):
        raise DownloadError(f"you cannot write in {anc}")
    return p


def preview(kind, repo, revision, env: Optional[dict] = None, info_fn: Optional[Callable] = None) -> dict:
    """What a download would fetch: size, files, licence, and whether it needs access. Reads Hugging Face only (no file is downloaded)."""
    kind, repo, rev = check_repo(kind, repo, revision)
    tok = token(env)
    try:
        if info_fn is not None:
            info = info_fn(kind, repo, rev, tok)
        else:
            from huggingface_hub import HfApi
            api = HfApi(token=tok)
            info = (api.model_info if kind == "model" else api.dataset_info)(repo, revision=rev, files_metadata=True)
    except Exception as e:
        raise DownloadError(_failure(e))
    files = sorted(((getattr(s, "rfilename", ""), int(getattr(s, "size", 0) or 0)) for s in (info.siblings or [])), key=lambda x: -x[1])
    total = sum(s for _, s in files)
    card = getattr(info, "card_data", None)
    lic = ""
    try:
        lic = str((card.to_dict() if hasattr(card, "to_dict") else card or {}).get("license") or "")
    except Exception:
        pass
    return {"kind": kind, "repo": repo, "revision": rev, "bytes": total, "files": len(files), "largest": [{"name": n, "bytes": s} for n, s in files[:8]],
            "gated": bool(getattr(info, "gated", False)), "private": bool(getattr(info, "private", False)), "license": lic,
            "sha": str(getattr(info, "sha", "") or "")[:12], "large": total > LARGE_BYTES}


# ---------------------------------------------------------------- remembered downloads
def records_file() -> Path:
    return paths.state_dir() / "downloads.json"


def records() -> list[dict]:
    try:
        data = json.loads(records_file().read_text())
        return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _remember(rec: dict) -> None:
    f = records_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    data = [r for r in records() if not (r.get("path") == rec["path"] and r.get("repo") == rec["repo"])] + [rec]
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, f)


def downloaded_models() -> list[str]:
    """Folders of models downloaded from this page that still exist."""
    return [r["path"] for r in records() if r.get("kind") == "model" and (Path(r["path"]) / "config.json").is_file()]


# ---------------------------------------------------------------- the one running task
def _tree_bytes(root: Path) -> int:
    total = 0
    for dirpath, _, names in os.walk(root):
        for n in names:
            try:
                total += os.path.getsize(os.path.join(dirpath, n))
            except OSError:
                pass
    return total


class Manager:
    """One background task at a time: a download, or the fingerprinting of a dataset. Poll ``status()``."""

    def __init__(self, env: Optional[dict] = None, popen: Callable = subprocess.Popen, python: Optional[str] = None):
        self._env = dict(os.environ if env is None else env)
        self._popen, self._python = popen, python or sys.executable
        self._lock = threading.Lock()
        self._task: Optional[dict] = None
        self._proc = None
        self._stop = threading.Event()

    def status(self) -> dict:
        with self._lock:
            return dict(self._task) if self._task else {"state": "idle"}

    def busy(self) -> bool:
        with self._lock:
            return bool(self._task and self._task["state"] in ("running", "hashing"))

    def _begin(self, task: dict) -> None:
        with self._lock:
            if self._task and self._task["state"] in ("running", "hashing"):
                raise DownloadError("another download or check is still running; wait for it or cancel it")
            self._stop = threading.Event()
            self._task = {"id": f"{int(time.time() * 1000)}", "started": time.time(), **task}

    def _set(self, **kw) -> None:
        with self._lock:
            if self._task:
                self._task.update(kw)

    # -- datasets: fingerprint in the background
    def hash_dataset(self, name, path, note: str = "", source: Optional[dict] = None, verify_only: bool = False, version=None) -> dict:
        dataversions._check_name(name)
        self._begin({"kind": "hash", "state": "hashing", "name": name, "total": 0, "done": 0, "message": "reading the files to fingerprint them"})
        stop = self._stop

        def work():
            try:
                if verify_only:
                    r = dataversions.verify(name, version, progress=lambda n: self._set(done=n), stop=stop)
                    self._set(state="done", result=r, message=f"{name} v{r['version']}: {r['status']}", finished=time.time())
                else:
                    root = dataversions._folder(path)
                    total = sum(f.stat().st_size for f in dataversions._files(root))
                    self._set(total=total)
                    v = dataversions.add(name, str(root), note, source, progress=lambda n: self._set(done=n), stop=stop)
                    self._set(state="done", result={"name": name, "version": v["version"], "new": v["new"]}, finished=time.time(),
                              message=f"{name} v{v['version']} " + ("registered" if v["new"] else "was already registered (same content)"))
            except dataversions.DataError as e:
                self._set(state="cancelled" if str(e) == "stopped" else "failed", message=str(e), finished=time.time())
            except Exception as e:
                self._set(state="failed", message=_clean(str(e))[:400], finished=time.time())

        threading.Thread(target=work, daemon=True).start()
        return self.status()

    # -- downloads
    def start_download(self, kind, repo, revision, target, confirm_large: bool = False, name: str = "",
                       info_fn: Optional[Callable] = None) -> dict:
        info = preview(kind, repo, revision, self._env, info_fn)
        dest = check_target(target)
        if info["large"] and not confirm_large:
            raise DownloadError(f"this is {info['bytes'] / 2**30:.0f} GB; confirm that you want a download this large on this machine")
        free = shutil.disk_usage(dest if dest.exists() else _existing_parent(dest)).free
        if info["bytes"] and free < info["bytes"] * 1.05:
            raise DownloadError(f"there is not enough space in {dest}: {info['bytes'] / 2**30:.1f} GB needed, {free / 2**30:.1f} GB free")
        dest.mkdir(parents=True, exist_ok=True)
        self._begin({"kind": "download", "state": "running", "repo": info["repo"], "revision": info["revision"], "what": info["kind"], "target": str(dest),
                     "total": info["bytes"], "done": _tree_bytes(dest), "message": "starting"})
        env = dict(self._env)
        tok = token(self._env)
        if tok:
            env["HF_TOKEN"] = tok                                        # through the environment only, never a command line
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        cmd = [self._python, "-m", "bhaskera.downloads", "run", "--kind", info["kind"], "--repo", info["repo"], "--revision", info["revision"], "--target", str(dest)]
        try:
            self._proc = self._popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        except OSError as e:
            self._set(state="failed", message=f"could not start the download: {e}", finished=time.time())
            return self.status()
        stop = self._stop
        threading.Thread(target=self._watch, args=(dest, info, name), kwargs={"stop": stop}, daemon=True).start()
        return self.status()

    def _watch(self, dest: Path, info: dict, name: str, stop: threading.Event) -> None:
        proc, tail = self._proc, collections.deque(maxlen=12)

        def read():
            for line in proc.stdout:
                tail.append(line.rstrip())

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        while proc.poll() is None:
            t0 = time.monotonic()
            self._set(done=_tree_bytes(dest), message="downloading")
            time.sleep(max(2.0, 4 * (time.monotonic() - t0)))          # a folder of very many files is counted less often
        reader.join(timeout=2)
        if stop.is_set():
            self._set(state="cancelled", message="cancelled; start the same download again to continue from where it stopped", finished=time.time(), done=_tree_bytes(dest))
            return
        if proc.returncode != 0:
            self._set(state="failed", message=_clean(" ".join(list(tail)[-3:]))[:400] or f"the download stopped (exit code {proc.returncode})", finished=time.time())
            return
        rec = {"kind": info["kind"], "repo": info["repo"], "revision": info["revision"], "path": str(dest), "bytes": _tree_bytes(dest), "date": int(time.time())}
        try:
            if info["kind"] == "model" and not (dest / "config.json").is_file():
                self._set(state="failed", message="the download finished but the folder has no config.json: it is not a Hugging Face model folder", finished=time.time())
                return
            _remember(rec)
            if info["kind"] == "dataset":
                ds = re.sub(r"[^A-Za-z0-9._-]+", "-", name or info["repo"].split("/")[-1]).strip("-.")[:64] or "dataset"
                self._set(state="hashing", message="downloaded; fingerprinting the files", total=rec["bytes"], done=0)
                v = dataversions.add(ds, str(dest), f"Downloaded from Hugging Face: {info['repo']} ({info['revision']})",
                                     {"type": "huggingface", "repo": info["repo"], "revision": info["revision"], "sha": info.get("sha", "")},
                                     progress=lambda n: self._set(done=n), stop=stop)
                self._set(state="done", finished=time.time(), result={"path": str(dest), "dataset": ds, "version": v["version"]},
                          message=f"downloaded to {dest} and registered as {ds} v{v['version']}")
            else:
                self._set(state="done", finished=time.time(), result={"path": str(dest)}, message=f"downloaded to {dest}")
        except Exception as e:
            self._set(state="failed", message=_clean(str(e))[:400], finished=time.time())

    def cancel(self) -> dict:
        with self._lock:
            running = bool(self._task and self._task["state"] in ("running", "hashing"))
            proc = self._proc
        if not running:
            return self.status()
        self._stop.set()
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                pass
        return self.status()


def _existing_parent(p: Path) -> Path:
    while not p.exists():
        p = p.parent
    return p


# ---------------------------------------------------------------- the process that downloads
def _run(args) -> int:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=args.repo, repo_type=args.kind, revision=args.revision, local_dir=args.target, max_workers=int(MAX_WORKERS),
                      token=os.environ.get("HF_TOKEN") or None)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m bhaskera.downloads", description="Download a model or dataset from Hugging Face (also used by the home page).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="download into a folder")
    r.add_argument("--kind", choices=KINDS, required=True)
    r.add_argument("--repo", required=True)
    r.add_argument("--revision", default="main")
    r.add_argument("--target", required=True)
    args = ap.parse_args(argv)
    try:
        check_repo(args.kind, args.repo, args.revision)
        check_target(args.target)
        return _run(args)
    except DownloadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"error: {_failure(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
