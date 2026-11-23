# Security

Bhaskera runs on shared clusters, where other people can log in to the same machines. This page says what the
framework protects, what it assumes, and how to report a problem.

## What is protected

| Part | Where it listens | How it is kept private |
|---|---|---|
| **Home page and proxy** (`bhaskera-dashboard ray`, on the login node) | `127.0.0.1`, a port derived from your user id | Only connections from **your own user** are accepted (checked per connection; your SSH tunnel counts). Requests for any other host name are refused, so a web site cannot reach it by pointing its name at your laptop (DNS rebinding). Requests that change something must come from the hub's own addresses. The Ray Dashboard is read-only through it. 64 connections and 10 MB per request at most. |
| **MLflow page** (`bhaskera-dashboard start`) | `127.0.0.1` | MLflow has no login, so it is not exposed on the network. A different `--host` prints a warning. |
| **Ray Dashboard** (`monitoring.dashboard_host`) | `127.0.0.1` | Its job API can run commands, so it is not exposed on the network. |
| **Prometheus and Grafana** of a job | `127.0.0.1` of the job's first node | Reached only through the hub's SSH link. |
| **Chat server** (`./submit.sh serve`) | `127.0.0.1` of the job's first node | Serves only connections from the user who started it, and only for `localhost` host names. The home page reaches it through its SSH link; the chat request must come from the Bhaskera page itself (same-site `Origin` and a custom header), and is limited to 1 MB, 50 messages and 1024 new tokens. |
| **Start buttons** (`/api/launch`) | same hub | The page sends a form, not a command. Same-user, host and Origin checks as the chat route, plus the custom header; 16 KB per request. `submit.sh` is run with an argument list (no shell). Recipes must be files in the folders the hub lists itself; nodes, hours and example limits are bounded whole numbers; run ids must belong to a run with a stored adapter; benchmark names allow letters, digits, `_`, `-` and commas. `SKIP_QOS_CHECK` and `SKIP_PREFLIGHT` are removed from the environment, one start runs at a time, and at most 12 starts per hour. `BHASKERA_HUB_LAUNCH=0` turns it off. |
| **Job logs** shown on the home page | `/api/log` | Only logs of jobs owned by the user who started the hub, found through Slurm (the log path is never taken from the request). |

## What it assumes

- Nodes of a job are not shared with other users (`--exclusive`). On a cluster where nodes are shared, other users on
  the first node can reach the job's Ray Dashboard, Prometheus and Grafana on its loopback address.
- The login node's `/proc/net/tcp` is readable. Without `/proc` (not Linux) the same-user check has nothing to look at
  and passes; with `/proc` but no answer it refuses.
- Recipes and cluster profiles are yours: they are not treated as untrusted input.

## Things to know

- `model.trust_remote_code: true` runs the Python files that come with a model, with your rights. Use it only for models you
  trust, from a folder nobody else can write to. `bhaskera-preflight` warns about it.
- Checkpoints, logs and the MLflow store use your normal file permissions (umask). On a shared file system, keep them in a
  folder only you can read if the data is sensitive.
- Exported adapters and registered models are ordinary files; nothing is signed.

## Guardrails against mistakes

These do not stop an attacker; they stop common errors from wasting GPU hours or data:

- A recipe key that does nothing is reported (`BHASKERA_STRICT_CONFIG=1` makes it an error).
- `submit.sh` checks the partition's time and job limits, and `bhaskera-preflight` checks memory, caches and kernels before
  anything is queued; every node is checked before training starts.
- Checkpoint retention keeps the newest checkpoints and never removes the newest one.
- Sweeps are limited to 64 trials and queue only what the partition allows.
- A faulty node found by a job is skipped by that user's next jobs for three days.

## Reporting a problem

Please open a private security advisory on the GitHub repository (Security tab), or email the maintainer named there, and
do not post details in a public issue until a fix is out.
