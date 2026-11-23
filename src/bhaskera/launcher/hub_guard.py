"""Guards for the home page and the proxy behind it (ray_hub.py).

The hub runs on a login node that many users share, and it is opened from a laptop browser that also visits other
web sites. Three things protect it:

* **Same user only.** A loopback port is not private on a shared machine: any user of the node can connect to it. Each
  connection is therefore accepted only when the process on the other end belongs to the user who started the hub
  (read from /proc/net/tcp; the SSH forward of that user's own login qualifies). If that cannot be told, the
  connection is refused.
* **Known addresses only.** Requests whose Host header is not localhost, 127.0.0.1 or one of the hub's own names
  (``job-<id>.localhost``, ``grafana-<id>.localhost``, ``mlflow.localhost``) are refused, so a web site cannot reach
  the hub by pointing its own name at 127.0.0.1 (DNS rebinding). Requests that change something must come from one of
  those addresses too (Origin check), and the Ray Dashboard is read-only through the hub: its job API can run
  commands on the cluster.
* **Limits.** At most 64 connections at a time and 10 MB per request body.
"""
from __future__ import annotations

import os
import re
import socket
from typing import Optional

ALLOWED_HOST = re.compile(r"^(localhost|127\.0\.0\.1|\[::1\]|job-\d+\.localhost|grafana-\d+\.localhost|mlflow\.localhost)(:\d{1,5})?$", re.IGNORECASE)
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")
MAX_BODY = 10 * 1024 * 1024
MAX_CHAT_BODY = 1024 * 1024
MAX_FORM_BODY = 16 * 1024
MAX_CONNECTIONS = 64

# Added to every page and answer the hub itself produces (not to what it relays from Ray, Grafana or MLflow).
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; "
                                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"),
}


def host_allowed(host: str) -> bool:
    return bool(ALLOWED_HOST.match((host or "").strip()))


def origin_allowed(origin: str) -> bool:
    """An Origin header ('http://job-7.localhost:8265') that names one of the hub's own addresses."""
    m = re.match(r"^https?://(.+)$", (origin or "").strip())
    return bool(m and host_allowed(m.group(1)))


def method_allowed(kind: str, method: str) -> bool:
    """The Ray Dashboard is read-only through the hub; MLflow and Grafana need POST to read their data."""
    return method in SAFE_METHODS or kind != "job"


def safe_link(url: str) -> str:
    """The url when it is a plain http(s) link, else an empty string (no javascript: or data: links)."""
    return url if re.match(r"^https?://", url or "") else ""


def _hex_addr(ip: str, port: int) -> str:
    return "%08X:%04X" % (int.from_bytes(socket.inet_aton(ip), "little"), port)


def uid_of_connection(proc_net_tcp: str, client: tuple, server: tuple) -> Optional[int]:
    """The owner of the client end of a TCP connection on this machine, from the text of /proc/net/tcp; None when the
    connection is not listed."""
    want_local, want_remote = _hex_addr(*client), _hex_addr(*server)
    for line in proc_net_tcp.splitlines()[1:]:
        fields = line.split()
        if len(fields) > 7 and fields[1].upper() == want_local and fields[2].upper() == want_remote:
            return int(fields[7])
    return None


def same_user_peer(client: tuple, server: tuple, uid: Optional[int] = None, proc: str = "/proc/net/tcp") -> bool:
    """Whether the connection comes from a process of this user. Without /proc (not Linux) there is nothing to check
    and the connection is allowed; with /proc but no answer it is refused."""
    uid = os.getuid() if uid is None else uid
    try:
        with open(proc) as f:
            text = f.read()
    except OSError:
        return True
    try:
        owner = uid_of_connection(text, (client[0], client[1]), (server[0], server[1]))
    except (OSError, ValueError):
        return False
    return owner == uid
