"""Which data a tokenised cache was made from: a fingerprint of the input files, mixed into the cache's name.

The cache name used to depend only on the model, sequence length, format and settings, so tokenising a *different* file with the same settings
into the same cache folder silently reused the old cache. Now the names, sizes and modification times of the input files are part of the name:
different or edited data gets its own cache, the same data finds the one it made. (Reading the files to hash them would be slow for large data;
the sizes and times change whenever a file is edited, replaced or added.)
"""
from __future__ import annotations

import hashlib
import os
from typing import Iterable


def files_signature(paths: Iterable[str]) -> str:
    """16 hex characters for a list of input files and folders (a folder counts every file inside it), or '' when there are none."""
    h = hashlib.sha256()
    n = 0
    for p in sorted(str(x) for x in paths):
        if os.path.isdir(p):
            for dirpath, dirnames, names in os.walk(p):
                dirnames.sort()
                for name in sorted(names):
                    f = os.path.join(dirpath, name)
                    st = os.stat(f)
                    h.update(f"{os.path.relpath(f, p)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
                    n += 1
        else:
            st = os.stat(p)
            h.update(f"{os.path.basename(p)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
            n += 1
    return h.hexdigest()[:16] if n else ""
