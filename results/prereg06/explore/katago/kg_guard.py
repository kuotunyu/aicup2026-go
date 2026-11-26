"""Guard for the prereg06 katago scripts: any open / listdir / scandir / stat under data/local raises.

Same idea as results/prereg05/explore/u1/u1_lib.install_guard, copied here so the katago scripts do not import
the prereg04 fusion modules. Stdlib only.
"""

import builtins
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
_FORBIDDEN = os.path.normcase(os.path.abspath(os.path.join(REPO, "data", "local")))


class LocalDataRefused(RuntimeError):
    pass


def _forbidden(p):
    try:
        if isinstance(p, int):
            return False
        s = os.path.normcase(os.path.abspath(os.fspath(p)))
    except TypeError:
        return False
    return s == _FORBIDDEN or s.startswith(_FORBIDDEN + os.sep) or ("data" + os.sep + "local") in s


def install_guard():
    if getattr(builtins, "_c10_kg_guard", False):
        return
    real_open, real_listdir, real_scandir, real_stat = builtins.open, os.listdir, os.scandir, os.stat

    def g_open(file, *a, **k):
        if _forbidden(file):
            raise LocalDataRefused(f"katago guard: refusing to open {file}")
        return real_open(file, *a, **k)

    def g_listdir(path="."):
        if _forbidden(path):
            raise LocalDataRefused(f"katago guard: refusing to list {path}")
        return real_listdir(path)

    def g_scandir(path="."):
        if _forbidden(path):
            raise LocalDataRefused(f"katago guard: refusing to scan {path}")
        return real_scandir(path)

    def g_stat(path, *a, **k):
        if _forbidden(path):
            raise LocalDataRefused(f"katago guard: refusing to stat {path}")
        return real_stat(path, *a, **k)

    builtins.open = g_open
    io.open = g_open
    os.listdir = g_listdir
    os.scandir = g_scandir
    os.stat = g_stat
    builtins._c10_kg_guard = True
