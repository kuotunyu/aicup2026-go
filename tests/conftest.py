"""Keep the import-time data/local guards inside the test modules that use them (one pytest process = one process per
test file, as far as the guards go).

tools/p06_player.py (through tools/p06_style.install_guard) and results/prereg06/explore/c1_headroom/c1h_lib.py install
a process-wide guard when they are imported: builtins.open, io.open, io.FileIO, os.open / stat / lstat / listdir /
scandir are wrapped and never restored. Run as one process per test file nobody noticed. In a single pytest process
(the readiness list of docs/TESTDAY.md: `PY -m pytest -q tests`) the first module importing one of them, already at
collection, guarded every module after it, and the tests that read the validation game files or synthesise label-named
files failed (test06 build_review, section 2.6: 12 failed, 102 errors; review 10/3).

Rule here: a test module runs under a guard exactly when its own import graph (module globals and the modules of
names imported from them, inside this repository) reaches a module that installs it at import; every other module
runs with the original functions. Concretely:
- after each test module is collected (imported), the original functions come back and the guard flags are cleared;
- before a module's tests, the guards it reaches are installed again by the tools' own install_guard() (nothing is
  re-implemented, no tool file is changed); after them, the originals come back.
A guard a test installs while it runs (an explicit install_guard(), the first import of such a tool inside a test)
stays for the rest of that module, as in its own process, and is removed after it.
"""

import builtins
import io
import os
import sys
import types

import pytest

_REPO = os.path.normcase(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OWN = tuple(os.path.join(_REPO, d) + os.sep for d in ("tools", "tests", "results", "colab"))   # never .venv
_SLOTS = ((builtins, "open"), (io, "open"), (io, "FileIO"), (os, "open"), (os, "stat"), (os, "lstat"),
          (os, "listdir"), (os, "scandir"))
_INSTALLERS = {   # module installing a guard at import -> (its builtins flag, module holding install_guard)
    "p06_player": ("_c10_p06_guard", "p06_style"),
    "c1h_lib": ("_c10_c1h_guard", "c1h_lib")}
_FLAGS = ("_c10_p06_guard", "_c10_c1h_guard", "_c10_p06_cnn_guard")
_PRISTINE = tuple(getattr(m, n) for m, n in _SLOTS)     # conftest is imported before any test module


def _reset():
    for (m, n), fn in zip(_SLOTS, _PRISTINE):
        setattr(m, n, fn)
    for f in _FLAGS:
        if hasattr(builtins, f):
            delattr(builtins, f)


def _in_repo(mod):
    f = getattr(mod, "__file__", None)
    return bool(f) and os.path.normcase(os.path.abspath(f)).startswith(_OWN)


def _reached(mod, seen):
    """Names of the guard-installing repository modules reachable from mod (module globals, and the modules that
    define the functions / classes imported by name)."""
    out = set()
    for v in list(vars(mod).values()):
        m = v
        if not isinstance(v, types.ModuleType):
            try:
                m = sys.modules.get(getattr(v, "__module__", None) or "")
            except Exception:  # noqa: BLE001  (an exotic object without a usable __module__)
                m = None
        if not isinstance(m, types.ModuleType) or id(m) in seen or not _in_repo(m):
            continue
        seen.add(id(m))
        if m.__name__ in _INSTALLERS:
            out.add(m.__name__)
        out |= _reached(m, seen)
    return out


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    yield
    if isinstance(collector, pytest.Module):
        _reset()                     # whatever its import installed never guards the next module's collection


@pytest.fixture(autouse=True, scope="module")
def _c10_import_guards(request):
    _reset()
    mod = request.module
    for name in sorted(_reached(mod, {id(mod)}) | ({mod.__name__} & set(_INSTALLERS))):
        installer = sys.modules.get(_INSTALLERS[name][1])
        if installer is not None and hasattr(installer, "install_guard"):
            installer.install_guard()
    yield
    _reset()
