"""unichart — a plotting and notebook dashboard tool built on Plotly.

    from unichart import UnichartNotebook

The plotting core lives in :mod:`unichart._core` and loads on first use: every
public name (``UnichartNotebook``, ``sniff_session``, ...) resolves through the
module ``__getattr__`` below. That keeps ``import unichart.cli`` — and so
``unichart --help`` / ``--version`` — free of the ~1 s pandas/Plotly import.

Submodules: :mod:`unichart.dashboard` (Dash boards), :mod:`unichart.terminal`
(the explorer GUI) and :mod:`unichart.cli` (the ``unichart`` command).

Code that *patches* a core global (``display``, ``_HELP_COLOR``) must patch
``unichart._core``: setting it on this package would shadow the name here
without reaching the functions that read it.
"""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING

try:
    from importlib.metadata import PackageNotFoundError, version as _version
    __version__ = _version('unichart')
except PackageNotFoundError:            # imported from a source tree, not installed
    __version__ = '0.0.0+unknown'
del PackageNotFoundError, _version

if TYPE_CHECKING:                       # let editors see the lazy names
    from ._core import *  # noqa: F401,F403


def _load_core():
    return _import_module('._core', __name__)


def __getattr__(name):
    if name == '__all__':
        # For `from unichart import *`: what _core defines, not what it imports.
        core = _load_core()
        return [n for n, v in vars(core).items()
                if not n.startswith('_') and not isinstance(v, type(core))
                and getattr(v, '__module__', core.__name__) == core.__name__]
    try:
        return getattr(_load_core(), name)
    except AttributeError:
        raise AttributeError(
            f"module 'unichart' has no attribute {name!r}") from None


def __dir__():
    return sorted(set(globals()) | set(dir(_load_core())))
