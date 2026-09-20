"""The file browser a naked ``load()`` opens, and what happens when there is none.

Covers ``_pick_files``' fallbacks rather than any real dialog: no test here puts
a window on a screen. The case that forced them is a Jupyter kernel started by
the VS Code Flatpak — its runtime python ships no Tk at all, so ``import
tkinter`` raises and the old code dead-ended on advice ('apt install
python3-tk') that installs into a host filesystem the kernel cannot see. The
same runtime does carry zenity, so a subprocess still reaches a real dialog.

Runs under pytest, and standalone (``python tests/test_file_picker.py``) for
environments without it.
"""

import os
import subprocess
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unichart
from unichart import UnichartNotebook

warnings.filterwarnings('ignore')


class _Completed:
    """Stands in for subprocess.CompletedProcess."""

    def __init__(self, returncode, stdout='', stderr=''):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _Dialogs:
    """Pretend a chosen set of dialog programs exists, and answer for them.

    Patches shutil.which and subprocess.run for the duration of a ``with``
    block; ``calls`` records the argv each program was asked to run.
    """

    def __init__(self, available=(), answer=None, missing_tkinter=False):
        self.available = set(available)
        self.answer = answer or (lambda program, argv: _Completed(1))
        self.missing_tkinter = missing_tkinter
        self.calls = []

    def __enter__(self):
        import shutil
        self._which, self._run = shutil.which, subprocess.run
        shutil.which = lambda name, *a, **kw: (
            f'/usr/bin/{name}' if name in self.available else None)
        subprocess.run = self._fake_run
        if self.missing_tkinter:
            self._saved_modules = {k: sys.modules.get(k)
                                   for k in ('tkinter', 'tkinter.filedialog')}
            sys.modules['tkinter'] = None          # import -> ImportError
            sys.modules['tkinter.filedialog'] = None
        return self

    def __exit__(self, *exc):
        import shutil
        shutil.which, subprocess.run = self._which, self._run
        if self.missing_tkinter:
            for name, module in self._saved_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        return False

    def _fake_run(self, argv, *a, **kw):
        program = Path(argv[0]).name
        self.calls.append(argv)
        return self.answer(program, argv)


def _env(**values):
    """Set env vars for a block, restoring whatever was there before."""

    class _Ctx:
        def __enter__(self):
            self.saved = {k: os.environ.get(k) for k in values}
            for k, v in values.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

        def __exit__(self, *exc):
            for k, v in self.saved.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
            return False

    return _Ctx()


# ---------------------------------------------------------------------------
# The fallback picker
# ---------------------------------------------------------------------------

def test_zenity_returns_picked_paths():
    picked = _Completed(0, '/data/a.csv\n/data/b.csv\n')
    with _Dialogs(available=['zenity'], answer=lambda p, argv: picked) as d:
        assert UnichartNotebook._pick_files_native() == ('/data/a.csv', '/data/b.csv')
    assert Path(d.calls[0][0]).name == 'zenity'


def test_zenity_separator_is_newline_not_pipe():
    # '|' is a legal filename character, so the default separator would split
    # paths that contain one.
    argv = UnichartNotebook._dialog_argv('zenity')
    assert '--separator=\n' in argv
    assert '--multiple' in argv
    assert any(a.startswith('--file-filter=Data files |') and '*.csv' in a for a in argv)


def test_cancel_is_not_failure():
    # rc 1 means the user cancelled: load nothing, quietly.
    with _Dialogs(available=['zenity'], answer=lambda p, argv: _Completed(1)):
        assert UnichartNotebook._pick_files_native() == ()


def test_falls_through_to_next_program_on_error():
    # rc 255 with GTK noise on stderr: zenity could not display, but the next
    # program might. stderr alone must not count as failure (see below).
    def answer(program, argv):
        if program == 'zenity':
            return _Completed(255, '', 'cannot open display')
        return _Completed(0, '/data/a.csv\n')

    with _Dialogs(available=['zenity', 'qarma'], answer=answer) as d:
        assert UnichartNotebook._pick_files_native() == ('/data/a.csv',)
    assert [Path(c[0]).name for c in d.calls] == ['zenity', 'qarma']


def test_only_zenity_compatible_programs_are_tried():
    # rc 1 is read as "cancelled", which is zenity's contract. A program that
    # exits 1 on a usage error would have it swallowed as a cancel, so nothing
    # joins this list without its CLI being checked first.
    assert UnichartNotebook._DIALOG_PROGRAMS == ('zenity', 'qarma')


def test_stderr_chatter_on_success_is_ignored():
    picked = _Completed(0, '/data/a.csv\n', 'Gtk-WARNING **: something')
    with _Dialogs(available=['zenity'], answer=lambda p, argv: picked):
        assert UnichartNotebook._pick_files_native() == ('/data/a.csv',)


def test_no_dialog_program_reports_none():
    # None, not (): "nothing to ask with" is not "user picked nothing".
    with _Dialogs(available=[]):
        assert UnichartNotebook._pick_files_native() is None


# ---------------------------------------------------------------------------
# _pick_files: tkinter first, then the fallback, then a useful error
# ---------------------------------------------------------------------------

def test_missing_tkinter_falls_back_to_dialog_program():
    picked = _Completed(0, '/data/a.csv\n')
    with _Dialogs(available=['zenity'], answer=lambda p, argv: picked,
                  missing_tkinter=True):
        assert UnichartNotebook._pick_files() == ('/data/a.csv',)


def test_missing_tkinter_and_no_dialog_raises():
    with _Dialogs(available=[], missing_tkinter=True):
        try:
            UnichartNotebook._pick_files()
        except RuntimeError as e:
            message = str(e)
        else:
            raise AssertionError('expected RuntimeError')
    assert 'tkinter is unavailable' in message
    assert "load('data.csv')" in message


# ---------------------------------------------------------------------------
# The error message, which has to fit the runtime it is printed in
# ---------------------------------------------------------------------------

def test_sandbox_message_does_not_advise_apt():
    # Inside the VS Code Flatpak, 'apt install python3-tk' installs where this
    # process will never see it.
    with _env(container='flatpak', DISPLAY=None, WAYLAND_DISPLAY='wayland-0'):
        message = UnichartNotebook._no_picker_message('tkinter is unavailable (x)')
    assert 'apt install' not in message
    assert 'sandbox' in message
    assert 'XWayland' in message           # DISPLAY unset, Wayland session


def test_host_message_advises_apt():
    with _env(container=None, DISPLAY=':0', WAYLAND_DISPLAY=None):
        message = UnichartNotebook._no_picker_message('tkinter is unavailable (x)')
    assert 'python3-tk' in message
    assert 'sandbox' not in message
    assert 'XWayland' not in message


def test_message_quotes_the_underlying_error():
    # The old code swallowed it with 'from None'; "no module" and "libtk8.6.so
    # is unloadable" need different fixes.
    with _env(container=None, DISPLAY=':0', WAYLAND_DISPLAY=None):
        message = UnichartNotebook._no_picker_message("boom (libtk8.6.so missing)")
    assert 'libtk8.6.so missing' in message


def _main():
    import traceback
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception:                                     # noqa: BLE001
            failed.append(name)
            print(f'FAIL  {name}')
            traceback.print_exc()
        else:
            print(f'PASS  {name}')
    print(f'\n{len(tests) - len(failed)}/{len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(_main())
