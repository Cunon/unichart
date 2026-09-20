"""Closing the board from the GUI.

The board owns the process it is served from, so its ✕ ends that process —
except inside a Jupyter kernel, where the process is the user's. These tests
cover the wiring around that: which controls are built, what the confirm
callback answers, that ``exit()`` in the terminal pane routes to the same
shutdown, and that nothing kills the process without a confirmation.

The shutdown itself is stubbed throughout. A test that really called
``os._exit`` would take the test runner with it.

Runs under pytest, and standalone (``python tests/test_quit.py``). Dash is
imported only inside the tests that need it — the ``exit()`` and shutdown ones
do not — so the module loads without the optional extra.
"""

import io
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unichart_terminal
from unichart import UnichartNotebook

warnings.filterwarnings('ignore')


def _notebook():
    with redirect_stdout(io.StringIO()):
        return UnichartNotebook()


def _ids(component, found=None):
    """Every id in a built layout, as strings."""
    found = set() if found is None else found
    if component is None or isinstance(component, str):
        return found
    if isinstance(component, (list, tuple)):
        for child in component:
            _ids(child, found)
        return found
    ident = getattr(component, 'id', None)
    if ident:
        found.add(ident if isinstance(ident, str) else str(ident))
    return _ids(getattr(component, 'children', None), found)


def _app(allow_quit):
    with redirect_stdout(io.StringIO()):
        return unichart_terminal.build_terminal_app(_notebook(),
                                                    allow_quit=allow_quit)


def _recorded_shutdown():
    """Replace ``_shutdown`` with a recorder. Returns (calls, restore)."""
    calls = []
    original = unichart_terminal._shutdown

    def record(delay=0.4, code=0):
        calls.append((delay, code))

    unichart_terminal._shutdown = record
    return calls, lambda: setattr(unichart_terminal, '_shutdown', original)


# ---------------------------------------------------------------------------
# What gets built
# ---------------------------------------------------------------------------

def test_the_board_offers_a_close_button(tmp_path=None):
    ids = _ids(_app(True).layout)
    assert 'term-quit' in ids                      # the top bar's button
    assert {'term-quit-yes', 'term-quit-cancel'} <= ids   # and its dialog


def test_a_kernel_board_has_no_close_button(tmp_path=None):
    """Closing it would close the kernel, so it is not offered."""
    app = _app(False)
    assert not [i for i in _ids(app.layout) if 'quit' in i]
    assert not [k for k in app.callback_map if 'quit' in k]


def test_the_default_follows_where_we_are_running(tmp_path=None):
    import unichart_dashboard
    original = unichart_dashboard._in_notebook
    try:
        for in_notebook in (True, False):
            unichart_dashboard._in_notebook = lambda: in_notebook
            with redirect_stdout(io.StringIO()):
                app = unichart_terminal.build_terminal_app(_notebook())
            assert ('term-quit' in _ids(app.layout)) is not in_notebook
    finally:
        unichart_dashboard._in_notebook = original


def test_terminal_tells_the_board_whether_it_owns_the_process(tmp_path=None):
    """``terminal`` resolves the gate; the board does not guess a second time."""
    import unichart_dashboard
    seen = {}

    def fake_build(notebook, title=None, startup=(), allow_quit=None):
        seen['allow_quit'] = allow_quit
        return type('_Stub', (), {'run': lambda self, **kw: None})()

    originals = (unichart_terminal.build_terminal_app,
                 unichart_dashboard._in_notebook)
    unichart_terminal.build_terminal_app = fake_build
    try:
        for in_notebook in (True, False):
            unichart_dashboard._in_notebook = lambda: in_notebook
            with redirect_stdout(io.StringIO()):
                unichart_terminal.terminal(nb=_notebook(), open_browser=False)
            assert seen['allow_quit'] is not in_notebook
    finally:
        (unichart_terminal.build_terminal_app,
         unichart_dashboard._in_notebook) = originals


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------

def test_the_button_asks_before_it_closes(tmp_path=None):
    """One click shows the dialog. Nothing is shut down by it."""
    app = _app(True)
    calls, restore = _recorded_shutdown()
    try:
        answer = _fire(app, 'term-quit')
        assert answer == 'term-modal'              # shown, not hidden
        assert calls == []
    finally:
        restore()


def test_cancel_puts_the_dialog_away(tmp_path=None):
    app = _app(True)
    calls, restore = _recorded_shutdown()
    try:
        assert _fire(app, 'term-quit-cancel') == 'term-modal hidden'
        assert calls == []
    finally:
        restore()


def test_confirming_shuts_down_and_says_goodbye(tmp_path=None):
    app = _app(True)
    calls, restore = _recorded_shutdown()
    try:
        # The goodbye state ships with the page: once the process is going
        # there is no round trip left to fetch it with.
        assert _fire(app, 'term-quit-yes') == 'term-modal closing'
        assert len(calls) == 1
    finally:
        restore()


def _fire(app, triggered_id):
    """Run the close callback as if ``triggered_id`` had been clicked.

    Dash hands a callback its trigger through a context variable and serializes
    what comes back, so both ends are faked here: set the context, call the
    registered function, read the class name out of the response.
    """
    import json

    from dash._utils import AttributeDict

    handler = app.callback_map['term-quit-modal.className']['callback']
    answer = handler(
        1, 1, 1,
        outputs_list={'id': 'term-quit-modal', 'property': 'className'},
        callback_context=AttributeDict(
            updated_props={},
            triggered_inputs=[{'prop_id': f'{triggered_id}.n_clicks',
                               'value': 1}]))
    if isinstance(answer, str) and answer.lstrip().startswith('{'):
        answer = json.loads(answer)['response']['term-quit-modal']['className']
    return answer


# ---------------------------------------------------------------------------
# exit() in the terminal pane
# ---------------------------------------------------------------------------

def test_exit_closes_the_board(tmp_path=None):
    """And leaves a line in the transcript rather than a SystemExit traceback."""
    calls, restore = _recorded_shutdown()
    try:
        result = unichart_terminal.Session(_notebook()).run('exit()')
    finally:
        restore()
    assert result['error'] is None
    assert 'closing' in result['text']
    assert len(calls) == 1


def test_quit_is_the_same_door(tmp_path=None):
    calls, restore = _recorded_shutdown()
    try:
        assert unichart_terminal.Session(_notebook()).run('quit()')['error'] is None
    finally:
        restore()
    assert len(calls) == 1


def test_an_exit_code_is_the_process_exit_code(tmp_path=None):
    """``exit(3)`` means 3, the way it would at a prompt."""
    calls, restore = _recorded_shutdown()
    try:
        unichart_terminal.Session(_notebook()).run('exit(3)')
    finally:
        restore()
    assert [code for _, code in calls] == [3]


def test_exit_in_a_kernel_explains_itself_instead(tmp_path=None):
    calls, restore = _recorded_shutdown()
    try:
        session = unichart_terminal.Session(_notebook(), allow_quit=False)
        result = session.run('exit()')
    finally:
        restore()
    assert calls == []
    assert 'kernel' in result['text']
    assert result['error'] is None


def test_bare_exit_says_what_to_type(tmp_path=None):
    """``exit`` with no parens echoes a hint, the way Python's does."""
    result = unichart_terminal.Session(_notebook()).run('exit')
    assert 'exit()' in result['text']
    assert result['error'] is None


# ---------------------------------------------------------------------------
# The shutdown itself
# ---------------------------------------------------------------------------

def test_shutdown_exits_the_process_after_the_response(tmp_path=None):
    """Not on the calling thread: the response has to reach the browser."""
    timers, exits = [], []

    class _FakeTimer:
        def __init__(self, delay, fn):
            timers.append(delay)
            self.fn = fn

        def start(self):
            self.fn()

    originals = (unichart_terminal.threading.Timer, unichart_terminal.os._exit)
    unichart_terminal.threading.Timer = _FakeTimer
    unichart_terminal.os._exit = exits.append
    try:
        unichart_terminal._shutdown()
    finally:
        (unichart_terminal.threading.Timer,
         unichart_terminal.os._exit) = originals
    assert timers and timers[0] > 0
    assert exits == [0]

    exits.clear()
    unichart_terminal.threading.Timer = _FakeTimer
    unichart_terminal.os._exit = exits.append
    try:
        unichart_terminal._shutdown(code=3)
    finally:
        (unichart_terminal.threading.Timer,
         unichart_terminal.os._exit) = originals
    assert exits == [3]


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

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
