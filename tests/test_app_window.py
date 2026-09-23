"""The explorer as a standalone app: its own window, its own icon.

Covers the seam between ``--app`` and the browser, not the browser itself:
which argv gets built, what happens on a machine with no Chromium, and that the
page really advertises an icon and a manifest. Whether the window decoration
and the taskbar entry come out right is the browser's business and can only be
seen by looking at a screen.

Runs under pytest, and standalone (``python tests/test_app_window.py``) for
environments without it. Dash is imported only inside the tests that need it.
"""

import io
import os
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unichart import cli
from unichart import terminal
from unichart import UnichartNotebook

warnings.filterwarnings('ignore')


class _StubAppBase:
    """Stands in for the Dash app: built, never really run."""

    def run(self, **kwargs):
        pass


def _terminal(in_notebook=False, **kwargs):
    """Run ``terminal`` against a stub app, returning what it would launch.

    The launcher is captured rather than called — the point of these tests is
    the argv, and actually starting a browser would open windows on whoever is
    running the suite. ``in_notebook`` fakes a Jupyter kernel, which
    ``terminal`` asks about through ``dashboard._in_notebook``.
    """
    captured = {'launched': None, 'timer': False}

    def fake_launcher(url, app_window):
        captured['url'] = url
        captured['app_window'] = app_window
        return lambda: captured.update(launched=url)

    class _FakeTimer:
        def __init__(self, delay, fn):
            self.fn = fn

        def start(self):
            captured['timer'] = True
            self.fn()

    class _StubApp(_StubAppBase):
        def run(self, **run_kwargs):
            captured['run'] = run_kwargs

    def fake_build(notebook, title=None, startup=(), allow_quit=None):
        captured['allow_quit'] = allow_quit
        return _StubApp()

    from unichart import dashboard
    originals = (terminal.build_terminal_app,
                 terminal._browser_launcher,
                 terminal.threading.Timer,
                 dashboard._in_notebook)
    terminal.build_terminal_app = fake_build
    terminal._browser_launcher = fake_launcher
    terminal.threading.Timer = _FakeTimer
    dashboard._in_notebook = lambda: in_notebook
    try:
        with redirect_stdout(io.StringIO()):
            terminal.terminal(nb=UnichartNotebook(), **kwargs)
    finally:
        (terminal.build_terminal_app,
         terminal._browser_launcher,
         terminal.threading.Timer,
         dashboard._in_notebook) = originals
    return captured


# ---------------------------------------------------------------------------
# The flag
# ---------------------------------------------------------------------------

def test_app_flag_reaches_explore(tmp_path):
    """--app is a board option: it has to survive the trip through the CLI."""
    from unichart import dashboard

    csv = tmp_path / 'runs.csv'
    csv.write_text('t,temp\n0,20\n1,21\n')

    seen = {}
    original = dashboard.explore
    dashboard.explore = lambda notebook=None, **kw: seen.update(kw)
    try:
        with redirect_stdout(io.StringIO()):
            assert cli.main([str(csv), '--app']) == 0
        assert seen['app_window'] is True
        assert seen['open_browser'] is True

        seen.clear()
        with redirect_stdout(io.StringIO()):
            assert cli.main([str(csv)]) == 0
        assert seen['app_window'] is False, 'a tab stays the default'
    finally:
        dashboard.explore = original


def test_no_browser_wins_over_app(tmp_path):
    """--no-browser is the headless flag; asking for a window can't undo it."""
    captured = _terminal(app_window=True, open_browser=False, port=0)
    assert captured['launched'] is None
    assert captured['timer'] is False


def test_app_window_opens_one(tmp_path):
    captured = _terminal(app_window=True, port=0)
    assert captured['app_window'] is True
    assert captured['launched'].startswith('http://127.0.0.1:')


# ---------------------------------------------------------------------------
# The launcher
# ---------------------------------------------------------------------------

def test_launcher_builds_an_app_mode_command(tmp_path):
    """``--app=URL`` is what makes the window chrome-less; the size is ours."""
    spawned = {}
    originals = (terminal._app_browser, terminal._spawn)
    terminal._app_browser = lambda: '/usr/bin/chromium'
    terminal._spawn = lambda argv, url: spawned.update(argv=argv)
    try:
        with redirect_stdout(io.StringIO()) as out:
            terminal._browser_launcher('http://127.0.0.1:8050/', True)()
    finally:
        (terminal._app_browser, terminal._spawn) = originals

    width, height = terminal.APP_WINDOW_SIZE
    assert spawned['argv'] == ['/usr/bin/chromium',
                              '--app=http://127.0.0.1:8050/',
                              f'--window-size={width},{height}']
    assert out.getvalue() == '', 'nothing to report when it worked'


def test_launcher_falls_back_to_a_tab_and_says_so(tmp_path):
    """A Firefox-only machine still gets a board, plus a line explaining why."""
    opened = {}
    originals = (terminal._app_browser,
                 terminal.webbrowser.open)
    terminal._app_browser = lambda: None
    terminal.webbrowser.open = lambda url: opened.update(url=url)
    try:
        out = io.StringIO()
        with redirect_stdout(out):
            terminal._browser_launcher('http://127.0.0.1:8050/', True)()
    finally:
        (terminal._app_browser,
         terminal.webbrowser.open) = originals

    assert opened['url'] == 'http://127.0.0.1:8050/'
    assert 'UNICHART_APP_BROWSER' in out.getvalue()


def test_app_browser_env_override(tmp_path):
    """The escape hatch for a Chromium build the search doesn't know."""
    exe = tmp_path / 'my-chrome'
    exe.write_text('#!/bin/sh\n')
    original = os.environ.get('UNICHART_APP_BROWSER')
    os.environ['UNICHART_APP_BROWSER'] = str(exe)
    try:
        assert terminal._app_browser() == str(exe)
        os.environ['UNICHART_APP_BROWSER'] = str(tmp_path / 'nope')
        assert terminal._app_browser() is None, \
            'a bad override is an error to see, not a silent fallback'
    finally:
        if original is None:
            del os.environ['UNICHART_APP_BROWSER']
        else:
            os.environ['UNICHART_APP_BROWSER'] = original


def test_a_bad_override_is_named_back(tmp_path):
    """Telling someone to set the variable they just set is no help at all."""
    original = os.environ.get('UNICHART_APP_BROWSER')
    os.environ['UNICHART_APP_BROWSER'] = str(tmp_path / 'nope')
    opened = {}
    open_original = terminal.webbrowser.open
    terminal.webbrowser.open = lambda url: opened.update(url=url)
    try:
        out = io.StringIO()
        with redirect_stdout(out):
            terminal._browser_launcher('http://127.0.0.1:8050/', True)()
    finally:
        terminal.webbrowser.open = open_original
        if original is None:
            del os.environ['UNICHART_APP_BROWSER']
        else:
            os.environ['UNICHART_APP_BROWSER'] = original

    assert 'nope' in out.getvalue(), 'the value that failed is the answer'
    assert opened['url'] == 'http://127.0.0.1:8050/'


# ---------------------------------------------------------------------------
# In a kernel
# ---------------------------------------------------------------------------

def test_a_window_from_a_kernel_means_the_external_board(tmp_path):
    """An app window is not an inline iframe, so asking for one leaves it."""
    captured = _terminal(in_notebook=True, app_window=True, port=0)
    assert captured['run']['jupyter_mode'] == 'external'
    assert captured['launched'] is not None
    assert captured['run']['host'] == '127.0.0.1'


def test_inline_still_wins_when_asked_for_explicitly(tmp_path):
    """The iframe resolves its own URL: pinning a host there blanks it."""
    captured = _terminal(in_notebook=True, app_window=True, port=0,
                         jupyter_mode='inline')
    assert captured['launched'] is None
    assert 'host' not in captured['run']
    assert captured['run']['jupyter_height'] == 860


def test_a_kernel_without_the_flag_is_unchanged(tmp_path):
    captured = _terminal(in_notebook=True, port=0)
    assert captured['run']['jupyter_mode'] == 'inline'
    assert captured['launched'] is None


# ---------------------------------------------------------------------------
# The icon
# ---------------------------------------------------------------------------

def test_the_board_serves_its_own_icon_and_manifest(tmp_path):
    """What gives the standalone window a face instead of Dash's plotly mark."""
    with redirect_stdout(io.StringIO()):
        app = terminal.build_terminal_app(
            UnichartNotebook(), title='Runs', banner=False)
    client = app.server.test_client()

    head = client.get('/').get_data(as_text=True).split('</head>')[0]
    assert 'href="/_unichart/icon.svg"' in head
    assert 'rel="manifest"' in head
    assert '{%favicon%}' not in head and 'favicon.ico' not in head, \
        'Dash\'s own icon is replaced, not competed with'

    icon = client.get('/_unichart/icon.svg')
    assert icon.status_code == 200
    assert icon.headers['Content-Type'].startswith('image/svg+xml')

    manifest = client.get('/_unichart/manifest.webmanifest')
    assert manifest.status_code == 200
    body = manifest.get_json()
    assert body['name'] == 'Runs', 'the window is named after the board'
    assert body['display'] == 'standalone'
    assert body['icons'][0]['src'].endswith('/_unichart/icon.svg')

    assert client.get('/_dash-layout').status_code == 200, \
        'the extra routes leave Dash alone'


# ---------------------------------------------------------------------------
# Standalone runner, for environments without pytest
# ---------------------------------------------------------------------------

def _main():
    import tempfile
    import traceback

    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failed = []
    for name, fn in tests:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fn(Path(tmp))
            except Exception:                                 # noqa: BLE001
                failed.append(name)
                print(f'FAIL  {name}')
                traceback.print_exc()
            else:
                print(f'PASS  {name}')
    print(f'\n{len(tests) - len(failed)}/{len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(_main())
