"""Sessions where the CLI and the explorer meet them.

Covers the seam rather than the session format itself: which files are
recognised as sessions, that a ``.json`` session now carries its plot call the
way a PNG always has, and that ``unichart``'s argument handling routes a session
away from ``nb.load``. The format's own round-trip guarantees live in
``demo_notebooks/session_save_load_demo.ipynb``.

Runs under pytest, and standalone (``python tests/test_session_cli.py``) for
environments without it. Dash is imported only inside the tests that need it, so
the rest run without the optional extra.
"""

import io
import json
import struct
import sys
import warnings
import zlib
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unichart_cli
from unichart import (UnichartNotebook, png_embed_text, read_png_session,
                      sniff_session)

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Fixtures, hand-rolled so the module runs without pytest too
# ---------------------------------------------------------------------------

def _frame():
    n1 = np.arange(20, dtype=float)
    return pd.DataFrame({'t': n1, 'temp': n1 * 1.5, 'press': n1 * 0.3})


def _notebook(tmp, plot=True):
    """A notebook loaded from a real csv, optionally with a plot drawn."""
    csv = tmp / 'runs.csv'
    _frame().to_csv(csv, index=False)
    with redirect_stdout(io.StringIO()):
        nb = UnichartNotebook()
        nb.load(str(csv))
        if plot:
            nb.plot('t', ['temp', 'press'])
    return nb, csv


def _blank_png():
    """The smallest valid PNG: signature, IHDR, IEND. No kaleido needed."""
    def chunk(ctype, payload):
        return (struct.pack('>I', len(payload)) + ctype + payload
                + struct.pack('>I', zlib.crc32(ctype + payload) & 0xffffffff))
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
            + chunk(b'IEND', b''))


class _Args:
    """Stands in for the argparse namespace ``_split_files`` reads."""

    def __init__(self, files, combine=False, set_col=None, name_col=None):
        self.files = [str(f) for f in files]
        self.combine, self.set_col, self.name_col = combine, set_col, name_col


# ---------------------------------------------------------------------------
# sniff_session
# ---------------------------------------------------------------------------

def test_sniff_session_recognises_every_shape(tmp_path):
    nb, csv = _notebook(tmp_path)
    session_json = tmp_path / 's.json'
    with redirect_stdout(io.StringIO()):
        nb.save_session(str(session_json))

    session_png = tmp_path / 'sess.png'
    session_png.write_bytes(png_embed_text(
        _blank_png(), 'unichart-session', session_json.read_text()))
    plain_png = tmp_path / 'plain.png'
    plain_png.write_bytes(_blank_png())
    data_json = tmp_path / 'data.json'
    _frame().to_json(data_json)

    assert sniff_session(session_json) == 'json'
    assert sniff_session(session_png) == 'png'
    # The three that must NOT be mistaken for sessions, or a data load breaks.
    assert sniff_session(data_json) is None
    assert sniff_session(plain_png) is None
    assert sniff_session(csv) is None
    # And nothing it is handed may raise, however broken.
    assert sniff_session(tmp_path / 'missing.json') is None
    assert sniff_session(b'\x00\x01\x02') is None
    assert sniff_session(None) is None

    # Bytes and paths must agree — the drop zone only ever has bytes.
    assert sniff_session(session_json.read_bytes()) == 'json'
    assert sniff_session(session_png.read_bytes()) == 'png'
    # A PNG that is a PNG but carries no session is not one.
    assert read_png_session(str(plain_png)) is None


# ---------------------------------------------------------------------------
# The plot call now rides along in JSON
# ---------------------------------------------------------------------------

def test_json_session_records_and_replays_the_plot_call(tmp_path):
    nb, _ = _notebook(tmp_path)
    path = tmp_path / 's.json'
    with redirect_stdout(io.StringIO()):
        nb.save_session(str(path))

    assert json.loads(path.read_text())['plot_call']['method'] == 'plot'

    with redirect_stdout(io.StringIO()):
        restored = UnichartNotebook()
        restored.load_session(str(path))
    assert restored.last_fig is not None, 'the plot should come back with the data'
    assert restored.last_x == 't'
    assert restored.last_y == ['temp', 'press']


def test_replay_survives_kwargs_json_cannot_round_trip(tmp_path):
    """Tuples, numpy scalars and hue columns are what str(o) would mangle."""
    nb, _ = _notebook(tmp_path, plot=False)
    with redirect_stdout(io.StringIO()):
        nb.plot('t', ['temp', 'press'], figsize=(10, 6), legend='above')
        path = tmp_path / 's.json'
        nb.save_session(str(path))

        restored = UnichartNotebook()
        restored.load_session(str(path))
    assert restored.last_fig is not None
    # A tuple comes back as a list; the plotters take either, and the figure is
    # the proof — not merely that the key survived.
    assert restored.last_fig.data


def test_unreplayable_plot_call_warns_instead_of_raising(tmp_path):
    """What _session_json_default's str(o) fallback does to a rich kwarg.

    A figsize written as its repr comes back as a string the plotter cannot do
    arithmetic on. The data and formatting are already restored by then, so that
    must cost a warning, not a traceback out of load_session.
    """
    nb, _ = _notebook(tmp_path)
    path = tmp_path / 's.json'
    with redirect_stdout(io.StringIO()):
        nb.save_session(str(path))

    session = json.loads(path.read_text())
    session['plot_call']['kwargs']['figsize'] = 'not a size'
    path.write_text(json.dumps(session))

    out = io.StringIO()
    with redirect_stdout(out):
        restored = UnichartNotebook()
        created = restored.load_session(str(path))     # must not raise
    assert len(created) == 1, 'the data still restores'
    assert 'could not replay' in out.getvalue()


def test_parms_whitelist_keeps_the_plotted_columns(tmp_path):
    """Recording the plot call also protects its columns from parms=."""
    nb, _ = _notebook(tmp_path)
    with redirect_stdout(io.StringIO()):
        nb.sets[0]['derived'] = nb.sets[0]['temp'] * 2    # force embedding
        path = tmp_path / 's.json'
        nb.save_session(str(path), parms=['derived'])
    frame = json.loads(path.read_text())['sets'][0]['source']['frame']
    for needed in ('t', 'temp', 'press'):
        assert needed in frame['columns'], f'{needed} is plotted and must survive'


# ---------------------------------------------------------------------------
# CLI argument handling
# ---------------------------------------------------------------------------

def test_split_files_separates_sessions_from_data(tmp_path):
    nb, csv = _notebook(tmp_path)
    session = tmp_path / 's.json'
    data_json = tmp_path / 'data.json'
    _frame().to_json(data_json)
    with redirect_stdout(io.StringIO()):
        nb.save_session(str(session))

    data, sessions = unichart_cli._split_files(
        _Args([csv, session, data_json]))
    assert data == [str(csv), str(data_json)]
    assert sessions == [str(session)]


def test_split_files_reports_missing_paths(tmp_path):
    try:
        unichart_cli._split_files(_Args([tmp_path / 'nope.csv']))
    except unichart_cli.CliError as exc:
        assert 'no such file' in str(exc)
    else:
        raise AssertionError('a missing FILE must be reported')


def test_read_flags_are_rejected_for_a_session_alone(tmp_path):
    nb, csv = _notebook(tmp_path)
    session = tmp_path / 's.json'
    with redirect_stdout(io.StringIO()):
        nb.save_session(str(session))

    for kwargs in ({'set_col': 't'}, {'name_col': 't'}, {'combine': True}):
        try:
            unichart_cli._split_files(_Args([session], **kwargs))
        except unichart_cli.CliError as exc:
            assert 'not sessions' in str(exc)
        else:
            raise AssertionError(f'{kwargs} should not be accepted alone')

    # With a data file on the line they have something to act on again.
    data, sessions = unichart_cli._split_files(
        _Args([csv, session], set_col='t'))
    assert data and sessions


def test_completion_methods_match_the_real_ones():
    """The check unichart_cli's own comment promises."""
    from unichart_dashboard import PLOT_METHODS
    assert unichart_cli.COMPLETION_METHODS == tuple(PLOT_METHODS)


def test_save_session_flag_is_completable():
    assert '--save-session' in unichart_cli.complete('unichart --save-')
    # A PNG must never be offered as a column source: _header_columns would
    # try to read it as a table.
    assert '.png' not in unichart_cli._DATA_SUFFIXES
    assert '.png' in unichart_cli._SESSION_SUFFIXES


def test_cli_routes_a_session_to_the_board_not_to_load(tmp_path, monkeypatch=None):
    """The serving path hands sessions to explore(), never to nb.load."""
    nb, _ = _notebook(tmp_path)
    session = tmp_path / 's.json'
    with redirect_stdout(io.StringIO()):
        nb.save_session(str(session))

    import unichart_dashboard
    seen = {}
    original = unichart_dashboard.explore

    def fake_explore(notebook=None, **kwargs):
        seen.update(kwargs)
        seen['sets'] = len(notebook.sets)
        return None

    unichart_dashboard.explore = fake_explore
    try:
        with redirect_stdout(io.StringIO()):
            code = unichart_cli.main([str(session)])
    finally:
        unichart_dashboard.explore = original

    assert code == 0
    assert seen['sessions'] == [str(session)]
    assert seen['sets'] == 0, 'the session is restored by the board, not before it'
    assert seen['dark'] is None, 'no --dark means the session keeps its theme'


def test_terminal_restores_sessions_as_startup_commands(tmp_path):
    """Order matters: the dark flip, then the session, then --dark, then panels.

    The session has to land after terminal()'s "match the board" dark flip or a
    light session loses its theme, and an explicit dark= has to land after the
    session or the flag loses to it.
    """
    import unichart_terminal
    nb, _ = _notebook(tmp_path, plot=False)
    session = tmp_path / 's.json'
    with redirect_stdout(io.StringIO()):
        nb.save_session(str(session))

    captured = {}

    class _StubApp:
        def run(self, **kwargs):
            pass

    def fake_build(notebook, title=None, startup=(), allow_quit=None):
        captured['startup'] = list(startup)
        captured['darkmode_before_startup'] = notebook.darkmode
        return _StubApp()

    original = unichart_terminal.build_terminal_app
    unichart_terminal.build_terminal_app = fake_build
    try:
        with redirect_stdout(io.StringIO()):
            unichart_terminal.terminal(
                nb=UnichartNotebook(), sessions=[str(session)], dark=True,
                panels=[{'method': 'plot', 'x': 't', 'y': ['temp']}],
                open_browser=False)
    finally:
        unichart_terminal.build_terminal_app = original

    startup = captured['startup']
    assert captured['darkmode_before_startup'] is True, 'the board flips dark first'
    assert startup[0] == f"nb.load_session({str(session)!r})"
    assert startup[1] == 'nb.toggle_darkmode(True)', '--dark must outlast the session'
    assert startup[2].startswith('plot('), 'panels draw on top of the session'


def test_terminal_leaves_the_theme_alone_without_dark(tmp_path):
    import unichart_terminal
    nb, _ = _notebook(tmp_path, plot=False)
    session = tmp_path / 's.json'
    with redirect_stdout(io.StringIO()):
        nb.save_session(str(session))

    captured = {}

    class _StubApp:
        def run(self, **kwargs):
            pass

    def fake_build(notebook, title=None, startup=(), allow_quit=None):
        captured['startup'] = list(startup)
        return _StubApp()

    original = unichart_terminal.build_terminal_app
    unichart_terminal.build_terminal_app = fake_build
    try:
        with redirect_stdout(io.StringIO()):
            unichart_terminal.terminal(nb=UnichartNotebook(),
                                       sessions=str(session), open_browser=False)
    finally:
        unichart_terminal.build_terminal_app = original

    assert captured['startup'] == [f"nb.load_session({str(session)!r})"], \
        'no --dark means nothing overrides the session theme'


def test_save_session_flag_writes_a_session_with_the_panel(tmp_path):
    _, csv = _notebook(tmp_path, plot=False)
    out = tmp_path / 'board.json'
    with redirect_stdout(io.StringIO()):
        code = unichart_cli.main([str(csv), '--panel', 'plot:t:temp,press',
                                  '--save-session', str(out)])
    assert code == 0
    session = json.loads(out.read_text())
    assert session['plot_call']['method'] == 'plot'
    assert session['plot_call']['kwargs']['y'] == ['temp', 'press']


def test_info_reports_a_plot_call_that_would_not_replay(tmp_path):
    """--info describes the file, so a broken call must still be reported."""
    nb, _ = _notebook(tmp_path)
    session = tmp_path / 's.json'
    with redirect_stdout(io.StringIO()):
        nb.save_session(str(session))
    payload = json.loads(session.read_text())
    payload['plot_call']['kwargs']['figsize'] = 'not a size'
    session.write_text(json.dumps(payload))

    out = io.StringIO()
    with redirect_stdout(out):
        code = unichart_cli.main([str(session), '--info'])
    assert code == 0
    assert "plot call: plot(x='t'" in out.getvalue()


def test_draw_panels_maps_table_and_contour_signatures(tmp_path):
    """table() takes cols/x_col, not x/y — the mapping render_panel documents."""
    nb, _ = _notebook(tmp_path, plot=False)
    with redirect_stdout(io.StringIO()):
        # Would raise "unexpected keyword argument 'x'" if x/y were passed through.
        unichart_cli._draw_panels(nb, [{'method': 'table', 'x': 't',
                                        'y': ['temp']}])
        # A one-item y unwraps for the methods that need a scalar.
        unichart_cli._draw_panels(nb, [{'method': 'histogram', 'x': 'temp'}])
    assert nb._last_plot_call['method'] == 'histogram'
    assert nb._last_plot_call['kwargs']['x'] == 'temp'


def test_info_on_a_session_reports_its_plot_call(tmp_path):
    nb, _ = _notebook(tmp_path)
    session = tmp_path / 's.json'
    with redirect_stdout(io.StringIO()):
        nb.save_session(str(session))

    out = io.StringIO()
    with redirect_stdout(out):
        code = unichart_cli.main([str(session), '--info'])
    assert code == 0
    text = out.getvalue()
    assert '1 dataset(s):' in text
    assert "plot call: plot(x='t'" in text


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
                argcount = fn.__code__.co_argcount
                fn(Path(tmp)) if argcount else fn()
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
