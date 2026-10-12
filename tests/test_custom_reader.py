"""Loading a file type unichart has no reader for.

``uc.load(path, reader=fn)`` and ``uc.register_reader(ext, fn)`` let a function
of the user's (path in, DataFrame out) read any format. Where there is no
reader, each surface asks for one: the notebook's error is a reader to paste
and edit, the explorer stages one in its terminal input, and the ``unichart``
command opens the explorer that way or points at ``--reader``.

Runs under pytest, and standalone (``python tests/test_custom_reader.py``) for
environments without it.
"""

import ast
import base64
import io
import json
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unichart import UnichartNotebook, cli
from unichart._core import _READER_STUB_BODY

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Fixtures, hand-rolled so the module runs without pytest too
# ---------------------------------------------------------------------------

# A rig's whitespace-separated log: a comment line, then a header and rows.
DAT = "# rig 4\nt  temp  press\n0  10.5  1.0\n1  11.0  1.1\n2  11.6  1.3\n"


def read_dat(path):
    return pd.read_csv(path, sep=r'\s+', comment='#')


def _dat(tmp, name='run1.dat'):
    path = tmp / name
    path.write_text(DAT)
    return path


def _quiet(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# The notebook
# ---------------------------------------------------------------------------

def test_an_unknown_type_raises_a_reader_to_start_from(tmp_path):
    dat = _dat(tmp_path)
    uc = _quiet(UnichartNotebook)
    try:
        uc.load(str(dat))
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError('an unknown type loaded without a reader')
    assert "no reader for '.dat' files (run1.dat)" in message
    assert 'def read_dat(path):' in message
    assert f"uc.load({str(dat)!r}, reader=read_dat)" in message
    assert "uc.register_reader('.dat', read_dat)" in message
    assert not uc.sets


def test_a_file_with_no_extension_is_offered_reader_only(tmp_path):
    bare = tmp_path / 'data'
    bare.write_text('a,b\n1,2\n')
    uc = _quiet(UnichartNotebook)
    try:
        uc.load(str(bare))
    except ValueError as exc:
        message = str(exc)
    assert 'it has no extension' in message
    assert 'reader=read_file' in message
    assert 'register_reader' not in message, 'there is no extension to register'
    _quiet(uc.load, str(bare), reader=lambda p: pd.read_csv(p))
    assert uc.sets[0].df['b'].tolist() == [2]


def test_reader_reads_one_load_and_gets_read_kwargs(tmp_path):
    dat = _dat(tmp_path)
    seen = {}

    def reader(path, **kwargs):
        seen.update(path=path, kwargs=kwargs)
        return read_dat(path)

    uc = _quiet(UnichartNotebook)
    _quiet(uc.load, str(dat), reader=reader, read_kwargs={'units': 'SI'})
    assert seen == {'path': str(dat), 'kwargs': {'units': 'SI'}}, \
        'the reader gets the path as a str, and read_kwargs as keywords'
    ds = uc.sets[0]
    assert ds.title == 'run1', 'the filename still titles the set'
    assert ds.df['temp'].tolist() == [10.5, 11.0, 11.6]
    assert ds.file_path == str(dat.resolve())
    assert ds._source['reader'] is True
    # One load only: the next one asks again.
    try:
        uc.load(str(dat))
    except ValueError:
        pass
    else:
        raise AssertionError('reader= leaked past its own load')


def test_reader_wins_over_a_built_in_extension(tmp_path):
    csv = tmp_path / 'euro.csv'
    csv.write_text('t;temp\n0;1,5\n1;2,5\n')
    uc = _quiet(UnichartNotebook)
    _quiet(uc.load, str(csv),
           reader=lambda p: pd.read_csv(p, sep=';', decimal=','))
    assert uc.sets[0].df['temp'].tolist() == [1.5, 2.5]


def test_register_reader_spellings_decorator_and_scope(tmp_path):
    uc = _quiet(UnichartNotebook)
    assert uc.register_reader('DAT', read_dat) is read_dat
    assert uc._custom_readers == {'.dat': read_dat}

    @uc.register_reader(['*.log', 'out'])
    def read_log(path):
        return read_dat(path)

    assert callable(read_log), 'the decorator hands the function back'
    assert set(uc._custom_readers) == {'.dat', '.log', '.out'}

    for name in ('RUN1.DAT', 'run2.log', 'run3.out'):
        _quiet(uc.load, str(_dat(tmp_path, name)))
    assert [ds.title for ds in uc.sets] == ['RUN1', 'run2', 'run3']

    other = _quiet(UnichartNotebook)
    assert other._custom_readers == {}, 'readers are per notebook'
    assert other._reader_for(tmp_path / 'run2.log') == (None, False)

    for bad in ('', '.', '*'):
        try:
            uc.register_reader(bad, read_dat)
        except ValueError as exc:
            assert 'reader=' in str(exc)
        else:
            raise AssertionError(f'register_reader accepted {bad!r}')
    try:
        uc.register_reader('.dat', 'read_dat')
    except TypeError:
        pass
    else:
        raise AssertionError('a non-callable reader was accepted')


def test_reader_results_are_coerced_or_rejected(tmp_path):
    dat = _dat(tmp_path)
    uc = _quiet(UnichartNotebook)
    _quiet(uc.load, str(dat), reader=lambda p: {'a': [1, 2]})
    _quiet(uc.load, str(dat), reader=lambda p: np.arange(6).reshape(3, 2))
    _quiet(uc.load, str(dat), reader=lambda p: pd.Series([1, 2], name='s'))
    assert [len(ds.df) for ds in uc.sets] == [2, 3, 2]

    def read_list(path):
        return [1, 2, 3]

    try:
        uc.load(str(dat), reader=read_list)
    except TypeError as exc:
        assert 'read_list() returned list for run1.dat' in str(exc)
    else:
        raise AssertionError('a list was accepted as a DataFrame')


def test_sets_from_a_custom_reader_are_embedded_in_sessions(tmp_path):
    """A session can't store the function, so it stores the rows.

    Restored in a fresh notebook that has no reader at all: a file reference
    here would be skipped on reload, and the data would silently go missing.
    """
    dat = _dat(tmp_path)
    csv = tmp_path / 'plain.csv'
    pd.DataFrame({'t': [0, 1], 'temp': [5.0, 6.0]}).to_csv(csv, index=False)
    uc = _quiet(UnichartNotebook)
    uc.register_reader('.dat', read_dat)
    _quiet(uc.load, [str(dat), str(csv)])

    session = tmp_path / 's.json'
    _quiet(uc.save_session, str(session))
    kinds = [e['source']['kind'] for e in json.loads(session.read_text())['sets']]
    assert kinds == ['embedded', 'file'], 'only the custom-read set is embedded'

    fresh = _quiet(UnichartNotebook)
    _quiet(fresh.load_session, str(session))
    assert [ds.title for ds in fresh.sets] == ['run1', 'plain']
    assert fresh.sets[0].df['temp'].tolist() == [10.5, 11.0, 11.6]


# ---------------------------------------------------------------------------
# The explorer board
# ---------------------------------------------------------------------------

def _board(uc=None, staged=()):
    from unichart import terminal

    uc = uc or _quiet(UnichartNotebook)
    app = _quiet(terminal.build_terminal_app, uc, banner=False, staged=staged)
    key = next(k for k in app.callback_map if 'term-cheats.data' in k)
    outputs = []
    for out in key.strip('.').split('...'):
        ident, prop = out.rsplit('.', 1)
        outputs.append({'id': json.loads(ident) if ident.startswith('{') else ident,
                        'property': prop})
    return uc, app, key, outputs


def _post(app, key, outputs, trigger, source='', files=()):
    """One dispatch round trip, as the browser would send it."""
    from unichart import terminal

    contents = [f"data:application/octet-stream;base64,"
                f"{base64.b64encode(raw).decode()}" for _, raw in files] or None
    names = [name for name, _ in files] or None
    body = {
        'output': key, 'outputs': outputs, 'changedPropIds': [trigger],
        'inputs': [{'id': 'term-submit', 'property': 'n_clicks',
                    'value': 1 if trigger == 'term-submit.n_clicks' else 0},
                   {'id': 'term-upload', 'property': 'contents', 'value': contents},
                   [{'id': {'type': 'term-example', 'index': i},
                     'property': 'n_clicks', 'value': 0}
                    for i in range(len(terminal.EXAMPLES))],
                   {'id': 'term-save', 'property': 'n_clicks', 'value': 0},
                   [{'id': {'type': 'term-chip', 'index': i},
                     'property': 'n_clicks', 'value': 0}
                    for i in range(len(terminal.CHEAT_SHEET))]],
        'state': [{'id': 'term-cheats', 'property': 'data', 'value': None},
                  {'id': 'term-input', 'property': 'value', 'value': source},
                  {'id': 'term-entries', 'property': 'data', 'value': []},
                  {'id': 'term-history', 'property': 'data', 'value': []},
                  {'id': 'term-upload', 'property': 'filename', 'value': names},
                  {'id': 'term-chart', 'property': 'figure', 'value': None}],
    }
    with redirect_stdout(io.StringIO()):
        reply = app.server.test_client().post('/_dash-update-component', json=body)
    assert reply.status_code in (200, 204), reply.data[:500]
    return (reply.get_json() or {}).get('response', {})


def test_reader_template_is_python_that_registers_then_loads():
    from unichart import terminal

    text = terminal._reader_template(['/x/a.dat', '/x/b.dat', '/x/c.log', '/x/bare'])
    ast.parse(text)
    assert text.count('def read_dat(path):') == 1, 'one reader per type'
    assert "uc.register_reader('.dat', read_dat)" in text
    assert "uc.register_reader('.log', read_log)" in text
    assert "uc.load('/x/a.dat')" in text and "uc.load('/x/b.dat')" in text
    assert "uc.load('/x/bare', reader=read_file)" in text
    assert text.index('register_reader') < text.index('uc.load('), \
        'registered before the loads that need it'


def test_dropping_an_unknown_type_stages_a_reader_then_running_it_loads():
    uc, app, key, outputs = _board()
    reply = _post(app, key, outputs, 'term-upload.contents',
                  files=[('run1.dat', DAT.encode())])

    staged = reply['term-input']['value']
    assert 'def read_dat(path):' in staged
    assert "uc.register_reader('.dat', read_dat)" in staged
    notes = [e['text'] for e in reply['term-entries']['data']]
    assert any('run1.dat: unichart has no reader for .dat files' in n for n in notes)
    assert not any(e['kind'] == 'err' for e in reply['term-entries']['data'])
    assert not uc.sets, 'nothing loads until the reader is run'
    dropped = Path(staged.rsplit("uc.load('", 1)[1].split("'")[0])
    assert dropped.read_text() == DAT, 'the drop lands on disk byte for byte'

    # The user fixes the guess for this format and presses Enter.
    fixed = staged.replace(_READER_STUB_BODY,
                           "pd.read_csv(path, sep=r'\\s+', comment='#')")
    reply = _post(app, key, outputs, 'term-submit.n_clicks', source=fixed)
    assert reply['term-input']['value'] == ''
    assert [ds.title for ds in uc.sets] == ['run1']
    assert uc.sets[0].df['press'].tolist() == [1.0, 1.1, 1.3]

    # Registered, so the next drop of that type just loads.
    reply = _post(app, key, outputs, 'term-upload.contents',
                  files=[('run2.dat', DAT.encode())])
    typed = [e['text'] for e in reply['term-entries']['data'] if e['kind'] == 'in']
    assert len(typed) == 1 and typed[0].startswith('uc.load(') \
        and typed[0].endswith("run2.dat')"), typed
    assert reply['term-input']['value'] == ''
    assert [ds.title for ds in uc.sets] == ['run1', 'run2']


def test_a_registered_reader_overrides_the_board_for_a_built_in_type():
    """A dropped .csv must reach the user's reader as the bytes it came as.

    The board reads its own types in memory and re-saves them as a comma csv;
    going that way, a ';'-separated file would be mangled before the reader
    ever saw it.
    """
    uc = _quiet(UnichartNotebook)
    uc.register_reader('.csv', lambda p: pd.read_csv(p, sep=';', decimal=','))
    uc, app, key, outputs = _board(uc)
    _post(app, key, outputs, 'term-upload.contents',
          files=[('euro.csv', b't;temp\n0;1,5\n1;2,5\n')])
    assert uc.sets[0].df['temp'].tolist() == [1.5, 2.5]


def test_the_file_picker_offers_registered_types():
    argv = UnichartNotebook._dialog_argv('zenity', ('.dat', '.csv'))
    data_filter = next(a for a in argv if a.startswith('--file-filter=Data'))
    assert data_filter.endswith('*.parquet *.dat'), 'extras follow, deduplicated'


def test_a_mixed_drop_loads_what_it_can_and_stages_the_rest():
    uc, app, key, outputs = _board()
    csv = pd.DataFrame({'t': [0, 1], 'temp': [5.0, 6.0]}).to_csv(index=False)
    reply = _post(app, key, outputs, 'term-upload.contents',
                  files=[('plain.csv', csv.encode()), ('run1.dat', DAT.encode())])
    assert [ds.title for ds in uc.sets] == ['plain']
    assert 'def read_dat(path):' in reply['term-input']['value']


def test_the_board_can_open_with_a_reader_staged(tmp_path):
    from unichart import terminal

    dat = _dat(tmp_path)
    _, app, _, _ = _board(staged=[str(dat)])
    textarea = next(c for c in _walk(app.layout)
                    if getattr(c, 'id', None) == 'term-input')
    assert f"uc.load({str(dat)!r})" in textarea.value
    assert 'run1.dat: unichart has no reader' in str(app.layout)

    # terminal(data=...) loads what it can read and stages the rest.
    csv = tmp_path / 'plain.csv'
    pd.DataFrame({'t': [0, 1]}).to_csv(csv, index=False)
    captured = {}

    class _StubApp:
        def run(self, **kwargs):
            pass

    def fake_build(notebook, title=None, startup=(), allow_quit=None, staged=()):
        captured['staged'] = list(staged)
        captured['sets'] = [ds.title for ds in notebook.sets]
        return _StubApp()

    original = terminal.build_terminal_app
    terminal.build_terminal_app = fake_build
    try:
        _quiet(terminal.terminal, uc=_quiet(UnichartNotebook),
               data=[str(dat), str(csv)], open_browser=False)
    finally:
        terminal.build_terminal_app = original
    assert captured == {'staged': [str(dat)], 'sets': ['plain']}


def _walk(component):
    yield component
    children = getattr(component, 'children', None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if child is not None and hasattr(child, 'to_plotly_json'):
            yield from _walk(child)


# ---------------------------------------------------------------------------
# The unichart command
# ---------------------------------------------------------------------------

def _script(tmp):
    script = tmp / 'readers.py'
    script.write_text("import pandas as pd\n\n"
                      "def read_dat(path):\n"
                      "    return pd.read_csv(path, sep=r'\\s+', comment='#')\n")
    return script


def _main(argv):
    out, err = io.StringIO(), io.StringIO()
    old_err, sys.stderr = sys.stderr, err
    try:
        with redirect_stdout(out):
            code = cli.main([str(a) for a in argv])
    finally:
        sys.stderr = old_err
    return code, out.getvalue(), err.getvalue()


def test_reader_flag_reads_the_files(tmp_path):
    dat, script = _dat(tmp_path), _script(tmp_path)
    code, out, err = _main([dat, '--reader', f'{script}:read_dat', '--info'])
    assert code == 0, err
    assert '1 dataset(s)' in out and 'run1' in out and 'temp' in out


def test_reader_flag_is_kept_for_the_board(tmp_path):
    """Registered for the type, so a .dat dropped on the board loads too."""
    from unichart import dashboard

    dat, script = _dat(tmp_path), _script(tmp_path)
    seen = {}

    def fake_explore(notebook=None, **kwargs):
        seen.update(kwargs, readers=dict(notebook._custom_readers),
                    sets=len(notebook.sets))

    original = dashboard.explore
    dashboard.explore = fake_explore
    try:
        code, _, err = _main([dat, '--reader', f'{script}:read_dat'])
    finally:
        dashboard.explore = original
    assert code == 0, err
    assert seen['sets'] == 1 and seen['data'] is None
    assert list(seen['readers']) == ['.dat']


def test_an_unknown_type_opens_the_board_with_a_reader_staged(tmp_path):
    from unichart import dashboard

    dat = _dat(tmp_path)
    csv = tmp_path / 'plain.csv'
    pd.DataFrame({'t': [0, 1]}).to_csv(csv, index=False)
    seen = {}

    def fake_explore(notebook=None, **kwargs):
        seen.update(kwargs, sets=[ds.title for ds in notebook.sets])

    original = dashboard.explore
    dashboard.explore = fake_explore
    try:
        code, _, err = _main([csv, dat])
    finally:
        dashboard.explore = original
    assert code == 0, err
    assert seen['data'] == [str(dat)], 'the .dat is handed to the board to stage'
    assert seen['sets'] == ['plain'], 'the csv still loads up front'


def test_without_a_board_an_unknown_type_points_at_reader(tmp_path):
    dat = _dat(tmp_path)
    for extra in (['--info'], ['--html', tmp_path / 'b.html'],
                  ['--save-session', tmp_path / 's.json'], ['--combine'],
                  ['--set-col', 't']):
        code, _, err = _main([dat] + extra)
        assert code == 2, (extra, err)
        assert f'no reader for {dat}' in err and '--reader' in err, (extra, err)
    _, _, err = _main([dat, '--combine'])
    assert '--combine reads every FILE at startup' in err


def test_bad_reader_specs_are_one_line_errors(tmp_path):
    dat, script = _dat(tmp_path), _script(tmp_path)
    for spec, expect in (('readers.py', 'expected SCRIPT.py:FUNC'),
                         (f'{tmp_path}/nope.py:f', 'no such file'),
                         (f'{script}:missing', "has no function 'missing'"),
                         ('no_such_module_xyz:f', 'could not load')):
        code, _, err = _main([dat, '--reader', spec, '--info'])
        assert code == 2 and expect in err, (spec, err)
    assert cli._load_reader('os.path:basename') is __import__('os').path.basename


def test_reader_flag_is_rejected_for_a_session_alone(tmp_path):
    uc = _quiet(UnichartNotebook)
    _quiet(uc.load, pd.DataFrame({'t': [0, 1]}))
    session = tmp_path / 's.json'
    _quiet(uc.save_session, str(session))
    code, _, err = _main([session, '--reader', 'os.path:basename', '--info'])
    assert code == 2 and '--reader applies to data files' in err, err


# ---------------------------------------------------------------------------
# Standalone runner, for environments without pytest
# ---------------------------------------------------------------------------

def _run_all():
    import tempfile
    import traceback

    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failed = []
    for name, fn in tests:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fn(Path(tmp)) if fn.__code__.co_argcount else fn()
            except Exception:                                 # noqa: BLE001
                failed.append(name)
                print(f'FAIL  {name}')
                traceback.print_exc()
            else:
                print(f'PASS  {name}')
    print(f'\n{len(tests) - len(failed)}/{len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(_run_all())
