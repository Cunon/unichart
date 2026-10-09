"""The example datasets on the board's sidebar.

Each example is a frame plus a cheat sheet written for its columns, so the
risk is a snippet that names a column, method or set the example doesn't
have. These tests load every example — into an empty notebook and on top of
data already there, which shifts its set indices — and run every snippet of
its cheat sheet through the terminal's own ``Session``. The board tests click
the example buttons through the real dispatch callback.

Runs under pytest, and standalone (``python tests/test_examples.py``). Dash is
imported only inside the board tests.
"""

import io
import json
import sys
import tempfile
import warnings
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unichart import terminal
from unichart import UnichartNotebook

warnings.filterwarnings('ignore')

_TMP = Path(tempfile.mkdtemp(prefix='unichart-examples-'))


def _notebook():
    with redirect_stdout(io.StringIO()):
        return UnichartNotebook()


def _loaded(example, prior):
    """A session with ``example`` loaded the way the board loads it, after
    ``prior`` sets of the engine example. Returns (session, start, sheet)."""
    session = terminal.Session(_notebook())
    if prior:
        path = _TMP / 'prior.csv'
        terminal.demo_frame().to_csv(path, index=False)
        session.run(f'uc.load({str(path)!r})')
    first = len(session.uc.sets)
    frame = example['frame']()
    path = _TMP / f"{example['key']}.csv"
    frame.to_csv(path, index=False)
    session.run(f'uc.load({str(path)!r})')
    assert len(session.uc.sets) - first == frame['SETNUMBER'].nunique()
    start, sheet = terminal.example_commands(example, first,
                                             frame['SETNUMBER'].nunique())
    return session, start, sheet


def _clean(result, command):
    """No traceback, no printed complaint, no stray repr echoed back."""
    assert result['error'] is None, f'{command}\n{result["error"]}'
    for line in result['text'].splitlines():
        low = line.strip().lower()
        assert not low.startswith(('warning', 'error', 'unknown')), \
            f'{command} printed: {line}'
        assert ' object at 0x' not in line, f'{command} echoed: {line}'


def test_every_example_starts_with_a_figure():
    for example in terminal.EXAMPLES:
        for prior in (False, True):
            session, start, _ = _loaded(example, prior)
            for command in start:
                result = session.run(command)
                _clean(result, command)
            assert result['figure'] is not None, example['key']
            if prior:
                # Only the example's own sets are drawn, not what was there.
                assert [ds.index for ds in session.uc.selected()][0] == 3


def test_every_snippet_runs_on_its_example():
    """Each snippet from a freshly started example, so none relies on another."""
    for example in terminal.EXAMPLES:
        for prior in (False, True):
            _, _, sheet = _loaded(example, prior)
            for i, (snippet, _) in enumerate(sheet):
                session, start, again = _loaded(example, prior)
                for command in start:
                    session.run(command)
                assert again[i][0] == snippet
                result = session.run(snippet)
                _clean(result, f"[{example['key']}, prior={prior}] {snippet}")
                if snippet.split('(')[0].endswith(('plot', 'plot_ymult', 'contour',
                                                   'box', 'bar', 'histogram',
                                                   'plot_marginal')) \
                        or '; plot(' in snippet or '; contour(' in snippet:
                    assert result['figure'] is not None, snippet


def test_indices_follow_where_the_example_landed():
    engine = terminal.EXAMPLES[0]
    _, sheet = terminal.example_commands(engine, 0, 3)
    assert "select(delta(base_idx=0, study_indices=[1, 2]" in sheet[3][0]
    start, sheet = terminal.example_commands(engine, 5, 3)
    assert start[0] == "select('5:8')"
    assert "delta(base_idx=5, study_indices=[6, 7]" in sheet[3][0]
    assert "restore('5:8')" in [s for s, _ in sheet]


def test_a_closing_semicolon_hides_the_value():
    session = terminal.Session(_notebook())
    assert session.run('[1, 2]')['text'] == '[1, 2]'
    assert session.run('[1, 2];')['text'] == ''
    assert session.run('[1, 2];  # quiet')['text'] == ''
    assert session.run("'a;'")['text'] == "'a;'"


# ---------------------------------------------------------------------------
# The board
# ---------------------------------------------------------------------------

def _board():
    uc = _notebook()
    app = terminal.build_terminal_app(uc, banner=False)
    key = next(k for k in app.callback_map if 'term-cheats.data' in k)
    outputs = []
    for out in key.strip('.').split('...'):
        ident, prop = out.rsplit('.', 1)
        outputs.append({'id': json.loads(ident) if ident.startswith('{') else ident,
                        'property': prop})
    return uc, app, key, outputs


def _click(app, key, outputs, trigger, examples, chips, cheats):
    """POST one dispatch round trip, as the browser would."""
    def ids(kind, values):
        return [{'id': {'type': kind, 'index': i}, 'property': 'n_clicks',
                 'value': v} for i, v in enumerate(values)]
    body = {
        'output': key, 'outputs': outputs, 'changedPropIds': [trigger],
        'inputs': [{'id': 'term-submit', 'property': 'n_clicks', 'value': 0},
                   {'id': 'term-upload', 'property': 'contents', 'value': None},
                   ids('term-example', examples),
                   {'id': 'term-save', 'property': 'n_clicks', 'value': 0},
                   ids('term-chip', chips)],
        'state': [{'id': 'term-cheats', 'property': 'data', 'value': cheats},
                  {'id': 'term-input', 'property': 'value', 'value': ''},
                  {'id': 'term-entries', 'property': 'data', 'value': []},
                  {'id': 'term-history', 'property': 'data', 'value': []},
                  {'id': 'term-upload', 'property': 'filename', 'value': None},
                  {'id': 'term-chart', 'property': 'figure', 'value': None}],
    }
    reply = app.server.test_client().post('/_dash-update-component', json=body)
    assert reply.status_code in (200, 204), reply.data[:500]
    return (reply.get_json() or {}).get('response', {})


def _prop(kind, index):
    return json.dumps({'index': index, 'type': kind}, separators=(',', ':')) + '.n_clicks'


def test_the_sidebar_offers_every_example():
    _, app, _, _ = _board()
    layout = str(app.layout)
    for i, example in enumerate(terminal.EXAMPLES):
        assert example['label'] in layout
    assert 'Load demo data' not in layout


def test_clicking_an_example_loads_it_and_swaps_the_cheat_sheet():
    uc, app, key, outputs = _board()
    n = len(terminal.EXAMPLES)
    default = [list(c) for c in terminal.CHEAT_SHEET]

    clicks = [0] * n
    clicks[1] = 1
    reply = _click(app, key, outputs, _prop('term-example', 1), clicks,
                   [0] * len(default), default)
    battery = terminal.EXAMPLES[1]
    assert reply['term-cheat-label']['children'] == f"Cheat sheet · {battery['label']}"
    assert [c[0] for c in reply['term-cheats']['data']] == \
        [s for s, _ in terminal.example_commands(battery, 0, 4)[1]]
    assert reply['term-chart']['figure']['data'], 'the example drew nothing'
    assert len(uc.sets) == 4

    # A second example lands after the first, and only it is selected.
    cheats = reply['term-cheats']['data']
    clicks[3] = 1
    reply = _click(app, key, outputs, _prop('term-example', 3), clicks,
                   [0] * len(cheats), cheats)
    typed = [e['text'] for e in reply['term-entries']['data'] if e['kind'] == 'in']
    assert typed[1] == "select('4:8')", typed
    assert [ds.index for ds in uc.selected()] == [4, 5, 6, 7]

    # A chip resolves against the sheet on screen, not the default one.
    cheats = reply['term-cheats']['data']
    chips = [0] * len(cheats)
    chips[1] = 1
    reply = _click(app, key, outputs, _prop('term-chip', 1), clicks, chips, cheats)
    assert reply['term-input']['value'] == cheats[1][0]


def test_rerendered_buttons_do_not_load_anything():
    uc, app, key, outputs = _board()
    default = [list(c) for c in terminal.CHEAT_SHEET]
    _click(app, key, outputs, _prop('term-example', 0),
           [0] * len(terminal.EXAMPLES), [0] * len(default), default)
    assert not uc.sets


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
