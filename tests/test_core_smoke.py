"""Smoke test for the plotting core: every plot method draws, nothing raises.

The rest of the suite covers the seams (CLI, sessions, quit, file picker); this
is the only file that drives the ~12k-line plotting API itself. It checks that
each call produces a figure with traces — not what the figure looks like; the
gallery (``gallery/make_gallery.py``) is where the looks get eyeballed.

Data is ``gallery/dyno_runs.csv``: four engine runs sharing one schema, loaded
the way the gallery loads it, so the calls here are ones known to make sense.

Runs under pytest, and standalone (``python tests/test_core_smoke.py``) for
environments without it.
"""

import inspect
import io
import json
import re
import sys
import tempfile
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unichart import UnichartNotebook
from unichart.dashboard import PLOT_METHODS, render_panel

warnings.filterwarnings('ignore')

DATA_CSV = Path(__file__).resolve().parent.parent / 'gallery' / 'dyno_runs.csv'


# ---------------------------------------------------------------------------
# Fixtures, hand-rolled so the module runs without pytest too
# ---------------------------------------------------------------------------

def _notebook():
    """A notebook holding the dyno runs, with its chatter swallowed."""
    with redirect_stdout(io.StringIO()):
        nb = UnichartNotebook()
        nb.set_copy_buttons(False)
        nb.load_df(pd.read_csv(DATA_CSV), set_idx_column='run_id',
                   set_name_column='run_name')
    return nb


def _draw(call):
    """Run ``call(nb)`` on a fresh notebook; return the figure it left behind."""
    nb = _notebook()
    with redirect_stdout(io.StringIO()):
        call(nb)
    fig = nb.last_fig
    assert isinstance(fig, go.Figure), f'no figure, got {type(fig).__name__}'
    assert len(fig.data) > 0, 'figure has no traces'
    return fig


# ---------------------------------------------------------------------------
# One test per plot method — every entry of dashboard.PLOT_METHODS
# ---------------------------------------------------------------------------

PLOT_CALLS = {
    'plot': lambda nb: nb.plot(x='time_s', y=['rpm', 'cht_c'], ncols=2),
    'plot_ymult': lambda nb: nb.plot_ymult(x='time_s',
                                           y=['rpm', 'cht_c', 'fuel_kgh']),
    'plot_marginal': lambda nb: nb.plot_marginal(x='rpm', y='torque_nm',
                                                 marginal='box'),
    'bar': lambda nb: nb.bar(x='phase', y=['fuel_kgh', 'eta_pct'], agg='mean',
                             barmode='group'),
    'box': lambda nb: nb.box(x='phase', y='eta_pct', points='outliers'),
    'histogram': lambda nb: nb.histogram(x='cht_c', nbins=40),
    'contour': lambda nb: nb.contour(x='rpm', y='torque_nm', z='eta_pct'),
    # The runs have no angle column; time_s read as degrees still draws.
    'polar': lambda nb: nb.polar(x='time_s', y='rpm'),
    'table': lambda nb: nb.table(cols=['rpm', 'eta_pct'], output='fig'),
}


def _plot_test(method):
    def test():
        _draw(PLOT_CALLS[method])
    test.__name__ = f'test_plot_method_{method}'
    test.__doc__ = f'nb.{method} draws a figure with traces.'
    return test


for _method in PLOT_METHODS:
    globals()[f'test_plot_method_{_method}'] = _plot_test(_method)


def test_every_plot_method_is_covered():
    """A new entry in PLOT_METHODS needs a call here, or it goes untested."""
    assert set(PLOT_CALLS) == set(PLOT_METHODS)


# ---------------------------------------------------------------------------
# The rest of the everyday surface
# ---------------------------------------------------------------------------

def test_by_sets_with_selection_and_query():
    def call(nb):
        nb.select([0, 3])
        nb.query('all', 'phase == "climb"')
        nb.plot(x='time_s', y=['cht_c', 'egt_c'], by='sets')
    _draw(call)


def test_subplot_queries():
    """Each subplot keeps only the rows its query matches; a single y gets a
    panel per query, titled by it, and a set's legend entry lands on the
    first panel it is actually drawn in."""
    nb = _notebook()
    with redirect_stdout(io.StringIO()):
        nb.plot('time_s', 'cht_c', subplot_queries=['rpm > 1e9', 'phase == "idle"'])
    fig = nb.last_fig
    assert [a.text for a in fig.layout.annotations][:2] == ['rpm > 1e9', 'phase == "idle"']
    assert {t.xaxis for t in fig.data} == {'x2'}       # nothing matched panel 1
    assert all(t.showlegend for t in fig.data)
    idle = pd.read_csv(DATA_CSV).query('phase == "idle"')
    assert sum(len(t.x) for t in fig.data) == len(idle)

    with redirect_stdout(io.StringIO()):
        nb.plot('time_s', ['cht_c', 'egt_c'], subplot_queries=[None, 'rpm > 3000'])
    full, part = ([t for t in nb.last_fig.data if t.xaxis == ax] for ax in ('x', 'x2'))
    assert sum(len(t.x) for t in part) < sum(len(t.x) for t in full)

    try:
        with redirect_stdout(io.StringIO()):
            nb.plot('time_s', 'cht_c', subplot_queries=['no_such_col > 1', None])
    except ValueError as e:
        assert 'no_such_col' in str(e)
    else:
        raise AssertionError('a query no set can evaluate should raise')


def test_subplot_scales():
    """Per-subplot ranges, keyed by parameter like scale(), override scale()
    on whichever axis shows the parameter: None keeps scale()'s limit and
    'auto' drops it."""
    nb = _notebook()
    with redirect_stdout(io.StringIO()):
        nb.scale('egt_c', (0, 1000))
        nb.plot('time_s', 'egt_c', subplot_scales=[
            None, {'egt_c': (100, 200), 'time_s': None},
            {'time_s': (0, 60), 'egt_c': 'auto'}])
    lay = nb.last_fig.layout
    assert tuple(lay.yaxis.range) == (0, 1000)
    assert tuple(lay.yaxis2.range) == (100, 200) and lay.xaxis2.range is None
    assert lay.yaxis3.range is None and tuple(lay.xaxis3.range) == (0, 60)
    assert not lay.annotations                       # scales alone: no titles

    with warnings.catch_warnings(record=True) as caught, \
            redirect_stdout(io.StringIO()):
        warnings.simplefilter('always')
        nb.plot('time_s', ['cht_c', 'egt_c'],
                subplot_scales=[{'cht_c': (0, 300)}, {'cht': (0, 1)}])
    assert any("['cht']" in str(w.message) for w in caught)
    assert tuple(nb.last_fig.layout.yaxis.range) == (0, 300)

    for bad in ([(1, 2)], [{'egt_c': (1, 2, 3)}], {'egt_c': (0, 1)}):
        try:
            nb.plot('time_s', 'egt_c', subplot_scales=bad)
        except TypeError:
            pass
        else:
            raise AssertionError(f'subplot_scales={bad!r} should raise')


def test_subplot_sets_queries_and_scales_together():
    """The three lists line up panel by panel, need the same length, are
    recorded in their JSON form, and are ignored (with a warning) by by='sets'."""
    nb = _notebook()
    with redirect_stdout(io.StringIO()):
        nb.plot('time_s', 'egt_c', subplot_sets=['0:2', 3],
                subplot_queries=[None, 'rpm > 2000'], subplot_scales=[None, {'egt_c': (0, 900)}])
    fig = nb.last_fig
    assert [a.text for a in fig.layout.annotations][:2] == ['Sets 0, 1', 'Set 3 · rpm > 2000']
    assert tuple(fig.layout.yaxis2.range) == (0, 900)
    assert nb._last_plot_call['kwargs']['subplot_scales'] == [None, {'egt_c': [0, 900]}]

    try:
        nb.plot('time_s', 'egt_c', subplot_sets=[0, 1], subplot_queries=['rpm > 1'])
    except ValueError as e:
        assert 'same number of entries' in str(e)
    else:
        raise AssertionError('mismatched lists should raise')

    with warnings.catch_warnings(record=True) as caught, \
            redirect_stdout(io.StringIO()):
        warnings.simplefilter('always')
        nb.plot('time_s', 'egt_c', by='sets', subplot_queries=['rpm > 1'],
                subplot_scales=[{'egt_c': (0, 1)}])
    said = [str(w.message) for w in caught]
    assert any('ignores subplot_queries' in s for s in said)
    assert any('ignores subplot_scales' in s for s in said)


def test_subplot_queries_and_scales_session():
    """A saved session replays them, and a column whitelist keeps the columns
    a subplot query names."""
    nb = _notebook()
    with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
        nb.plot('time_s', 'cht_c', subplot_queries=['rpm > 3000', None],
                subplot_scales=[{'cht_c': (0, 300)}, None])
        path = Path(tmp) / 'session.json'
        nb.save_session(path, embed_data='all', parms=['time_s', 'cht_c'])

        fresh = UnichartNotebook()
        fresh.set_copy_buttons(False)
        fresh.load_session(path)
    assert 'rpm' in fresh.sets[0].columns
    assert [len(t.x) for t in fresh.last_fig.data] == [len(t.x) for t in nb.last_fig.data]
    assert tuple(fresh.last_fig.layout.yaxis.range) == (0, 300)


def _axis_titles(fig, cells, secondary_y=False):
    """``(x title, y title)`` of each ``(row, col)`` subplot in ``cells``."""
    subs = [fig.get_subplot(r, c, secondary_y=secondary_y) for r, c in cells]
    return [(s.xaxis.title.text, s.yaxis.title.text) for s in subs]


def test_axis_labels():
    """labels='all' (the default) titles every subplot; 'auto' keeps a title
    its column / row shares only on the outer subplot, and 'outer' keeps only
    those whatever the others say."""
    nb = _notebook()
    cells = [(1, 1), (1, 2), (2, 1)]
    y = ['cht_c', 'egt_c', 'rpm']
    with redirect_stdout(io.StringIO()):
        nb.plot('time_s', y, ncols=2)
    assert _axis_titles(nb.last_fig, cells) == [('time_s', v) for v in y]

    with redirect_stdout(io.StringIO()):
        nb.plot('time_s', y, ncols=2, labels='auto')
    # Column 1's x title moves to its bottom subplot; column 2's single
    # subplot is its bottom one. Row 1's y titles differ, so both stay.
    assert _axis_titles(nb.last_fig, cells) == [
        (None, 'cht_c'), ('time_s', 'egt_c'), ('time_s', 'rpm')]
    assert nb._last_plot_call['kwargs']['labels'] == 'auto'

    with redirect_stdout(io.StringIO()):
        nb.plot('time_s', y, ncols=2, labels='outer')
    assert _axis_titles(nb.last_fig, cells) == [
        (None, 'cht_c'), ('time_s', None), ('time_s', 'rpm')]

    nb.y_label = 'Temperature'
    with redirect_stdout(io.StringIO()):
        nb.plot('time_s', ['cht_c', 'egt_c'], labels='auto')
    assert _axis_titles(nb.last_fig, [(1, 1), (1, 2)]) == [
        ('time_s', 'Temperature'), ('time_s', None)]

    try:
        nb.plot('time_s', y, labels='shared')
    except ValueError as e:
        assert 'shared' in str(e)
    else:
        raise AssertionError('an unknown labels mode should raise')


def test_axis_labels_by_sets_and_elsewhere():
    """by='sets' places its secondary y title on the last subplot of each row;
    'ymult' warns that it ignores labels; bar keeps its outer titles."""
    nb = _notebook()
    cells = [(1, 1), (1, 2), (2, 1), (2, 2)]
    with redirect_stdout(io.StringIO()):
        nb.plot('time_s', ['cht_c', 'egt_c'], by='sets', ncols=2, labels='auto')
    fig = nb.last_fig
    assert _axis_titles(fig, cells) == [
        (None, 'cht_c'), (None, None), ('time_s', 'cht_c'), ('time_s', None)]
    assert [y2 for _, y2 in _axis_titles(fig, cells, secondary_y=True)] == [
        None, 'egt_c', None, 'egt_c']

    with redirect_stdout(io.StringIO()):
        nb.plot('time_s', ['cht_c', 'egt_c'], by='sets', ncols=2)
    assert {t for pair in _axis_titles(nb.last_fig, cells) for t in pair} == {
        'time_s', 'cht_c'}

    with warnings.catch_warnings(record=True) as caught, \
            redirect_stdout(io.StringIO()):
        warnings.simplefilter('always')
        nb.plot('time_s', ['cht_c', 'egt_c'], by='ymult', labels='outer')
    assert any("ignores labels='outer'" in str(w.message) for w in caught)

    with redirect_stdout(io.StringIO()):
        nb.bar('phase', ['fuel_kgh', 'eta_pct', 'rpm'], ncols=2)
    assert [x for x, _ in _axis_titles(nb.last_fig, [(1, 1), (1, 2), (2, 1)])] == [
        None, 'phase', 'phase']


def test_formatting_and_decorations():
    def call(nb):
        nb.var_format('cht_c', color='firebrick', linestyle='--')
        nb.color(0, 'black')
        nb.marker('all', 'o')
        nb.markersize('all', 4)
        nb.line('cht_c', 150, color='firebrick', linestyle='--', label='limit')
        nb.highlight('time_s', (300, 450), color='orange', alpha=0.12)
        nb.plot(x='time_s', y='cht_c')
    _draw(call)


def test_markersize_zero_hides_markers():
    """markersize 0 turns a set's (or a variable's) markers off, as marker
    'none' does, in every builder that draws points."""
    def modes(nb, call):
        with redirect_stdout(io.StringIO()):
            call()
        return {tr.name: tr.mode for tr in nb.last_fig.data
                if tr.type in ('scatter', 'scattergl', 'scatterpolar')}

    nb = _notebook()
    nb.markersize(0, 0)
    got = modes(nb, lambda: nb.plot(x='time_s', y='cht_c'))
    off = [m for n, m in got.items() if n.startswith('0:')]
    assert off and all('markers' not in m for m in off), got
    assert any('markers' in m for n, m in got.items() if not n.startswith('0:')), got

    nb.linestyle(0, '-')
    got = modes(nb, lambda: nb.plot(x='time_s', y='cht_c'))
    assert all(m == 'lines' for n, m in got.items() if n.startswith('0:')), got

    nb.markersize('all', 0)
    for call in (lambda: nb.plot(x='time_s', y='cht_c', by='sets'),
                 lambda: nb.plot_marginal(x='rpm', y='torque_nm'),
                 lambda: nb.polar(x='time_s', y='rpm')):
        got = modes(nb, call)
        assert got and all('markers' not in m for m in got.values()), got

    nb = _notebook()
    nb.markersize('cht_c', 0)
    got = modes(nb, lambda: nb.plot_ymult(x='time_s', y=['cht_c', 'rpm']))
    assert all('markers' not in m for n, m in got.items() if 'cht_c' in n), got
    assert any('markers' in m for n, m in got.items() if 'rpm' in n), got


def test_full_legend_with_reference_line():
    """The full-legend fit (what save_png and static images use) counts a
    reference line's legend entry — a shape, so plotly>=5.16."""
    def call(nb):
        nb.set_default_format(legend_scroll=False)
        nb.line('cht_c', 150, color='firebrick', legend='CHT limit')
        nb.plot(x='time_s', y='cht_c')
    _draw(call)


def _custom_entries(fig):
    """The legend_entry traces: legend-only, one None point each (an ``r``
    point on a polar plot, an ``x`` one elsewhere)."""
    def one_none(tr):
        pts = getattr(tr, 'r' if tr.type == 'scatterpolar' else 'x', None)
        return pts is not None and len(pts) == 1 and pts[0] is None
    return {tr.name: tr for tr in fig.data if one_none(tr)}


def test_custom_legend_entries():
    """legend_entry() puts entries no set draws into every plot's legend,
    styled as asked, ordered around the sets, and edited or removed by name."""
    nb = _notebook()
    nb.legend_entry('Spec limit', color='red', linestyle='--')
    nb.legend_entry('Test point', marker='x')                  # marker only
    nb.legend_entry('Model', marker='circle-open', linestyle='-', linewidth=3)
    nb.legend_entry('Note', linestyle='none')                  # text only
    nb.legend_entry('Key', position='start', color='teal', marker='s')
    for method, call in PLOT_CALLS.items():
        if method == 'table':                   # a go.Table: no legend
            continue
        with redirect_stdout(io.StringIO()):
            call(nb)
        entries = _custom_entries(nb.last_fig)
        assert list(entries) == ['Spec limit', 'Test point', 'Model', 'Note',
                                 'Key'], method
        ranks = [tr.legendrank for tr in nb.last_fig.data
                 if tr.name not in entries and tr.legendrank is not None]
        assert entries['Key'].legendrank < min(ranks, default=1000), method
        assert entries['Spec limit'].legendrank > max(ranks, default=1000), method

    entries = _custom_entries(nb.last_fig)
    assert entries['Spec limit'].mode == 'lines'
    assert entries['Spec limit'].line.dash == 'dash'
    assert entries['Test point'].mode == 'markers'
    assert entries['Model'].mode == 'lines+markers'
    assert entries['Model'].marker.symbol == 'circle-open'
    assert entries['Model'].line.width == 3
    assert entries['Note'].mode == 'none'
    assert entries['Key'].marker.color == 'teal'
    # No color given: the legend's text color, so it follows dark mode.
    assert entries['Test point'].marker.color == 'black'
    with redirect_stdout(io.StringIO()):
        nb.toggle_darkmode(True)
        nb.plot(x='time_s', y='cht_c')
    assert _custom_entries(nb.last_fig)['Test point'].marker.color == 'white'

    # Edit one field, rename in place, remove, list.
    nb.legend_entry('Spec limit', color='orange')
    assert nb.legend_entries['Spec limit']['linestyle'] == 'dash'
    nb.legend_entry('Spec limit', rename='Redline')
    nb.legend_entry('Note', 'reset')
    listing = nb.legend_entry()
    assert list(listing) == ['Redline', 'Test point', 'Model', 'Key']
    assert listing['Redline']['color'] == 'orange'
    listing['Redline']['color'] = 'pink'                      # a copy
    assert nb.legend_entries['Redline']['color'] == 'orange'

    # Nothing to toggle: they stay lit when the sets start hidden, and go
    # with the legend.
    nb.suppress_legends = True
    with redirect_stdout(io.StringIO()):
        nb.plot(x='time_s', y='cht_c')
    assert {tr.visible for tr in _custom_entries(nb.last_fig).values()} == {True}
    assert {tr.visible for tr in nb.last_fig.data
            if tr.name not in listing} == {'legendonly'}
    with redirect_stdout(io.StringIO()):
        nb.plot(x='time_s', y='cht_c', legend='off')
    assert not _custom_entries(nb.last_fig)

    with redirect_stdout(io.StringIO()):
        nb.reset_format('lines')
        assert nb.legend_entries
        nb.reset_format('legend_entries')
    assert nb.legend_entries == {}


def test_custom_legend_entry_checks():
    """A bad legend_entry() argument raises and stores nothing; Plotly's own
    marker and dash names are accepted alongside Matplotlib's."""
    nb = _notebook()
    nb.legend_entry('A', color='red')
    for bad in (lambda: nb.legend_entry('B', 'red'),          # not a style slot
                lambda: nb.legend_entry('B', color='not-a-color'),
                lambda: nb.legend_entry('B', marker='q'),
                lambda: nb.legend_entry('B', linestyle='~'),
                lambda: nb.legend_entry('B', markersize=0),
                lambda: nb.legend_entry('B', position='middle'),
                lambda: nb.legend_entry('A', color='blue', marker='q'),
                lambda: nb.legend_entry('A', rename='all'),
                lambda: nb.legend_entry('missing', rename='C'),
                lambda: nb.legend_entry('A', 'reset', color='red')):
        try:
            bad()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError('expected an error')
    assert nb.legend_entries == {'A': {
        'color': 'red', 'marker': None, 'linestyle': 'solid',
        'markersize': None, 'linewidth': None, 'position': 'end'}}
    nb.legend_entry('B', marker='star-triangle-up-open', linestyle='longdashdot')
    nb.legend_entry('B', linestyle='5px,3px')
    try:
        nb.legend_entries = {'C': {'color': 'not-a-color'}}
    except ValueError:
        pass
    else:
        raise AssertionError('expected an error')
    assert list(nb.legend_entries) == ['A', 'B']
    nb.legend_entry('all', 'reset')
    assert nb.legend_entries == {}
    # Assigning follows legend_entry's rule: a marker alone draws no line.
    nb.legend_entries = {'Z': {'marker': 'o'}}
    assert nb.legend_entries['Z']['marker'] == 'circle'
    assert nb.legend_entries['Z']['linestyle'] is None


def test_custom_legend_entries_session():
    """Entries are saved with the session; a session saved before they
    existed leaves the current ones alone."""
    nb = _notebook()
    nb.legend_entry('Spec limit', color='red', linestyle='--')
    nb.legend_entry('Key', position='start', marker='x')
    with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
        path = Path(tmp) / 'session.json'
        nb.plot(x='time_s', y='cht_c')
        nb.save_session(path)
        fresh = UnichartNotebook()
        fresh.load_session(path)
        assert fresh.legend_entries == nb.legend_entries
        assert list(fresh.legend_entries) == ['Spec limit', 'Key']
        assert set(_custom_entries(fresh.last_fig)) == {'Spec limit', 'Key'}

        session = json.loads(path.read_text())
        del session['notebook']['legend_entries']
        path.write_text(json.dumps(session))
        older = _notebook()
        older.legend_entry('Mine')
        older.load_session(path)
    assert list(older.legend_entries) == ['Mine']


def test_hue_and_polynomial_trend():
    def call(nb):
        nb.select(3)
        nb.hue(3, 'eta_pct')
        nb.reg_order('all', 2)
        nb.plot(x='rpm', y='torque_nm')
    _draw(call)


def test_delta_sets_plot():
    def call(nb):
        before = len(nb.list_sets(output='df'))
        nb.delta(base_idx=0, study_indices=[1, 2], align_on='time_s',
                 delta_parms=['cht_c', 'eta_pct'])
        assert len(nb.list_sets(output='df')) == before + 2
        nb.select([before, before + 1])
        nb.plot(x='time_s', y=['DL_cht_c', 'DLPCT_eta_pct'])
    _draw(call)


def test_delta_multi_key_and_match_modes():
    """align_on takes several keys (a flight envelope); match= picks nearest,
    interp, exact or positional pairing. A study 2% above a linear base reads
    exactly 2% under N-D linear interpolation, at off-grid study points."""
    f = lambda h, m: 1000 + 0.01 * h + 500 * m
    A, M = np.meshgrid(np.arange(0, 40001, 5000), np.arange(0.3, 0.91, 0.1))
    base = pd.DataFrame({'altitude': A.ravel(), 'mach_number': M.ravel()})
    base['thrust'] = f(base.altitude, base.mach_number)
    rng = np.random.default_rng(0)
    study = pd.DataFrame({'altitude': rng.uniform(2000, 38000, 30),
                          'mach_number': rng.uniform(0.35, 0.85, 30)})
    study['thrust'] = 1.02 * f(study.altitude, study.mach_number)
    with redirect_stdout(io.StringIO()):
        nb = UnichartNotebook()
        nb.set_copy_buttons(False)
        nb.load_df(base, title='base')
        nb.load_df(study, title='study')
        keys = ['altitude', 'mach_number']
        interp = nb.delta(0, 1, align_on=keys, delta_parms='thrust', match='interp')[0]
        near = nb.delta(0, 1, align_on=keys, delta_parms='thrust', keep_parms=True)[0]
        tight = nb.delta(0, 1, align_on=keys, delta_parms='thrust',
                         tolerance={'altitude': 1, 'mach_number': 0.001})[0]
        grid = nb.delta(0, 1, align_on=keys, delta_parms='thrust',
                        x_ins=[(10000, 0.5), (50000, 0.5)])[0]
        row = nb.delta(0, 1, match='row', delta_parms='thrust')[0]
    np.testing.assert_allclose(interp.df['DLPCT_thrust'], 2.0)
    assert list(interp.df.columns[:2]) == keys
    assert near.df['thrust_BASE'].isin(base['thrust']).all()
    assert tight.df['DL_thrust'].isna().all()
    assert grid.df['DLPCT_thrust'].iloc[0] == pytest.approx(2.0)
    assert np.isnan(grid.df['DLPCT_thrust'].iloc[1])     # outside the envelope
    assert len(row.df) == len(study)
    assert interp.delta_sets['align_on'] == keys
    assert interp.delta_sets['match'] == 'interp'


def test_summary_and_table_frames():
    nb = _notebook()
    with redirect_stdout(io.StringIO()):
        summary = nb.summary(cols=['rpm', 'eta_pct'], output='df')
        table = nb.table(cols=['rpm', 'eta_pct'], output='df')
    assert isinstance(summary, pd.DataFrame) and not summary.empty
    assert isinstance(table, pd.DataFrame) and not table.empty


def test_table_copy_query_button():
    """table()'s toolbar gets a Copy query button, and the page gets each
    column's query spelling and raw kind (its cells are display text by
    then); summary() shares the widget but has no data columns to query."""
    import unichart._core as core
    nb = _notebook()
    shown = []
    real_display, core.display = core.display, shown.append
    try:
        with redirect_stdout(io.StringIO()):
            nb.table(cols=['phase', 'rpm'])
            nb.summary(cols=['rpm'])
            odd = pd.DataFrame({'</script>': [1.0]})
            nb._display_html_table(odd, query_cols=nb._query_columns(odd))
    finally:
        core.display = real_display
    table_html, summary_html, odd_html = (obj.data for obj in shown)

    assert '<button class="uc-query-btn"' in table_html
    assert ('var QUERY_COLS = [{"ref": "Set", "kind": "num"}, '
            '{"ref": "Dataset", "kind": "str"}, {"ref": "phase", "kind": "str"}, '
            '{"ref": "rpm", "kind": "num"}];') in table_html
    assert '<button class="uc-query-btn"' not in summary_html
    assert 'var QUERY_COLS = null;' in summary_html
    # A column name can't close the script tag early.
    assert odd_html.count('</script>') == 1

    frame = pd.DataFrame({'rpm': [1.0, None], 'Power (kW)': [1, 2],
                          'class': ['001', None], 'ok': [True, False],
                          'when': pd.to_datetime(['2024-01-05', None])})
    assert UnichartNotebook._query_columns(frame) == [
        {'ref': 'rpm', 'kind': 'num'}, {'ref': '`Power (kW)`', 'kind': 'num'},
        {'ref': '`class`', 'kind': 'str'}, {'ref': 'ok', 'kind': 'bool'},
        {'ref': 'when', 'kind': 'other'}]


def test_copy_button_shows_with_its_figure():
    """The ⧉ copy button belongs to the figure's display, not to the plot
    call: a plot that is never shown leaves no button behind, and a figure
    shown again (display(fig), fig.show(), uc.last_fig) gets one each time."""
    import copy
    import pickle

    import plotly.io as pio
    from plotly.basedatatypes import BaseFigure

    shown = []
    real_button = UnichartNotebook.__dict__['_display_copy_button']
    real_display, real_show = BaseFigure._ipython_display_, pio.show
    UnichartNotebook._display_copy_button = staticmethod(lambda: shown.append('button'))
    BaseFigure._ipython_display_ = lambda fig: shown.append('plot')
    pio.show = lambda fig, *args, **kwargs: shown.append('plot')
    try:
        nb = _notebook()
        with redirect_stdout(io.StringIO()):
            nb.set_copy_buttons(True)
            nb.plot(x='rpm', y='torque_nm')
            fig = nb.plot(x='rpm', y='eta_pct')
        assert shown == [], 'plotting alone must display nothing'
        assert isinstance(fig, go.Figure)
        fig._ipython_display_()
        fig.show()
        assert shown == ['button', 'plot'] * 2

        # Switched to the button class in place, the figure still edits,
        # copies and pickles like any other. Checked before the next plot,
        # which empties this one to free memory (_clear_last_fig).
        fig.update_layout(title_text='edited')
        fig.data[0].name = 'renamed'
        assert fig.layout.title.text == 'edited'
        assert json.loads(fig.to_json())['data'][0]['name'] == 'renamed'
        for twin in (copy.deepcopy(fig), pickle.loads(pickle.dumps(fig))):
            assert type(twin) is type(fig)
            assert json.loads(twin.to_json()) == json.loads(fig.to_json())

        shown.clear()
        with redirect_stdout(io.StringIO()):
            nb.set_copy_buttons(False)
            plain = nb.plot(x='rpm', y='eta_pct')
        assert type(plain) is go.Figure
        plain._ipython_display_()
        assert shown == ['plot']
    finally:
        UnichartNotebook._display_copy_button = real_button
        BaseFigure._ipython_display_, pio.show = real_display, real_show


def test_styles_and_dark_mode():
    def call(nb):
        nb.set_plot_style('plotly')
        nb.toggle_darkmode(True)
        nb.set_plot_size(5, 3)                 # inches, like figsize
        nb.plot(x='time_s', y=['torque_nm', 'eta_pct'])
    _draw(call)


def test_session_round_trip():
    """save_session → load_session restores the sets and redraws the plot."""
    nb = _notebook()
    with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
        nb.select([0, 2])
        nb.color(0, 'red')
        nb.plot(x='time_s', y='rpm')
        path = Path(tmp) / 'session.json'
        nb.save_session(path)

        fresh = UnichartNotebook()
        fresh.set_copy_buttons(False)
        fresh.load_session(path)
    assert len(fresh.list_sets(output='df')) == len(nb.list_sets(output='df'))
    assert isinstance(fresh.last_fig, go.Figure)
    assert len(fresh.last_fig.data) == len(nb.last_fig.data)


GRID = dict(x='time_s', y=['rpm', 'torque_nm', 'cht_c', 'eta_pct'], ncols=2)


def _grid_domains(fig):
    """Where each panel of a 2x2 grid sits, which is what the gaps decide."""
    return [(tuple(fig.layout[f'xaxis{i}'].domain),
             tuple(fig.layout[f'yaxis{i}'].domain)) for i in ('', 2, 3, 4)]


def test_spacing_attributes_draw_like_the_arguments():
    """nb.hspace / nb.vspace draw what hspace= / vspace= draw, and a per-call
    value still wins over them."""
    builtin = _grid_domains(_draw(lambda nb: nb.plot(**GRID)))
    per_call = _grid_domains(_draw(
        lambda nb: nb.plot(**GRID, hspace=110, vspace='90px')))
    assert per_call != builtin

    def standing(nb):
        nb.hspace, nb.vspace = 110, '90px'
        nb.plot(**GRID)
    assert _grid_domains(_draw(standing)) == per_call

    def overridden(nb):
        nb.hspace, nb.vspace = 300, 300
        nb.plot(**GRID, hspace=110, vspace='90px')
    assert _grid_domains(_draw(overridden)) == per_call


# Attribute: (a good value, values its check must refuse).
PLOT_DEFAULT_ATTRS = {
    'hspace': (110, (-5, 'wide', True, [80])),
    'vspace': ('40px', (-1, 'tall')),
    'ncols': (2, (0, 1.5, True, '2')),
    'nrows': (3, (-1, 2.0)),
    'legend_scroll': (False, (0, 'no')),
    'suppress_legends': (True, (1, 'yes')),
    'title_gap': (20, (-1, 'wide', True)),
    'legend_gap': (24.5, (-3, '20px', False)),
}


def _refuses(nb, attr, bad):
    """Assigning ``bad`` raises and leaves the setting as it was."""
    before = getattr(nb, attr)
    try:
        setattr(nb, attr, bad)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError(f'nb.{attr} = {bad!r} was accepted')
    assert getattr(nb, attr) == before, f'nb.{attr} changed on a refused value'


def _session_copy(nb):
    """A fresh notebook restored from ``nb``'s saved session."""
    with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
        path = Path(tmp) / 'session.json'
        nb.save_session(path)
        fresh = UnichartNotebook()
        fresh.load_session(path)
    return fresh


def test_plot_default_attributes():
    """Each attribute is its set_default_format argument's setting: checked on
    assignment, cleared by None or reset_format('defaults'), kept by sessions."""
    nb = _notebook()
    for attr, (good, bads) in PLOT_DEFAULT_ATTRS.items():
        assert getattr(nb, attr) is None, attr
        other = _notebook()
        other.set_default_format(**{attr: good})
        assert getattr(other, attr) == good, attr
        setattr(nb, attr, good)
        assert nb._plot_defaults[attr] == good, attr
        for bad in bads:
            _refuses(nb, attr, bad)

    fresh = _session_copy(nb)
    for attr, (good, _) in PLOT_DEFAULT_ATTRS.items():
        assert getattr(fresh, attr) == good, attr
    # The store's session key is 'plot_defaults', now also a method's name:
    # restoring it must not shadow the method.
    assert fresh.plot_defaults() == {}

    nb.hspace = None
    assert nb.hspace is None and nb.vspace == '40px'
    with redirect_stdout(io.StringIO()):
        nb.reset_format('defaults')
    assert all(getattr(nb, attr) is None for attr in PLOT_DEFAULT_ATTRS)


def test_grid_and_legend_attributes_draw_like_the_arguments():
    four = dict(x='time_s', y=['rpm', 'torque_nm', 'cht_c', 'eta_pct'])
    per_call = _grid_domains(_draw(lambda nb: nb.plot(**four, ncols=1)))
    assert per_call != _grid_domains(_draw(lambda nb: nb.plot(**four)))

    def standing(nb):
        nb.ncols = 1
        nb.plot(**four)
    assert _grid_domains(_draw(standing)) == per_call

    def hidden(nb):
        nb.suppress_legends = True
        nb.plot(**four)
    assert {tr.visible for tr in _draw(hidden).data} == {'legendonly'}


def test_figsize_attribute():
    nb = _notebook()
    assert nb.figsize == (12, 8)
    nb.figsize = [10, 6]
    assert nb.figsize == (10, 6)             # stored as a tuple, like the method's
    for bad in ('big', (10,), (10, 6, 1), (10, -1), (True, 5)):
        _refuses(nb, 'figsize', bad)
    nb.set_default_format(figsize=(9, 5))
    assert nb.figsize == (9, 5)
    assert _session_copy(nb).figsize == (9, 5)
    nb.figsize = None
    assert nb.figsize == (12, 8)

    def standing(nb):
        nb.figsize = (10, 6)
        nb.plot(x='time_s', y='rpm')
    fig = _draw(standing)
    per_call = _draw(lambda nb: nb.plot(x='time_s', y='rpm', figsize=(10, 6)))
    size = (fig.layout.width, fig.layout.height)
    assert size == (per_call.layout.width, per_call.layout.height)
    assert size != (1200, 800)


def test_plot_style_attribute():
    """nb.plot_style = 'plotly' is set_plot_style('plotly'), minus the message:
    same figure, loaded sets restyled too."""
    def by_attr(nb):
        out = io.StringIO()
        with redirect_stdout(out):
            nb.plot_style = 'plotly'
        assert out.getvalue() == '', 'the attribute printed'
        nb.plot(x='time_s', y='rpm')

    def by_method(nb):
        nb.set_plot_style('plotly')
        nb.plot(x='time_s', y='rpm')
    assert _draw(by_attr).to_json() == _draw(by_method).to_json()

    nb = _notebook()
    assert nb.plot_style == 'matplotlib'
    nb.plot_style = 'plotly'
    assert nb.sets[0].color == '#636EFA'     # Plotly's palette, not tab10
    assert _session_copy(nb).plot_style == 'plotly'
    for bad in ('seaborn', 3, ['plotly']):
        _refuses(nb, 'plot_style', bad)
    nb.plot_style = 'mpl'
    assert nb.plot_style == 'matplotlib'
    nb.plot_style = 'plotly'
    nb.plot_style = None
    assert nb.plot_style == 'matplotlib'


def test_font_size_attributes():
    nb = _notebook()
    assert nb.legend_size is None
    nb.legend_size = 'large'
    other = _notebook()
    with redirect_stdout(io.StringIO()):
        other.set_font_sizes(legend='large')
    assert nb.legend_size == other.legend_size == 14.0
    assert nb.get_font_sizes()['legend'] == 14.0
    for bad in ('gigantic', 0, -3, True):
        _refuses(nb, 'legend_size', bad)

    nb.axes_tick_size = 11
    assert _session_copy(nb).axes_tick_size == 11.0
    nb.axes_tick_size = 'reset'
    assert nb.axes_tick_size is None
    with redirect_stdout(io.StringIO()):
        nb.reset_format('fonts')
    assert nb.legend_size is None

    def standing(nb):
        nb.legend_size = 'large'
        nb.plot(x='time_s', y='rpm')
    assert _draw(standing).layout.legend.font.size == 14.0

    # set_font_sizes checks once, not again through the attribute's setter.
    with warnings.catch_warnings(record=True) as caught, redirect_stdout(io.StringIO()):
        warnings.simplefilter('always')
        nb.set_font_sizes(legend=80)
    assert len([w for w in caught if 'unusually large' in str(w.message)]) == 1


def test_font_sizes_reach_hue_colorbars_and_group_titles():
    """colorbar_size reaches a numeric hue's colorbar, which lives on a layout
    coloraxis rather than a trace, and legend_size a grouped legend's titles."""
    def hue(nb):
        nb.set_font_sizes(colorbar=21, legend=17)
        nb.plot('time_s', 'cht_c', hue='rpm')
    fig = _draw(hue)
    used = {t.marker.coloraxis for t in fig.data if t.marker.coloraxis}
    assert used
    for ca in used:
        cb = fig.layout[ca].colorbar
        assert cb.tickfont.size == cb.title.font.size == 21

    def no_hue(nb):
        nb.set_font_sizes(colorbar=21)
        nb.plot('time_s', 'cht_c')
    assert not _draw(no_hue).layout.coloraxis.to_plotly_json()   # unused: left alone

    def grouped(nb):
        nb.set_font_sizes(legend=17)
        nb.plot_ymult('time_s', ['cht_c', 'egt_c'])
    legend = _draw(grouped).layout.legend
    assert legend.font.size == legend.grouptitlefont.size == 17


def test_plot_size_attribute():
    """nb.plot_size is set_plot_size's setting, in the same inches."""
    nb = _notebook()
    assert nb.plot_size is None
    nb.set_plot_size(4.6, 3.0)
    assert nb.plot_size == (4.6, 3.0)         # what was set, not ~460 px

    def by_attr(nb):
        nb.plot_size = (4.6, 3)
        nb.plot(x='time_s', y=['rpm', 'cht_c'], ncols=2)

    def by_method(nb):
        nb.set_plot_size(4.6, 3)
        nb.plot(x='time_s', y=['rpm', 'cht_c'], ncols=2)
    a, b = _draw(by_attr), _draw(by_method)
    assert (a.layout.width, a.layout.height) == (b.layout.width, b.layout.height)
    assert a.layout.width > 900               # two 4.6in panels, not 4.6px ones

    for bad in ('big', (4.6,), (4.6, 3, 1), (0, 3), (-1, 3), ('4', 3), (True, 3)):
        _refuses(nb, 'plot_size', bad)
    nb.plot_size = [5, None]                  # width only; a list is fine
    assert nb.plot_size == (5, None)
    nb.plot_size = (None, None)
    assert nb.plot_size is None

    # Per-panel vs whole grid is its own attribute, which this leaves alone.
    nb.set_plot_size(10, 6, per_subplot=False)
    nb.plot_size = (8, 5)
    assert nb.plot_size_per_subplot is False
    with redirect_stdout(io.StringIO()):
        nb.reset_format('plot_size')
    assert nb.plot_size is None and nb.plot_size_per_subplot is True


def test_plot_size_in_sessions():
    """Saved in inches under plot_size_in, and in px under plot_size, the key
    older sessions hold and an older unichart reads. Both kinds load right."""
    nb = _notebook()
    nb.plot_size = (4.6, 3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'session.json'
        with redirect_stdout(io.StringIO()):
            nb.save_session(path)
        saved = json.loads(path.read_text(encoding='utf-8'))
        state = saved['notebook']
        assert state['plot_size_in'] == [4.6, 3]
        assert state['plot_size'] == [4.6 * 100, 300]
        with redirect_stdout(io.StringIO()):
            fresh = UnichartNotebook()
            fresh.load_session(path)
        assert fresh.plot_size == (4.6, 3)

        del state['plot_size_in']             # as saved before the key existed
        state['plot_size'] = [460, None]
        path.write_text(json.dumps(saved), encoding='utf-8')
        with redirect_stdout(io.StringIO()):
            older = UnichartNotebook()
            older.load_session(path)
        assert older.plot_size == (4.6, None)


TWO = dict(x='rpm', y=['torque_nm', 'power_kw'])


def _top(call):
    """``(top margin, legend y, height)`` of the figure ``call(nb)`` draws."""
    lay = _draw(call).layout
    return lay.margin.t, lay.legend.y, lay.height


def _band_growth(titles, setup=lambda nb: None, **kwargs):
    """How much the top margin grows when the panels get ``titles``."""
    def draw(subplot_titles):
        def call(nb):
            setup(nb)
            nb.plot(**TWO, subplot_titles=subplot_titles, **kwargs)
        return _top(call)[0]
    return draw(titles) - draw(None)


def test_top_band_makes_room_for_subplot_titles():
    """The top row's subplot titles hang above the panels, into the top
    margin: it grows by their height so they clear the legend (or, without
    one, the suptitle), and a taller title — bigger font, more lines — gets
    more."""
    line = 16.7 * 1.3                        # matplotlib style's 12 pt titles
    one = _band_growth(['Torque', 'Power'])
    assert one >= line
    assert _band_growth(['Torque<br>(N·m)', 'Power']) >= one + line - 0.5
    big = _band_growth(['Torque', 'Power'],
                       setup=lambda nb: nb.set_font_sizes(subplot_title=30))
    assert big >= one + (30 - 16.7) * 1.3 - 0.5
    # A side legend: the titles clear the suptitle instead.
    assert _band_growth(['Torque<br>(N·m)', 'Power'], legend='right') >= line
    # Pinned plot area: same room, and the pinned panel height holds.
    pinned = _band_growth(['Torque', 'Power'],
                          setup=lambda nb: nb.set_plot_size(4, 2.5))
    assert pinned >= line

    def pinned_titled(nb):
        nb.set_plot_size(4, 2.5)
        nb.plot(**TWO, subplot_titles=['Torque<br>(N·m)', 'Power'])
    fig = _draw(pinned_titled)
    yax = fig.layout.yaxis.domain
    m = fig.layout.margin
    paper_h = fig.layout.height - m.t - (80 if m.b is None else m.b)
    assert abs(paper_h * (yax[1] - yax[0]) - 250) < 1


def test_title_and_legend_gaps():
    """title_gap is the space under the suptitle (the legend moves down by
    it), legend_gap the space under the legend; each grows the top margin by
    exactly the change, and 0 still keeps the subplot titles off the legend."""
    def gaps(title_gap=None, legend_gap=None):
        def call(nb):
            nb.title_gap, nb.legend_gap = title_gap, legend_gap
            nb.plot(**TWO, subplot_titles=['Torque', 'Power'], suptitle='Dyno')
        return _top(call)
    t0, y0, h = gaps()
    t1, y1, _ = gaps(title_gap=40)
    assert abs((t1 - t0) - 28) < 1e-6 and abs((y0 - y1) * h - 28) < 1e-6
    t2, y2, _ = gaps(legend_gap=40)
    assert abs((t2 - t0) - 28) < 1e-6 and y2 == y0
    t3, _, _ = gaps(title_gap=0, legend_gap=0)
    assert t0 - t3 == 24
    assert _band_growth(['Torque', 'Power'], setup=lambda nb: setattr(
        nb, 'legend_gap', 0)) >= 16.7 * 1.3


def test_session_skips_a_bad_attribute_value():
    """A saved value an attribute's check refuses is skipped with a warning;
    the rest of the session still restores."""
    nb = _notebook()
    nb.legend_size = 'large'
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'session.json'
        with redirect_stdout(io.StringIO()):
            nb.save_session(path)
        session = json.loads(path.read_text(encoding='utf-8'))
        session['notebook']['figsize'] = 'big'
        path.write_text(json.dumps(session), encoding='utf-8')
        out = io.StringIO()
        with redirect_stdout(out):
            fresh = UnichartNotebook()
            fresh.load_session(path)
    assert 'session value for figsize ignored' in out.getvalue()
    assert fresh.figsize == (12, 8)
    assert fresh.legend_size == 14.0


# ---------------------------------------------------------------------------
# Method defaults: plot_defaults() and its siblings
# ---------------------------------------------------------------------------

DEFAULTED_METHODS = PLOT_METHODS + ['summary', 'save_png']


def _call(nb, method, *args, **kwargs):
    """Run ``nb.<method>(...)`` quietly; return the arguments it recorded."""
    with redirect_stdout(io.StringIO()):
        getattr(nb, method)(*args, **kwargs)
    return nb._last_plot_call['kwargs']


def _n_panels(fig):
    return sum(1 for key in fig.layout.to_plotly_json()
               if re.fullmatch(r'xaxis\d*', key))


def test_every_defaulted_method_has_a_sister():
    """<method>_defaults takes its method's arguments, plus reset, and help()
    shows them, listed under their own category."""
    nb = _notebook()
    for method in DEFAULTED_METHODS:
        own = inspect.signature(getattr(nb, method)).parameters
        sister = inspect.signature(getattr(nb, f'{method}_defaults')).parameters
        assert [p for p in sister if p != 'reset'] == list(own), method
        assert sister['reset'].kind is inspect.Parameter.KEYWORD_ONLY, method

    out = io.StringIO()
    with redirect_stdout(out):
        nb.help('plot_defaults')
    assert "plot_defaults(x=None, y=None, by='vars', " in out.getvalue()
    assert 'reset=False' in out.getvalue()

    out = io.StringIO()
    with redirect_stdout(out):
        nb.help()
    listed = out.getvalue().split('\nMethod defaults\n')[1].split('\nStyling')[0]
    for method in DEFAULTED_METHODS:
        assert f'  • {method}_defaults(' in listed, method


def test_method_defaults_fill_only_what_a_call_leaves_out():
    nb = _notebook()
    nb.plot_defaults(x='time_s', y=['rpm', 'cht_c'], legend='right')
    nb.plot_defaults(ncols=2)                      # calls add up
    assert nb.plot_defaults() == {'x': 'time_s', 'y': ['rpm', 'cht_c'],
                                  'legend': 'right', 'ncols': 2}
    call = _call(nb, 'plot')
    assert (call['x'], call['y'], call['legend'], call['ncols']) == (
        'time_s', ['rpm', 'cht_c'], 'right', 2)

    # Passed arguments win: positionally, by keyword, and as None.
    call = _call(nb, 'plot', 'rpm', 'torque_nm', legend='off', ncols=None)
    assert (call['x'], call['y'], call['legend']) == ('rpm', 'torque_nm', 'off')
    assert 'ncols' not in call

    # ncols/nrows resolve as a pair: a passed nrows skips the stored ncols.
    call = _call(nb, 'plot', nrows=1)
    assert call['nrows'] == 1 and 'ncols' not in call

    # The positional spelling stores the same thing.
    other = _notebook()
    other.plot_defaults('time_s', 'rpm')
    assert other.plot_defaults() == {'x': 'time_s', 'y': 'rpm'}


def test_method_defaults_outrank_the_notebook_wide_ones():
    per_call = _grid_domains(_draw(lambda nb: nb.plot(**GRID)))

    def stored(nb):
        nb.ncols = 1                               # every plot method
        nb.plot_defaults(ncols=2)                  # plot alone
        nb.plot(x=GRID['x'], y=GRID['y'])
    assert _grid_domains(_draw(stored)) == per_call


def test_only_the_called_method_fills_in():
    """plot(by='ymult') hands off to plot_ymult without plot_ymult's stored
    defaults, and a method that raised leaves defaults working."""
    nb = _notebook()
    nb.plot_ymult_defaults(legend_group_by='vars')
    call = _call(nb, 'plot', x='time_s', y=['rpm', 'cht_c'], by='ymult')
    assert nb._last_plot_call['method'] == 'plot_ymult'
    assert 'legend_group_by' not in call
    assert _call(nb, 'plot_ymult')['legend_group_by'] == 'vars'

    nb.bar_defaults(agg='max')
    try:
        with redirect_stdout(io.StringIO()):
            nb.bar(x='phase', y='no_such_column')
    except ValueError:
        pass
    else:
        raise AssertionError('bar drew a column no dataset has')
    assert _call(nb, 'bar', x='phase', y='fuel_kgh')['agg'] == 'max'


def test_method_defaults_clearing_and_checks():
    nb = _notebook()
    nb.plot_defaults(legend='right', ncols=2, by='sets')
    nb.plot_defaults(legend=None, by='vars')       # plot's own defaults
    nb.plot_defaults(ncols='reset')
    assert nb.plot_defaults() == {}
    nb.plot_defaults(x='time_s', suptitle='Run')
    nb.plot_defaults(reset=True, y='rpm')          # cleared first, then stored
    assert nb.plot_defaults() == {'y': 'rpm'}

    # Stored and returned values are copies.
    cols = ['rpm']
    nb.table_defaults(cols=cols)
    cols.append('eta_pct')
    nb.table_defaults()['cols'].append('cht_c')
    assert nb.table_defaults() == {'cols': ['rpm']}

    # sig_figs and decimals are alternatives.
    nb.table_defaults(sig_figs=3)
    nb.table_defaults(decimals=2)
    assert nb.table_defaults() == {'cols': ['rpm'], 'decimals': 2}
    try:
        nb.table_defaults(sig_figs=3, decimals=2)
    except ValueError:
        pass
    else:
        raise AssertionError('stored both sig_figs and decimals')
    with redirect_stdout(io.StringIO()):
        frame = nb.table(sig_figs=4, output='df')  # skips the stored decimals
    assert isinstance(frame, pd.DataFrame)

    # What the method wouldn't take is refused, and nothing is stored.
    for bad in (lambda: nb.bar_defaults(legend='right'),
                lambda: nb.summary_defaults(1, 2, 3, 4, 5, 6, 7)):
        try:
            bad()
        except TypeError as exc:
            assert '_defaults()' in str(exc)
        else:
            raise AssertionError('took an argument its method refuses')
    assert nb.bar_defaults() == nb.summary_defaults() == {}

    # Applied-format resets leave them; resetting the defaults clears them.
    nb.box_defaults(points='all')
    with redirect_stdout(io.StringIO()):
        nb.reset_format()
    assert nb.box_defaults() == {'points': 'all'}
    with redirect_stdout(io.StringIO()):
        nb.reset_format('defaults')
    assert nb.box_defaults() == nb.plot_defaults() == nb.table_defaults() == {}
    nb.box_defaults(points='all')
    with redirect_stdout(io.StringIO()):
        nb.set_default_format(reset=True)
    assert nb.box_defaults() == {}


def test_method_defaults_in_sessions():
    """Sessions keep them, minus values with no JSON form (with a warning), and
    replay the saved plot as it was called, not with the stored defaults."""
    nb = _notebook()
    nb.plot_defaults(by='sets')
    with redirect_stdout(io.StringIO()):
        nb.plot(x='time_s', y='rpm', by='vars')    # overrides the stored by
    assert _n_panels(nb.last_fig) == 1
    nb.table_defaults(cols=['rpm'], sig_figs=3)
    nb.bar_defaults(agg=max)                       # a function: no JSON form
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'session.json'
        out = io.StringIO()
        with redirect_stdout(out):
            nb.save_session(path)
        assert 'bar_defaults(agg=...) not saved' in out.getvalue()
        session = json.loads(path.read_text(encoding='utf-8'))
        state = session['notebook']
        assert state['method_defaults'] == {
            'plot': {'by': 'sets'}, 'table': {'cols': ['rpm'], 'sig_figs': 3}}
        assert state['plot_defaults']['ncols'] is None  # the older store, as before

        with redirect_stdout(io.StringIO()):
            fresh = UnichartNotebook()
            fresh.set_copy_buttons(False)
            fresh.load_session(path)
        assert fresh.plot_defaults() == {'by': 'sets'}
        assert fresh.table_defaults() == {'cols': ['rpm'], 'sig_figs': 3}
        assert fresh.bar_defaults() == {}
        assert _n_panels(fresh.last_fig) == 1      # by='vars', as called

        # From a newer unichart: what this one can't take is skipped, alone.
        state['method_defaults'] = {'bar': {'agg': 'sum', 'legend': 'right'},
                                    'teleport': {'x': 1}}
        path.write_text(json.dumps(session), encoding='utf-8')
        out = io.StringIO()
        with redirect_stdout(out):
            newer = UnichartNotebook()
            newer.load_session(path, replay=False)
        assert 'bar_defaults(legend=) ignored' in out.getvalue()
        assert "defaults for 'teleport' ignored" in out.getvalue()
        assert newer.bar_defaults() == {'agg': 'sum'}


def _set_means(nb, idx, col):
    """``col``'s per-phase mean and row count for set ``idx``, by phase."""
    g = nb.sets[idx].cols(['phase', col]).groupby('phase')[col]
    return g.mean().to_dict(), g.count().to_dict()


def test_bar_overlay_sets():
    """overlay_sets draws the chosen sets as one marker per category (the agg
    of their own rows) in their own style, never as bars, selected or not."""
    nb = _notebook()
    nb.select(0)
    with redirect_stdout(io.StringIO()):
        fig = nb.bar(x='phase', y='fuel_kgh', overlay_sets=[1, 2])
    bars = [t for t in fig.data if t.type == 'bar']
    marks = [t for t in fig.data if t.type == 'scatter']
    assert [t.name for t in bars] == [nb.sets[0].title_format]
    assert [t.name for t in marks] == [nb.sets[1].title_format, nb.sets[2].title_format]
    for t, idx in zip(marks, (1, 2)):
        means, counts = _set_means(nb, idx, 'fuel_kgh')
        drawn = dict(zip(t.x, t.y))
        assert drawn.keys() == means.keys()
        assert all(abs(drawn[k] - means[k]) < 1e-9 for k in means)
        assert dict(zip(t.x, t.customdata)) == counts       # the set's own n
        assert t.marker.color == nb.sets[idx].color
        assert t.offsetgroup == 'overlay_sets' and t.showlegend
    assert marks[0].marker.symbol != marks[1].marker.symbol
    assert nb._last_plot_call['kwargs']['overlay_sets'] == [1, 2]

    # A selected overlay set is drawn as markers only, not as bars too.
    nb.select('all')
    with redirect_stdout(io.StringIO()):
        fig = nb.bar(x='phase', y=['fuel_kgh', 'eta_pct'], overlay_sets='1:3')
    bar_names = {t.name for t in fig.data if t.type == 'bar'}
    assert bar_names == {nb.sets[0].title_format, nb.sets[3].title_format}
    marks = [t for t in fig.data if t.type == 'scatter']
    assert len(marks) == 4                                  # 2 sets x 2 panels
    assert sum(t.showlegend for t in marks) == 2            # one entry per set


def test_bar_overlay_sets_by_sets_and_errors():
    """by='sets' repeats each overlay set on every panel over the variable's
    bar; dataset_x ignores it with a warning; bad input says why."""
    nb = _notebook()
    nb.select([0, 3])
    with redirect_stdout(io.StringIO()):
        fig = nb.bar(x='phase', y=['fuel_kgh', 'eta_pct'], by='sets', overlay_sets=1)
    marks = [t for t in fig.data if t.type == 'scatter']
    # 2 panels (sets 0 and 3; set 1 gets none) x 2 variables
    assert len(fig.layout.annotations) >= 2
    assert {a.text for a in fig.layout.annotations} >= {nb.sets[0].title_format,
                                                       nb.sets[3].title_format}
    assert nb.sets[1].title_format not in {a.text for a in fig.layout.annotations}
    assert len(marks) == 4
    assert {t.offsetgroup for t in marks} == {'var_fuel_kgh', 'var_eta_pct'}
    assert sum(t.showlegend for t in marks) == 1

    with warnings.catch_warnings(record=True) as caught, \
            redirect_stdout(io.StringIO()):
        warnings.simplefilter('always')
        nb.bar(y='fuel_kgh', by='dataset_x', overlay_sets=1)
    assert any('overlay_sets' in str(w.message) for w in caught)

    nb.select(1)
    out = io.StringIO()
    with redirect_stdout(out):
        assert nb.bar(x='phase', y='fuel_kgh', overlay_sets=1) is None
    assert 'every selected set is an overlay set' in out.getvalue()

    nb.select(0)
    nb.sets[0]['only_in_set_0'] = 1.0
    try:
        nb.bar(x='phase', y='only_in_set_0', overlay_sets=1)
    except ValueError as e:
        assert 'overlay set' in str(e)
    else:
        raise AssertionError('a y missing from every overlay set should raise')


def test_bar_overlay_sets_paired():
    """A dict pairs overlay sets with particular bar sets: their markers ride
    those sets' bars (by='vars') or panels (by='sets'); bad pairings raise;
    the pairing survives a session save (JSON turns the keys into strings)."""
    nb = _notebook()
    nb.select([0, 3])
    with redirect_stdout(io.StringIO()):
        fig = nb.bar(x='phase', y=['fuel_kgh', 'eta_pct'], overlay_sets={0: 1, 3: [1, 2]})
    groups = sorted((t.name, t.offsetgroup) for t in fig.data if t.type == 'scatter')
    s1, s2 = nb.sets[1].title_format, nb.sets[2].title_format
    # per panel: set 1 over sets 0 and 3, set 2 over set 3 -> doubled for 2 panels
    assert groups == sorted([(s1, 'set_0'), (s1, 'set_3'), (s2, 'set_3')] * 2)
    marks = [t for t in fig.data if t.type == 'scatter']
    assert sum(t.showlegend for t in marks) == 2

    with redirect_stdout(io.StringIO()):
        fig = nb.bar(x='phase', y='fuel_kgh', by='sets', overlay_sets={0: 1, 3: 2})
    on_panel = {(t.name, t.xaxis) for t in fig.data if t.type == 'scatter'}
    assert on_panel == {(s1, 'x'), (s2, 'x2')}

    for spec, text in (({2: 1}, 'is not selected'),
                       ({0: 3, 3: 1}, 'is itself an overlay set'),
                       ({0: 99}, 'matches no dataset')):
        try:
            with redirect_stdout(io.StringIO()):
                nb.bar(x='phase', y='fuel_kgh', overlay_sets=spec)
        except ValueError as e:
            assert text in str(e), (spec, str(e))
        else:
            raise AssertionError(f'{spec} should raise')

    with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
        fig = nb.bar(x='phase', y='fuel_kgh', overlay_sets={0: 1, 3: 2})
        path = Path(tmp) / 'session.json'
        nb.save_session(path, embed_data='all')
        fresh = UnichartNotebook()
        fresh.set_copy_buttons(False)
        fresh.load_session(path)
    replayed = [(t.name, t.offsetgroup) for t in fresh.last_fig.data if t.type == 'scatter']
    assert replayed == [(t.name, t.offsetgroup) for t in fig.data if t.type == 'scatter']


def test_bar_overlay_sets_follow_var_format_style():
    """An overlay set's glyph follows its y variable's var_format style (tick /
    whisker, stem linestyle and width) while keeping the set's own color; a
    whisker stems to the bar it sits on, or is a bare tick with none."""
    nb = _notebook()
    nb.select([0, 3])
    nb.var_format('fuel_kgh', style='whisker', linewidth=3)
    with redirect_stdout(io.StringIO()):
        fig = nb.bar(x='phase', y='fuel_kgh', overlay_sets={0: 1, 3: 2})
    bars = {t.name: dict(zip(t.x, t.y)) for t in fig.data if t.type == 'bar'}
    for t in (t for t in fig.data if t.type == 'scatter'):
        idx = int(t.name.split(':')[0])
        anchor = nb.sets[{1: 0, 2: 3}[idx]].title_format
        assert t.marker.symbol == 'line-ew' and t.marker.color == nb.sets[idx].color
        assert t.error_y.thickness == 3
        # The stem spans value -> paired bar top (up when above, down when below).
        for xv, v, up, down in zip(t.x, t.y, t.error_y.array, t.error_y.arrayminus):
            assert abs((v + up - down) - bars[anchor][xv]) < 1e-9

    nb.var_format('fuel_kgh', linestyle='--')
    with redirect_stdout(io.StringIO()):
        fig = nb.bar(x='phase', y='fuel_kgh', by='sets', overlay_sets={0: 1, 3: 2})
    stems = [t for t in fig.data if t.name.endswith('(stem)')]
    assert len(stems) == 2 and all(t.line.dash == 'dash' for t in stems)

    # Unpaired over two bar sets: no single bar to stem to, so a bare tick.
    with redirect_stdout(io.StringIO()):
        fig = nb.bar(x='phase', y='fuel_kgh', overlay_sets=1)
    mark, = [t for t in fig.data if t.type == 'scatter']
    assert mark.marker.symbol == 'line-ew' and mark.error_y.array is None

    nb.var_format('fuel_kgh', style='reset', linestyle='reset', linewidth='reset')
    with redirect_stdout(io.StringIO()):
        fig = nb.bar(x='phase', y='fuel_kgh', overlay_sets=1)
    mark, = [t for t in fig.data if t.type == 'scatter']
    assert mark.marker.symbol != 'line-ew'                  # back to the set's marker


def test_warnings_still_point_at_the_callers_line():
    """The defaults wrapper adds a frame, which the warnings aimed at the
    user's line step past. A DeprecationWarning only shows by default when it
    points at __main__."""
    nb = _notebook()
    nb.sets[0]['only_in_set_0'] = 1.0
    with warnings.catch_warnings(record=True) as caught, \
            redirect_stdout(io.StringIO()):
        warnings.simplefilter('always')
        nb.histogram(x='cht_c', opacity=0.5)
        nb.bar(x='phase', y='fuel_kgh', by='sets', color='red')
        nb.bar(x='phase', y='only_in_set_0')
    expected = ("'opacity' is deprecated", "bar(by='sets') colors bars",
                "'only_in_set_0' is missing")
    for text in expected:
        hits = [w for w in caught if text in str(w.message)]
        assert hits, f'no warning saying {text!r}'
        assert all(Path(w.filename).resolve() == Path(__file__).resolve()
                   for w in hits), (text, [w.filename for w in hits])


def test_board_controls_win_over_method_defaults():
    """A board panel passes its x / y / legend controls, so stored defaults
    fill only the rest."""
    nb = _notebook()
    nb.plot_defaults(x='rpm', legend='off', suptitle='Stored title')
    with redirect_stdout(io.StringIO()):
        render_panel(nb, 'plot', 'time_s', ['cht_c'], dataset_indices=[0, 1],
                     legend='right')
    call = nb._last_plot_call['kwargs']
    assert (call['x'], call['legend'], call['suptitle']) == (
        'time_s', 'right', 'Stored title')


BOARD_PANELS = [
    {'method': 'plot', 'x': 'time_s', 'y': ['rpm']},
    {'method': 'plot_ymult', 'x': 'time_s', 'y': ['rpm', 'cht_c']},
    {'method': 'plot_marginal', 'x': 'rpm', 'y': ['torque_nm']},
    {'method': 'bar', 'x': 'phase', 'y': ['fuel_kgh']},
    {'method': 'box', 'x': 'phase', 'y': ['eta_pct']},
    {'method': 'histogram', 'x': 'cht_c', 'y': []},
    {'method': 'contour', 'x': 'rpm', 'y': ['torque_nm'], 'z': 'eta_pct'},
    {'method': 'polar', 'x': 'time_s', 'y': ['rpm']},
    {'method': 'polar', 'x': 'time_s', 'y': ['rpm'], 'z': 'eta_pct'},
    {'method': 'table', 'x': None, 'y': ['rpm', 'eta_pct']},
]

# render_panel never raises: a failing panel comes back as an empty figure
# carrying this color's error annotation (dashboard._error_figure).
_ERROR_COLOR = '#c0392b'


def test_render_panel_every_method():
    """Each board panel type renders traces, not an error figure."""
    assert {p['method'] for p in BOARD_PANELS} == set(PLOT_METHODS)
    nb = _notebook()
    for spec in BOARD_PANELS:
        with redirect_stdout(io.StringIO()):
            fig = render_panel(nb, spec['method'], spec['x'], spec['y'],
                               dataset_indices=[0, 1], z=spec.get('z'))
        notes = [a.text for a in fig.layout.annotations
                 if a.font and a.font.color == _ERROR_COLOR]
        assert not notes, f"{spec['method']}: {notes}"
        assert len(fig.data) > 0, f"{spec['method']}: no traces"


def test_dashboard_to_html():
    """The static board export writes a page with every panel painted."""
    nb = _notebook()
    with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
        path = Path(tmp) / 'board.html'
        nb.dashboard_to_html(BOARD_PANELS, str(path))
        page = path.read_text(encoding='utf-8')
    assert _ERROR_COLOR not in page
    assert page.count('Plotly.newPlot') >= len(BOARD_PANELS) - 1  # table may be HTML
    # A panel whose call went wrong quietly is a newPlot with no traces.
    assert not re.search(r'Plotly\.newPlot\(\s*"[^"]*",\s*\[\]', page)


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
