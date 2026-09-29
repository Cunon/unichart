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

import io
import json
import re
import sys
import tempfile
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
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


def test_full_legend_with_reference_line():
    """The full-legend fit (what save_png and static images use) counts a
    reference line's legend entry — a shape, so plotly>=5.16."""
    def call(nb):
        nb.set_default_format(legend_scroll=False)
        nb.line('cht_c', 150, color='firebrick', legend='CHT limit')
        nb.plot(x='time_s', y='cht_c')
    _draw(call)


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


def test_summary_and_table_frames():
    nb = _notebook()
    with redirect_stdout(io.StringIO()):
        summary = nb.summary(cols=['rpm', 'eta_pct'], output='df')
        table = nb.table(cols=['rpm', 'eta_pct'], output='df')
    assert isinstance(summary, pd.DataFrame) and not summary.empty
    assert isinstance(table, pd.DataFrame) and not table.empty


def test_styles_and_dark_mode():
    def call(nb):
        nb.set_plot_style('plotly')
        nb.toggle_darkmode(True)
        nb.set_plot_size(500, 300)
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
        assert nb.plot_defaults[attr] == good, attr
        for bad in bads:
            _refuses(nb, attr, bad)

    fresh = _session_copy(nb)
    for attr, (good, _) in PLOT_DEFAULT_ATTRS.items():
        assert getattr(fresh, attr) == good, attr

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


BOARD_PANELS = [
    {'method': 'plot', 'x': 'time_s', 'y': ['rpm']},
    {'method': 'plot_ymult', 'x': 'time_s', 'y': ['rpm', 'cht_c']},
    {'method': 'plot_marginal', 'x': 'rpm', 'y': ['torque_nm']},
    {'method': 'bar', 'x': 'phase', 'y': ['fuel_kgh']},
    {'method': 'box', 'x': 'phase', 'y': ['eta_pct']},
    {'method': 'histogram', 'x': 'cht_c', 'y': []},
    {'method': 'contour', 'x': 'rpm', 'y': ['torque_nm'], 'z': 'eta_pct'},
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
