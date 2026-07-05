"""On-the-fly Dash dashboards that combine multiple unichart figures.

`unichart.py` plotting methods each return a Plotly ``go.Figure`` and cache the
most recent one in ``nb.last_fig``. This module wires those figures into an
interactive Dash board with one shared data context: a header bar owns the
dataset selection and the light/dark theme for *every* panel, and each panel
keeps only the controls that are genuinely its own (plot type, x / y / z
variables, title, legend position).

Typical use, inline in a Jupyter notebook::

    from unichart_dashboard import dashboard

    dashboard(nb, panels=[
        {'method': 'plot', 'x': 'time', 'y': 'temp'},
        {'method': 'bar',  'x': 'cat',  'y': 'val'},
    ], ncols=2, title='Test rig overview')

Every panel plots the datasets picked in the header, so the whole board always
shows one consistent slice of the data. A panel that must always show specific
datasets (e.g. a contour map of set 0 with overlay sets on top) can pin them
with ``datasets=[0]`` in its spec; pinned panels ignore the header picker and
carry a "pinned" badge so the exception is visible.

Dash is an optional dependency; it is imported lazily so importing the core
toolkit never requires it.
"""

import base64
import inspect
import re
import socket
import threading
from html import escape as _escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


# Control-owned kwargs: these come from the live UI controls, so a panel's
# passthrough ``kwargs`` must never shadow them.
_CONTROL_KEYS = {'x', 'y', 'z', 'suptitle', 'legend'}


# Dataset .select is shared, mutable notebook state. Panel renders flip it,
# read the figure, then restore it, so renders must not interleave.
_RENDER_LOCK = threading.Lock()

# Plot methods offered in the per-panel "plot type" switch. contour is included
# but needs a z column (and optionally overlay_sets) supplied via the panel's
# ``kwargs`` passthrough, e.g. kwargs={'z': 'EFF', 'overlay_sets': [1, 2]}. A
# contour panel with no z renders a graceful in-panel error. table renders a
# go.Table of the selected columns (the y control picks the columns); it has a
# different signature, so render_panel dispatches it through a dedicated branch.
PLOT_METHODS = ['plot', 'plot_ymult', 'bar', 'box', 'histogram', 'contour', 'table']

# Methods whose signature accepts a `legend=` argument (above/right/off). The
# legend control is only shown — and only passed through — for these.
_LEGEND_METHODS = {'plot', 'plot_ymult'}

# Methods that take a `z=` column (mapped to color). The z dropdown is only
# shown — and only passed through — for these.
_Z_METHODS = {'contour'}

# Methods with a different signature from the x/y plotters: handled by a
# dedicated branch in render_panel rather than the generic x/y/legend dispatch.
_TABLE_METHODS = {'table'}

LEGEND_POSITIONS = ['above', 'right', 'off']


# ---------------------------------------------------------------------------
# Theme — one set of chrome tokens for the whole board
# ---------------------------------------------------------------------------

# Figures inherit the board's UI font so chart text and chrome text read as one
# surface (unichart's plotly templates default to Open Sans, which clashes).
_FIG_FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Chrome tokens, light and dark. The dark block also remaps Dash 4's own
# component tokens (--Dash-*) so dcc dropdowns/inputs — including their Radix
# popovers, which portal to <body> — follow the board theme instead of staying
# white-on-white. Everything is keyed off <html data-theme="...">, which the
# theme switch updates clientside.
_APP_CSS = """
:root {
  --page: #f9f9f7;
  --surface: #fcfcfb;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --hairline: #e1e0d9;
  --border: rgba(11, 11, 11, 0.10);
  --accent: #2a78d6;
  --accent-weak: rgba(42, 120, 214, 0.10);
  --hover: rgba(11, 11, 11, 0.03);
  --Dash-Fill-Interactive-Strong: var(--accent);
}
html[data-theme="dark"] {
  --page: #0d0d0d;
  --surface: #1a1a19;
  --ink: #ffffff;
  --ink-2: #c3c2b7;
  --muted: #898781;
  --hairline: #2c2c2a;
  --border: rgba(255, 255, 255, 0.10);
  --accent: #3987e5;
  --accent-weak: rgba(57, 135, 229, 0.16);
  --hover: rgba(255, 255, 255, 0.05);
  --Dash-Text-Strong: rgba(255, 255, 255, 0.92);
  --Dash-Text-Primary: rgba(255, 255, 255, 0.87);
  --Dash-Text-Weak: rgba(255, 255, 255, 0.60);
  --Dash-Text-Disabled: rgba(255, 255, 255, 0.38);
  --Dash-Stroke-Strong: rgba(255, 255, 255, 0.42);
  --Dash-Stroke-Weak: rgba(255, 255, 255, 0.14);
  --Dash-Fill-Interactive-Weak: rgba(255, 255, 255, 0.06);
  --Dash-Fill-Inverse-Strong: #232322;
  --Dash-Fill-Primary-Hover: rgba(255, 255, 255, 0.05);
  --Dash-Fill-Primary-Active: rgba(255, 255, 255, 0.09);
  --Dash-Fill-Disabled: rgba(255, 255, 255, 0.12);
  --Dash-Shading-Strong: rgba(0, 0, 0, 0.55);
  --Dash-Shading-Weak: rgba(0, 0, 0, 0.35);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--page);
  color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 13px;
}
.board { padding: 14px; }
.board-header {
  display: flex; flex-wrap: wrap; align-items: center; gap: 10px 22px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 10px 14px; margin-bottom: 14px;
}
.board-title { font-size: 15px; font-weight: 650; margin-right: 6px; }
.board-group { display: flex; align-items: center; gap: 8px; }
.board-actions { margin-left: auto; display: flex; align-items: center; gap: 8px; }
.control-label {
  font-size: 10.5px; font-weight: 600; letter-spacing: 0.05em;
  text-transform: uppercase; color: var(--muted);
}
.chip-list { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.chip-list label {
  display: inline-flex; align-items: center; gap: 6px; margin: 0;
  padding: 3px 10px; border: 1px solid var(--hairline); border-radius: 999px;
  font-size: 12.5px; color: var(--ink-2); cursor: pointer; user-select: none;
  transition: border-color 120ms, background 120ms;
}
.chip-list label:hover { border-color: var(--muted); background: var(--hover); }
.chip-list label:has(input:checked) {
  border-color: var(--accent); background: var(--accent-weak); color: var(--ink);
}
.chip-list label:has(input:focus-visible) {
  outline: 2px solid var(--accent); outline-offset: 1px;
}
.chip-list input { position: absolute; opacity: 0; width: 1px; height: 1px; margin: 0; }
.chip-dot {
  width: 9px; height: 9px; border-radius: 50%; flex: none;
  box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.15);
}
.btn {
  font: inherit; font-size: 12px; color: var(--ink-2); background: transparent;
  border: 1px solid var(--hairline); border-radius: 7px; padding: 4px 11px;
  cursor: pointer; transition: border-color 120ms, background 120ms;
}
.btn:hover { color: var(--ink); border-color: var(--muted); background: var(--hover); }
.board-grid { display: grid; gap: 14px; }
.panel-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 10px 12px 12px;
}
.panel-head { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.panel-title-input {
  flex: 1 1 auto; min-width: 60px; font: inherit; font-size: 14px; font-weight: 650;
  color: var(--ink); background: transparent; border: none;
  border-bottom: 1px dashed transparent; padding: 2px 1px;
}
.panel-title-input:hover { border-bottom-color: var(--hairline); }
.panel-title-input:focus { outline: none; border-bottom-color: var(--accent); }
.panel-title-input::placeholder { color: var(--muted); font-weight: 500; }
/* Locked board: the title is display-only — no edit affordance, no caret. */
.panel-title-input.locked { cursor: default; }
.panel-title-input.locked:hover { border-bottom-color: transparent; }
.panel-title-input.locked::placeholder { color: transparent; }
.pin-badge {
  flex: none; font-size: 10.5px; color: var(--muted);
  border: 1px dashed var(--hairline); border-radius: 999px; padding: 1px 8px;
  white-space: nowrap;
}
.panel-controls { display: flex; flex-wrap: wrap; gap: 6px 10px; margin-bottom: 8px; }
.control { display: flex; flex-direction: column; gap: 3px; min-width: 108px; }
.hidden { display: none !important; }
.dash-dropdown { font-size: 12.5px; }
.panel-graph-wrap { border-top: 1px solid var(--hairline); padding-top: 6px; }
/* Static-export data tables: to_html renders table panels as real HTML
   tables (sortable, theme-token colors); the live board still draws
   go.Table, so these rules are inert there. */
.table-wrap {
  overflow: auto; border: 1px solid var(--hairline); border-radius: 6px;
  margin-top: 6px;
}
.data-table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
.data-table th, .data-table td {
  padding: 4px 10px; text-align: left; border-bottom: 1px solid var(--hairline);
}
.data-table th {
  position: sticky; top: 0; background: var(--surface); cursor: pointer;
  user-select: none; font-size: 10.5px; font-weight: 600;
  letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted);
  white-space: nowrap;
}
.data-table th:hover { color: var(--ink); }
.data-table th[data-dir="asc"]::after { content: " ▲"; font-size: 8px; }
.data-table th[data-dir="desc"]::after { content: " ▼"; font-size: 8px; }
.data-table th.num, .data-table td.num { text-align: right; }
.data-table tbody tr:last-child td { border-bottom: none; }
.data-table tbody tr:hover { background: var(--hover); }
"""

# Dash requires the {%...%} placeholders verbatim, and f-strings would choke on
# them, so the page template is assembled with plain .replace() below.
_INDEX_TEMPLATE = """<!DOCTYPE html>
<html data-theme="__THEME__">
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
__CSS__
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


def _index_string(initial_theme):
    """The app's HTML shell: chrome CSS inlined, initial theme baked into the
    <html> tag so a dark board never flashes light while the page boots."""
    return (_INDEX_TEMPLATE
            .replace('__THEME__', initial_theme)
            .replace('__CSS__', _APP_CSS))


def _pick_port(preferred):
    """Return a usable localhost port, preferring ``preferred``.

    Re-running a dashboard cell leaves the previous server bound to its port, so
    reusing the same number raises ``Address already in use``. If ``preferred`` is
    free we keep it; otherwise the OS hands us any open port (bind to 0)."""
    for candidate in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(('127.0.0.1', candidate))
                return s.getsockname()[1]
            except OSError:
                continue
    return preferred  # pragma: no cover - both binds failing is pathological


def _require_dash():
    """Import Dash lazily, with a friendly error if it isn't installed."""
    try:
        import dash  # noqa: F401
        from dash import (Dash, dcc, html, dash_table, no_update,
                          Input, Output, State, MATCH, ALL)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "unichart_dashboard requires Dash. Install it with "
            "`pip install dash` (or add it to requirements.txt)."
        ) from exc
    return (Dash, dcc, html, dash_table, no_update,
            Input, Output, State, MATCH, ALL)


def _all_columns(nb):
    """Sorted union of column names across every dataset on the notebook."""
    cols = set()
    for ds in nb.sets:
        cols.update(str(c) for c in ds.columns)
    return sorted(cols)


def _selected_indices(nb):
    """Indices of currently-selected datasets."""
    return [ds.index for ds in nb.sets if ds.select]


def _passthrough_kwargs(method_fn, extra):
    """Filter a panel's passthrough ``kwargs`` to what ``method_fn`` will accept.

    Panels carry method-specific options (``nbins``, ``barmode``, ``points``,
    ``by``, ...). Because the plot-type switch can change the method at runtime, a
    kwarg meant for one method (e.g. ``nbins`` for histogram) must be silently
    dropped for another (e.g. bar) instead of raising. We keep a kwarg only if the
    target method names it, or accepts ``**kwargs``. Control-owned keys (x/y/
    suptitle/legend) are never taken from here."""
    if not extra:
        return {}
    params = inspect.signature(method_fn).parameters
    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                         for p in params.values())
    return {k: v for k, v in extra.items()
            if k not in _CONTROL_KEYS and (accepts_var_kw or k in params)}


def render_panel(nb, method, x, y, dataset_indices, suptitle=None, legend='above',
                 size=None, extra=None, z=None, darkmode=None):
    """Render one panel to a ``go.Figure`` using the notebook's plot methods.

    Dispatches to ``nb.<method>`` with the given x/y/suptitle (and legend for the
    methods that support it) and reads the figure from ``nb.last_fig`` (so it
    works even under ``static_images`` mode). On any error an empty figure
    carrying the error text is returned so one bad panel can't take down the
    board.

    ``extra`` is an optional dict of method-specific passthrough kwargs (e.g.
    ``{'nbins': 20}`` for histogram, ``{'barmode': 'stack', 'agg': 'sum'}`` for
    bar, ``{'by': 'sets'}`` for plot). It is filtered to what the current method
    accepts (see :func:`_passthrough_kwargs`) so it stays valid across plot-type
    switches and never overrides the live x/y/suptitle/legend controls.

    ``darkmode`` overrides ``nb.darkmode`` for this render only (None leaves the
    notebook's setting untouched) — it is how the board's theme switch restyles
    every figure without permanently flipping the user's notebook.

    The notebook is left exactly as it was found. Rendering a panel would
    otherwise mutate shared state: ``.select`` (which datasets are active),
    ``darkmode`` and the ``last_*`` plot memory (``last_x``/``last_y``/
    ``last_fig``/...). All of it is snapshotted and restored in ``finally``, so
    interacting with the board never changes what a later ``nb.plot()`` cell
    does. ``nb.last_fig`` is additionally cleared before dispatch so the
    notebook's own ``_clear_last_fig`` cannot gut the user's previously cached
    figure in place.

    ``size`` is an optional ``(width_px, height_px)``. When given, it is stamped
    onto the figure as an explicit, non-autosize dimension. This matters for the
    inline-Jupyter board: a freshly-created Dash iframe has zero size at first
    paint, so an autosize/responsive figure draws a 0x0 (blank) canvas until an
    interaction triggers a resize. An explicit size paints correctly on load.

    This is the pure core of the Dash callback and is callable directly in tests.
    """
    if method not in PLOT_METHODS:
        return _stamp(_error_figure(f"Unknown plot method: {method!r}"), size)

    chosen = set(dataset_indices or [])
    if not chosen:
        return _stamp(_error_figure(
            "No datasets selected — pick one in the header."), size)

    with _RENDER_LOCK:
        # Snapshot every piece of notebook state a render touches, so the user's
        # notebook is untouched after the board runs.
        select_snapshot = [(ds, ds.select) for ds in nb.sets]
        darkmode_snapshot = nb.darkmode
        state_snapshot = {k: getattr(nb, k) for k in vars(nb)
                          if k.startswith('last_')}
        try:
            for ds in nb.sets:
                ds.select = ds.index in chosen
            if darkmode is not None:
                nb.darkmode = bool(darkmode)
            # Null last_fig first so the method's _clear_last_fig doesn't empty
            # (data=[], layout={}) the figure the user had cached before the board.
            nb.last_fig = None

            # The y control is multi-select (a list), but some methods (e.g.
            # histogram) require a scalar y. Unwrap a single selection so every
            # method works; keep the list only when genuinely multi-valued.
            if isinstance(y, (list, tuple)) and len(y) == 1:
                y = y[0]

            method_fn = getattr(nb, method)
            kwargs = _passthrough_kwargs(method_fn, extra)

            if method in _TABLE_METHODS:
                # table() has a different signature from the x/y plotters: the
                # multi-select y control names the columns to show (cols), the x
                # control is the interpolation x-axis (x_col; inert in raw mode),
                # and suptitle is the table title. output='fig' returns the
                # styled go.Table figure (sig_figs / dark-mode already applied)
                # without table()'s HTML display side effect. Controls win over
                # the passthrough kwargs, and output is forced last so a stray
                # output= in a panel's kwargs can't reintroduce the display call.
                if y:
                    kwargs['cols'] = y
                if x:
                    kwargs['x_col'] = x
                if suptitle:
                    kwargs['title'] = suptitle
                kwargs['output'] = 'fig'
                fig = method_fn(**kwargs)
                if fig is None:
                    return _stamp(_error_figure(
                        "No table data — pick column(s) under y."), size)
                if not suptitle:
                    fig.update_layout(title_text='')
                return _stamp(fig, size)

            kwargs['x'] = x
            kwargs['y'] = y
            if method in _Z_METHODS and z:
                kwargs['z'] = z
            if suptitle:
                kwargs['suptitle'] = suptitle
            if method in _LEGEND_METHODS:
                kwargs['legend'] = legend

            method_fn(**kwargs)
            fig = nb.last_fig
            if fig is None:
                return _stamp(_error_figure("No figure produced (no data / selection?)"), size)
            if not suptitle:
                # In the board, the card header owns the panel title. With no
                # explicit suptitle, unichart still auto-titles the figure
                # ("x vs [y]"); blank it so the title isn't said twice.
                fig.update_layout(title_text='')
            # No defensive copy: the finally below restores nb.last_fig to the
            # snapshot, so this figure is detached from the notebook the moment we
            # return. _clear_last_fig only guts whatever nb.last_fig points at, so
            # it can never reach this one. (Skipping go.Figure(fig) avoids deep-
            # copying every trace, which is the per-render cost on large data.)
            return _stamp(fig, size)
        except Exception as exc:  # noqa: BLE001 - surface any plotting error in-panel
            return _stamp(_error_figure(f"{type(exc).__name__}: {exc}"), size)
        finally:
            for ds, was in select_snapshot:
                ds.select = was
            nb.darkmode = darkmode_snapshot
            # Drop any last_* attribute the render created (e.g. contour's
            # last_z on a notebook that had never plotted a contour), then
            # restore the snapshotted values.
            for k in [k for k in vars(nb)
                      if k.startswith('last_') and k not in state_snapshot]:
                delattr(nb, k)
            for k, v in state_snapshot.items():
                setattr(nb, k, v)


def _stamp(fig, size):
    """Stamp board styling onto a figure so every panel reads as one surface.

    - Explicit (width, height): unichart bakes figsize into width/height via
      _base_layout; we overwrite it with the panel size so panels paint at a
      known size regardless of their container (with ``size=None`` the figure
      keeps whatever dimensions it already has).
    - Transparent paper: the card's surface color shows through the figure
      margins, so light/dark figures sit flush on light/dark cards.
    - The board's UI font, so chart text matches the chrome.
    """
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      font_family=_FIG_FONT)
    if size is not None:
        fig.update_layout(autosize=False, width=size[0], height=size[1])
    return fig


def _error_figure(message):
    """A blank figure that displays an error message in the panel."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       xref='paper', yref='paper', x=0.5, y=0.5,
                       font=dict(color='#c0392b'))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=20, r=20, t=20, b=20))
    return fig


def _placeholder(size):
    """An empty, correctly-sized figure for a panel's initial state.

    Panels are not rendered at build time; the initial callback paints each one
    (and the dcc.Loading spinner covers the gap). This avoids the load-time double
    render — once at build, once via the callback — which is costly on large data.
    The placeholder is sized so the panel doesn't collapse before its figure lands."""
    fig = go.Figure()
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor='rgba(0,0,0,0)')
    return _stamp(fig, size)


def _decode_array(v):
    """Decode a Plotly value into a plain Python list / ndarray.

    Plotly (v6) JSON-encodes numeric arrays as base64 "typed arrays"
    (``{'dtype': 'f8', 'bdata': '...', 'shape': '2, 3'}``); categorical/string
    arrays stay as plain lists. A figure read from a dcc.Graph ``State`` therefore
    carries this mix, so the exporter must decode the typed-array form."""
    if isinstance(v, dict) and 'bdata' in v:
        arr = np.frombuffer(base64.b64decode(v['bdata']), dtype=np.dtype(v['dtype']))
        shape = v.get('shape')
        if shape:
            arr = arr.reshape([int(s) for s in str(shape).split(',')])
        return arr
    return v


def _is_2d(seq):
    """True if seq is a 2-D ndarray or a sequence of sequences (contour z grid)."""
    if isinstance(seq, np.ndarray):
        return seq.ndim == 2
    return (isinstance(seq, (list, tuple)) and len(seq) > 0
            and isinstance(seq[0], (list, tuple, np.ndarray)))


def _figure_to_dataframe(figure):
    """Flatten a (JSON) figure's traces into one tidy DataFrame for CSV export.

    Each trace contributes rows tagged with a ``trace`` column. 1-D traces
    (scatter / line / bar / box / histogram, and contour overlays) export their
    ``x``/``y``/``z`` arrays column-wise; a 2-D ``z`` (contour grid) is unrolled to
    long ``(x, y, z)`` rows over the grid. Traces with no data are skipped."""
    frames = []
    for i, tr in enumerate(figure.get('data', []) if figure else []):
        name = tr.get('name') or tr.get('type') or f'trace{i}'
        x = _decode_array(tr.get('x'))
        y = _decode_array(tr.get('y'))
        z = _decode_array(tr.get('z'))
        if _is_2d(z):
            xs = x if x is not None else list(range(len(z[0]) if len(z) else 0))
            ys = y if y is not None else list(range(len(z)))
            rows = [(xj, yi, zrow[j])
                    for yi, zrow in zip(ys, z)
                    for j, xj in enumerate(xs)]
            df = pd.DataFrame(rows, columns=['x', 'y', 'z'])
        else:
            cols = {k: list(v) for k, v in (('x', x), ('y', y), ('z', z))
                    if v is not None}
            if not cols:
                continue
            n = max(len(v) for v in cols.values())
            cols = {k: list(v) + [None] * (n - len(v)) for k, v in cols.items()}
            df = pd.DataFrame(cols)
        df.insert(0, 'trace', name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# Trace legendgroups that encode the source dataset index: the by-vars
# plotters stamp group_{i}, plot_ymult stamps set_{i} (its default 'sets'
# grouping), contour stamps overlay_{i} on overlay sets. The static export's
# global dataset filter matches these to map traces back to datasets.
_DATASET_TAG_RE = re.compile(r'^(?:group_|set_|overlay_)(\d+)$')

# Client-side half of the static export's global dataset filter: the header
# chips toggle trace visibility across every filterable panel (marked
# data-gsel) via Plotly.restyle. TAG must stay in sync with _DATASET_TAG_RE.
# Panels are rendered with *every* dataset, so init() hides the ones that were
# unselected at export time; it waits for window load so plotly.js has painted
# the figures first.
_GLOBAL_SELECT_JS = r"""
(function () {
  var TAG = /^(?:group_|set_|overlay_)(\d+)$/;
  function apply() {
    var on = {};
    document.querySelectorAll('.gsel input').forEach(function (el) {
      if (el.checked) { on[el.value] = true; }
    });
    document.querySelectorAll('[data-gsel] .plotly-graph-div')
      .forEach(function (gd) {
        if (!gd.data) { return; }
        var idx = [], vis = [];
        gd.data.forEach(function (tr, i) {
          var m = TAG.exec(tr.legendgroup || '');
          if (m) { idx.push(i); vis.push(on[m[1]] === true); }
        });
        if (idx.length) { Plotly.restyle(gd, {visible: vis}, idx); }
      });
  }
  document.querySelectorAll('.gsel input').forEach(function (el) {
    el.addEventListener('change', apply);
  });
  function init() {
    if (document.querySelector('.gsel input:not(:checked)')) { apply(); }
  }
  if (document.readyState === 'complete') { init(); }
  else { window.addEventListener('load', init); }
})();
"""


# Click-to-sort for the static export's HTML tables: clicking a header sorts
# by that column (toggling asc/desc, arrow via the th's data-dir attribute).
# Columns marked .num (every cell parses as a number) sort numerically;
# missing values ('-', the table's NaN fill) always sort last.
_SORTABLE_TABLE_JS = r"""
(function () {
  document.querySelectorAll('table.data-table th').forEach(function (th) {
    th.addEventListener('click', function () {
      var table = th.closest('table');
      var dir = th.dataset.dir === 'asc' ? 'desc' : 'asc';
      table.querySelectorAll('th').forEach(function (h) {
        delete h.dataset.dir;
      });
      th.dataset.dir = dir;
      var col = th.cellIndex;
      var numeric = th.classList.contains('num');
      var rows = Array.prototype.slice.call(table.tBodies[0].rows);
      rows.sort(function (ra, rb) {
        var a = ra.cells[col].textContent.trim();
        var b = rb.cells[col].textContent.trim();
        var ma = (a === '' || a === '-'), mb = (b === '' || b === '-');
        if (ma || mb) { return ma && mb ? 0 : (ma ? 1 : -1); }
        var res = numeric ? Number(a) - Number(b) : a.localeCompare(b);
        return dir === 'asc' ? res : -res;
      });
      rows.forEach(function (r) { table.tBodies[0].appendChild(r); });
    });
  });
})();
"""


def _table_columns(trace):
    """(header names, column value lists) from a ``go.Table`` trace.

    Strips the ``<b>`` wrappers ``table()`` puts around header text. This is
    the shared extraction step for both table displays built from the trace:
    the static export's HTML table and the live board's DataTable."""
    headers = [re.sub(r'</?b>', '', str(h)) for h in (trace.header.values or [])]
    columns = [list(col) for col in (trace.cells.values or [])]
    return headers, columns


def _numish(v):
    """True if a table cell reads as a number (missing markers count, so an
    otherwise-numeric column with NaN holes still sorts numerically)."""
    s = str(v).strip()
    if s in ('', '-'):
        return True
    try:
        float(s)
        return True
    except ValueError:
        return False


def _table_records(trace, sig_figs=None):
    """DataTable ``(data, columns)`` for a live-board table panel.

    Built from the same ``go.Table`` trace the panel used to display, so the
    content matches ``table()`` exactly. Numeric-looking columns are converted
    to real numbers and typed ``'numeric'`` — with ``sig_figs`` applied the
    trace's cells are pre-formatted *strings*, which DataTable's native sort
    would order lexically ("9.5" > "10.2"). The d3 ``r`` (significant-digit
    decimal) format re-applies the same rounding for display. ``'-'`` (the
    table's NaN fill) stays text and shows as-is.
    """
    from dash.dash_table.Format import Format, Scheme

    headers, columns = _table_columns(trace)
    specs, converted = [], []
    for h, col in zip(headers, columns):
        spec = {'name': h, 'id': h}
        if col and all(_numish(v) for v in col):
            spec['type'] = 'numeric'
            # table() applies sig_figs to float columns only (ints pass
            # through untouched), so only re-apply the display rounding where
            # it did: a pure-int column keeps its plain rendering.
            if sig_figs and not all(isinstance(v, (int, np.integer))
                                    for v in col):
                spec['format'] = Format(precision=sig_figs,
                                        scheme=Scheme.decimal)
            col = [float(v) if isinstance(v, str) and v.strip() not in ('', '-')
                   else v for v in col]
        specs.append(spec)
        converted.append(col)
    data = [dict(zip(headers, row)) for row in zip(*converted)]
    return data, specs


def _table_html(trace, width, height):
    """Render a ``go.Table`` trace as a sortable HTML table for the export.

    A plotly table is a canvas: its headers can't take clicks, so it can't be
    sorted. The static export therefore shows table panels as real ``<table>``
    markup — ``_SORTABLE_TABLE_JS`` adds click-to-sort headers, the theme
    tokens style it (so it follows light/dark better than go.Table's baked
    colors), and it scrolls inside its card past ``height``. Cell values come
    straight from the trace, so ``table()``'s sig_figs / NaN-fill formatting
    carries over unchanged.
    """
    headers, columns = _table_columns(trace)
    num = [bool(col) and all(_numish(v) for v in col) for col in columns]

    def _cell(tag, j, text):
        cls = ' class="num"' if j < len(num) and num[j] else ''
        return f'<{tag}{cls}>{_escape(text)}</{tag}>'

    ths = ''.join(_cell('th', j, h) for j, h in enumerate(headers))
    rows = []
    for row in zip(*columns):
        tds = ''.join(_cell('td', j, str(v)) for j, v in enumerate(row))
        rows.append(f'<tr>{tds}</tr>')
    return (f'<div class="table-wrap" style="width: {width}px; '
            f'max-height: {height}px">'
            f'<table class="data-table"><thead><tr>{ths}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _normalize_y(y):
    """Y control is multi-select; normalize seeds to a list of strings."""
    if y is None:
        return []
    if isinstance(y, (list, tuple)):
        return [str(v) for v in y]
    return [str(y)]


def build_app(nb, panels, ncols=2, width=600, height=420, title=None,
              controls=True):
    """Build (but do not run) the Dash app for the given panels.

    Returns the configured ``dash.Dash`` instance with its layout and callbacks
    registered. Factored out of :func:`dashboard` so the app/layout can be
    constructed and inspected without starting a server.

    The board has one shared data context: the header's dataset picker and
    light/dark switch are Inputs to every panel's render, so all panels always
    show the same dataset selection and theme. A panel spec with a ``datasets``
    key is pinned to those indices instead (it ignores the picker and shows a
    badge).

    ``controls`` toggles a locked-down, presentation view. With ``controls=False``
    every editing affordance is hidden — the per-panel plot-type / x / y / z /
    legend dropdowns, the export buttons, the header's dataset picker and theme
    switch, and title editing — leaving a clean grid of titled figure cards that
    render exactly the seeded panels. The controls stay in the DOM (hidden, not
    removed) so the render/export callback graph is untouched; the figures paint
    from their seeded values just as they would interactively. It is a clean
    presentation layer, not a security boundary.

    ``width`` / ``height`` are the explicit px size of each panel's figure. They
    are stamped onto every figure so panels paint reliably inline (see
    :func:`render_panel`) instead of collapsing to 0x0 in a fresh Dash iframe.
    """
    (Dash, dcc, html, dash_table, no_update,
     Input, Output, State, MATCH, ALL) = _require_dash()

    if not nb.sets:
        raise ValueError("The notebook has no datasets loaded.")
    if not panels:
        raise ValueError("Provide at least one panel.")

    col_options = _all_columns(nb)
    default_selected = _selected_indices(nb)
    size = (width, height)
    initial_theme = 'dark' if getattr(nb, 'darkmode', False) else 'light'
    board_title = title or 'unichart dashboard'

    header = _header_div(html, dcc, nb, board_title, default_selected,
                         initial_theme, controls)
    grid = html.Div(
        [_panel_div(html, dcc, dash_table, i, panel, col_options, size,
                    controls)
         for i, panel in enumerate(panels)],
        className='board-grid',
        style={'gridTemplateColumns': f'repeat({ncols}, max-content)'},
    )

    app = Dash(__name__, title=board_title)
    app.index_string = _index_string(initial_theme)
    app.layout = html.Div([header, grid], className='board')

    @app.callback(
        Output({'type': 'panel-graph', 'index': MATCH}, 'figure'),
        # Table panels display in a DataTable (native, sortable) instead of
        # the graph; the render callback fills whichever the method needs and
        # flips which wrapper is hidden.
        Output({'type': 'panel-table', 'index': MATCH}, 'data'),
        Output({'type': 'panel-table', 'index': MATCH}, 'columns'),
        Output({'type': 'panel-graph-box', 'index': MATCH}, 'className'),
        Output({'type': 'panel-table-wrap', 'index': MATCH}, 'className'),
        Input({'type': 'panel-method', 'index': MATCH}, 'value'),
        Input({'type': 'panel-x', 'index': MATCH}, 'value'),
        Input({'type': 'panel-y', 'index': MATCH}, 'value'),
        Input({'type': 'panel-z', 'index': MATCH}, 'value'),
        Input({'type': 'panel-legend', 'index': MATCH}, 'value'),
        # The board-level data context: every panel re-renders when the shared
        # dataset selection or theme changes, so the whole board stays on one
        # slice of the data.
        Input('board-datasets', 'value'),
        Input('board-theme', 'value'),
        # Per-panel passthrough kwargs and the dataset pin are fixed at build
        # time (no control), so they ride along as State.
        State({'type': 'panel-kwargs', 'index': MATCH}, 'data'),
        State({'type': 'panel-pin', 'index': MATCH}, 'data'),
        # Initial-call-enabled on purpose: on load the callback paints each panel
        # once (over the placeholder), via the same path used on interaction.
        #
        # The panel title is deliberately NOT an input: it lives in the card
        # header (chrome), not inside the figure, so editing it must not cost a
        # re-render. It only names CSV exports (read there as State).
    )
    def _update_panel(method, x, y, z, legend, board_datasets, theme,
                      extra, pin):
        datasets = pin if pin is not None else board_datasets
        fig = render_panel(nb, method, x, y, datasets, suptitle=None,
                           legend=legend, size=size, extra=extra, z=z,
                           darkmode=(theme == 'dark'))
        if (method in _TABLE_METHODS and fig.data
                and fig.data[0].type == 'table'):
            data, columns = _table_records(
                fig.data[0], sig_figs=(extra or {}).get('sig_figs'))
            return no_update, data, columns, 'hidden', ''
        # Non-table methods — and a table render that *failed* (render_panel
        # returns an error figure, not a table trace) — paint the graph.
        return fig, no_update, no_update, '', 'hidden'

    # Keep the page's data-theme attribute in sync with the theme switch. On
    # <html> (not the app div) because Dash 4's dropdown popovers portal to
    # <body> and must still inherit the theme tokens.
    app.clientside_callback(
        "function(theme){ document.documentElement.setAttribute("
        "'data-theme', theme || 'light'); return ''; }",
        Output('board-theme-sink', 'children'),
        Input('board-theme', 'value'),
    )

    # Show the z / legend controls only for plot types that accept them.
    # Clientside so toggling is instant and never queues behind a render on
    # the lock.
    app.clientside_callback(
        "function(method){ return method === 'contour' ? "
        "'control' : 'control hidden'; }",
        Output({'type': 'panel-z-wrap', 'index': MATCH}, 'className'),
        Input({'type': 'panel-method', 'index': MATCH}, 'value'),
    )
    app.clientside_callback(
        "function(method){ return (method === 'plot' || method === 'plot_ymult')"
        " ? 'control' : 'control hidden'; }",
        Output({'type': 'panel-legend-wrap', 'index': MATCH}, 'className'),
        Input({'type': 'panel-method', 'index': MATCH}, 'value'),
    )

    # Export the panel's plotted data as CSV. Reads the *currently displayed*
    # figure or table via State (no re-render, no lock): a table panel exports
    # its DataTable records (a go.Table figure has no x/y/z to flatten), any
    # other panel flattens its figure's traces.
    @app.callback(
        Output({'type': 'panel-download', 'index': MATCH}, 'data'),
        Input({'type': 'panel-export', 'index': MATCH}, 'n_clicks'),
        State({'type': 'panel-graph', 'index': MATCH}, 'figure'),
        State({'type': 'panel-table', 'index': MATCH}, 'data'),
        State({'type': 'panel-method', 'index': MATCH}, 'value'),
        State({'type': 'panel-suptitle', 'index': MATCH}, 'value'),
        prevent_initial_call=True,
    )
    def _export_panel(n_clicks, figure, table_data, method, suptitle):
        if not n_clicks:
            return None
        if method in _TABLE_METHODS:
            df = pd.DataFrame(table_data or [])
        else:
            df = _figure_to_dataframe(figure)
        stem = re.sub(r'[^\w.-]+', '_', (suptitle or 'panel').strip()) or 'panel'
        return dcc.send_data_frame(df.to_csv, f'{stem}.csv', index=False)

    # Board-level export: stack every panel's flattened data into one CSV, with a
    # `panel` column naming each (its suptitle, else its position).
    @app.callback(
        Output('board-download', 'data'),
        Input('board-export', 'n_clicks'),
        State({'type': 'panel-graph', 'index': ALL}, 'figure'),
        State({'type': 'panel-table', 'index': ALL}, 'data'),
        State({'type': 'panel-method', 'index': ALL}, 'value'),
        State({'type': 'panel-suptitle', 'index': ALL}, 'value'),
        prevent_initial_call=True,
    )
    def _export_board(n_clicks, figures, tables, methods, suptitles):
        if not n_clicks:
            return None
        frames = []
        for i, (figure, table_data, method, suptitle) in enumerate(
                zip(figures, tables, methods, suptitles)):
            if method in _TABLE_METHODS:
                df = pd.DataFrame(table_data or [])
            else:
                df = _figure_to_dataframe(figure)
            if df.empty:
                continue
            df.insert(0, 'panel', (suptitle or '').strip() or f'panel {i}')
            frames.append(df)
        if not frames:
            return None
        combined = pd.concat(frames, ignore_index=True)
        return dcc.send_data_frame(combined.to_csv, 'dashboard_data.csv', index=False)

    return app


def _dataset_chip(html, ds):
    """A dataset option label: the dataset's plot color as a dot, then its name.

    The dot uses ``ds.color`` — the same color unichart gives the dataset's
    traces — so the header selection maps visually onto the curves in every
    panel."""
    return html.Span(
        [html.Span(className='chip-dot', style={'background': ds.color}),
         html.Span(ds.title_format)],
        style={'display': 'inline-flex', 'alignItems': 'center', 'gap': '6px'},
    )


def _header_div(html, dcc, nb, board_title, default_selected, initial_theme,
                controls=True):
    """The board header: title, the shared dataset picker, the theme switch and
    the whole-board CSV export. This is the single place data selection and
    display style are controlled for every (unpinned) panel.

    ``controls=False`` (locked board) hides the picker, theme switch and export
    button so only the board title shows. The components stay in the DOM — hidden,
    not removed — so they keep seeding the render callback (the panels still paint
    the default selection and theme)."""
    grp = 'board-group' if controls else 'board-group hidden'
    datasets = dcc.Checklist(
        id='board-datasets',
        options=[{'label': _dataset_chip(html, ds), 'value': ds.index}
                 for ds in nb.sets],
        value=list(default_selected),
        className='chip-list',
    )
    theme = dcc.RadioItems(
        id='board-theme',
        options=[{'label': 'light', 'value': 'light'},
                 {'label': 'dark', 'value': 'dark'}],
        value=initial_theme,
        className='chip-list',
    )
    return html.Div(
        [
            html.Div(board_title, className='board-title'),
            html.Div([html.Span('datasets', className='control-label'), datasets],
                     className=grp),
            html.Div([html.Span('theme', className='control-label'), theme],
                     className=grp),
            html.Div(
                [
                    html.Button('⬇ export all panels', id='board-export',
                                n_clicks=0,
                                className='btn' if controls else 'btn hidden',
                                title="Download every panel's data as one CSV"),
                    dcc.Download(id='board-download'),
                    # Invisible sink for the clientside theme callback.
                    html.Div(id='board-theme-sink', className='hidden'),
                ],
                className='board-actions',
            ),
        ],
        className='board-header',
    )


def _control(html, label, component, wrap_id=None, hidden=False):
    """Label a control and stack it in a small flex column.

    ``wrap_id`` gives the wrapper Div an id (so a callback can target it);
    ``hidden`` starts it collapsed (used for the z / legend controls on plot
    types that don't take them)."""
    div_kwargs = {'className': 'control hidden' if hidden else 'control'}
    if wrap_id is not None:
        div_kwargs['id'] = wrap_id
    return html.Div(
        [html.Label(label, className='control-label'), component],
        **div_kwargs,
    )


def _panel_div(html, dcc, dash_table, i, panel, col_options, size,
               controls=True):
    """One panel card: an editable title row, a strip of the panel's own
    controls (plot type / variables), and its display. Dataset selection and
    theme deliberately have no controls here — they live in the header.

    The card carries both a graph and a (sortable) DataTable, each in its own
    wrapper; the render callback fills whichever the current plot type needs
    and hides the other, so switching to/from ``table`` at runtime just flips
    wrappers.

    ``controls=False`` (locked board) hides the control strip and the csv button
    and makes the title read-only display text. The components stay in the DOM so
    the render/export callbacks are unaffected; only their chrome is hidden."""
    width, height = size
    method = panel.get('method', 'plot')
    x = panel.get('x')
    y = _normalize_y(panel.get('y'))
    # z is a first-class control; accept it top-level, falling back to a z left
    # inside kwargs (older specs / convenience).
    z = panel.get('z') or (panel.get('kwargs') or {}).get('z')
    suptitle = panel.get('suptitle')
    legend = panel.get('legend', 'above')
    # A 'datasets' key pins this panel to fixed dataset indices: it opts out of
    # the header picker (needed e.g. for a contour map of one set with overlay
    # sets drawn on top). None = follow the board selection.
    pin = panel.get('datasets')
    extra = panel.get('kwargs') or {}

    head = html.Div(
        [
            dcc.Input(id={'type': 'panel-suptitle', 'index': i},
                      type='text', value=suptitle or '',
                      placeholder=f'Panel {i + 1}', debounce=True,
                      readOnly=not controls,
                      className='panel-title-input' if controls
                      else 'panel-title-input locked'),
            html.Span(f"pinned: sets {', '.join(str(d) for d in pin)}",
                      className='pin-badge',
                      title='This panel always plots these dataset indices; '
                            'the header picker does not affect it.')
            if pin is not None else None,
            html.Button('⬇ csv', id={'type': 'panel-export', 'index': i},
                        n_clicks=0,
                        className='btn' if controls else 'btn hidden',
                        title="Download this panel's plotted data as CSV"),
        ],
        className='panel-head',
    )

    controls_row = html.Div(
        [
            _control(html, 'plot type', dcc.Dropdown(
                id={'type': 'panel-method', 'index': i},
                options=PLOT_METHODS, value=method, clearable=False)),
            _control(html, 'x', dcc.Dropdown(
                id={'type': 'panel-x', 'index': i},
                options=col_options, value=x)),
            _control(html, 'y', dcc.Dropdown(
                id={'type': 'panel-y', 'index': i},
                options=col_options, value=y, multi=True)),
            # z: only relevant for contour, so hidden unless the plot type is
            # contour (toggled by the clientside callback in build_app).
            _control(html, 'z', dcc.Dropdown(
                id={'type': 'panel-z', 'index': i},
                options=col_options, value=z),
                wrap_id={'type': 'panel-z-wrap', 'index': i},
                hidden=(method not in _Z_METHODS)),
            # legend: only plot/plot_ymult accept it, so hidden elsewhere
            # (same clientside toggle pattern as z).
            _control(html, 'legend', dcc.Dropdown(
                id={'type': 'panel-legend', 'index': i},
                options=LEGEND_POSITIONS, value=legend, clearable=False),
                wrap_id={'type': 'panel-legend-wrap', 'index': i},
                hidden=(method not in _LEGEND_METHODS)),
        ],
        className='panel-controls' if controls else 'panel-controls hidden',
    )

    # Don't render at build time — the initial callback paints this panel once
    # (the spinner covers the wait). Saves a full render per panel on load.
    graph = dcc.Graph(id={'type': 'panel-graph', 'index': i},
                      figure=_placeholder(size),
                      config={'displaylogo': False},
                      style={'height': f'{height}px', 'width': f'{width}px'})

    # The table display for table panels: DataTable gives native click-to-sort
    # headers, which the go.Table canvas can't (it swallows clicks). Styled
    # with the theme tokens so it follows light/dark like the rest of the
    # chrome; data/columns are filled by the render callback.
    table = dash_table.DataTable(
        id={'type': 'panel-table', 'index': i},
        data=[], columns=[],
        sort_action='native',
        page_action='none',
        fixed_rows={'headers': True},
        style_table={'width': f'{width}px', 'maxHeight': f'{height}px',
                     'overflowY': 'auto',
                     'border': '1px solid var(--hairline)',
                     'borderRadius': '6px'},
        style_header={'backgroundColor': 'var(--surface)',
                      'color': 'var(--muted)', 'fontWeight': '600',
                      'fontSize': '10.5px', 'textTransform': 'uppercase',
                      'letterSpacing': '0.05em', 'border': 'none',
                      'borderBottom': '1px solid var(--hairline)'},
        style_cell={'fontFamily': _FIG_FONT, 'fontSize': '12.5px',
                    'padding': '4px 10px',
                    'backgroundColor': 'transparent', 'color': 'var(--ink)',
                    'border': 'none',
                    'borderBottom': '1px solid var(--hairline)'},
    )

    # Graph and table live side by side; the render callback hides whichever
    # the current method doesn't use. Seed the initial hidden state from the
    # seeded method so a table panel doesn't flash an empty graph placeholder.
    is_table = method in _TABLE_METHODS
    graph_box = html.Div(graph, id={'type': 'panel-graph-box', 'index': i},
                         className='hidden' if is_table else '')
    table_wrap = html.Div(table, id={'type': 'panel-table-wrap', 'index': i},
                          className='' if is_table else 'hidden')

    # Renders are serialized by a lock, so a slow panel (or one queued behind
    # another) would otherwise sit silently with a stale figure. dcc.Loading shows
    # a spinner while this panel's callback runs — including on initial load
    # (show_initially, default True), where it covers the empty placeholder until
    # the first render lands. overlay_style keeps the (dimmed) current figure
    # visible with the spinner on top, so a re-render reads as "this panel is
    # updating" rather than blanking out. delay_show is small so the spinner is
    # actually perceptible without flickering on sub-100ms renders.
    content = dcc.Loading(
        children=html.Div([graph_box, table_wrap]), type='circle',
        delay_show=120, color='#3987e5',
        overlay_style={'visibility': 'visible', 'opacity': 0.5},
        style={'height': f'{height}px', 'width': f'{width}px'},
    )

    # Holds this panel's passthrough kwargs / dataset pin for the callback to
    # read as State.
    store = dcc.Store(id={'type': 'panel-kwargs', 'index': i}, data=extra)
    pin_store = dcc.Store(id={'type': 'panel-pin', 'index': i},
                          data=list(pin) if pin is not None else None)
    # Sink for the CSV produced by the export button.
    download = dcc.Download(id={'type': 'panel-download', 'index': i})

    return html.Div(
        [head, controls_row, store, pin_store, download,
         html.Div(content, className='panel-graph-wrap')],
        className='panel-card',
    )


def dashboard(nb, panels, ncols=2, width=600, height=420, title=None,
              controls=True, jupyter_mode='inline', port=8050, debug=False,
              **run_kwargs):
    """Build and launch an interactive Dash board combining unichart figures.

    The board shows one consistent slice of the data: the header's dataset
    picker and light/dark theme switch drive every panel, and each panel adds
    only its own plot type / variable controls.

    Parameters
    ----------
    nb : UnichartNotebook
        The notebook whose datasets and plot methods drive the panels. Its
        current ``.select`` state seeds the header's dataset picker, and its
        ``darkmode`` seeds the theme switch.
    panels : list[dict]
        One dict per panel. Recognized keys: ``method`` (one of
        :data:`PLOT_METHODS`, default ``'plot'``), ``x``, ``y`` (str or list),
        ``z`` (the contour color column; its dropdown is shown only for contour
        panels), ``suptitle`` (the panel's card title — drawn in the card
        header, not inside the figure, and used to name CSV exports),
        ``legend`` (``above``/``right``/``off``; shown
        only for plot/plot_ymult), ``datasets`` (pin the panel to these dataset
        indices — it then ignores the header picker and shows a "pinned" badge),
        and ``kwargs`` (method-specific passthrough options, e.g. ``{'nbins':
        20}`` for histogram, ``{'barmode': 'stack', 'agg': 'sum'}`` for bar,
        ``{'overlay_sets': [1, 2]}`` for contour, ``{'x_in': [10, 15],
        'sig_figs': 3}`` for a table). Each kwarg is applied only when the
        active method accepts it, so it survives plot-type switches. For a
        ``table`` panel the ``y`` control selects the columns to show, ``x``
        is the interpolation x-axis (used only with ``x_in``), and the panel
        title is the table title. Table panels display as a DataTable with
        native sorting — click a column header to sort (numeric columns sort
        numerically, including under ``sig_figs`` formatting).
    ncols : int
        Number of columns in the panel grid.
    width, height : int
        Explicit px size of each panel's figure. An explicit size is what makes
        panels paint reliably in the inline Jupyter iframe.
    title : str
        Board title shown in the header (and as the page title).
    controls : bool
        ``True`` (default) shows the full interactive chrome. ``False`` renders
        a locked-down presentation board: the per-panel plot-type / x / y / z /
        legend dropdowns, the export buttons, the header's dataset picker and
        theme switch, and title editing are all hidden, leaving a clean grid of
        titled figure cards. The panels still render exactly as seeded (and honor
        their ``datasets`` pins); it is a display layer, not a security boundary.
    jupyter_mode : str
        Passed to ``Dash.run`` — ``'inline'`` (default) renders in the notebook
        cell; ``'external'`` / ``'tab'`` open a browser.
    port : int
        Preferred port. If it is already taken (e.g. a previous board is still
        running from an earlier cell run), a free port is chosen automatically so
        re-running the cell doesn't raise ``Address already in use``.
    debug, **run_kwargs
        Forwarded to ``Dash.run``.

    Returns
    -------
    dash.Dash
        The running app instance (useful for inspection / further wiring).
    """
    app = build_app(nb, panels, ncols=ncols, width=width, height=height,
                    title=title, controls=controls)
    # The inline iframe defaults to ~650px and would clip a multi-row board, so
    # size it to fit all rows plus the header (caller can override via
    # jupyter_height=...). A locked board drops the per-panel control strip, so
    # each row needs less vertical room.
    if jupyter_mode == 'inline' and 'jupyter_height' not in run_kwargs:
        nrows = -(-len(panels) // ncols)          # ceil division
        per_row = height + (150 if controls else 70)
        run_kwargs['jupyter_height'] = nrows * per_row + 110
    port = _pick_port(port)
    app.run(jupyter_mode=jupyter_mode, port=port, debug=debug, **run_kwargs)
    return app


def to_html(nb, panels, path, ncols=2, width=600, height=420, title=None,
            embed_js='cdn', global_select=True):
    """Write the board to a self-contained static HTML file.

    A live ``dashboard`` is a Dash server: its panels are painted by server-side
    callbacks, so there is nothing to "save" to a standalone page. This renders
    each panel once — exactly as the board would, honoring the same panel specs,
    the notebook's current dataset selection (or a panel's ``datasets`` pin) and
    its ``darkmode`` — and lays the figures out in the board's card grid, written
    to one HTML file at ``path``.

    The result is the locked / presentation view frozen to disk, with one live
    board control carried over: a global dataset filter in the header (see
    ``global_select``). The Plotly charts stay fully interactive (hover, zoom,
    pan, modebar), but the editing chrome (dropdowns, theme switch) is gone.
    ``table`` panels are written as real HTML tables with click-to-sort
    column headers (a ``go.Table`` canvas can't take header clicks), scrolling
    inside their card past ``height``. Seed the slice first
    (``nb.toggle_darkmode(True)``, dataset ``.select`` flags) if you want a
    specific one — it becomes the file's initial state.

    Parameters mirror :func:`dashboard` (``ncols``, ``width``, ``height``,
    ``title``), plus:

    embed_js : {'cdn', True, 'directory'}
        How plotly.js is included (passed to ``plotly.io.to_html`` for the first
        figure; the rest reuse it). ``'cdn'`` (default) keeps the file small
        (~200 KB) but needs internet to open. ``True`` embeds the full library
        (~5 MB) for a truly offline, self-contained file — use it for email or an
        air-gapped machine. ``'directory'`` references a ``plotly.min.js`` sitting
        next to the HTML.
    global_select : bool
        ``True`` (default) recreates the live header's dataset picker as chips
        that work offline: panels are rendered with *every* dataset embedded,
        and a small script toggles trace visibility across all panels at once
        (traces map back to datasets via the legendgroups unichart stamps, see
        ``_DATASET_TAG_RE``). The chips seed from the notebook's current
        ``.select`` flags, so the file opens showing exactly the export-time
        selection. Panels the filter cannot reach keep that selection frozen
        and carry a badge: pinned panels (``datasets=[...]``) stay pinned, and
        ``table`` / ``contour`` / ``by='sets'`` panels are marked "static"
        (their figures aren't built one-trace-per-dataset, so visibility
        toggling can't re-slice them). Unchecking every chip leaves empty
        axes — unlike the live board there is no server to draw the "no
        datasets" hint. ``False`` writes the fully frozen snapshot.

    Rendering leaves the notebook untouched (``render_panel`` snapshots and
    restores ``.select`` / ``darkmode`` / ``last_*``). Returns ``path``.
    """
    if not nb.sets:
        raise ValueError("The notebook has no datasets loaded.")
    if not panels:
        raise ValueError("Provide at least one panel.")

    theme = 'dark' if getattr(nb, 'darkmode', False) else 'light'
    board_title = title or 'unichart dashboard'
    default_selected = _selected_indices(nb)
    all_indices = [ds.index for ds in nb.sets]

    static_badge = ('<span class="pin-badge" title="This panel type cannot be '
                    're-filtered client-side; it always shows the datasets '
                    'selected at export.">static</span>')

    cards = []
    any_filtered = False
    any_tables = False
    plotly_included = False
    for i, panel in enumerate(panels):
        pin = panel.get('datasets')
        method = panel.get('method', 'plot')
        z = panel.get('z') or (panel.get('kwargs') or {}).get('z')

        # suptitle=None mirrors the live board: the figure's auto-title is
        # blanked and the panel title is drawn in the card header instead, so
        # it isn't said twice.
        def _render(sel):
            return render_panel(
                nb, method, panel.get('x'), panel.get('y'), sel,
                suptitle=None, legend=panel.get('legend', 'above'),
                size=(width, height), extra=panel.get('kwargs'), z=z,
                darkmode=nb.darkmode,
            )

        badge = ''
        filtered = False
        if pin is not None:
            # Pinned panels ignore the dataset filter, exactly like the live
            # board's picker; the badge says so (only needed when there is a
            # filter to be exempt from).
            fig = _render(pin)
            if global_select:
                badge = ('<span class="pin-badge" title="This panel always '
                         'plots these dataset indices; the dataset filter '
                         'does not affect it.">pinned: sets '
                         f"{', '.join(str(d) for d in pin)}</span>")
        elif (global_select and method not in _TABLE_METHODS
                and method not in _Z_METHODS):
            # Embed every dataset so the filter can bring any of them in; the
            # on-load script hides the initially-unselected ones. Only keep
            # this render if its traces actually carry dataset tags — a
            # by='sets' panel (subplot per dataset, untagged traces) can't be
            # filtered, so it falls back to the export-time selection.
            fig = _render(all_indices)
            filtered = any(_DATASET_TAG_RE.match(tr.legendgroup or '')
                           for tr in fig.data)
            if not filtered:
                if set(default_selected) != set(all_indices):
                    fig = _render(default_selected)
                badge = static_badge
        else:
            # table (one combined go.Table) and contour (z-grid computed from
            # the data, only overlays are tagged) can't be re-sliced by trace
            # visibility; freeze them on the export-time selection.
            fig = _render(default_selected)
            if global_select:
                badge = static_badge
        any_filtered |= filtered

        if (method in _TABLE_METHODS and fig.data
                and fig.data[0].type == 'table'):
            # Table panels become real, sortable HTML tables (an error render
            # falls through: it is a plain figure, not a table trace).
            body = _table_html(fig.data[0], width, height)
            any_tables = True
        else:
            # Bundle plotly.js once (first plotly-rendered figure), then
            # reference it from the rest.
            body = pio.to_html(
                fig, include_plotlyjs=(False if plotly_included else embed_js),
                full_html=False, config={'displaylogo': False})
            plotly_included = True
        name = _escape(panel.get('suptitle') or f'Panel {i + 1}')
        gsel_attr = ' data-gsel="1"' if filtered else ''
        cards.append(
            f'<div class="panel-card"{gsel_attr}>'
            '<div class="panel-head">'
            f'<span class="panel-title-input locked">{name}</span>{badge}</div>'
            f'<div class="panel-graph-wrap">{body}</div></div>'
        )

    grid = (f'<div class="board-grid" style="grid-template-columns: '
            f'repeat({ncols}, max-content)">' + ''.join(cards) + '</div>')

    # The header shows the dataset filter only when some panel responds to it;
    # chips reuse the live board's chip-list CSS and seed from the current
    # selection (the same state that just rendered the static fallbacks).
    header_bits = [f'<div class="board-title">{_escape(board_title)}</div>']
    show_filter = global_select and any_filtered
    if show_filter:
        sel = set(default_selected)
        chips = ''.join(
            f'<label><input type="checkbox" value="{ds.index}"'
            f'{" checked" if ds.index in sel else ""}>'
            f'<span class="chip-dot" style="background:'
            f'{_escape(str(ds.color))}"></span>'
            f'<span>{_escape(str(ds.title_format))}</span></label>'
            for ds in nb.sets)
        header_bits.append(
            '<div class="board-group"><span class="control-label">datasets'
            f'</span><div class="chip-list gsel">{chips}</div></div>')
    header = '<div class="board-header">' + ''.join(header_bits) + '</div>'
    script = f'<script>{_GLOBAL_SELECT_JS}</script>' if show_filter else ''
    if any_tables:
        script += f'<script>{_SORTABLE_TABLE_JS}</script>'

    # Same chrome CSS and data-theme hook as the live board (_APP_CSS), so the
    # static file looks identical and honors light/dark.
    document = (
        f'<!DOCTYPE html>\n<html data-theme="{theme}">\n<head>\n'
        f'    <meta charset="utf-8">\n    <title>{_escape(board_title)}</title>\n'
        f'    <style>\n{_APP_CSS}\n    </style>\n</head>\n'
        f'<body>\n    <div class="board">{header}{grid}</div>{script}\n'
        '</body>\n</html>\n'
    )
    with open(path, 'w', encoding='utf-8') as f:
        f.write(document)
    return path
