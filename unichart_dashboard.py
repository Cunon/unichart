"""On-the-fly Dash dashboards that combine multiple unichart figures.

`unichart.py` plotting methods each return a Plotly ``go.Figure`` and cache the
most recent one in ``nb.last_fig``. This module wires those figures into an
interactive Dash board: a grid of panels, each with its own controls (plot type,
x / y variables, dataset on/off, suptitle, legend position) that re-plot live.

Typical use, inline in a Jupyter notebook::

    from unichart_dashboard import dashboard

    dashboard(nb, panels=[
        {'method': 'plot', 'x': 'time', 'y': 'temp'},
        {'method': 'bar',  'x': 'cat',  'y': 'val', 'datasets': [0, 1]},
    ], ncols=2)

Dash is an optional dependency; it is imported lazily so importing the core
toolkit never requires it.
"""

import base64
import inspect
import re
import socket
import threading

import numpy as np
import pandas as pd
import plotly.graph_objects as go


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

# Methods whose signature accepts a `legend=` argument (above/right/off).
_LEGEND_METHODS = {'plot', 'plot_ymult'}

# Methods that take a `z=` column (mapped to color). The z dropdown is only
# shown — and only passed through — for these.
_Z_METHODS = {'contour'}

# Methods with a different signature from the x/y plotters: handled by a
# dedicated branch in render_panel rather than the generic x/y/legend dispatch.
_TABLE_METHODS = {'table'}

LEGEND_POSITIONS = ['above', 'right', 'off']


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
        from dash import Dash, dcc, html, Input, Output, State, MATCH, ALL
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "unichart_dashboard requires Dash. Install it with "
            "`pip install dash` (or add it to requirements.txt)."
        ) from exc
    return Dash, dcc, html, Input, Output, State, MATCH, ALL


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
                 size=None, extra=None, z=None):
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

    The notebook is left exactly as it was found. Rendering a panel would
    otherwise mutate shared state: ``.select`` (which datasets are active) and the
    ``last_*`` plot memory (``last_x``/``last_y``/``last_fig``/...). All of it is
    snapshotted and restored in ``finally``, so interacting with the board never
    changes what a later ``nb.plot()`` cell does. ``nb.last_fig`` is additionally
    cleared before dispatch so the notebook's own ``_clear_last_fig`` cannot gut
    the user's previously cached figure in place.

    ``size`` is an optional ``(width_px, height_px)``. When given, it is stamped
    onto the figure as an explicit, non-autosize dimension. This matters for the
    inline-Jupyter board: a freshly-created Dash iframe has zero size at first
    paint, so an autosize/responsive figure draws a 0x0 (blank) canvas until an
    interaction triggers a resize. An explicit size paints correctly on load.

    This is the pure core of the Dash callback and is callable directly in tests.
    """
    if method not in PLOT_METHODS:
        return _size(_error_figure(f"Unknown plot method: {method!r}"), size)

    chosen = set(dataset_indices or [])

    with _RENDER_LOCK:
        # Snapshot every piece of notebook state a render touches, so the user's
        # notebook is untouched after the board runs.
        select_snapshot = [(ds, ds.select) for ds in nb.sets]
        state_snapshot = {k: getattr(nb, k) for k in vars(nb)
                          if k.startswith('last_')}
        try:
            for ds in nb.sets:
                ds.select = ds.index in chosen
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
                    return _size(_error_figure(
                        "No table data — pick column(s) under y."), size)
                return _size(fig, size)

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
                return _size(_error_figure("No figure produced (no data / selection?)"), size)
            # No defensive copy: the finally below restores nb.last_fig to the
            # snapshot, so this figure is detached from the notebook the moment we
            # return. _clear_last_fig only guts whatever nb.last_fig points at, so
            # it can never reach this one. (Skipping go.Figure(fig) avoids deep-
            # copying every trace, which is the per-render cost on large data.)
            return _size(fig, size)
        except Exception as exc:  # noqa: BLE001 - surface any plotting error in-panel
            return _size(_error_figure(f"{type(exc).__name__}: {exc}"), size)
        finally:
            for ds, was in select_snapshot:
                ds.select = was
            # Drop any last_* attribute the render created (e.g. contour's
            # last_z on a notebook that had never plotted a contour), then
            # restore the snapshotted values.
            for k in [k for k in vars(nb)
                      if k.startswith('last_') and k not in state_snapshot]:
                delattr(nb, k)
            for k, v in state_snapshot.items():
                setattr(nb, k, v)


def _size(fig, size):
    """Stamp an explicit (width, height) onto a figure so it paints at a known
    size regardless of its container. unichart bakes figsize into width/height
    via _base_layout; we overwrite it with the panel size. With ``size=None`` the
    figure keeps whatever dimensions it already has."""
    if size is not None:
        fig.update_layout(autosize=False, width=size[0], height=size[1])
    return fig


def _error_figure(message):
    """A blank figure that displays an error message in the panel."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       xref='paper', yref='paper', x=0.5, y=0.5,
                       font=dict(color='crimson'))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=20, r=20, t=20, b=20))
    return fig


def _placeholder(size):
    """An empty, correctly-sized figure for a panel's initial state.

    Panels are not rendered at build time; the initial callback paints each one
    (and the dcc.Loading spinner covers the gap). This avoids the load-time double
    render — once at build, once via the callback — which is costly on large data.
    The placeholder is sized so the panel doesn't collapse before its figure lands."""
    fig = go.Figure()
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    return _size(fig, size)


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


def _normalize_y(y):
    """Y control is multi-select; normalize seeds to a list of strings."""
    if y is None:
        return []
    if isinstance(y, (list, tuple)):
        return [str(v) for v in y]
    return [str(y)]


def build_app(nb, panels, ncols=2, width=600, height=420):
    """Build (but do not run) the Dash app for the given panels.

    Returns the configured ``dash.Dash`` instance with its layout and the single
    pattern-matching callback registered. Factored out of :func:`dashboard` so
    the app/layout can be constructed and inspected without starting a server.

    ``width`` / ``height`` are the explicit px size of each panel's figure. They
    are stamped onto every figure so panels paint reliably inline (see
    :func:`render_panel`) instead of collapsing to 0x0 in a fresh Dash iframe.
    """
    Dash, dcc, html, Input, Output, State, MATCH, ALL = _require_dash()

    if not nb.sets:
        raise ValueError("The notebook has no datasets loaded.")
    if not panels:
        raise ValueError("Provide at least one panel.")

    col_options = _all_columns(nb)
    dataset_options = [{'label': ds.title_format, 'value': ds.index}
                       for ds in nb.sets]
    default_selected = _selected_indices(nb)
    size = (width, height)

    # Board-level toolbar: export every panel's data into one combined CSV.
    toolbar = html.Div(
        [
            html.Button('⬇ export all panels', id='board-export', n_clicks=0,
                        title="Download every panel's data as one CSV",
                        style={'fontSize': '12px', 'cursor': 'pointer'}),
            dcc.Download(id='board-download'),
        ],
        style={'padding': '8px 8px 0 8px'},
    )
    grid = html.Div(
        [_panel_div(html, dcc, i, panel, col_options, dataset_options,
                    default_selected, size)
         for i, panel in enumerate(panels)],
        style={'display': 'grid',
               'gridTemplateColumns': f'repeat({ncols}, max-content)',
               'gap': '16px', 'padding': '8px'},
    )

    app = Dash(__name__)
    app.layout = html.Div([toolbar, grid])

    @app.callback(
        Output({'type': 'panel-graph', 'index': MATCH}, 'figure'),
        Input({'type': 'panel-method', 'index': MATCH}, 'value'),
        Input({'type': 'panel-x', 'index': MATCH}, 'value'),
        Input({'type': 'panel-y', 'index': MATCH}, 'value'),
        Input({'type': 'panel-z', 'index': MATCH}, 'value'),
        Input({'type': 'panel-datasets', 'index': MATCH}, 'value'),
        Input({'type': 'panel-suptitle', 'index': MATCH}, 'value'),
        Input({'type': 'panel-legend', 'index': MATCH}, 'value'),
        # Per-panel passthrough kwargs are fixed at build time (no control), so
        # they ride along as State rather than triggering renders themselves.
        State({'type': 'panel-kwargs', 'index': MATCH}, 'data'),
        # Initial-call-enabled on purpose: on load the callback paints each panel
        # once (over the placeholder), via the same path used on interaction.
    )
    def _update_panel(method, x, y, z, datasets, suptitle, legend, extra):
        return render_panel(nb, method, x, y, datasets, suptitle, legend,
                            size=size, extra=extra, z=z)

    # Show the z dropdown only when the panel's plot type is contour. Clientside
    # so toggling is instant and never queues behind a render on the lock.
    app.clientside_callback(
        "function(method){ var s = {display:'flex', flexDirection:'column', "
        "minWidth:'120px'}; if (method !== 'contour'){ s.display = 'none'; } "
        "return s; }",
        Output({'type': 'panel-z-wrap', 'index': MATCH}, 'style'),
        Input({'type': 'panel-method', 'index': MATCH}, 'value'),
    )

    # Export the panel's plotted data as CSV. Reads the *currently displayed*
    # figure via State (no re-render, no lock) and flattens its traces.
    @app.callback(
        Output({'type': 'panel-download', 'index': MATCH}, 'data'),
        Input({'type': 'panel-export', 'index': MATCH}, 'n_clicks'),
        State({'type': 'panel-graph', 'index': MATCH}, 'figure'),
        State({'type': 'panel-suptitle', 'index': MATCH}, 'value'),
        prevent_initial_call=True,
    )
    def _export_panel(n_clicks, figure, suptitle):
        if not n_clicks:
            return None
        df = _figure_to_dataframe(figure)
        stem = re.sub(r'[^\w.-]+', '_', (suptitle or 'panel').strip()) or 'panel'
        return dcc.send_data_frame(df.to_csv, f'{stem}.csv', index=False)

    # Board-level export: stack every panel's flattened data into one CSV, with a
    # `panel` column naming each (its suptitle, else its position).
    @app.callback(
        Output('board-download', 'data'),
        Input('board-export', 'n_clicks'),
        State({'type': 'panel-graph', 'index': ALL}, 'figure'),
        State({'type': 'panel-suptitle', 'index': ALL}, 'value'),
        prevent_initial_call=True,
    )
    def _export_board(n_clicks, figures, suptitles):
        if not n_clicks:
            return None
        frames = []
        for i, (figure, suptitle) in enumerate(zip(figures, suptitles)):
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


# Base style for a labelled control column (also the "shown" state of the
# toggleable z control; the clientside callback flips between this and hidden).
_CONTROL_STYLE = {'display': 'flex', 'flexDirection': 'column', 'minWidth': '120px'}


def _control(html, label, component, wrap_id=None, hidden=False):
    """Label a control and stack it in a small flex column.

    ``wrap_id`` gives the wrapper Div an id (so a callback can target it);
    ``hidden`` starts it collapsed (used for the z control on non-contour panels).
    """
    style = {**_CONTROL_STYLE, 'display': 'none'} if hidden else dict(_CONTROL_STYLE)
    div_kwargs = {'style': style}
    if wrap_id is not None:
        div_kwargs['id'] = wrap_id
    return html.Div(
        [html.Label(label, style={'fontSize': '11px', 'color': '#555'}), component],
        **div_kwargs,
    )


def _panel_div(html, dcc, i, panel, col_options, dataset_options,
               default_selected, size):
    """One panel: a row of controls above its graph."""
    width, height = size
    method = panel.get('method', 'plot')
    x = panel.get('x')
    y = _normalize_y(panel.get('y'))
    # z is a first-class control; accept it top-level, falling back to a z left
    # inside kwargs (older specs / convenience).
    z = panel.get('z') or (panel.get('kwargs') or {}).get('z')
    suptitle = panel.get('suptitle')
    legend = panel.get('legend', 'above')
    selected = panel.get('datasets', default_selected)
    extra = panel.get('kwargs') or {}

    controls = html.Div(
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
            _control(html, 'datasets', dcc.Checklist(
                id={'type': 'panel-datasets', 'index': i},
                options=dataset_options, value=list(selected),
                style={'fontSize': '12px'})),
            _control(html, 'suptitle', dcc.Input(
                id={'type': 'panel-suptitle', 'index': i},
                type='text', value=suptitle or '',
                debounce=True, style={'width': '140px'})),
            _control(html, 'legend', dcc.Dropdown(
                id={'type': 'panel-legend', 'index': i},
                options=LEGEND_POSITIONS, value=legend, clearable=False)),
            html.Button(
                '⬇ data', id={'type': 'panel-export', 'index': i}, n_clicks=0,
                title="Download this panel's plotted data as CSV",
                style={'fontSize': '11px', 'alignSelf': 'flex-end',
                       'height': '30px', 'cursor': 'pointer'}),
        ],
        style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px',
               'alignItems': 'flex-start', 'marginBottom': '6px'},
    )

    # Don't render at build time — the initial callback paints this panel once
    # (the spinner covers the wait). Saves a full render per panel on load.
    graph = dcc.Graph(id={'type': 'panel-graph', 'index': i},
                      figure=_placeholder(size),
                      style={'height': f'{height}px', 'width': f'{width}px'})

    # Renders are serialized by a lock, so a slow panel (or one queued behind
    # another) would otherwise sit silently with a stale figure. dcc.Loading shows
    # a spinner while this panel's callback runs — including on initial load
    # (show_initially, default True), where it covers the empty placeholder until
    # the first render lands. overlay_style keeps the (dimmed) current figure
    # visible with the spinner on top, so a re-render reads as "this panel is
    # updating" rather than blanking out. delay_show is small so the spinner is
    # actually perceptible without flickering on sub-100ms renders.
    graph = dcc.Loading(
        children=graph, type='circle', delay_show=120,
        overlay_style={'visibility': 'visible', 'opacity': 0.5},
        style={'height': f'{height}px', 'width': f'{width}px'},
    )

    # Holds this panel's passthrough kwargs for the callback to read as State.
    store = dcc.Store(id={'type': 'panel-kwargs', 'index': i}, data=extra)
    # Sink for the CSV produced by the export button.
    download = dcc.Download(id={'type': 'panel-download', 'index': i})

    return html.Div([controls, store, download, graph],
                    style={'border': '1px solid #ddd', 'borderRadius': '6px',
                           'padding': '8px'})


def dashboard(nb, panels, ncols=2, width=600, height=420, jupyter_mode='inline',
              port=8050, debug=False, **run_kwargs):
    """Build and launch an interactive Dash board combining unichart figures.

    Parameters
    ----------
    nb : UnichartNotebook
        The notebook whose datasets and plot methods drive the panels.
    panels : list[dict]
        One dict per panel. Recognized keys: ``method`` (one of
        :data:`PLOT_METHODS`, default ``'plot'``), ``x``, ``y`` (str or list),
        ``z`` (the contour color column; its dropdown is shown only for contour
        panels), ``suptitle``, ``legend`` (``above``/``right``/``off``),
        ``datasets`` (list of dataset indices to select initially; defaults to the
        notebook's current selection), and ``kwargs`` (method-specific passthrough
        options, e.g. ``{'nbins': 20}`` for histogram, ``{'barmode': 'stack',
        'agg': 'sum'}`` for bar, ``{'overlay_sets': [1, 2]}`` for contour,
        ``{'x_in': [10, 15], 'sig_figs': 3}`` for a table). Each kwarg is applied
        only when the active method accepts it, so it survives plot-type switches.
        For a ``table`` panel the ``y`` control selects the columns to show, ``x``
        is the interpolation x-axis (used only with ``x_in``), and ``suptitle`` is
        the table title.
    ncols : int
        Number of columns in the panel grid.
    width, height : int
        Explicit px size of each panel's figure. An explicit size is what makes
        panels paint reliably in the inline Jupyter iframe.
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
    app = build_app(nb, panels, ncols=ncols, width=width, height=height)
    # The inline iframe defaults to ~650px and would clip a multi-row board, so
    # size it to fit all rows (caller can override via jupyter_height=...).
    if jupyter_mode == 'inline' and 'jupyter_height' not in run_kwargs:
        nrows = -(-len(panels) // ncols)          # ceil division
        run_kwargs['jupyter_height'] = nrows * (height + 150) + 40
    port = _pick_port(port)
    app.run(jupyter_mode=jupyter_mode, port=port, debug=debug, **run_kwargs)
    return app
