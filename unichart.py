import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import warnings
import numbers
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import re
import inspect
from scipy.interpolate import griddata
import functools
import gc
from pathlib import Path

# -----------------------------------------------------------------------------
# Constants & Mappers (Translation Layer)
# -----------------------------------------------------------------------------

MARKER_MAP_MPL_TO_PLOTLY = {
    'o': 'circle', 's': 'square', 'D': 'diamond', 'd': 'diamond-tall',
    'v': 'triangle-down', '^': 'triangle-up', '<': 'triangle-left', '>': 'triangle-right',
    'p': 'pentagon', '*': 'star', 'h': 'hexagon', 'H': 'hexagon2',
    'x': 'x', 'X': 'x-thin', '+': 'cross', '|': 'line-ns', '_': 'line-ew', '.': 'circle-dot'
}

LINESTYLE_MAP_MPL_TO_PLOTLY = {
    '-': 'solid', '--': 'dash', '-.': 'dashdot', ':': 'dot',
    'None': None, ' ': None, '': None
}

FONT_SIZE_MAP = {
    'xs':     8,
    'xsmall': 8,
    'sm':     10,
    'small':  10,
    'md':     12,
    'medium': 12,
    'base':   12,
    'lg':     14,
    'large':  14,
    'xl':     18,
    'xlarge': 18,
    'xxl':    24,
    '2xl':    24,
    'xxxl':   28,
    '3xl':    28,
    'huge':   32,
}

def get_plotly_marker(mpl_marker):
    # Pass through if already a Plotly-native name
    if mpl_marker in MARKER_MAP_MPL_TO_PLOTLY.values():
        return mpl_marker
    return MARKER_MAP_MPL_TO_PLOTLY.get(mpl_marker, 'circle')

def get_plotly_linestyle(mpl_style):
    # Pass through if already a Plotly-native name
    if mpl_style in LINESTYLE_MAP_MPL_TO_PLOTLY.values():
        return mpl_style
    return LINESTYLE_MAP_MPL_TO_PLOTLY.get(mpl_style, 'solid')

def validate_color(value):
    """Return True if value is a string (Plotly accepts named colors, hex, and rgb strings)."""
    if not isinstance(value, str):
        return False
    return True

def validate_marker(value):
    return value in MARKER_MAP_MPL_TO_PLOTLY or value is None

def validate_linestyle(value):
    return value in LINESTYLE_MAP_MPL_TO_PLOTLY or value is None

def marker_map(index):
    markers = list(MARKER_MAP_MPL_TO_PLOTLY.keys())
    return markers[index % len(markers)]


class CyclicList(list):
    """A ``list`` whose *integer* indexing wraps around (cycles) modulo its
    length, so a short map still answers any index:

        >>> CyclicList(['a', 'b'])[3]
        'b'

    Slicing and all other list behavior are unchanged. An empty CyclicList
    raises ``IndexError`` on integer access, like a normal empty list.
    """

    def __getitem__(self, index):
        if isinstance(index, int) and len(self):
            return super().__getitem__(index % len(self))
        return super().__getitem__(index)

def _generate_contour_grid(x_data, y_data, z_data, res=100, method='linear'):
    """
    Interpolates scattered x, y, z data into a uniform 2D grid for contour plotting.
    Leaves data outside the convex hull as NaN.
    """
    valid = ~(np.isnan(x_data) | np.isnan(y_data) | np.isnan(z_data))
    x_val = x_data[valid]
    y_val = y_data[valid]
    z_val = z_data[valid]

    if len(x_val) < 4:
        return x_val.values, y_val.values, z_val.values

    xi = np.linspace(x_val.min(), x_val.max(), res)
    yi = np.linspace(y_val.min(), y_val.max(), res)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    zi_grid = griddata((x_val, y_val), z_val, (xi_grid, yi_grid), method=method)

    return xi, yi, zi_grid

# -----------------------------------------------------------------------------
# Private Helpers
# -----------------------------------------------------------------------------

def _calc_grid(n, nrows, ncols):
    if nrows is None and ncols is None:
        ncols = min(3, max(1, int(np.ceil(np.sqrt(n)))))
        nrows = int(np.ceil(n / ncols))
    elif nrows is None:
        nrows = int(np.ceil(n / ncols))
    elif ncols is None:
        ncols = int(np.ceil(n / nrows))
    return nrows, ncols

# Top-of-figure spacing estimates (px) used to reserve room for a (possibly
# multi-line) suptitle above a horizontal "above" legend so the legend can't
# grow up into the title. Per-line title height scales with the title font size:
# _base_layout assumes the default font, and _apply_fonts re-reserves the space
# once a custom suptitle size (set via set_font_sizes) is known.
_TITLE_TOP_PAD = 12             # gap above the first title line
_DEFAULT_TITLE_FONT_PX = 18     # approximates Plotly's default suptitle font
_TITLE_LINE_FACTOR = 1.45       # title line height = font size * this
_LEGEND_ROW_PX = 26             # space reserved for the first legend row
_LEGEND_GAP    = 12             # gap title→legend and legend→plot


def _title_lines(text):
    """Number of rendered lines in a Plotly title string. Counts <br> and
    treats a literal newline as a line break too (it is normalized to <br>
    elsewhere before rendering)."""
    if not text:
        return 1
    return text.replace('\n', '<br>').count('<br>') + 1


def _top_space(title_text, figsize, has_above_legend, title_font_size=None):
    """Geometry that reserves vertical space for a (possibly multi-line)
    suptitle so a horizontal "above" legend can't cover it.

    ``title_font_size`` is the suptitle font size in px; when ``None`` the
    Plotly default is assumed. Per-line height scales with it so a large custom
    title font still gets enough room.

    Returns ``(top_margin_px, title_pos, legend_pos)``. ``title_pos`` is merged
    into ``layout.title`` (pins it to the top of the figure container);
    ``legend_pos`` is merged into ``layout.legend`` when ``has_above_legend``
    (pins the legend's *top* just below the title, in container coords, so extra
    legend rows grow downward toward the plot rather than up into the title).
    ``legend_pos`` is ``None`` when there is no above-legend.
    """
    height = (figsize[1] * 100) if figsize else 800
    font = title_font_size or _DEFAULT_TITLE_FONT_PX
    n_lines = _title_lines(title_text)
    line_px = font * _TITLE_LINE_FACTOR
    # Headroom above the first line grows with the font so large titles aren't
    # clipped at the figure's top edge.
    top_pad = _TITLE_TOP_PAD + max(0.0, font - _DEFAULT_TITLE_FONT_PX) * 0.9
    title_band = top_pad + n_lines * line_px + _LEGEND_GAP
    if has_above_legend:
        top_margin = title_band + _LEGEND_ROW_PX + _LEGEND_GAP
    else:
        top_margin = title_band + _TITLE_TOP_PAD

    title_pos = {'y': 1.0 - top_pad / height, 'yanchor': 'top', 'yref': 'container'}
    legend_pos = None
    if has_above_legend:
        legend_pos = {'orientation': 'h', 'yanchor': 'top',
                      'y': 1.0 - title_band / height, 'yref': 'container'}
    return top_margin, title_pos, legend_pos


def _above_legend_layout(suptitle, figsize):
    """Ready ``(legend, top_margin)`` for a standalone horizontal above-legend,
    matching the geometry ``_base_layout`` applies. Used where the legend is set
    via a direct ``update_layout`` rather than through ``_base_layout``."""
    top_margin, _, legend_pos = _top_space(suptitle, figsize, True)
    legend = {'xanchor': 'left', 'x': 0, **legend_pos}
    return legend, top_margin


# Footer (bottom text box) spacing. Mirrors the suptitle math but reserves
# space at the *bottom*; the default font is Plotly's annotation default.
_FOOTER_PAD = 10              # gap below the last footer line / above the band
_DEFAULT_FOOTER_FONT_PX = 12  # Plotly's default annotation font size


def _bottom_space(footer_text, footer_font_size=None):
    """Bottom margin (px) to reserve for a (possibly multi-line) footer, *in
    addition* to the axis-label margin, so the footer sits below the labels
    without overlapping them. Per-line height scales with the footer font."""
    font = footer_font_size or _DEFAULT_FOOTER_FONT_PX
    n_lines = _title_lines(footer_text)
    return _FOOTER_PAD + n_lines * font * _TITLE_LINE_FACTOR + _FOOTER_PAD


def _base_layout(darkmode, suptitle, figsize, **extra):
    incoming_title = extra.pop('title', {})
    if isinstance(incoming_title, str):
        incoming_title = {'text': incoming_title}
    if 'text' not in incoming_title:
        incoming_title['text'] = suptitle
    # Normalize "\n" to Plotly's "<br>" so newline-style titles both render and
    # get counted as multiple lines for space reservation.
    if incoming_title.get('text'):
        incoming_title['text'] = incoming_title['text'].replace('\n', '<br>')

    incoming_legend = extra.pop('legend', None)
    has_above = (isinstance(incoming_legend, dict)
                 and incoming_legend.get('orientation') == 'h')

    top_margin, title_pos, legend_pos = _top_space(
        incoming_title.get('text'), figsize, has_above)

    title_defaults = {'x': 0.5, 'xanchor': 'center'}
    merged_title = {**title_defaults, **incoming_title, **title_pos}

    default_margin = {'t': top_margin}
    incoming_margin = extra.pop('margin', {})
    merged_margin = {**default_margin, **incoming_margin}

    args = {
        'template': "plotly_dark" if darkmode else "plotly_white",
        'title': merged_title,
        'margin': merged_margin,
        **extra
    }
    if legend_pos is not None:
        # caller's x/xanchor/font win; vertical geometry (legend_pos) is forced.
        legend_defaults = {'xanchor': 'left', 'x': 0}
        args['legend'] = {**legend_defaults, **incoming_legend, **legend_pos}
    elif incoming_legend is not None:
        args['legend'] = incoming_legend
    if figsize:
        args['width'] = figsize[0] * 100
        args['height'] = figsize[1] * 100
    return args

def _build_xbins(bin_size, bin_start, bin_end):
    d = {}
    if bin_size is not None: d['size'] = bin_size
    if bin_start is not None: d['start'] = bin_start
    if bin_end is not None: d['end'] = bin_end
    return d or None

def _show_or_return(fig, return_axes):
    if return_axes:
        return fig
    fig.show()
    return fig

def _subplot_refs(row, col, ncols):
    """Return the (xref, yref) axis name strings for a subplot at (row, col) in an ncols grid."""
    idx = (row - 1) * ncols + col
    xref = 'x' if idx == 1 else f'x{idx}'
    yref = 'y' if idx == 1 else f'y{idx}'
    return xref, yref

# -----------------------------------------------------------------------------
# Variable Format Resolver (used by multi-y plot)
# -----------------------------------------------------------------------------
_VAR_FORMAT_KEYS = ('color', 'marker', 'linestyle', 'markersize', 'linewidth', 'alpha', 'style')

# Rendering styles for bar-plot overlay columns (the ``markers=`` argument of
# ``unibar``/``unibar_per_dataset``). 'marker' is the classic symbol overlay;
# 'tick' draws a horizontal dash at the value (bullet-chart target); 'whisker'
# adds a stem connecting that dash to the top of its bar.
_OVERLAY_STYLES = ('marker', 'tick', 'whisker')

# Sentinel for ``default_format['marker']`` meaning "assign per-index from
# marker_map" (the historical behavior). A concrete marker string instead pins
# every future dataset to that symbol, and ``None`` turns markers off — distinct
# from "use the index map", which is why this needs its own sentinel object.
_MARKER_BY_INDEX = object()

# Sentinel for set_default_format's ``marker`` parameter meaning "not supplied,
# leave unchanged". Kept separate from _MARKER_BY_INDEX so that an explicit
# ``marker=None`` (markers off) is distinguishable from "caller passed nothing".
_UNSET = object()

# Built-in per-dataset style defaults applied to newly loaded sets. Each
# UnichartNotebook copies these into ``self.default_format``; ``set_default_format``
# overrides them so that *future* loaded datasets (and ``reset_format``) pick up
# the new styling — the markersize/linewidth analogue of color_map/marker_map.
# ``marker`` defaults to _MARKER_BY_INDEX (per-index from marker_map); setting it
# to a symbol or to None overrides that for future sets. Color stays index-only.
_DATASET_FORMAT_DEFAULTS = {
    'marker':     _MARKER_BY_INDEX,
    'markersize': 10,
    'linestyle':  None,
    'linewidth':  2,
    'edgewidth':  1,
    'edge_color': 'black',
    'alpha':      1,
    'fill':       True,
}

def _resolve_var_format(dataset, variable, variable_formats=None):
    """
    Per-attribute precedence: variable_formats wins, else dataset attr.

    Used by the multi-y-axis plot. Returns a flat dict containing the
    final color/marker/linestyle/markersize/linewidth/alpha that should
    be applied for a given (dataset, variable) pair.
    """
    variable_formats = variable_formats or {}
    var_fmt = variable_formats.get(variable, {})
    return {
        'color':      var_fmt.get('color',      dataset.color),
        'marker':     var_fmt.get('marker',     dataset.marker),
        'linestyle':  var_fmt.get('linestyle',  dataset.linestyle),
        'markersize': var_fmt.get('markersize', dataset.markersize),
        'linewidth':  var_fmt.get('linewidth',  dataset.linewidth),
        'alpha':      var_fmt.get('alpha',      dataset.alpha),
        'edge_color': getattr(dataset, 'edge_color', 'black'),
        'edgewidth':  getattr(dataset, 'edgewidth', 1),
        'fill':       getattr(dataset, 'fill', True),
    }


def _overlay_marker_kw(style, symbol, color, size, alpha, edge_color,
                       values=None, bar_values=None,
                       stem_dash='solid', stem_width=2):
    """Marker (and, for 'whisker', error-bar) kwargs for a bar-overlay Scatter
    trace, per the _OVERLAY_STYLES contract. 'tick' and 'whisker' render a
    horizontal 'line-ew' dash at the value — line symbols only draw their
    outline, so the column color goes on ``marker.line``. 'whisker'
    additionally draws a stem spanning the gap between ``values`` (the overlay
    column) and ``bar_values`` (the bar column it sits on).

    The error-bar stem only exists for ``stem_dash='solid'`` (Plotly error
    bars cannot dash); dashed stems are drawn by a companion line trace built
    with ``_whisker_stem_kw``, and ``stem_dash=None`` means no stem at all."""
    if style not in ('tick', 'whisker'):
        return dict(marker=dict(
            symbol=symbol, size=size, color=color, opacity=alpha,
            line=dict(width=1.5, color=edge_color),
        ))
    kw = dict(marker=dict(
        symbol='line-ew', size=max(size, 16), opacity=alpha,
        color=color, line=dict(width=3, color=color),
    ))
    if style == 'whisker' and bar_values is not None and stem_dash == 'solid':
        delta = values - bar_values           # + when value sits above its bar
        kw['error_y'] = dict(
            type='data', symmetric=False,
            array=(-delta).clip(lower=0),     # stem up to a bar top above the value
            arrayminus=delta.clip(lower=0),   # stem down to a bar top below the value
            color=color, thickness=stem_width, width=0,
        )
    return kw


def _whisker_stem_kw(x, values, bar_values, color, alpha, dash, width):
    """Kwargs for a whisker stem drawn as explicit None-separated line
    segments (one vertical segment per bar, value -> bar top). Used instead of
    the error-bar stem when a dashed linestyle is requested."""
    xs, ys = [], []
    for xi, v, b in zip(x, values, bar_values):
        if pd.isna(v) or pd.isna(b):
            continue
        xs += [xi, xi, None]
        ys += [v, b, None]
    return dict(x=xs, y=ys, mode='lines',
                line=dict(color=color, dash=dash, width=width),
                opacity=alpha, hoverinfo='skip', showlegend=False)


def _fill_marker_kw(color, fill, edgewidth=1):
    """Marker color/outline keys honoring a fill toggle: a solid ``color`` fill
    when ``fill`` is True, or a hollow marker (transparent fill, ``color``
    outline) when False. Mirrors the hollow-marker handling in ``uniplot``."""
    if fill:
        return {'color': color}
    return {'color': 'rgba(0,0,0,0)', 'line': dict(width=edgewidth, color=color)}

# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------

_SET_ID_COL = '_SET_ID'

# Above this many points a scatter trace renders with WebGL (go.Scattergl)
# instead of SVG — SVG creates one DOM node per point and locks the browser
# on dense transient data. Set to None to always use SVG.
WEBGL_POINT_THRESHOLD = 10_000


def _scatter_cls(n_points):
    """Trace class for a scatter of ``n_points``: Scattergl past the WebGL
    threshold, plain Scatter below it (or when the threshold is disabled)."""
    if WEBGL_POINT_THRESHOLD is None:
        return go.Scatter
    return go.Scattergl if n_points > WEBGL_POINT_THRESHOLD else go.Scatter


def _data_col_indexer(cdf):
    """Integer positions of every combined-frame column except _SET_ID."""
    return np.flatnonzero(cdf.columns.to_numpy() != _SET_ID_COL)


class _DatasetFrameView:
    """Mutable per-set view onto a UnichartNotebook's combined DataFrame.

    Reads return a per-set slice (with _SET_ID hidden). Writes scatter values
    back into the combined frame, NaN-filling other sets when a new column is
    introduced. Exists so that legacy `ds._df_full[col] = values` patterns
    continue to work after the storage refactor.

    Unlike ``Dataset.df``, reads here are ownership-agnostic: the slice spans
    every combined-frame column, including other sets' all-NaN phantoms.
    """

    def __init__(self, dataset):
        self._dataset = dataset

    def _slice(self):
        nb = self._dataset._notebook
        cdf = nb._combined_df
        pos = nb._set_positions(self._dataset._set_id)
        return cdf.iloc[pos, _data_col_indexer(cdf)]

    def __getitem__(self, key):
        return self._slice()[key]

    @staticmethod
    def _is_object_like(value):
        """True if value carries string/object (non-numeric) data."""
        if value is None:
            return False
        if isinstance(value, str):
            return True
        if isinstance(value, pd.Series):
            return not pd.api.types.is_numeric_dtype(value)
        if isinstance(value, np.ndarray):
            return value.dtype == object or value.dtype.kind in ('U', 'S')
        # scalar: numbers (incl. bool) are numeric-friendly, everything else isn't
        return not isinstance(value, numbers.Number)

    def __setitem__(self, key, value):
        notebook = self._dataset._notebook
        cdf = notebook._combined_df
        mask = (cdf[_SET_ID_COL] == self._dataset._set_id)
        target_idx = cdf.index[mask]

        # Normalize the value and validate length for array-likes.
        if isinstance(value, pd.Series):
            # Align by label; labels absent from this set are dropped,
            # missing target labels become NaN.
            assign_val = value.reindex(target_idx)
        elif hasattr(value, '__len__') and not isinstance(value, str):
            n = len(target_idx)
            if len(value) != n:
                raise ValueError(
                    f"Length mismatch assigning '{key}' to set "
                    f"{self._dataset.index}: {len(value)} values for {n} rows.")
            assign_val = np.asarray(value)
        else:
            assign_val = value

        # Choose/repair the column dtype. A masked .loc assignment cannot
        # upcast in place on modern pandas, so a string into a float64 column
        # raises. Create new columns — and widen existing numeric ones — to
        # object when the incoming value is non-numeric.
        incoming_object = self._is_object_like(assign_val)
        if key not in cdf.columns:
            cdf[key] = pd.Series(
                None if incoming_object else np.nan,
                index=cdf.index,
                dtype=object if incoming_object else float,
            )
        elif incoming_object and cdf[key].dtype != object:
            cdf[key] = cdf[key].astype(object)

        cdf.loc[mask, key] = assign_val
        # Writing through the view claims the column for this set — including
        # the case of filling NaNs into a column another set introduced.
        self._dataset._own_cols.add(key)
        notebook._reapply_all_queries()

    @property
    def columns(self):
        return self._slice().columns

    @property
    def index(self):
        return self._slice().index

    @property
    def empty(self):
        return self._slice().empty

    def __len__(self):
        return len(self._slice())

    def __repr__(self):
        return repr(self._slice())

    def __getattr__(self, name):
        return getattr(self._slice(), name)


class Dataset:
    """Façade over a slice of a UnichartNotebook's combined DataFrame.

    Holds only formatting/state attributes plus a reference to the parent
    notebook and an internal set id. The actual row data lives in
    `notebook._combined_df`, keyed by the `_SET_ID` column.
    """

    def __init__(self, notebook, set_id, index=0, title=None, display_parms=None,
                 own_cols=None):
        if not hasattr(notebook, '_combined_df'):
            raise TypeError(
                "Dataset must be constructed by a UnichartNotebook; "
                "use UnichartNotebook.load_df(df, ...) instead.")
        self._notebook = notebook
        self._set_id = set_id
        self._query = None
        self._query_mask = None
        self._select = True

        # Columns this set actually owns. Other sets' columns exist in the
        # combined frame as all-NaN phantoms for this set; ownership keeps
        # them out of `df`/`columns`. None (legacy construction) means
        # "everything currently in the combined frame".
        if own_cols is None:
            own_cols = set(notebook._combined_df.columns)
        self._own_cols = {c for c in own_cols if c != _SET_ID_COL}

        if title:
            self.title = title
        else:
            cdf = notebook._combined_df
            pos = notebook._set_positions(set_id)
            if "TITLE" in cdf.columns and len(pos):
                self.title = str(cdf["TITLE"].iloc[pos[0]])
            else:
                self.title = "Untitled"

        self.index = index
        self.title_format = f"{self.title} {index}"

        self._color = notebook._color_at(index)

        # Per-dataset style defaults come from the notebook so set_default_format
        # controls how future loaded datasets look. Falls back to the built-ins.
        fmt = getattr(notebook, 'default_format', _DATASET_FORMAT_DEFAULTS)
        # Marker is per-index by default; a default marker (symbol or None=off)
        # set via set_default_format overrides the marker_map assignment.
        default_marker = fmt.get('marker', _MARKER_BY_INDEX)
        self._marker = (notebook._marker_at(index)
                        if default_marker is _MARKER_BY_INDEX else default_marker)
        self._edge_color = fmt.get('edge_color', 'black')
        self._fill = fmt.get('fill', True)
        self._linestyle = fmt.get('linestyle', None)
        self.markersize = fmt.get('markersize', 10)
        self.alpha = fmt.get('alpha', 1)
        self.hue = ""
        self.hue_palette = "Jet"
        self.hue_order = None
        self.reg_order = None
        self.style = None
        self.linewidth = fmt.get('linewidth', 2)
        self.edgewidth = fmt.get('edgewidth', 1)
        self.set_type = 1
        self.data_type = 'discrete'
        self.delta_sets = None
        self.file_path = None
        self._display_parms = coerce_display_parms(display_parms)
        self._plot_type = 'scatter'
        self._order = None

    def _raw_df(self):
        """All rows for this set, unmasked, with _SET_ID stripped."""
        cdf = self._notebook._combined_df
        pos = self._notebook._set_positions(self._set_id)
        return cdf.iloc[pos, _data_col_indexer(cdf)]

    def _masked_positions(self):
        """Integer row positions of this set in the combined frame, with the
        query mask applied. Positions come from the notebook's per-set cache,
        so no full-column scan or full-width row copy is needed."""
        pos = self._notebook._set_positions(self._set_id)
        if self._query_mask is not None:
            cdf = self._notebook._combined_df
            qm = self._query_mask.reindex(cdf.index, fill_value=False).to_numpy()
            pos = pos[qm[pos]]
        return pos

    def _own_col_positions(self):
        """Integer positions of this set's own columns within the combined
        frame (reconciling ownership with any untracked column changes first)."""
        self._notebook._reconcile_columns()
        own = self._own_cols
        cols = self._notebook._combined_df.columns
        return [i for i, c in enumerate(cols) if c != _SET_ID_COL and c in own]

    @property
    def columns(self):
        """Column labels this set actually owns, in combined-frame order —
        without materializing any rows. All-NaN phantom columns introduced by
        *other* sets sharing the combined frame are excluded. Ownership is
        kept current by the write APIs and by automatic reconciliation of
        columns added/removed directly on ``nb.df``; after in-place value
        surgery on the live frame, call ``nb.refresh_own_columns(rescan=True)``.
        """
        cols = self._notebook._combined_df.columns
        return pd.Index([cols[i] for i in self._own_col_positions()])

    def cols(self, keys, masked=True):
        """Rows of just the requested column(s) — far cheaper than ``self.df[keys]``
        on wide frames, since only the named columns are copied. Missing keys are
        silently skipped; duplicated column labels keep their first occurrence
        (matching the plotters' dedup behavior). Returns a fresh copy.

        Explicitly requested keys are served from the combined frame whether or
        not this set owns them (a phantom column comes back all-NaN).
        """
        cdf = self._notebook._combined_df
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        col_pos = []
        for k in dict.fromkeys(keys):
            locs = cdf.columns.get_indexer_for([k])
            locs = locs[locs >= 0]
            if len(locs):
                col_pos.append(locs[0])
        pos = (self._masked_positions() if masked
               else self._notebook._set_positions(self._set_id))
        return cdf.iloc[pos, col_pos]

    def __getitem__(self, key):
        """Read a column (or columns) for this set, e.g. ``ds['FN']``.

        Copies only the requested column(s), not the full set width.
        """
        if isinstance(key, list):
            return self.cols(key)
        return self._notebook._combined_df[key].iloc[self._masked_positions()]

    def __setitem__(self, key, value):
        """Write a column back into the combined frame for this set.

        This is the supported way to add or modify columns, e.g.
        ``ds['NEW'] = ds['A'] * 2``. Writing through ``ds.df[...]`` does NOT
        persist, because ``df`` returns a fresh slice (a copy) each call.
        """
        self._df_full[key] = value

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        if value is None or value in self.columns:
            self._order = value
        else:
            raise ValueError(f"Invalid order column: {value}")

    @property
    def df(self):
        """Read-only view of this set's rows (query mask applied), restricted
        to the columns this set owns — all-NaN phantom columns introduced by
        other sets are excluded (see :attr:`columns`).

        Returns a fresh copy each call, so assigning to it does NOT persist:
        use ``ds['col'] = ...`` (or ``nb.set_column``) to write back.
        """
        cdf = self._notebook._combined_df
        return cdf.iloc[self._masked_positions(), self._own_col_positions()]

    @df.setter
    def df(self, value):
        # _replace_set_rows reapplies every set's query mask (the global index
        # is rebuilt on replacement, which would otherwise stale other masks).
        self._notebook._replace_set_rows(self._set_id, value)

    @property
    def _df_full(self):
        return _DatasetFrameView(self)

    @_df_full.setter
    def _df_full(self, value):
        self._notebook._replace_set_rows(self._set_id, value)

    @property
    def query(self):
        return self._query

    @query.setter
    def query(self, value):
        self._query = value
        self._apply_query()

    def _apply_query(self):
        cdf = self._notebook._combined_df
        if not self._query:
            self._query_mask = None
            return
        set_rows = cdf.iloc[self._notebook._set_positions(self._set_id)]
        try:
            filtered = set_rows.query(self._query)
        except Exception as e:
            raise ValueError(f"Query error: {e}")
        if filtered.empty:
            print(f"No data in set {self.index} after query: {self._query}. Turning Set Off...")
            self._select = False
            self._query_mask = None
            return
        mask = pd.Series(False, index=cdf.index)
        mask.loc[filtered.index] = True
        self._query_mask = mask

    @property
    def color(self): return self._color

    @color.setter
    def color(self, value): self._color = value
        
    @property
    def select(self): return self._select

    @select.setter
    def select(self, value):
        if str(value).lower() in ['true', '1', 't', 'on']:
            self._select = True
        elif str(value).lower() in ['false', '0', 'f', 'off']:
            self._select = False
        else:
            raise ValueError(f"Invalid value for select: {value}")

    @property
    def edge_color(self): return self._edge_color

    @edge_color.setter
    def edge_color(self, value): self._edge_color = value

    @property
    def fill(self): return self._fill

    @fill.setter
    def fill(self, value):
        if str(value).lower() in ['true', '1', 't', 'on']:
            self._fill = True
        elif str(value).lower() in ['false', '0', 'f', 'off']:
            self._fill = False
        else:
            raise ValueError(f"Invalid value for fill: {value}")

    @property
    def plot_type(self): return self._plot_type

    @plot_type.setter
    def plot_type(self, value):
        valid_plot_types = ['scatter', 'contour', 'histogram']
        if value in valid_plot_types:
            self._plot_type = value
        else:
            raise ValueError(f"Invalid plot_type: {value}")
        
    @property
    def marker(self): return self._marker

    @marker.setter
    def marker(self, value): self._marker = value

    @property
    def linestyle(self): return self._linestyle

    @linestyle.setter
    def linestyle(self, value): self._linestyle = value

    @property
    def linewidth(self): return self._linewidth

    @linewidth.setter
    def linewidth(self, value):
        if isinstance(value, (int, float)) and value >= 0:
            self._linewidth = value
        else:
            raise ValueError(f"Invalid linewidth: {value}")

    @property
    def edgewidth(self): return self._edgewidth

    @edgewidth.setter
    def edgewidth(self, value):
        if isinstance(value, (int, float)) and value >= 0:
            self._edgewidth = value
        else:
            raise ValueError(f"Invalid edgewidth: {value}")

    @property
    def display_parms(self): return self._display_parms

    @display_parms.setter
    def display_parms(self, value):
        self._display_parms = coerce_display_parms(value)

    def sel_query(self, query):
        self.query = query

    def update_format_dict(self, format_options):
        for key, value in format_options.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid format key: {key}")

    def get_format_dict(self):
        return {
            'title': self.title,
            'color': self.color,
            'marker': self.marker,
            'edge_color': self.edge_color,
            'fill': self.fill,
            'linestyle': self.linestyle,
            'markersize': self.markersize,
            'alpha': self.alpha,
            'hue': self.hue,
            'hue_palette': self.hue_palette,
            'hue_order': self.hue_order,
            'reg_order': self.reg_order,
            'index': self.index,
            'style': self.style,
            'display_parms': self.display_parms,
            'plot_type': self.plot_type,
            'linewidth': self.linewidth,
            'edgewidth': self.edgewidth,
        }
    
    def set_format_option(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise ValueError(f"Invalid format key: {key}")

    def get_title(self):
        return self.title

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def coerce_display_parms(value):
    """Normalise a ``display_parms`` value into a clean list of column names.

    Accepts ``None`` (-> ``[]``), a single string (-> ``[name]``), or any
    iterable of names (list, tuple, pandas.Index, numpy array, ...). Entries
    are stringified, stripped of blanks, and de-duplicated while preserving
    order. A non-iterable, non-string value raises ``ValueError`` so genuine
    mistakes still surface.
    """
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, dict):
        raise ValueError("display_parms must be a column name or list of names, not a dict")
    elif hasattr(value, '__iter__'):
        items = list(value)
    else:
        raise ValueError(f"display_parms must be a column name or list of names, got {type(value).__name__}")
    out = []
    for item in items:
        name = str(item).strip()
        if name and name not in out:
            out.append(name)
    return out


def build_hover_data(df, parms):
    """Build robust hover ``customdata`` + template lines for ``display_parms``.

    Returns ``(customdata, lines)`` where ``customdata`` is an ndarray suitable
    for a trace's ``customdata`` (or ``None`` when there is nothing to show) and
    ``lines`` is a list of ``"<br>name: %{customdata[i]...}"`` template
    fragments to append to a hovertemplate.

    Each parm is formatted by dtype so that "certain data" renders cleanly:

    * datetimes are pre-rendered to readable strings (never raw epoch ints or
      ``...T00:00:00`` ISO blobs, and ``NaT`` shows blank);
    * booleans show ``True`` / ``False`` rather than ``1`` / ``0``;
    * integers print in full, without scientific notation;
    * floats use general precision (``.6g``);
    * anything else (categories, strings, objects) is shown as-is;
    * missing values render blank instead of ``NaN`` / ``None``.

    Parms missing from ``df`` are skipped. Building each column independently
    also avoids the mixed-dtype ``to_numpy()`` collapse that previously turned
    timestamps into opaque objects.
    """
    cols = []
    lines = []
    for parm in parms:
        if parm not in df.columns:
            continue
        s = df[parm]
        i = len(cols)
        if pd.api.types.is_datetime64_any_dtype(s):
            # any non-midnight time component -> include H:M:S. The notna mask
            # is required because NaT != NaT is True in pandas.
            has_time = bool((s.notna() & (s != s.dt.normalize())).any())
            fmt = '%Y-%m-%d %H:%M:%S' if has_time else '%Y-%m-%d'
            col = s.dt.strftime(fmt).where(s.notna(), '')
            cols.append(col.to_numpy())
            lines.append(f"<br>{parm}: %{{customdata[{i}]}}")
        elif pd.api.types.is_bool_dtype(s):
            col = np.where(s.to_numpy(), 'True', 'False')
            cols.append(col)
            lines.append(f"<br>{parm}: %{{customdata[{i}]}}")
        elif pd.api.types.is_integer_dtype(s):
            cols.append(s.to_numpy())
            # plain %{customdata} prints integers in full (no .5g sci notation)
            lines.append(f"<br>{parm}: %{{customdata[{i}]}}")
        elif pd.api.types.is_numeric_dtype(s):
            cols.append(s.to_numpy())
            lines.append(f"<br>{parm}: %{{customdata[{i}]:.6g}}")
        else:
            col = s.astype(object).where(s.notna(), '')
            cols.append(col.to_numpy())
            lines.append(f"<br>{parm}: %{{customdata[{i}]}}")
    if not cols:
        return None, []
    customdata = np.column_stack(cols) if len(cols) > 1 else cols[0].reshape(-1, 1)
    return customdata, lines


def table_read(df, x_col, y_col, x_in, kind='linear', fill_value='extrapolate', bounds_error=False):
    """Interpolate values from a DataFrame column using 1-D interpolation.

    This function sorts the DataFrame by the x column, constructs a 1-D
    interpolator for the (x_col, y_col) pairs using scipy.interpolate.interp1d,
    and evaluates the interpolator at the provided x_in points.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the source data. Must contain columns named
        x_col and y_col.
    x_col : str
        Name of the column to use as the independent variable (x-axis).
    y_col : str
        Name of the column to use as the dependent variable (y-axis).
    x_in : array-like or scalar
        Points at which to evaluate the interpolator. Can be a scalar,
        list, or numpy array.
    kind : str or int, optional
        Specifies the kind of interpolation to use. Passed directly to
        scipy.interpolate.interp1d (e.g. 'linear', 'nearest', 'zero',
        'slinear', 'quadratic', 'cubic', or an integer for spline order).
        Default is 'linear'.
    fill_value : float or (float, float) or {'extrapolate'}, optional
        Value to use for points outside the interpolation range. Matches
        interp1d's fill_value behaviour. Default is 'extrapolate'.
    bounds_error : bool, optional
        If True, raise a ValueError when attempting to interpolate outside
        the range of x values. If False, use fill_value. Default is False.

    Returns
    -------
    numpy.ndarray or scalar
        Interpolated y values corresponding to x_in. The type mirrors the
        output of scipy.interpolate.interp1d evaluated at x_in (scalar or
        array).

    Notes
    -----
    - The DataFrame is sorted by x_col before building the interpolator to
      ensure monotonic x values as expected by interp1d.
    - For best results, ensure x_col contains unique values; duplicate x
      values can lead to undefined behavior from interp1d.

    """
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Columns '{x_col}' and '{y_col}' must be present in the DataFrame.")

    df_sorted = df.sort_values(by=x_col)
    x_values = df_sorted[x_col].values
    y_values = df_sorted[y_col].values

    f = interp1d(
        x_values,
        y_values,
        kind=kind,
        fill_value=fill_value,
        bounds_error=bounds_error,
    )

    y_interp = f(x_in)
    return y_interp

def _parse_reg_spec(spec):
    """Normalize reg_order spec into (kind, param)."""
    if not spec:
        return None, None
    if isinstance(spec, bool):
        return None, None
    if isinstance(spec, numbers.Number):
        return ('poly', int(spec)) if spec > 0 else (None, None)

    aliases = {
        'linear': ('poly', 1), 'lin': ('poly', 1),
        'quadratic': ('poly', 2), 'cubic': ('poly', 3),
        'poly': ('poly', None),
        'log': ('log', None), 'logarithmic': ('log', None),
        'exp': ('exp', None), 'exponential': ('exp', None),
        'power': ('power', None), 'pow': ('power', None),
        'lowess': ('lowess', 0.3), 'loess': ('lowess', 0.3),
        'spline': ('spline', 3), 'cubic_spline': ('spline', 3),
        'ma': ('ma', None), 'moving_average': ('ma', None), 'rolling': ('ma', None),
    }

    if isinstance(spec, str):
        s = spec.lower().strip()
        if s in aliases: return aliases[s]
        if s.startswith('poly'):
            try: return 'poly', int(s[4:])
            except ValueError: pass
        raise ValueError(f"Unknown regression type: {spec!r}")

    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        kind, param = spec
        kind = str(kind).lower().strip()
        if kind in aliases:
            return aliases[kind][0], param
        raise ValueError(f"Unknown regression kind: {kind!r}")

    raise ValueError(f"Invalid reg_order spec: {spec!r}")


def _calculate_regression(df, x_col, y_col, spec):
    """
    Compute a regression curve. Returns (x_array, y_array, label).
    Returns (None, None, None) when spec is falsy or fit cannot be computed.
    """
    kind, param = _parse_reg_spec(spec)
    if kind is None:
        return None, None, None

    df_clean = df.dropna(subset=[x_col, y_col]).sort_values(by=x_col)
    x = df_clean[x_col].to_numpy(dtype=float)
    y = df_clean[y_col].to_numpy(dtype=float)
    if len(x) < 2:
        return None, None, None

    x_lin = np.linspace(x.min(), x.max(), 200)

    try:
        if kind == 'poly':
            order = int(param) if param else 1
            if len(x) < order + 1:
                return None, None, None
            p = np.poly1d(np.polyfit(x, y, order))
            label = 'Linear' if order == 1 else f'LS{order}'
            return x_lin, p(x_lin), label

        if kind == 'log':
            mask = x > 0
            if mask.sum() < 2: return None, None, None
            a, b = np.polyfit(np.log(x[mask]), y[mask], 1)
            y_lin = np.full_like(x_lin, np.nan)
            pos = x_lin > 0
            y_lin[pos] = a * np.log(x_lin[pos]) + b
            return x_lin, y_lin, 'Log'

        if kind == 'exp':
            mask = y > 0
            if mask.sum() < 2: return None, None, None
            b, log_a = np.polyfit(x[mask], np.log(y[mask]), 1)
            return x_lin, np.exp(log_a) * np.exp(b * x_lin), 'Exp'

        if kind == 'power':
            mask = (x > 0) & (y > 0)
            if mask.sum() < 2: return None, None, None
            b, log_a = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
            y_lin = np.full_like(x_lin, np.nan)
            pos = x_lin > 0
            y_lin[pos] = np.exp(log_a) * np.power(x_lin[pos], b)
            return x_lin, y_lin, 'Power'

        if kind == 'lowess':
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
            except ImportError:
                warnings.warn("LOWESS requires statsmodels. Install with `pip install statsmodels`.")
                return None, None, None
            frac = float(param) if param is not None else 0.3
            res = lowess(y, x, frac=frac, return_sorted=True)
            return res[:, 0], res[:, 1], f'LOWESS({frac:.2f})'

        if kind == 'spline':
            from scipy.interpolate import UnivariateSpline
            k = max(1, min(5, int(param) if param else 3))
            ux, idx = np.unique(x, return_index=True)
            uy = y[idx]
            if len(ux) < k + 1:
                return None, None, None
            spl = UnivariateSpline(ux, uy, k=k)
            return x_lin, spl(x_lin), f'Spline{k}'

        if kind == 'ma':
            window = int(param) if param else max(3, len(x) // 20)
            window = max(2, min(window, len(x)))
            ma = pd.Series(y).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
            return x, ma, f'MA({window})'

    except Exception as e:
        warnings.warn(f"Regression '{kind}' failed: {e}")
        return None, None, None

    return None, None, None

# -----------------------------------------------------------------------------
# Main Plotting Functions
# -----------------------------------------------------------------------------
def uniplot(list_of_datasets, x, y, z=None, plot_type=None, color=None, hue=None, marker=None,
            markersize=10, marker_edge_color="black", linestyle=None, hue_palette="Jet",
            hue_order=None, line=False, suppress_msg=False, return_axes=False, axes=None,
            suptitle=None, xlabel=None, ylabel=None, subplot_titles=None,
            darkmode=False, interactive=True, display_parms=None, grid=True,
            legend='above', legend_ncols=1, figsize=(12, 8), ncols=None, nrows=None, x_lim=None, y_lim=None,
            axis_limits=None):

    axis_limits = axis_limits or {}
    x_list = x if isinstance(x, list) else [x]
    y_list = y if isinstance(y, list) else [y]

    if len(x_list) == len(y_list):
        pairs = list(zip(x_list, y_list))
    elif len(x_list) == 1:
        pairs = [(x_list[0], yi) for yi in y_list]
    elif len(y_list) == 1:
        pairs = [(xi, y_list[0]) for xi in x_list]
    else:
        raise ValueError(
            f"x and y must be the same length, or one must be a single value. "
            f"Got len(x)={len(x_list)}, len(y)={len(y_list)}."
        )

    n_plots = len(pairs)
    if n_plots == 0: raise ValueError("At least one x/y pair is required.")

    nrows, ncols = _calc_grid(n_plots, nrows, ncols)

    numeric_hue_info = {}
    for _ds in list_of_datasets:
        if not _ds.select: continue
        _fmt = _ds.get_format_dict()
        _cur_hue = _fmt.get('hue') or hue
        if not _cur_hue or _cur_hue in numeric_hue_info: continue
        if _cur_hue in _ds.columns and pd.api.types.is_numeric_dtype(_ds[_cur_hue]):
            _idx = len(numeric_hue_info) + 1
            numeric_hue_info[_cur_hue] = {
                'ca_name': 'coloraxis' if _idx == 1 else f'coloraxis{_idx}',
                'palette': _fmt.get('hue_palette', 'Jet'),
                'lim': axis_limits.get(_cur_hue),
            }

    right_margin = max(80, len(numeric_hue_info) * 90)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles, shared_xaxes=False)
    fig.update_layout(**_base_layout(
        darkmode, None, figsize,
        title={'text': suptitle or (f"{x} vs {[str(yi) for yi in y_list]}" if len(x_list) == 1 else f"{x_list} vs {y_list}"), 'x': 0.5, 'xanchor': 'center'},
        showlegend=(legend != 'off'),
        margin=dict(r=right_margin),
        **({'legend': dict(orientation="h")} if legend == 'above' else {}),
    ))

    for dataset in list_of_datasets:
        if not dataset.select: continue

        fmt = dataset.get_format_dict()
        cur_title = fmt.get('title')
        cur_hue = fmt.get('hue') or hue
        cur_color = color or fmt.get('color')
        cur_marker = marker or fmt.get('marker')
        cur_linestyle = linestyle or fmt.get('linestyle')
        cur_markersize = fmt.get('markersize', markersize)
        cur_linewidth = fmt.get('linewidth', 2)
        cur_alpha = fmt.get('alpha', 1)
        cur_reg_order = fmt.get('reg_order')
        cur_idx = fmt.get('index')
        hover_parms = display_parms or fmt.get('display_parms', [])

        base_cols = dataset.columns
        cols_upper = None  # built lazily on first case-insensitive miss
        valid_hover = [p for p in hover_parms if p in base_cols]
        hue_in_cols = bool(cur_hue) and cur_hue in base_cols
        ds_order = dataset.order
        order_in_cols = bool(ds_order) and ds_order != 'index' and ds_order in base_cols

        # Resolve every pair's columns up front so the set can be fetched —
        # and sorted — once as a narrow frame. Sorting the full set width to
        # plot a handful of columns dominated plot time on wide frames.
        resolved_pairs = []
        for idx_p, (x_name, yi) in enumerate(pairs):
            if yi not in base_cols: continue
            if x_name in base_cols:
                x_col = x_name
            else:
                if cols_upper is None:
                    cols_upper = {c.upper(): c for c in base_cols}
                x_col = cols_upper.get(str(x_name).upper())
                if not x_col: continue
            resolved_pairs.append((idx_p, x_col, yi))
        if not resolved_pairs: continue

        needed = []
        for _, x_col, yi in resolved_pairs:
            needed.extend((x_col, yi))
        if hue_in_cols: needed.append(cur_hue)
        needed.extend(valid_hover)
        if order_in_cols: needed.append(ds_order)

        sorted_base_df = dataset.cols(list(dict.fromkeys(needed)))
        if order_in_cols:
            sorted_base_df = sorted_base_df.sort_values(by=ds_order)
        else:
            sorted_base_df = sorted_base_df.sort_index()

        for idx_p, x_col, yi in resolved_pairs:
            row = idx_p // ncols + 1
            col = idx_p % ncols + 1

            req_cols = [x_col, yi]
            if hue_in_cols: req_cols.append(cur_hue)
            req_cols.extend(valid_hover)

            df = sorted_base_df[list(dict.fromkeys(req_cols))]

            # x/y are already serialized as the trace's own arrays, so only the
            # hover parms ride in customdata (as a plain ndarray, not a frame) —
            # shipping x/y there as well doubled the figure payload.
            custom_data, hover_lines = build_hover_data(df, valid_hover)

            ht = f"<b><u>Set: {cur_idx}</u></b><br><b>{cur_title}</b><br>{x_col}: %{{x:.2f}}<br>{yi}: %{{y:.2f}}"
            ht += "".join(hover_lines)
            ht += "<extra></extra>"

            # A line shows when a linestyle is set (regression draws its own
            # trace, so the raw series stays point-only). Markers show unless
            # explicitly turned off with marker=None; if neither is on, the
            # trace draws nothing ('none').
            show_line = bool(cur_linestyle) and not cur_reg_order
            show_marker = cur_marker is not None
            mode_parts = []
            if show_line: mode_parts.append('lines')
            if show_marker: mode_parts.append('markers')
            mode = "+".join(mode_parts) if mode_parts else 'none'

            marker_dict = dict(
                size=cur_markersize, symbol=get_plotly_marker(cur_marker),
                line=dict(width=fmt.get('edgewidth', 1), color=fmt.get('edge_color', 'black')),
                opacity=cur_alpha
            )
            line_dict = dict(width=cur_linewidth, dash=get_plotly_linestyle(cur_linestyle))

            # Hue, when active, drives the per-point marker color and takes
            # precedence over the set color / fill toggle below.
            if cur_hue and cur_hue in df.columns:
                hue_data = df[cur_hue]
                if pd.api.types.is_numeric_dtype(hue_data):
                    info = numeric_hue_info.get(cur_hue, {})
                    marker_dict['color'] = hue_data
                    marker_dict['coloraxis'] = info.get('ca_name', 'coloraxis')
                else:
                    hue_series = hue_data.astype('category')
                    marker_dict['color'] = hue_series.cat.codes
                    marker_dict['colorscale'] = fmt.get('hue_palette', 'Jet')
                    marker_dict['showscale'] = False
            elif fmt.get('fill', True):
                marker_dict['color'] = cur_color
            else:
                # No fill: hollow marker whose outline takes the set color.
                marker_dict['color'] = 'rgba(0,0,0,0)'
                marker_dict['line'] = dict(width=fmt.get('edgewidth', 1), color=cur_color)
            line_dict['color'] = cur_color

            fig.add_trace(_scatter_cls(len(df))(
                x=df[x_col], y=df[yi], mode=mode,
                name=f"{cur_idx}: {cur_title}",
                legendgroup=f"group_{cur_idx}",
                marker=marker_dict, line=line_dict,
                customdata=custom_data, hovertemplate=ht,
                showlegend=(idx_p == 0)
            ), row=row, col=col)

            if cur_reg_order:
                rx, ry, fit_label = _calculate_regression(df, x_col, yi, cur_reg_order)
                if rx is not None:
                    fig.add_trace(go.Scatter(
                        x=rx, y=ry, mode='lines',
                        name=f"{cur_idx}: {cur_title} Fit ({fit_label})",
                        legendgroup=f"group_{cur_idx}",
                        line=dict(color=cur_color, width=cur_linewidth, dash=get_plotly_linestyle(cur_linestyle)),
                        opacity=0.7, hoverinfo='skip', showlegend=False
                    ), row=row, col=col)

    coloraxis_updates = {}
    for hue_col, info in numeric_hue_info.items():
        idx = list(numeric_hue_info.keys()).index(hue_col)
        ca_def = dict(
            colorscale=info['palette'],
            colorbar=dict(title=hue_col, x=1.02 + idx * 0.12, thickness=15),
        )
        if info['lim']:
            ca_def['cmin'] = info['lim'][0]
            ca_def['cmax'] = info['lim'][1]
        coloraxis_updates[info['ca_name']] = ca_def
    if coloraxis_updates:
        fig.update_layout(**coloraxis_updates)

    for idx_p, (x_name, yi) in enumerate(pairs):
        row = idx_p // ncols + 1
        col = idx_p % ncols + 1

        axis_title = ylabel if ylabel else yi
        x_axis_title = xlabel if xlabel else x_name

        fig.update_yaxes(title_text=axis_title, title_standoff=15, row=row, col=col)
        fig.update_xaxes(title_text=x_axis_title, title_standoff=15, row=row, col=col)

    for idx_p, (x_name, yi) in enumerate(pairs):
        row = idx_p // ncols + 1
        col = idx_p % ncols + 1
        if y_lim: fig.update_yaxes(range=y_lim, row=row, col=col)
        if x_lim: fig.update_xaxes(range=x_lim, row=row, col=col)
    if not grid:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

    return _show_or_return(fig, return_axes)

def uniplot_per_dataset(list_of_datasets, x, y, display_parms=None,
                        suptitle=None, figsize=(12, 8), ncols=None, nrows=None,
                        darkmode=False, x_lim=None, y_lim=None,
                        axis_limits=None, return_axes=False):

    active_datasets = [d for d in list_of_datasets if d.select]
    if not active_datasets:
        return None

    y_list = y if isinstance(y, list) else [y]
    axis_limits = axis_limits or {}
    n_sets = len(active_datasets)
    nrows, ncols = _calc_grid(n_sets, nrows, ncols)

    use_secondary = len(y_list) >= 2
    if len(y_list) > 2:
        warnings.warn(
            f"{len(y_list)} y-variables requested; only the first 2 get dedicated "
            "axes per subplot. Variables 3+ will share the secondary axis."
        )

    specs = [[{'secondary_y': use_secondary} for _ in range(ncols)] for _ in range(nrows)]
    sp_titles = [ds.title_format for ds in active_datasets]
    sp_titles += [""] * (nrows * ncols - len(sp_titles))

    fig = make_subplots(
        rows=nrows, cols=ncols, specs=specs,
        subplot_titles=sp_titles,
        horizontal_spacing=0.12 if use_secondary else 0.08,
        vertical_spacing=0.15,
    )
    fig.update_layout(**_base_layout(
        darkmode, suptitle or "Dataset Comparison", figsize,
        showlegend=True, margin=dict(r=60),
    ))

    color_cycle = px.colors.qualitative.Plotly
    primary_y = y_list[0]
    secondary_ys = y_list[1:]

    for idx_ds, dataset in enumerate(active_datasets):
        row = (idx_ds // ncols) + 1
        col = (idx_ds % ncols) + 1

        base_cols = dataset.columns
        if x in base_cols:
            x_col = x
        else:
            cols_upper = {c.upper(): c for c in base_cols}
            x_col = cols_upper.get(str(x).upper())
            if not x_col:
                continue

        valid_hover = [p for p in (display_parms or []) if p in base_cols]
        req_cols = list(dict.fromkeys(
            [x_col] + [yi for yi in y_list if yi in base_cols]
            + valid_hover
            + ([dataset.order] if dataset.order and dataset.order != 'index'
               and dataset.order in base_cols else [])
        ))
        df = dataset.cols(req_cols)
        if dataset.order == 'index':
            df = df.sort_index()
        elif dataset.order:
            df = df.sort_values(by=dataset.order)

        line_dict = dict(width=dataset.linewidth)
        if dataset.linestyle:
            line_dict['dash'] = get_plotly_linestyle(dataset.linestyle)

        hover_cd, hover_lines = build_hover_data(df, valid_hover)
        hover_suffix = "".join(hover_lines)

        if primary_y in df.columns:
            color0 = color_cycle[0]
            ht = (f"<b>{dataset.title}</b><br>{x_col}: %{{x:.2f}}"
                  f"<br>{primary_y}: %{{y:.2f}}{hover_suffix}<extra></extra>")
            fig.add_trace(
                _scatter_cls(len(df))(
                    x=df[x_col], y=df[primary_y],
                    mode='lines+markers' if dataset.linestyle else 'markers',
                    name=primary_y, legendgroup=primary_y,
                    showlegend=(idx_ds == 0),
                    marker=dict(size=dataset.markersize or 6,
                                **_fill_marker_kw(color0, dataset.fill, dataset.edgewidth)),
                    line=dict(color=color0, **line_dict),
                    customdata=hover_cd, hovertemplate=ht,
                ),
                row=row, col=col,
                secondary_y=False if use_secondary else None,
            )
            kw = dict(title_text=primary_y,
                      title_font=dict(color=color0),
                      tickfont=dict(color=color0),
                      row=row, col=col)
            if primary_y in axis_limits:
                kw['range'] = axis_limits[primary_y]
            if use_secondary:
                fig.update_yaxes(secondary_y=False, **kw)
            else:
                fig.update_yaxes(**kw)

        for k, yi in enumerate(secondary_ys):
            if yi not in df.columns:
                continue
            color_k = color_cycle[(k + 1) % len(color_cycle)]
            ht = (f"<b>{dataset.title}</b><br>{x_col}: %{{x:.2f}}"
                  f"<br>{yi}: %{{y:.2f}}{hover_suffix}<extra></extra>")
            fig.add_trace(
                _scatter_cls(len(df))(
                    x=df[x_col], y=df[yi],
                    mode='lines+markers' if dataset.linestyle else 'markers',
                    name=yi, legendgroup=yi,
                    showlegend=(idx_ds == 0),
                    marker=dict(
                        size=dataset.markersize or 6,
                        symbol='circle' if k == 0 else 'diamond',
                        **_fill_marker_kw(color_k, dataset.fill, dataset.edgewidth),
                    ),
                    line=dict(color=color_k, **line_dict),
                    customdata=hover_cd, hovertemplate=ht,
                ),
                row=row, col=col, secondary_y=True,
            )

        if use_secondary and secondary_ys:
            color1 = color_cycle[1]
            kw = dict(title_text=secondary_ys[0],
                      title_font=dict(color=color1),
                      tickfont=dict(color=color1),
                      showgrid=False, row=row, col=col)
            if secondary_ys[0] in axis_limits:
                kw['range'] = axis_limits[secondary_ys[0]]
            fig.update_yaxes(secondary_y=True, **kw)

        fig.update_xaxes(title_text=x, row=row, col=col)
        if x_lim:
            fig.update_xaxes(range=x_lim, row=row, col=col)

    return _show_or_return(fig, return_axes)

def unibar(list_of_datasets, x, y, markers=None, variable_formats=None,
           barmode='group', color=None, 
           suptitle=None, xlabel=None, ylabel=None, subplot_titles=None,
           darkmode=False, figsize=(12, 8), ncols=None, nrows=None, 
           y_lim=None, return_axes=False):
    y_list = y if isinstance(y, list) else [y]
    markers_list = markers if isinstance(markers, list) else ([markers] if markers else [])
    variable_formats = variable_formats or {}
    n_y = len(y_list)
    nrows, ncols = _calc_grid(n_y, nrows, ncols)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles or y_list)
    # scattermode='group' makes the marker/tick/whisker overlays honor their
    # offsetgroup so they sit over their own bar instead of the category
    # center. Only valid when bars themselves are offset (barmode='group').
    fig.update_layout(**_base_layout(
        darkmode, suptitle or f"Bar Comparison: {x}", figsize,
        barmode=barmode, showlegend=True,
        legend=dict(orientation="h"),
        **({'scattermode': 'group'} if barmode == 'group' else {})
    ))

    edge_default = 'white' if darkmode else 'black'

    # Track which (marker_col, y_subplot) legend entries we've already shown,
    # for marker columns that are explicitly styled via var_format.
    marker_legend_shown = set()

    for ds in list_of_datasets:
        if not ds.select: continue
        df = ds.cols([c for c in dict.fromkeys([x] + y_list + markers_list)
                      if c in ds.columns])
        offset_group = f"set_{ds.index}"

        for idx_y, yi in enumerate(y_list):
            row, col = (idx_y // ncols) + 1, (idx_y % ncols) + 1
            if yi not in df.columns: continue

            fig.add_trace(go.Bar(
                x=df[x], y=df[yi],
                name=f"{ds.index}: {ds.title}",
                legendgroup=f"group_{ds.index}",
                offsetgroup=offset_group,
                alignmentgroup="bars",
                marker_color=ds.color if not color else color,
                opacity=ds.alpha,
                showlegend=(idx_y == 0)             # bars always get one legend entry per set
            ), row=row, col=col)

            for m_idx, m_col in enumerate(markers_list):
                if m_col not in df.columns: continue

                var_fmt = variable_formats.get(m_col, {})
                styled = bool(var_fmt)              # styled = has any var_format override

                m_symbol = get_plotly_marker(var_fmt.get('marker') or marker_map(m_idx + 1))
                m_color  = var_fmt.get('color') or (color if color else ds.color)
                m_size   = var_fmt.get('markersize', max(ds.markersize, 10))
                m_alpha  = var_fmt.get('alpha', ds.alpha)
                m_style  = var_fmt.get('style', 'marker')
                # linestyle/linewidth drive the whisker stem: dash style and
                # thickness. 'None' linestyles suppress the stem entirely.
                m_dash   = (get_plotly_linestyle(var_fmt['linestyle'])
                            if var_fmt.get('linestyle') is not None else 'solid')
                m_lw     = var_fmt.get('linewidth', 2)

                # Legend strategy:
                #   styled marker      -> ONE entry per (marker_col, subplot), named just the column
                #   unstyled marker    -> one entry per (dataset, marker_col), grouped with dataset
                if styled:
                    legend_key = (m_col, idx_y)
                    show = legend_key not in marker_legend_shown
                    if show:
                        marker_legend_shown.add(legend_key)
                    trace_name = m_col
                    legend_group = f"marker_{m_col}"
                else:
                    show = (idx_y == 0)
                    trace_name = f"{ds.index}: {ds.title} — {m_col}"
                    legend_group = f"group_{ds.index}"

                fig.add_trace(go.Scatter(
                    x=df[x], y=df[m_col],
                    mode='markers',
                    name=trace_name,
                    legendgroup=legend_group,
                    offsetgroup=offset_group,
                    alignmentgroup="bars",
                    **_overlay_marker_kw(m_style, m_symbol, m_color, m_size,
                                         m_alpha, edge_default,
                                         values=df[m_col], bar_values=df[yi],
                                         stem_dash=m_dash, stem_width=m_lw),
                    showlegend=show,
                    hovertemplate=(f"<b>{ds.title}</b><br>{x}: %{{x}}<br>"
                                   f"{m_col}: %{{y:.4g}}<extra></extra>")
                ), row=row, col=col)

                # Dashed stems can't ride on the marker trace's error bars
                # (always solid), so they get a companion line trace tied to
                # the same legend group.
                if m_style == 'whisker' and m_dash and m_dash != 'solid':
                    fig.add_trace(go.Scatter(
                        name=f"{trace_name} (stem)",
                        legendgroup=legend_group,
                        offsetgroup=offset_group,
                        alignmentgroup="bars",
                        **_whisker_stem_kw(df[x], df[m_col], df[yi],
                                           m_color, m_alpha, m_dash, m_lw),
                    ), row=row, col=col)

    fig.update_xaxes(title_text=xlabel or x)
    fig.update_yaxes(title_text=ylabel or "Value")
    if y_lim: fig.update_yaxes(range=y_lim)

    return _show_or_return(fig, return_axes)
def unibar_per_dataset(list_of_datasets, x, y, markers=None, variable_formats=None,
                       barmode='group',
                       suptitle=None, figsize=(12, 8), ncols=None, nrows=None, 
                       darkmode=False, y_lim=None, return_axes=False):
    """
    Grouped Bar Chart. Subplots are organized by Dataset.

    variable_formats applies to BOTH bar variables and marker columns in
    this view, since color encodes variable (not dataset) within each subplot.
    """
    active_ds = [d for d in list_of_datasets if d.select]
    y_list = y if isinstance(y, list) else [y]
    markers_list = markers if isinstance(markers, list) else ([markers] if markers else [])
    variable_formats = variable_formats or {}
    n_sets = len(active_ds)
    nrows, ncols = _calc_grid(n_sets, nrows, ncols)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[d.title_format for d in active_ds])
    color_cycle = px.colors.qualitative.Plotly
    fig.update_layout(**_base_layout(
        darkmode, suptitle or "Dataset Bar Comparison", figsize,
        barmode=barmode,
        legend=dict(orientation="h"),
        **({'scattermode': 'group'} if barmode == 'group' else {})
    ))

    edge_default = 'white' if darkmode else 'black'

    for idx_ds, ds in enumerate(active_ds):
        row, col = (idx_ds // ncols) + 1, (idx_ds % ncols) + 1
        df = ds.cols([c for c in dict.fromkeys([x] + y_list + markers_list)
                      if c in ds.columns])

        for idx_y, yi in enumerate(y_list):
            if yi not in df.columns: continue
            offset_group = f"var_{yi}"

            var_fmt = variable_formats.get(yi, {})
            bar_color = var_fmt.get('color') or color_cycle[idx_y % len(color_cycle)]
            bar_alpha = var_fmt.get('alpha', 1.0)

            fig.add_trace(go.Bar(
                x=df[x], y=df[yi],
                name=yi,
                legendgroup=yi,
                offsetgroup=offset_group,
                alignmentgroup="bars",
                marker_color=bar_color,
                opacity=bar_alpha,
                showlegend=(idx_ds == 0)
            ), row=row, col=col)

        for m_idx, m_col in enumerate(markers_list):
            if m_col not in df.columns: continue

            var_fmt = variable_formats.get(m_col, {})
            m_symbol = get_plotly_marker(var_fmt.get('marker') or marker_map(m_idx))
            m_color  = var_fmt.get('color') or color_cycle[(len(y_list) + m_idx) % len(color_cycle)]
            m_size   = var_fmt.get('markersize', 12)
            m_alpha  = var_fmt.get('alpha', 1.0)
            m_style  = var_fmt.get('style', 'marker')
            m_dash   = (get_plotly_linestyle(var_fmt['linestyle'])
                        if var_fmt.get('linestyle') is not None else 'solid')
            m_lw     = var_fmt.get('linewidth', 2)

            # Color encodes variable in this view, so an overlay column isn't
            # tied to one bar series: tick/whisker glyphs attach to the FIRST
            # plotted y variable's bars (classic markers stay at the category
            # center, as before).
            anchor = next((yy for yy in y_list if yy in df.columns), None)
            attach = m_style in ('tick', 'whisker') and anchor is not None
            group_kw = ({'offsetgroup': f"var_{anchor}", 'alignmentgroup': "bars"}
                        if attach else {})

            fig.add_trace(go.Scatter(
                x=df[x], y=df[m_col],
                mode='markers',
                name=m_col,
                legendgroup=f"marker_{m_col}",
                **group_kw,
                **_overlay_marker_kw(m_style, m_symbol, m_color, m_size,
                                     m_alpha, edge_default,
                                     values=df[m_col],
                                     bar_values=df[anchor] if attach else None,
                                     stem_dash=m_dash, stem_width=m_lw),
                showlegend=(idx_ds == 0),
                hovertemplate=f"<b>{m_col}</b><br>{x}: %{{x}}<br>%{{y:.4g}}<extra></extra>"
            ), row=row, col=col)

            if m_style == 'whisker' and attach and m_dash and m_dash != 'solid':
                fig.add_trace(go.Scatter(
                    name=f"{m_col} (stem)",
                    legendgroup=f"marker_{m_col}",
                    **group_kw,
                    **_whisker_stem_kw(df[x], df[m_col], df[anchor],
                                       m_color, m_alpha, m_dash, m_lw),
                ), row=row, col=col)

    fig.update_xaxes(title_text=x)
    if y_lim: fig.update_yaxes(range=y_lim)

    return _show_or_return(fig, return_axes)

def unibox(list_of_datasets, x, y, boxmode='group', points='outliers', notched=False,
           color=None, suptitle=None, xlabel=None, ylabel=None, subplot_titles=None,
           darkmode=False, figsize=(12, 8), ncols=None, nrows=None, 
           y_lim=None, return_axes=False):
    """
    Boxplot version of uniplot.
    Subplots are organized by Y-variables.
    """
    y_list = y if isinstance(y, list) else [y]
    n_y = len(y_list)
    nrows, ncols = _calc_grid(n_y, nrows, ncols)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles or y_list)
    fig.update_layout(**_base_layout(
        darkmode, suptitle or f"Boxplot Comparison: {x}", figsize,
        boxmode=boxmode, showlegend=True,
        margin=dict(r=80),
        legend=dict(orientation="h"),
    ))

    for ds in list_of_datasets:
        if not ds.select: continue
        df = ds.cols([c for c in dict.fromkeys([x] + y_list) if c in ds.columns])

        for idx_y, yi in enumerate(y_list):
            row, col = (idx_y // ncols) + 1, (idx_y % ncols) + 1
            if yi not in df.columns: continue

            fig.add_trace(go.Box(
                x=df[x], 
                y=df[yi],
                name=f"{ds.index}: {ds.title}",
                legendgroup=f"group_{ds.index}",
                marker_color=ds.color if not color else color,
                opacity=ds.alpha,
                boxpoints=points,
                notched=notched,
                line=dict(width=ds.linewidth),
                showlegend=(idx_y == 0)
            ), row=row, col=col)

    fig.update_xaxes(title_text=xlabel or x)
    fig.update_yaxes(title_text=ylabel or "Value")
    if y_lim: fig.update_yaxes(range=y_lim)

    return _show_or_return(fig, return_axes)

def unibox_per_dataset(list_of_datasets, x, y, boxmode='group', points='outliers', notched=False,
                       suptitle=None, figsize=(12, 8), ncols=None, nrows=None, 
                       darkmode=False, y_lim=None, return_axes=False):
    """
    Boxplot version of uniplot_per_dataset.
    Subplots are organized by Dataset.
    """
    active_ds = [d for d in list_of_datasets if d.select]
    y_list = y if isinstance(y, list) else [y]
    n_sets = len(active_ds)
    nrows, ncols = _calc_grid(n_sets, nrows, ncols)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[d.title_format for d in active_ds])
    color_cycle = px.colors.qualitative.Plotly
    fig.update_layout(**_base_layout(
        darkmode, suptitle or "Dataset Box Comparison", figsize,
        boxmode=boxmode, showlegend=True,
        margin=dict(r=80),
        legend=dict(orientation="h"),
    ))

    for idx_ds, ds in enumerate(active_ds):
        row, col = (idx_ds // ncols) + 1, (idx_ds % ncols) + 1
        df = ds.cols([c for c in dict.fromkeys([x] + y_list) if c in ds.columns])

        for idx_y, yi in enumerate(y_list):
            if yi not in df.columns: continue

            fig.add_trace(go.Box(
                x=df[x], 
                y=df[yi],
                name=yi,
                legendgroup=yi,
                marker_color=color_cycle[idx_y % len(color_cycle)],
                boxpoints=points,
                notched=notched,
                showlegend=(idx_ds == 0)
            ), row=row, col=col)

    fig.update_xaxes(title_text=x)
    if y_lim: fig.update_yaxes(range=y_lim)

    return _show_or_return(fig, return_axes)


def unihistogram(list_of_datasets, x, y=None, histfunc='sum', nbins=None,
                 bin_size=None, bin_start=None, bin_end=None,
                 histnorm='', barmode='overlay', alpha=0.7,
                 color=None, suptitle=None, subplot_titles=None, darkmode=False,
                 figsize=(12, 8), ncols=None, nrows=None, x_lim=None, return_axes=False,
                 opacity=None):
    """
    Create a unified histogram for a list of datasets.
    Subplots are organized by Variable (x).
    """
    if opacity is not None:
        warnings.warn("'opacity' is deprecated, use 'alpha'", DeprecationWarning, stacklevel=2)
        alpha = opacity
    x_list = x if isinstance(x, list) else [x]
    n_x = len(x_list)
    nrows, ncols = _calc_grid(n_x, nrows, ncols)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles or x_list)
    fig.update_layout(**_base_layout(
        darkmode, suptitle or "Distribution Comparison", figsize,
        barmode=barmode, showlegend=True,
        legend=dict(orientation="h")
    ))

    xbins = _build_xbins(bin_size, bin_start, bin_end)

    for ds in list_of_datasets:
        if not ds.select: continue
        df = ds.cols([c for c in dict.fromkeys(x_list + ([y] if y else []))
                      if c in ds.columns])

        use_color = color if color else ds.color

        for idx_x, xi in enumerate(x_list):
            row, col = (idx_x // ncols) + 1, (idx_x % ncols) + 1
            
            if xi not in df.columns: continue

            subset_cols = [xi] if y is None else [xi, y]
            if y and y not in df.columns: continue
            
            clean_data = df.dropna(subset=subset_cols)
            if clean_data.empty: continue

            trace_args = dict(
                x=clean_data[xi],
                name=f"{ds.index}: {ds.title}",
                legendgroup=f"group_{ds.index}",
                marker_color=use_color,
                opacity=alpha,
                nbinsx=nbins,
                xbins=xbins,
                histnorm=histnorm,
                showlegend=(idx_x == 0)
            )

            if y:
                trace_args['y'] = clean_data[y]
                trace_args['histfunc'] = histfunc

            fig.add_trace(go.Histogram(**trace_args), row=row, col=col)

    if x_lim: fig.update_xaxes(range=x_lim)

    y_label = f"Sum of {y}" if y else ("Density" if "density" in histnorm else "Count")
    fig.update_yaxes(title_text=y_label)

    return _show_or_return(fig, return_axes)

def unihistogram_by_dataset(list_of_datasets, x, y=None, histfunc='sum', nbins=None,
                            bin_size=None, bin_start=None, bin_end=None,
                            histnorm='', barmode='overlay', alpha=0.7,
                            variable_formats=None,
                            color=None, suptitle=None, figsize=(12, 8), ncols=None, nrows=None,
                            darkmode=False, x_lim=None, return_axes=False,
                            opacity=None):
    """
    Create a unified histogram where Subplots are organized by Dataset.

    With one subplot per dataset, the histogrammed x-variables are what color
    distinguishes. A per-variable ``variable_formats`` override supplies that
    variable's ``color`` (and/or ``alpha``), taking precedence over the global
    ``color`` arg and the default per-variable color cycle. Other format
    attributes have no meaning for a histogram and are ignored.
    """
    if opacity is not None:
        warnings.warn("'opacity' is deprecated, use 'alpha'", DeprecationWarning, stacklevel=2)
        alpha = opacity
    variable_formats = variable_formats or {}
    active_ds = [d for d in list_of_datasets if d.select]
    x_list = x if isinstance(x, list) else [x]
    n_sets = len(active_ds)

    if not active_ds:
        print("No datasets selected.")
        return None

    nrows, ncols = _calc_grid(n_sets, nrows, ncols)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[d.title_format for d in active_ds])
    color_cycle = px.colors.qualitative.Plotly
    fig.update_layout(**_base_layout(
        darkmode, suptitle or "Dataset Distribution Analysis", figsize,
        barmode=barmode, showlegend=True,
        legend=dict(orientation="h")
    ))

    xbins = _build_xbins(bin_size, bin_start, bin_end)

    for idx_ds, ds in enumerate(active_ds):
        row, col = (idx_ds // ncols) + 1, (idx_ds % ncols) + 1
        df = ds.cols([c for c in dict.fromkeys(x_list + ([y] if y else []))
                      if c in ds.columns])

        for idx_x, xi in enumerate(x_list):
            if xi not in df.columns: continue
            
            subset_cols = [xi] if y is None else [xi, y]
            if y and y not in df.columns: continue
            
            clean_data = df.dropna(subset=subset_cols)
            if clean_data.empty: continue
            
            var_fmt = variable_formats.get(xi, {})
            if var_fmt.get('color'):
                use_color = var_fmt['color']
            elif color:
                use_color = color
            elif len(x_list) == 1:
                use_color = ds.color
            else:
                use_color = color_cycle[idx_x % len(color_cycle)]
            use_alpha = var_fmt.get('alpha', alpha)

            trace_args = dict(
                x=clean_data[xi],
                name=xi,
                legendgroup=xi,
                marker_color=use_color,
                opacity=use_alpha,
                nbinsx=nbins,
                xbins=xbins,
                histnorm=histnorm,
                showlegend=(idx_ds == 0)
            )

            if y:
                trace_args['y'] = clean_data[y]
                trace_args['histfunc'] = histfunc

            fig.add_trace(go.Histogram(**trace_args), row=row, col=col)

    if x_lim: fig.update_xaxes(range=x_lim)

    y_label = f"Sum of {y}" if y else ("Density" if "density" in histnorm else "Count")
    fig.update_yaxes(title_text=y_label)

    return _show_or_return(fig, return_axes)

def _add_contour_overlays(fig, overlay_datasets, x, y, n_subplots, ncols, darkmode):
    """Draw each overlay dataset's ``(x, y)`` on top of every contour subplot.

    Used by the contour builders to lay discrete sample points — or, when a set
    has a ``linestyle``, a connected boundary line — over the interpolated
    field. The same overlay sets are repeated on all ``n_subplots`` cells
    (contour subplots share the same x/y axes), added *after* the contour traces
    so they sit above an opaque ``contours_coloring='fill'``.

    Each set keeps its own plot style: ``color``, ``marker``, ``markersize``,
    ``linestyle``/``linewidth``, ``alpha``, and ``fill`` (hollow markers when
    off). The mode mirrors :func:`uniplot` — a set with a ``linestyle`` draws as
    a line (so it can trace a contour boundary), otherwise as markers. Points
    are connected in the set's own order (its ``order`` column when set, else
    row order), so an explicitly drawn boundary keeps its shape; regression
    fits are not applied. The legend entry shows once (first subplot), grouped
    per dataset.
    """
    if not overlay_datasets:
        return

    edge_default = 'white' if darkmode else 'black'

    # Fetch and style each overlay set once (the same trace repeats on every
    # cell, so re-slicing the set per cell only multiplied the cost), then add
    # traces cell-major to keep the original draw/legend order.
    prepared = []
    for ds in overlay_datasets:
        ds_cols = ds.columns
        if x not in ds_cols or y not in ds_cols:
            continue

        order_in_cols = ds.order and ds.order != 'index' and ds.order in ds_cols
        df = ds.cols(list(dict.fromkeys([x, y] + ([ds.order] if order_in_cols else []))))

        # Connect points in the set's own order so a hand-drawn boundary
        # keeps its shape (an `order` column if set, else original rows).
        if order_in_cols:
            df = df.sort_values(by=ds.order)

        # A linestyle means "draw a line" (trace a boundary); markers show
        # unless turned off with marker=None.
        want_lines = bool(ds.linestyle)
        want_markers = ds.marker is not None
        mode = '+'.join(['lines'] * want_lines + ['markers'] * want_markers) or 'none'

        marker_dict = dict(
            symbol=get_plotly_marker(ds.marker),
            size=ds.markersize,
            color=ds.color if ds.fill else 'rgba(0,0,0,0)',
            line=dict(width=ds.edgewidth,
                      color=ds.color if not ds.fill
                      else (ds.edge_color or edge_default)),
        )
        prepared.append((ds, df, mode, marker_dict))

    for cell in range(n_subplots):
        row, col = (cell // ncols) + 1, (cell % ncols) + 1
        for ds, df, mode, marker_dict in prepared:
            fig.add_trace(go.Scatter(
                x=df[x], y=df[y],
                mode=mode,
                name=f"{ds.index}: {ds.title}",
                legendgroup=f"overlay_{ds.index}",
                marker=marker_dict,
                line=dict(width=ds.linewidth, color=ds.color,
                          dash=get_plotly_linestyle(ds.linestyle)),
                opacity=ds.alpha,
                showlegend=(cell == 0),
                hovertemplate=(f"<b>{ds.title}</b><br>{x}: %{{x:.3g}}<br>"
                               f"{y}: %{{y:.3g}}<extra></extra>")
            ), row=row, col=col)

def unicontour(list_of_datasets, x, y, z, contours_coloring='fill', colorscale=None,
               interpolate=True, interp_res=100, interp_method='linear',
               ncontours=None, overlay_datasets=None,
               suptitle=None, xlabel=None, ylabel=None, subplot_titles=None,
               darkmode=False, figsize=(12, 8), ncols=None, nrows=None,
               axis_limits=None, return_axes=False):
    """
    Create a unified contour plot for a list of datasets.
    Subplots are organized by Z-variables.
    """
    z_list = z if isinstance(z, list) else [z]
    n_z = len(z_list)
    active_ds = [d for d in list_of_datasets if d.select]
    axis_limits = axis_limits or {}

    if not active_ds:
        print("No datasets selected.")
        return None

    nrows, ncols = _calc_grid(n_z, nrows, ncols)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles or z_list, horizontal_spacing=0.15)
    fig.update_layout(**_base_layout(
        darkmode, None, figsize,
        title={'text': suptitle or f"Contour: {y} vs {x}", 'x': 0.5, 'xanchor': 'center', 'y': 0.98, 'yanchor': 'top', 'yref': 'container'},
        showlegend=True, margin=dict(r=100),
        legend=dict(orientation="h"),
    ))

    for idx_ds, ds in enumerate(active_ds):
        df = ds.cols([c for c in dict.fromkeys([x, y] + z_list) if c in ds.columns])
        if x not in df.columns or y not in df.columns:
            continue

        for idx_z, zi in enumerate(z_list):
            if zi not in df.columns: continue
            row, col = (idx_z // ncols) + 1, (idx_z % ncols) + 1

            use_coloring = 'lines' if len(active_ds) > 1 and contours_coloring == 'fill' else contours_coloring

            clean_df = df.dropna(subset=[x, y, zi])
            if clean_df.empty: continue

            if interpolate:
                plot_x, plot_y, plot_z = _generate_contour_grid(
                    clean_df[x], clean_df[y], clean_df[zi], 
                    res=interp_res, method=interp_method
                )
                if plot_x is None:
                    print(f"Skipping contour trace for set {ds.index} (Z={zi}): Insufficient points or collinear data.")
                    continue
            else:
                plot_x, plot_y, plot_z = clean_df[x], clean_df[y], clean_df[zi]

            z_lim = axis_limits.get(zi)
            zmin, zmax = z_lim if z_lim else (None, None)

            subplot_idx = (row - 1) * ncols + col
            x_axis_name = f"xaxis{subplot_idx}" if subplot_idx > 1 else "xaxis"
            y_axis_name = f"yaxis{subplot_idx}" if subplot_idx > 1 else "yaxis"
            
            try:
                x_domain = fig.layout[x_axis_name].domain
                y_domain = fig.layout[y_axis_name].domain
                cb_x = x_domain[1] + 0.01               
                cb_y = sum(y_domain) / 2                
                cb_len = y_domain[1] - y_domain[0]      
            except KeyError:
                cb_x, cb_y, cb_len = 1.02, 0.5, 1.0     

            fig.add_trace(go.Contour(
                x=plot_x, y=plot_y, z=plot_z,
                zmin=zmin, zmax=zmax,  
                name=f"{ds.index}: {ds.title}",
                legendgroup=f"group_{ds.index}",
                colorscale=colorscale or ds.hue_palette,
                contours_coloring=use_coloring,
                ncontours=ncontours, 
                line=dict(width=ds.linewidth, color=ds.color if use_coloring=='lines' else None),
                showscale=(idx_ds == 0),
                showlegend=(idx_z == 0),
                colorbar=dict(
                    title=zi,
                    x=cb_x,
                    y=cb_y,
                    len=cb_len,
                    thickness=15
                ),
                hovertemplate=f"<b>{ds.title}</b><br>{x}: %{{x:.3g}}<br>{y}: %{{y:.3g}}<br>{zi}: %{{z:.3g}}<extra></extra>"
            ), row=row, col=col)

    _add_contour_overlays(fig, overlay_datasets, x, y, n_z, ncols, darkmode)

    fig.update_xaxes(title_text=xlabel or x)
    fig.update_yaxes(title_text=ylabel or y)

    return _show_or_return(fig, return_axes)

def unicontour_per_dataset(list_of_datasets, x, y, z, contours_coloring='fill', colorscale=None,
                           interpolate=True, interp_res=100, interp_method='linear',
                           ncontours=None, overlay_datasets=None,
                           suptitle=None, figsize=(12, 8), ncols=None, nrows=None,
                           darkmode=False, axis_limits=None, return_axes=False):
    """
    Contour plot where Subplots are organized by Dataset.
    """
    active_ds = [d for d in list_of_datasets if d.select]
    z_list = z if isinstance(z, list) else [z]
    n_sets = len(active_ds)
    axis_limits = axis_limits or {}

    if not active_ds:
        print("No datasets selected.")
        return None

    nrows, ncols = _calc_grid(n_sets, nrows, ncols)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[d.title_format for d in active_ds], horizontal_spacing=0.15)
    fig.update_layout(**_base_layout(
        darkmode, None, figsize,
        title={'text': suptitle or "Dataset Contour Comparison", 'x': 0.5, 'y': 0.98, 'yref': 'container'},
        showlegend=True, margin=dict(r=100),
        legend=dict(orientation="h"),
    ))

    for idx_ds, ds in enumerate(active_ds):
        row, col = (idx_ds // ncols) + 1, (idx_ds % ncols) + 1
        df = ds.cols([c for c in dict.fromkeys([x, y] + z_list) if c in ds.columns])

        if x not in df.columns or y not in df.columns: continue

        for idx_z, zi in enumerate(z_list):
            if zi not in df.columns: continue
            
            use_coloring = 'lines' if len(z_list) > 1 and contours_coloring == 'fill' else contours_coloring

            clean_df = df.dropna(subset=[x, y, zi])
            if clean_df.empty: continue

            if interpolate:
                plot_x, plot_y, plot_z = _generate_contour_grid(
                    clean_df[x], clean_df[y], clean_df[zi], 
                    res=interp_res, method=interp_method
                )
            else:
                plot_x, plot_y, plot_z = clean_df[x], clean_df[y], clean_df[zi]

            z_lim = axis_limits.get(zi)
            zmin, zmax = z_lim if z_lim else (None, None)

            subplot_idx = (row - 1) * ncols + col
            x_axis_name = f"xaxis{subplot_idx}" if subplot_idx > 1 else "xaxis"
            y_axis_name = f"yaxis{subplot_idx}" if subplot_idx > 1 else "yaxis"
            
            try:
                x_domain = fig.layout[x_axis_name].domain
                y_domain = fig.layout[y_axis_name].domain
                cb_x = x_domain[1] + 0.01 + (idx_z * 0.05) 
                cb_y = sum(y_domain) / 2
                cb_len = y_domain[1] - y_domain[0]
            except KeyError:
                cb_x, cb_y, cb_len = 1.02 + (idx_z * 0.05), 0.5, 1.0

            fig.add_trace(go.Contour(
                x=plot_x, y=plot_y, z=plot_z,
                zmin=zmin, zmax=zmax,  
                name=zi,
                legendgroup=zi,
                colorscale=colorscale or ds.hue_palette,
                contours_coloring=use_coloring,
                ncontours=ncontours,
                showscale=(idx_ds == 0),
                showlegend=(idx_ds == 0),
                colorbar=dict(
                    title=zi,
                    x=cb_x,
                    y=cb_y,
                    len=cb_len,
                    thickness=15
                ),
                hovertemplate=f"<b>{zi}</b><br>{x}: %{{x:.3g}}<br>{y}: %{{y:.3g}}<br>Value: %{{z:.3g}}<extra></extra>"
            ), row=row, col=col)

    _add_contour_overlays(fig, overlay_datasets, x, y, n_sets, ncols, darkmode)

    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)

    return _show_or_return(fig, return_axes)

def unibar_datasets_as_x(list_of_datasets, y, agg='mean', variable_formats=None,
                         suptitle=None, darkmode=False,
                         figsize=(12, 8), axis_limits=None, return_axes=False):
    """
    Creates a single grouped bar chart where the X-axis is the Dataset name,
    and the bars are the different Y-variables, each scaled to their own Y-axis.
    Includes an 'agg' parameter to handle multi-row datasets.

    Per-variable ``variable_formats`` overrides apply here: a variable's
    ``color`` recolors its bar and its Y-axis (overriding the default color
    cycle), and ``alpha`` sets the bar opacity. Other format attributes
    (marker/linestyle/etc.) have no meaning in this chart and are ignored.
    """
    active_ds = [d for d in list_of_datasets if d.select]
    if not active_ds:
        print("No datasets selected.")
        return None

    y_list = y if isinstance(y, list) else [y]
    axis_limits = axis_limits or {}
    variable_formats = variable_formats or {}
    color_cycle = px.colors.qualitative.Plotly

    fig = go.Figure()

    x_labels = [f"{ds.index}: {ds.title}" for ds in active_ds]

    extras_count = max(0, len(y_list) - 2)
    width_per_axis = 0.08
    required_space = extras_count * width_per_axis
    x_domain_end = max(0.5, 1.0 - required_space) 

    for idx_y, yi in enumerate(y_list):
        var_fmt = variable_formats.get(yi, {})
        var_color = var_fmt.get('color', color_cycle[idx_y % len(color_cycle)])
        var_alpha = var_fmt.get('alpha')

        y_data = []
        for ds in active_ds:
            if yi in ds.columns:
                valid_data = ds[yi].dropna()
                if valid_data.empty:
                    val = None
                elif agg == 'mean': val = valid_data.mean()
                elif agg == 'sum': val = valid_data.sum()
                elif agg == 'max': val = valid_data.max()
                elif agg == 'min': val = valid_data.min()
                elif agg == 'median': val = valid_data.median()
                elif agg == 'first': val = valid_data.iloc[0]
                elif agg == 'last': val = valid_data.iloc[-1]
                else:
                    print(f"Warning: Unknown agg '{agg}', defaulting to mean.")
                    val = valid_data.mean()
                y_data.append(val)
            else:
                y_data.append(None)

        y_axis_name = "y" if idx_y == 0 else f"y{idx_y + 1}"

        fig.add_trace(go.Bar(
            name=yi,
            x=x_labels,
            y=y_data,
            yaxis=y_axis_name,
            offsetgroup=str(idx_y),
            marker_color=var_color,
            opacity=var_alpha if var_alpha is not None else 1.0
        ))

        axis_layout = dict(
            title=yi,
            title_font=dict(color=var_color),
            tickfont=dict(color=var_color),
            showgrid=(idx_y == 0) 
        )

        if yi in axis_limits:
            axis_layout['range'] = axis_limits[yi]

        if idx_y == 0:
            fig.update_layout(yaxis=axis_layout)
        elif idx_y == 1:
            axis_layout.update(dict(overlaying='y', side='right', anchor='x'))
            fig.update_layout(yaxis2=axis_layout)
        else:
            pos = x_domain_end + ((idx_y - 1) * width_per_axis)
            axis_layout.update(dict(overlaying='y', side='right', anchor='free', position=pos))
            fig.update_layout({f"yaxis{idx_y + 1}": axis_layout})

    fig.update_layout(**_base_layout(
        darkmode, suptitle or f"Variables by Dataset ({agg})", figsize,
        barmode='group',
        xaxis=dict(domain=[0, x_domain_end], title="Dataset"),
        margin=dict(r=50 + (extras_count * 80)),
        legend=dict(orientation="h")
    ))

    return _show_or_return(fig, return_axes)

def unibox_datasets_as_x(list_of_datasets, y, boxmode='group', points='outliers', notched=False,
                         variable_formats=None,
                         suptitle=None, darkmode=False, figsize=(12, 8), axis_limits=None, return_axes=False):
    """
    Creates a single grouped box plot where the X-axis is the Dataset name,
    and the boxes are the different Y-variables, each scaled to their own Y-axis.

    Each Y-variable is colored from the default palette unless a per-variable
    ``variable_formats`` override supplies a ``color`` (and/or ``alpha``); the
    override color also drives that variable's Y-axis title/tick color.
    """
    active_ds = [d for d in list_of_datasets if d.select]
    if not active_ds:
        print("No datasets selected.")
        return None

    y_list = y if isinstance(y, list) else [y]
    axis_limits = axis_limits or {}
    variable_formats = variable_formats or {}
    color_cycle = px.colors.qualitative.Plotly

    fig = go.Figure()

    extras_count = max(0, len(y_list) - 2)
    width_per_axis = 0.08
    required_space = extras_count * width_per_axis
    x_domain_end = max(0.5, 1.0 - required_space)

    for idx_y, yi in enumerate(y_list):
        var_fmt = variable_formats.get(yi, {})
        var_color = var_fmt.get('color') or color_cycle[idx_y % len(color_cycle)]
        var_alpha = var_fmt.get('alpha')

        all_y_data = []
        all_x_labels = []

        for ds in active_ds:
            if yi in ds.columns:
                valid_data = ds[yi].dropna()
                if not valid_data.empty:
                    all_y_data.extend(valid_data.values)
                    label = f"{ds.index}: {ds.title}"
                    all_x_labels.extend([label] * len(valid_data))

        if not all_y_data:
            continue

        y_axis_name = "y" if idx_y == 0 else f"y{idx_y + 1}"

        box_kwargs = dict(
            name=yi,
            x=all_x_labels,
            y=all_y_data,
            yaxis=y_axis_name,
            offsetgroup=str(idx_y),
            marker_color=var_color,
            boxpoints=points,
            notched=notched,
        )
        if var_alpha is not None:
            box_kwargs['opacity'] = var_alpha
        fig.add_trace(go.Box(**box_kwargs))

        axis_layout = dict(
            title=yi,
            title_font=dict(color=var_color),
            tickfont=dict(color=var_color),
            showgrid=(idx_y == 0) 
        )

        if yi in axis_limits:
            axis_layout['range'] = axis_limits[yi]

        if idx_y == 0:
            fig.update_layout(yaxis=axis_layout)
        elif idx_y == 1:
            axis_layout.update(dict(overlaying='y', side='right', anchor='x'))
            fig.update_layout(yaxis2=axis_layout)
        else:
            pos = x_domain_end + ((idx_y - 1) * width_per_axis)
            axis_layout.update(dict(overlaying='y', side='right', anchor='free', position=pos))
            fig.update_layout({f"yaxis{idx_y + 1}": axis_layout})

    fig.update_layout(**_base_layout(
        darkmode, suptitle or "Variables by Dataset", figsize,
        boxmode=boxmode,
        xaxis=dict(domain=[0, x_domain_end], title="Dataset"),
        margin=dict(r=50 + (extras_count * 80)),
        legend=dict(orientation="h"),
    ))

    return _show_or_return(fig, return_axes)


# -----------------------------------------------------------------------------
# Multi-Y-Axis Scatter/Line Plot
# -----------------------------------------------------------------------------
def uniplot_ymultaxis(list_of_datasets, x, y,
                        variable_formats=None, display_parms=None,
                        suptitle=None, xlabel=None,
                        darkmode=False, figsize=(12, 8),
                        x_lim=None, axis_limits=None,
                        legend='right', legend_group_by='sets', return_axes=False):
    """
    Single-plot, multi-Y-axis scatter/line chart.

    All selected datasets are overlaid on the same x-axis. Each y variable
    in `y` gets its own y-axis (left for the first, right for the second,
    further right for the third+). One trace per (dataset × variable).

    Formatting precedence (per attribute):
        variable_formats[var][attr]   →   if set, use this
        dataset.<attr>                →   otherwise, fall back to dataset

    This is intentionally per-attribute, so you can do things like
    "set linestyle on the variable, but let color come from the dataset",
    or vice versa.

    Parameters
    ----------
    list_of_datasets : list[Dataset]
    x : str
        X-column name (shared across all traces).
    y : str | list[str]
        One or more y-column names. Each gets its own y-axis.
    variable_formats : dict[str, dict] | None
        Per-variable overrides, e.g.
            {'Temp': {'linestyle': '--'}, 'Pressure': {'color': 'blue'}}
        Recognized keys: color, marker, linestyle, markersize, linewidth, alpha.
    display_parms : list[str] | None
        Extra columns to surface in the hover tooltip.
    axis_limits : dict[str, tuple] | None
        Per-column (min, max) limits. Applies to x and to any y axis.
    legend : 'right' | 'above' | 'off'
    """
    variable_formats = variable_formats or {}
    axis_limits = axis_limits or {}

    active = [d for d in list_of_datasets if d.select]
    if not active:
        print("No datasets selected.")
        return None

    y_list = y if isinstance(y, list) else [y]
    if not y_list:
        raise ValueError("At least one y variable is required.")

    # Domain math: extra y-axes (3+) live to the right of the plot.
    extras = max(0, len(y_list) - 2)
    width_per_axis = 0.06
    x_domain_end = max(0.5, 1.0 - extras * width_per_axis)

    # Axis label colors: only apply when explicitly set in variable_formats.
    axis_label_colors = {yi: variable_formats.get(yi, {}).get('color') for yi in y_list}

    fig = go.Figure()

    for ds in active:
        base_cols = ds.columns
        if x not in base_cols:
            continue

        ds_hover = display_parms if display_parms is not None else getattr(ds, 'display_parms', [])
        valid_hover = [p for p in (ds_hover or []) if p in base_cols]

        # Fetch only the columns this plot touches, then sort the narrow
        # frame — sorting the full set width dominated on wide frames.
        order_col = getattr(ds, 'order', None)
        order_in_cols = bool(order_col) and order_col != 'index' and order_col in base_cols
        needed = [x] + [yi for yi in y_list if yi in base_cols] + valid_hover
        if order_in_cols:
            needed.append(order_col)
        sorted_base_df = ds.cols(list(dict.fromkeys(needed)))
        if order_in_cols:
            sorted_base_df = sorted_base_df.sort_values(by=order_col)
        else:
            sorted_base_df = sorted_base_df.sort_index()

        for idx_y, yi in enumerate(y_list):
            if yi not in base_cols:
                continue

            fmt = _resolve_var_format(ds, yi, variable_formats)

            req_cols = list(dict.fromkeys([x, yi] + valid_hover))
            df = sorted_base_df[req_cols]

            parts = []
            if fmt['linestyle']:
                parts.append('lines')
            if fmt['marker'] is not None:   # marker=None turns markers off
                parts.append('markers')
            mode = '+'.join(parts) if parts else 'none'

            ht = (f"<b>Set {ds.index}: {ds.title}</b><br>"
                  f"<b>{yi}</b><br>"
                  f"{x}: %{{x:.4g}}<br>"
                  f"{yi}: %{{y:.4g}}")
            customdata, hover_lines = build_hover_data(df, valid_hover)
            ht += "".join(hover_lines)
            ht += "<extra></extra>"

            y_axis_name = "y" if idx_y == 0 else f"y{idx_y + 1}"

            if fmt.get('fill', True):
                marker_kw = dict(
                    color=fmt['color'],
                    line=dict(width=fmt['edgewidth'], color=fmt['edge_color']),
                )
            else:
                # No fill: hollow marker whose outline takes the set color.
                marker_kw = dict(
                    color='rgba(0,0,0,0)',
                    line=dict(width=fmt['edgewidth'], color=fmt['color']),
                )

            fig.add_trace(_scatter_cls(len(df))(
                x=df[x], y=df[yi],
                mode=mode,
                name=f"{ds.index}: {ds.title}" if legend_group_by == 'vars' else yi,
                yaxis=y_axis_name,
                legendgroup=f"var_{yi}" if legend_group_by == 'vars' else f"set_{ds.index}",
                legendgrouptitle_text=yi if legend_group_by == 'vars' else f"{ds.index}: {ds.title}",
                marker=dict(
                    size=fmt['markersize'],
                    symbol=get_plotly_marker(fmt['marker']),
                    opacity=fmt['alpha'],
                    **marker_kw,
                ),
                line=dict(
                    width=fmt['linewidth'],
                    dash=get_plotly_linestyle(fmt['linestyle']) or 'solid',
                    color=fmt['color'],
                ),
                customdata=customdata,
                hovertemplate=ht,
            ))

    # Color axes only when explicitly set via variable_formats; otherwise use Plotly's default
    for idx_y, yi in enumerate(y_list):
        ax_color = axis_label_colors[yi]
        title_kw = dict(text=yi)
        if ax_color:
            title_kw['font'] = dict(color=ax_color)
        ax = dict(title=title_kw, showgrid=(idx_y == 0))
        if ax_color:
            ax['tickfont'] = dict(color=ax_color)
        if yi in axis_limits:
            ax['range'] = axis_limits[yi]

        if idx_y == 0:
            fig.update_layout(yaxis=ax)
        elif idx_y == 1:
            ax.update(dict(overlaying='y', side='right', anchor='x'))
            fig.update_layout(yaxis2=ax)
        else:
            pos = x_domain_end + (idx_y - 1) * width_per_axis
            ax.update(dict(overlaying='y', side='right', anchor='free', position=pos))
            fig.update_layout({f"yaxis{idx_y + 1}": ax})

    layout_extras = dict(
        showlegend=(legend != 'off'),
        xaxis=dict(domain=[0, x_domain_end], title=xlabel or x),
        margin=dict(r=60 + extras * 70),
    )
    if x_lim:
        layout_extras['xaxis']['range'] = x_lim
    elif x in axis_limits:
        layout_extras['xaxis']['range'] = axis_limits[x]

    if legend == 'above':
        # vertical geometry filled in by _base_layout; keep the centered x.
        layout_extras['legend'] = dict(orientation='h', xanchor='center', x=0.5)

    fig.update_layout(**_base_layout(
        darkmode,
        suptitle or f"{', '.join(y_list)} vs {x}",
        figsize,
        **layout_extras,
    ))

    return _show_or_return(fig, return_axes)

class UnichartNotebook:
    """Interactive multi-dataset plotting environment for notebooks.

    A notebook holds any number of datasets (``nb.sets``, each a :class:`Dataset`)
    backed by a single shared DataFrame, and turns them into Plotly figures with a
    concise, stateful API. The typical workflow is:

        1. Load data          -> ``load_df`` / ``load`` / ``load_clipboard``
        2. Select what to show -> ``select`` / ``omit`` / ``query``
        3. Plot                -> ``plot`` / ``plot_ymult``
        4. Style               -> ``color`` / ``marker`` / ``var_format`` /
                                  ``set_default_format``
        5. Analyse             -> ``delta`` / ``table`` / ``combine_sets``

    Plot calls remember their last arguments (``last_x``, ``last_y``, ...), so
    follow-up styling and analysis calls can be made without re-specifying them.
    Call ``nb.help()`` for a categorized method listing, or ``nb.help('name')``
    for the full documentation of a single method.
    """

    def __init__(self):
        self.sets = []
        self._combined_df = pd.DataFrame({_SET_ID_COL: pd.Series(dtype='int64')})
        self._next_set_id = 0
        # Per-set row-position cache for the combined frame; see _set_positions.
        self._set_row_pos = {}
        # Column-ownership reconciliation state; see _reconcile_columns.
        self._cols_snapshot = self._combined_df.columns
        self._known_cols = set(self._combined_df.columns)

        # State Memory
        self.last_x = None
        self.last_y = None
        self.last_format = 'stack'
        self.last_ymult_format = 'color'
        self.darkmode = False 
        self.last_ncols = None
        self.last_nrows = None
        self.last_fig = None

        # When True, plotting methods return a flat inline PNG instead of an
        # interactive Plotly figure. Interactive figures embed plotly.js in the
        # notebook (large files, esp. with many plots); static PNGs keep size
        # down. last_fig still caches the real figure, so save_png/re-styling
        # keep working. Toggle via set_static_images. static_scale = PNG
        # resolution multiplier. Requires 'kaleido' (falls back to interactive).
        self.static_images = False
        self.static_scale = 2

        self.suptitle = None
        # Optional text box pinned to the bottom of the figure (caption/footnote).
        # Like suptitle: set per-call via footer= or persist as this attribute.
        # None = current behavior (no footer, no extra bottom margin reserved).
        self.footer = None

        # Default figure size (width, height) in inches, used by the plot methods
        # whenever a call doesn't pass figsize=. Change via set_default_format.
        # Distinct from set_plot_size, which pins the inner plot area; figsize
        # sets the overall figure dimensions.
        self.figsize = (12, 8)

        # Persistent per-call plotting defaults, set via set_default_format.
        # None = unset, so the relevant plot method falls back to its own built-in
        # (resolved through _apply_default). An explicit per-call argument always
        # wins over the stored default. Cleared by set_default_format(reset=True).
        self.plot_defaults = {
            'legend': None, 'suppress_legends': None,
            'ncols': None, 'nrows': None,
            'barmode': None, 'agg': None, 'histfunc': None,
            'histnorm': None, 'alpha': None, 'boxmode': None, 'points': None,
        }

        # Plot Decorations
        self.plot_title = None
        self.x_label = None
        self.y_label = None
        
        self.display_parms = []
        self.axis_limits = {} 
        self.lines = {}       
        self.highlights = {}  

        self.parm_description_dict = {}

        # Per-variable formatting overrides (used by plot_ymult / uniplot_ymultaxis).
        # Shape: {variable_name: {attr: value}} where attr ∈ _VAR_FORMAT_KEYS.
        self.variable_formats = {}

        # User-customizable color map: the ordered list of colors assigned to
        # datasets by index, mirroring marker_map for markers. Replace it with
        # your own list (e.g. nb.color_map = ['#FF0000', '#00FF00', ...]) to
        # choose the colors new/reset datasets and integer color() lookups use.
        # Integer indexing cycles, so nb.color_map[3] works on a 2-color map.
        self.color_map = px.colors.qualitative.Plotly

        # User-customizable marker map: the ordered list of marker symbols
        # assigned to datasets by index (the marker analogue of color_map).
        # Replace it with your own list (e.g. nb.marker_map = ['o', 's', '^'])
        # to choose the markers new/reset datasets and integer marker() lookups
        # use. Integer indexing cycles, so nb.marker_map[3] works on a 2-marker map.
        self.marker_map = list(MARKER_MAP_MPL_TO_PLOTLY.keys())

        # Per-dataset style defaults (marker/markersize/linestyle/linewidth/
        # edgewidth/edge_color/alpha/fill) applied to datasets as they're loaded.
        # Change via set_default_format to restyle *future* loaded datasets; color
        # stays controlled by color_map. Marker defaults to per-index (marker_map)
        # but set_default_format can pin it to a symbol or disable it (None).
        self.default_format = dict(_DATASET_FORMAT_DEFAULTS)

        # Optional fixed inner plot-area size (px, w/h, either may be None) so
        # plots stay the same size regardless of suptitle/legend/margins. Set
        # via set_plot_size; applied in _finalize. None = size driven by figsize.
        self.plot_size = None

        self.suptitle_size = None
        self.footer_size = None
        self.legend_size = None
        self.axes_title_size = None
        self.axes_tick_size = None
        self.subplot_title_size = None
        self.colorbar_size = None
        self.hover_size = None
        self.table_header_size = None
        self.table_cell_size = None

        print("UniChart Notebook Environment Initialized.")

    # ------------------------------------------------------------------
    # Backwards compatibility
    # ------------------------------------------------------------------
    @property
    def uset(self):
        return self.sets

    @uset.setter
    def uset(self, value):
        self.sets = value

    # ------------------------------------------------------------------
    # Data Management
    # ------------------------------------------------------------------
    def _set_positions(self, set_id):
        """Cached integer row positions of a set within the combined frame.

        Valid because row identity only changes through _register_sets /
        _replace_set_rows / clear_data, which all reset the cache; column
        writes leave row positions untouched.
        """
        pos = self._set_row_pos.get(set_id)
        if pos is None:
            ids = self._combined_df[_SET_ID_COL].to_numpy()
            pos = np.flatnonzero(ids == set_id)
            self._set_row_pos[set_id] = pos
        return pos

    def _reconcile_columns(self):
        """Sync per-set column ownership with the combined frame's columns.

        Cheap fast path: pandas replaces the columns Index object whenever a
        column is added, removed, or renamed, so an identity check catches any
        untracked column-set change (e.g. a direct ``nb.df['NEW'] = ...``).
        Brand-new columns are claimed by the sets that actually hold data in
        them; removed columns are forgotten everywhere. The one thing this
        cannot see is an in-place *value* write into an existing column
        (``nb.df.loc[...] = ...``) — the API write paths track those
        themselves, and ``refresh_own_columns(rescan=True)`` heals after
        direct surgery.
        """
        cdf = self._combined_df
        if cdf.columns is self._cols_snapshot:
            return
        current = set(cdf.columns)
        added = current - self._known_cols
        removed = self._known_cols - current
        added.discard(_SET_ID_COL)
        if removed:
            for ds in self.sets:
                ds._own_cols -= removed
        for col in added:
            col_vals = cdf[col]
            if isinstance(col_vals, pd.DataFrame):   # duplicated label
                notna = col_vals.notna().any(axis=1).to_numpy()
            else:
                notna = col_vals.notna().to_numpy()
            for ds in self.sets:
                if notna[self._set_positions(ds._set_id)].any():
                    ds._own_cols.add(col)
        self._cols_snapshot = cdf.columns
        self._known_cols = current

    def _snapshot_columns(self):
        """Mark the combined frame's current columns as reconciled."""
        self._cols_snapshot = self._combined_df.columns
        self._known_cols = set(self._combined_df.columns)

    def refresh_own_columns(self, rescan=False):
        """Re-sync per-set column ownership with the combined frame.

        Ownership normally maintains itself: loading, ``ds['col'] = ...``,
        ``add_column``, ``set_column`` and df replacement all track it, and
        columns added/removed/renamed directly on ``nb.df`` are reconciled
        automatically from the data. The one blind spot is filling values
        *in place* into existing columns of the live frame (e.g.
        ``nb.df.loc[rows, 'COL'] = ...``) for a set that didn't own that
        column. Call with ``rescan=True`` after that kind of surgery: every
        column holding any data in a set's rows is claimed by that set.
        Rescan only ever adds ownership, it never revokes it.
        """
        self._reconcile_columns()
        if not rescan:
            return
        cdf = self._combined_df
        all_cols = list(cdf.columns)
        for ds in self.sets:
            pos = self._set_positions(ds._set_id)
            missing_pos = [i for i, c in enumerate(all_cols)
                           if c != _SET_ID_COL and c not in ds._own_cols]
            if not len(pos) or not missing_pos:
                continue
            sub = cdf.iloc[pos, missing_pos]
            has_data = sub.notna().any().to_numpy()
            ds._own_cols.update(np.asarray(sub.columns)[has_data].tolist())

    def _register_set(self, df, title):
        """Append a new set to the combined frame and create the façade Dataset.

        Outer-concats so missing columns are NaN-filled in either direction.
        Returns the new Dataset.
        """
        return self._register_sets([(df, title)])[0]

    def _register_sets(self, frames_and_titles):
        """Append several new sets with a single combined-frame rebuild.

        Each (df, title) pair becomes one Dataset. Concatenating once keeps a
        multi-set load O(total rows); registering one set at a time re-copies
        the whole accumulated frame per set.
        """
        if not frames_and_titles:
            return []

        # Settle any untracked column changes against the *current* frame
        # before the rebuild makes them indistinguishable from the new sets'.
        self._reconcile_columns()

        tagged_frames, metas = [], []
        for df, title in frames_and_titles:
            if _SET_ID_COL in df.columns:
                df = df.drop(columns=_SET_ID_COL)
            set_id = self._next_set_id
            self._next_set_id += 1
            tagged = df.copy()
            tagged[_SET_ID_COL] = set_id
            tagged_frames.append(tagged)
            metas.append((set_id, title, set(df.columns)))

        frames = (tagged_frames if self._combined_df.empty
                  else [self._combined_df, *tagged_frames])
        if len(frames) == 1:
            self._combined_df = frames[0].reset_index(drop=True)
        else:
            # Row-stacking sets that introduce disjoint columns leaves the result
            # with one block per column (a 122-col frame ends up 122 blocks). The
            # concat itself is silent, but the fragmented frame is slow for column
            # access and trips a "DataFrame is highly fragmented" PerformanceWarning
            # on the next per-column write (e.g. ds['x']=..., set_column). The
            # trailing .copy() is a deliberate defragmentation — it consolidates the
            # blocks; do not remove it as a redundant copy.
            self._combined_df = pd.concat(
                frames, ignore_index=True, sort=False
            ).copy()
        self._set_row_pos = {}
        self._snapshot_columns()

        created = []
        for set_id, title, own_cols in metas:
            ds = Dataset(self, set_id, index=len(self.sets), title=title,
                         own_cols=own_cols)
            self.sets.append(ds)
            created.append(ds)
        self._reapply_all_queries()
        return created

    def _reapply_all_queries(self):
        """Recompute every dataset's query mask. Call after combined-frame mutations."""
        for ds in self.sets:
            ds._apply_query()

    def _replace_set_rows(self, set_id, new_df):
        """Replace all rows belonging to set_id with new_df (re-tagged with the same id)."""
        self._reconcile_columns()
        cdf = self._combined_df
        kept = cdf.loc[cdf[_SET_ID_COL] != set_id]
        if _SET_ID_COL in new_df.columns:
            new_df = new_df.drop(columns=_SET_ID_COL)
        tagged = new_df.copy()
        tagged[_SET_ID_COL] = set_id
        if kept.empty:
            self._combined_df = tagged.reset_index(drop=True)
        else:
            # Trailing .copy() consolidates the row-stacked blocks; see the note in
            # _register_set. Without it the frame stays one-block-per-column and the
            # next per-column write warns about fragmentation.
            self._combined_df = pd.concat(
                [kept, tagged], ignore_index=True, sort=False).copy()
        self._set_row_pos = {}
        # The replacement frame defines this set's columns from scratch.
        for ds in self.sets:
            if ds._set_id == set_id:
                ds._own_cols = set(new_df.columns) - {_SET_ID_COL}
                break
        self._snapshot_columns()
        # ignore_index rebuilds every row label, which staled all query masks
        # (they are keyed to the global index). Recompute them all.
        self._reapply_all_queries()

    def load_df(self, df, title=None, set_name_column=None, set_idx_column=None, load_cols_as_vars=False, combined=False):
        """Split a DataFrame into one Dataset per unique set_idx_column value, or load it as one.

        ``df`` may be a single DataFrame or a list of DataFrames. For a list, ``combined=True``
        concatenates them into one set, while ``combined=False`` loads each DataFrame separately.
        """
        if isinstance(df, (list, tuple)):
            if combined:
                df = pd.concat(df, ignore_index=True)
            else:
                for single_df in df:
                    self.load_df(single_df, title=title, set_name_column=set_name_column,
                                 set_idx_column=set_idx_column, load_cols_as_vars=load_cols_as_vars)
                return

        df = df.copy()

        if not title:
            if set_name_column and set_name_column in df.columns:
                pass
            elif "TITLE" in df.columns:
                set_name_column = "TITLE"
            else:
                df["TITLE"] = "Dataset"
                set_name_column = "TITLE"

            if set_idx_column and set_idx_column in df.columns:
                pass
            elif "SETNUMBER" in df.columns:
                set_idx_column = "SETNUMBER"
            elif "INDEX" in df.columns:
                set_idx_column = "INDEX"
            else:
                df["SETNUMBER"] = df.index

        if set_idx_column and set_idx_column in df.columns:
            # Collect every group first and register them in one batch — one
            # combined-frame rebuild for the whole file instead of one per set.
            groups = []
            for set_index, df_subset in df.groupby(set_idx_column):
                if title:
                    final_title = title
                elif set_name_column and set_name_column in df_subset.columns:
                    final_title = str(df_subset.iloc[0][set_name_column])
                elif "TITLE" in df_subset.columns:
                    final_title = str(df_subset.iloc[0]["TITLE"])
                else:
                    final_title = f"Group {set_index}"
                groups.append((df_subset, final_title))

            for ds in self._register_sets(groups):
                print(f"Loaded Set {ds.index}: {ds.title}")
        else:
            ds = self._register_set(df, title if title else "Untitled")
            print(f"Loaded Set {ds.index}: {ds.title}")

        if load_cols_as_vars:
            names = {str(c): str(c) for c in df.columns if str(c).isidentifier()}
            globals().update(names)
            skipped = [c for c in df.columns if str(c) not in names]
            if skipped:
                print(f"Could not create variables for {len(skipped)} column(s) "
                      f"whose names are not valid identifiers: {skipped[:10]}"
                      f"{'...' if len(skipped) > 10 else ''}")

    _FILE_READERS = {
        ".csv": lambda path, kw: pd.read_csv(path, **kw),
        ".tsv": lambda path, kw: pd.read_csv(path, sep="\t", **kw),
        ".txt": lambda path, kw: pd.read_csv(path, sep="\t", **kw),
        ".xlsx": lambda path, kw: pd.read_excel(path, **kw),
        ".xls": lambda path, kw: pd.read_excel(path, **kw),
        ".json": lambda path, kw: pd.read_json(path, **kw),
        ".parquet": lambda path, kw: pd.read_parquet(path, **kw),
    }

    def load(self, source, title=None, set_name_column=None, set_idx_column=None,
             load_cols_as_vars=False, combined=False, read_kwargs=None):
        """Load datasets from DataFrames, filepaths, dicts, or numpy arrays.

        ``source`` may be a single item or a list of items; each is coerced to a DataFrame
        and loaded via load_df. Supported files: .csv, .tsv, .txt, .xlsx, .xls, .json, .parquet
        (``read_kwargs`` is passed through to the pandas reader). For a list, ``combined=True``
        merges everything into one set while ``combined=False`` loads each separately. A file's
        title defaults to its filename when no title or SETNUMBER/INDEX split column is present.
        """
        sources = list(source) if isinstance(source, (list, tuple)) else [source]
        coerced = [self._coerce_to_df(s, read_kwargs) for s in sources]

        def resolve_title(frame, default_title):
            if title is not None or set_idx_column is not None:
                return title
            if "SETNUMBER" in frame.columns or "INDEX" in frame.columns:
                return None
            return default_title

        if combined:
            frame = pd.concat([df for df, _ in coerced], ignore_index=True)
            default_title = next((dt for _, dt in coerced if dt), None)
            self.load_df(frame, title=resolve_title(frame, default_title),
                         set_name_column=set_name_column, set_idx_column=set_idx_column,
                         load_cols_as_vars=load_cols_as_vars)
            return

        for frame, default_title in coerced:
            self.load_df(frame, title=resolve_title(frame, default_title),
                         set_name_column=set_name_column, set_idx_column=set_idx_column,
                         load_cols_as_vars=load_cols_as_vars)

    def _coerce_to_df(self, source, read_kwargs=None):
        """Coerce a single source into a (DataFrame, default_title) pair. default_title is the
        filename stem for file inputs, otherwise None."""
        if isinstance(source, pd.DataFrame):
            return source.copy(), None
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"No such file: {path}")
            reader = self._FILE_READERS.get(path.suffix.lower())
            if reader is None:
                raise ValueError(
                    f"Unsupported file type '{path.suffix}' for {path}; "
                    f"supported: {', '.join(self._FILE_READERS)}")
            return reader(path, read_kwargs or {}), path.stem
        if isinstance(source, dict):
            return pd.DataFrame(source), None
        if isinstance(source, np.ndarray):
            return pd.DataFrame(source), None
        raise TypeError(
            f"load() cannot handle source of type {type(source).__name__}; "
            f"pass a DataFrame, filepath, dict, or numpy array.")

    def load_clipboard(self, **kwargs):
        """Quickly load data from system clipboard."""
        try:
            df = pd.read_clipboard(**kwargs)
            self.load_df(df, title="Clipboard Data")
        except Exception as e:
            print(f"Error reading clipboard: {e}")

    def clear_data(self):
        self.sets = []
        self._combined_df = pd.DataFrame({_SET_ID_COL: pd.Series(dtype='int64')})
        self._next_set_id = 0
        self._set_row_pos = {}
        self._snapshot_columns()
        print("All datasets cleared.")

    # ------------------------------------------------------------------
    # Combined-frame bulk operations
    # ------------------------------------------------------------------
    @property
    def df(self):
        """The combined DataFrame across all sets (with the _SET_ID column visible).

        Returned live — mutations affect plotting. Prefer `add_column` /
        `set_column` for safe writes; they reapply per-set query masks and
        track per-set column ownership. Adding or dropping columns directly
        here is detected and reconciled automatically (ownership is assigned
        from the data); filling values *in place* into existing columns is
        not — call `refresh_own_columns(rescan=True)` afterwards.
        """
        return self._combined_df

    def add_column(self, name, value):
        """Add or replace a column on the combined frame across all sets.

        `value` may be a scalar, a Series aligned to the combined frame's index,
        or a callable receiving the combined frame and returning either.
        """
        if callable(value):
            value = value(self._combined_df)
        self._reconcile_columns()
        self._combined_df[name] = value
        # An all-sets write: every set owns the column, even where the
        # assigned values happen to be NaN.
        for ds in self.sets:
            ds._own_cols.add(name)
        self._snapshot_columns()
        self._reapply_all_queries()

    def set_column(self, uset_slice, col, value):
        """Write `value` into `col` for the selected sets only.

        Adds the column (NaN-filled elsewhere) if it doesn't exist. `value` may
        be a scalar or an array-like aligned to the total number of selected rows.
        """
        targets = self._get_uset_slice(uset_slice)
        if not targets:
            return
        target_ids = {ds._set_id for ds in targets}
        cdf = self._combined_df
        mask = cdf[_SET_ID_COL].isin(target_ids)
        if col not in cdf.columns:
            cdf[col] = pd.NA
        if hasattr(value, '__len__') and not isinstance(value, str):
            n = int(mask.sum())
            if len(value) != n:
                raise ValueError(
                    f"Length mismatch assigning '{col}': {len(value)} values for {n} rows.")
            cdf.loc[mask, col] = list(value)
        else:
            cdf.loc[mask, col] = value
        # The targeted sets claim the column — including the case of filling
        # NaNs into a column some other set introduced.
        for ds in targets:
            ds._own_cols.add(col)
        self._reapply_all_queries()

    def set_color_palette(self, palette, uset_slice='all'):
        """Recolor the selected sets using a Plotly qualitative palette name or color list."""
        targets = self._get_uset_slice(uset_slice)
        if not targets:
            return
        if isinstance(palette, str):
            try:
                colors = getattr(px.colors.qualitative, palette)
            except AttributeError:
                try:
                    colors = pcolors.sample_colorscale(palette, len(targets))
                except Exception as e:
                    raise ValueError(f"Unknown palette: {palette!r} ({e})")
        else:
            colors = list(palette)
        for i, ds in enumerate(targets):
            ds.color = colors[i % len(colors)]

    def set_title(self, uset_slice, title):
        """
        Update the title of the specified dataset(s).
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.title = str(title)
            ds.title_format = f"{ds.title} {ds.index}"

    # ------------------------------------------------------------------
    # Selection & Filtering
    # ------------------------------------------------------------------
    def _get_uset_slice(self, uset_slice):
        """Normalize a selector into a list of Dataset objects.

        Accepts:
            None | 'all'    -> all datasets
            int             -> dataset at that index (negative indices count
                               from the end, so -1 is the last dataset)
            Dataset         -> wrapped in a list
            list            -> mixed list of any of the above
        Unknown inputs print a warning and return [].

        Note: dataset titles are deliberately *not* accepted as selectors. A
        bare string is reserved for variable/parameter targeting in the
        formatting setters (see :meth:`_var_targets`), so titles would be
        ambiguous here.
        """
        if uset_slice is None or uset_slice == 'all':
            return list(self.sets)

        if isinstance(uset_slice, Dataset):
            return [uset_slice]

        if isinstance(uset_slice, int) and not isinstance(uset_slice, bool):
            if -len(self.sets) <= uset_slice < len(self.sets):
                return [self.sets[uset_slice]]
            return []

        if isinstance(uset_slice, (list, tuple, set)):
            result = []
            seen_ids = set()
            for item in uset_slice:
                for d in self._get_uset_slice(item):
                    if id(d) not in seen_ids:
                        result.append(d)
                        seen_ids.add(id(d))
            return result

        print(f"Warning: don't know how to interpret {uset_slice!r} as a dataset selector.")
        return []

    @staticmethod
    def _var_targets(target):
        """Autodetect whether a formatting-setter target names variable(s).

        Dataset selectors are ints, ``'all'``/None, ``Dataset`` objects, or
        lists thereof. Since titles are no longer valid selectors, a bare
        string (other than ``'all'``) — or a list/tuple of such strings —
        unambiguously denotes variable/parameter name(s).

        Returns the list of variable names when ``target`` names variables,
        otherwise ``None`` (meaning: treat as a dataset selector).
        """
        if isinstance(target, str) and target != 'all':
            return [target]
        if isinstance(target, (list, tuple)) and target and all(
                isinstance(t, str) and t != 'all' for t in target):
            return list(target)
        return None

    @property
    def color_map(self):
        """The dataset color sequence — a :class:`CyclicList`, so integer
        indexing wraps (``color_map[3]`` works on a 2-color map). Assigning any
        list (or palette sequence) coerces it to a CyclicList automatically.
        """
        return self._color_map

    @color_map.setter
    def color_map(self, value):
        self._color_map = value if isinstance(value, CyclicList) else CyclicList(value)

    def _color_at(self, index):
        """Color assigned to a 0-based ``index``, cycling ``self.color_map``.

        Parallels :meth:`_marker_at` for colors. Falls back to the default
        Plotly palette if ``color_map`` has been set to an empty list.
        """
        cmap = self.color_map or CyclicList(px.colors.qualitative.Plotly)
        return cmap[index]

    @property
    def marker_map(self):
        """The dataset marker sequence — a :class:`CyclicList`, so integer
        indexing wraps (``marker_map[3]`` works on a 2-marker map). Assigning any
        list coerces it to a CyclicList automatically.
        """
        return self._marker_map

    @marker_map.setter
    def marker_map(self, value):
        self._marker_map = value if isinstance(value, CyclicList) else CyclicList(value)

    def _marker_at(self, index):
        """Marker assigned to a 0-based ``index``, cycling ``self.marker_map``.

        Falls back to the default marker set if ``marker_map`` has been set to
        an empty list.
        """
        mmap = self.marker_map or CyclicList(MARKER_MAP_MPL_TO_PLOTLY.keys())
        return mmap[index]

    def select(self, uset_slice=None):
        """Select the specified dataset(s)."""
        for ds in self.sets: ds.select = False
        for ds in self._get_uset_slice(uset_slice):
            ds.select = True

    def selected(self):
        """Get the currently selected datasets."""
        return [ds for ds in self.sets if ds.select]

    def omit(self, uset_slice=None):
        for ds in self._get_uset_slice(uset_slice):
            ds.select = False

    def restore(self, uset_slice=None):
        targets = self.sets if uset_slice == "all" else self._get_uset_slice(uset_slice)
        for ds in targets:
            ds.select = True

    def query(self, uset_slice=None, query_str=None):
        targets = list(self._get_uset_slice(uset_slice))
        if not targets:
            return

        if not query_str:
            for ds in targets:
                ds._query = None
                ds._query_mask = None
            return

        cdf = self._combined_df
        target_ids = {ds._set_id for ds in targets}
        set_mask = cdf[_SET_ID_COL].isin(target_ids)
        subset = cdf.loc[set_mask]
        try:
            filtered = subset.query(query_str)
        except Exception as e:
            raise ValueError(f"Query error: {e}")

        kept_by_set = filtered.groupby(_SET_ID_COL, sort=False).groups
        for ds in targets:
            ds._query = query_str
            keep_idx = kept_by_set.get(ds._set_id)
            if keep_idx is None or len(keep_idx) == 0:
                print(f"No data in set {ds.index} after query: {query_str}. Turning Set Off...")
                ds._select = False
                ds._query_mask = None
                continue
            mask = pd.Series(False, index=cdf.index)
            mask.loc[keep_idx] = True
            ds._query_mask = mask

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def color(self, uset_slice=None, color_val=None):
        """
        Set the primary color for the target dataset(s) or variable(s).

        The target is autodetected: an int / list of ints / ``Dataset`` / 'all'
        / None selects dataset(s); a variable/parameter name (or list of names)
        applies a per-variable override via :meth:`var_format`, which takes
        precedence over dataset formatting at plot time.

        Args:
            uset_slice (int, list, 'all', or str): Dataset selector, or
                variable name(s) to target.
            color_val (str or int): A color spec, or an integer set index.
                - str: a standard color name ('red'), hex code ('#FF5733'),
                or RGB/RGBA string ('rgb(255, 0, 0)'). Pass ``'reset'`` when
                targeting a variable to clear its color override.
                - int: resolves to the color that set index would be assigned
                from ``color_map`` at load time, independent of any recoloring
                that set has since received. E.g. color(5, 2) gives set 5 the
                ``color_map`` color of set 2.

        Examples
        --------
        nb.color('all', 'red')              # every dataset red
        nb.color(0, 'blue')                 # dataset 0 blue
        nb.color('Temperature', 'blue')     # the Temperature variable blue
        nb.color('Temperature', 'reset')    # clear that variable override
        """
        if isinstance(color_val, int) and not isinstance(color_val, bool):
            color_val = self._color_at(color_val)
        variables = self._var_targets(uset_slice)
        if variables is not None:
            for v in variables:
                self.var_format(v, color=color_val)
            return
        for ds in self._get_uset_slice(uset_slice):
            ds.color = color_val

    def marker(self, uset_slice=None, marker_val=None):
        """
        Set the marker style for the target dataset(s) or variable(s).

        The target is autodetected: an int / list of ints / ``Dataset`` / 'all'
        / None selects dataset(s); a variable/parameter name (or list of names)
        applies a per-variable override via :meth:`var_format`, which takes
        precedence over dataset formatting at plot time.

        Args:
            uset_slice (int, list, 'all', or str): Dataset selector, or
                variable name(s) to target.
            marker_val (str or int): A marker spec, or an integer set index.
                - str: Matplotlib-style marker ('o', 's', '^', 'D', '.')
                or Plotly-style marker ('circle', 'square'). Pass ``'reset'``
                when targeting a variable to clear its marker override.
                - int: resolves to the marker that set index would be assigned
                from ``marker_map`` at load time, independent of any later
                restyling. E.g. marker(5, 2) gives set 5 the ``marker_map``
                marker of set 2.

        Examples
        --------
        nb.marker('all', 's')             # every dataset uses squares
        nb.marker('Pressure', '^')        # the Pressure variable uses triangles
        """
        if isinstance(marker_val, int) and not isinstance(marker_val, bool):
            marker_val = self._marker_at(marker_val)
        variables = self._var_targets(uset_slice)
        if variables is not None:
            for v in variables:
                self.var_format(v, marker=marker_val)
            return
        for ds in self._get_uset_slice(uset_slice):
            ds.marker = marker_val

    def linestyle(self, uset_slice=None, style_val=None):
        """
        Set the line style for the target dataset(s) or variable(s).

        The target is autodetected: an int / list of ints / ``Dataset`` / 'all'
        / None selects dataset(s); a variable/parameter name (or list of names)
        applies a per-variable override via :meth:`var_format`, which takes
        precedence over dataset formatting at plot time.

        Args:
            uset_slice (int, list, 'all', or str): Dataset selector, or
                variable name(s) to target.
            style_val (str): Matplotlib-style string ('-', '--', '-.', ':')
                             or Plotly string ('solid', 'dash', 'dashdot', 'dot').
                             Pass ``'reset'`` when targeting a variable to clear
                             its linestyle override.

        Examples
        --------
        nb.linestyle('all', '--')             # dash every dataset
        nb.linestyle('Temperature', ':')      # dot the Temperature variable
        """
        variables = self._var_targets(uset_slice)
        if variables is not None:
            for v in variables:
                self.var_format(v, linestyle=style_val)
            return
        for ds in self._get_uset_slice(uset_slice):
            ds.linestyle = style_val

    def markersize(self, uset_slice=None, size_val=None):
        """
        Set the marker size for the target dataset(s) or variable(s).

        The target is autodetected: a dataset selector (int / list / 'all' /
        None / ``Dataset``) sets it per dataset; a variable/parameter name (or
        list of names) applies a per-variable override via :meth:`var_format`.
        Pass ``'reset'`` when targeting a variable to clear the override.
        """
        variables = self._var_targets(uset_slice)
        if variables is not None:
            for v in variables:
                self.var_format(v, markersize=size_val)
            return
        for ds in self._get_uset_slice(uset_slice):
            ds.markersize = size_val

    def linewidth(self, uset_slice=None, width_val=None):
        """
        Set the line thickness for the target dataset(s) or variable(s).

        The target is autodetected: a dataset selector (int / list / 'all' /
        None / ``Dataset``) sets it per dataset; a variable/parameter name (or
        list of names) applies a per-variable override via :meth:`var_format`.
        Pass ``'reset'`` when targeting a variable to clear the override.
        """
        variables = self._var_targets(uset_slice)
        if variables is not None:
            for v in variables:
                self.var_format(v, linewidth=width_val)
            return
        for ds in self._get_uset_slice(uset_slice):
            ds.linewidth = width_val

    def edgewidth(self, uset_slice, width_val):
        """
        Set the marker edge width (outline thickness) for the specified dataset(s).

        Args:
            uset_slice (int, list, or 'all'): The dataset index or indices to modify.
            width_val (int or float): The thickness of the marker edge. Must be >= 0.
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.edgewidth = width_val

    def fill(self, uset_slice, fill_val):
        """
        Set the fill state (whether markers are solid or hollow) for the specified dataset(s).

        Args:
            uset_slice (int, list, or 'all'): The dataset index or indices to modify.
            fill_val (bool, int, or str): True/False (or 'on'/'off', '1'/'0', 't'/'f') 
                                          to enable or disable marker fill.
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.fill = fill_val

    def hue(self, uset_slice, col_name):
        """
        Map a dataframe column to the color scale for the specified dataset(s).
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.hue = col_name

    def hue_palette(self, uset_slice, hue_palette):
        """
        Set the color scale/palette used when `hue` is mapped to a variable.
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.hue_palette = hue_palette

    def alpha(self, uset_slice=None, alpha_val=None):
        """
        Set the opacity (alpha) for the target dataset(s) or variable(s).

        The target is autodetected: a dataset selector (int / list / 'all' /
        None / ``Dataset``) sets it per dataset; a variable/parameter name (or
        list of names) applies a per-variable override via :meth:`var_format`.
        Pass ``'reset'`` when targeting a variable to clear the override.
        """
        variables = self._var_targets(uset_slice)
        if variables is not None:
            for v in variables:
                self.var_format(v, alpha=alpha_val)
            return
        for ds in self._get_uset_slice(uset_slice):
            ds.alpha = alpha_val

    def plot_type(self, uset_slice, type_val):
        for ds in self._get_uset_slice(uset_slice):
            ds.plot_type = type_val

    def reg_order(self, uset_slice, order):
        """
        Set a regression/trendline for the specified dataset(s).
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.reg_order = order

    # ------------------------------------------------------------------
    # Variable-level formatting overrides
    # ------------------------------------------------------------------
    def var_format(self, variable, color=None, marker=None, linestyle=None,
                   markersize=None, linewidth=None, alpha=None, style=None):
        """
        Set persistent per-variable formatting overrides.

        Variable formatting takes precedence over Dataset formatting on a
        per-attribute basis — anything you don't set still falls back to
        the Dataset's value at plot time.

        Pass the string 'reset' as the value to clear a single attribute.

        `style` only affects bar-plot overlay columns (the `markers=` argument
        of `nb.bar`): 'marker' (default symbol overlay), 'tick' (horizontal
        dash at the value), or 'whisker' (dash plus a stem down/up to the top
        of the bar). For `style='whisker'`, `linestyle` sets the stem's dash
        pattern ('-', '--', '-.', ':'; 'None' hides the stem) and `linewidth`
        its thickness.

        Examples
        --------
        nb.var_format('Temperature', linestyle='--')         # all Temp lines dashed
        nb.var_format('Pressure', color='blue', marker='s')  # Pressure forced blue squares
        nb.var_format('Pressure', color='reset')             # remove just the color override
        nb.var_format('EGT_LIMIT', style='whisker', color='red')
        """
        if style is not None and style != 'reset' and style not in _OVERLAY_STYLES:
            raise ValueError(f"style must be one of {_OVERLAY_STYLES}, got {style!r}")
        fmt = self.variable_formats.setdefault(variable, {})
        pairs = {'color': color, 'marker': marker, 'linestyle': linestyle,
                 'markersize': markersize, 'linewidth': linewidth, 'alpha': alpha,
                 'style': style}
        for k, v in pairs.items():
            if v is None:
                continue
            if v == 'reset':
                fmt.pop(k, None)
            else:
                fmt[k] = v
        if not fmt:
            del self.variable_formats[variable]
        return self.variable_formats.get(variable, {})

    def clear_var_format(self, variable=None):
        """Clear variable formatting. Pass None (or no arg) to clear everything."""
        if variable is None:
            self.variable_formats.clear()
        else:
            self.variable_formats.pop(variable, None)

    def list_var_formats(self):
        """Pretty-print current variable-level formatting."""
        if not self.variable_formats:
            print("No variable-level formatting set.")
            return
        print("Variable-level formatting (overrides dataset attributes):")
        for var, fmt in self.variable_formats.items():
            items = ", ".join(f"{k}={v!r}" for k, v in fmt.items())
            print(f"  {var}: {items}")

    def reset_format(self, uset_slice=None, sets=True, vars=True,
                     lines=True, highlights=True, scale=True, fonts=True):
        """
        Reset formatting state back to defaults.

        Parameters
        ----------
        uset_slice : int | list | 'all' | None
            Which datasets to reset. None/'all' resets every dataset.
            Ignored when `sets=False`.
        sets : bool
            Reset per-dataset visual attributes (color, marker, linestyle,
            markersize, linewidth, edgewidth, alpha, hue, hue_palette,
            reg_order) back to their Dataset.__init__ defaults.
        vars : bool
            Clear all variable-level formatting overrides (variable_formats).
        lines : bool
            Clear all stored reference lines.
        highlights : bool
            Clear all stored highlight regions.
        scale : bool
            Clear all stored axis limits (axis_limits).
        fonts : bool
            Reset all font-size overrides to None (use Plotly defaults).

        Examples
        --------
        nb.reset_format()                  # reset everything
        nb.reset_format(sets=False)        # keep per-dataset formatting, clear rest
        nb.reset_format(uset_slice=[0,1])  # only reset datasets 0 and 1
        nb.reset_format(lines=True, highlights=True, sets=False, vars=False,
                        scale=False, fonts=False)  # only clear decorations
        """
        if sets:
            targets = (self.sets if uset_slice is None
                       else self._get_uset_slice(uset_slice))
            fmt = getattr(self, 'default_format', _DATASET_FORMAT_DEFAULTS)
            default_marker = fmt.get('marker', _MARKER_BY_INDEX)
            for ds in targets:
                ds._color      = self._color_at(ds.index)
                ds._marker     = (self._marker_at(ds.index)
                                  if default_marker is _MARKER_BY_INDEX else default_marker)
                ds._linestyle  = fmt.get('linestyle', None)
                ds.markersize  = fmt.get('markersize', 10)
                ds.linewidth   = fmt.get('linewidth', 2)
                ds.edgewidth   = fmt.get('edgewidth', 1)
                ds.alpha       = fmt.get('alpha', 1)
                ds._edge_color = fmt.get('edge_color', 'black')
                ds._fill       = fmt.get('fill', True)
                ds.hue         = ""
                ds.hue_palette = "Jet"
                ds.hue_order   = None
                ds.reg_order   = None

        if vars:
            self.variable_formats.clear()

        if lines:
            self.lines.clear()

        if highlights:
            self.highlights.clear()

        if scale:
            self.axis_limits.clear()

        if fonts:
            for attr in ('suptitle_size', 'footer_size', 'legend_size', 'axes_title_size',
                         'axes_tick_size', 'subplot_title_size',
                         'colorbar_size', 'hover_size'):
                setattr(self, attr, None)

        parts = []
        if sets:      parts.append("dataset formatting")
        if vars:      parts.append("variable formats")
        if lines:     parts.append("lines")
        if highlights: parts.append("highlights")
        if scale:     parts.append("axis limits")
        if fonts:     parts.append("font sizes")
        print(f"Reset: {', '.join(parts) if parts else 'nothing'}.")

    def _apply_default(self, key, value, builtin):
        """Resolve a per-call plotting arg: an explicit ``value`` (not None) wins;
        else the stored ``self.plot_defaults[key]`` if set; else the method's own
        ``builtin``. Lets each method keep its native default while sharing one
        notebook-level override (e.g. barmode's built-in differs per method)."""
        if value is not None:
            return value
        stored = self.plot_defaults.get(key)
        return stored if stored is not None else builtin

    def _resolve_grid(self, ncols, nrows):
        """Apply the standing ncols/nrows default only when neither was passed,
        resolving them as a pair so a one-off ``ncols=`` doesn't pull the default
        ``nrows``. Returns ``(ncols, nrows)``."""
        if ncols is None and nrows is None:
            dn, dr = self.plot_defaults.get('ncols'), self.plot_defaults.get('nrows')
            if dn is not None or dr is not None:
                return dn, dr
        return ncols, nrows

    def set_default_format(self, markersize=None, linestyle=None, linewidth=None,
                           edgewidth=None, edge_color=None, alpha=None, fill=None,
                           marker=_UNSET, figsize=None, legend=None,
                           suppress_legends=None, ncols=None, nrows=None,
                           barmode=None, agg=None, histfunc=None, histnorm=None,
                           boxmode=None, points=None, reset=False):
        """Set notebook-wide defaults for styling and for the plot methods.

        Two kinds of default live here. **Per-dataset styles** (markersize,
        linestyle, linewidth, edgewidth, edge_color, alpha, fill, marker) are the
        markersize/linewidth analogue of ``color_map``/``marker_map``, applied to
        *future* loaded datasets; already-loaded sets keep their styling until
        ``reset_format()`` re-applies the new defaults. **Figure / per-call
        defaults** (figsize, legend, suppress_legends, ncols, nrows, barmode, agg,
        histfunc, histnorm, points, boxmode) seed the matching argument of the
        plot methods whenever a call doesn't pass its own value; an explicit
        per-call argument always wins. Only the values you pass change; others
        persist. Color remains controlled by ``color_map``.

        ``reset=True`` restores *all* of the above — per-dataset styles, figsize,
        and the per-call defaults — to their built-ins, and ignores other args.

        Parameters
        ----------
        markersize, linewidth, edgewidth : float (>= 0)
        alpha : float in [0, 1]
            Default opacity for per-dataset styles *and* the histogram bar
            opacity default (the latter overridable per-call via ``histogram(alpha=)``).
        linestyle : matplotlib/Plotly dash name (e.g. '--', 'dash') or None
        edge_color : color string (named, hex, or rgb)
        fill : bool (or truthy/falsy string) — filled vs. hollow markers
        marker : marker symbol, None, or 'map'
            A marker symbol pins every future dataset to that symbol; ``None``
            turns markers off (a line-only plot shows just lines; with neither
            line nor marker the trace draws nothing); ``'map'`` restores the
            default per-index assignment from ``marker_map``.
        figsize : (width, height) tuple of positive numbers (inches)
            Default figure size for all plot methods. Distinct from
            ``set_plot_size``, which pins the inner plot area.
        legend : 'above' | 'right' | 'off'
            Default legend placement for ``plot`` / ``plot_ymult`` (built-in 'above').
        suppress_legends : bool
            Default for all plot methods (built-in False).
        ncols, nrows : positive int
            Default subplot grid. Takes precedence over the sticky "remember the
            last grid" memory in ``plot``, but an explicit per-call ncols/nrows
            still wins. Resolved as a pair: setting one leaves the other auto.
        barmode : 'group' | 'stack' | 'overlay' | 'relative'
            Default bar mode for ``bar`` (built-in 'group') and ``histogram``
            (built-in 'overlay'). Validated against the union of both; a value
            valid for only one method errors when the other method runs.
        agg : aggregation name for ``bar`` (built-in 'mean').
        histfunc, histnorm : for ``histogram`` (built-ins 'sum', '').
        boxmode, points : for ``box`` (built-ins 'group', 'outliers').

        Examples
        --------
        nb.set_default_format(markersize=6, linestyle='--', linewidth=1)
        nb.set_default_format(marker=None)   # turn markers off for future sets
        nb.set_default_format(figsize=(10, 6), legend='right', ncols=2)
        nb.set_default_format(reset=True)    # clear styles, figsize, and defaults
        """
        if reset:
            self.default_format = dict(_DATASET_FORMAT_DEFAULTS)
            self.figsize = (12, 8)
            self.plot_defaults = {k: None for k in self.plot_defaults}
            print("Default format, figsize, and plot defaults reset to built-ins.")
            return

        # Validate everything into locals first; commit only at the end so a bad
        # arg can't leave the notebook in a half-updated state.
        new_figsize = None
        if figsize is not None:
            if (not isinstance(figsize, (tuple, list)) or len(figsize) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or v <= 0 for v in figsize)):
                raise ValueError(
                    f"figsize must be a (width, height) tuple of positive "
                    f"numbers, got {figsize!r}")
            new_figsize = tuple(figsize)

        def _num(name, val, lo=0.0, hi=None):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be numeric, got {type(val).__name__}")
            if val < lo or (hi is not None and val > hi):
                rng = f">= {lo}" if hi is None else f"in [{lo}, {hi}]"
                raise ValueError(f"{name} must be {rng}, got {val}")
            return val

        updates = {}
        if marker is not _UNSET:
            if isinstance(marker, str) and marker.lower() == 'map':
                updates['marker'] = _MARKER_BY_INDEX
            elif validate_marker(marker):
                updates['marker'] = marker
            else:
                valid = ', '.join(sorted(map(str, MARKER_MAP_MPL_TO_PLOTLY)))
                raise ValueError(
                    f"Invalid marker {marker!r}. Use None (off), 'map' "
                    f"(per-index), or one of: {valid}")
        if markersize is not None: updates['markersize'] = _num('markersize', markersize)
        if linewidth  is not None: updates['linewidth']  = _num('linewidth', linewidth)
        if edgewidth  is not None: updates['edgewidth']  = _num('edgewidth', edgewidth)
        if alpha      is not None: updates['alpha']      = _num('alpha', alpha, 0.0, 1.0)
        if linestyle  is not None:
            if not (validate_linestyle(linestyle)
                    or linestyle in LINESTYLE_MAP_MPL_TO_PLOTLY.values()):
                valid = ', '.join(sorted(map(str, LINESTYLE_MAP_MPL_TO_PLOTLY)))
                raise ValueError(f"Invalid linestyle '{linestyle}'. Valid: {valid}")
            updates['linestyle'] = linestyle
        if edge_color is not None:
            if not validate_color(edge_color):
                raise ValueError(f"edge_color must be a color string, got {edge_color!r}")
            updates['edge_color'] = edge_color
        if fill is not None:
            s = str(fill).lower()
            if s in ('true', '1', 't', 'on'):
                updates['fill'] = True
            elif s in ('false', '0', 'f', 'off'):
                updates['fill'] = False
            else:
                raise ValueError(f"Invalid value for fill: {fill}")

        # ---- Figure / per-call plotting defaults ---------------------------
        pd_updates = {}
        if 'alpha' in updates:
            pd_updates['alpha'] = updates['alpha']  # also seeds histogram opacity
        if legend is not None:
            if legend not in ('above', 'right', 'off'):
                raise ValueError(
                    f"legend must be 'above', 'right', or 'off', got {legend!r}")
            pd_updates['legend'] = legend
        if suppress_legends is not None:
            if not isinstance(suppress_legends, bool):
                raise TypeError("suppress_legends must be bool, got "
                                f"{type(suppress_legends).__name__}")
            pd_updates['suppress_legends'] = suppress_legends
        for _name, _val in (('ncols', ncols), ('nrows', nrows)):
            if _val is not None:
                if isinstance(_val, bool) or not isinstance(_val, int) or _val < 1:
                    raise ValueError(f"{_name} must be a positive integer, got {_val!r}")
                pd_updates[_name] = _val
        if barmode is not None:
            valid = ('group', 'stack', 'overlay', 'relative')
            if barmode not in valid:
                raise ValueError(f"barmode must be one of {valid}, got {barmode!r}")
            pd_updates['barmode'] = barmode
        # Chart-mode defaults validated leniently — Plotly/pandas reject bad values
        # at draw time, so we don't track their allowed-value lists here.
        if agg      is not None: pd_updates['agg']      = agg
        if histfunc is not None: pd_updates['histfunc'] = histfunc
        if histnorm is not None: pd_updates['histnorm'] = histnorm
        if boxmode  is not None: pd_updates['boxmode']  = boxmode
        if points   is not None: pd_updates['points']   = points

        # All validation passed — commit.
        if new_figsize is not None:
            self.figsize = new_figsize
        self.default_format.update(updates)
        self.plot_defaults.update(pd_updates)

    def reg_info(self, uset_slice=None):
        """Print the regression type, equation, and fit stats (R², RMSE, MAE) for each dataset."""
        KIND_LABELS = {
            'poly':   lambda p: "Linear (degree 1)" if p == 1 else f"Polynomial (degree {p})",
            'log':    lambda p: "Logarithmic",
            'exp':    lambda p: "Exponential",
            'power':  lambda p: "Power law",
            'lowess': lambda p: f"LOWESS (frac={p})",
            'spline': lambda p: "Cubic spline",
            'ma':     lambda p: f"Moving average (window={p})" if p else "Moving average",
        }

        def _fit_regression(kind, param, ds, x_col, y_col, label):
            formula = label if kind not in ('poly', 'log', 'exp', 'power') else None
            if kind is None or x_col is None or y_col is None:
                return formula, None
            if x_col not in ds.columns or y_col not in ds.columns:
                return formula, None
            df = ds.cols([x_col, y_col])
            try:
                df_c = df.dropna(subset=[x_col, y_col]).sort_values(by=x_col)
                x = df_c[x_col].to_numpy(dtype=float)
                y = df_c[y_col].to_numpy(dtype=float)
                if len(x) < 2:
                    return formula, None

                y_pred = None

                def _term(c, power, first):
                    exp_str = {0: '', 1: 'x', 2: 'x²', 3: 'x³', 4: 'x⁴', 5: 'x⁵'}.get(power, f'x^{power}')
                    if first:
                        return f"{c:.4g}{exp_str}"
                    return (f"+ {c:.4g}" if c >= 0 else f"- {abs(c):.4g}") + exp_str

                if kind == 'poly':
                    order = int(param) if param else 1
                    if len(x) < order + 1:
                        return formula, None
                    p = np.poly1d(np.polyfit(x, y, order))
                    y_pred = p(x)
                    parts = [_term(c, order - i, i == 0) for i, c in enumerate(p.coeffs)]
                    formula = "y = " + " ".join(parts)

                elif kind == 'log':
                    mask = x > 0
                    if mask.sum() < 2:
                        return formula, None
                    a, b = np.polyfit(np.log(x[mask]), y[mask], 1)
                    y_pred = np.where(x > 0, a * np.log(np.where(x > 0, x, 1)) + b, np.nan)
                    b_part = f"+ {b:.4g}" if b >= 0 else f"- {abs(b):.4g}"
                    formula = f"y = {a:.4g}·ln(x) {b_part}"

                elif kind == 'exp':
                    mask = y > 0
                    if mask.sum() < 2:
                        return formula, None
                    b, log_a = np.polyfit(x[mask], np.log(y[mask]), 1)
                    A = np.exp(log_a)
                    y_pred = A * np.exp(b * x)
                    formula = f"y = {A:.4g}·e^({b:.4g}x)"

                elif kind == 'power':
                    mask = (x > 0) & (y > 0)
                    if mask.sum() < 2:
                        return formula, None
                    b, log_a = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
                    A = np.exp(log_a)
                    y_pred = np.where(x > 0, A * np.power(np.where(x > 0, x, 1), b), np.nan)
                    formula = f"y = {A:.4g}·x^{b:.4g}"

                elif kind == 'lowess':
                    try:
                        from statsmodels.nonparametric.smoothers_lowess import lowess
                        frac = float(param) if param is not None else 0.3
                        y_pred = lowess(y, x, frac=frac, return_sorted=True)[:, 1]
                    except ImportError:
                        return formula, None

                elif kind == 'spline':
                    from scipy.interpolate import UnivariateSpline
                    k = max(1, min(5, int(param) if param else 3))
                    ux, uidx = np.unique(x, return_index=True)
                    if len(ux) < k + 1:
                        return formula, None
                    y_pred = UnivariateSpline(ux, y[uidx], k=k)(x)

                elif kind == 'ma':
                    window = int(param) if param else max(3, len(x) // 20)
                    window = max(2, min(window, len(x)))
                    y_pred = pd.Series(y).rolling(window=window, center=True, min_periods=1).mean().to_numpy()

                if y_pred is not None:
                    valid = ~(np.isnan(y) | np.isnan(y_pred))
                    y_v, yp_v = y[valid], y_pred[valid]
                    if len(y_v) >= 2:
                        ss_res = np.sum((y_v - yp_v) ** 2)
                        ss_tot = np.sum((y_v - np.mean(y_v)) ** 2)
                        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None
                        return formula, {
                            'n':    int(valid.sum()),
                            'r2':   r2,
                            'rmse': float(np.sqrt(np.mean((y_v - yp_v) ** 2))),
                            'mae':  float(np.mean(np.abs(y_v - yp_v))),
                        }

            except Exception:
                pass

            return formula, None

        lx, ly = self.last_x, self.last_y
        x_col = lx[0] if isinstance(lx, list) else lx
        y_col = ly[0] if isinstance(ly, list) else ly

        result = {}
        for ds in (self.sets if uset_slice is None or uset_slice == 'all'
                   else self._get_uset_slice(uset_slice)):
            raw = ds.reg_order
            kind, param = _parse_reg_spec(raw)
            label = KIND_LABELS.get(kind, lambda p: str(kind))(param) if kind is not None else None
            formula, stats = _fit_regression(kind, param, ds, x_col, y_col, label)
            result[ds.index] = {'raw': raw, 'kind': kind, 'param': param, 'label': label, 'formula': formula, 'stats': stats}

        any_reg = any(v['kind'] is not None for v in result.values())
        xy_header = f"Regression info for y='{y_col}' vs x='{x_col}':" if (x_col and y_col) \
            else "Regression info (no prior x/y columns set):"
        print(xy_header)
        for idx, info in result.items():
            ds = self.sets[idx]
            reg_str = info['label'] or "None"
            formula_str = f"  →  {info['formula']}" if info['formula'] else ""
            print(f"Set {idx} ({ds.title}) [y='{y_col}' vs x='{x_col}']: {reg_str}{formula_str}")
            if info['stats']:
                s = info['stats']
                r2_str = f"R²={s['r2']:.4f}" if s['r2'] is not None else "R²=N/A"
                print(f"   {r2_str}   RMSE={s['rmse']:.4g}   MAE={s['mae']:.4g}   n={s['n']}")

        if not any_reg:
            print("No regression functions are currently set.")

        return result

    def set_display_parms(self, uset_slice, parms):
        """
        Update the display parameters (columns shown on hover) for the specified dataset(s).
        """
        if not isinstance(parms, list):
            parms = [parms]
            
        for ds in self._get_uset_slice(uset_slice):
            ds.display_parms = parms

    def toggle_darkmode(self, state=None):
        """
        Toggle between dark and light mode for plots.
        state: bool (optional) - Force specific state (True=Dark, False=Light)
        """
        if state is not None:
            self.darkmode = bool(state)
        else:
            self.darkmode = not self.darkmode
            
        mode = "Dark" if self.darkmode else "Light"
        print(f"Plot theme set to: {mode} Mode")
    
    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def delta(self, base_idx, study_indices, align_on=None, delta_parms=None,
              passed_parms=None, keep_parms=None,
              direction='nearest', tolerance=None,
              x_ins=None, interp='both', kind=None, name_as_study=False,
              name_by='index'):
        """Compute deltas (absolute and %) between each study dataset and the base.

        The resulting delta dataset is anchored on the STUDY dataset's rows: every
        column of the study set is carried through under its original name. On top
        of that, for each delta parameter ``P`` the result gains:

            DL_<P>      study value minus base value
            DLPCT_<P>   100 * (study - base) / base   (NaN where base == 0)

        Base-side values are surfaced only under a standardized ``<name>_BASE``
        name, via two complementary controls — nothing from the base set is
        added under any other name:

            keep_parms    keep the raw base value of selected *delta* parameters
                          (a subset of delta_parms, or True / 'all' for all of them)
            passed_parms  pass through additional base columns for context

        The delta set inherits the study set's colour and marker, so it reads as a
        visual continuation of the study series it was derived from. Pass
        ``name_as_study=True`` to also title it after the study set.

        Parameters
        ----------
        base_idx : int
            Index of the baseline (reference) dataset.
        study_indices : int | list | 'all'
            Dataset(s) to compare against the base. The base itself is always skipped.
        align_on : str | None
            Column to align on (nearest-match merge). Defaults to last_x.
        delta_parms : str | list | None
            Columns to compute deltas for. Defaults to last_y.
        passed_parms : str | list | None
            Extra BASE column(s) to carry into the result as ``<name>_BASE``.
            Intended for context columns that are not themselves being deltaed.
        keep_parms : str | list | bool | None
            Which delta parameters' raw BASE values to keep, as ``<name>_BASE``.
            Pass a subset of delta_parms, or True / 'all' to keep every one.
            None / False keeps no base values (only the deltas and study columns).
        direction : 'nearest' | 'forward' | 'backward'
            Passed to merge_asof — controls which base row matches each study row.
            Only used in the default (non-``x_ins``) row-matching mode.
        tolerance : numeric | None
            Maximum allowed distance between matched align_on values. Unmatched rows get NaN.
            Only used in the default (non-``x_ins``) row-matching mode.

        Interpolation mode (``x_ins``)
        ------------------------------
        By default the result is anchored on the study dataset's rows, with each
        base row matched by a nearest/forward/backward ``merge_asof``. Pass
        ``x_ins`` (a scalar or list-like of ``align_on`` values) to instead place
        the result rows at exactly those ``align_on`` values, reading the base
        and/or study values off an interpolated curve at each point — mirroring
        :meth:`table`'s interpolation.

        x_ins : scalar | list | None
            ``align_on`` values that define the rows of the new dataset. When
            given, ``align_on`` must be numeric. ``None`` (default) keeps the
            original ``merge_asof`` row-matching behaviour.
        interp : 'base' | 'study' | 'both'
            Which side(s) to interpolate onto ``x_ins`` (default ``'both'``).
            The non-selected side — and any non-numeric column on either side —
            is read from the row whose ``align_on`` is nearest each requested
            value. Note every column of an interpolated side is fitted, including
            study context columns that are not delta parameters, since the whole
            row is synthetic.
        kind : str | tuple | None
            Regression/interpolation spec applied to the interpolated side(s),
            accepting the same specs as ``reg_order`` (e.g. ``'poly2'``, ``'log'``,
            ``'exp'``, ``'power'``, ``(kind, param)`` tuples). When ``None`` each
            side falls back to its own dataset's ``reg_order``; when no spec is
            set at all, values are interpolated piecewise-linearly through the raw
            points. The result gains ``BASE_METHOD`` / ``STUDY_METHOD`` columns
            naming how each side's numeric values were produced (the regression
            label, ``'Table'`` for 1-D linear interpolation, or ``'Nearest'``).
            Only used when ``x_ins`` is given.
        name_as_study : bool
            When ``True``, title each delta set after its study set instead of the
            default ``'Set {study} rel. to Set {base}'``. The study set's colour
            and marker are inherited regardless of this flag.
        name_by : 'index' | 'name'
            What identifies each set in the default title: ``'index'`` (default)
            uses the set indices, ``'name'`` uses the set titles. Ignored when
            ``name_as_study=True``.
        """
        if x_ins is not None and interp not in ('base', 'study', 'both'):
            raise ValueError("interp must be 'base', 'study', or 'both'.")
        if name_by not in ('index', 'name', 'title', 'settitle'):
            raise ValueError("name_by must be 'index' or 'name'.")
        # Resolve align_on from last plot state
        if align_on is None:
            lx = self.last_x
            align_on = lx[0] if isinstance(lx, list) else lx
        if align_on is None:
            raise ValueError("align_on is required when no prior plot exists.")
        if isinstance(align_on, list):
            raise ValueError("align_on must be a single column name, not a list.")

        # Resolve delta_parms from last plot state
        if delta_parms is None:
            ly = self.last_y
            delta_parms = ly if isinstance(ly, list) else ([ly] if ly is not None else [])
        if not isinstance(delta_parms, list):
            delta_parms = [delta_parms]
        delta_parms = [p for p in delta_parms if p is not None]
        if not delta_parms:
            raise ValueError("delta_parms is required when no prior plot exists.")

        # align_on is the merge key, not a delta target. If it slipped into
        # delta_parms (e.g. a SETNUMBER/INDEX column that is also last_x, or an
        # explicit list that includes the align column), drop it — the base side
        # keeps the key un-renamed, so a '<align_on>_BASE' column never exists and
        # deltaing the key against itself is meaningless.
        if align_on in delta_parms:
            print(f"Note: '{align_on}' is the alignment key, not a delta parameter — "
                  f"ignoring it in delta_parms.")
            delta_parms = [p for p in delta_parms if p != align_on]
            if not delta_parms:
                raise ValueError(
                    "delta_parms contained only the alignment column; nothing to delta.")

        # Normalize study-side passthrough specs
        def _as_list(v):
            if v is None:
                return []
            return list(v) if isinstance(v, (list, tuple, set)) else [v]
        passed_parms = _as_list(passed_parms)

        # keep_parms: True/'all' -> every delta parm; None/False -> none; else a subset.
        if keep_parms is True or (isinstance(keep_parms, str) and keep_parms.lower() == 'all'):
            keep_parms = list(delta_parms)
        elif keep_parms is None or keep_parms is False:
            keep_parms = []
        else:
            keep_parms = _as_list(keep_parms)

        if not (0 <= base_idx < len(self.sets)):
            raise IndexError(f"base_idx {base_idx} is out of range (have {len(self.sets)} datasets).")

        base_ds = self.sets[base_idx]
        if align_on not in base_ds.columns:
            raise ValueError(f"align_on column '{align_on}' not found in base dataset '{base_ds.title}'.")
        if x_ins is not None and not pd.api.types.is_numeric_dtype(base_ds[align_on]):
            raise ValueError(
                f"x_ins interpolation requires a numeric align_on; '{align_on}' is "
                f"not numeric in base dataset '{base_ds.title}'.")

        def _nearest_key(src_df, xcol):
            """Sort permutation and sorted x for ``xcol`` — computed once per side
            and shared by every `_read_col_at` nearest lookup on that side."""
            x = src_df[xcol].to_numpy(dtype=float)
            order = np.argsort(x)
            return order, x[order]

        def _read_col_at(src_df, xcol, ycol, x_arr, do_interp, spec, order, x_sorted):
            """Read ``ycol`` from ``src_df`` at the ``x_arr`` positions of ``xcol``.

            Numeric columns on an interpolated side are read off the regression
            curve (``spec``) when one fits, else by 1-D linear interpolation
            through the raw points. Non-numeric columns, and any column on a
            non-interpolated side, carry the value from the row whose ``xcol`` is
            nearest each requested point (``order``/``x_sorted`` from
            ``_nearest_key``). Returns ``(values, method_label)``.
            """
            col = src_df[ycol]
            if do_interp and pd.api.types.is_numeric_dtype(col):
                rx, ry, fit_label = (_calculate_regression(src_df, xcol, ycol, spec)
                                     if spec else (None, None, None))
                if rx is not None:
                    return np.interp(x_arr, rx, ry), fit_label
                return table_read(src_df, xcol, ycol, x_arr, kind='linear'), 'Table'
            y_sorted = col.to_numpy()[order]
            if len(x_sorted) == 1:
                nearest = np.zeros(len(x_arr), dtype=np.intp)
            else:
                idx = np.clip(np.searchsorted(x_sorted, x_arr), 1, len(x_sorted) - 1)
                left, right = x_sorted[idx - 1], x_sorted[idx]
                nearest = np.where(x_arr - left <= right - x_arr, idx - 1, idx)
                # Among duplicate x values take the first occurrence, matching the
                # full argmin scan this replaces.
                nearest = np.searchsorted(x_sorted, x_sorted[nearest], side='left')
            return y_sorted[nearest], 'Nearest'

        # Exclude the base from study targets to avoid a trivial zero-delta set
        targets = [ds for ds in self._get_uset_slice(study_indices) if ds.index != base_idx]
        if not targets:
            print("No study datasets to process (base dataset excluded if present in selection).")
            return []

        # --- Base side: the reference. Only its column labels are needed up front,
        # to resolve which delta parms exist on both sides; the base's values are
        # pulled per study as a narrow '<name>_BASE' slice inside the loop. The
        # result now carries the STUDY set, so the full-width prep is per-study.
        base_cols = list(base_ds.columns)

        created = []

        for study_ds in targets:
            study_cols = study_ds.columns
            if align_on not in study_cols:
                print(f"Warning: skipping '{study_ds.title}' — align_on column '{align_on}' not found.")
                continue
            if x_ins is not None and not pd.api.types.is_numeric_dtype(study_ds[align_on]):
                print(f"Warning: skipping '{study_ds.title}' — x_ins interpolation needs a "
                      f"numeric align_on, but '{align_on}' is not numeric there.")
                continue

            valid_parms = [p for p in delta_parms
                           if p in base_cols and p in study_cols]
            skipped = sorted(set(delta_parms) - set(valid_parms))
            if skipped:
                print(f"Warning: skipping columns not present in both datasets: {skipped}")
            if not valid_parms:
                print(f"Warning: skipping '{study_ds.title}' — no valid delta columns found.")
                continue

            # Resolve base-side keep (a subset of delta parms) and passthrough columns.
            # align_on is always the merge key and is never duplicated as a *_BASE col.
            keep_valid = [p for p in keep_parms if p != align_on and p in valid_parms]
            keep_dropped = [p for p in keep_parms if p != align_on and p not in valid_parms]
            if keep_dropped:
                print(f"Warning: keep_parms not among valid delta columns (ignored): "
                      f"{sorted(set(keep_dropped))}")

            passed_valid = [c for c in passed_parms
                            if c != align_on and c in base_cols]
            passed_missing = [c for c in passed_parms
                              if c != align_on and c not in base_cols]
            if passed_missing:
                print(f"Warning: passed_parms not in base '{base_ds.title}' (ignored): {passed_missing}")

            # --- Study side: every study column, anchored and sorted on align_on. ---
            # study_ds.df materializes a fresh slice of the combined frame; ownership
            # already excludes other sets' phantom columns (e.g. the DL_/DLPCT_/
            # METHOD outputs of a previous delta) from the view, so the NaN scan
            # below only spans the study set's own width. It drops the study set's
            # own genuinely all-NaN columns so they are not carried into the result
            # as empty context — but never the align key or an actual delta parameter.
            study_raw = study_ds.df
            if study_raw.columns.duplicated().any():
                study_raw = study_raw.loc[:, ~study_raw.columns.duplicated()]
            df_study = study_raw.sort_values(align_on).reset_index(drop=True)
            study_all_nan = df_study.isna().all()
            phantom = [c for c in df_study.columns
                       if c != align_on and c not in valid_parms and study_all_nan[c]]
            if phantom:
                df_study = df_study.drop(columns=phantom)

            # --- Base side: align_on + (delta parms ∪ passthroughs), renamed *_BASE. ---
            # ds.cols keeps the first occurrence of any duplicated label and only
            # copies the named columns, never the base set's full width.
            base_need = list(dict.fromkeys([align_on] + valid_parms + passed_valid))
            df_base = (base_ds.cols(base_need)
                       .sort_values(align_on).reset_index(drop=True))
            df_base = df_base.rename(
                columns={c: f"{c}_BASE" for c in df_base.columns if c != align_on})

            # Guard against a pathological study column already named like a *_BASE col;
            # base values win for that name so the merge stays clean.
            overlap = (set(df_study.columns) & set(df_base.columns)) - {align_on}
            if overlap:
                print(f"Warning: study column(s) collide with base '*_BASE' names and were "
                      f"dropped in favor of base values: {sorted(overlap)}")
                df_study = df_study.drop(columns=list(overlap))

            # Build `merged`: align_on + study columns (original names) + base
            # columns (renamed *_BASE). Two ways to populate it, both yielding
            # the same column shape so the delta math below is shared:
            #   - default: nearest/forward/backward merge_asof on the study rows.
            #   - x_ins:   rows at the requested align_on values, each side read
            #              off an interpolated curve (or nearest raw row).
            base_methods = study_methods = None
            if x_ins is None:
                merge_kwargs = dict(on=align_on, direction=direction)
                if tolerance is not None:
                    merge_kwargs['tolerance'] = tolerance
                merged = pd.merge_asof(df_study, df_base, **merge_kwargs)
            else:
                x_arr = np.atleast_1d(x_ins).astype(float)
                base_spec = kind if kind is not None else base_ds.reg_order
                study_spec = kind if kind is not None else study_ds.reg_order
                base_interp = interp in ('base', 'both')
                study_interp = interp in ('study', 'both')

                # Collect columns in a dict and build the frame once — per-column
                # df[c] = ... inserts fragment the frame (PerformanceWarning) on
                # wide study sets.
                data = {align_on: x_arr}
                base_methods, study_methods = set(), set()
                study_key = _nearest_key(df_study, align_on)
                base_key = _nearest_key(df_base, align_on)
                for c in df_study.columns:
                    if c == align_on:
                        continue
                    data[c], m = _read_col_at(
                        df_study, align_on, c, x_arr, study_interp, study_spec, *study_key)
                    if pd.api.types.is_numeric_dtype(df_study[c]):
                        study_methods.add(m)
                for c in df_base.columns:
                    if c == align_on:
                        continue
                    data[c], m = _read_col_at(
                        df_base, align_on, c, x_arr, base_interp, base_spec, *base_key)
                    if pd.api.types.is_numeric_dtype(df_base[c]):
                        base_methods.add(m)
                merged = pd.DataFrame(data)

            # Only parms that are numeric on BOTH sides can be subtracted. Anything
            # else (strings, categoricals, datetimes, object dtype) is carried
            # through as a study/base side-by-side pair instead of crashing the
            # subtraction. dtype is checked post-merge so unmatched rows (NaN) and
            # any merge upcasting are reflected.
            numeric_parms, nonnumeric_parms = [], []
            for parm in valid_parms:
                s_num = pd.api.types.is_numeric_dtype(merged[parm])
                b_num = pd.api.types.is_numeric_dtype(merged[f"{parm}_BASE"])
                (numeric_parms if (b_num and s_num) else nonnumeric_parms).append(parm)

            if nonnumeric_parms:
                print(f"Warning: '{study_ds.title}' — cannot compute a numeric delta for "
                      f"non-numeric column(s) {nonnumeric_parms}; carrying study and base "
                      f"values side-by-side (as '<name>' and '<name>_BASE') instead.")

            # Deltas (numeric parms only): study value keeps its original name,
            # base value is *_BASE. Deltas stay base-referenced (study − base, and
            # % of base). Appended in one concat rather than per-column inserts,
            # which fragment the frame.
            delta_cols = {}
            for parm in numeric_parms:
                s_col, b_col = parm, f"{parm}_BASE"
                delta_cols[f"DL_{parm}"] = merged[s_col] - merged[b_col]
                delta_cols[f"DLPCT_{parm}"] = np.where(
                    merged[b_col] == 0, np.nan,
                    100 * ((merged[s_col] - merged[b_col]) / merged[b_col])
                )
            if delta_cols:
                merged = pd.concat(
                    [merged, pd.DataFrame(delta_cols, index=merged.index)], axis=1)

            # Which base *_BASE columns survive into the result: explicit keeps,
            # every non-numeric parm (so a carried string is actually comparable),
            # and the passthroughs.
            base_keep_cols = ({f"{p}_BASE" for p in keep_valid}
                              | {f"{p}_BASE" for p in nonnumeric_parms}
                              | {f"{c}_BASE" for c in passed_valid})

            # Assemble result with a predictable, plot-friendly column order:
            #   1. align_on
            #   2. per-parm block:
            #        numeric     -> <P>, <P>_BASE (if kept), DL_<P>, DLPCT_<P>
            #        non-numeric -> <P>, <P>_BASE            (no delta)
            #   3. remaining study context columns (full study set, original names)
            #   4. base passthrough columns (<name>_BASE)
            # Built as an ordered name list + one selection (not per-column
            # inserts, which fragment the frame on wide study sets).
            ordered = [align_on]
            for parm in valid_parms:
                ordered.append(parm)                              # study value (original name)
                b_col = f"{parm}_BASE"
                if parm in nonnumeric_parms:
                    ordered.append(b_col)                         # base value, no delta
                else:
                    if b_col in base_keep_cols:
                        ordered.append(b_col)                     # base value (kept)
                    ordered.append(f"DL_{parm}")
                    ordered.append(f"DLPCT_{parm}")

            ordered.extend(c for c in df_study.columns            # remaining study context
                           if c not in ordered)
            ordered.extend(b_col for c in passed_valid            # base passthroughs
                           if (b_col := f"{c}_BASE") not in ordered)
            result = merged.loc[:, list(dict.fromkeys(ordered))].copy()

            # On the x_ins path, every row is synthetic: record how each side's
            # numeric values were produced (regression label / 'Table' / 'Nearest').
            if x_ins is not None:
                result['BASE_METHOD'] = '/'.join(sorted(base_methods)) if base_methods else 'Nearest'
                result['STUDY_METHOD'] = '/'.join(sorted(study_methods)) if study_methods else 'Nearest'

            if numeric_parms:
                nan_frac = result[f"DL_{numeric_parms[0]}"].isna().mean()
                if nan_frac > 0.5:
                    print(f"Warning: '{study_ds.title}' — {nan_frac:.0%} of delta rows are NaN "
                          f"(large alignment gaps; consider tolerance= or a different direction=).")

            if name_as_study:
                new_title = study_ds.title
            elif name_by.lower() in ['name', 'title', 'settitle']:
                new_title = f"{study_ds.title} rel. to {base_ds.title}"
            else:
                new_title = f"Set {study_ds.index} rel. to Set {base_ds.index}"
            ds = self._register_set(result, new_title)
            ds.set_type = 'delta'
            ds.delta_sets = {'base': base_ds.index, 'study': study_ds.index}
            # Inherit the study set's colour and marker so the delta reads as a
            # visual continuation of the study series it was derived from.
            ds.color = study_ds.color
            ds.marker = study_ds.marker
            if x_ins is not None:
                ds.delta_sets['x_ins'] = [float(v) for v in x_arr]
                ds.delta_sets['interp'] = interp
            print(f"Loaded Set {ds.index}: {new_title}")
            created.append(ds)

        return created
        
    def combine_sets(self, uset_slice, title=None, ignore_index=True):
        """Concatenate multiple datasets row-wise into a new dataset.

        Parameters
        ----------
        uset_slice : int | list | 'all'
            Datasets to combine. Must resolve to at least 2 datasets.
        title : str | None
            Title for the new dataset. Defaults to 'Combined 0-1-2-...' using source indices.
        ignore_index : bool
            Reset the row index in the combined DataFrame (default True). Set to False to
            preserve the original indices, which may be useful if they carry meaning.
        """
        sources = self._get_uset_slice(uset_slice)
        if len(sources) < 2:
            print(f"Warning: combine_sets requires at least 2 datasets (got {len(sources)}).")
            return None

        col_sets = [set(ds.columns) for ds in sources]
        shared = col_sets[0].intersection(*col_sets[1:])
        all_cols = set().union(*col_sets)
        only_in_some = all_cols - shared
        if only_in_some:
            print(f"Warning: {len(only_in_some)} column(s) not present in all datasets — "
                  f"those cells will be NaN: {sorted(only_in_some)}")

        combined = pd.concat([ds.df for ds in sources], ignore_index=ignore_index)

        idx_str = '-'.join(str(ds.index) for ds in sources)
        new_title = title or f"Combined {idx_str}"
        new_ds = self._register_set(combined, new_title)
        print(f"Loaded Set {new_ds.index}: {new_title} ({len(combined)} rows from {len(sources)} datasets)")
        return new_ds

    def combine(self, uset_slice, title=None, ignore_index=True):
        """Alias for :meth:`combine_sets`: concatenate datasets into a new set."""
        return self.combine_sets(uset_slice, title=title, ignore_index=ignore_index)

    # ------------------------------------------------------------------
    # Axes Based Decorations (Lines/Highlights/Scale)
    # ------------------------------------------------------------------
    def line(self, column, level, color='red', linestyle=None, dash=None):
        """Add a vertical or horizontal line to the next plot.

        Args:
            column (str): The variable the line is keyed to (x-var -> vertical
                line; y-var -> horizontal line). Use 'all' with level='clear'.
            level (float or 'clear'): The line position, or 'clear' to remove
                the line(s) for ``column`` ('all' clears every line).
            color (str): Line color.
            linestyle (str, optional): Line style — Matplotlib-style
                ('-', '--', '-.', ':') or Plotly-style ('solid', 'dash',
                'dashdot', 'dot'). Defaults to 'dash'.
            dash (str, optional): Deprecated alias for ``linestyle``, kept for
                backwards compatibility. Ignored if ``linestyle`` is given.
        """
        if level == 'clear':
            if column == 'all':
                self.lines.clear()
            else:
                self.lines.pop(column, None)
            return

        # linestyle is the preferred name; dash is the legacy alias. When
        # neither is supplied, preserve the original default of 'dash'.
        style = linestyle if linestyle is not None else (dash if dash is not None else 'dash')

        if column not in self.lines: self.lines[column] = []
        plotly_dash = LINESTYLE_MAP_MPL_TO_PLOTLY.get(style, style)
        self.lines[column].append({'level': level, 'color': color, 'dash': plotly_dash})

    def highlight(self, column, range_tuple, color='yellow', alpha=0.2, opacity=None):
        """Add a highlighted region to the next plot."""
        if opacity is not None:
            warnings.warn("'opacity' is deprecated, use 'alpha'", DeprecationWarning, stacklevel=2)
            alpha = opacity
        if range_tuple == 'clear':
            if column == 'all':
                self.highlights.clear()
            else:
                self.highlights.pop(column, None)
            return

        if column not in self.highlights: self.highlights[column] = []
        self.highlights[column].append({'range': range_tuple, 'color': color, 'alpha': alpha})
        
    def scale(self, column, range_tuple):
        """
        Set specific axis limits for one or more parameters.

        ``column`` may be a single parameter name or a list/tuple of names,
        in which case the same ``range_tuple`` is applied to each.
        """
        if isinstance(column, (list, tuple)):
            columns = column
        else:
            columns = [column]

        for col in columns:
            if range_tuple == 'clear' or range_tuple is None:
                if col in self.axis_limits:
                    del self.axis_limits[col]
                    print(f"Limits cleared for '{col}'.")
            else:
                if isinstance(range_tuple, (list, tuple)) and len(range_tuple) == 2:
                    self.axis_limits[col] = range_tuple
                    print(f"Limits set for '{col}': {range_tuple}")
                else:
                    raise ValueError(f"Invalid range for {col}. Must be a tuple (min, max).")

    # ------------------------------------------------------------------
    # Font Management
    # ------------------------------------------------------------------
    def set_font_sizes(self, suptitle=None, footer=None, legend=None, axes_title=None,
                    axes_tick=None, subplot_title=None, colorbar=None,
                    hover=None, table_header=None, table_cell=None, all=None, reset=False):
        """
        Configure font sizes for plot and table elements. Settings persist across plots.

        Parameters
        ----------
        suptitle, footer, legend, axes_title, axes_tick, subplot_title, colorbar, hover : float or str
            Font sizes for plot elements.
        table_header : float or str
            Font size for table header row.
        table_cell : float or str
            Font size for table cell content.
        all : float or str
            Set all font sizes at once (overridden by individual parameters).
        reset : bool
            Reset all font sizes to defaults.
        """
        keys = ('suptitle_size', 'footer_size', 'legend_size', 'axes_title_size', 'axes_tick_size',
                'subplot_title_size', 'colorbar_size', 'hover_size', 'table_header_size', 'table_cell_size')

        if reset:
            for k in keys:
                setattr(self, k, None)
            return

        def _validate(name, value):
            if value is None:
                return None
            if isinstance(value, str):
                resolved = FONT_SIZE_MAP.get(value.lower())
                if resolved is None:
                    valid = ', '.join(sorted(FONT_SIZE_MAP))
                    raise ValueError(f"{name}: unknown size name '{value}'. Valid names: {valid}")
                value = resolved
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric or a size name, got {type(value).__name__}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            if value > 72:
                warnings.warn(f"{name}={value} is unusually large for a font size.")
            return float(value)

        base = _validate('all', all)
        resolved = {
            'suptitle_size':      _validate('suptitle', suptitle)           if suptitle      is not None else base,
            'footer_size':        _validate('footer', footer)               if footer        is not None else base,
            'legend_size':        _validate('legend', legend)               if legend        is not None else base,
            'axes_title_size':    _validate('axes_title', axes_title)       if axes_title    is not None else base,
            'axes_tick_size':     _validate('axes_tick', axes_tick)         if axes_tick     is not None else base,
            'subplot_title_size': _validate('subplot_title', subplot_title) if subplot_title is not None else base,
            'colorbar_size':      _validate('colorbar', colorbar)           if colorbar      is not None else base,
            'hover_size':         _validate('hover', hover)                 if hover         is not None else base,
            'table_header_size':  _validate('table_header', table_header)   if table_header  is not None else base,
            'table_cell_size':    _validate('table_cell', table_cell)       if table_cell    is not None else base,
        }
        for k, v in resolved.items():
            if v is not None:
                setattr(self, k, v)


    def get_font_sizes(self):
        """Return a dict of currently configured font sizes (None = unset/default)."""
        return {
            'suptitle':       self.suptitle_size,
            'footer':         self.footer_size,
            'legend':         self.legend_size,
            'axes_title':     self.axes_title_size,
            'axes_tick':      self.axes_tick_size,
            'subplot_title':  getattr(self, 'subplot_title_size', None),
            'colorbar':       getattr(self, 'colorbar_size', None),
            'hover':          getattr(self, 'hover_size', None),
            'table_header':   getattr(self, 'table_header_size', None),
            'table_cell':     getattr(self, 'table_cell_size', None),
        }

    # ------------------------------------------------------------------
    # Plot-area sizing
    # ------------------------------------------------------------------
    def set_plot_size(self, width=None, height=None, reset=False):
        """Pin the inner plot-area size so plots stay the same size regardless of
        suptitle lines, legend rows, or other margin changes.

        ``width``/``height`` are in inches (same units as ``figsize``). The figure
        is grown to ``plot_area + margins``, so the drawing region is held constant
        while margins absorb the title/legend. Pass only the dimension(s) you want
        to pin — ``None`` leaves that dimension driven by ``figsize``. Each call
        replaces the previous setting (calling with only ``height`` drops a prior
        ``width`` pin). ``reset=True`` (or both ``None``) clears it.

        Note: the arithmetic is exact when Plotly's margin ``autoexpand`` stays
        inert, which it does for the suptitle and the (downward-growing) legend.
        A colorbar (contour/hue) or very long tick labels can still autoexpand the
        right/left margin and make that dimension come out slightly short.
        """
        if reset or (width is None and height is None):
            self.plot_size = None
            return

        def _v(name, val):
            if val is None:
                return None
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be numeric (inches), got {type(val).__name__}")
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")
            return val * 100

        self.plot_size = (_v('width', width), _v('height', height))

    def _enforce_plot_size(self, fig):
        """Resize the figure so its inner plot area matches ``self.plot_size``.
        No-op unless a plot size is pinned. Runs after fonts/margins are final."""
        if fig is None or self.plot_size is None:
            return fig
        pw, ph = self.plot_size
        m = fig.layout.margin

        def mv(val, default):
            return default if val is None else val

        if pw is not None:
            fig.update_layout(width=pw + mv(m.l, 80) + mv(m.r, 80))
        if ph is not None:
            fig.update_layout(height=ph + mv(m.t, 100) + mv(m.b, 80))
        return fig

    def _apply_footer(self, fig, footer):
        """Add a bottom text box (footer/caption) and reserve room for it below
        the x-axis labels. No-op when ``footer`` is falsy, so the default
        behavior (no footer) is unchanged. Uses ``self.footer_size`` if set.

        The footer is an annotation in paper coords with a negative ``y`` (i.e.
        in the bottom margin). Its position is computed against the *final* plot
        area height, including the ``set_plot_size`` height pin which is applied
        right after this in ``_finalize``."""
        if fig is None or not footer:
            return fig
        text = footer.replace('\n', '<br>')
        m = fig.layout.margin
        band = _bottom_space(text, self.footer_size)
        base_b = m.b if m.b is not None else 80
        new_b = base_b + band
        fig.update_layout(margin=dict(b=new_b))

        # Plot-area height in px (one paper-y unit). If the height is pinned via
        # set_plot_size, that is the final inner height; otherwise derive it.
        if self.plot_size is not None and self.plot_size[1] is not None:
            plot_h = self.plot_size[1]
        else:
            t = m.t if m.t is not None else 100
            plot_h = max(1.0, (fig.layout.height or 800) - t - new_b)

        # Sit the footer's bottom edge _FOOTER_PAD px above the figure's bottom,
        # below the axis labels (which occupy base_b); extra lines grow upward.
        y = -(new_b - _FOOTER_PAD) / plot_h
        ann = dict(text=text, showarrow=False, align='center',
                   x=0.5, xref='paper', xanchor='center',
                   y=y, yref='paper', yanchor='bottom')
        if self.footer_size is not None:
            ann['font'] = dict(size=self.footer_size)
        fig.add_annotation(**ann)
        return fig

    def _apply_fonts(self, fig):
        """Apply stored font sizes to a Plotly figure."""
        if fig is None:
            return fig

        layout_updates = {}
        if self.suptitle_size is not None:
            layout_updates['title_font'] = dict(size=self.suptitle_size)
        if self.legend_size is not None:
            layout_updates['legend'] = dict(font=dict(size=self.legend_size))
        if getattr(self, 'hover_size', None) is not None:
            layout_updates['hoverlabel'] = dict(font=dict(size=self.hover_size))
        if layout_updates:
            fig.update_layout(**layout_updates)

        # With the (possibly larger) custom title font now applied, re-reserve
        # the top space so a bigger suptitle can't collide with an above-legend.
        # _base_layout sized it for the default font; redo it for the real size.
        if self.suptitle_size is not None and fig.layout.title.text:
            height = fig.layout.height or 800
            leg = fig.layout.legend
            has_above = (leg.orientation == 'h' and leg.yref == 'container')
            top_margin, title_pos, legend_pos = _top_space(
                fig.layout.title.text, (None, height / 100), has_above,
                title_font_size=self.suptitle_size)
            geo = {'margin': dict(t=top_margin), 'title': dict(y=title_pos['y'])}
            if legend_pos is not None:
                geo['legend'] = dict(y=legend_pos['y'])
            fig.update_layout(**geo)

        sp_size = getattr(self, 'subplot_title_size', None)
        if sp_size is not None and fig.layout.annotations:
            for ann in fig.layout.annotations:
                ann.font = dict(size=sp_size)

        x_updates, y_updates = {}, {}
        if self.axes_title_size is not None:
            x_updates['title_font'] = dict(size=self.axes_title_size)
            y_updates['title_font'] = dict(size=self.axes_title_size)
        if self.axes_tick_size is not None:
            x_updates['tickfont'] = dict(size=self.axes_tick_size)
            y_updates['tickfont'] = dict(size=self.axes_tick_size)
        if x_updates:
            fig.update_xaxes(**x_updates)
        if y_updates:
            fig.update_yaxes(**y_updates)

        cb_size = getattr(self, 'colorbar_size', None)
        if cb_size is not None:
            for trace in fig.data:
                cb = getattr(trace, 'colorbar', None)
                if cb is not None:
                    cb.tickfont = dict(size=cb_size)
                    if cb.title is not None:
                        try:
                            cb.title.font = dict(size=cb_size)
                        except (AttributeError, ValueError):
                            cb.title = dict(text=str(cb.title), font=dict(size=cb_size))

        return fig

    def _finalize(self, fig, suppress_legends, footer=None):
        """Shared tail for every plotting method: apply font sizes, add the
        optional footer, optionally collapse traces to legend-only, and cache the
        figure as ``last_fig``.

        This is the one step a new plotting method must not forget — keeping the
        ``self.last_fig`` cache contract in a single place. Grid sizing, decorations,
        and axis-limit ranges stay in each method, since those differ per plot type.

        ``footer`` is added after ``_apply_fonts`` (so the subplot-title font loop
        doesn't resize it) and before ``_enforce_plot_size`` (so the reserved
        bottom margin is included when pinning the plot area).
        """
        fig = self._apply_fonts(fig)
        fig = self._apply_footer(fig, footer)
        fig = self._enforce_plot_size(fig)
        if fig is not None and suppress_legends:
            fig.update_traces(visible='legendonly')
        self.last_fig = fig
        if self.static_images and fig is not None:
            return self._render_static(fig)
        return fig

    @staticmethod
    def _suppress_mathjax():
        """Disable Plotly's MathJax so static images don't carry the
        'Loading [MathJax]...' artifact. Shared by save_png and the static
        render path so the two can't drift."""
        import plotly.io as pio
        pio.defaults.mathjax = None

    def _render_static(self, fig):
        """Render ``fig`` to a flat inline PNG to keep notebook size down.

        Falls back to returning the interactive figure (warning once) if static
        rendering fails — e.g. 'kaleido' isn't installed — so a missing dep
        degrades to interactive plots rather than making every plot vanish."""
        try:
            from IPython.display import Image
            self._suppress_mathjax()
            return Image(fig.to_image(format="png", scale=self.static_scale))
        except Exception as e:
            if not getattr(self, '_static_warned', False):
                print(f"Static image render failed ({e}); falling back to "
                      "interactive plots. Is 'kaleido' installed?")
                self._static_warned = True
            return fig

    def set_static_images(self, enabled=True, scale=None):
        """Return plots as flat inline PNGs instead of interactive Plotly HTML.

        Interactive figures embed plotly.js in the notebook, bloating file size
        — especially with many plots. Enabling this makes every plotting method
        return a static PNG (via ``IPython.display.Image``) instead, while
        ``last_fig`` still caches the real figure so ``save_png`` and re-styling
        keep working. Requires the 'kaleido' package; if it's missing, plots
        fall back to interactive automatically.

        Note: with static mode on, plotting methods return an ``Image``, not a
        Plotly ``Figure``, so you can't chain ``.update_layout(...)`` on the
        return value — use ``nb.last_fig`` for that.

        Parameters
        ----------
        enabled : bool — turn static images on (default) or off.
        scale : float > 0, optional — PNG resolution multiplier (default 2).

        Examples
        --------
        nb.set_static_images()            # on, keeps notebook small
        nb.set_static_images(scale=3)     # higher-resolution PNGs
        nb.set_static_images(False)       # back to interactive figures
        """
        self.static_images = bool(enabled)
        if scale is not None:
            if (isinstance(scale, bool) or not isinstance(scale, (int, float))
                    or scale <= 0):
                raise ValueError(f"scale must be a positive number, got {scale!r}")
            self.static_scale = scale
        state = "on" if self.static_images else "off"
        print(f"Static images {state} (scale={self.static_scale}).")

    # ------------------------------------------------------------------
    # Main Plot Function
    # ------------------------------------------------------------------
    def plot(self, x=None, y=None, by='vars', figsize=None, ncols=None, nrows=None,
                subplot_titles=None, suptitle=None, footer=None, suppress_legends=None,
                legend=None, **kwargs):
        """
        Main plotting wrapper.

        Parameters:
        -----------
        by : str, optional
            'vars'    (default) - Subplot per Y variable.
            'sets' / 'datasets' - Subplot per Dataset.
            'ymult'           - Single plot, multiple Y axes (delegates to plot_ymult).
        legend : str, optional
            Legend placement, matching ``plot_ymult`` (default 'above', or the
            ``set_default_format(legend=)`` default):
            'above' - horizontal legend above the plot.
            'right' - vertical legend to the right of the plot.
            'off'   - hide the legend.
        """
        if figsize is None: figsize = self.figsize
        legend = self._apply_default('legend', legend, 'above')
        suppress_legends = self._apply_default('suppress_legends', suppress_legends, False)

        # Delegate to the multi-y wrapper if requested
        if by == 'ymult':
            return self.plot_ymult(x=x, y=y, suptitle=suptitle,
                                     figsize=figsize, legend=legend,
                                     suppress_legends=suppress_legends)

        self._clear_last_fig()

        if x is None: x = self.last_x
        if y is None: y = self.last_y
        self.last_x = x
        self.last_y = y

        # Grid precedence: explicit call arg > standing default > sticky last grid
        ncols, nrows = self._resolve_grid(ncols, nrows)
        if ncols is None and nrows is None and (
                self.last_ncols is not None or self.last_nrows is not None):
            ncols, nrows = self.last_ncols, self.last_nrows
        self.last_ncols = ncols
        self.last_nrows = nrows

        if by == 'sets' or by == 'datasets':
            fig = uniplot_per_dataset(
                list_of_datasets=self.sets,
                x=x,
                y=y,
                display_parms=self.display_parms,
                suptitle=suptitle or self.suptitle,
                figsize=figsize,
                ncols=ncols,
                nrows=nrows,
                darkmode=self.darkmode,
                x_lim=self.axis_limits.get(x) if isinstance(x, str) else None,
                y_lim=None,
                axis_limits=self.axis_limits, 
                return_axes=True 
            )
            mode = 'sets'
            
        else:
            plot_args = {
                'list_of_datasets': self.sets,
                'x': x,
                'y': y,
                'darkmode': self.darkmode,
                'display_parms': self.display_parms,
                'suptitle': suptitle or self.suptitle,
                'xlabel': self.x_label,
                'ylabel': self.y_label,
                'subplot_titles': subplot_titles,
                'return_axes': True,
                'figsize': figsize,
                'ncols': ncols,
                'nrows': nrows,
                'axis_limits': self.axis_limits,
            }
            plot_args.update(kwargs)
            fig = uniplot(**plot_args)
            mode = 'vars'

        if fig is None: return

        x_list = x if isinstance(x, list) else [x]
        y_list = y if isinstance(y, list) else [y]
        active_sets = [d for d in self.sets if d.select]

        if len(x_list) == len(y_list):
            plot_pairs = list(zip(x_list, y_list))
        elif len(x_list) == 1:
            plot_pairs = [(x_list[0], yi) for yi in y_list]
        elif len(y_list) == 1:
            plot_pairs = [(xi, y_list[0]) for xi in x_list]
        else:
            plot_pairs = [(x_list[0], yi) for yi in y_list]

        n_items = len(plot_pairs) if mode == 'vars' else len(active_sets)
        calc_ncols = max(1, _calc_grid(n_items, nrows, ncols)[1])

        fig = self._apply_decorations(
            fig, x_list, y_list, mode, calc_ncols,
            plot_pairs if mode == 'vars' else None
        )

        if mode == 'vars':
            for idx, (xi, yi) in enumerate(plot_pairs):
                r, c = (idx // calc_ncols) + 1, (idx % calc_ncols) + 1
                if xi in self.axis_limits:
                    fig.update_xaxes(range=self.axis_limits[xi], row=r, col=c)
                if yi in self.axis_limits:
                    fig.update_yaxes(range=self.axis_limits[yi], row=r, col=c)

        if legend == 'off':
            fig.update_layout(showlegend=False)
        elif legend == 'right':
            _top, _, _ = _top_space(suptitle or self.suptitle, figsize, False)
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation='v', xanchor='left', x=1.02,
                            yanchor='top', y=1),
                margin=dict(r=160, t=_top),
            )
        else:  # 'above' (default)
            _legend, _top = _above_legend_layout(suptitle or self.suptitle, figsize)
            fig.update_layout(legend=_legend, margin=dict(r=80, t=_top))
        return self._finalize(fig, suppress_legends, footer=footer or self.footer)

    # ------------------------------------------------------------------
    # Multi-Y plot wrapper
    # ------------------------------------------------------------------
    def plot_ymult(self, x=None, y=None, suptitle=None, footer=None, figsize=None,
                     legend=None, legend_group_by='sets', suppress_legends=None):
        """
        Single plot, multiple Y-axes. All selected datasets overlay on the same x-axis.
        Applies all notebook-level formatting: axis_limits, variable_formats, lines, highlights.

        ``legend`` (default 'above') and ``suppress_legends`` (default False) fall
        back to the ``set_default_format`` defaults when not passed.
        """
        if figsize is None: figsize = self.figsize
        legend = self._apply_default('legend', legend, 'above')
        suppress_legends = self._apply_default('suppress_legends', suppress_legends, False)
        self._clear_last_fig()
        if x is None: x = self.last_x
        if y is None: y = self.last_y
        self.last_x, self.last_y = x, y

        y_list = y if isinstance(y, list) else [y]

        fig = uniplot_ymultaxis(
            list_of_datasets=self.sets,
            x=x, y=y_list,
            variable_formats=self.variable_formats,
            display_parms=self.display_parms,
            suptitle=suptitle or self.suptitle,
            xlabel=self.x_label,
            darkmode=self.darkmode,
            figsize=figsize,
            x_lim=self.axis_limits.get(x) if isinstance(x, str) else None,
            axis_limits=self.axis_limits,
            legend=legend,
            legend_group_by=legend_group_by,
            return_axes=True,
        )
        if fig is None:
            return None

        # Apply line/highlight decorations using the y_list → axis map.
        # Vertical lines on x span all axes; horizontal lines on a y-var
        # are drawn on the y-axis assigned to that variable.
        yref_for = {yi: ('y' if i == 0 else f'y{i+1}') for i, yi in enumerate(y_list)}

        for col, lines in self.lines.items():
            if col == x:
                for l in lines:
                    fig.add_vline(x=l['level'],
                                  line_dash=l['dash'] or 'solid',
                                  line_color=l['color'])
            elif col in yref_for:
                yref = yref_for[col]
                for l in lines:
                    fig.add_shape(type='line', x0=0, x1=1,
                                  y0=l['level'], y1=l['level'],
                                  xref='paper', yref=yref,
                                  line=dict(color=l['color'], dash=l['dash'] or 'solid'))

        for col, hls in self.highlights.items():
            if col == x:
                for h in hls:
                    fig.add_vrect(x0=h['range'][0], x1=h['range'][1],
                                  fillcolor=h['color'], opacity=h['alpha'],
                                  layer='below', line_width=0)
            elif col in yref_for:
                yref = yref_for[col]
                for h in hls:
                    fig.add_shape(type='rect', x0=0, x1=1,
                                  y0=h['range'][0], y1=h['range'][1],
                                  xref='paper', yref=yref,
                                  fillcolor=h['color'], opacity=h['alpha'],
                                  layer='below', line_width=0)

        return self._finalize(fig, suppress_legends, footer=footer or self.footer)

    # ------------------------------------------------------------------
    # Interactive Dash dashboard wrapper
    # ------------------------------------------------------------------
    def dashboard(self, panels, **kwargs):
        """Launch an interactive Dash board combining multiple unichart figures.

        Thin wrapper around :func:`unichart_dashboard.dashboard`; imported lazily
        so the optional Dash dependency isn't required to use the rest of the
        toolkit. See that function for ``panels`` and keyword options.
        """
        from unichart_dashboard import dashboard as _dashboard
        return _dashboard(self, panels, **kwargs)

    def dashboard_to_html(self, panels, path, **kwargs):
        """Export a dashboard to a self-contained static HTML file.

        Thin wrapper around :func:`unichart_dashboard.to_html`; imported lazily.
        Renders each panel once and writes an offline board whose charts stay
        interactive (hover / zoom / modebar) and, by default, keep a global
        dataset filter in the header (``global_select=True``) that re-slices
        every panel client-side, seeded from the notebook's current selection.
        The rest of the editing chrome (dropdowns, theme switch) is dropped.
        See that function for the keyword options (``ncols``, ``width``,
        ``height``, ``title``, ``embed_js``, ``global_select``).
        """
        from unichart_dashboard import to_html as _to_html
        return _to_html(self, panels, path, **kwargs)

    # ------------------------------------------------------------------
    # The bar Command
    # ------------------------------------------------------------------
    def bar(self, x=None, y=None, markers=None, by='vars', barmode=None, agg=None,
            color=None, suptitle=None, footer=None, figsize=None, ncols=None, nrows=None, suppress_legends=None):
        """
        Unified interface for Bar Charts.

        `color` forces a single bar color for the default `by='vars'` view
        (mirroring `box`/`histogram`); the `by='sets'` and `by='dataset_x'`
        views color per dataset / per variable and ignore it.

        `barmode` (default 'group'), `agg` (default 'mean') and `suppress_legends`
        (default False) fall back to the `set_default_format` defaults when not passed.

        Marker overlay formatting is controlled via `var_format`. Examples:
            nb.var_format('EGT_LIMIT', color='red', marker='*', markersize=18)
            nb.bar(x='PHASE', y='EGT', markers='EGT_LIMIT')

        An overlay column can also render as a tick or whisker instead of a
        marker symbol, via the `style` key of `var_format`:
            nb.var_format('EGT_LIMIT', style='tick')     # horizontal dash at the value
            nb.var_format('EGT_LIMIT', style='whisker')  # dash + stem to the bar top
        """
        if figsize is None: figsize = self.figsize
        barmode = self._apply_default('barmode', barmode, 'group')
        agg = self._apply_default('agg', agg, 'mean')
        suppress_legends = self._apply_default('suppress_legends', suppress_legends, False)
        ncols, nrows = self._resolve_grid(ncols, nrows)
        self._clear_last_fig()

        if x is None: x = self.last_x
        if y is None: y = self.last_y
        self.last_x, self.last_y = x, y

        y_list = y if isinstance(y, list) else [y]

        if by == 'dataset_x':
            if markers:
                print("Warning: `markers` is not supported with by='dataset_x'.")
            fig = unibar_datasets_as_x(
                list_of_datasets=self.sets, y=y_list, agg=agg,
                variable_formats=self.variable_formats,         # <-- pass through
                suptitle=suptitle or self.suptitle, figsize=figsize,
                darkmode=self.darkmode, axis_limits=self.axis_limits, return_axes=True
            )
            if fig:
                fig = self._apply_decorations(fig, [], y_list, 'global', 1)
                fig = self._finalize(fig, suppress_legends, footer=footer or self.footer)
            return fig

        elif by in ['sets', 'datasets']:
            fig = unibar_per_dataset(
                list_of_datasets=self.sets, x=x, y=y, markers=markers,
                variable_formats=self.variable_formats,         # <-- pass through
                barmode=barmode,
                suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, return_axes=True
            )
        else:
            fig = unibar(
                list_of_datasets=self.sets, x=x, y=y, markers=markers,
                variable_formats=self.variable_formats,         # <-- pass through
                barmode=barmode, color=color,
                suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, return_axes=True 
            )

        if fig:
            if x in self.axis_limits:
                fig.update_xaxes(range=self.axis_limits[x])

            active_sets = [d for d in self.sets if d.select]
            n_items = len(active_sets) if by in ['sets', 'datasets'] else len(y_list)
            calc_ncols = max(1, _calc_grid(n_items, nrows, ncols)[1])

            if by in ['sets', 'datasets']:
                primary_y = y_list[0]
                if primary_y in self.axis_limits:
                    fig.update_yaxes(range=self.axis_limits[primary_y])
            else:
                for idx, yi in enumerate(y_list):
                    if yi in self.axis_limits:
                        r = (idx // calc_ncols) + 1
                        c = (idx % calc_ncols) + 1
                        fig.update_yaxes(range=self.axis_limits[yi], row=r, col=c)

            dec_mode = 'sets' if by in ['sets', 'datasets'] else 'vars'
            dec_items = [(x, yi) for yi in y_list] if dec_mode == 'vars' else None
            fig = self._apply_decorations(fig, [], y_list, dec_mode, calc_ncols, dec_items)

            fig = self._finalize(fig, suppress_legends, footer=footer or self.footer)

        return fig

    # ------------------------------------------------------------------
    # The box Command
    # ------------------------------------------------------------------
    def box(self, x=None, y=None, by='vars', boxmode=None, points=None, notched=False,
                color=None, suptitle=None, footer=None, figsize=None, ncols=None, nrows=None, suppress_legends=None):
        """
        Unified interface for Box Plots.

        `boxmode` (default 'group'), `points` (default 'outliers') and
        `suppress_legends` (default False) fall back to the `set_default_format`
        defaults when not passed.
        """
        if figsize is None: figsize = self.figsize
        boxmode = self._apply_default('boxmode', boxmode, 'group')
        points = self._apply_default('points', points, 'outliers')
        suppress_legends = self._apply_default('suppress_legends', suppress_legends, False)
        ncols, nrows = self._resolve_grid(ncols, nrows)
        self._clear_last_fig()

        if x is None: x = self.last_x
        if y is None: y = self.last_y
        self.last_x, self.last_y = x, y

        y_list = y if isinstance(y, list) else [y]

        if by == 'dataset_x':
            fig = unibox_datasets_as_x(
                list_of_datasets=self.sets, y=y_list, boxmode=boxmode,
                points=points, notched=notched, variable_formats=self.variable_formats,
                suptitle=suptitle or self.suptitle,
                figsize=figsize, darkmode=self.darkmode, axis_limits=self.axis_limits, return_axes=True
            )
            if fig:
                fig = self._apply_decorations(fig, [], y_list, 'global', 1)
                _legend, _top = _above_legend_layout(suptitle or self.suptitle, figsize)
                fig.update_layout(legend=_legend, margin=dict(r=80, t=_top))
                fig = self._finalize(fig, suppress_legends, footer=footer or self.footer)
            return fig

        elif by in ['sets', 'datasets']:
            primary_y = y_list[0]
            y_limit = self.axis_limits.get(primary_y)
            fig = unibox_per_dataset(
                list_of_datasets=self.sets, x=x, y=y, boxmode=boxmode,
                points=points, notched=notched,
                suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, y_lim=y_limit, return_axes=True
            )
        else:
            fig = unibox(
                list_of_datasets=self.sets, x=x, y=y, boxmode=boxmode,
                points=points, notched=notched, color=color,
                suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, y_lim=None, return_axes=True
            )
            
            if fig:
                calc_ncols = max(1, _calc_grid(len(y_list), nrows, ncols)[1])

                for idx, yi in enumerate(y_list):
                    if yi in self.axis_limits:
                        r = (idx // calc_ncols) + 1
                        c = (idx % calc_ncols) + 1
                        fig.update_yaxes(range=self.axis_limits[yi], row=r, col=c)

        if fig:
            if x in self.axis_limits:
                fig.update_xaxes(range=self.axis_limits[x])
            if by in ['sets', 'datasets']:
                fig = self._apply_decorations(fig, [], y_list, 'sets', 1)
            else:
                _nc = max(1, _calc_grid(len(y_list), nrows, ncols)[1])
                fig = self._apply_decorations(fig, [], y_list, 'vars', _nc,
                                              [(x, yi) for yi in y_list])
            _legend, _top = _above_legend_layout(suptitle or self.suptitle, figsize)
            fig.update_layout(legend=_legend, margin=dict(r=80, t=_top))
            fig = self._finalize(fig, suppress_legends, footer=footer or self.footer)

        return fig

    # ------------------------------------------------------------------
    # The histogram Command
    # ------------------------------------------------------------------
    def histogram(self, x=None, y=None, histfunc=None, by='vars', nbins=None,
                    bin_size=None, bin_start=None, bin_end=None,
                    histnorm=None, barmode=None, alpha=None,
                    color=None, suptitle=None, footer=None, figsize=None, ncols=None, nrows=None, suppress_legends=None,
                    opacity=None):
        """
        Unified interface for Histograms.

        `histfunc` (default 'sum'), `histnorm` (default ''), `barmode`
        (default 'overlay'), `alpha` (default 0.7) and `suppress_legends`
        (default False) fall back to the `set_default_format` defaults when not passed.
        """
        if figsize is None: figsize = self.figsize
        if opacity is not None:
            warnings.warn("'opacity' is deprecated, use 'alpha'", DeprecationWarning, stacklevel=2)
            alpha = opacity
        histfunc = self._apply_default('histfunc', histfunc, 'sum')
        histnorm = self._apply_default('histnorm', histnorm, '')
        barmode = self._apply_default('barmode', barmode, 'overlay')
        alpha = self._apply_default('alpha', alpha, 0.7)
        suppress_legends = self._apply_default('suppress_legends', suppress_legends, False)
        ncols, nrows = self._resolve_grid(ncols, nrows)
        self._clear_last_fig()

        if x is None: x = self.last_x
        self.last_x = x
        
        limit = None
        if isinstance(x, str):
            limit = self.axis_limits.get(x)
        elif isinstance(x, list) and len(x) == 1:
            limit = self.axis_limits.get(x[0])

        x_list = x if isinstance(x, list) else [x]

        if by in ['sets', 'datasets']:
            fig = unihistogram_by_dataset(
                list_of_datasets=self.sets, x=x, y=y, histfunc=histfunc, nbins=nbins,
                bin_size=bin_size, bin_start=bin_start, bin_end=bin_end,
                histnorm=histnorm, barmode=barmode, alpha=alpha,
                variable_formats=self.variable_formats, color=color,
                suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, x_lim=limit, return_axes=True
            )
            if fig:
                fig = self._apply_decorations(fig, x_list, [], 'sets', 1)
        else:
            fig = unihistogram(
                list_of_datasets=self.sets, x=x, y=y, histfunc=histfunc, nbins=nbins,
                bin_size=bin_size, bin_start=bin_start, bin_end=bin_end,
                histnorm=histnorm, barmode=barmode, alpha=alpha, color=color,
                suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, x_lim=limit, return_axes=True
            )
            if fig:
                _nc = max(1, _calc_grid(len(x_list), nrows, ncols)[1])
                fig = self._apply_decorations(fig, x_list, [], 'vars', _nc,
                                              [(xi, None) for xi in x_list])

        return self._finalize(fig, suppress_legends, footer=footer or self.footer)
        
    # ------------------------------------------------------------------
    # The contour Command
    # ------------------------------------------------------------------
    def contour(self, x=None, y=None, z=None, by='vars', contours_coloring='fill',
                    colorscale=None, interpolate=True, interp_res=100, interp_method='linear',
                    ncontours=None, overlay_sets=None,
                    suptitle=None, footer=None, figsize=None, ncols=None, nrows=None, suppress_legends=None):
        """
        Unified interface for Contour Plots.

        ``overlay_sets`` selects dataset(s) — using the usual selector (int /
        list / 'all' / ``Dataset``) — whose ``(x, y)`` data is drawn on top of
        the contour, honoring each set's full plot style. A set with a
        ``linestyle`` is drawn as a connected line (so it can trace a boundary
        over the field); otherwise its points show as markers. Color, marker,
        size, linewidth and ``fill`` are all respected. The same overlay sets
        are drawn on every subplot (including ``by='sets'``, where each subplot
        is a different dataset). Defaults to ``None`` (no overlay).

        ``suppress_legends`` (default False) falls back to the
        ``set_default_format`` default when not passed.
        """
        if figsize is None: figsize = self.figsize
        suppress_legends = self._apply_default('suppress_legends', suppress_legends, False)
        ncols, nrows = self._resolve_grid(ncols, nrows)
        self._clear_last_fig()

        if x is None: x = self.last_x
        if y is None: y = self.last_y
        if z is None: z = getattr(self, 'last_z', None)

        self.last_x, self.last_y, self.last_z = x, y, z

        if z is None:
            print("Error: Contour plots require a 'z' variable to map to color.")
            return

        overlay_datasets = (self._get_uset_slice(overlay_sets)
                            if overlay_sets is not None else [])

        limit_x = self.axis_limits.get(x)
        limit_y = self.axis_limits.get(y)

        if by in ['sets', 'datasets']:
            fig = unicontour_per_dataset(
                list_of_datasets=self.sets, x=x, y=y, z=z,
                contours_coloring=contours_coloring, colorscale=colorscale,
                interpolate=interpolate, interp_res=interp_res, interp_method=interp_method,
                ncontours=ncontours, overlay_datasets=overlay_datasets,
                suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, axis_limits=self.axis_limits, return_axes=True
            )
        else:
            fig = unicontour(
                list_of_datasets=self.sets, x=x, y=y, z=z,
                contours_coloring=contours_coloring, colorscale=colorscale,
                interpolate=interpolate, interp_res=interp_res, interp_method=interp_method,
                ncontours=ncontours, overlay_datasets=overlay_datasets,
                suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, axis_limits=self.axis_limits, return_axes=True
            )
            
        if fig:
            z_list = z if isinstance(z, list) else [z]
            if by in ['sets', 'datasets']:
                fig = self._apply_decorations(fig, [x], [y], 'sets', 1)
            else:
                _nc = max(1, _calc_grid(len(z_list), nrows, ncols)[1])
                fig = self._apply_decorations(fig, [x], [y], 'vars', _nc,
                                              [(x, y) for _ in z_list])
            if limit_x: fig.update_xaxes(range=limit_x)
            if limit_y: fig.update_yaxes(range=limit_y)
            fig = self._finalize(fig, suppress_legends, footer=footer or self.footer)

        return fig

    def table_read(self, uset_slice, x_col, y_col, x_in,
                   kind=None, fill_value='extrapolate', bounds_error=False):
        """Interpolate y values at ``x_in`` from the selected dataset(s).

        Wraps the module-level :func:`table_read`, calling it once per dataset
        in ``uset_slice`` against the dataset's (query-masked) ``df``.

        The interpolation ``kind`` defaults to each dataset's ``reg_order``
        (falling back to ``'linear'`` when that is unset/falsy), unless the
        caller passes an explicit ``kind``. Note this low-level wrapper passes
        ``kind`` straight to scipy's ``interp1d``, so it only accepts interp1d
        kinds — regression-only specs ('poly2', 'log', tuples) will raise here.
        Use :meth:`table` (which routes through the regression machinery) to
        read values that match a ``reg_order`` curve.

        Returns a dict keyed by ``ds.index`` mapping to the interpolated values.
        """
        results = {}
        for ds in self._get_uset_slice(uset_slice):
            ds_kind = kind if kind is not None else (ds.reg_order or 'linear')
            results[ds.index] = table_read(
                ds.cols([x_col, y_col]), x_col, y_col, x_in,
                kind=ds_kind, fill_value=fill_value, bounds_error=bounds_error,
            )
        return results

    # ------------------------------------------------------------------
    # Column statistics
    # ------------------------------------------------------------------
    def _aggregate(self, uset_slice, column, func, name):
        """Reduce ``column`` over the selected dataset(s) with ``func``.

        Datasets are chosen with the usual selector (int / list / 'all' / None /
        ``Dataset``). The query-masked ``column`` from every matching set is
        concatenated, NaNs are dropped, and ``func`` is applied. With multiple
        sets selected the statistic is computed over their combined values.
        Returns a scalar, or ``None`` (with a message) when no data is found.
        """
        parts = [ds[column] for ds in self._get_uset_slice(uset_slice)
                 if column in ds.columns]
        if not parts:
            print(f"{name}: column {column!r} not found in the selected dataset(s).")
            return None
        data = pd.concat(parts, ignore_index=True).dropna()
        if data.empty:
            print(f"{name}: no values for {column!r} in the selected dataset(s).")
            return None
        return func(data)

    def max(self, uset_slice, column):
        """Highest value of ``column`` in the selected dataset(s). E.g.
        ``uc.max(6, 'T4F')`` returns the maximum of column T4F in set 6."""
        return self._aggregate(uset_slice, column, lambda s: s.max(), 'max')

    def min(self, uset_slice, column):
        """Lowest value of ``column`` in the selected dataset(s)."""
        return self._aggregate(uset_slice, column, lambda s: s.min(), 'min')

    def mean(self, uset_slice, column):
        """Mean of ``column`` in the selected dataset(s)."""
        return self._aggregate(uset_slice, column, lambda s: s.mean(), 'mean')

    def median(self, uset_slice, column):
        """Median of ``column`` in the selected dataset(s)."""
        return self._aggregate(uset_slice, column, lambda s: s.median(), 'median')

    # ------------------------------------------------------------------
    # The table Command
    # ------------------------------------------------------------------
    def table(self, cols=None, title=None, x_col=None, x_in=None, kind=None,
              sig_figs=None, output=None):
        """
        Build a table of column values from the currently selected datasets.

        The method has two modes:

        * **Raw mode** (default) — show the actual rows from each dataset.
        * **Interpolation mode** (when ``x_in`` is given) — show y values
          looked up at the x values you ask for, interpolating or extrapolating
          as needed so they line up with the plotted curve.

        Parameters
        ----------
        cols : str or list of str, optional
            Column(s) to include. Defaults to the columns from the last plot
            (``self.last_x`` + ``self.last_y``). In interpolation mode these are
            the y column(s) to look up.
        title : str, optional
            Reserved for a table title (currently unused).
        x_col : str, optional
            The x column to interpolate against. Defaults to ``self.last_x``.
            Only used in interpolation mode.
        x_in : scalar or list-like, optional
            One or more x values to look up. Supplying this switches the method
            into interpolation mode.
        kind : str or tuple, optional
            Regression/curve spec used for the lookup, accepting the same values
            as a dataset's ``reg_order`` (e.g. ``'poly2'``, ``'log'``, ``'exp'``,
            ``'power'``, ``'spline'``, ``'lowess'``, ``'ma'``, or
            ``(kind, param)`` tuples). Defaults to each dataset's own
            ``reg_order`` so the table matches the plotted curve. When no spec is
            set, values are interpolated piecewise-linearly through the raw
            points.
        sig_figs : int, optional
            Round every float column to this many significant figures for
            display, keeping ordinary decimal notation (no scientific notation).
            Affects the rendered HTML table and Markdown output only; the
            ``output='df'`` DataFrame keeps its full-precision numeric values.
        output : {None, 'df', 'md', 'fig'}, optional
            What to return:

            - ``None`` (default): render and display the styled HTML table.
            - ``'df'``: return the assembled :class:`pandas.DataFrame`.
            - ``'md'``: return a GitHub-flavored Markdown string.
            - ``'fig'``: return the styled Plotly ``go.Figure`` (a ``go.Table``),
              with ``sig_figs`` and dark-mode already applied. Useful for
              embedding the table alongside other figures (e.g. in a dashboard
              panel) without triggering the HTML display side effect.

        Interpolation mode details
        --------------------------
        For each selected dataset, every numeric y column is evaluated at each
        value in ``x_in``. Non-numeric y columns instead carry the value from
        the row whose x is nearest the requested x. Two extra columns describe
        each looked-up value:

        ``INTERPOLATED``
            Where the requested x sits relative to the raw data:

            - ``'In set'`` — x matches an existing data point.
            - ``'Interpolated'`` — x falls between raw points.
            - ``'Extrapolated'`` — x falls outside the data range.

            Values read off a fitted curve never come from a raw point, so they
            are only ever ``'Interpolated'`` or ``'Extrapolated'``.

        ``METHOD``
            How the value was produced:

            - the regression type (e.g. ``'Linear'``, ``'LS2'``, ``'Log'``) when
              a ``reg_order``/``kind`` spec is used,
            - ``'1d interp vs <x_col>'`` for piecewise-linear table
              interpolation between raw points (no spec),
            - ``None`` for exact, in-set points.

        Examples
        --------
        Show the raw columns from the last plot::

            chart.table()

        Show specific columns::

            chart.table(cols=['speed', 'power'])

        Look up ``power`` at a few speeds using each dataset's fitted curve::

            chart.table(cols='power', x_col='speed', x_in=[10, 15, 20])

        Force a quadratic fit and return the result as a DataFrame::

            df = chart.table(cols='power', x_in=[10, 15, 20],
                             kind='poly2', output='df')
        """
        if output is not None and output not in ('df', 'md', 'fig'):
            print(f"Unknown output mode '{output}'. Use None, 'df', 'md', or 'fig'.")
            return
        if sig_figs is not None and (not isinstance(sig_figs, int) or
                                     isinstance(sig_figs, bool) or sig_figs < 1):
            print("sig_figs must be a positive integer.")
            return
        combined_dfs = []

        if x_in is not None:
            xc = x_col or self.last_x
            if xc is None:
                print("Interpolation mode requires an x column (pass x_col= or run a plot first).")
                return

            if cols is not None:
                y_cols = cols if isinstance(cols, list) else [cols]
            elif self.last_y is not None:
                y_cols = self.last_y if isinstance(self.last_y, list) else [self.last_y]
            else:
                print("No y column(s) specified for interpolation.")
                return
            y_cols = [c for c in y_cols if c != xc]

            x_arr = np.atleast_1d(x_in)

            for ds in self.sets:
                if not ds.select:
                    continue

                ds_cols = ds.columns
                if xc not in ds_cols:
                    continue
                valid_ycols = [c for c in y_cols if c in ds_cols]
                if not valid_ycols:
                    continue
                df = ds.cols([xc] + valid_ycols)

                spec = kind if kind is not None else ds.reg_order
                existing = df[xc].to_numpy(dtype=float)

                # Track whether any displayed numeric value was read off a
                # fitted regression curve. When it was, the value comes from the
                # model rather than a raw row, so the point is interpolated even
                # if its x matches an existing data point. ``reg_label`` records
                # the regression type (matching the plot label) for the METHOD
                # column.
                curve_used = False
                reg_label = None

                subset = pd.DataFrame({xc: x_arr})
                for yc in valid_ycols:
                    if pd.api.types.is_numeric_dtype(df[yc]):
                        # Use the same regression model as the plot (any
                        # reg_order kind), reading values off the fitted curve.
                        # Fall back to piecewise-linear interpolation through the
                        # raw points when no regression spec is set.
                        rx, ry, fit_label = (_calculate_regression(df, xc, yc, spec)
                                             if spec else (None, None, None))
                        if rx is not None:
                            subset[yc] = np.interp(x_arr, rx, ry)
                            curve_used = True
                            reg_label = fit_label
                        else:
                            subset[yc] = table_read(df, xc, yc, x_arr, kind='linear')
                    else:
                        # Non-interpolatable (string/categorical) column: carry the
                        # value from the row whose x is nearest to each requested x.
                        order = np.argsort(existing)
                        x_sorted = existing[order]
                        y_sorted = df[yc].to_numpy()[order]
                        nearest = np.abs(x_sorted[:, None] - x_arr[None, :]).argmin(axis=0)
                        subset[yc] = y_sorted[nearest]
                # INTERPOLATED classifies each requested x relative to the
                # dataset: ``'In set'`` when x matches an existing data point,
                # ``'Extrapolated'`` when x falls outside the data range, and
                # ``'Interpolated'`` when x falls between raw points. A value
                # read off a fitted curve never comes from a raw point, so it is
                # only ever ``'Interpolated'`` or ``'Extrapolated'``.
                # METHOD records how each row's value was produced: the
                # regression type for fitted curves, ``'1d interp vs <x_col>'``
                # for 1-D table interpolation between raw points, and ``None``
                # for exact (in-set) points.
                xmin = np.nanmin(existing)
                xmax = np.nanmax(existing)
                if curve_used:
                    subset['INTERPOLATED'] = [
                        'Extrapolated' if (xv < xmin or xv > xmax)
                        else 'Interpolated'
                        for xv in x_arr
                    ]
                    subset['METHOD'] = reg_label
                else:
                    status = []
                    for xv in x_arr:
                        if np.any(np.isclose(existing, xv)):
                            status.append('In set')
                        elif xv < xmin or xv > xmax:
                            status.append('Extrapolated')
                        else:
                            status.append('Interpolated')
                    subset['INTERPOLATED'] = status
                    subset['METHOD'] = [
                        None if s == 'In set' else f"1d interp vs {xc}"
                        for s in status
                    ]
                subset.insert(0, 'Dataset', ds.title)
                subset.insert(0, 'Set', ds.index)
                combined_dfs.append(subset)

            if not combined_dfs:
                print("No data found for the specified columns in selected datasets.")
                return
        else:
            if cols is None:
                if self.last_x is None or self.last_y is None:
                    print("No columns specified and no previous plot variables defined.")
                    return

                y_part = self.last_y if isinstance(self.last_y, list) else [self.last_y]
                target_cols = [self.last_x] + y_part
            else:
                target_cols = cols if isinstance(cols, list) else [cols]

            for ds in self.sets:
                if not ds.select:
                    continue

                valid_cols = [c for c in target_cols if c in ds.columns]

                if not valid_cols:
                    continue

                subset = ds.cols(valid_cols)        # fresh narrow copy — safe to mutate
                subset.insert(0, 'Dataset', ds.title)
                subset.insert(0, 'Set', ds.index)
                combined_dfs.append(subset)

            if not combined_dfs:
                print("No data found for the specified columns in selected datasets.")
                return

        final_df = pd.concat(combined_dfs, ignore_index=True)

        # Capture float columns before fillna (which can turn columns
        # containing NaN into object dtype) so sig_figs formatting below knows
        # which columns to round.
        float_cols = (list(final_df.select_dtypes(include='float').columns)
                      if sig_figs is not None else [])

        final_df = final_df.fillna('-')

        if output == 'df':
            return final_df

        if sig_figs is not None:
            def _to_sig(v):
                # Round to sig_figs significant figures, rendered as a plain
                # decimal string (never scientific notation). Non-floats (e.g.
                # the '-' fill value or string columns) pass through unchanged.
                if not isinstance(v, float) or not np.isfinite(v):
                    return v
                if v == 0:
                    return f"{0:.{sig_figs - 1}f}"
                digits = sig_figs - int(np.floor(np.log10(abs(v)))) - 1
                if digits <= 0:
                    return f"{round(v, digits):.0f}"
                return f"{v:.{digits}f}"

            for c in float_cols:
                final_df[c] = final_df[c].map(_to_sig)

        if output == 'md':
            try:
                return final_df.to_markdown(index=False)
            except ImportError:
                print("Markdown output requires the 'tabulate' package "
                      "(pip install tabulate).")
                return

        if self.darkmode:
            header_color = 'rgb(30, 30, 30)'
            cell_color = 'rgb(50, 50, 50)'
            font_color = 'white'
            line_color = 'rgb(70, 70, 70)'
        else:
            header_color = 'rgb(230, 230, 230)'
            cell_color = 'white'
            font_color = 'black'
            line_color = 'rgb(200, 200, 200)'

        # Bold header text via HTML — works in all Plotly versions
        header_values = [f"<b>{c}</b>" for c in final_df.columns]

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=header_values,
                fill_color=header_color,
                align='left',
                font=dict(color=font_color, size=12),     # <-- removed weight='bold'
                line_color=line_color
            ),
            cells=dict(
                values=[final_df[k].tolist() for k in final_df.columns],
                fill_color=cell_color,
                align='left',
                font=dict(color=font_color, size=11),
                line_color=line_color,
                height=25
            )
        )])

        layout_args = {
            'title': {'text': title or "Data Table", 'x': 0.5},
            'template': "plotly_dark" if self.darkmode else "plotly_white",
            'margin': dict(l=20, r=20, t=50, b=20),
        }
        fig.update_layout(**layout_args)

        self.last_fig = fig
        fig = self._apply_fonts(fig)

        if output == 'fig':
            return fig

        display_df = final_df.copy()
        for col in display_df.columns:
            if display_df[col].dtype in ['float64', 'float32']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.5g}" if isinstance(x, (int, float)) and x != '-' else x)

        header_size = self.table_header_size or 22
        cell_size = self.table_cell_size or 20
        title_size = self.suptitle_size or header_size + 4

        html_table = display_df.to_html(index=False, escape=False)
        if title:
            caption_html = (
                f'<caption style="caption-side:top;text-align:center;'
                f'font-weight:600;color:#000;font-size:{title_size}px;'
                f'padding:6px 8px;background-color:#e8e8e8;'
                f'border:1px solid #ccc;border-bottom:none;'
                f'font-family:-apple-system, BlinkMacSystemFont, \'Segoe UI\', Arial, sans-serif;">'
                f'{title}</caption>'
            )
            html_table = html_table.replace('<table', '<table', 1)
            html_table = html_table.replace('>', f'>{caption_html}', 1)

        styled_html = f"""
        <div style="margin-top:8px; margin-bottom:8px; overflow-x:auto;">
            {html_table}
        </div>
        <style>
        table {{
            border-collapse: collapse;
            width: auto;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
        }}
        th {{
            background-color: #e8e8e8;
            color: #000;
            font-weight: 600;
            font-size: {header_size}px;
            padding: 3px 8px;
            text-align: center;
            border: 1px solid #ccc;
        }}
        td {{
            color: #000;
            font-size: {cell_size}px;
            padding: 2px 8px;
            text-align: center;
            border: 1px solid #ddd;
        }}
        tr:nth-child(odd) td {{ background-color: #ffffff; }}
        tr:nth-child(even) td {{ background-color: #f3f3f3; }}
        tr:hover td {{ background-color: #ececec; }}
        </style>
        """

        display(HTML(styled_html))

    def save_png(self, filename="plot.png", scale=3, width=None, height=None):
        """
        Save the last generated plot to a PNG file.
        Requires the 'kaleido' package to be installed.
        """
        if self.last_fig is None:
            print("No plot to save. Please run .plot() first.")
            return

        try:
            self._suppress_mathjax()

            self.last_fig.write_image(filename, scale=scale, width=width, height=height)
            print(f"Plot saved to {filename}")

        except ValueError as e:
            print(f"Error saving image (ensure 'kaleido' is installed): {e}")
        except Exception as e:
            print(f"Error saving image: {e}")

    def list_sets(self):
        """
        Print a formatted table of all loaded datasets.
        """
        if not self.sets:
            print("No datasets loaded.")
            return

        rows = []
        for ds in self.sets:
            selected = "✓" if ds.select else "X"
            # Shape from cached row positions and own-column count — no
            # full-width materialization.
            shape = f"{len(ds._masked_positions())} x {len(ds._own_col_positions())}"
            query_info = str(ds.query)
            rows.append([
                f"Set {ds.index}",
                ds.title,
                selected,
                shape,
                query_info
            ])

        headers = ["Set", "Title", "Selected", "Shape", "Query?" ]
        col_widths = [
            max(len(row[i]) for row in [headers] + rows) + 2
            for i in range(len(headers))
        ]

        header_str = "".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        sep = "-" * sum(col_widths)

        output_lines = [header_str, sep]
        for row in rows:
            line = "".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))
            output_lines.append(line)

        print("\nLoaded Datasets:")
        print("\n".join(output_lines))

    def list_parms(self, set_number=None, search_string=None, use_regex=False):
        """
        List the parameters (columns) available in the loaded datasets.
        """
        import fnmatch
        
        if set_number is None:
            target_sets = self.selected()
            if not target_sets:
                target_sets = self.sets
        else:
            target_sets = self._get_uset_slice(set_number)
            
        if not target_sets:
            print("No datasets available to list parameters from.")
            return []
            
        all_cols = set()
        for ds in target_sets:
            all_cols.update(ds.df.columns)
            
        if search_string:
            try:
                if not use_regex:
                    if not any(c in search_string for c in ['*', '?', '[', ']']):
                        search_string = f"*{search_string}*"
                    pattern_str = fnmatch.translate(search_string)
                else:
                    pattern_str = search_string
                    
                pattern = re.compile(pattern_str, re.IGNORECASE)
                filtered_cols = [col for col in all_cols if pattern.search(str(col))]
                
            except re.error as e:
                print(f"Invalid search pattern '{search_string}': {e}")
                return []
        else:
            filtered_cols = list(all_cols)
            
        filtered_cols.sort(key=lambda x: str(x).lower())
        
        print(f"Found {len(filtered_cols)} parameters", end="")
        if search_string:
            print(f" matching '{search_string}'", end="")
        if set_number is not None:
            print(f" in set(s) {set_number}:")
        else:
            print(" in active datasets:")
            
        for col in filtered_cols:
            desc = self.parm_description_dict.get(col, "No description available.")
            print(f"  - {str(col).ljust(25)} : {desc}")
            
        return filtered_cols

    def summary(self, cols=None, print_table=False):
        """
        Build a statistical summary DataFrame. Optionally print a formatted table.
        """
        headers = ["Set", "Title", "Query", "Variable", "Count", "Min", "Mean", "Max", "Std"]

        if cols is None:
            y_part = self.last_y if isinstance(self.last_y, list) else [self.last_y] if self.last_y else []
            x_part = [self.last_x] if self.last_x else []
            target_cols = x_part + y_part
        else:
            target_cols = cols if isinstance(cols, list) else [cols]

        if not target_cols:
            if print_table:
                print("No columns specified and no previous plot variables defined.")
            return pd.DataFrame(columns=headers)

        active_ds = self.selected()
        if not active_ds:
            if print_table:
                print("No datasets selected. Cannot generate summary.")
            return pd.DataFrame(columns=headers)

        records = []
        for ds in active_ds:
            query_disp = str(ds.query) if ds.query else "-"
            ds_cols = ds.columns

            for col in target_cols:
                if col in ds_cols:
                    data = ds[col].dropna()

                    if data.empty:
                        records.append({
                            "Set": ds.index, "Title": ds.title, "Query": query_disp,
                            "Variable": col, "Count": 0,
                            "Min": np.nan, "Mean": np.nan, "Max": np.nan, "Std": np.nan,
                        })
                    elif pd.api.types.is_numeric_dtype(data):
                        records.append({
                            "Set": ds.index, "Title": ds.title, "Query": query_disp,
                            "Variable": col, "Count": len(data),
                            "Min": data.min(), "Mean": data.mean(),
                            "Max": data.max(), "Std": data.std(),
                        })
                    else:
                        records.append({
                            "Set": ds.index, "Title": ds.title, "Query": query_disp,
                            "Variable": col, "Count": len(data),
                            "Min": np.nan, "Mean": np.nan, "Max": np.nan, "Std": np.nan,
                        })

        df = pd.DataFrame(records, columns=headers)

        if print_table:
            if df.empty:
                print(f"None of the selected datasets contain the specified columns: {target_cols}")
                return df

            display_rows = []
            for _, r in df.iterrows():
                q = r["Query"]
                q_disp = (q[:27] + '...') if isinstance(q, str) and len(q) > 30 else q
                display_rows.append([
                    f"Set {r['Set']}",
                    str(r['Title'])[:20],
                    q_disp,
                    str(r['Variable'])[:15],
                    f"{int(r['Count'])}",
                    f"{r['Min']:.4g}" if pd.notna(r['Min']) else "-",
                    f"{r['Mean']:.4g}" if pd.notna(r['Mean']) else "-",
                    f"{r['Max']:.4g}" if pd.notna(r['Max']) else "-",
                    f"{r['Std']:.4g}" if pd.notna(r['Std']) else "-",
                ])

            col_widths = [max(len(str(item)) for item in col) + 2 for col in zip(*([headers] + display_rows))]
            header_str = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
            sep = "-" * sum(col_widths)

            print(f"\nStatistical Summary for: {', '.join(target_cols)}")
            print(header_str)
            print(sep)
            for row in display_rows:
                print("".join(str(val).ljust(w) for val, w in zip(row, col_widths)))

        return df

    # Method groupings for help(). A method left out of every list still shows,
    # under "Other" (help() fills that bucket by set-difference), so a newly
    # added method is never silently hidden; names here that no longer exist are
    # simply skipped.
    _HELP_CATEGORIES = [
        ("Loading & data",   ['load', 'load_df', 'load_clipboard', 'combine_sets',
                              'combine', 'add_column', 'set_column']),
        ("Selection",        ['select', 'selected', 'omit', 'query', 'restore',
                              'clear_data']),
        ("Plotting",         ['plot', 'plot_ymult', 'plot_type', 'bar', 'box',
                              'contour', 'histogram', 'line', 'highlight',
                              'save_png', 'dashboard']),
        ("Styling & format", ['color', 'marker', 'markersize', 'alpha', 'fill',
                              'linestyle', 'linewidth', 'edgewidth', 'hue',
                              'hue_palette', 'reg_order', 'set_color_palette',
                              'var_format', 'clear_var_format', 'list_var_formats',
                              'set_display_parms', 'set_title', 'set_default_format',
                              'reset_format', 'set_font_sizes', 'get_font_sizes',
                              'toggle_darkmode', 'scale', 'set_plot_size',
                              'set_static_images']),
        ("Analysis & stats", ['delta', 'table', 'table_read', 'summary', 'reg_info',
                              'min', 'max', 'mean', 'median']),
        ("Info",             ['list_sets', 'list_parms', 'refresh_own_columns',
                              'help']),
    ]

    def _help_sig(self, name):
        """Signature of a public method with ``self`` dropped (rendered off the
        bound method), falling back gracefully when inspect can't build one."""
        try:
            return str(inspect.signature(getattr(self, name)))
        except (ValueError, TypeError):
            return "(...)"

    def _help_method_line(self, name, func):
        """Two-line overview entry: signature + first docstring line."""
        doc = inspect.getdoc(func)
        preview = doc.split('\n')[0] if doc else "No description available."
        print(f"  • {name}{self._help_sig(name)}")
        print(f"      → {preview}")

    @staticmethod
    def _help_attr_line(name, val):
        s = str(val)
        if len(s) > 100:
            s = s[:100] + "..."
        return f"  • {name}: {type(val).__name__} = {s}"

    def _help_topic(self, key):
        """Detailed help for a single method or a category (see :meth:`help`)."""
        cls = self.__class__
        func = getattr(cls, key, None)
        if callable(func) and not key.startswith('_'):
            print("=" * 70)
            print(f"📖 {key}{self._help_sig(key)}")
            print("=" * 70)
            print(inspect.getdoc(func) or "No description available.")
            return
        for label, names in self._HELP_CATEGORIES:
            if key.lower() == label.lower():
                print(f"📂 {label}")
                print("-" * 70)
                for n in names:
                    f = getattr(cls, n, None)
                    if f is not None:
                        self._help_method_line(n, f)
                return
        import difflib
        public = [n for n, _ in inspect.getmembers(cls, inspect.isfunction)
                  if not n.startswith('_')]
        close = difflib.get_close_matches(key, public, n=5)
        print(f"No method or category named {key!r}.")
        print("Did you mean: " + ", ".join(close) + " ?" if close
              else "Call nb.help() for the full list.")

    def help(self, topic=None):
        """Show a categorized overview of the notebook API, or detailed help
        for a single method or category.

        Parameters
        ----------
        topic : str | None
            ``None`` (default) prints the categorized method and attribute
            overview. A method name (e.g. ``'delta'``) prints that method's full
            signature and docstring; a category name (e.g. ``'Plotting'``) lists
            just that group. An unknown topic suggests the closest matches.
        """
        if topic is not None:
            self._help_topic(str(topic).strip())
            return

        print("=" * 70)
        print("📚 UnichartNotebook HELP")
        print("=" * 70)

        cls = self.__class__
        doc = inspect.getdoc(cls)
        if doc:
            print("\n📋 CLASS DESCRIPTION:")
            print(doc)

        # Methods, grouped. Anything not mapped falls into "Other" via
        # set-difference so nothing is ever hidden.
        public = {n: f for n, f in inspect.getmembers(cls, inspect.isfunction)
                  if not n.startswith('_')}
        print("\n🔍 PUBLIC METHODS  (call nb.help('name') for full details):")
        print("-" * 70)
        shown = set()
        for label, names in self._HELP_CATEGORIES:
            entries = [n for n in names if n in public]
            if not entries:
                continue
            print(f"\n{label}")
            for n in entries:
                self._help_method_line(n, public[n])
                shown.add(n)
        leftover = sorted(set(public) - shown)
        if leftover:
            print("\nOther")
            for n in leftover:
                self._help_method_line(n, public[n])

        # Attributes, split into user-facing config vs internal plot memory
        # (the volatile ``last_*`` cache). The rule is prefix-based rather than a
        # fixed list, so a new attribute defaults to Config and stays visible.
        print("\n🛠️  ATTRIBUTES:")
        print("-" * 70)
        attrs = {a: v for a, v in self.__dict__.items() if not a.startswith('_')}
        if not attrs:
            print("No public instance attributes found.")
        else:
            config = {a: v for a, v in attrs.items() if not a.startswith('last_')}
            state = {a: v for a, v in attrs.items() if a.startswith('last_')}
            if config:
                print("\nConfig")
                for name in sorted(config):
                    print(self._help_attr_line(name, config[name]))
            if state:
                print("\nState (last-plot memory)")
                for name in sorted(state):
                    print(self._help_attr_line(name, state[name]))

        print("\n💡 QUICK START:")
        print("-" * 70)
        print("1. Load data:       nb.load_df(df, title='MyData')")
        print("2. Select datasets: nb.select([0, 1])")
        print("3. Plot:            nb.plot(x='time', y='value')")
        print("4. Multi-Y plot:    nb.plot_ymult(x='time', y=['Temp', 'Pressure'])")
        print("5. Variable format: nb.var_format('Temp', linestyle='--')")
        print("6. Method details:  nb.help('delta')")

        print("\n" + "=" * 70)

    # ------------------------------------------------------------------
    # Interactive "GUI" Replacement
    # ------------------------------------------------------------------
    def _clear_last_fig(self):
        """Drop the previous figure's trace data to free memory before the next plot."""
        if self.last_fig is not None:
            self.last_fig.data = []
            self.last_fig.layout = {}
            self.last_fig = None

    def _apply_decorations(self, fig, x_vars, y_vars, mode, calc_ncols, plot_items=None):
        """
        Apply stored lines and highlights to a figure.
        """
        x_list = x_vars if isinstance(x_vars, list) else ([x_vars] if x_vars else [])
        y_list = y_vars if isinstance(y_vars, list) else ([y_vars] if y_vars else [])

        for col_name, col_lines in self.lines.items():
            if col_name in x_list:
                if mode == 'vars' and plot_items:
                    for idx, (xi, yi) in enumerate(plot_items):
                        if xi == col_name:
                            r, c = (idx // calc_ncols) + 1, (idx % calc_ncols) + 1
                            xref, yref = _subplot_refs(r, c, calc_ncols)
                            for l in col_lines:
                                fig.add_shape(
                                    type='line', x0=l['level'], x1=l['level'], y0=0, y1=1,
                                    xref=xref, yref=f'{yref} domain',
                                    line=dict(color=l['color'], dash=l['dash'] or 'solid')
                                )
                else:
                    for l in col_lines:
                        fig.add_vline(x=l['level'], line_dash=l['dash'] or 'solid', line_color=l['color'])

            if col_name in y_list:
                if mode == 'vars' and plot_items:
                    for idx, (xi, yi) in enumerate(plot_items):
                        if yi == col_name:
                            r, c = (idx // calc_ncols) + 1, (idx % calc_ncols) + 1
                            xref, yref = _subplot_refs(r, c, calc_ncols)
                            for l in col_lines:
                                fig.add_shape(
                                    type='line', x0=0, x1=1, y0=l['level'], y1=l['level'],
                                    xref=f'{xref} domain', yref=yref,
                                    line=dict(color=l['color'], dash=l['dash'] or 'solid')
                                )
                else:
                    for l in col_lines:
                        fig.add_hline(y=l['level'], line_dash=l['dash'] or 'solid', line_color=l['color'])

        for col_name, hls in self.highlights.items():
            if col_name in x_list:
                if mode == 'vars' and plot_items:
                    for idx, (xi, yi) in enumerate(plot_items):
                        if xi == col_name:
                            r, c = (idx // calc_ncols) + 1, (idx % calc_ncols) + 1
                            xref, yref = _subplot_refs(r, c, calc_ncols)
                            for h in hls:
                                fig.add_shape(
                                    type='rect', x0=h['range'][0], x1=h['range'][1], y0=0, y1=1,
                                    xref=xref, yref=f'{yref} domain',
                                    fillcolor=h['color'], opacity=h['alpha'], layer='below', line_width=0
                                )
                else:
                    for h in hls:
                        fig.add_vrect(x0=h['range'][0], x1=h['range'][1], fillcolor=h['color'],
                                      opacity=h['alpha'], layer='below', line_width=0)

            if col_name in y_list:
                if mode == 'vars' and plot_items:
                    for idx, (xi, yi) in enumerate(plot_items):
                        if yi == col_name:
                            r, c = (idx // calc_ncols) + 1, (idx % calc_ncols) + 1
                            xref, yref = _subplot_refs(r, c, calc_ncols)
                            for h in hls:
                                fig.add_shape(
                                    type='rect', x0=0, x1=1, y0=h['range'][0], y1=h['range'][1],
                                    xref=f'{xref} domain', yref=yref,
                                    fillcolor=h['color'], opacity=h['alpha'], layer='below', line_width=0
                                )
                else:
                    for h in hls:
                        fig.add_hrect(y0=h['range'][0], y1=h['range'][1], fillcolor=h['color'],
                                      opacity=h['alpha'], layer='below', line_width=0)

        return fig