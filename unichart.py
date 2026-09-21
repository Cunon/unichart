import copy
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import warnings
import numbers
# Jupyter is optional: plain scripts and browser (Pyodide) environments import
# unichart without ipywidgets/IPython installed. `display` degrades to print;
# a shimmed IPython.display (e.g. the web terminal's) is used when present.
try:
    import ipywidgets as widgets
except ImportError:
    widgets = None
try:
    from IPython.display import display, clear_output, HTML
    _IPYTHON_DISPLAY, _FALLBACK_DISPLAY = display, None
except ImportError:
    def display(*objs, **kwargs):
        for obj in objs:
            print(getattr(obj, 'data', obj))
    _IPYTHON_DISPLAY, _FALLBACK_DISPLAY = None, display

    def clear_output(*args, **kwargs):
        pass

    class HTML:
        def __init__(self, data=''):
            self.data = data


def _display_renders():
    """Whether ``display(fig)`` would actually render rather than print a repr:
    a Jupyter kernel is live, or ``display`` has been rebound (the web
    terminal's capture shim). Lets optional displays — load_session's replay —
    stay quiet in plain scripts."""
    if display is _FALLBACK_DISPLAY:
        return False
    if display is _IPYTHON_DISPLAY:
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except ImportError:
            return False
    return True
import re
import inspect
from scipy.interpolate import griddata
import functools
import gc
import json
import os
import sys
import base64
import struct
import zlib
from datetime import datetime
from pathlib import Path

# -----------------------------------------------------------------------------
# Help coloring
# -----------------------------------------------------------------------------
# ``help()`` prints ANSI so one code path serves every front end: a real
# terminal renders it, the explorer's web terminal converts it to spans, and
# anything that can't gets plain text. The ~10 lines of gating are duplicated
# from unichart_cli rather than imported — the CLI depends on this module, not
# the other way round.
#
# _HELP_COLOR is the override the explorer terminal sets: it runs commands under
# redirect_stdout, so auto-detection there would always see "not a tty".
_HELP_COLOR = None                      # None = auto, True/False = forced

try:
    from IPython import get_ipython as _get_ipython
except ImportError:                     # plain script / Pyodide
    _get_ipython = None

_HELP_ANSI = {
    'head': '\033[1;33m',               # section headings
    'label': '\033[1m',                 # category / group labels
    'name': '\033[36m',                 # method and attribute names
    'sig': '\033[2m',                   # signatures, types, section underlines
    'lit': '\033[32m',                  # ``literals`` in a docstring
    'off': '\033[0m',
}

# Docstring structure, numpydoc-style: a heading is any line underlined with
# dashes, a parameter is `name : type` at the left margin, and ``...`` marks a
# literal. Painting is line-at-a-time and never nested — an inner reset would
# cut a heading's color short halfway through the line.
#
# A literal's backticks are dropped when it is painted: the color already says
# "this is code", and the markup is noise on screen. Uncolored output keeps
# them, so piped text stays the docstring as written.
_DOC_RULE_RE = re.compile(r'^-{3,}\s*$')
_DOC_PARAM_RE = re.compile(r'^([A-Za-z_*][\w, *]*) : (.+)$')
# Bounded so an unbalanced pair can't swallow half a docstring; [^`] spans
# newlines on purpose, because a literal often wraps across two lines.
_DOC_LITERAL_RE = re.compile(r'``[^`]{1,160}``')


def _color_docstring(text):
    """Paint a docstring's headings, parameter names/types and literals."""
    if not _help_color_on():
        return text
    def unmark(fragment):
        # Headings and parameter types carry their own color, so their
        # literals only need the markup dropped.
        return _DOC_LITERAL_RE.sub(lambda m: m.group(0)[2:-2], fragment)

    lines = text.split('\n')
    painted, body = [], []

    def flush():
        # Prose is painted a block at a time, not a line at a time: a literal
        # that wraps across two lines is one match only if the regex can see
        # both of them.
        if body:
            painted.append(_DOC_LITERAL_RE.sub(
                lambda m: _hc(m.group(0)[2:-2], 'lit'), '\n'.join(body)))
            body.clear()

    for i, line in enumerate(lines):
        following = lines[i + 1] if i + 1 < len(lines) else ''
        if line.strip() and _DOC_RULE_RE.match(following):
            flush()
            painted.append(_hc(unmark(line), 'head'))
        elif _DOC_RULE_RE.match(line):
            flush()
            painted.append(_hc(line, 'sig'))
        elif _DOC_PARAM_RE.match(line):
            flush()
            name, kind = _DOC_PARAM_RE.match(line).groups()
            painted.append(f"{_hc(name, 'name')} : {_hc(unmark(kind), 'sig')}")
        else:
            body.append(line)
    flush()
    return '\n'.join(painted)


def _help_color_on():
    """Whether ``help()`` should emit ANSI right now."""
    if _HELP_COLOR is not None:
        return bool(_HELP_COLOR)
    if os.environ.get('NO_COLOR'):      # set and non-empty, per no-color.org
        return False
    if os.environ.get('TERM') == 'dumb':
        return False
    try:
        if sys.stdout.isatty():
            return True
    except Exception:
        pass
    # A Jupyter kernel renders ANSI in stream output — IPython's own tracebacks
    # are colored the same way — so a notebook gets color even though its
    # stdout is not a tty. (_get_ipython is resolved at import: help() asks this
    # question once per painted fragment, a few hundred times a call.)
    if _get_ipython is None:
        return False
    try:
        shell = _get_ipython()
    except Exception:
        return False
    return shell is not None and type(shell).__name__ == 'ZMQInteractiveShell'


def _hc(text, key):
    """``text`` wrapped in the ANSI for ``key``, or unchanged when off."""
    if not _help_color_on():
        return text
    return f"{_HELP_ANSI[key]}{text}{_HELP_ANSI['off']}"


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

# Dash patterns Plotly accepts for gridlines (axis.griddash). Matplotlib-style
# aliases ('-', '--', ...) are also accepted via LINESTYLE_MAP_MPL_TO_PLOTLY.
GRID_DASH_OPTIONS = ('solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot')

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


# -----------------------------------------------------------------------------
# Plot styles (Plotly look vs. Matplotlib look)
# -----------------------------------------------------------------------------
# unichart draws through Plotly, whose house style is recognizable: no axis
# spines, a drawn zero line, its own color cycle and font. Figures are styled to
# approximate Matplotlib's rcParams defaults instead, so they sit next to
# Matplotlib output without reading as foreign. ``nb.set_plot_style('plotly')``
# opts back into Plotly's native look.
#
# It takes two halves, because a Plotly template can only carry *layout*
# defaults:
#   * layout    — the templates below (backgrounds, spines, ticks, grid, fonts),
#                 applied to the finished figure in UnichartNotebook._finalize.
#   * per-trace — color and markersize live on each Dataset and are written
#                 explicitly into every trace, so no template can reach them;
#                 set_plot_style swaps color_map / default_format instead.

PLOT_STYLES = ('matplotlib', 'plotly')

# What a fresh UnichartNotebook starts with, and what reset_format('defaults')
# returns to.
DEFAULT_PLOT_STYLE = 'matplotlib'

_PLOT_STYLE_ALIASES = {
    'matplotlib': 'matplotlib', 'mpl': 'matplotlib',
    'pyplot': 'matplotlib', 'plt': 'matplotlib',
    'plotly': 'plotly',
    # 'default'/'reset' track DEFAULT_PLOT_STYLE, so they keep meaning "however
    # unichart ships" rather than naming one particular look.
    'default': DEFAULT_PLOT_STYLE, 'reset': DEFAULT_PLOT_STYLE,
}

# Matplotlib's default property cycle (rcParams['axes.prop_cycle'] — "tab10").
# Spelled out rather than pulled from px.colors.qualitative so the values are
# readable here and don't ride on Plotly's palette naming.
MPL_COLOR_CYCLE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

# rcParams['font.family'] resolves to DejaVu Sans; the rest are fallbacks for
# machines without it (browsers render these figures, not Matplotlib).
MPL_FONT_FAMILY = 'DejaVu Sans, Verdana, Geneva, sans-serif'

# Matplotlib font sizes are points at the figure DPI; Plotly's are pixels.
# unichart maps one figsize inch to 100 px (see _base_layout), which is
# Matplotlib's own default DPI, so points convert as px = pt * 100 / 72.
_MPL_PT_TO_PX = 100.0 / 72.0


def _mpl_px(points):
    """Matplotlib points -> unichart pixels (at the 100 px/inch figure scale)."""
    return round(points * _MPL_PT_TO_PX, 1)


# rcParams sizes mapped onto unichart's font slots (see set_font_sizes):
# figure.titlesize and axes.titlesize are 'large' (12 pt), the rest 'medium'
# (10 pt) against font.size = 10. These are *style defaults* — an explicit
# set_font_sizes value always wins (resolved in _font_size).
MPL_FONT_SIZES = {
    'suptitle_size':      _mpl_px(12),
    'subplot_title_size': _mpl_px(12),
    'axes_title_size':    _mpl_px(10),
    'axes_tick_size':     _mpl_px(10),
    'legend_size':        _mpl_px(10),
    'footer_size':        _mpl_px(10),
    'colorbar_size':      _mpl_px(10),
    'hover_size':         _mpl_px(10),
}

# Per-dataset defaults the Matplotlib style overrides: rcParams['lines.markersize']
# is 6 pt (~8.3 px) against unichart's 10, and rcParams['image.cmap'] is viridis
# against unichart's Jet (contours and hue-colored scatters write the colorscale
# explicitly per trace, so no template can reach it). lines.linewidth (1.5 pt
# ~= 2.1 px) already rounds to unichart's default of 2, so linewidth is
# deliberately left alone rather than clobbered for nothing.
MPL_DATASET_FORMAT = {'markersize': _mpl_px(6), 'hue_palette': 'Viridis'}

_MPL_TEMPLATE = 'unichart_matplotlib'
_MPL_TEMPLATE_DARK = 'unichart_matplotlib_dark'


def _build_mpl_template(darkmode):
    """A Plotly template approximating Matplotlib's default rcParams (light) or
    its ``dark_background`` style (dark), layered over Plotly's own
    plotly_white / plotly_dark so unset attributes keep sane values.

    Faithful to Matplotlib: white/black figure and axes face colors, black/white
    spines on all four sides (axes.spines.* default True, axes.linewidth 0.8),
    outward ticks (xtick.direction 'out', major.size 3.5, width 0.8), *no* zero
    lines, the tab10 color cycle, viridis as the default colormap, DejaVu Sans,
    and a light-gray legend frame (legend.edgecolor '0.8').

    Deliberate departures:
      * Grid *visibility* is left to unichart's own ``grid=`` argument (default
        on) rather than Matplotlib's axes.grid=False, so the style can't fight
        an explicit per-plot choice; only its color/width is styled to match
        (linewidth 0.8, solid), at half the opacity of Matplotlib's grid.color
        '0.8' so the grid sits further behind the data. Override via nb.grid().
      * Dark mode keeps the tab10 cycle and dims the grid (a translucent gray
        rather than dark_background's white grid), with the pastel cycle also
        dropped.
    """
    tpl = copy.deepcopy(pio.templates['plotly_dark' if darkmode else 'plotly_white'])
    ink = 'white' if darkmode else 'black'          # axes.edgecolor / text.color
    face = 'black' if darkmode else 'white'         # figure/axes.facecolor
    # ~#b0b0b0 / #555555 at 50% alpha; translucency keeps the grid visually
    # behind dense data instead of competing with it.
    grid = 'rgba(85,85,85,0.5)' if darkmode else 'rgba(176,176,176,0.5)'
    viridis = list(pcolors.sequential.Viridis)      # rcParams['image.cmap']

    axis = dict(
        showline=True, linecolor=ink, linewidth=0.8, mirror=True,
        ticks='outside', tickcolor=ink, ticklen=3.5, tickwidth=0.8,
        zeroline=False,
        showgrid=True, gridcolor=grid, gridwidth=0.8, griddash='solid',
    )
    tpl.layout.update(
        font=dict(family=MPL_FONT_FAMILY, color=ink),
        paper_bgcolor=face,
        plot_bgcolor=face,
        colorway=MPL_COLOR_CYCLE,
        colorscale=dict(sequential=viridis, sequentialminus=viridis[::-1]),
        xaxis=axis,
        yaxis=axis,
        legend=dict(bgcolor=face, bordercolor='#cccccc', borderwidth=0.8),
    )
    return tpl


def _register_plot_style_templates():
    """Register the Matplotlib-flavored templates with Plotly (idempotent), so
    they can be referred to by name anywhere a template name is accepted."""
    pio.templates[_MPL_TEMPLATE] = _build_mpl_template(False)
    pio.templates[_MPL_TEMPLATE_DARK] = _build_mpl_template(True)


_register_plot_style_templates()


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


# Subplot spacing. Plotly's make_subplots defaults are fractions of the paper
# area that shrink with the grid (0.3/(rows-1) vertically, 0.2/(cols-1)
# horizontally): a third of the figure on a 2-row grid, too little to fit a
# subplot title plus tick labels on a 6-row one. unichart instead budgets a
# fixed number of pixels per gap and converts that to the fraction
# make_subplots wants, so a gap is the same physical size whatever the grid.
_SUBPLOT_HSPACE_PX = 80   # y tick labels + y title of the panel to the right
_SUBPLOT_VSPACE_PX = 70   # x tick labels + x title above, subplot title below
# Paper-area estimate when nothing is pinned: figsize minus the default
# margins (80 left/right; a title/legend band plus 80 at the bottom).
_SUBPLOT_MARGIN_W_PX = 160
_SUBPLOT_MARGIN_H_PX = 180
# Never let the gaps eat more than this share of the paper, so a crowded grid
# degrades to cramped panels rather than a make_subplots ValueError.
_SUBPLOT_MAX_GAP_SHARE = 0.6


def _parse_spacing(name, val):
    """Normalise an ``hspace``/``vspace`` value to ``('px', n)`` or
    ``('frac', f)``; ``None`` passes through. Numbers below 1 are a fraction
    of the plot area (Plotly's convention); 1 and above are pixels. Strings
    may carry a ``px`` suffix (``'60px'``)."""
    if val is None:
        return None
    unit = None
    if isinstance(val, str):
        txt = val.strip().lower()
        if txt.endswith('px'):
            txt, unit = txt[:-2].strip(), 'px'
        try:
            val = float(txt)
        except ValueError:
            raise ValueError(
                f"{name} must be a number (fraction < 1, or pixels >= 1) or a "
                f"string like '60px', got {val!r}") from None
    if isinstance(val, bool) or not isinstance(val, numbers.Real):
        raise TypeError(f"{name} must be numeric, got {type(val).__name__}")
    if val < 0:
        raise ValueError(f"{name} must be >= 0, got {val}")
    if unit == 'px' or val >= 1:
        return ('px', float(val))
    return ('frac', float(val))


def _subplot_spacing(nrows, ncols, figsize, hspace=None, vspace=None,
                     spacing_ref=None, h_px=None, v_px=None):
    """``make_subplots`` spacing kwargs for an ``nrows`` x ``ncols`` grid.

    ``hspace``/``vspace`` are the user values (see ``_parse_spacing``); a
    ``None`` falls back to the pixel budget ``h_px``/``v_px`` (a caller that
    needs extra room, e.g. for per-panel colorbars, raises it) or the module
    defaults. A pixel gap is converted against ``spacing_ref``, which says
    what the paper will be: ``{'panel': (w, h)}`` when ``set_plot_size`` pins
    each panel (the paper is then ``n`` panels plus ``n-1`` gaps),
    ``{'paper': (w, h)}`` when it pins the whole grid, or ``None`` to
    estimate from ``figsize``. Either entry may hold ``None`` for an unpinned
    dimension, which then also falls back to the ``figsize`` estimate.
    """
    ref = spacing_ref or {}
    fig_w = (figsize[0] * 100 if figsize else 1200) - _SUBPLOT_MARGIN_W_PX
    fig_h = (figsize[1] * 100 if figsize else 800) - _SUBPLOT_MARGIN_H_PX
    out = {}
    for key, n, val, default, ref_idx, fig_px in (
            ('horizontal_spacing', ncols, hspace, h_px or _SUBPLOT_HSPACE_PX, 0, fig_w),
            ('vertical_spacing', nrows, vspace, v_px or _SUBPLOT_VSPACE_PX, 1, fig_h)):
        if n < 2:
            continue
        kind, amount = _parse_spacing(key, val) or ('px', default)
        if kind == 'px':
            panel = (ref.get('panel') or (None, None))[ref_idx]
            paper = (ref.get('paper') or (None, None))[ref_idx]
            if panel is not None:
                # Plotly's spacing is per gap; the paper is n panels + n-1 gaps.
                frac = amount / (n * panel + (n - 1) * amount)
            else:
                frac = amount / max(paper or fig_px, 1.0)
        else:
            frac = amount
        cap = _SUBPLOT_MAX_GAP_SHARE / (n - 1)
        if frac > cap:
            warnings.warn(
                f"{key} of {frac:.3f} leaves too little room for {n} "
                f"{'columns' if ref_idx == 0 else 'rows'}; clamped to {cap:.3f}. "
                "Use a larger figsize, set_plot_size, or a smaller hspace/vspace.",
                UserWarning, stacklevel=3)
            frac = cap
        out[key] = frac
    return out

# Top-of-figure spacing estimates (px) used to reserve room for a (possibly
# multi-line) suptitle above a horizontal "above" legend so the legend can't
# grow up into the title. Per-line title height scales with the title font size:
# _base_layout assumes the default font, and _apply_fonts re-reserves the space
# once a custom suptitle size (set via set_font_sizes) is known.
_TITLE_TOP_PAD = 12             # gap above the first title line
_DEFAULT_TITLE_FONT_PX = 18     # approximates Plotly's default suptitle font
_TITLE_LINE_FACTOR = 1.45       # title line height = font size * this
_LEGEND_ROW_PX = 26             # space reserved per legend row
_LEGEND_GAP    = 12             # gap title→legend and legend→plot
# Extra top margin reserved only when set_plot_size is pinning the height:
# Plotly's rendered legend runs taller than _LEGEND_ROW_PX, and without the
# slack its autoexpand takes the difference out of the plot area.
_PINNED_TOP_SLACK = 12
_PINNED_ROW_SLACK = 4           # ...growing with each reserved legend row
# Legend height cap (px) that switches off Plotly's legend scrollbar. Left to
# itself Plotly caps a horizontal legend at 50% of the figure height and a side
# legend at the plot height, then scrolls — which an image can't do. See
# UnichartNotebook._fit_full_legend.
_FULL_LEGEND_MAXHEIGHT = 100_000
# Legend entry/column widths used only on the pinned path (see
# UnichartNotebook._legend_row_estimate). _legend_rows leans small on purpose —
# a little overlap is cheap when Plotly's autoexpand can absorb it — but here
# under-reserving shrinks the pinned plot area, so these are calibrated against
# the legend Plotly actually renders. A grouped legend's columns are narrower
# per character than a flowed entry but carry more fixed padding.
# Horizontal room (px) one extra right-hand y axis needs for its tick labels and
# its rotated title. Stacked extra axes were historically spaced by a fixed
# *fraction* of the plot area, so the room per axis shrank with the plot: at a
# small figsize — or under a set_plot_size pin — one axis's tick labels land on
# top of the next axis's title. 51px is exactly what a default-size figure
# already produces (and reads cleanly at), so the floor is invisible there and
# only bites where the plot has actually been squeezed.
_YAXIS_SLOT_PX = 51


def _yaxis_slot_fraction(default_fraction, plot_width_px):
    """Paper fraction to leave between stacked right-hand y axes.

    Widens the historical fraction only where it would fall below
    ``_YAXIS_SLOT_PX``, so wide figures keep exactly the layout they had and
    only the cramped ones move."""
    if not plot_width_px or plot_width_px <= 0:
        return default_fraction
    return max(default_fraction, _YAXIS_SLOT_PX / plot_width_px)


_PINNED_ENTRY_BASE_PX = 40
_PINNED_ENTRY_CHAR_PX = 10
_PINNED_GROUP_BASE_PX = 46
_PINNED_GROUP_CHAR_PX = 7
# Full-legend size estimates (UnichartNotebook._fit_full_legend), calibrated
# against legends Plotly rendered. A legend row is the legend font's line
# height (with a floor) plus padding, and every legend group is followed by
# Plotly's default tracegroupgap. Across a horizontal legend an entry takes its
# glyph and padding plus ~0.5 font-px per character, and a group column ~0.45
# font-px per character of its title, both including Plotly's itemgap; they
# lean wide, since over-estimating only leaves whitespace.
_FULL_ROW_LINE_FACTOR = 1.3
_FULL_ROW_MIN_PX = 16
_FULL_ROW_PAD_PX = 3
_FULL_GROUP_GAP_PX = 10
_FULL_LEGEND_PAD_PX = 2
_FULL_ENTRY_BASE_PX = 50
_FULL_ENTRY_CHAR_EM = 0.5
_FULL_GROUP_BASE_PX = 30
_FULL_GROUP_CHAR_EM = 0.45


def _flow_rows(widths, usable):
    """Greedy wrap: how many rows a run of items of these pixel widths takes
    at ``usable`` px per row. Shared by both legend-layout estimates."""
    rows, cur = 1, 0
    for w in widths:
        if cur + min(w, usable) > usable:
            rows, cur = rows + 1, min(w, usable)
        else:
            cur += min(w, usable)
    return rows
_LEGEND_ENTRY_BASE_PX = 34      # legend glyph + inter-entry padding
_LEGEND_CHAR_PX = 7             # approx px per character of entry text


def _legend_rows(names, fig_width_px):
    """Estimate how many rows a horizontal legend with these entry names wraps
    to at the given figure width. Plotly does the real wrapping in JS, so this
    is a greedy approximation that leans slightly generous — under-reserving
    makes the legend overlap the subplot titles, over-reserving just adds a
    little whitespace."""
    if not names:
        return 1
    usable = max(200, (fig_width_px or 1200) - 40)
    rows, cur = 1, 0
    for n in names:
        w = min(_LEGEND_ENTRY_BASE_PX + _LEGEND_CHAR_PX * len(str(n)), usable)
        if cur + w > usable:
            rows, cur = rows + 1, w
        else:
            cur += w
    return rows


def _title_lines(text):
    """Number of rendered lines in a Plotly title string. Counts <br> and
    treats a literal newline as a line break too (it is normalized to <br>
    elsewhere before rendering)."""
    if not text:
        return 1
    return text.replace('\n', '<br>').count('<br>') + 1


def _top_space(title_text, figsize, has_above_legend, title_font_size=None,
               legend_rows=1):
    """Geometry that reserves vertical space for a (possibly multi-line)
    suptitle so a horizontal "above" legend can't cover it.

    ``title_font_size`` is the suptitle font size in px; when ``None`` the
    Plotly default is assumed. Per-line height scales with it so a large custom
    title font still gets enough room. ``legend_rows`` is the number of rows
    the above-legend is expected to wrap to (see ``_legend_rows``); the top
    margin reserves space for all of them so a wrapping legend can't spill
    over the subplot titles.

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
        top_margin = title_band + max(1, legend_rows) * _LEGEND_ROW_PX + _LEGEND_GAP
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
    legend_rows = extra.pop('legend_rows', 1)
    has_above = (isinstance(incoming_legend, dict)
                 and incoming_legend.get('orientation') == 'h')

    top_margin, title_pos, legend_pos = _top_space(
        incoming_title.get('text'), figsize, has_above, legend_rows=legend_rows)

    title_defaults = {'x': 0.5, 'xanchor': 'center'}
    merged_title = {**title_defaults, **incoming_title, **title_pos}

    default_margin = {'t': top_margin}
    incoming_margin = extra.pop('margin', {})
    merged_margin = {**default_margin, **incoming_margin}

    args = {
        'template': "plotly_dark" if darkmode else "plotly_white",
        'title': merged_title,
        'margin': merged_margin,
        # Stashed so _apply_fonts can re-reserve the same legend rows when it
        # redoes the top-space math for a custom suptitle font.
        'meta': {'uc_legend_rows': legend_rows},
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

def _label_outer_axes(fig, n, nrows, ncols, xlabel=None, ylabel=None):
    """Axis titles for an ``n``-panel subplot grid without per-panel repetition,
    mirroring matplotlib's ``label_outer()``: the x title goes only on each
    column's bottom-most *occupied* panel (the last grid row may have empty
    trailing cells), and the y title only on the first column."""
    for c in range(1, ncols + 1):
        occupied = [(i // ncols) + 1 for i in range(n) if (i % ncols) + 1 == c]
        if occupied and xlabel:
            fig.update_xaxes(title_text=xlabel, row=max(occupied), col=c)
    if ylabel:
        for r in range(1, nrows + 1):
            if (r - 1) * ncols < n:     # row has at least one panel
                fig.update_yaxes(title_text=ylabel, row=r, col=1)


def _subplot_refs(row, col, ncols):
    """Return the (xref, yref) axis name strings for a subplot at (row, col) in an ncols grid."""
    idx = (row - 1) * ncols + col
    xref = 'x' if idx == 1 else f'x{idx}'
    yref = 'y' if idx == 1 else f'y{idx}'
    return xref, yref

# -----------------------------------------------------------------------------
# Reference-line labels (see UnichartNotebook.line)
# -----------------------------------------------------------------------------
# Labels are Plotly annotations tagged with _LINE_LABEL_NAME so _apply_fonts can
# tell them apart from subplot titles (which it restyles wholesale).
_LINE_LABEL_NAME = '_uc_line_label'
_LINE_LABEL_PAD = 4     # px gap between the text and the line / plot edge

# Position tokens, resolved at draw time because a column can be the x-var of one
# plot (vertical line) and the y-var of another (horizontal line). "Along" tokens
# slide the label along the line; "side" tokens pick which side of it the text
# sits on. The names mirror Plotly's own ``annotation_position`` vocabulary.
_LINE_LABEL_ALONG = {
    'vertical':   {'bottom': 0.0, 'middle': 0.5, 'center': 0.5, 'top': 1.0},
    'horizontal': {'left': 0.0, 'middle': 0.5, 'center': 0.5, 'right': 1.0},
}
_LINE_LABEL_SIDES = {
    'vertical':   {'left': 'left', 'right': 'right'},
    'horizontal': {'top': 'top', 'above': 'top', 'bottom': 'bottom', 'below': 'bottom'},
}
_LINE_LABEL_DEFAULT_SIDE = {'vertical': 'right', 'horizontal': 'top'}

# -----------------------------------------------------------------------------
# Watermarks (see UnichartNotebook.watermark)
# -----------------------------------------------------------------------------
# A watermark is a single Plotly layout image tagged with _WATERMARK_NAME so
# _apply_watermark can replace its own previous entry instead of stacking a new
# copy every time a figure is re-finalized.
_WATERMARK_NAME = '_uc_watermark'

# Built-in look, used for whatever the user hasn't set via nb.watermark().
# Deliberately faint and centered: the common case is a logo washed behind the
# data. 'contain' keeps the image's aspect ratio inside the size box, so the
# user never has to know the file's pixel dimensions.
_WATERMARK_DEFAULTS = {
    'source': None, 'opacity': 0.15, 'position': 'center',
    'size': 0.3, 'layer': 'below', 'sizing': 'contain',
}

# Position tokens -> (axis, value). 'c' means "center", which fills whichever
# axis the other tokens left free ('center' alone = dead center, 'center left'
# = vertically centered on the left edge). Mirrors the vocabulary of the
# reference-line labels above, plus the upper/lower aliases Matplotlib users type.
_WATERMARK_TOKENS = {
    'left': ('h', 'left'), 'right': ('h', 'right'),
    'top': ('v', 'top'), 'upper': ('v', 'top'),
    'bottom': ('v', 'bottom'), 'lower': ('v', 'bottom'),
    'center': ('c', None), 'centre': ('c', None), 'middle': ('c', None),
}
# Paper coordinate + matching anchor for each resolved token, so a corner
# watermark sits flush inside the plot area rather than half outside it.
_WATERMARK_H = {'left': (0.0, 'left'), 'center': (0.5, 'center'), 'right': (1.0, 'right')}
_WATERMARK_V = {'bottom': (0.0, 'bottom'), 'middle': (0.5, 'middle'), 'top': (1.0, 'top')}

_WATERMARK_MIME = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                   '.gif': 'image/gif', '.webp': 'image/webp', '.bmp': 'image/bmp',
                   '.svg': 'image/svg+xml'}
# Leading bytes -> mime, for raw-bytes sources where there's no filename to read.
_WATERMARK_MAGIC = ((b'\x89PNG\r\n\x1a\n', 'image/png'), (b'\xff\xd8\xff', 'image/jpeg'),
                    (b'GIF87a', 'image/gif'), (b'GIF89a', 'image/gif'),
                    (b'BM', 'image/bmp'), (b'<svg', 'image/svg+xml'),
                    (b'<?xml', 'image/svg+xml'))


def _parse_watermark_position(position):
    """Resolve a watermark ``position`` into ``(x, y, xanchor, yanchor)`` in
    paper coordinates (0-1 across the plot area).

    ``position`` is either a string of space-separated tokens ('center',
    'bottom right', 'top', 'center left', ...) or an explicit ``(x, y)`` pair
    of paper coordinates, which is anchored on the image's own center so the
    numbers name where the watermark's middle lands.
    """
    if isinstance(position, (list, tuple)):
        if len(position) != 2:
            raise ValueError("position tuple must be (x, y) in paper coords "
                             f"(0-1), got {len(position)} values")
        x, y = position
        for name, val in (('x', x), ('y', y)):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"position {name} must be numeric (paper coords), "
                                f"got {type(val).__name__}")
        return float(x), float(y), 'center', 'middle'

    if not isinstance(position, str):
        raise TypeError("position must be a position string or an (x, y) pair, "
                        f"got {type(position).__name__}")

    horiz = vert = None
    centers = 0
    for tok in position.replace(',', ' ').lower().split():
        entry = _WATERMARK_TOKENS.get(tok)
        if entry is None:
            valid = ', '.join(sorted(_WATERMARK_TOKENS))
            raise ValueError(f"position: unknown token {tok!r}. Valid tokens: "
                             f"{valid} (or an (x, y) pair in paper coords).")
        axis, val = entry
        if axis == 'h':
            horiz = val
        elif axis == 'v':
            vert = val
        else:
            centers += 1

    # 'center' fills the axes the explicit tokens left open; on its own (or with
    # nothing at all) that means dead center.
    if centers or (horiz is None and vert is None):
        if horiz is None:
            horiz = 'center'
        if vert is None:
            vert = 'middle'
    horiz = horiz or 'center'
    vert = vert or 'middle'

    x, xanchor = _WATERMARK_H[horiz]
    y, yanchor = _WATERMARK_V[vert]
    return x, y, xanchor, yanchor


def _watermark_data_uri(source):
    """Normalize a watermark ``source`` into something Plotly can actually draw.

    Plotly only renders ``layout.images`` sources it can fetch: a URL or a data
    URI. A bare file path fails *silently* — blank in the browser and blank in
    kaleido, with no exception — so local files are read here and inlined as
    base64 data URIs. Inlining is also what keeps the watermark in the PNGs
    produced by ``save_png``, ``set_static_images`` and the '⧉ copy' button,
    none of which can resolve an external reference.

    Accepts a path (str / Path), an existing ``data:`` URI or ``http(s)`` URL
    (both passed through untouched), raw image bytes, or a PIL Image.
    """
    if source is None:
        return None

    # PIL Image: re-encode as PNG so transparency survives.
    if hasattr(source, 'save') and hasattr(source, 'mode'):
        import io
        buf = io.BytesIO()
        source.save(buf, format='PNG')
        source = buf.getvalue()

    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
        if raw[:4] == b'RIFF' and raw[8:12] == b'WEBP':
            mime = 'image/webp'      # bare 'RIFF' also matches WAV/AVI
        else:
            mime = next((m for magic, m in _WATERMARK_MAGIC if raw.startswith(magic)),
                        'image/png')
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

    if isinstance(source, os.PathLike):
        source = os.fspath(source)
    if not isinstance(source, str):
        raise TypeError("watermark source must be a file path, URL, data URI, "
                        f"bytes or a PIL Image, got {type(source).__name__}")

    text = source.strip()
    if text.startswith('data:') or text.startswith(('http://', 'https://')):
        return text                     # already fetchable by the browser

    path = os.path.expanduser(text)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"watermark image not found: {source!r}. Pass a readable file path, "
            "an http(s) URL, a data: URI, image bytes, or a PIL Image.")
    ext = os.path.splitext(path)[1].lower()
    if ext not in _WATERMARK_MIME:
        valid = ', '.join(sorted(_WATERMARK_MIME))
        raise ValueError(f"unsupported watermark image type {ext or '(none)'!r}. "
                         f"Supported extensions: {valid}.")
    with open(path, 'rb') as fh:
        data = fh.read()
    return f"data:{_WATERMARK_MIME[ext]};base64,{base64.b64encode(data).decode('ascii')}"



def _parse_line_label_position(position, orientation):
    """Resolve a ``label_position`` spec into ``(along_fraction, side)``.

    ``position`` is either a number (0-1 along the line) or a string of
    space-separated tokens, which may mix an along-token or bare number with a
    side-token — e.g. ``'top'``, ``'bottom left'``, ``0.25``, ``'0.25 left'``.
    ``orientation`` is 'vertical' or 'horizontal' and decides how each token
    reads: 'top' slides a vertical line's label but flips a horizontal one's to
    the upper side of the line.
    """
    along_map = _LINE_LABEL_ALONG[orientation]
    side_map = _LINE_LABEL_SIDES[orientation]
    along = side = None

    if position is None:
        tokens = []
    elif isinstance(position, bool):
        raise TypeError("label_position must be a number (0-1) or a position string, got bool")
    elif isinstance(position, (int, float)):
        along, tokens = float(position), []
    elif isinstance(position, str):
        tokens = position.replace(',', ' ').lower().split()
    elif isinstance(position, (list, tuple)):
        tokens = [str(t).lower() for t in position]
    else:
        raise TypeError("label_position must be a number (0-1) or a position string, "
                        f"got {type(position).__name__}")

    for tok in tokens:
        try:
            along = float(tok)
            continue
        except ValueError:
            pass
        if tok in along_map:
            along = along_map[tok]
        elif tok in side_map:
            side = side_map[tok]
        else:
            valid = ', '.join(sorted(set(along_map) | set(side_map)))
            raise ValueError(f"label_position: unknown token '{tok}' for a {orientation} "
                             f"reference line. Valid tokens: {valid} (or a 0-1 number).")

    if along is None:
        along = 1.0                                   # default: far end of the line
    if not 0.0 <= along <= 1.0:
        raise ValueError(f"label_position fraction must be between 0 and 1, got {along}")
    return along, side or _LINE_LABEL_DEFAULT_SIDE[orientation]


def _line_label_annotation(line_spec, orientation, default_size=None):
    """Build the annotation kwargs for a reference line's label, or None.

    Coordinates come back in the system Plotly uses for axis-spanning shapes:
    the along-line coordinate is a 0-1 axis-*domain* fraction and the
    across-line coordinate is the line's data ``level``. Callers pair that with
    the matching refs ('x' + 'y domain' for a vertical line, 'x domain' + 'y'
    for a horizontal one).
    """
    text = line_spec.get('label')
    if not text:
        return None

    along, side = _parse_line_label_position(line_spec.get('label_position'), orientation)
    level = line_spec['level']
    size = line_spec.get('label_size') or default_size
    color = line_spec.get('label_color') or line_spec.get('color')

    if orientation == 'vertical':
        # Slide along y (domain); the side picks which flank of the line the text
        # sits on. At the ends, anchor inward so the text stays in the plot area.
        anchor_along = 'top' if along > 0.9 else 'bottom' if along < 0.1 else 'middle'
        shift_along = -_LINE_LABEL_PAD if along > 0.9 else _LINE_LABEL_PAD if along < 0.1 else 0
        anns = dict(x=level, y=along,
                    xanchor='left' if side == 'right' else 'right',
                    yanchor=anchor_along,
                    xshift=_LINE_LABEL_PAD if side == 'right' else -_LINE_LABEL_PAD,
                    yshift=shift_along)
    else:
        anchor_along = 'right' if along > 0.9 else 'left' if along < 0.1 else 'center'
        shift_along = -_LINE_LABEL_PAD if along > 0.9 else _LINE_LABEL_PAD if along < 0.1 else 0
        anns = dict(x=along, y=level,
                    xanchor=anchor_along,
                    yanchor='bottom' if side == 'top' else 'top',
                    xshift=shift_along,
                    yshift=_LINE_LABEL_PAD if side == 'top' else -_LINE_LABEL_PAD)

    font = {k: v for k, v in (('size', size), ('color', color)) if v is not None}
    anns.update(text=str(text), showarrow=False, name=_LINE_LABEL_NAME, font=font)
    return anns


def _prefix_annotation(annotation):
    """Turn ``_line_label_annotation`` output into ``annotation_*`` kwargs for
    ``add_vline``/``add_hline``. Empty dict when there is no label.

    Plotly derives an annotation's placement from the shape unless a property is
    already set, so every key here survives verbatim; the shape's own refs give
    us 'x' + 'y domain' (vline) / 'x domain' + 'y' (hline), which is the
    coordinate system ``_line_label_annotation`` returns.
    """
    return {f'annotation_{k}': v for k, v in (annotation or {}).items()}

# -----------------------------------------------------------------------------
# Variable Format Resolver (used by multi-y plot)
# -----------------------------------------------------------------------------
_VAR_FORMAT_KEYS = ('color', 'marker', 'linestyle', 'markersize', 'linewidth', 'alpha',
                    'alpha_marker', 'alpha_line', 'style')

# CSS named colors (the set Plotly accepts), for ``_color_with_alpha``. Plotly
# only validates names; it ships no name -> value table.
_CSS_NAMED_COLORS = dict(kv.split(':') for kv in (
    "aliceblue:f0f8ff antiquewhite:faebd7 aqua:00ffff aquamarine:7fffd4 azure:f0ffff "
    "beige:f5f5dc bisque:ffe4c4 black:000000 blanchedalmond:ffebcd blue:0000ff "
    "blueviolet:8a2be2 brown:a52a2a burlywood:deb887 cadetblue:5f9ea0 chartreuse:7fff00 "
    "chocolate:d2691e coral:ff7f50 cornflowerblue:6495ed cornsilk:fff8dc crimson:dc143c "
    "cyan:00ffff darkblue:00008b darkcyan:008b8b darkgoldenrod:b8860b darkgray:a9a9a9 "
    "darkgreen:006400 darkgrey:a9a9a9 darkkhaki:bdb76b darkmagenta:8b008b "
    "darkolivegreen:556b2f darkorange:ff8c00 darkorchid:9932cc darkred:8b0000 "
    "darksalmon:e9967a darkseagreen:8fbc8f darkslateblue:483d8b darkslategray:2f4f4f "
    "darkslategrey:2f4f4f darkturquoise:00ced1 darkviolet:9400d3 deeppink:ff1493 "
    "deepskyblue:00bfff dimgray:696969 dimgrey:696969 dodgerblue:1e90ff firebrick:b22222 "
    "floralwhite:fffaf0 forestgreen:228b22 fuchsia:ff00ff gainsboro:dcdcdc ghostwhite:f8f8ff "
    "gold:ffd700 goldenrod:daa520 gray:808080 green:008000 greenyellow:adff2f grey:808080 "
    "honeydew:f0fff0 hotpink:ff69b4 indianred:cd5c5c indigo:4b0082 ivory:fffff0 khaki:f0e68c "
    "lavender:e6e6fa lavenderblush:fff0f5 lawngreen:7cfc00 lemonchiffon:fffacd "
    "lightblue:add8e6 lightcoral:f08080 lightcyan:e0ffff lightgoldenrodyellow:fafad2 "
    "lightgray:d3d3d3 lightgreen:90ee90 lightgrey:d3d3d3 lightpink:ffb6c1 lightsalmon:ffa07a "
    "lightseagreen:20b2aa lightskyblue:87cefa lightslategray:778899 lightslategrey:778899 "
    "lightsteelblue:b0c4de lightyellow:ffffe0 lime:00ff00 limegreen:32cd32 linen:faf0e6 "
    "magenta:ff00ff maroon:800000 mediumaquamarine:66cdaa mediumblue:0000cd "
    "mediumorchid:ba55d3 mediumpurple:9370db mediumseagreen:3cb371 mediumslateblue:7b68ee "
    "mediumspringgreen:00fa9a mediumturquoise:48d1cc mediumvioletred:c71585 "
    "midnightblue:191970 mintcream:f5fffa mistyrose:ffe4e1 moccasin:ffe4b5 navajowhite:ffdead "
    "navy:000080 oldlace:fdf5e6 olive:808000 olivedrab:6b8e23 orange:ffa500 orangered:ff4500 "
    "orchid:da70d6 palegoldenrod:eee8aa palegreen:98fb98 paleturquoise:afeeee "
    "palevioletred:db7093 papayawhip:ffefd5 peachpuff:ffdab9 peru:cd853f pink:ffc0cb "
    "plum:dda0dd powderblue:b0e0e6 purple:800080 rebeccapurple:663399 red:ff0000 "
    "rosybrown:bc8f8f royalblue:4169e1 saddlebrown:8b4513 salmon:fa8072 sandybrown:f4a460 "
    "seagreen:2e8b57 seashell:fff5ee sienna:a0522d silver:c0c0c0 skyblue:87ceeb "
    "slateblue:6a5acd slategray:708090 slategrey:708090 snow:fffafa springgreen:00ff7f "
    "steelblue:4682b4 tan:d2b48c teal:008080 thistle:d8bfd8 tomato:ff6347 turquoise:40e0d0 "
    "violet:ee82ee wheat:f5deb3 white:ffffff whitesmoke:f5f5f5 yellow:ffff00 "
    "yellowgreen:9acd32"
).split())


def _color_with_alpha(color, alpha):
    """Return ``color`` as an ``rgba(...)`` string carrying ``alpha``.

    Plotly has no ``line.opacity``, so a line-only opacity has to ride in the
    line color itself. Accepts hex (``#rgb``, ``#rrggbb``, ``#rrggbbaa``),
    ``rgb()``/``rgba()`` strings (an existing alpha is replaced), and CSS
    named colors. Anything unrecognized is returned unchanged.
    """
    if not isinstance(color, str):
        return color
    c = color.strip().lower()
    rgb = None
    if c in _CSS_NAMED_COLORS:
        c = '#' + _CSS_NAMED_COLORS[c]
    if c.startswith('#'):
        h = c[1:]
        if len(h) in (3, 4):
            h = ''.join(ch * 2 for ch in h)
        if len(h) in (6, 8) and all(ch in '0123456789abcdef' for ch in h):
            rgb = tuple(str(int(h[i:i + 2], 16)) for i in (0, 2, 4))
    elif c.startswith(('rgb(', 'rgba(')) and c.endswith(')'):
        parts = [x.strip() for x in c[c.index('(') + 1:-1].split(',')]
        if len(parts) >= 3:
            rgb = tuple(parts[:3])
    if rgb is None:
        return color
    return f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:g})'


def _split_alpha(fmt, color, marker_dict, line_dict):
    """Resolve ``alpha`` / ``alpha_marker`` / ``alpha_line`` for one trace.

    ``alpha`` is the trace-level opacity and covers everything. When
    ``alpha_marker`` or ``alpha_line`` is set, the trace opacity drops to 1
    and each part carries its own value instead — markers through
    ``marker.opacity``, the line through an rgba color — with the part that
    has no value of its own keeping ``alpha``. Mutates ``marker_dict`` and
    ``line_dict`` in place and returns the trace-level opacity to use.
    """
    alpha = fmt.get('alpha')
    alpha = 1 if alpha is None else alpha
    a_m, a_l = fmt.get('alpha_marker'), fmt.get('alpha_line')
    if a_m is None and a_l is None:
        return alpha
    marker_dict['opacity'] = alpha if a_m is None else a_m
    line_dict['color'] = _color_with_alpha(color, alpha if a_l is None else a_l)
    return 1

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
    # Marker-only / line-only opacity; None means "same as alpha".
    'alpha_marker': None,
    'alpha_line':    None,
    'fill':       True,
    # Precision used when a value is *displayed* — table/summary cells, plot
    # hover readouts. Two spellings of one knob, so at most one is ever set:
    # sig_figs counts significant figures, decimals counts places after the
    # point. Both None = every display keeps its own built-in precision
    # (.5g table cells, .4g statistics, .2g/.4g/.6g hover).
    'sig_figs':   None,
    'decimals':   None,
    # Colorscale for contours and hue-colored scatters. Lives here (rather than
    # only on Dataset) so a plot style can swap it — see MPL_DATASET_FORMAT.
    'hue_palette': 'Jet',
}

# Per-dataset *styling* a derived set copies from the set it was built from
# (see ``UnichartNotebook._inherit_set_format``). How a series is drawn — plus
# ``sig_figs``/``decimals``, how precisely its values are written out:
# the attributes ``_reset_set_attrs`` restores, minus the two that are not
# styling. ``reg_order`` is left out because a trendline is analysis, not
# appearance — a delta should not silently acquire the study set's fit — and
# ``plot_type`` because the repo treats it as structural (which is why the bulk
# reset sweep skips it too).
_INHERITED_FORMAT_ATTRS = (
    'color', 'marker', 'linestyle', 'markersize', 'linewidth', 'edgewidth',
    'alpha', 'alpha_marker', 'alpha_line', 'edge_color', 'fill', 'hue',
    'hue_palette', 'hue_order', 'style', 'display_parms', 'zorder',
    'sig_figs', 'decimals',
)

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
        'alpha_marker': var_fmt.get('alpha_marker', getattr(dataset, 'alpha_marker', None)),
        'alpha_line':    var_fmt.get('alpha_line',    getattr(dataset, 'alpha_line', None)),
        'edge_color': getattr(dataset, 'edge_color', 'black'),
        'edgewidth':  getattr(dataset, 'edgewidth', 1),
        'fill':       getattr(dataset, 'fill', True),
    }


_STYLE_BY_LINESTYLES = ['-', '--', '-.', ':']

_STYLE_BY_ALIASES = {
    'color': 'color', 'colors': 'color',
    'marker': 'marker', 'markers': 'marker',
    'linestyle': 'linestyle', 'linestyles': 'linestyle',
    'line': 'linestyle', 'ls': 'linestyle',
}


def _style_by_formats(y_list, style_by, variable_formats=None,
                      color_cycle=None, marker_cycle=None):
    """
    Expand ``style_by`` into per-variable formats, auto-cycling the requested
    attribute(s) by the variable's position in ``y_list``.

    ``style_by`` is one or more of 'color' / 'marker' / 'linestyle', given as
    a string ('marker', 'color+linestyle', 'color, marker') or a list. Any
    attribute already present in ``variable_formats`` for a variable wins over
    the auto-cycled value, so explicit ``var_format`` calls behave as usual.

    ``color_cycle`` / ``marker_cycle`` are the sequences to cycle — callers pass
    the notebook's ``color_map`` / ``marker_map`` so the auto-styling follows the
    active palette (and therefore the active plot style). Omitting them falls
    back to the built-in defaults, which is what those maps hold anyway.
    """
    variable_formats = variable_formats or {}
    if not style_by:
        return variable_formats
    tokens = (style_by.replace('+', ' ').replace(',', ' ').split()
              if isinstance(style_by, str) else list(style_by))
    attrs = []
    for t in tokens:
        key = _STYLE_BY_ALIASES.get(str(t).lower())
        if key is None:
            raise ValueError(
                f"style_by accepts 'color', 'marker', 'linestyle', or a "
                f"combination like 'color+marker'; got {t!r}")
        if key not in attrs:
            attrs.append(key)
    color_cycle = list(color_cycle) if color_cycle else px.colors.qualitative.Plotly
    marker_cycle = (list(marker_cycle) if marker_cycle
                    else list(MARKER_MAP_MPL_TO_PLOTLY.keys()))
    merged = dict(variable_formats)
    for idx, var in enumerate(y_list):
        auto = {}
        if 'color' in attrs:
            auto['color'] = color_cycle[idx % len(color_cycle)]
        if 'marker' in attrs:
            auto['marker'] = marker_cycle[idx % len(marker_cycle)]
        if 'linestyle' in attrs:
            auto['linestyle'] = _STYLE_BY_LINESTYLES[idx % len(_STYLE_BY_LINESTYLES)]
        auto.update(variable_formats.get(var, {}))
        merged[var] = auto
    return merged


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


def _union_ranges(ranges):
    """Union a list of (min, max) axis-range tuples into one spanning range.

    ``None`` entries are ignored. Returns ``None`` when nothing is supplied,
    the sole tuple unchanged when only one contributes (preserving a
    deliberately reversed axis), else the (min, max) spanning all endpoints.

    Used so a ``scale()`` set on a bar-overlay column (marker/tick/whisker),
    which shares the bar's y-axis, widens that axis to keep both the bars and
    the overlay in frame instead of being silently dropped."""
    ranges = [r for r in ranges if r is not None]
    if not ranges:
        return None
    if len(ranges) == 1:
        return tuple(ranges[0])
    pts = [p for r in ranges for p in r]
    return (min(pts), max(pts))

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


def _compact_indices(indices):
    """Render set indices compactly, collapsing runs: [0, 1, 2, 5] -> '0-2, 5'."""
    runs = []
    for i in sorted(indices):
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


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
        self._assign(key, value)

    def _assign(self, key, value, reapply=True):
        """Body of ``__setitem__``. ``reapply=False`` skips the notebook-wide
        query refresh so a caller writing several sets in one logical operation
        can refresh once at the end instead of once per set."""
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

        # Overwriting a column this set already owns changes values its
        # source file cannot reproduce.
        if key in self._dataset._own_cols:
            self._dataset._mark_source_modified()
        cdf.loc[mask, key] = assign_val
        # Writing through the view claims the column for this set — including
        # the case of filling NaNs into a column another set introduced.
        self._dataset._own_cols.add(key)
        if reapply:
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
        # "index: title" — matches how legends/trace names identify a set.
        self.title_format = f"{index}: {self.title}"

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
        self.alpha_marker = fmt.get('alpha_marker', None)
        self.alpha_line = fmt.get('alpha_line', None)
        self.hue = ""
        self.hue_palette = fmt.get('hue_palette', 'Jet')
        self.hue_order = None
        self.reg_order = None
        self.style = None
        self.linewidth = fmt.get('linewidth', 2)
        self.edgewidth = fmt.get('edgewidth', 1)
        # Mutually exclusive, and each setter clears the other, so the raw
        # attributes must exist before either is assigned.
        self._sig_figs = None
        self._decimals = None
        self.sig_figs = fmt.get('sig_figs', None)
        self.decimals = fmt.get('decimals', None)
        self.set_type = 1
        self.data_type = 'discrete'
        self.delta_sets = None
        # Data provenance, populated by the load paths. ``file_path`` is the
        # absolute source file (None for in-memory data); ``_source`` carries
        # everything needed to re-create this set from its source again
        # (read_kwargs, group split column/key, and the column set at load
        # time so save_session can tell when derived columns were added).
        # Cleared by df replacement, which severs the link to the source.
        self.file_path = None
        self._source = None
        self._display_parms = coerce_display_parms(display_parms)
        self._plot_type = 'scatter'
        self._order = None
        self._zorder = 0

    def _mark_source_modified(self):
        """Record that a column this set loaded from its source has since been
        overwritten. The source file no longer reproduces the set's values, so
        save_session must embed the rows instead of writing a file reference
        (added/removed columns are detected separately, via ``load_cols``)."""
        if self._source is not None:
            self._source['modified'] = True

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
    def zorder(self):
        """Draw order: sets are plotted in ascending ``zorder`` (default 0),
        later traces on top; ties keep load order."""
        return self._zorder

    @zorder.setter
    def zorder(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid zorder (expected a number): {value!r}")
        self._zorder = value

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
    def sig_figs(self): return self._sig_figs

    @sig_figs.setter
    def sig_figs(self, value):
        # None = unset: every display keeps its own built-in precision.
        if value is None or (isinstance(value, int) and not isinstance(value, bool)
                             and value >= 1):
            self._sig_figs = value
            if value is not None:
                self._decimals = None     # the two are alternatives
        else:
            raise ValueError(
                f"Invalid sig_figs: {value!r} (expected a positive integer or None)")

    @property
    def decimals(self): return self._decimals

    @decimals.setter
    def decimals(self, value):
        # The fixed-places spelling of sig_figs; 0 means whole numbers.
        if value is None or (isinstance(value, int) and not isinstance(value, bool)
                             and value >= 0):
            self._decimals = value
            if value is not None:
                self._sig_figs = None     # the two are alternatives
        else:
            raise ValueError(
                f"Invalid decimals: {value!r} (expected a non-negative integer or None)")

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
            'alpha_marker': self.alpha_marker,
            'alpha_line': self.alpha_line,
            'hue': self.hue,
            'hue_palette': self.hue_palette,
            'hue_order': self.hue_order,
            'reg_order': self.reg_order,
            'zorder': self.zorder,
            'index': self.index,
            'style': self.style,
            'display_parms': self.display_parms,
            'plot_type': self.plot_type,
            'linewidth': self.linewidth,
            'edgewidth': self.edgewidth,
            'sig_figs': self.sig_figs,
            'decimals': self.decimals,
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


def _hover_fmt(sig_figs, default, decimals=None):
    """d3 number format for a hover readout: ``.<n>g`` when the dataset carries
    a ``sig_figs``, ``.<n>f`` when it carries ``decimals`` instead, else the
    call site's own built-in format."""
    if sig_figs:
        return f".{sig_figs}g"
    if decimals is not None:
        return f".{decimals}f"
    return default


def build_hover_data(df, parms, sig_figs=None, decimals=None):
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
    * floats use general precision (``.6g``, or the dataset's ``sig_figs`` /
      ``decimals`` when it has one);
    * anything else (categories, strings, objects) is shown as-is;
    * missing values render blank instead of ``NaN`` / ``None``.

    Parms missing from ``df`` are skipped. Building each column independently
    also avoids the mixed-dtype ``to_numpy()`` collapse that previously turned
    timestamps into opaque objects.
    """
    float_fmt = _hover_fmt(sig_figs, '.6g', decimals)
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
            lines.append(f"<br>{parm}: %{{customdata[{i}]:{float_fmt}}}")
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
def _in_draw_order(datasets):
    """Datasets stable-sorted into draw order: ascending ``zorder`` (default
    0), ties keeping load order — so ``zorder=1`` lifts one set above the
    zorder-0 rest and negative values push a set behind. Plotly renders
    later-added traces on top, so the overlaid plot builders iterate this
    order; the per-dataset subplot variants keep panel order instead. Legend
    order is kept by set index via per-set ``legendrank`` stamps at the
    trace-add sites."""
    return sorted(datasets, key=lambda d: getattr(d, 'zorder', 0))


def _xy_pairs(x, y):
    """Pair up ``x`` and ``y`` (each a name or a list) the way the grid plots
    do: equal-length lists zip, a single value broadcasts against the list."""
    x_list = x if isinstance(x, list) else [x]
    y_list = y if isinstance(y, list) else [y]
    if len(x_list) == len(y_list):
        return list(zip(x_list, y_list))
    if len(x_list) == 1:
        return [(x_list[0], yi) for yi in y_list]
    if len(y_list) == 1:
        return [(xi, y_list[0]) for xi in x_list]
    raise ValueError(
        f"x and y must be the same length, or one must be a single value. "
        f"Got len(x)={len(x_list)}, len(y)={len(y_list)}."
    )


def _numeric_hue_info(list_of_datasets, hue, axis_limits):
    """The numeric hue columns among the selected datasets, each assigned its
    own layout coloraxis: ``{hue_col: {'ca_name', 'palette', 'lim'}}``. A
    dataset's own ``hue`` wins over the call-level one; categorical hues are
    left out (they color through a per-trace colorscale, with no colorbar)."""
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
    return numeric_hue_info


def _apply_hue_coloraxes(fig, numeric_hue_info):
    """Lay out one colorbar per numeric hue found by ``_numeric_hue_info``,
    stacked to the right of the plot area."""
    coloraxis_updates = {}
    for idx, (hue_col, info) in enumerate(numeric_hue_info.items()):
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


def _marker_line_style(fmt, color, marker, markersize, linewidth, linestyle,
                       hue, df, numeric_hue_info):
    """``marker`` / ``line`` dicts plus trace opacity for one dataset's scatter
    trace, from its format dict and the already-resolved call-level values.
    Hue, when active and present in ``df``, drives the per-point marker color
    and takes precedence over the set color / fill toggle."""
    marker_dict = dict(
        size=markersize, symbol=get_plotly_marker(marker),
        line=dict(width=fmt.get('edgewidth', 1), color=fmt.get('edge_color', 'black')),
    )
    line_dict = dict(width=linewidth, dash=get_plotly_linestyle(linestyle))
    if hue and hue in df.columns:
        hue_data = df[hue]
        if pd.api.types.is_numeric_dtype(hue_data):
            info = numeric_hue_info.get(hue, {})
            marker_dict['color'] = hue_data
            marker_dict['coloraxis'] = info.get('ca_name', 'coloraxis')
        else:
            hue_series = hue_data.astype('category')
            marker_dict['color'] = hue_series.cat.codes
            marker_dict['colorscale'] = fmt.get('hue_palette', 'Jet')
            marker_dict['showscale'] = False
    elif fmt.get('fill', True):
        marker_dict['color'] = color
    else:
        # No fill: hollow marker whose outline takes the set color.
        marker_dict['color'] = 'rgba(0,0,0,0)'
        marker_dict['line'] = dict(width=fmt.get('edgewidth', 1), color=color)
    line_dict['color'] = color
    opacity = _split_alpha(fmt, color, marker_dict, line_dict)
    return marker_dict, line_dict, opacity


def uniplot(list_of_datasets, x, y, z=None, plot_type=None, color=None, hue=None, marker=None,
            markersize=10, marker_edge_color="black", linestyle=None, hue_palette="Jet",
            hue_order=None, line=False, suppress_msg=False, return_axes=False, axes=None,
            suptitle=None, xlabel=None, ylabel=None, subplot_titles=None,
            darkmode=False, interactive=True, display_parms=None, grid=True,
            legend='above', legend_ncols=1, figsize=(12, 8), ncols=None, nrows=None, x_lim=None, y_lim=None,
            axis_limits=None, hspace=None, vspace=None, spacing_ref=None):

    axis_limits = axis_limits or {}
    x_list = x if isinstance(x, list) else [x]
    y_list = y if isinstance(y, list) else [y]
    pairs = _xy_pairs(x, y)

    n_plots = len(pairs)
    if n_plots == 0: raise ValueError("At least one x/y pair is required.")

    nrows, ncols = _calc_grid(n_plots, nrows, ncols)

    numeric_hue_info = _numeric_hue_info(list_of_datasets, hue, axis_limits)

    # Match unicontour's r=100 per colorbar: 90 leaves the colorbar's tick
    # labels just wide enough to trip Plotly's margin autoexpand, which then
    # eats into the plot area and breaks a set_plot_size width pin.
    right_margin = max(80, len(numeric_hue_info) * 100)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles, shared_xaxes=False,
                        **_subplot_spacing(nrows, ncols, figsize, hspace, vspace, spacing_ref))
    fig.update_layout(**_base_layout(
        darkmode, None, figsize,
        title={'text': suptitle or (f"{x} vs {[str(yi) for yi in y_list]}" if len(x_list) == 1 else f"{x_list} vs {y_list}"), 'x': 0.5, 'xanchor': 'center'},
        showlegend=(legend != 'off'),
        margin=dict(r=right_margin),
        **({'legend': dict(orientation="h")} if legend == 'above' else {}),
    ))

    for dataset in _in_draw_order(list_of_datasets):
        if not dataset.select: continue

        fmt = dataset.get_format_dict()
        cur_title = fmt.get('title')
        cur_hue = fmt.get('hue') or hue
        cur_color = color or fmt.get('color')
        cur_marker = marker or fmt.get('marker')
        cur_linestyle = linestyle or fmt.get('linestyle')
        cur_markersize = fmt.get('markersize', markersize)
        cur_linewidth = fmt.get('linewidth', 2)
        cur_reg_order = fmt.get('reg_order')
        cur_idx = fmt.get('index')
        cur_sig = fmt.get('sig_figs')
        cur_dec = fmt.get('decimals')
        num_fmt = _hover_fmt(cur_sig, '.2f', cur_dec)
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
            custom_data, hover_lines = build_hover_data(df, valid_hover,
                                                        cur_sig, cur_dec)

            ht = (f"<b><u>Set: {cur_idx}</u></b><br><b>{cur_title}</b><br>"
                  f"{x_col}: %{{x:{num_fmt}}}<br>{yi}: %{{y:{num_fmt}}}")
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

            marker_dict, line_dict, cur_opacity = _marker_line_style(
                fmt, cur_color, cur_marker, cur_markersize, cur_linewidth,
                cur_linestyle, cur_hue, df, numeric_hue_info)

            fig.add_trace(_scatter_cls(len(df))(
                x=df[x_col], y=df[yi], mode=mode,
                name=f"{cur_idx}: {cur_title}",
                legendgroup=f"group_{cur_idx}",
                legendrank=1000 + cur_idx,
                marker=marker_dict, line=line_dict,
                opacity=cur_opacity,
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
                        line=dict(color=line_dict['color'], width=cur_linewidth,
                                  dash=get_plotly_linestyle(cur_linestyle)),
                        opacity=cur_opacity, hoverinfo='skip', showlegend=False
                    ), row=row, col=col)

    _apply_hue_coloraxes(fig, numeric_hue_info)

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
                        axis_limits=None, return_axes=False,
                        hspace=None, vspace=None, spacing_ref=None):

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
        # A secondary y axis hangs its tick labels and title off the right of
        # each panel, so the column gap has to hold those too.
        **_subplot_spacing(nrows, ncols, figsize, hspace, vspace, spacing_ref,
                           h_px=150 if use_secondary else None),
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

        ds_sig, ds_dec = dataset.sig_figs, dataset.decimals
        num_fmt = _hover_fmt(ds_sig, '.2f', ds_dec)
        hover_cd, hover_lines = build_hover_data(df, valid_hover, ds_sig, ds_dec)
        hover_suffix = "".join(hover_lines)

        if primary_y in df.columns:
            color0 = color_cycle[0]
            ht = (f"<b>{dataset.title}</b><br>{x_col}: %{{x:{num_fmt}}}"
                  f"<br>{primary_y}: %{{y:{num_fmt}}}{hover_suffix}<extra></extra>")
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
            ht = (f"<b>{dataset.title}</b><br>{x_col}: %{{x:{num_fmt}}}"
                  f"<br>{yi}: %{{y:{num_fmt}}}{hover_suffix}<extra></extra>")
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
           barmode='group', color=None, agg='mean', categorical_x=True,
           suptitle=None, xlabel=None, ylabel=None, subplot_titles=None,
           darkmode=False, figsize=(12, 8), ncols=None, nrows=None,
           y_lim=None, return_axes=False, hspace=None, vspace=None, spacing_ref=None):
    """
    Grouped Bar Chart. Subplots are organized by Y-variable.

    Each dataset's rows are reduced to one value per ``x`` category with
    ``agg`` (see ``_resolve_agg``; ``agg=False`` draws the rows as they are),
    for the bar columns and the overlay columns alike. Hover text reports
    ``<agg> <column>: <value> (n=<rows>)``. ``categorical_x`` renders the
    x-axis as categories (evenly spaced bars even for numeric x).

    ``markers`` overlay columns pair positionally with the y variables — the
    i-th marker column draws on the i-th y variable's subplot, attached to
    that variable's bars (extras fall back to the first subplot). Pairing
    keeps a high-valued limit column from stretching the y-scale of
    unrelated subplots.

    The x-axis title appears only on each column's bottom-most panel. By
    default each panel names its own variable on the y-axis (no subplot
    titles); passing ``ylabel`` reverts to a single shared y title on the
    first column, and ``subplot_titles`` restores per-panel titles.
    """
    y_list = y if isinstance(y, list) else [y]
    markers_list = markers if isinstance(markers, list) else ([markers] if markers else [])
    variable_formats = variable_formats or {}
    agg_func, agg_name = _resolve_agg(agg)
    n_y = len(y_list)
    nrows, ncols = _calc_grid(n_y, nrows, ncols)
    active_ds = _in_draw_order(d for d in list_of_datasets if d.select)

    # Expected legend entries, for row estimation: one bar entry per set, one
    # figure-wide entry per styled marker column, one per (set, column) for
    # unstyled marker columns. Mirrors the legend bookkeeping below.
    legend_names = [f"{d.index}: {d.title}" for d in active_ds]
    for m_col in markers_list:
        if variable_formats.get(m_col):
            legend_names.append(m_col)
        else:
            legend_names += [f"{d.index}: {d.title} — {m_col}" for d in active_ds]

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles,
                        **_subplot_spacing(nrows, ncols, figsize, hspace, vspace, spacing_ref))
    # scattermode='group' makes the marker/tick/whisker overlays honor their
    # offsetgroup so they sit over their own bar instead of the category
    # center. Only valid when bars themselves are offset (barmode='group').
    fig.update_layout(**_base_layout(
        darkmode, suptitle or f"Bar Comparison: {x}", figsize,
        barmode=barmode, showlegend=True,
        legend=dict(orientation="h"),
        legend_rows=_legend_rows(legend_names, figsize[0] * 100 if figsize else None),
        **({'scattermode': 'group'} if barmode == 'group' else {})
    ))

    edge_default = 'white' if darkmode else 'black'

    # Legend bookkeeping. An entry appears with the FIRST trace actually added
    # (not blindly the first subplot, which loses the entry when that panel
    # happens to lack the column):
    #   bars               -> one entry per set
    #   styled markers     -> ONE figure-wide entry per column, named the column
    #   unstyled markers   -> one entry per (set, column), grouped with the set
    # legendrank keeps sets contiguous with styled overlays trailing, however
    # the traces interleave.
    bar_legend_shown = set()
    marker_legend_shown = set()

    for ds in active_ds:
        if x not in ds.columns: continue
        df = ds.cols([c for c in dict.fromkeys([x] + y_list + markers_list)
                      if c in ds.columns])
        df, n_df = _bar_frame(df, x, list(df.columns), agg_func)
        offset_group = f"set_{ds.index}"

        for idx_y, yi in enumerate(y_list):
            row, col = (idx_y // ncols) + 1, (idx_y % ncols) + 1
            if yi not in df.columns: continue

            show_bar = ds.index not in bar_legend_shown
            if show_bar:
                bar_legend_shown.add(ds.index)
            fig.add_trace(go.Bar(
                x=df[x], y=df[yi],
                name=f"{ds.index}: {ds.title}",
                legendgroup=f"group_{ds.index}",
                legendrank=1000 + ds.index,
                offsetgroup=offset_group,
                alignmentgroup="bars",
                marker_color=ds.color if not color else color,
                opacity=ds.alpha,
                showlegend=show_bar,
                customdata=None if n_df is None else n_df[yi],
                hovertemplate=(f"<b>{ds.title}</b><br>{x}: %{{x}}<br>"
                               f"{_agg_hover(agg_name, yi, n_df, ds.sig_figs, ds.decimals)}"
                               f"<extra></extra>")
            ), row=row, col=col)

        # Overlay columns: positional pairing (see docstring). Each column is
        # drawn once per set, on its paired subplot only.
        for m_idx, m_col in enumerate(markers_list):
            if m_col not in df.columns: continue
            anchor_idx = m_idx if m_idx < n_y else 0
            anchor_y = y_list[anchor_idx]
            if anchor_y not in df.columns: continue
            row, col = (anchor_idx // ncols) + 1, (anchor_idx % ncols) + 1

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

            if styled:
                legend_key = m_col
                trace_name = m_col
                legend_group = f"marker_{m_col}"
                legend_rank = 2000 + m_idx
            else:
                legend_key = (ds.index, m_col)
                trace_name = f"{ds.index}: {ds.title} — {m_col}"
                legend_group = f"group_{ds.index}"
                legend_rank = 1000 + ds.index
            show = legend_key not in marker_legend_shown
            if show:
                marker_legend_shown.add(legend_key)

            fig.add_trace(go.Scatter(
                x=df[x], y=df[m_col],
                mode='markers',
                name=trace_name,
                legendgroup=legend_group,
                legendrank=legend_rank,
                offsetgroup=offset_group,
                alignmentgroup="bars",
                **_overlay_marker_kw(m_style, m_symbol, m_color, m_size,
                                     m_alpha, edge_default,
                                     values=df[m_col], bar_values=df[anchor_y],
                                     stem_dash=m_dash, stem_width=m_lw),
                showlegend=show,
                customdata=None if n_df is None else n_df[m_col],
                hovertemplate=(f"<b>{ds.title}</b><br>{x}: %{{x}}<br>"
                               f"{_agg_hover(agg_name, m_col, n_df, ds.sig_figs, ds.decimals)}"
                               f"<extra></extra>")
            ), row=row, col=col)

            # Dashed stems can't ride on the marker trace's error bars
            # (always solid), so they get a companion line trace tied to
            # the same legend group.
            if m_style == 'whisker' and m_dash and m_dash != 'solid':
                fig.add_trace(go.Scatter(
                    name=f"{trace_name} (stem)",
                    legendgroup=legend_group,
                    legendrank=legend_rank,
                    offsetgroup=offset_group,
                    alignmentgroup="bars",
                    **_whisker_stem_kw(df[x], df[m_col], df[anchor_y],
                                       m_color, m_alpha, m_dash, m_lw),
                ), row=row, col=col)

    _label_outer_axes(fig, n_y, nrows, ncols, xlabel or x, ylabel)
    if not ylabel:
        # No shared ylabel: each panel names its own variable on the y-axis,
        # replacing the subplot titles that used to carry that information.
        for idx_y, yi in enumerate(y_list):
            fig.update_yaxes(title_text=yi,
                             row=(idx_y // ncols) + 1, col=(idx_y % ncols) + 1)
    if y_lim: fig.update_yaxes(range=y_lim)
    if categorical_x: fig.update_xaxes(type='category')

    return _show_or_return(fig, return_axes)


def unibar_per_dataset(list_of_datasets, x, y, markers=None, variable_formats=None,
                       barmode='group', agg='mean', categorical_x=True,
                       suptitle=None, xlabel=None, ylabel=None, subplot_titles=None,
                       figsize=(12, 8), ncols=None, nrows=None,
                       darkmode=False, y_lim=None, return_axes=False,
                       hspace=None, vspace=None, spacing_ref=None):
    """
    Grouped Bar Chart. Subplots are organized by Dataset.

    Rows are reduced to one value per ``x`` category with ``agg`` exactly as
    in ``unibar`` (``agg=False`` for raw rows); hover reports the reducer and
    the row count. ``categorical_x`` renders the x-axis as categories.

    variable_formats applies to BOTH bar variables and marker columns in
    this view, since color encodes variable (not dataset) within each subplot.

    ``markers`` overlay columns pair positionally with the y variables: a
    tick/whisker glyph for the i-th marker column sits on (and, for
    'whisker', stems to) the i-th y variable's bar in each group. Marker
    columns beyond the number of y variables attach to the first y variable.

    The x-axis title appears only on each column's bottom-most panel;
    ``ylabel``, when given, only on the first column.
    """
    active_ds = [d for d in list_of_datasets if d.select]
    y_list = y if isinstance(y, list) else [y]
    markers_list = markers if isinstance(markers, list) else ([markers] if markers else [])
    variable_formats = variable_formats or {}
    agg_func, agg_name = _resolve_agg(agg)
    n_sets = len(active_ds)
    nrows, ncols = _calc_grid(n_sets, nrows, ncols)

    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=subplot_titles or [d.title_format for d in active_ds])
    color_cycle = px.colors.qualitative.Plotly
    fig.update_layout(**_base_layout(
        darkmode, suptitle or "Dataset Bar Comparison", figsize,
        barmode=barmode,
        legend=dict(orientation="h"),
        legend_rows=_legend_rows(y_list + markers_list,
                                 figsize[0] * 100 if figsize else None),
        **({'scattermode': 'group'} if barmode == 'group' else {})
    ))

    edge_default = 'white' if darkmode else 'black'

    # One legend entry per variable / marker column, shown with the first
    # trace actually added (not blindly the first subplot, which loses the
    # entry when that particular set lacks the column). legendrank keeps bar
    # variables first and overlays trailing regardless of which set showed them.
    bar_legend_shown = set()
    marker_legend_shown = set()

    for idx_ds, ds in enumerate(active_ds):
        row, col = (idx_ds // ncols) + 1, (idx_ds % ncols) + 1
        if x not in ds.columns: continue
        df = ds.cols([c for c in dict.fromkeys([x] + y_list + markers_list)
                      if c in ds.columns])
        df, n_df = _bar_frame(df, x, list(df.columns), agg_func)

        for idx_y, yi in enumerate(y_list):
            if yi not in df.columns: continue
            offset_group = f"var_{yi}"

            var_fmt = variable_formats.get(yi, {})
            bar_color = var_fmt.get('color') or color_cycle[idx_y % len(color_cycle)]
            bar_alpha = var_fmt.get('alpha', 1.0)

            show_bar = yi not in bar_legend_shown
            if show_bar:
                bar_legend_shown.add(yi)
            fig.add_trace(go.Bar(
                x=df[x], y=df[yi],
                name=yi,
                legendgroup=yi,
                legendrank=1000 + idx_y,
                offsetgroup=offset_group,
                alignmentgroup="bars",
                marker_color=bar_color,
                opacity=bar_alpha,
                showlegend=show_bar,
                customdata=None if n_df is None else n_df[yi],
                hovertemplate=(f"<b>{yi}</b><br>{x}: %{{x}}<br>"
                               f"{_agg_hover(agg_name, yi, n_df, ds.sig_figs, ds.decimals)}"
                               f"<extra></extra>")
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

            # Positional pairing, as in unibar / unibar_datasets_as_x: the i-th
            # overlay column attaches its tick/whisker glyph to the i-th y
            # variable's bars (extras fall back to the first y). Classic
            # markers stay at the category center. If this set lacks the
            # paired bar column the glyph is drawn unattached (no stem) rather
            # than on some other variable's bar.
            anchor = y_list[m_idx if m_idx < len(y_list) else 0] if y_list else None
            attach = m_style in ('tick', 'whisker') and anchor in df.columns
            group_kw = ({'offsetgroup': f"var_{anchor}", 'alignmentgroup': "bars"}
                        if attach else {})

            show_marker = m_col not in marker_legend_shown
            if show_marker:
                marker_legend_shown.add(m_col)
            fig.add_trace(go.Scatter(
                x=df[x], y=df[m_col],
                mode='markers',
                name=m_col,
                legendgroup=f"marker_{m_col}",
                legendrank=2000 + m_idx,
                **group_kw,
                **_overlay_marker_kw(m_style, m_symbol, m_color, m_size,
                                     m_alpha, edge_default,
                                     values=df[m_col],
                                     bar_values=df[anchor] if attach else None,
                                     stem_dash=m_dash, stem_width=m_lw),
                showlegend=show_marker,
                customdata=None if n_df is None else n_df[m_col],
                hovertemplate=(f"<b>{m_col}</b><br>{x}: %{{x}}<br>"
                               f"{_agg_hover(agg_name, m_col, n_df, ds.sig_figs, ds.decimals)}"
                               f"<extra></extra>")
            ), row=row, col=col)

            if m_style == 'whisker' and attach and m_dash and m_dash != 'solid':
                fig.add_trace(go.Scatter(
                    name=f"{m_col} (stem)",
                    legendgroup=f"marker_{m_col}",
                    legendrank=2000 + m_idx,
                    **group_kw,
                    **_whisker_stem_kw(df[x], df[m_col], df[anchor],
                                       m_color, m_alpha, m_dash, m_lw),
                ), row=row, col=col)

    _label_outer_axes(fig, n_sets, nrows, ncols, xlabel or x, ylabel)
    if y_lim: fig.update_yaxes(range=y_lim)
    if categorical_x: fig.update_xaxes(type='category')

    return _show_or_return(fig, return_axes)

def unibox(list_of_datasets, x, y, boxmode='group', points='outliers', notched=False,
           color=None, suptitle=None, xlabel=None, ylabel=None, subplot_titles=None,
           darkmode=False, figsize=(12, 8), ncols=None, nrows=None, 
           y_lim=None, return_axes=False, hspace=None, vspace=None, spacing_ref=None):
    """
    Boxplot version of uniplot.
    Subplots are organized by Y-variables.

    The x-axis title appears only on each column's bottom-most panel. By
    default each panel names its own variable on the y-axis (no subplot
    titles); passing ``ylabel`` reverts to a single shared y title on the
    first column, and ``subplot_titles`` restores per-panel titles.
    """
    y_list = y if isinstance(y, list) else [y]
    n_y = len(y_list)
    nrows, ncols = _calc_grid(n_y, nrows, ncols)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles,
                        **_subplot_spacing(nrows, ncols, figsize, hspace, vspace, spacing_ref))
    fig.update_layout(**_base_layout(
        darkmode, suptitle or f"Boxplot Comparison: {x}", figsize,
        boxmode=boxmode, showlegend=True,
        margin=dict(r=80),
        legend=dict(orientation="h"),
    ))

    for ds in _in_draw_order(list_of_datasets):
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
                legendrank=1000 + ds.index,
                marker_color=ds.color if not color else color,
                opacity=ds.alpha,
                boxpoints=points,
                notched=notched,
                line=dict(width=ds.linewidth),
                showlegend=(idx_y == 0)
            ), row=row, col=col)

    _label_outer_axes(fig, n_y, nrows, ncols, xlabel or x, ylabel)
    if not ylabel:
        # No shared ylabel: each panel names its own variable on the y-axis,
        # replacing the subplot titles that used to carry that information.
        for idx_y, yi in enumerate(y_list):
            fig.update_yaxes(title_text=yi,
                             row=(idx_y // ncols) + 1, col=(idx_y % ncols) + 1)
    if y_lim: fig.update_yaxes(range=y_lim)

    return _show_or_return(fig, return_axes)

def unibox_per_dataset(list_of_datasets, x, y, boxmode='group', points='outliers', notched=False,
                       suptitle=None, figsize=(12, 8), ncols=None, nrows=None, 
                       darkmode=False, y_lim=None, return_axes=False,
                       hspace=None, vspace=None, spacing_ref=None):
    """
    Boxplot version of uniplot_per_dataset.
    Subplots are organized by Dataset.
    """
    active_ds = [d for d in list_of_datasets if d.select]
    y_list = y if isinstance(y, list) else [y]
    n_sets = len(active_ds)
    nrows, ncols = _calc_grid(n_sets, nrows, ncols)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[d.title_format for d in active_ds],
                        **_subplot_spacing(nrows, ncols, figsize, hspace, vspace, spacing_ref))
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
                 opacity=None, hspace=None, vspace=None, spacing_ref=None):
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

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles or x_list,
                        **_subplot_spacing(nrows, ncols, figsize, hspace, vspace, spacing_ref))
    fig.update_layout(**_base_layout(
        darkmode, suptitle or "Distribution Comparison", figsize,
        barmode=barmode, showlegend=True,
        legend=dict(orientation="h")
    ))

    xbins = _build_xbins(bin_size, bin_start, bin_end)

    for ds in _in_draw_order(list_of_datasets):
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
                legendrank=1000 + ds.index,
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
                            opacity=None, hspace=None, vspace=None, spacing_ref=None):
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

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[d.title_format for d in active_ds],
                        **_subplot_spacing(nrows, ncols, figsize, hspace, vspace, spacing_ref))
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

# -----------------------------------------------------------------------------
# Marginal distribution plots (see UnichartNotebook.plot_marginal)
# -----------------------------------------------------------------------------
# A scatter of y against x with the distribution of each variable drawn in a
# strip hugging the matching axis: x's above the plot, y's to its right. Each
# (x, y) pair — or each dataset, for the per-dataset variant — occupies one
# 2x2 *block* of subplot cells:
#
#         ┌─────────────┬────┐
#         │  x marginal │    │   <- top-right cell is empty (axes hidden)
#         ├─────────────┼────┤
#         │             │ y  │
#         │    main     │ m. │
#         │             │    │
#         └─────────────┴────┘
#
# Blocks tile a regular grid sized by _calc_grid, with the usual inter-block
# gaps from _subplot_spacing; the domains inside a block are laid out by hand
# so the marginals sit a few px off the main panel instead of a full gap away.
_MARGINAL_KINDS = ('histogram', 'box', 'violin', 'rug', 'kde')
_MARGINAL_GAP_PX = 8        # main panel -> marginal strip, per side
_MARGINAL_KDE_POINTS = 200  # evaluation grid for the 'kde' kind
_MARGINAL_RUG_SIZE = 10     # px height of a rug tick


def _resolve_marginal_kinds(marginal, marginal_x, marginal_y):
    """Resolve the ``marginal`` / ``marginal_x`` / ``marginal_y`` trio to one
    kind per side, ``None`` meaning no strip on that side. A side's own value
    wins; ``None`` inherits ``marginal``; ``False`` switches the side off."""
    def one(side_val):
        val = marginal if side_val is None else side_val
        if val is None or val is False:
            return None
        if val not in _MARGINAL_KINDS:
            raise ValueError(
                f"marginal must be one of {_MARGINAL_KINDS} or False, got {val!r}")
        return val
    return one(marginal_x), one(marginal_y)


def _marginal_block_cells(block_idx, ncols_blocks):
    """``(main, top, right, corner)`` subplot ``(row, col)`` cells of a block
    in the ``2*nrows x 2*ncols_blocks`` cell grid that the blocks tile."""
    br, bc = divmod(block_idx, ncols_blocks)
    r0, c0 = 2 * br + 1, 2 * bc + 1
    return (r0 + 1, c0), (r0, c0), (r0 + 1, c0 + 1), (r0, c0 + 1)


def _marginal_block_refs(block_idx, ncols_blocks):
    """``(main, top, right)`` ``(xref, yref)`` axis names for a block — what
    ``_apply_decorations`` needs to draw reference lines on each panel."""
    main, top, right, _ = _marginal_block_cells(block_idx, ncols_blocks)
    n_cell_cols = 2 * ncols_blocks
    return tuple(_subplot_refs(r, c, n_cell_cols) for r, c in (main, top, right))


def _marginal_gap_fraction(n_blocks, main_share, figsize, spacing_ref, axis_idx):
    """Paper fraction of ``_MARGINAL_GAP_PX`` along one axis (0 = width,
    1 = height), against the same size reference ``_subplot_spacing`` uses:
    a ``set_plot_size`` panel pin (the main panel, which is ``main_share`` of
    its block), a paper pin, or the ``figsize`` estimate."""
    ref = spacing_ref or {}
    fig_px = ((figsize[axis_idx] * 100 if figsize else (1200, 800)[axis_idx])
              - (_SUBPLOT_MARGIN_W_PX, _SUBPLOT_MARGIN_H_PX)[axis_idx])
    panel = (ref.get('panel') or (None, None))[axis_idx]
    paper = (ref.get('paper') or (None, None))[axis_idx]
    if panel is not None:
        paper = n_blocks * panel / max(main_share, 1e-6)
    return _MARGINAL_GAP_PX / max(paper or fig_px, 1.0)


def _marginal_trace(kind, values, axis, color, alpha, name, legendgroup, legendrank,
                    nbins=None, xbins=None, histnorm=''):
    """One distribution trace of ``values`` for the strip hugging the main
    plot's ``axis``: ``'x'`` is the strip above (drawn along x), ``'y'`` the
    strip to the right (drawn along y). Returns ``None`` when the kind can't
    be drawn for these values (a KDE on fewer than two distinct points)."""
    along_x = (axis == 'x')
    common = dict(name=name, legendgroup=legendgroup, legendrank=legendrank,
                  showlegend=False)
    data_kw = {'x': values} if along_x else {'y': values}

    if kind == 'histogram':
        kw = dict(common, marker_color=color, opacity=alpha, histnorm=histnorm, **data_kw)
        if along_x:
            kw.update(nbinsx=nbins, xbins=xbins)
        else:
            kw.update(nbinsy=nbins, ybins=xbins)
        return go.Histogram(**kw)

    if kind == 'box':
        return go.Box(marker_color=color, line_color=color, opacity=alpha,
                      orientation='h' if along_x else 'v', **common, **data_kw)

    if kind == 'violin':
        return go.Violin(line_color=color, fillcolor=_color_with_alpha(color, 0.5),
                         opacity=alpha, points=False, orientation='h' if along_x else 'v',
                         **common, **data_kw)

    if kind == 'rug':
        zeros = np.zeros(len(values))
        pos = {'x': values, 'y': zeros} if along_x else {'x': zeros, 'y': values}
        return _scatter_cls(len(values))(
            mode='markers', hoverinfo='skip', opacity=alpha,
            marker=dict(symbol='line-ns-open' if along_x else 'line-ew-open',
                        color=color, size=_MARGINAL_RUG_SIZE, line=dict(width=1, color=color)),
            **common, **pos)

    if kind == 'kde':
        from scipy.stats import gaussian_kde
        arr = np.asarray(values, dtype=float)
        if len(arr) < 2 or np.ptp(arr) == 0:
            return None
        grid = np.linspace(arr.min(), arr.max(), _MARGINAL_KDE_POINTS)
        dens = gaussian_kde(arr)(grid)
        pos = {'x': grid, 'y': dens, 'fill': 'tozeroy'} if along_x else \
              {'x': dens, 'y': grid, 'fill': 'tozerox'}
        return go.Scatter(mode='lines', line=dict(color=color, width=2),
                          fillcolor=_color_with_alpha(color, 0.3), opacity=alpha,
                          hovertemplate=f"{name}<br>%{{{'x' if along_x else 'y'}:.3g}}: "
                                        f"%{{{'y' if along_x else 'x'}:.3g}}<extra></extra>",
                          **common, **pos)

    raise ValueError(f"unknown marginal kind {kind!r}")


def _render_marginal(panels, marginal_x, marginal_y, marginal_size,
                     nbins, xbins, histnorm, alpha,
                     color, hue, marker, markersize, display_parms,
                     suptitle, xlabel, ylabel, darkmode, figsize, ncols, nrows,
                     axis_limits, legend, hspace, vspace, spacing_ref):
    """Shared builder behind ``unimarginal`` / ``unimarginal_per_dataset``.

    ``panels`` is a list of ``(title, datasets, x_col, y_col)`` — one block
    each. Everything that differs between the two public variants is decided
    by the caller in how it builds that list.
    """
    if not panels:
        return None
    if not (0 < marginal_size < 0.6):
        raise ValueError(f"marginal_size must be between 0 and 0.6, got {marginal_size!r}")
    axis_limits = axis_limits or {}
    n_blocks = len(panels)
    nrows_b, ncols_b = _calc_grid(n_blocks, nrows, ncols)
    n_rows_cells, n_cols_cells = 2 * nrows_b, 2 * ncols_b

    all_datasets = []
    for _, dss, _, _ in panels:
        all_datasets.extend(d for d in dss if d not in all_datasets)
    numeric_hue_info = _numeric_hue_info(all_datasets, hue, axis_limits)
    right_margin = max(80, len(numeric_hue_info) * 100)

    # make_subplots only gets the grid; every domain is overwritten below.
    fig = make_subplots(rows=n_rows_cells, cols=n_cols_cells,
                        column_widths=[1 - marginal_size, marginal_size] * ncols_b,
                        row_heights=[marginal_size, 1 - marginal_size] * nrows_b,
                        horizontal_spacing=0.001, vertical_spacing=0.001)
    fig.update_layout(**_base_layout(
        darkmode, None, figsize,
        title={'text': suptitle, 'x': 0.5, 'xanchor': 'center'},
        showlegend=(legend != 'off'), barmode='overlay',
        margin=dict(r=right_margin),
        **({'legend': dict(orientation="h")} if legend == 'above' else {}),
    ))

    # ---- block geometry (paper coords) -----------------------------------
    sp = _subplot_spacing(nrows_b, ncols_b, figsize, hspace, vspace, spacing_ref)
    hs, vs = sp.get('horizontal_spacing', 0.0), sp.get('vertical_spacing', 0.0)
    block_w = (1 - hs * (ncols_b - 1)) / ncols_b
    block_h = (1 - vs * (nrows_b - 1)) / nrows_b
    main_share = 1 - marginal_size
    gap_x = _marginal_gap_fraction(ncols_b, main_share, figsize, spacing_ref, 0) if marginal_y else 0.0
    gap_y = _marginal_gap_fraction(nrows_b, main_share, figsize, spacing_ref, 1) if marginal_x else 0.0

    legend_seen = set()
    for b_idx, (title, datasets, x_col, y_col) in enumerate(panels):
        br, bc = divmod(b_idx, ncols_b)
        bx0 = bc * (block_w + hs)
        by1 = 1 - br * (block_h + vs)
        bx1, by0 = bx0 + block_w, by1 - block_h
        # Clamp: the last row/column's far edge lands a rounding error
        # outside [0, 1], which Plotly rejects as a domain value.
        bx0, bx1, by0, by1 = (min(1.0, max(0.0, v)) for v in (bx0, bx1, by0, by1))

        main_w = (block_w - gap_x) * main_share if marginal_y else block_w
        main_h = (block_h - gap_y) * main_share if marginal_x else block_h
        main_xdom = [bx0, bx0 + main_w]
        main_ydom = [by0, by0 + main_h]

        main, top, right, corner = _marginal_block_cells(b_idx, ncols_b)
        main_xref, main_yref = _subplot_refs(*main, n_cols_cells)

        fig.update_xaxes(domain=main_xdom, title_text=xlabel or x_col, title_standoff=15,
                         row=main[0], col=main[1])
        fig.update_yaxes(domain=main_ydom, title_text=ylabel or y_col, title_standoff=15,
                         row=main[0], col=main[1])
        if x_col in axis_limits:
            fig.update_xaxes(range=axis_limits[x_col], row=main[0], col=main[1])
        if y_col in axis_limits:
            fig.update_yaxes(range=axis_limits[y_col], row=main[0], col=main[1])

        # The strips share the main panel's data axis (``matches``) and hide
        # every tick label: the count/density axis carries no information a
        # reader needs, and the data axis is already labelled on the main panel.
        if marginal_x:
            fig.update_xaxes(domain=main_xdom, matches=main_xref, showticklabels=False,
                             row=top[0], col=top[1])
            fig.update_yaxes(domain=[by1 - (block_h - gap_y) * marginal_size, by1],
                             showticklabels=False, row=top[0], col=top[1])
        else:
            fig.update_xaxes(visible=False, row=top[0], col=top[1])
            fig.update_yaxes(visible=False, row=top[0], col=top[1])
        if marginal_y:
            fig.update_xaxes(domain=[bx1 - (block_w - gap_x) * marginal_size, bx1],
                             showticklabels=False, row=right[0], col=right[1])
            fig.update_yaxes(domain=main_ydom, matches=main_yref, showticklabels=False,
                             row=right[0], col=right[1])
        else:
            fig.update_xaxes(visible=False, row=right[0], col=right[1])
            fig.update_yaxes(visible=False, row=right[0], col=right[1])
        fig.update_xaxes(visible=False, row=corner[0], col=corner[1])
        fig.update_yaxes(visible=False, row=corner[0], col=corner[1])

        if title:
            # Same placement make_subplots gives its subplot_titles (centered
            # over the block, sitting on its top edge), so _apply_fonts'
            # subplot-title sweep restyles these the same way.
            fig.add_annotation(text=title, xref='paper', yref='paper',
                               x=(bx0 + bx1) / 2, y=by1, xanchor='center', yanchor='bottom',
                               showarrow=False, font=dict(size=16))

        # ---- traces ------------------------------------------------------
        for dataset in _in_draw_order(datasets):
            if not dataset.select: continue
            if x_col not in dataset.columns or y_col not in dataset.columns: continue

            fmt = dataset.get_format_dict()
            cur_title = fmt.get('title')
            cur_hue = fmt.get('hue') or hue
            cur_color = color or fmt.get('color')
            cur_marker = marker or fmt.get('marker')
            cur_linestyle = fmt.get('linestyle')
            cur_markersize = fmt.get('markersize', markersize)
            cur_linewidth = fmt.get('linewidth', 2)
            cur_reg_order = fmt.get('reg_order')
            cur_idx = fmt.get('index')
            cur_sig = fmt.get('sig_figs')
            cur_dec = fmt.get('decimals')
            num_fmt = _hover_fmt(cur_sig, '.2f', cur_dec)
            hover_parms = display_parms or fmt.get('display_parms', [])
            valid_hover = [p for p in hover_parms if p in dataset.columns]
            hue_in_cols = bool(cur_hue) and cur_hue in dataset.columns
            ds_order = dataset.order
            order_in_cols = bool(ds_order) and ds_order != 'index' and ds_order in dataset.columns

            needed = [x_col, y_col] + valid_hover
            if hue_in_cols: needed.append(cur_hue)
            if order_in_cols: needed.append(ds_order)
            df = dataset.cols(list(dict.fromkeys(needed)))
            df = df.sort_values(by=ds_order) if order_in_cols else df.sort_index()
            df = df.dropna(subset=[x_col, y_col])
            if df.empty: continue

            custom_data, hover_lines = build_hover_data(df, valid_hover,
                                                        cur_sig, cur_dec)
            ht = (f"<b><u>Set: {cur_idx}</u></b><br><b>{cur_title}</b><br>"
                  f"{x_col}: %{{x:{num_fmt}}}<br>{y_col}: %{{y:{num_fmt}}}")
            ht += "".join(hover_lines) + "<extra></extra>"

            show_line = bool(cur_linestyle) and not cur_reg_order
            show_marker = cur_marker is not None
            mode_parts = (['lines'] if show_line else []) + (['markers'] if show_marker else [])
            mode = "+".join(mode_parts) if mode_parts else 'none'

            marker_dict, line_dict, cur_opacity = _marker_line_style(
                fmt, cur_color, cur_marker, cur_markersize, cur_linewidth,
                cur_linestyle, cur_hue, df, numeric_hue_info)

            name = f"{cur_idx}: {cur_title}"
            group, rank = f"group_{cur_idx}", 1000 + cur_idx
            fig.add_trace(_scatter_cls(len(df))(
                x=df[x_col], y=df[y_col], mode=mode, name=name,
                legendgroup=group, legendrank=rank,
                marker=marker_dict, line=line_dict, opacity=cur_opacity,
                customdata=custom_data, hovertemplate=ht,
                showlegend=(cur_idx not in legend_seen),
            ), row=main[0], col=main[1])
            legend_seen.add(cur_idx)

            if cur_reg_order:
                rx, ry, fit_label = _calculate_regression(df, x_col, y_col, cur_reg_order)
                if rx is not None:
                    fig.add_trace(go.Scatter(
                        x=rx, y=ry, mode='lines',
                        name=f"{name} Fit ({fit_label})", legendgroup=group,
                        line=dict(color=line_dict['color'], width=cur_linewidth,
                                  dash=get_plotly_linestyle(cur_linestyle)),
                        opacity=cur_opacity, hoverinfo='skip', showlegend=False,
                    ), row=main[0], col=main[1])

            # The strips take the set color even when hue colors the points:
            # a distribution has no per-point identity to carry a hue.
            for kind, axis, cell, col_name in ((marginal_x, 'x', top, x_col),
                                               (marginal_y, 'y', right, y_col)):
                if not kind: continue
                trace = _marginal_trace(kind, df[col_name], axis, cur_color, alpha,
                                        name, group, rank, nbins, xbins, histnorm)
                if trace is not None:
                    fig.add_trace(trace, row=cell[0], col=cell[1])

    _apply_hue_coloraxes(fig, numeric_hue_info)
    return fig


def unimarginal(list_of_datasets, x, y, marginal='histogram', marginal_x=None, marginal_y=None,
                marginal_size=0.2, nbins=None, bin_size=None, bin_start=None, bin_end=None,
                histnorm='', alpha=0.7, color=None, hue=None, marker=None, markersize=10,
                display_parms=None, suptitle=None, xlabel=None, ylabel=None, subplot_titles=None,
                darkmode=False, figsize=(12, 8), ncols=None, nrows=None,
                x_lim=None, y_lim=None, axis_limits=None, legend='above', return_axes=False,
                hspace=None, vspace=None, spacing_ref=None):
    """Scatter of ``y`` against ``x`` with marginal distribution strips.

    One block per (x, y) pair — pairs form as in ``uniplot`` (equal-length
    lists zip, a single value broadcasts) — with every selected dataset
    overlaid in each block. ``marginal`` picks the strip kind for both sides
    ('histogram', 'box', 'violin', 'rug' or 'kde'); ``marginal_x`` /
    ``marginal_y`` override one side, or switch it off with ``False``.
    ``marginal_size`` is the strip's share of the block (default 0.2). The
    histogram binning arguments (``nbins``, ``bin_*``, ``histnorm``) and
    ``alpha`` apply to the strips only; the scatter takes each dataset's
    own styling, as ``uniplot`` does.
    """
    pairs = _xy_pairs(x, y)
    y_list = y if isinstance(y, list) else [y]
    m_x, m_y = _resolve_marginal_kinds(marginal, marginal_x, marginal_y)
    titles = subplot_titles or [None] * len(pairs)
    if len(titles) < len(pairs):
        titles = list(titles) + [None] * (len(pairs) - len(titles))
    active = [d for d in list_of_datasets if d.select]
    panels = [(titles[i], active, xi, yi) for i, (xi, yi) in enumerate(pairs)]

    axis_limits = dict(axis_limits or {})
    if x_lim and len(pairs) == 1: axis_limits[pairs[0][0]] = x_lim
    if y_lim and len(pairs) == 1: axis_limits[pairs[0][1]] = y_lim

    default_title = (f"{x} vs {[str(yi) for yi in y_list]}" if not isinstance(x, list)
                     else f"{x} vs {y_list}")
    fig = _render_marginal(
        panels, m_x, m_y, marginal_size, nbins, _build_xbins(bin_size, bin_start, bin_end),
        histnorm, alpha, color, hue, marker, markersize, display_parms,
        suptitle or default_title, xlabel, ylabel, darkmode, figsize, ncols, nrows,
        axis_limits, legend, hspace, vspace, spacing_ref)
    if fig is None:
        return None
    return _show_or_return(fig, return_axes)


def unimarginal_per_dataset(list_of_datasets, x, y, marginal='histogram', marginal_x=None,
                            marginal_y=None, marginal_size=0.2, nbins=None, bin_size=None,
                            bin_start=None, bin_end=None, histnorm='', alpha=0.7, color=None,
                            hue=None, marker=None, markersize=10, display_parms=None,
                            suptitle=None, xlabel=None, ylabel=None,
                            darkmode=False, figsize=(12, 8), ncols=None, nrows=None,
                            x_lim=None, y_lim=None, axis_limits=None, legend='above',
                            return_axes=False, hspace=None, vspace=None, spacing_ref=None):
    """``unimarginal`` with one block per selected dataset for a single
    ``x`` / ``y`` pair (a one-element list is accepted for either)."""
    if isinstance(x, list):
        if len(x) != 1:
            raise ValueError("plot_marginal(by='sets') takes a single x variable")
        x = x[0]
    if isinstance(y, list):
        if len(y) != 1:
            raise ValueError("plot_marginal(by='sets') takes a single y variable")
        y = y[0]
    m_x, m_y = _resolve_marginal_kinds(marginal, marginal_x, marginal_y)
    active = [d for d in list_of_datasets if d.select]
    if not active:
        print("No datasets selected.")
        return None
    panels = [(d.title_format, [d], x, y) for d in active]

    axis_limits = dict(axis_limits or {})
    if x_lim: axis_limits[x] = x_lim
    if y_lim: axis_limits[y] = y_lim

    fig = _render_marginal(
        panels, m_x, m_y, marginal_size, nbins, _build_xbins(bin_size, bin_start, bin_end),
        histnorm, alpha, color, hue, marker, markersize, display_parms,
        suptitle or "Dataset Comparison", xlabel, ylabel, darkmode, figsize, ncols, nrows,
        axis_limits, legend, hspace, vspace, spacing_ref)
    if fig is None:
        return None
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
    for ds in _in_draw_order(overlay_datasets):
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
        line_dict = dict(width=ds.linewidth, color=ds.color,
                         dash=get_plotly_linestyle(ds.linestyle))
        opacity = _split_alpha(ds.get_format_dict(), ds.color, marker_dict, line_dict)
        prepared.append((ds, df, mode, marker_dict, line_dict, opacity))

    for cell in range(n_subplots):
        row, col = (cell // ncols) + 1, (cell % ncols) + 1
        for ds, df, mode, marker_dict, line_dict, opacity in prepared:
            fig.add_trace(go.Scatter(
                x=df[x], y=df[y],
                mode=mode,
                name=f"{ds.index}: {ds.title}",
                legendgroup=f"overlay_{ds.index}",
                legendrank=1000 + ds.index,
                marker=marker_dict,
                line=line_dict,
                opacity=opacity,
                showlegend=(cell == 0),
                hovertemplate=(f"<b>{ds.title}</b><br>"
                               f"{x}: %{{x:{_hover_fmt(ds.sig_figs, '.3g', ds.decimals)}}}<br>"
                               f"{y}: %{{y:{_hover_fmt(ds.sig_figs, '.3g', ds.decimals)}}}"
                               f"<extra></extra>")
            ), row=row, col=col)

def _contour_line_style(ds, coloring):
    """The ``go.Contour.line`` dict giving a set's contour boundaries its style.

    Boundaries take the set's normal line style — ``linewidth``, ``linestyle``
    (mapped to a Plotly dash) and ``color`` — the same values a line plot uses.
    ``color`` is left out when Plotly would discard it: under
    ``contours_coloring='lines'`` the boundaries are colored from the
    colorscale by z, and a line color has no effect there.
    """
    line = dict(width=ds.linewidth, dash=get_plotly_linestyle(ds.linestyle))
    if coloring != 'lines':
        line['color'] = ds.color
    return line


def unicontour(list_of_datasets, x, y, z, contours_coloring='fill', colorscale=None,
               interpolate=True, interp_res=100, interp_method='linear',
               ncontours=None, overlay_datasets=None,
               suptitle=None, xlabel=None, ylabel=None, subplot_titles=None,
               darkmode=False, figsize=(12, 8), ncols=None, nrows=None,
               axis_limits=None, return_axes=False,
               hspace=None, vspace=None, spacing_ref=None):
    """
    Create a unified contour plot for a list of datasets.
    Subplots are organized by Z-variables.

    Each set's contour boundaries are drawn in its own style — ``color``,
    ``linestyle`` and ``linewidth``. With more than one set the default
    ``contours_coloring='fill'`` becomes ``'none'`` (bare boundaries), since
    overlapping fills would hide each other and this is what lets the sets'
    colors tell them apart; the legend then carries the set styles instead of
    a colorbar. Passing ``contours_coloring='lines'`` explicitly keeps
    Plotly's colorscale-by-z line coloring, which ignores a line color
    (``linestyle``/``linewidth`` still apply).
    """
    z_list = z if isinstance(z, list) else [z]
    n_z = len(z_list)
    active_ds = _in_draw_order(d for d in list_of_datasets if d.select)
    axis_limits = axis_limits or {}

    if not active_ds:
        print("No datasets selected.")
        return None

    nrows, ncols = _calc_grid(n_z, nrows, ncols)

    # Filled contours from several sets would hide each other, so a multi-set
    # plot draws bare boundaries instead. 'none' — not Plotly's 'lines'
    # coloring, which paints the boundaries from the colorscale by z and
    # ignores line.color — is what lets each set keep its own color, linestyle
    # and linewidth.
    use_coloring = ('none' if len(active_ds) > 1 and contours_coloring == 'fill'
                    else contours_coloring)
    show_colorbars = use_coloring != 'none'

    # A panel with a colorbar carries it (bar, tick labels, title) on its
    # right, which the column gap has to hold on top of the next panel's y
    # axis; without one the normal gap and right margin are enough.
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles or z_list,
                        **_subplot_spacing(nrows, ncols, figsize, hspace, vspace, spacing_ref,
                                           h_px=160 if show_colorbars else None))
    fig.update_layout(**_base_layout(
        darkmode, None, figsize,
        title={'text': suptitle or f"Contour: {y} vs {x}", 'x': 0.5, 'xanchor': 'center', 'y': 0.98, 'yanchor': 'top', 'yref': 'container'},
        showlegend=True, margin=dict(r=100) if show_colorbars else {},
        legend=dict(orientation="h"),
    ))

    for idx_ds, ds in enumerate(active_ds):
        df = ds.cols([c for c in dict.fromkeys([x, y] + z_list) if c in ds.columns])
        if x not in df.columns or y not in df.columns:
            continue

        for idx_z, zi in enumerate(z_list):
            if zi not in df.columns: continue
            row, col = (idx_z // ncols) + 1, (idx_z % ncols) + 1

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
                legendrank=1000 + ds.index,
                colorscale=colorscale or ds.hue_palette,
                contours_coloring=use_coloring,
                ncontours=ncontours, 
                line=_contour_line_style(ds, use_coloring),
                showscale=(show_colorbars and idx_ds == 0),
                showlegend=(idx_z == 0),
                colorbar=dict(
                    title=zi,
                    x=cb_x,
                    y=cb_y,
                    len=cb_len,
                    thickness=15
                ),
                hovertemplate=(f"<b>{ds.title}</b><br>"
                               f"{x}: %{{x:{_hover_fmt(ds.sig_figs, '.3g', ds.decimals)}}}<br>"
                               f"{y}: %{{y:{_hover_fmt(ds.sig_figs, '.3g', ds.decimals)}}}<br>"
                               f"{zi}: %{{z:{_hover_fmt(ds.sig_figs, '.3g', ds.decimals)}}}"
                               f"<extra></extra>")
            ), row=row, col=col)

    _add_contour_overlays(fig, overlay_datasets, x, y, n_z, ncols, darkmode)

    fig.update_xaxes(title_text=xlabel or x)
    fig.update_yaxes(title_text=ylabel or y)

    return _show_or_return(fig, return_axes)

def unicontour_per_dataset(list_of_datasets, x, y, z, contours_coloring='fill', colorscale=None,
                           interpolate=True, interp_res=100, interp_method='linear',
                           ncontours=None, overlay_datasets=None,
                           suptitle=None, figsize=(12, 8), ncols=None, nrows=None,
                           darkmode=False, axis_limits=None, return_axes=False,
                           hspace=None, vspace=None, spacing_ref=None):
    """
    Contour plot where Subplots are organized by Dataset.

    Boundaries take the set's ``color``, ``linestyle`` and ``linewidth``. With
    several z variables in one panel the default ``'fill'`` coloring becomes
    ``'lines'`` so the fills don't hide each other; each z keeps its own
    colorbar and colors its lines from the colorscale, so the set's color
    doesn't apply in that case (its linestyle and linewidth still do).
    """
    active_ds = [d for d in list_of_datasets if d.select]
    z_list = z if isinstance(z, list) else [z]
    n_sets = len(active_ds)
    axis_limits = axis_limits or {}

    if not active_ds:
        print("No datasets selected.")
        return None

    nrows, ncols = _calc_grid(n_sets, nrows, ncols)

    # One colorbar per z variable per panel, stacked to the right of it.
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[d.title_format for d in active_ds],
                        **_subplot_spacing(nrows, ncols, figsize, hspace, vspace, spacing_ref,
                                           h_px=160 + 52 * (len(z_list) - 1)))
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
                line=_contour_line_style(ds, use_coloring),
                showscale=(idx_ds == 0),
                showlegend=(idx_ds == 0),
                colorbar=dict(
                    title=zi,
                    x=cb_x,
                    y=cb_y,
                    len=cb_len,
                    thickness=15
                ),
                hovertemplate=(f"<b>{zi}</b><br>"
                               f"{x}: %{{x:{_hover_fmt(ds.sig_figs, '.3g', ds.decimals)}}}<br>"
                               f"{y}: %{{y:{_hover_fmt(ds.sig_figs, '.3g', ds.decimals)}}}<br>"
                               f"Value: %{{z:{_hover_fmt(ds.sig_figs, '.3g', ds.decimals)}}}"
                               f"<extra></extra>")
            ), row=row, col=col)

    _add_contour_overlays(fig, overlay_datasets, x, y, n_sets, ncols, darkmode)

    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)

    return _show_or_return(fig, return_axes)

_BAR_AGG_NAMES = ('mean', 'sum', 'min', 'max', 'median', 'std', 'var',
                  'count', 'first', 'last')


def _resolve_agg(agg):
    """Normalise a ``bar(agg=)`` value to ``(func, name)``.

    ``func`` is what pandas' ``Series.agg`` / ``GroupBy.agg`` accept: one of
    the built-in reducer names, any other ``pd.Series`` reducer method name
    ('nunique', 'sem', 'prod', ...), or a callable taking a Series. ``name``
    labels the reducer in hover text and titles. ``agg=False`` (or 'raw')
    means "no aggregation, draw the rows as they are" and resolves to
    ``(None, None)``. Anything else raises ``ValueError`` — silently falling
    back to a mean would mislabel the chart.
    """
    if agg is False or agg == 'raw':
        return None, None
    if isinstance(agg, str):
        if agg in _BAR_AGG_NAMES or callable(getattr(pd.Series, agg, None)):
            return agg, agg
        raise ValueError(
            f"agg must be one of {_BAR_AGG_NAMES}, another pandas Series "
            f"reducer name, a callable, or False for raw rows; got {agg!r}")
    if callable(agg):
        return agg, getattr(agg, '__name__', 'agg')
    raise ValueError(
        f"agg must be a reducer name, a callable, or False; got {agg!r}")


def _agg_series(series, func):
    """Reduce one column to a scalar with a resolved ``func``; NaN when the
    series has no valid values. 'first'/'last' are handled here because
    ``Series.first``/``Series.last`` are offset-based, not reducers."""
    valid = series.dropna()
    if valid.empty:
        return np.nan
    if func == 'first': return valid.iloc[0]
    if func == 'last': return valid.iloc[-1]
    return valid.agg(func)


def _agg_column(ds, col, func):
    """Aggregate one dataset column to a scalar; None when the column is
    absent or all-NaN."""
    if col not in ds.columns:
        return None
    val = _agg_series(ds[col], func)
    return None if pd.isna(val) else val


def _agg_count(ds, col):
    """Number of valid (non-NaN) points ``_agg_column`` aggregates for one
    column — surfaced in the dataset_x bar hover as ``n=``."""
    if col not in ds.columns:
        return 0
    return int(ds[col].notna().sum())


def _bar_frame(df, x, cols, func):
    """Reduce ``df`` to one row per ``x`` category for the bar backends.

    Returns ``(values, counts)``: ``values`` has ``x`` plus each of ``cols``
    reduced by ``func`` within its category (categories keep first-appearance
    order, matching Plotly's category axis); ``counts`` has the same shape
    and holds the number of valid rows behind each value, for the ``n=`` in
    hover text. With ``func`` None (``agg=False``) the frame is returned as is
    and ``counts`` is None — one bar per row.
    """
    if func is None:
        return df, None
    cols = [c for c in cols if c != x]
    grouped = df.groupby(x, sort=False, observed=True)
    values = grouped[cols].agg(func)      # 'first'/'last' skip NaN here
    counts = grouped[cols].count()
    return values.reset_index(), counts.reset_index()


def _agg_hover(agg_name, col, counts_col, sig_figs=None, decimals=None):
    """Hover fragment for one value: ``mean EGT: 643 (n=64)`` when aggregated,
    else the plain ``EGT: 643``. The set's ``sig_figs`` / ``decimals`` replace
    the built-in ``.4g`` precision."""
    fmt = _hover_fmt(sig_figs, '.4g', decimals)
    if agg_name is None or counts_col is None:
        return f"{col}: %{{y:{fmt}}}"
    return f"{agg_name} {col}: %{{y:{fmt}}} (n=%{{customdata}})"


def unibar_datasets_as_x(list_of_datasets, y, agg='mean', markers=None, variable_formats=None,
                         suptitle=None, darkmode=False,
                         figsize=(12, 8), axis_limits=None, return_axes=False):
    """
    Creates a single grouped bar chart where the X-axis is the Dataset name,
    and the bars are the different Y-variables, each scaled to their own Y-axis.
    Includes an 'agg' parameter to handle multi-row datasets. Hovering a bar
    (or overlay glyph) reports the aggregate as ``<agg>: <value> (n=<points>)``,
    where n counts the valid (non-NaN) rows that fed the aggregation.

    ``markers`` columns are aggregated with the same ``agg`` rule and overlaid
    as marker/tick/whisker glyphs (per the ``style`` key of their
    ``variable_formats`` entry). Overlays pair positionally with the y
    variables: the i-th marker column rides on the i-th y variable's axis
    and, for tick/whisker, attaches to its bars (so ``y=['T41','RU'],
    markers=['T41_LIMIT','RU_LIMIT']`` puts each limit on its own variable's
    scale). Markers beyond the number of y variables fall back to the first
    axis.

    Per-variable ``variable_formats`` overrides apply here: a variable's
    ``color`` recolors its bar and its Y-axis (overriding the default color
    cycle), and ``alpha`` sets the bar opacity. Marker columns additionally
    honor ``marker``/``markersize``/``style`` and ``linestyle``/``linewidth``
    (the whisker stem).
    """
    active_ds = [d for d in list_of_datasets if d.select]
    if not active_ds:
        print("No datasets selected.")
        return None

    y_list = y if isinstance(y, list) else [y]
    markers_list = markers if isinstance(markers, list) else ([markers] if markers else [])
    axis_limits = axis_limits or {}
    variable_formats = variable_formats or {}
    color_cycle = px.colors.qualitative.Plotly
    agg_func, agg_name = _resolve_agg(agg)
    if agg_func is None:
        raise ValueError("by='dataset_x' reduces each dataset to one bar per "
                         "variable, so it needs an aggregation (agg=False is not allowed).")

    fig = go.Figure()

    x_labels = [f"{ds.index}: {ds.title}" for ds in active_ds]

    extras_count = max(0, len(y_list) - 2)
    width_per_axis = _yaxis_slot_fraction(
        0.08, (figsize[0] * 100 if figsize else 1200) - 80
              - (50 + extras_count * 80))
    required_space = extras_count * width_per_axis
    x_domain_end = max(0.5, 1.0 - required_space) 

    for idx_y, yi in enumerate(y_list):
        var_fmt = variable_formats.get(yi, {})
        var_color = var_fmt.get('color', color_cycle[idx_y % len(color_cycle)])
        var_alpha = var_fmt.get('alpha')

        y_data = [_agg_column(ds, yi, agg_func) for ds in active_ds]

        y_axis_name = "y" if idx_y == 0 else f"y{idx_y + 1}"

        fig.add_trace(go.Bar(
            name=yi,
            x=x_labels,
            y=y_data,
            yaxis=y_axis_name,
            offsetgroup=str(idx_y),
            marker_color=var_color,
            opacity=var_alpha if var_alpha is not None else 1.0,
            customdata=[_agg_count(ds, yi) for ds in active_ds],
            hovertemplate=(f"<b>{yi}</b><br>%{{x}}<br>"
                           f"{agg_name}: %{{y:.4g}} (n=%{{customdata}})<extra></extra>")
        ))

        axis_layout = dict(
            title=yi,
            title_font=dict(color=var_color),
            tickfont=dict(color=var_color),
            showgrid=(idx_y == 0)
        )

        # A marker column pairs positionally to the idx-th y variable's axis
        # (see the marker loop below), so its scale() unions into that axis.
        marker_lims = [axis_limits[markers_list[mi]]
                       for mi in range(len(markers_list))
                       if (mi if mi < len(y_list) else 0) == idx_y
                       and markers_list[mi] in axis_limits]
        yr = _union_ranges([axis_limits.get(yi)] + marker_lims)
        if yr is not None:
            axis_layout['range'] = yr

        if idx_y == 0:
            fig.update_layout(yaxis=axis_layout)
        elif idx_y == 1:
            axis_layout.update(dict(overlaying='y', side='right', anchor='x'))
            fig.update_layout(yaxis2=axis_layout)
        else:
            pos = x_domain_end + ((idx_y - 1) * width_per_axis)
            axis_layout.update(dict(overlaying='y', side='right', anchor='free', position=pos))
            fig.update_layout({f"yaxis{idx_y + 1}": axis_layout})

    edge_default = 'white' if darkmode else 'black'

    for m_idx, m_col in enumerate(markers_list):
        m_vals = pd.Series([_agg_column(ds, m_col, agg_func) for ds in active_ds],
                           dtype=float)
        if m_vals.isna().all():
            continue

        # Positional pairing: the i-th marker column anchors to the i-th y
        # variable (its axis, its bars); extras fall back to the first.
        anchor_idx = m_idx if m_idx < len(y_list) else 0
        anchor_axis = 'y' if anchor_idx == 0 else f"y{anchor_idx + 1}"
        anchor_vals = pd.Series([_agg_column(ds, y_list[anchor_idx], agg_func)
                                 for ds in active_ds], dtype=float)

        var_fmt = variable_formats.get(m_col, {})
        m_symbol = get_plotly_marker(var_fmt.get('marker') or marker_map(m_idx))
        m_color  = var_fmt.get('color') or color_cycle[(len(y_list) + m_idx) % len(color_cycle)]
        m_size   = var_fmt.get('markersize', 12)
        m_alpha  = var_fmt.get('alpha', 1.0)
        m_style  = var_fmt.get('style', 'marker')
        m_dash   = (get_plotly_linestyle(var_fmt['linestyle'])
                    if var_fmt.get('linestyle') is not None else 'solid')
        m_lw     = var_fmt.get('linewidth', 2)

        # Tick/whisker glyphs attach to the anchor variable's bars via its
        # offsetgroup (classic markers stay at the category center).
        attach = m_style in ('tick', 'whisker') and anchor_vals.notna().any()
        group_kw = ({'offsetgroup': str(anchor_idx), 'alignmentgroup': 'bars'}
                    if attach else {})

        fig.add_trace(go.Scatter(
            x=x_labels, y=m_vals,
            mode='markers',
            name=m_col,
            legendgroup=f"marker_{m_col}",
            yaxis=anchor_axis,
            **group_kw,
            **_overlay_marker_kw(m_style, m_symbol, m_color, m_size,
                                 m_alpha, edge_default,
                                 values=m_vals,
                                 bar_values=anchor_vals if attach else None,
                                 stem_dash=m_dash, stem_width=m_lw),
            customdata=[_agg_count(ds, m_col) for ds in active_ds],
            hovertemplate=(f"<b>{m_col}</b><br>%{{x}}<br>"
                           f"{agg_name}: %{{y:.4g}} (n=%{{customdata}})<extra></extra>")
        ))

        if m_style == 'whisker' and attach and m_dash and m_dash != 'solid':
            fig.add_trace(go.Scatter(
                name=f"{m_col} (stem)",
                legendgroup=f"marker_{m_col}",
                yaxis=anchor_axis,
                **group_kw,
                **_whisker_stem_kw(x_labels, m_vals, anchor_vals,
                                   m_color, m_alpha, m_dash, m_lw),
            ))

    fig.update_layout(**_base_layout(
        darkmode, suptitle or f"Variables by Dataset ({agg_name})", figsize,
        barmode='group',
        xaxis=dict(domain=[0, x_domain_end], title="Dataset"),
        margin=dict(r=50 + (extras_count * 80)),
        legend=dict(orientation="h"),
        scattermode='group'
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
    width_per_axis = _yaxis_slot_fraction(
        0.08, (figsize[0] * 100 if figsize else 1200) - 80
              - (50 + extras_count * 80))
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
                        legend='right', legend_group_by='sets', style_by=None,
                        color_cycle=None, marker_cycle=None,
                        return_axes=False):
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
    style_by : str | list[str] | None
        Auto-differentiate the y variables by cycling one or more attributes:
        'color', 'marker', 'linestyle', or a combination ('color+marker',
        ['marker', 'linestyle'], ...). Attributes set in ``variable_formats``
        still win per-variable; auto colors also tint that variable's axis.
    """
    variable_formats = variable_formats or {}
    axis_limits = axis_limits or {}

    active = _in_draw_order(d for d in list_of_datasets if d.select)
    if not active:
        print("No datasets selected.")
        return None

    y_list = y if isinstance(y, list) else [y]
    if not y_list:
        raise ValueError("At least one y variable is required.")

    variable_formats = _style_by_formats(y_list, style_by, variable_formats,
                                         color_cycle=color_cycle,
                                         marker_cycle=marker_cycle)

    # Domain math: extra y-axes (3+) live to the right of the plot. The slot
    # each one gets is a fraction of the plot area, floored at _YAXIS_SLOT_PX so
    # a narrow figure can't crush the axes into each other.
    extras = max(0, len(y_list) - 2)
    ymult_right_margin = max(80, 60 + extras * 70)
    width_per_axis = _yaxis_slot_fraction(
        0.06, (figsize[0] * 100 if figsize else 1200) - 80 - ymult_right_margin)
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
        ds_sig = getattr(ds, 'sig_figs', None)
        ds_dec = getattr(ds, 'decimals', None)
        num_fmt = _hover_fmt(ds_sig, '.4g', ds_dec)

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
                  f"{x}: %{{x:{num_fmt}}}<br>"
                  f"{yi}: %{{y:{num_fmt}}}")
            customdata, hover_lines = build_hover_data(df, valid_hover,
                                                       ds_sig, ds_dec)
            ht += "".join(hover_lines)
            ht += "<extra></extra>"

            y_axis_name = "y" if idx_y == 0 else f"y{idx_y + 1}"

            # Markers carry ``alpha`` (or ``alpha_marker``) directly; the
            # line only goes translucent when ``alpha_line`` asks for it.
            m_alpha = fmt['alpha'] if fmt.get('alpha_marker') is None else fmt['alpha_marker']
            l_color = (fmt['color'] if fmt.get('alpha_line') is None
                       else _color_with_alpha(fmt['color'], fmt['alpha_line']))

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
                legendrank=1000 if legend_group_by == 'vars' else 1000 + ds.index,
                legendgrouptitle_text=yi if legend_group_by == 'vars' else f"{ds.index}: {ds.title}",
                marker=dict(
                    size=fmt['markersize'],
                    symbol=get_plotly_marker(fmt['marker']),
                    opacity=m_alpha,
                    **marker_kw,
                ),
                line=dict(
                    width=fmt['linewidth'],
                    dash=get_plotly_linestyle(fmt['linestyle']) or 'solid',
                    color=l_color,
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
        # 60 is enough for a bare right edge, but the second y variable already
        # hangs an axis there (its labels need ~78px); under-reserving lets
        # Plotly's autoexpand take the difference out of the plot area.
        margin=dict(r=ymult_right_margin),
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

# ----------------------------------------------------------------------
# PNG text-chunk helpers (save_png embeds the plotting session in the image)
# ----------------------------------------------------------------------
_PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
_PNG_SESSION_KEYWORD = 'unichart-session'


def _png_chunks(data):
    """Yield ``(type, payload)`` for every chunk of PNG bytes ``data``."""
    if not data.startswith(_PNG_SIGNATURE):
        raise ValueError("not a PNG byte string")
    pos = len(_PNG_SIGNATURE)
    while pos + 8 <= len(data):
        length, = struct.unpack('>I', data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        yield ctype, data[pos + 8:pos + 8 + length]
        pos += 12 + length


def _png_chunk(ctype, payload):
    crc = zlib.crc32(ctype + payload) & 0xffffffff
    return struct.pack('>I', len(payload)) + ctype + payload + struct.pack('>I', crc)


def png_embed_text(data, keyword, text):
    """Return PNG bytes ``data`` with ``text`` stored under ``keyword`` as a
    compressed ``iTXt`` chunk (UTF-8) just before ``IEND``. An existing text
    chunk with the same keyword is replaced. Pixels are untouched, and every
    PNG reader ignores the chunk, so the image displays as before."""
    key = keyword.encode('latin-1')
    out = [_PNG_SIGNATURE]
    for ctype, payload in _png_chunks(data):
        if ctype in (b'iTXt', b'zTXt', b'tEXt') and payload.split(b'\x00', 1)[0] == key:
            continue
        if ctype == b'IEND':
            body = (key + b'\x00'            # keyword
                    + b'\x01\x00'            # compressed, method 0 (zlib)
                    + b'\x00' + b'\x00'      # language tag, translated keyword
                    + zlib.compress(text.encode('utf-8'), 9))
            out.append(_png_chunk(b'iTXt', body))
        out.append(_png_chunk(ctype, payload))
    return b''.join(out)


def png_read_text(data, keyword):
    """The text stored under ``keyword`` in PNG bytes ``data`` (``iTXt``,
    ``zTXt`` or ``tEXt`` chunk), or None if there is none."""
    key = keyword.encode('latin-1')
    for ctype, payload in _png_chunks(data):
        if ctype not in (b'iTXt', b'zTXt', b'tEXt'):
            continue
        k, _, rest = payload.partition(b'\x00')
        if k != key:
            continue
        if ctype == b'tEXt':
            return rest.decode('latin-1')
        if ctype == b'zTXt':
            return zlib.decompress(rest[1:]).decode('latin-1')
        compressed, rest = rest[0], rest[2:]
        _lang, _, rest = rest.partition(b'\x00')
        _translated, _, rest = rest.partition(b'\x00')
        if compressed:
            rest = zlib.decompress(rest)
        return rest.decode('utf-8')
    return None


def read_png_session(path):
    """The unichart session embedded in a PNG by :meth:`UnichartNotebook.save_png`,
    as a dict — the same structure as a ``save_session`` file plus a
    ``plot_call`` entry (method name and arguments). None if the PNG carries no
    session. Use :meth:`UnichartNotebook.load_session` on the PNG to actually
    restore and replot it; this is for inspecting what an image contains."""
    with open(path, 'rb') as fh:
        data = fh.read()
    text = png_read_text(data, _PNG_SESSION_KEYWORD)
    return json.loads(text) if text else None


# How much of a non-PNG file to look at when sniffing. json.dump writes keys in
# insertion order and _build_session sets 'unichart_session' first, so the marker
# is always in the opening bytes of a file this library wrote.
_SESSION_SNIFF_BYTES = 8192


def sniff_session(source):
    """``'png'``, ``'json'`` or None for a session file — by path or by bytes.

    The cheap "is this a session?" question, for callers dispatching a file the
    user handed over: the ``unichart`` command deciding between :meth:`load` and
    :meth:`load_session`, or the explorer's drop zone deciding between a data
    frame and a restore. Answers without parsing a possibly huge file: a PNG is
    identified by its signature and its text chunk, a JSON by the
    ``"unichart_session"`` marker in the opening bytes.

    That marker check is positional, so a session file whose keys have been
    re-ordered by hand sniffs as ``None`` — it then takes the ordinary data
    path, which is what happens today, rather than failing. Never raises:
    anything unreadable, half-typed or not a session is None.
    """
    try:
        if isinstance(source, (bytes, bytearray)):
            head = bytes(source[:_SESSION_SNIFF_BYTES])
            data = bytes(source)
        else:
            with open(source, 'rb') as fh:
                head = fh.read(_SESSION_SNIFF_BYTES)
            data = None
        if head.startswith(_PNG_SIGNATURE):
            if data is None:
                with open(source, 'rb') as fh:
                    data = fh.read()
            return 'png' if png_read_text(data, _PNG_SESSION_KEYWORD) else None
        return 'json' if b'"unichart_session"' in head else None
    except Exception:                                         # noqa: BLE001
        return None


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
        # The last plotting call (method + resolved arguments), so save_png
        # can embed everything needed to remake the figure. See _record_plot_call.
        self._last_plot_call = None
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

        # When True (default), every interactive plot gets a small '⧉ copy'
        # button displayed above it that copies the rendered plot to the
        # clipboard as a PNG (in-browser, no kaleido). Static-image mode
        # skips the button: flat PNGs already support the browser's native
        # right-click → Copy Image. Toggle via set_copy_buttons.
        self.copy_buttons = True

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
            'legend': None, 'suppress_legends': None, 'legend_scroll': None,
            'ncols': None, 'nrows': None, 'hspace': None, 'vspace': None,
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

        # Overall look of the figures: 'matplotlib' (default) or 'plotly'.
        # Orthogonal to darkmode — each style has a light and a dark variant.
        # Installs plot_style, the style's color_map / default_format entries and
        # its font-size fallbacks (_style_font_defaults), so it must run *after*
        # color_map and default_format exist. Change via set_plot_style.
        self._apply_style_defaults(DEFAULT_PLOT_STYLE)

        # Optional fixed plot-area size (px, w/h, either may be None) so plots
        # stay the same size — and shape — regardless of suptitle/legend/margins
        # or how many subplots the call produces. Set via set_plot_size; applied
        # in _finalize. None = size driven by figsize. plot_size_per_subplot
        # decides whether the pinned size is one panel (default) or the whole
        # subplot grid.
        self.plot_size = None
        self.plot_size_per_subplot = True

        # Sticky gridline formatting set via nb.grid(). Shape:
        # {'x'|'y': {'major'|'minor': {'visible'|'color'|'width'|'dash': value}}}.
        # Empty inner dicts = leave the plot style's grid alone. Applied to every
        # figure in _finalize (see _apply_grid); cleared by nb.grid(reset=True)
        # or reset_format('grid').
        self.grid_format = self._empty_grid_format()

        # Sticky watermark image set via nb.watermark(). Shape:
        # {'source': <data URI or URL>, 'source_repr': <as typed>, 'opacity':,
        # 'position':, 'size': (w, h), 'layer':, 'sizing':}. Empty = no
        # watermark; any key absent falls back to _WATERMARK_DEFAULTS. Applied
        # to every figure in _finalize (see _apply_watermark); cleared by
        # nb.watermark(reset=True) or reset_format('watermark').
        self.watermark_format = {}

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

        # A pure append preserves existing rows' labels and positions whenever
        # the current index is already a clean 0..n-1 range (every internal
        # rebuild path guarantees this; only direct index surgery on nb.df can
        # break it). In that case existing query masks stay exact — new rows
        # belong to the new, query-less sets and _masked_positions reindexes
        # masks over the longer index with fill_value=False — so the O(sets ×
        # frame width) query re-evaluation below can be skipped.
        idx = self._combined_df.index
        labels_preserved = (isinstance(idx, pd.RangeIndex)
                            and idx.start == 0 and idx.step == 1)

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
        if not labels_preserved:
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
        # The replacement frame defines this set's columns from scratch. It
        # also severs the link to any source file: the rows no longer come
        # from it, so save_session must embed this set's data instead.
        for ds in self.sets:
            if ds._set_id == set_id:
                ds._own_cols = set(new_df.columns) - {_SET_ID_COL}
                ds.file_path = None
                ds._source = None
                break
        self._snapshot_columns()
        # ignore_index rebuilds every row label, which staled all query masks
        # (they are keyed to the global index). Recompute them all.
        self._reapply_all_queries()

    @staticmethod
    def _dedupe_columns(df):
        """Return df with duplicate column labels renamed reader-style (X, X.1, X.2).

        Duplicate labels poison every later column alignment against the combined
        frame — pd.concat with another set raises InvalidIndexError ("Reindexing
        only valid with uniquely valued Index objects"), possibly loads later than
        the frame that introduced them.
        """
        if not df.columns.duplicated().any():
            return df
        counts, used, new_cols = {}, set(), []
        for col in df.columns:
            new = col
            while new in used:
                counts[col] = counts.get(col, 0) + 1
                new = f"{col}.{counts[col]}"
            used.add(new)
            new_cols.append(new)
        renamed = sorted(str(c) for c in counts)
        print(f"Renamed duplicate column(s) to be unique: {renamed}")
        return df.set_axis(new_cols, axis=1)

    def load_df(self, df, title=None, set_name_column=None, set_idx_column=None, load_cols_as_vars=False, combined=False):
        """Split a DataFrame into one Dataset per unique set_idx_column value, or load it as one.

        ``df`` may be a single DataFrame or a list of DataFrames. For a list, ``combined=True``
        concatenates them into one set, while ``combined=False`` loads each DataFrame separately.

        Returns the list of Datasets created.
        """
        if isinstance(df, (list, tuple)):
            if combined:
                df = pd.concat([self._dedupe_columns(d) for d in df],
                               ignore_index=True)
            else:
                created = []
                for single_df in df:
                    created.extend(self.load_df(
                        single_df, title=title, set_name_column=set_name_column,
                        set_idx_column=set_idx_column, load_cols_as_vars=load_cols_as_vars))
                return created

        df = self._dedupe_columns(df).copy()

        # Columns this call adds to the frame, recorded in each set's
        # provenance so a file-referenced session set gets them back on
        # reload (see _apply_synth_cols).
        synth_cols = {}
        if not title:
            if set_name_column and set_name_column in df.columns:
                pass
            elif "TITLE" in df.columns:
                set_name_column = "TITLE"
            else:
                df["TITLE"] = "Dataset"
                set_name_column = "TITLE"
                synth_cols["TITLE"] = {'kind': 'const', 'value': "Dataset"}

            if set_idx_column and set_idx_column in df.columns:
                pass
            elif "SETNUMBER" in df.columns:
                set_idx_column = "SETNUMBER"
            elif "INDEX" in df.columns:
                set_idx_column = "INDEX"
            else:
                df["SETNUMBER"] = df.index
                synth_cols["SETNUMBER"] = {'kind': 'index'}

        if set_idx_column and set_idx_column in df.columns:
            # Collect every group first and register them in one batch — one
            # combined-frame rebuild for the whole file instead of one per set.
            groups, group_keys = [], []
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
                group_keys.append(set_index)

            created = self._register_sets(groups)
            for ds, key in zip(created, group_keys):
                ds._source = {'path': None, 'read_kwargs': None,
                              'set_idx_column': set_idx_column,
                              'group_key': key.item() if hasattr(key, 'item') else key,
                              'synth_cols': dict(synth_cols),
                              'load_cols': set(ds._own_cols)}
                print(f"Loaded Set {ds.index}: {ds.title}")
        else:
            ds = self._register_set(df, title if title else "Untitled")
            ds._source = {'path': None, 'read_kwargs': None,
                          'set_idx_column': None, 'group_key': None,
                          'synth_cols': dict(synth_cols),
                          'load_cols': set(ds._own_cols)}
            created = [ds]
            print(f"Loaded Set {ds.index}: {ds.title}")

        if load_cols_as_vars:
            names = {str(c): str(c) for c in df.columns if str(c).isidentifier()}
            globals().update(names)
            skipped = [c for c in df.columns if str(c) not in names]
            if skipped:
                print(f"Could not create variables for {len(skipped)} column(s) "
                      f"whose names are not valid identifiers: {skipped[:10]}"
                      f"{'...' if len(skipped) > 10 else ''}")
        return created

    _FILE_READERS = {
        ".csv": lambda path, kw: pd.read_csv(path, **kw),
        ".tsv": lambda path, kw: pd.read_csv(path, sep="\t", **kw),
        ".txt": lambda path, kw: pd.read_csv(path, sep="\t", **kw),
        ".xlsx": lambda path, kw: pd.read_excel(path, **kw),
        ".xls": lambda path, kw: pd.read_excel(path, **kw),
        ".json": lambda path, kw: pd.read_json(path, **kw),
        ".parquet": lambda path, kw: pd.read_parquet(path, **kw),
    }

    # Dialog programs that can stand in for tkinter. This is what a sandboxed
    # kernel has: the VS Code Flatpak's runtime python ships no Tk, but the
    # runtime does carry zenity, so a subprocess still reaches a real dialog.
    # Only the zenity CLI is listed (qarma is a clone of it) because the rc==1
    # reading below is zenity's: a program whose usage errors also exit 1 would
    # have them read back as "the user cancelled", which is the one failure
    # worth never guessing at. Adding kdialog or yad means verifying that first.
    _DIALOG_PROGRAMS = ('zenity', 'qarma')

    @classmethod
    def _dialog_argv(cls, program):
        """Build the argv that asks ``program`` for a multi-select open dialog."""
        patterns = [f"*{ext}" for ext in cls._FILE_READERS]
        # The default separator is '|', which is a legal filename character, so
        # ask for newlines instead.
        return [program, '--file-selection', '--multiple', '--separator=\n',
                '--title=Select data file(s) to load',
                f"--file-filter=Data files | {' '.join(patterns)}",
                '--file-filter=All files | *']

    @classmethod
    def _pick_files_native(cls):
        """Pick files with an OS dialog program instead of tkinter.

        Returns the chosen paths, an empty tuple when the dialog was cancelled,
        or ``None`` when no dialog program could be run at all — the caller
        needs to tell "user picked nothing" apart from "there was nothing to
        ask", and only the first of those means load nothing.
        """
        import shutil
        import subprocess

        for program in cls._DIALOG_PROGRAMS:
            exe = shutil.which(program)
            if exe is None:
                continue
            argv = cls._dialog_argv(program)
            argv[0] = exe
            try:
                proc = subprocess.run(argv, capture_output=True, text=True)
            except OSError:
                continue
            if proc.returncode == 1:
                return ()          # zenity's cancel; its usage errors are 255
            if proc.returncode != 0:
                # Couldn't display (no Wayland/X11 socket, say). Another
                # program might still manage it, so keep looking. GTK chatters
                # on stderr even on success, so stderr alone proves nothing.
                continue
            # A picked path can be a document-portal mount
            # (/run/user/<uid>/doc/...) when the dialog ran behind a portal.
            # Those read fine now but are not stable, so a session saved from
            # one may not reload — see annotate() in load().
            return tuple(line for line in proc.stdout.split('\n') if line)
        return None

    @staticmethod
    def _no_picker_message(reason):
        """Build the "no file browser" error, with advice that fits the runtime.

        The advice has to branch on where the interpreter lives: inside a
        sandbox (the VS Code Flatpak runs its kernels on a runtime python that
        ships no Tk at all) a host ``apt install python3-tk`` installs into a
        filesystem this process cannot see, so the stock advice would send the
        caller down a dead end. The underlying exception is quoted rather than
        swallowed — "tkinter is unavailable" and "libtk8.6.so is unloadable"
        need different fixes.
        """
        hints = []
        if os.environ.get('container') or os.path.exists('/.flatpak-info'):
            hints.append(
                "this interpreter runs inside a sandbox (e.g. a kernel started "
                "by the VS Code Flatpak), whose runtime ships no Tk — installing "
                "python3-tk on the host is invisible to it, so run the kernel "
                "from a host interpreter instead")
        else:
            hints.append("install Tk (e.g. 'sudo apt install python3-tk')")
        if not os.environ.get('DISPLAY') and os.environ.get('WAYLAND_DISPLAY'):
            hints.append(
                "tkinter also talks X11 only, and DISPLAY is unset in this "
                "Wayland session, so it needs XWayland even once Tk is present")
        return (f"load() could not open a file browser: {reason}. "
                + "; ".join(hints)
                + ". Or pass a filepath: load('data.csv').")

    @classmethod
    def _pick_files(cls):
        """Open the OS file-chooser and return the selected paths as a tuple.

        Returns an empty tuple when the dialog is cancelled. Raises
        ``RuntimeError`` when no dialog can be opened at all — tkinter missing
        (Pyodide, a sandboxed runtime built without Tk, or Debian/Ubuntu
        without ``python3-tk``) or no display reachable (a headless server, or
        a sandboxed kernel with no Wayland/X11 socket) — since the caller asked
        for a picker and silently loading nothing would be the more confusing
        outcome.

        tkinter is imported lazily: unichart imports fine without it, and only
        this method needs it.
        """
        try:
            import tkinter
            from tkinter import filedialog
        except ImportError as e:
            picked = cls._pick_files_native()
            if picked is not None:
                return picked
            raise RuntimeError(
                cls._no_picker_message(f"tkinter is unavailable ({e})")) from None

        patterns = " ".join(f"*{ext}" for ext in cls._FILE_READERS)
        root = None
        tcl_error = None
        try:
            root = tkinter.Tk()
            root.withdraw()
            # Without this the dialog can open behind the browser window when
            # called from a Jupyter kernel.
            root.attributes('-topmost', True)
            paths = filedialog.askopenfilenames(
                parent=root, title="Select data file(s) to load",
                filetypes=[("Data files", patterns), ("All files", "*.*")])
        except tkinter.TclError as e:
            # Tk is installed but cannot reach a display — a Wayland-only
            # session without XWayland lands here, where zenity still works.
            # Dealt with below rather than here so the dead root is torn down
            # before the fallback dialog opens.
            tcl_error = e
        finally:
            if root is not None:
                # Leaving the root alive hangs the *next* dialog in the same
                # kernel, so tear it down even when the dialog raised.
                try:
                    root.update()
                    root.destroy()
                except Exception:
                    pass

        if tcl_error is not None:
            picked = cls._pick_files_native()
            if picked is not None:
                return picked
            raise RuntimeError(cls._no_picker_message(
                f"tkinter could not open a display ({tcl_error})")) from None

        # Cancel gives '' on some platforms and () on others.
        return tuple(paths) if paths else ()

    def load(self, source=None, title=None, set_name_column=None, set_idx_column=None,
             load_cols_as_vars=False, combined=False, read_kwargs=None):
        """Load datasets from DataFrames, filepaths, dicts, or numpy arrays.

        Called with no ``source`` (or ``source=None``), opens the OS file
        browser and loads whatever is picked there — several files at once is
        fine, and cancelling loads nothing.

        ``source`` may be a single item or a list of items; each is coerced to a DataFrame
        and loaded via load_df. Supported files: .csv, .tsv, .txt, .xlsx, .xls, .json, .parquet
        (``read_kwargs`` is passed through to the pandas reader). For a list, ``combined=True``
        merges everything into one set while ``combined=False`` loads each separately. A file's
        title defaults to its filename when no title or SETNUMBER/INDEX split column is present.
        """
        if source is None:
            source = self._pick_files()
            if not source:
                print("No files selected.")
                return

        sources = list(source) if isinstance(source, (list, tuple)) else [source]
        coerced = [self._coerce_to_df(s, read_kwargs) for s in sources]

        def resolve_title(frame, default_title):
            if title is not None or set_idx_column is not None:
                return title
            if "SETNUMBER" in frame.columns or "INDEX" in frame.columns:
                return None
            return default_title

        def annotate(created, src_path):
            # File-backed sets remember where they came from so save_session
            # can store a reference instead of embedding the rows.
            if src_path is None:
                return
            for ds in created:
                ds.file_path = str(src_path)
                if ds._source is not None:
                    ds._source['path'] = str(src_path)
                    ds._source['read_kwargs'] = dict(read_kwargs) if read_kwargs else None

        if combined:
            frame = pd.concat([df for df, _, _ in coerced], ignore_index=True)
            default_title = next((dt for _, dt, _ in coerced if dt), None)
            created = self.load_df(frame, title=resolve_title(frame, default_title),
                                   set_name_column=set_name_column, set_idx_column=set_idx_column,
                                   load_cols_as_vars=load_cols_as_vars)
            # A combined load is only reproducible from a single source file.
            if len(coerced) == 1:
                annotate(created, coerced[0][2])
            return

        for frame, default_title, src_path in coerced:
            created = self.load_df(frame, title=resolve_title(frame, default_title),
                                   set_name_column=set_name_column, set_idx_column=set_idx_column,
                                   load_cols_as_vars=load_cols_as_vars)
            annotate(created, src_path)

    def _coerce_to_df(self, source, read_kwargs=None):
        """Coerce a single source into a (DataFrame, default_title, path) triple.
        default_title is the filename stem for file inputs; path is the resolved
        source file, or None for in-memory sources."""
        if isinstance(source, pd.DataFrame):
            return self._dedupe_columns(source).copy(), None, None
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"No such file: {path}")
            reader = self._FILE_READERS.get(path.suffix.lower())
            if reader is None:
                raise ValueError(
                    f"Unsupported file type '{path.suffix}' for {path}; "
                    f"supported: {', '.join(self._FILE_READERS)}")
            return reader(path, read_kwargs or {}), path.stem, path.resolve()
        if isinstance(source, dict):
            return pd.DataFrame(source), None, None
        if isinstance(source, np.ndarray):
            return pd.DataFrame(source), None, None
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

    # ------------------------------------------------------------------
    # Sessions (save/restore data sources + formatting)
    # ------------------------------------------------------------------

    # Per-set attributes captured by save_session and restored verbatim by
    # load_session (title/query/select/order are handled separately because
    # they need special treatment on restore).
    _SESSION_SET_FORMAT_ATTRS = (
        'color', 'marker', 'linestyle', 'markersize', 'linewidth', 'edgewidth',
        'alpha', 'alpha_marker', 'alpha_line', 'edge_color', 'fill', 'hue',
        'hue_palette', 'hue_order', 'reg_order', 'style', 'zorder',
        'plot_type', 'set_type', 'data_type', 'display_parms', 'delta_sets',
        'sig_figs', 'decimals',
    )

    # Notebook-level formatting captured by save_session. plot_style and
    # default_format are handled separately (style install order matters and
    # default_format carries the _MARKER_BY_INDEX sentinel).
    _SESSION_NB_ATTRS = (
        'darkmode', 'suptitle', 'footer', 'plot_title', 'x_label', 'y_label',
        'display_parms', 'axis_limits', 'lines', 'highlights',
        'parm_description_dict', 'variable_formats', 'color_map', 'marker_map',
        'figsize', 'plot_defaults', 'plot_size', 'plot_size_per_subplot',
        'grid_format', 'watermark_format',
        'suptitle_size', 'footer_size', 'legend_size', 'axes_title_size',
        'axes_tick_size', 'subplot_title_size', 'colorbar_size', 'hover_size',
        'table_header_size', 'table_cell_size',
        'static_images', 'static_scale', 'copy_buttons',
    )

    _SESSION_MARKER_SENTINEL = '__MARKER_BY_INDEX__'

    @staticmethod
    def _session_json_default(o):
        """json.dump fallback for numpy/pandas values inside session state."""
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        if isinstance(o, (set, frozenset)):
            return sorted(map(str, o))
        if isinstance(o, Path):
            return str(o)
        return str(o)

    def _record_plot_call(self, method, frame_locals):
        """Remember the plotting call now in progress as ``{'method', 'kwargs'}``.

        Called by each public plotting method right after it has resolved
        ``x``/``y`` from ``last_x``/``last_y``, with its ``locals()``; the
        method's signature picks the parameters out (``**kwargs`` flattened),
        and arguments still at their signature default are dropped so the
        record reads like the call the user typed. ``save_png`` embeds this
        alongside the session so ``load_session`` can replay the exact plot.
        """
        sig = inspect.signature(getattr(type(self), method))
        call = {}
        for name, prm in sig.parameters.items():
            if name == 'self' or prm.kind == prm.VAR_POSITIONAL:
                continue
            if prm.kind == prm.VAR_KEYWORD:
                call.update(frame_locals.get(name) or {})
                continue
            value = frame_locals.get(name)
            if prm.default is not prm.empty:
                try:
                    if value is prm.default or (
                            not isinstance(value, (np.ndarray, pd.Series))
                            and value == prm.default):
                        continue
                except (TypeError, ValueError):
                    pass
            call[name] = value
        self._last_plot_call = {'method': method, 'kwargs': call}

    def _set_source_frame(self, ds):
        """This set's rows, unmasked, restricted to its own columns — what an
        embedded session entry stores."""
        cdf = self._combined_df
        pos = self._set_positions(ds._set_id)
        return cdf.iloc[pos, ds._own_col_positions()].reset_index(drop=True)

    @staticmethod
    def _apply_synth_cols(df, synth_cols):
        """Re-add the columns :meth:`load_df` synthesised at load time
        (recorded in ``_source['synth_cols']``) to a freshly re-read file
        frame, so a file-referenced session set restores with the same
        columns it had originally."""
        for col, spec in (synth_cols or {}).items():
            if col in df.columns:
                continue
            kind = spec.get('kind') if isinstance(spec, dict) else None
            if kind == 'index':
                df[col] = df.index
            elif kind == 'const':
                df[col] = spec.get('value')
        return df

    @staticmethod
    def _session_relpath(file_path, session_path):
        """``file_path`` relative to the session file's directory, or None
        when no relative path exists (e.g. different drives)."""
        try:
            return os.path.relpath(str(file_path), str(Path(session_path).resolve().parent))
        except ValueError:
            return None

    @staticmethod
    def _resolve_session_file(src, session_path):
        """Locate a file-referenced session set's source file: the absolute
        path as saved, falling back to the saved session-relative path so a
        session file moved together with its data still loads."""
        fpath = Path(src['path'])
        if fpath.is_file():
            return fpath
        rel = src.get('rel_path')
        if rel:
            candidate = Path(session_path).resolve().parent / rel
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"source file missing: {fpath}")

    @staticmethod
    def _embed_frame(df):
        """Encode a DataFrame as a JSON-safe dict, column-wise with recorded
        dtypes. Serialized by Python's json (not pandas.to_json) because the
        stdlib writes floats via repr — an exact float64 round-trip, where
        to_json truncates to at most 15 significant digits."""
        columns, dtypes, data = [], {}, {}
        for i in range(df.shape[1]):
            s = df.iloc[:, i]
            name = str(df.columns[i])
            columns.append(name)
            dtypes[name] = str(s.dtype)
            if pd.api.types.is_datetime64_any_dtype(s):
                vals = [None if pd.isna(v) else v.isoformat() for v in s]
            else:
                # tolist() converts numpy scalars to Python ones; float NaN
                # survives json (NaN literal) but NaT/pd.NA would not.
                vals = [None if v is pd.NaT or v is pd.NA else v
                        for v in s.tolist()]
            data[name] = vals
        return {'columns': columns, 'dtypes': dtypes, 'data': data}

    @staticmethod
    def _unembed_frame(payload):
        """Rebuild a DataFrame from :meth:`_embed_frame` output."""
        series = {}
        for name in payload['columns']:
            dtype = payload.get('dtypes', {}).get(name)
            s = pd.Series(payload['data'][name],
                          dtype='object' if dtype == 'object' else None)
            if dtype and dtype != 'object':
                try:
                    s = s.astype(dtype)
                except (TypeError, ValueError):
                    pass
            series[name] = s
        return pd.DataFrame(series)[payload['columns']]

    def save_session(self, path, embed_data=True, parms=None):
        """Save a reloadable plotting session: every set's data source, query
        and formatting, plus the notebook-level formatting state.

        Each set is saved either as a **file reference** (path + read options +
        group key, when the set was loaded from a file that still exists and
        the set's columns are unchanged since loading) or as **embedded data**
        (the rows themselves, stored in the session file). Restore with
        :meth:`load_session`. File references store both the absolute path
        and the path relative to the session file, so a session moved
        together with its data files still loads.

        The most recent plotting call is recorded too, so ``load_session``
        replays it and the figure comes back with the data — the same thing
        :meth:`save_png` embeds in a PNG.

        Parameters
        ----------
        path : str | Path
            Destination ``.json`` file.
        embed_data : bool | 'all'
            ``True`` (default): embed the rows of any set that cannot be
            reproduced from a file — in-memory loads, derived sets (delta/
            combine), df-replaced sets, sets with added or overwritten
            columns, or sets whose source file has gone missing. ``False``: save file references
            only; non-reproducible sets are skipped with a warning. ``'all'``:
            embed every set's rows, making the session file fully
            self-contained (survives source-file edits/deletion).
        parms : str | list of str, optional
            Whitelist of columns to embed. By default an embedded set stores
            every column it owns; with a whitelist only these columns (plus
            any a set's query, hue, style or recorded plot call needs, added
            automatically) are written, which keeps sessions small when the
            data is wide. Has no
            effect on file-referenced sets, which re-read the whole file. A
            set with none of the columns is skipped with a warning.

        Notes
        -----
        Column writes through ``ds[col] = ...``, :meth:`set_column` and
        :meth:`add_column` are tracked, but in-place edits made directly on
        ``nb.df`` (``nb.df.loc[...] = ...``) are not detectable and would be
        lost on reload — pass ``embed_data='all'`` to be safe. Queries are
        saved as expressions and re-run on load.
        """
        path = Path(path)
        session, stats = self._build_session(embed_data, path,
                                             plot_call=self._last_plot_call,
                                             parms=parms)
        with open(path, 'w') as fh:
            json.dump(session, fh, indent=1, default=self._session_json_default)
        print(f"Saved session to {path}: " + self._session_summary(stats)
              + self._plot_call_note())

    def _plot_call_note(self):
        """The hint both session writers append when there is no plot to replay."""
        return "" if self._last_plot_call else "; no plot call recorded"

    @staticmethod
    def _session_summary(stats):
        n_sets, n_file, n_embedded, skipped = stats
        msg = f"{n_sets} set(s) ({n_file} file-referenced, {n_embedded} embedded)"
        if skipped:
            msg += (f"; skipped {len(skipped)} set(s) with no file source "
                    f"{skipped} (use embed_data=True to include them)")
        return msg

    @staticmethod
    def _session_needed_cols(ds, plot_call=None):
        """Columns a whitelisted embed must keep for ``ds`` to restore and
        replot: any its query expression names, its hue/style columns, and the
        columns the recorded plot call passes (x, y, z, style_by, ...)."""
        own = set(ds._own_cols)
        needed = set()
        if ds._query:
            tokens = {m[0] or m[1] for m in
                      re.findall(r"`([^`]+)`|([A-Za-z_]\w*)", ds._query)}
            needed |= own & tokens
        for attr in ('hue', 'style'):
            val = getattr(ds, attr, None)
            if isinstance(val, str) and val in own:
                needed.add(val)
        for val in (plot_call or {}).get('kwargs', {}).values():
            vals = val if isinstance(val, (list, tuple)) else [val]
            needed |= own & {v for v in vals if isinstance(v, str)}
        return needed

    def _build_session(self, embed_data, path, plot_call=None, parms=None):
        """Assemble the session dict :meth:`save_session` writes (and
        :meth:`save_png` embeds). ``path`` is the file the session will live
        in — file references are also stored relative to it. ``plot_call``
        (see :meth:`_record_plot_call`) is included when given so the plot can
        be replayed; ``parms`` whitelists the columns embedded sets store
        (see :meth:`save_session`). Returns
        ``(session, (n_sets, n_file, n_embedded, skipped))``.
        """
        embed_all = (embed_data == 'all')
        keep = None if parms is None else set(coerce_display_parms(parms))
        set_entries, n_file, n_embedded, skipped = [], 0, 0, []

        for ds in self.sets:
            src = ds._source or {}
            file_ok = False
            if not embed_all and src.get('path'):
                file_ok = Path(src['path']).is_file()
                if file_ok and src.get('load_cols') is not None:
                    # Columns added since load (ds['NEW']=..., set_column, ...)
                    # would not come back from the file — embed instead.
                    file_ok = set(src['load_cols']) == set(ds._own_cols)
                if file_ok and src.get('modified'):
                    # A loaded column was overwritten through the write APIs
                    # (ds[col]=..., set_column, add_column) — embed instead.
                    file_ok = False
                if file_ok and src.get('read_kwargs'):
                    try:
                        json.dumps(src['read_kwargs'])
                    except TypeError:
                        file_ok = False

            if file_ok:
                source = {'kind': 'file', 'path': str(src['path']),
                          'rel_path': self._session_relpath(src['path'], path),
                          'read_kwargs': src.get('read_kwargs'),
                          'set_idx_column': src.get('set_idx_column'),
                          'group_key': src.get('group_key'),
                          'synth_cols': src.get('synth_cols') or None}
                n_file += 1
            elif embed_data:
                frame = self._set_source_frame(ds)
                if keep is not None:
                    wanted = keep | self._session_needed_cols(ds, plot_call)
                    cols = [c for c in frame.columns if c in wanted]
                    if not cols:
                        print(f"Warning: skipping set {ds.index} ({ds.title}): "
                              f"none of the whitelisted columns {sorted(keep)} in it")
                        skipped.append(ds.index)
                        continue
                    frame = frame[cols]
                source = {'kind': 'embedded', 'frame': self._embed_frame(frame)}
                n_embedded += 1
            else:
                skipped.append(ds.index)
                continue

            entry = {
                'title': ds.title,
                'query': ds._query,
                'select': ds._select,
                'order': ds._order,
                'source': source,
                'format': {a: getattr(ds, a, None)
                           for a in self._SESSION_SET_FORMAT_ATTRS},
            }
            set_entries.append(entry)

        nb_state = {a: getattr(self, a, None) for a in self._SESSION_NB_ATTRS}
        nb_state['plot_style'] = getattr(self, 'plot_style', None)
        nb_state['default_format'] = {
            k: (self._SESSION_MARKER_SENTINEL if v is _MARKER_BY_INDEX else v)
            for k, v in self.default_format.items()}

        session = {'unichart_session': 1,
                   'saved': datetime.now().isoformat(timespec='seconds'),
                   'notebook': nb_state,
                   'sets': set_entries}
        if plot_call is not None:
            session['plot_call'] = plot_call
        return session, (len(set_entries), n_file, n_embedded, skipped)

    def load_session(self, path, restore_notebook_format=True, replay=True):
        """Restore a session saved by :meth:`save_session` — or embedded in a
        PNG by :meth:`save_png` — reload every set from its file reference or
        embedded data, then reapply titles, queries, select flags and all
        formatting. The plot call stored with the session is then replayed
        (``replay=True``) so the figure comes back as ``last_fig`` and is
        displayed.

        Sets are appended after any already-loaded sets (load into a fresh
        notebook to reproduce the saved indices exactly; ``delta_sets``
        base/study references are shifted to match the new positions either
        way). File-referenced sets re-read their source file (by absolute
        path, else by the saved session-relative path), so edits to the file
        since saving show up — and a vanished file or group key means that
        set is skipped with a warning.

        Parameters
        ----------
        path : str | Path
            Session ``.json`` file written by :meth:`save_session`.
        restore_notebook_format : bool
            Also restore notebook-level formatting (plot style, color/marker
            maps, default format, labels, lines/highlights, axis limits, font
            sizes, ...). Default True; pass False to keep the current
            notebook-level settings and only load the sets.
        replay : bool
            When the session records a plot call (both :meth:`save_session`
            and :meth:`save_png` record the last one), call that plotting
            method again with the same arguments after restoring. The figure is
            displayed and cached as ``last_fig``. A call that will not replay —
            a kwarg JSON could not round-trip, say — warns and leaves the
            restored data and formatting in place. Pass False to only restore
            the data and formatting.

        Returns the list of Datasets created.
        """
        with open(path, 'rb') as fh:
            head = fh.read(len(_PNG_SIGNATURE))
        if head == _PNG_SIGNATURE:
            session = read_png_session(path)
            if session is None:
                raise ValueError(f"{path} carries no embedded unichart session "
                                 "(saved without embed_session, or not by unichart).")
        else:
            with open(path) as fh:
                session = json.load(fh)
        if 'unichart_session' not in session:
            raise ValueError(f"{path} is not a unichart session file.")

        created = self._restore_session(session, path, restore_notebook_format)
        base_offset = len(self.sets) - len(created)
        print(f"Loaded session from {path}: {len(created)} set(s) restored"
              + (f" (appended after {base_offset} existing)" if base_offset else ""))

        call = session.get('plot_call')
        if replay and isinstance(call, dict) and call.get('method'):
            method = getattr(self, call['method'], None)
            if method is None:
                print(f"Warning: cannot replay unknown plot method {call['method']!r}")
            else:
                kwargs = dict(call.get('kwargs') or {})
                try:
                    result = method(**kwargs)
                except Exception as exc:                      # noqa: BLE001
                    # JSON has no type for a tuple limit or a numpy scalar, and
                    # _session_json_default falls back to str(o) — so a kwarg can
                    # come back as something the method won't take. The data and
                    # formatting are already restored and are what matter; warn
                    # the way a missing source file does rather than losing them
                    # to a traceback.
                    print(f"Warning: could not replay {call['method']}(): {exc}")
                else:
                    if result is not None and _display_renders():
                        display(result)
        return created

    @classmethod
    def from_session(cls, path, **kwargs):
        """A fresh notebook with ``path`` (a session ``.json`` or a PNG from
        :meth:`save_png`) loaded — ``UnichartNotebook.from_session('plot.png')``
        remakes the plot from the image alone. ``kwargs`` go to
        :meth:`load_session`."""
        nb = cls()
        nb.load_session(path, **kwargs)
        return nb

    def _restore_session(self, session, path, restore_notebook_format=True):
        """Apply a session dict (see :meth:`_build_session`) to this notebook;
        ``path`` locates session-relative file references. Returns the
        Datasets created."""
        base_offset = len(self.sets)
        frames, kept_entries, file_cache, resolved_paths = [], [], {}, {}

        for i, entry in enumerate(session.get('sets', [])):
            src = entry.get('source') or {}
            kind = src.get('kind')
            try:
                if kind == 'embedded':
                    df = self._unembed_frame(src['frame'])
                elif kind == 'file':
                    fpath = self._resolve_session_file(src, path)
                    resolved_paths[i] = str(fpath)
                    cache_key = (str(fpath),
                                 json.dumps(src.get('read_kwargs') or {},
                                            sort_keys=True, default=str),
                                 json.dumps(src.get('synth_cols') or {},
                                            sort_keys=True, default=str))
                    if cache_key not in file_cache:
                        reader = self._FILE_READERS.get(fpath.suffix.lower())
                        if reader is None:
                            raise ValueError(f"unsupported file type: {fpath.suffix}")
                        file_cache[cache_key] = self._apply_synth_cols(
                            self._dedupe_columns(
                                reader(fpath, src.get('read_kwargs') or {})),
                            src.get('synth_cols'))
                    df = file_cache[cache_key]
                    idx_col = src.get('set_idx_column')
                    if idx_col is not None:
                        key = src.get('group_key')
                        if idx_col not in df.columns:
                            raise ValueError(f"split column '{idx_col}' missing from {fpath}")
                        groups = dict(iter(df.groupby(idx_col)))
                        match = key if key in groups else next(
                            (k for k in groups if str(k) == str(key)), None)
                        if match is None:
                            raise ValueError(f"group {key!r} not found in {fpath}")
                        df = groups[match]
                    df = df.copy()
                else:
                    raise ValueError(f"unknown source kind: {kind!r}")
            except Exception as e:
                print(f"Warning: skipping session set {i} "
                      f"({entry.get('title', '?')}): {e}")
                continue
            frames.append((df, entry.get('title') or 'Untitled'))
            kept_entries.append((i, entry))

        created = self._register_sets(frames)

        for ds, (entry_pos, entry) in zip(created, kept_entries):
            fmt = entry.get('format') or {}
            for attr in self._SESSION_SET_FORMAT_ATTRS:
                if attr not in fmt:
                    continue
                value = fmt[attr]
                if attr == 'delta_sets' and isinstance(value, dict):
                    value = dict(value)
                    for ref in ('base', 'study'):
                        if isinstance(value.get(ref), int):
                            value[ref] += base_offset
                try:
                    setattr(ds, attr, value)
                except (ValueError, TypeError) as e:
                    print(f"Warning: set {ds.index}: could not restore "
                          f"{attr}={value!r} ({e})")
            # 'index' is a valid order sentinel the setter rejects; assign
            # directly, as _inherit_set_format does.
            ds._order = entry.get('order')
            if entry.get('query'):
                try:
                    ds.query = entry['query']
                except ValueError as e:
                    print(f"Warning: set {ds.index}: could not restore query "
                          f"{entry['query']!r} ({e})")
            # After the query: an emptied query flips select off, and the
            # saved flag is the state the user actually had.
            ds._select = bool(entry.get('select', True))

            # Re-attach provenance so the restored session can be re-saved.
            src = entry.get('source') or {}
            if src.get('kind') == 'file':
                fpath = resolved_paths.get(entry_pos, src['path'])
                ds.file_path = fpath
                ds._source = {'path': fpath,
                              'read_kwargs': src.get('read_kwargs'),
                              'set_idx_column': src.get('set_idx_column'),
                              'group_key': src.get('group_key'),
                              'synth_cols': dict(src.get('synth_cols') or {}),
                              'load_cols': set(ds._own_cols)}

        if restore_notebook_format and isinstance(session.get('notebook'), dict):
            nb_state = session['notebook']
            style = nb_state.get('plot_style')
            if style:
                # Install the style first; explicit saved values below win.
                self._apply_style_defaults(style)
            if isinstance(nb_state.get('default_format'), dict):
                self.default_format = {
                    k: (_MARKER_BY_INDEX if v == self._SESSION_MARKER_SENTINEL else v)
                    for k, v in nb_state['default_format'].items()}
            for attr in self._SESSION_NB_ATTRS:
                if attr in nb_state:
                    setattr(self, attr, nb_state[attr])
            # JSON turns tuples into lists; these two are used as tuples.
            for attr in ('figsize', 'plot_size'):
                val = getattr(self, attr, None)
                if isinstance(val, list):
                    setattr(self, attr, tuple(val))

        return created

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
        # assigned values happen to be NaN. Sets that already owned it have
        # had source-loaded values overwritten.
        for ds in self.sets:
            if name in ds._own_cols:
                ds._mark_source_modified()
            ds._own_cols.add(name)
        self._snapshot_columns()
        self._reapply_all_queries()

    def set_column(self, uset_slice, col, value):
        """Write `value` into `col` for the selected sets only.

        Adds the column (NaN-filled elsewhere) if it doesn't exist. `value` may be:

        - a **scalar**, broadcast to every row of every selected set;
        - an **array-like** sized to the *total* rows of the selected sets, laid
          out in set order (a Series is consumed positionally, by length — it is
          not aligned by label);
        - a **callable** taking each target `Dataset` in turn and returning a
          scalar or a per-set array-like — the per-set-formula form::

              nb.set_column([0, 2], 'THRUST', lambda d: k[d.index] * d['N1'] ** 2)

          Each set is written exactly as ``d[col] = fn(d)`` would write it, so a
          returned Series aligns by label and a set carrying an active query
          leaves its filtered-out rows NaN.
        """
        targets = self._get_uset_slice(uset_slice)
        if not targets:
            return
        if callable(value):
            # Delegate per set to the dataset write path, which already owns
            # label alignment, length checks, dtype selection and ownership.
            # Queries are refreshed once at the end, not once per set, so this
            # stays a single logical write like the value path below.
            for ds in targets:
                ds._df_full._assign(col, value(ds), reapply=False)
            self._reapply_all_queries()
            return
        target_ids = {ds._set_id for ds in targets}
        cdf = self._combined_df
        mask = cdf[_SET_ID_COL].isin(target_ids)
        if hasattr(value, '__len__') and not isinstance(value, str):
            n = int(mask.sum())
            if len(value) != n:
                raise ValueError(
                    f"Length mismatch assigning '{col}': {len(value)} values for {n} rows.")
            assign_val = np.asarray(value)
        else:
            assign_val = value
        # Same dtype rule as ``ds[col] = ...``: numeric writes land in a float
        # column, and only genuinely non-numeric data creates (or widens an
        # existing numeric column to) object dtype.
        incoming_object = _DatasetFrameView._is_object_like(assign_val)
        if col not in cdf.columns:
            cdf[col] = pd.Series(
                None if incoming_object else np.nan,
                index=cdf.index,
                dtype=object if incoming_object else float,
            )
        elif incoming_object and cdf[col].dtype != object:
            cdf[col] = cdf[col].astype(object)
        cdf.loc[mask, col] = assign_val
        # The targeted sets claim the column — including the case of filling
        # NaNs into a column some other set introduced. Sets that already
        # owned it have had source-loaded values overwritten.
        for ds in targets:
            if col in ds._own_cols:
                ds._mark_source_modified()
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
            ds.title_format = f"{ds.index}: {ds.title}"

    # ------------------------------------------------------------------
    # Selection & Filtering
    # ------------------------------------------------------------------
    # One range-shorthand token: "1:10", ":5", "5:", "::2", "-3:", or a bare int.
    # Only ``:`` separates a range, which leaves ``-`` free for negative indices.
    _RANGE_TOKEN_RE = re.compile(r'^\s*-?\d*\s*(?::\s*-?\d*\s*){0,2}$')

    @classmethod
    def _is_range_str(cls, text):
        """True if ``text`` is range shorthand (``"1:10"``, ``"0,2,5:8"``, ...).

        Used both to parse selectors and to stop :meth:`_var_targets` from
        mistaking a range string for a variable name.
        """
        if not isinstance(text, str) or not text.strip():
            return False
        tokens = text.split(',')
        return all(t.strip() and cls._RANGE_TOKEN_RE.match(t) for t in tokens)

    @staticmethod
    def _parse_range_token(token):
        """Turn one shorthand token into an ``int`` or a ``slice``."""
        token = token.strip()
        if ':' not in token:
            return int(token)
        parts = [p.strip() for p in token.split(':')]
        return slice(*(int(p) if p else None for p in parts))

    def _get_uset_slice(self, uset_slice):
        """Normalize a selector into a list of Dataset objects.

        Accepts:
            None | 'all'    -> all datasets
            int             -> dataset at that index (negative indices count
                               from the end, so -1 is the last dataset)
            slice | range   -> the datasets at those positions
            str             -> range shorthand: comma-separated ints and/or
                               ``start:stop[:step]`` slices, e.g. ``"1:10"``
                               (sets 1-9), ``"0,3,7:"``, ``"::2"``, ``"-3:"``
            Dataset         -> wrapped in a list
            list            -> mixed list of any of the above
        Unknown inputs print a warning and return [].

        Slice/range selectors follow Python semantics: the stop bound is
        exclusive and out-of-range bounds clamp rather than raise, so
        ``"1:999"`` runs to the last set. A bare int that is out of range still
        returns ``[]``.

        Note: dataset *titles* are deliberately not accepted as selectors. A
        non-numeric string is reserved for variable/parameter targeting in the
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

        if isinstance(uset_slice, slice):
            return list(self.sets[uset_slice])

        if isinstance(uset_slice, str):
            if not self._is_range_str(uset_slice):
                print(f"Warning: don't know how to interpret {uset_slice!r} as a "
                      f"dataset selector. Expected 'all' or range shorthand such "
                      f"as '1:10' or '0,3,7:'.")
                return []
            return self._get_uset_slice(
                [self._parse_range_token(t) for t in uset_slice.split(',')])

        if isinstance(uset_slice, (list, tuple, set, range)):
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

    @classmethod
    def _var_targets(cls, target):
        """Autodetect whether a formatting-setter target names variable(s).

        Dataset selectors are ints, ``'all'``/None, slices/ranges, range
        shorthand strings (``"1:10"``), ``Dataset`` objects, or lists thereof.
        Since titles are not valid selectors, any *other* bare string — or a
        list/tuple of such strings — unambiguously denotes variable/parameter
        name(s).

        Returns the list of variable names when ``target`` names variables,
        otherwise ``None`` (meaning: treat as a dataset selector).
        """
        def _is_var_name(t):
            if not isinstance(t, str) or t == 'all':
                return False
            # Only the *new* shorthand spellings (they contain ':' or ',') are
            # claimed as dataset selectors. A bare numeric string stays a
            # variable name, so a column literally called '2020' still formats.
            if (':' in t or ',' in t) and cls._is_range_str(t):
                return False
            return True

        if _is_var_name(target):
            return [target]
        if isinstance(target, (list, tuple)) and target and all(
                _is_var_name(t) for t in target):
            return list(target)
        return None

    @staticmethod
    def _var_names(variable):
        """Normalize a variable argument to a list of variable names.

        A single name (a string, or any other non-iterable column key)
        becomes a one-element list; a non-string iterable — list, tuple,
        set, pandas Index — is taken as a collection of names. Unlike
        :meth:`_var_targets` this is used where the argument is *known* to
        name variables, so no dataset-selector autodetection happens.
        """
        if isinstance(variable, str) or not hasattr(variable, '__iter__'):
            names = [variable]
        else:
            names = list(variable)
        for name in names:
            try:
                hash(name)
            except TypeError:
                raise TypeError("variable names must be hashable, got "
                                f"{name!r}") from None
        return names

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
        """Select the specified dataset(s), deselecting everything else.

        ``uset_slice`` accepts an index, a list of indices, a ``slice``/
        ``range``, or range shorthand as a string — see
        :meth:`_get_uset_slice`. Shorthand follows Python slice semantics
        (exclusive stop)::

            nb.select("1:10")     # sets 1-9
            nb.select("0,3,7:")   # set 0, set 3, and set 7 to the end
            nb.select("::2")      # every other set
            nb.select("-3:")      # the last three sets
        """
        for ds in self.sets: ds.select = False
        for ds in self._get_uset_slice(uset_slice):
            ds.select = True

    def selected(self):
        """Get the currently selected datasets."""
        return [ds for ds in self.sets if ds.select]

    def omit(self, uset_slice=None):
        """Deselect the given dataset(s), leaving the rest of the selection
        untouched. Takes the same selectors as :meth:`select`, e.g.
        ``nb.omit("5:8")``."""
        for ds in self._get_uset_slice(uset_slice):
            ds.select = False

    def restore(self, uset_slice=None):
        """Re-select the given dataset(s), e.g. ``nb.restore("5:8")``. Takes the
        same selectors as :meth:`select`."""
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
                or RGB/RGBA string ('rgb(255, 0, 0)'). Pass ``'reset'`` to
                restore the default: on a variable it clears the color
                override, on dataset(s) it restores the ``color_map`` color.
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
        nb.color('all', 'reset')            # back to the color_map colors
        """
        if isinstance(color_val, int) and not isinstance(color_val, bool):
            color_val = self._color_at(color_val)
        variables = self._var_targets(uset_slice)
        if variables is not None:
            for v in variables:
                self.var_format(v, color=color_val)
            return
        self._set_or_reset(uset_slice, 'color', color_val)

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
                to restore the default: on a variable it clears the marker
                override, on dataset(s) it restores the ``marker_map`` /
                ``set_default_format`` marker.
                - int: resolves to the marker that set index would be assigned
                from ``marker_map`` at load time, independent of any later
                restyling. E.g. marker(5, 2) gives set 5 the ``marker_map``
                marker of set 2.

        Examples
        --------
        nb.marker('all', 's')             # every dataset uses squares
        nb.marker('Pressure', '^')        # the Pressure variable uses triangles
        nb.marker('all', 'reset')         # back to the marker_map markers
        """
        if isinstance(marker_val, int) and not isinstance(marker_val, bool):
            marker_val = self._marker_at(marker_val)
        variables = self._var_targets(uset_slice)
        if variables is not None:
            for v in variables:
                self.var_format(v, marker=marker_val)
            return
        self._set_or_reset(uset_slice, 'marker', marker_val)

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
                             Pass ``'reset'`` to restore the default (clears a
                             variable override; restores the default linestyle
                             on dataset(s)).

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
        self._set_or_reset(uset_slice, 'linestyle', style_val)

    def markersize(self, uset_slice=None, size_val=None):
        """
        Set the marker size for the target dataset(s) or variable(s).

        The target is autodetected: a dataset selector (int / list / 'all' /
        None / ``Dataset``) sets it per dataset; a variable/parameter name (or
        list of names) applies a per-variable override via :meth:`var_format`.
        Pass ``'reset'`` to restore the default (clears a variable override;
        restores the default markersize on dataset(s)).
        """
        variables = self._var_targets(uset_slice)
        if variables is not None:
            for v in variables:
                self.var_format(v, markersize=size_val)
            return
        self._set_or_reset(uset_slice, 'markersize', size_val)

    def linewidth(self, uset_slice=None, width_val=None):
        """
        Set the line thickness for the target dataset(s) or variable(s).

        The target is autodetected: a dataset selector (int / list / 'all' /
        None / ``Dataset``) sets it per dataset; a variable/parameter name (or
        list of names) applies a per-variable override via :meth:`var_format`.
        Pass ``'reset'`` to restore the default (clears a variable override;
        restores the default linewidth on dataset(s)).
        """
        variables = self._var_targets(uset_slice)
        if variables is not None:
            for v in variables:
                self.var_format(v, linewidth=width_val)
            return
        self._set_or_reset(uset_slice, 'linewidth', width_val)

    # Smallest sensible value for each spelling of the precision knob:
    # 1 significant figure, or 0 decimal places (whole numbers).
    _PRECISION_MIN = {'sig_figs': 1, 'decimals': 0}

    def _set_precision(self, attr, uset_slice, value):
        """Shared body of :meth:`sig_figs` and :meth:`decimals`.

        The two are spellings of one knob — round a displayed value to n
        significant figures, or to n places after the point — so setting either
        clears the other, and ``'reset'`` clears both. ``attr`` says which one
        this call sets; everything else (the one-argument notebook-wide form,
        the selector + value form, the report, the sentinels) is identical.
        """
        other = 'decimals' if attr == 'sig_figs' else 'sig_figs'
        low = self._PRECISION_MIN[attr]

        def _valid(v):
            if isinstance(v, bool) or not isinstance(v, int) or v < low:
                print(f"{attr} must be a "
                      f"{'positive' if low else 'non-negative'} integer.")
                return False
            return True

        def _reset(targets):
            # Both attributes, so a reset always lands back on the notebook
            # default whichever spelling the set was carrying.
            for ds in self._get_uset_slice(targets):
                self._reset_set_attrs(ds, ('sig_figs', 'decimals'))

        def _describe(sig, dec):
            if sig:
                return f"{sig} sig figs"
            if dec is not None:
                return f"{dec} decimals"
            return "off"

        # Report: no arguments at all.
        if uset_slice is None and value is None:
            sig = self.default_format.get('sig_figs')
            dec = self.default_format.get('decimals')
            print(f"Display precision: {_describe(sig, dec)} (notebook default)")
            overrides = {ds.index: (ds.sig_figs, ds.decimals) for ds in self.sets
                         if (ds.sig_figs, ds.decimals) != (sig, dec)}
            if overrides:
                print("  per-set: " + ", ".join(
                    f"{i}: {_describe(s, d)}" for i, (s, d) in overrides.items()))
            return self.default_format.get(attr)

        # One positional argument is the notebook-wide form: nb.sig_figs(3).
        # A lone int would otherwise read as a set index with no value.
        if value is None:
            if isinstance(uset_slice, str):
                if uset_slice == 'reset':
                    self.default_format['sig_figs'] = None
                    self.default_format['decimals'] = None
                    _reset('all')
                    return
                # A selector with no value ('all', '1:3', ...) — the integer
                # message below would only confuse.
                print(f"{attr}({uset_slice!r}) needs a value too: "
                      f"nb.{attr}({uset_slice!r}, {low + 3}).")
                return
            if not _valid(uset_slice):
                print(f"To target sets, pass a selector and a value: "
                      f"nb.{attr}(0, {low + 3}).")
                return
            self.default_format[attr] = uset_slice
            self.default_format[other] = None
            self._set_or_reset('all', attr, uset_slice)
            return

        if self._var_targets(uset_slice) is not None:
            print(f"{attr} is a per-dataset setting, not a per-variable one. "
                  f"Use nb.{attr}(<sets>, n) or nb.{attr}(n) for all sets.")
            return
        if isinstance(value, str) and value == 'reset':
            _reset(uset_slice)
            return
        if not _valid(value):
            return
        self._set_or_reset(uset_slice, attr, value)

    def sig_figs(self, uset_slice=None, figs_val=None):
        """
        Set how many significant figures displayed values are rounded to.

        Applies wherever unichart *shows* a number rather than stores it: the
        cells of :meth:`table`, the statistics in :meth:`summary`, and the
        hover readouts of the plots (x/y/z values and the ``display_parms``
        lines). The data itself is never touched — ``table(output='df')`` and
        ``summary(output='df')`` still return full precision, as does
        ``ds.df``.

        Two forms, told apart by how many arguments you pass:

        * ``nb.sig_figs(3)`` — one value: the notebook-wide setting. It becomes
          the default for datasets loaded later *and* is applied to every set
          already loaded.
        * ``nb.sig_figs(2, 3)`` — a dataset selector then a value: only those
          sets, exactly like :meth:`markersize` and friends. Use this form
          (``nb.sig_figs(3, 4)``) when the number you mean is a set index.

        ``nb.sig_figs()`` reports the current precision and returns the
        notebook-wide value. ``'reset'`` in place of a value restores the
        built-in precision — for the named sets, or (one-argument form) for
        the notebook and every set.

        :meth:`decimals` is the same knob counting places after the point
        instead; a value either rounds to significant figures or to decimals,
        so setting one clears the other.

        Unlike the styling setters this is per *dataset* only; a variable name
        as the target is rejected rather than routed to :meth:`var_format`.

        Args:
            uset_slice (int, list, 'all', Dataset, or int): Dataset selector,
                or — alone — the notebook-wide number of significant figures.
            figs_val (int or 'reset'): Significant figures (>= 1), or 'reset'.

        Examples:
            nb.sig_figs(4)          # 4 sig figs everywhere
            nb.sig_figs(0, 6)       # set 0 shows 6, the rest keep the default
            nb.sig_figs(0, 'reset') # set 0 back to the notebook default
            nb.sig_figs('reset')    # built-in precision everywhere
        """
        return self._set_precision('sig_figs', uset_slice, figs_val)

    def decimals(self, uset_slice=None, dec_val=None):
        """
        Set how many decimal places displayed values are rounded to.

        The fixed-places sister of :meth:`sig_figs`, with the same reach (the
        cells of :meth:`table`, the statistics in :meth:`summary`, and the
        plots' hover readouts), the same two forms, and the same promise that
        only the *display* changes — ``output='df'`` and ``ds.df`` keep full
        precision. Trailing zeros are kept, so ``decimals=2`` shows ``1.5`` as
        ``1.50``, and ``decimals=0`` gives whole numbers.

        * ``nb.decimals(2)`` — one value: the notebook-wide setting, applied to
          the sets already loaded and inherited by those loaded later.
        * ``nb.decimals(1, 2)`` — a dataset selector then a value. Use this
          form when the number you mean is a set index.

        ``nb.decimals()`` reports the current precision. ``'reset'`` restores
        the built-in precision, for the named sets or (one argument) for the
        notebook and every set. Since a value is rounded either to significant
        figures or to decimal places, setting this clears :attr:`sig_figs`.

        Note that the Markdown output of ``table``/``summary`` re-renders plain
        numeric columns without the trailing zeros; the HTML table, the
        ``'fig'`` table and the hover readouts keep them.

        Args:
            uset_slice (int, list, 'all', Dataset, or int): Dataset selector,
                or — alone — the notebook-wide number of decimal places.
            dec_val (int or 'reset'): Decimal places (>= 0), or 'reset'.

        Examples:
            nb.decimals(2)          # two places everywhere
            nb.decimals(0, 3)       # set 0 shows three, the rest the default
            nb.decimals(0)          # whole numbers everywhere (one value form)
            nb.decimals('reset')    # built-in precision everywhere
        """
        return self._set_precision('decimals', uset_slice, dec_val)

    def zorder(self, uset_slice=None, z_val=None):
        """
        Set the draw order (z-order) for the specified dataset(s).

        Datasets are drawn in ascending ``zorder`` (default 0) with later
        traces on top; ties keep load order. So ``nb.zorder(2, 1)`` lifts set
        2 above everything else without touching the others, and a negative
        value pushes a set behind. Applies wherever sets share axes — plot,
        bar, box, hist, contour (incl. overlay sets) and plot_ymult; the
        per-dataset subplot variants keep their panel order. Legend order
        stays by set index regardless of zorder. Pass ``'reset'`` to restore
        the default (0).

        Args:
            uset_slice (int, list, 'all', or Dataset): Dataset selector.
            z_val (int or float): Draw priority, or ``'reset'``.
        """
        self._set_or_reset(uset_slice, 'zorder', z_val)

    def edgewidth(self, uset_slice, width_val):
        """
        Set the marker edge width (outline thickness) for the specified dataset(s).

        Args:
            uset_slice (int, list, or 'all'): The dataset index or indices to modify.
            width_val (int, float, or 'reset'): The thickness of the marker edge
                                                (>= 0), or 'reset' for the default.
        """
        self._set_or_reset(uset_slice, 'edgewidth', width_val)

    def fill(self, uset_slice, fill_val):
        """
        Set the fill state (whether markers are solid or hollow) for the specified dataset(s).

        Args:
            uset_slice (int, list, or 'all'): The dataset index or indices to modify.
            fill_val (bool, int, or str): True/False (or 'on'/'off', '1'/'0', 't'/'f')
                                          to enable or disable marker fill, or
                                          'reset' for the default.
        """
        self._set_or_reset(uset_slice, 'fill', fill_val)

    def hue(self, uset_slice, col_name):
        """
        Map a dataframe column to the color scale for the specified dataset(s).
        Pass 'reset' to remove the hue mapping.
        """
        self._set_or_reset(uset_slice, 'hue', col_name)

    def hue_palette(self, uset_slice, hue_palette):
        """
        Set the color scale/palette used when `hue` is mapped to a variable.
        Pass 'reset' to restore the default palette.
        """
        self._set_or_reset(uset_slice, 'hue_palette', hue_palette)

    def alpha(self, uset_slice=None, alpha_val=None):
        """
        Set the opacity (alpha) for the target dataset(s) or variable(s).

        The target is autodetected: a dataset selector (int / list / 'all' /
        None / ``Dataset``) sets it per dataset; a variable/parameter name (or
        list of names) applies a per-variable override via :meth:`var_format`.
        Pass ``'reset'`` to restore the default (clears a variable override;
        restores the default alpha on dataset(s)).
        """
        variables = self._var_targets(uset_slice)
        if variables is not None:
            for v in variables:
                self.var_format(v, alpha=alpha_val)
            return
        self._set_or_reset(uset_slice, 'alpha', alpha_val)

    @staticmethod
    def _check_alpha(name, value):
        """Validate a partial-opacity value: None/'reset' pass through,
        anything else must be a number in [0, 1]."""
        if value is None or (isinstance(value, str) and value == 'reset'):
            return value
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{name} must be a number in [0, 1] (or 'reset'), "
                            f"got {value!r}")
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
        return value

    def alpha_marker(self, uset_slice=None, alpha_val=None):
        """
        Set the opacity of just the *markers* for the target dataset(s) or
        variable(s), leaving the line at its ``alpha``.

        Same target autodetection as :meth:`alpha`. Pass ``'reset'`` (or
        ``None``) to go back to sharing the dataset's ``alpha``.

        Examples
        --------
        nb.alpha_marker(0, 0.3)          # faint points, full-strength line
        nb.alpha_marker('CHT1', 0.5)     # per-variable override
        nb.alpha_marker(0, 'reset')
        """
        alpha_val = self._check_alpha('alpha_marker', alpha_val)
        variables = self._var_targets(uset_slice)
        if variables is not None:
            for v in variables:
                self.var_format(v, alpha_marker=alpha_val if alpha_val is not None else 'reset')
            return
        self._set_or_reset(uset_slice, 'alpha_marker', alpha_val)

    def alpha_line(self, uset_slice=None, alpha_val=None):
        """
        Set the opacity of just the *line* for the target dataset(s) or
        variable(s), leaving the markers at their ``alpha``.

        Same target autodetection as :meth:`alpha`. Pass ``'reset'`` (or
        ``None``) to go back to sharing the dataset's ``alpha``.

        Examples
        --------
        nb.alpha_line(0, 0.3)             # faint connecting line, solid points
        nb.alpha_line('CHT1', 0.5)        # per-variable override
        nb.alpha_line(0, 'reset')
        """
        alpha_val = self._check_alpha('alpha_line', alpha_val)
        variables = self._var_targets(uset_slice)
        if variables is not None:
            for v in variables:
                self.var_format(v, alpha_line=alpha_val if alpha_val is not None else 'reset')
            return
        self._set_or_reset(uset_slice, 'alpha_line', alpha_val)

    def plot_type(self, uset_slice, type_val):
        """Set the plot type ('scatter', 'contour', 'histogram') for the
        specified dataset(s). Pass 'reset' to restore 'scatter'."""
        self._set_or_reset(uset_slice, 'plot_type', type_val)

    def reg_order(self, uset_slice, order):
        """
        Set a regression/trendline for the specified dataset(s).

        The fit is drawn as its own line trace in the dataset's color and
        linestyle; the raw series stays point-only while a trendline is
        active (a set ``linestyle`` no longer connects the raw points).

        Args:
            uset_slice (int, list, 'all', or ``Dataset``): The dataset(s) to modify.
            order: The fit spec — an int, a name, a ``(kind, param)`` pair, or
                'reset'/None to remove the trendline.

        Available specs
        ---------------
        ``n`` (int)            Least-squares polynomial of degree n (n <= 0 → no fit).
        'linear', 'lin'        Straight-line least-squares fit (degree 1).
        'quadratic'            Degree-2 polynomial.
        'cubic'                Degree-3 polynomial.
        'poly'                 Polynomial, degree 1 unless a param is given.
        'polyN'                Degree N, e.g. 'poly4'.
        'log', 'logarithmic'   y = a·ln(x) + b; needs points with x > 0.
        'exp', 'exponential'   y = a·e^(bx); needs points with y > 0.
        'power', 'pow'         y = a·x^b; needs points with x > 0 and y > 0.
        'lowess', 'loess'      Locally weighted smoothing, param = frac
                               (default 0.3). Requires ``statsmodels``.
        'spline', 'cubic_spline'
                               Univariate spline, param = degree k (default 3,
                               clamped to 1-5). Requires ``scipy``.
        'ma', 'moving_average', 'rolling'
                               Centered rolling mean, param = window (default
                               max(3, npoints // 20), clamped to 2-npoints).

        Pass a ``(kind, param)`` tuple/list to override the parameter, e.g.
        ``('lowess', 0.5)``, ``('ma', 10)``, ``('spline', 5)``, ``('poly', 4)``.
        Note that a bare ``True`` is ignored — pass a spec, not a flag.

        An unrecognized name or spec raises ``ValueError``. A fit that cannot be
        computed (too few points, or none surviving the x/y > 0 masks) is
        skipped silently; LOWESS without ``statsmodels``, and any unexpected
        failure inside the fit, warn instead.

        Each fit carries a short kind label — 'Linear', 'LS2'/'LS3'… for higher
        polynomials, 'Log', 'Exp', 'Power', 'LOWESS(0.30)', 'Spline3', 'MA(7)'.
        On a plot it names the fit trace ("<n>: <title> Fit (Linear)"), which is
        hidden from the legend; in :meth:`table` and :meth:`delta` it appears as
        the method label for values read off the curve. Those two methods take
        the same specs via their ``kind`` argument and fall back to each
        dataset's own ``reg_order`` when none is given.

        Examples
        --------
        nb.reg_order('all', 'linear')         # straight-line fit everywhere
        nb.reg_order(0, 3)                    # cubic polynomial on set 0
        nb.reg_order(1, 'exp')                # exponential fit on set 1
        nb.reg_order([0, 2], ('lowess', 0.5)) # smoother LOWESS on sets 0 and 2
        nb.reg_order('all', 'reset')          # remove the trendlines
        """
        self._set_or_reset(uset_slice, 'reg_order', order)

    def copy_format(self, source, targets=None):
        """
        Copy one dataset's formatting onto other dataset(s).

        Copies the full styling bundle — color, marker, linestyle, markersize,
        linewidth, edgewidth, alpha, edge_color, fill, hue, hue_palette,
        hue_order, style, display_parms, sig_figs and decimals — so the
        targets read as visual twins of the source. Nothing analytical or identifying carries over:
        titles, queries, the select flag, ``plot_type`` and ``reg_order`` all
        stay the target's own. Column-dependent attributes (``hue``,
        ``display_parms``) only copy where the target actually has those
        columns; otherwise the target keeps its current value.

        Args:
            source (int or Dataset): The single dataset to copy from
                (negative indices count from the end).
            targets (int, list, 'all', Dataset, or None): The dataset(s) to
                copy onto. None or 'all' targets every other dataset. The
                source itself is always skipped, so ``copy_format(0)`` restyles
                everything else to match set 0.

        Examples
        --------
        nb.copy_format(0, 2)          # set 2 now styled like set 0
        nb.copy_format(0, [1, 2])     # sets 1 and 2 styled like set 0
        nb.copy_format(-1)            # every other set styled like the last one
        """
        src = self._get_uset_slice(source)
        if len(src) != 1:
            raise ValueError(
                f"copy_format source must resolve to exactly one dataset, "
                f"got {len(src)} from {source!r}")
        src = src[0]
        for ds in self._get_uset_slice(targets):
            if ds is src:
                continue
            self._inherit_set_format(ds, src)

    # ------------------------------------------------------------------
    # Variable-level formatting overrides
    # ------------------------------------------------------------------
    def var_format(self, variable, color=None, marker=None, linestyle=None,
                   markersize=None, linewidth=None, alpha=None, style=None,
                   alpha_marker=None, alpha_line=None, reset=False):
        """
        Set persistent per-variable formatting overrides.

        ``variable`` may be a single variable/parameter name or a list of
        names, in which case every name gets the same overrides.

        Variable formatting takes precedence over Dataset formatting on a
        per-attribute basis — anything you don't set still falls back to
        the Dataset's value at plot time.

        There are two ways to undo an override:

        - ``var_format('CHT', color='reset')`` clears just that one attribute.
        - ``var_format('CHT', reset=True)`` clears *every* override on the
          variable(s), and ignores any other arguments passed in the same
          call (same convention as ``set_default_format(reset=True)``).

        `alpha_marker` / `alpha_line` set the opacity of just the markers or
        just the line (see :meth:`alpha_marker`, :meth:`alpha_line`); the
        other part keeps `alpha`.

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
        nb.var_format(['CHT1', 'CHT2'], marker='x')          # same style for both
        nb.var_format(['CHT1', 'CHT2'], marker='reset')      # clear it on both
        nb.var_format('Pressure', reset=True)                # drop all Pressure overrides
        nb.var_format(['CHT1', 'CHT2'], reset=True)          # ...on both

        Returns
        -------
        dict
            The variable's overrides for a single name, or a
            ``{name: overrides}`` dict when a list of names was passed.
        """
        if reset:
            self.clear_var_format(variable)
            if isinstance(variable, str) or not hasattr(variable, '__iter__'):
                return {}
            return {n: {} for n in self._var_names(variable)}
        if style is not None and style != 'reset' and style not in _OVERLAY_STYLES:
            raise ValueError(f"style must be one of {_OVERLAY_STYLES}, got {style!r}")
        names = self._var_names(variable)
        pairs = {'color': color, 'marker': marker, 'linestyle': linestyle,
                 'markersize': markersize, 'linewidth': linewidth, 'alpha': alpha,
                 'alpha_marker': alpha_marker, 'alpha_line': alpha_line,
                 'style': style}
        for name in names:
            fmt = self.variable_formats.setdefault(name, {})
            for k, v in pairs.items():
                if v is None:
                    continue
                if v == 'reset':
                    fmt.pop(k, None)
                else:
                    fmt[k] = v
            if not fmt:
                del self.variable_formats[name]
        if isinstance(variable, str) or not hasattr(variable, '__iter__'):
            return self.variable_formats.get(variable, {})
        return {n: self.variable_formats.get(n, {}) for n in names}

    def clear_var_format(self, variable=None):
        """Clear variable formatting for one variable or a list of variables.
        Pass None (or no arg) to clear everything.
        Equivalent to ``reset_format(vars=variable)`` / ``reset_format('vars')``."""
        if variable is None:
            self.variable_formats.clear()
        else:
            for name in self._var_names(variable):
                self.variable_formats.pop(name, None)

    def list_var_formats(self):
        """Pretty-print current variable-level formatting."""
        if not self.variable_formats:
            print("No variable-level formatting set.")
            return
        print("Variable-level formatting (overrides dataset attributes):")
        for var, fmt in self.variable_formats.items():
            items = ", ".join(f"{k}={v!r}" for k, v in fmt.items())
            print(f"  {var}: {items}")

    # ------------------------------------------------------------------
    # Resetting formatting — reset_format() is the single hub; every
    # formatting setter also accepts the universal 'reset' sentinel.
    # ------------------------------------------------------------------

    # Scope names reset_format understands, plus forgiving aliases.
    _RESET_SCOPES = ('sets', 'vars', 'lines', 'highlights', 'scales',
                     'fonts', 'plot_size', 'grid', 'watermark', 'defaults')
    _RESET_ALIASES = {'set': 'sets', 'var': 'vars', 'variable': 'vars',
                      'variables': 'vars', 'line': 'lines',
                      'highlight': 'highlights', 'scale': 'scales',
                      'limits': 'scales', 'axis_limits': 'scales',
                      'font': 'fonts', 'font_sizes': 'fonts',
                      'plotsize': 'plot_size', 'default': 'defaults',
                      'grids': 'grid', 'gridlines': 'grid',
                      'grid_format': 'grid', 'watermarks': 'watermark',
                      'watermark_format': 'watermark'}
    # What a plain reset_format() sweeps. 'defaults' (the values behind
    # set_default_format) is deliberately excluded — resetting applied
    # formatting shouldn't silently discard the user's chosen defaults;
    # ask for it by name or with 'all'.
    _RESET_DEFAULT_SWEEP = ('sets', 'vars', 'lines', 'highlights', 'scales',
                            'fonts', 'plot_size', 'grid', 'watermark')

    def _reset_set_attrs(self, ds, attrs=None):
        """Restore per-dataset formatting attribute(s) to the current defaults:
        color from ``color_map``, marker from ``marker_map`` (or the
        ``set_default_format`` marker), the rest from ``default_format``.
        ``attrs`` is an iterable of attribute names; None means the full
        formatting sweep (everything except ``plot_type``, which only resets
        when asked for by name so contour/histogram sets survive a bulk reset).
        """
        fmt = getattr(self, 'default_format', _DATASET_FORMAT_DEFAULTS)
        default_marker = fmt.get('marker', _MARKER_BY_INDEX)
        defaults = {
            'color':       lambda: self._color_at(ds.index),
            'marker':      lambda: (self._marker_at(ds.index)
                                    if default_marker is _MARKER_BY_INDEX
                                    else default_marker),
            'linestyle':   lambda: fmt.get('linestyle', None),
            'markersize':  lambda: fmt.get('markersize', 10),
            'linewidth':   lambda: fmt.get('linewidth', 2),
            'edgewidth':   lambda: fmt.get('edgewidth', 1),
            'alpha':       lambda: fmt.get('alpha', 1),
            'alpha_marker': lambda: fmt.get('alpha_marker', None),
            'alpha_line':    lambda: fmt.get('alpha_line', None),
            'edge_color':  lambda: fmt.get('edge_color', 'black'),
            'fill':        lambda: fmt.get('fill', True),
            'hue':         lambda: "",
            'hue_palette': lambda: fmt.get('hue_palette', 'Jet'),
            'hue_order':   lambda: None,
            'reg_order':   lambda: None,
            'zorder':      lambda: 0,
            'sig_figs':    lambda: fmt.get('sig_figs', None),
            'decimals':    lambda: fmt.get('decimals', None),
            'plot_type':   lambda: 'scatter',
        }
        if attrs is None:
            attrs = [a for a in defaults if a != 'plot_type']
        for attr in attrs:
            setattr(ds, attr, defaults[attr]())

    def _inherit_set_format(self, ds, source):
        """Copy every per-dataset formatting attribute from ``source`` onto ``ds``.

        Styling only — see :attr:`_INHERITED_FORMAT_ATTRS` — so a derived set
        (see :meth:`delta`) reads as a visual continuation of the set it was
        built from without inheriting anything analytical.

        Never copied: ``reg_order`` (a trendline is analysis, not appearance),
        ``plot_type``, and the identity/state attributes ``title``, ``index``,
        ``set_type``, ``delta_sets``, the select flag and any query — those stay
        the new set's own. Attributes that name a column (``hue``,
        ``hue_order``, ``display_parms``, ``order``) only carry over for columns
        that survived into ``ds``; the new set keeps its own default for the rest.
        """
        cols = set(ds.columns)
        src_hue = getattr(source, 'hue', '')
        hue_ok = bool(src_hue) and src_hue in cols
        for attr in _INHERITED_FORMAT_ATTRS:
            value = getattr(source, attr, None)
            if attr in ('hue', 'hue_order'):
                if not hue_ok:
                    continue          # stale hue column: keep the new set's default
            elif attr == 'display_parms':
                value = [p for p in (value or []) if p in cols]
                if not value:
                    continue
            setattr(ds, attr, value)

        # ``order`` validates against the target's columns, so an unmatched one
        # would raise on a set that is already registered. 'index' is a valid
        # sentinel the plotters honour but the setter rejects, hence the direct
        # assignment.
        src_order = getattr(source, 'order', None)
        if src_order == 'index' or (src_order and src_order in cols):
            ds._order = src_order

    def _set_or_reset(self, uset_slice, attr, value):
        """Assign a per-dataset formatting attribute across a selector,
        honoring the universal ``'reset'`` sentinel (restore that attribute
        to its current default via :meth:`_reset_set_attrs`)."""
        for ds in self._get_uset_slice(uset_slice):
            if isinstance(value, str) and value == 'reset':
                self._reset_set_attrs(ds, (attr,))
            else:
                setattr(ds, attr, value)

    def _reset_defaults(self):
        """Restore default_format, figsize, and the per-call plot defaults to
        built-ins. Shared by set_default_format(reset=True) and
        reset_format('defaults')."""
        self.default_format = dict(_DATASET_FORMAT_DEFAULTS)
        self.figsize = (12, 8)
        self.plot_defaults = {k: None for k in self.plot_defaults}
        # The plot style is a default too (it drives color_map, default_format
        # and the font-size fallbacks), so a defaults reset returns to the
        # shipped style. Loaded datasets are left alone; reset_format('sets')
        # restyles those.
        self._apply_style_defaults(DEFAULT_PLOT_STYLE)

    def reset_format(self, *what, uset_slice=None, sets=None, vars=None,
                     lines=None, highlights=None, scale=None, fonts=None):
        """
        Reset formatting back to defaults — the one entry point for all of it.

        Call with no arguments to reset every piece of *applied* formatting
        (dataset styles, variable overrides, lines, highlights, axis limits,
        font sizes, pinned plot size, gridline formatting, watermark). Pass
        scope names
        to reset only those.
        The *defaults themselves* (``set_default_format`` state, figsize,
        per-call plot defaults, the ``set_plot_style`` look) are only reset when
        you ask for ``'defaults'`` explicitly, or reset everything with
        ``'all'``.

        Parameters
        ----------
        *what : scope name(s) and/or dataset selector(s)
            Scope names: 'sets', 'vars', 'lines', 'highlights', 'scales',
            'fonts', 'plot_size', 'grid', 'watermark', 'defaults', or 'all'
            (everything, defaults included). Common aliases work too ('scale', 'variables',
            'gridlines', ...).
            A non-string positional (int, list, Dataset) is a dataset
            selector, same as ``uset_slice``.
        uset_slice : int | list | Dataset | 'all' | None
            Which datasets the 'sets' scope applies to (None/'all' = every
            dataset). Passing a selector with no scope names resets *only*
            those datasets' formatting.
        sets, vars, lines, highlights, scale, fonts : bool or name(s)
            Fine-grained include/exclude, kept from the old API: ``True``
            adds the scope, ``False`` removes it from the sweep. For
            ``vars`` / ``lines`` / ``highlights`` / ``scale`` you may instead
            pass a variable/column name (or list of names) to reset just
            those entries, e.g. ``reset_format(vars='CHT', lines='ALT')``.

        Every formatting setter also accepts ``'reset'`` as its value for
        per-attribute resets: ``nb.color(0, 'reset')``,
        ``nb.color('CHT', 'reset')``, ``nb.line('all', 'reset')``,
        ``nb.scale('all', 'reset')``, ``nb.var_format('CHT', color='reset')``.

        Examples
        --------
        nb.reset_format()                       # all applied formatting
        nb.reset_format('all')                  # ...plus the stored defaults
        nb.reset_format('lines', 'scales')      # just those scopes
        nb.reset_format([0, 1])                 # only datasets 0 and 1
        nb.reset_format('sets', uset_slice=0)   # same, explicit form
        nb.reset_format(vars='CHT')             # one variable's overrides
        nb.reset_format(sets=False)             # everything except datasets
        """
        # --- Parse positionals: scope names vs dataset selectors -----------
        scopes, selectors, full = [], [], False
        for item in what:
            if isinstance(item, str):
                key = self._RESET_ALIASES.get(item.lower(), item.lower())
                if key in ('all', 'everything'):
                    full = True
                elif key in self._RESET_SCOPES:
                    scopes.append(key)
                else:
                    valid = ', '.join(self._RESET_SCOPES + ('all',))
                    raise ValueError(
                        f"Unknown reset scope {item!r}. Valid scopes: {valid}. "
                        f"To clear one variable's overrides use "
                        f"reset_format(vars={item!r}).")
            else:
                selectors.append(item)
        if selectors:
            merged = selectors[0] if len(selectors) == 1 else selectors
            uset_slice = merged if uset_slice is None else [uset_slice, merged]

        # --- Determine the active scopes ------------------------------------
        flag_map = {'sets': sets, 'vars': vars, 'lines': lines,
                    'highlights': highlights, 'scales': scale, 'fonts': fonts}
        any_flags = any(v is not None for v in flag_map.values())
        if full:
            active = set(self._RESET_SCOPES)
        elif scopes:
            active = set(scopes)
        elif uset_slice is not None and not any_flags:
            active = {'sets'}          # bare selector: just re-style those sets
        else:
            active = set(self._RESET_DEFAULT_SWEEP)

        # --- Boolean / targeted kwargs adjust the active set ----------------
        targeted = {}                  # scope -> specific names to clear
        for scope, val in flag_map.items():
            if val is None:
                continue
            if val is True:
                active.add(scope)
            elif val is False:
                active.discard(scope)
            elif scope in ('vars', 'lines', 'highlights', 'scales'):
                targeted[scope] = [val] if isinstance(val, str) else list(val)
                active.add(scope)
            else:
                raise TypeError(f"{scope} must be True or False, got {val!r}")

        # --- Perform the resets (defaults first, so 'sets' picks them up) ---
        done = []
        if 'defaults' in active:
            self._reset_defaults()
            done.append("style/plot defaults")
        if 'sets' in active:
            targets = self._get_uset_slice(uset_slice)
            for ds in targets:
                self._reset_set_attrs(ds)
            label = "dataset formatting"
            if uset_slice is not None and uset_slice != 'all':
                label += f" ({len(targets)} set{'s' if len(targets) != 1 else ''})"
            done.append(label)
        for scope, store, label in (('vars', self.variable_formats, "variable formats"),
                                    ('lines', self.lines, "lines"),
                                    ('highlights', self.highlights, "highlights"),
                                    ('scales', self.axis_limits, "axis limits")):
            if scope not in active:
                continue
            if scope in targeted:
                for name in targeted[scope]:
                    store.pop(name, None)
                done.append(f"{label} ({', '.join(targeted[scope])})")
            else:
                store.clear()
                done.append(label)
        if 'fonts' in active:
            self.set_font_sizes(reset=True)
            done.append("font sizes")
        if 'plot_size' in active:
            self.plot_size = None
            self.plot_size_per_subplot = True
            done.append("plot size")
        if 'grid' in active:
            self.grid_format = self._empty_grid_format()
            done.append("grid format")
        if 'watermark' in active:
            self.watermark_format = {}
            done.append("watermark")
        print(f"Reset: {', '.join(done) if done else 'nothing'}.")

    def _apply_default(self, key, value, builtin):
        """Resolve a per-call plotting arg: an explicit ``value`` (not None) wins;
        else the stored ``self.plot_defaults[key]`` if set; else the method's own
        ``builtin``. Lets each method keep its native default while sharing one
        notebook-level override (e.g. barmode's built-in differs per method)."""
        if value is not None:
            return value
        stored = self.plot_defaults.get(key)
        return stored if stored is not None else builtin

    @staticmethod
    def _check_bar_columns(active_sets, columns):
        """Fail loudly on a column no selected dataset has, warn on one only some
        have. Bar charts tolerate heterogeneous datasets (a column is drawn where
        present), but a column absent everywhere would otherwise yield a blank
        panel or a bare pandas KeyError with no hint of which name was wrong."""
        for col in dict.fromkeys(c for c in columns if c is not None):
            lacking = [f"{d.index}: {d.title}" for d in active_sets if col not in d.columns]
            if len(lacking) == len(active_sets):
                raise ValueError(f"Column {col!r} not found in any selected dataset "
                                 f"({', '.join(lacking)}).")
            if lacking:
                warnings.warn(f"Column {col!r} is missing from dataset(s) "
                              f"{', '.join(lacking)}; drawn only where present.",
                              UserWarning, stacklevel=3)

    def _resolve_grid(self, ncols, nrows):
        """Apply the standing ncols/nrows default only when neither was passed,
        resolving them as a pair so a one-off ``ncols=`` doesn't pull the default
        ``nrows``. Returns ``(ncols, nrows)``."""
        if ncols is None and nrows is None:
            dn, dr = self.plot_defaults.get('ncols'), self.plot_defaults.get('nrows')
            if dn is not None or dr is not None:
                return dn, dr
        return ncols, nrows

    def _spacing_kwargs(self, hspace, vspace):
        """Subplot-gap kwargs for a gridded builder: the per-call ``hspace`` /
        ``vspace`` (validated here so a typo fails before anything is drawn),
        falling back to the ``set_default_format`` defaults, plus the
        ``spacing_ref`` that lets a pixel gap be converted against the size
        ``set_plot_size`` will make the figure rather than ``figsize``."""
        hspace = self._apply_default('hspace', hspace, None)
        vspace = self._apply_default('vspace', vspace, None)
        _parse_spacing('hspace', hspace)
        _parse_spacing('vspace', vspace)
        ref = None
        if self.plot_size is not None:   # already in px, None for an unpinned dim
            ref = {'panel' if self.plot_size_per_subplot else 'paper': self.plot_size}
        return {'hspace': hspace, 'vspace': vspace, 'spacing_ref': ref}

    def set_default_format(self, markersize=None, linestyle=None, linewidth=None,
                           edgewidth=None, edge_color=None, alpha=None, fill=None,
                           marker=_UNSET, hue_palette=None, sig_figs=None,
                           decimals=None, alpha_marker=_UNSET,
                           alpha_line=_UNSET, figsize=None, legend=None,
                           suppress_legends=None, ncols=None, nrows=None,
                           hspace=None, vspace=None,
                           barmode=None, agg=None, histfunc=None, histnorm=None,
                           boxmode=None, points=None, legend_scroll=None,
                           reset=False):
        """Set notebook-wide defaults for styling and for the plot methods.

        Two kinds of default live here. **Per-dataset styles** (markersize,
        linestyle, linewidth, edgewidth, edge_color, alpha, fill, marker,
        hue_palette — the colorscale for contours and hue-colored scatters) are the
        markersize/linewidth analogue of ``color_map``/``marker_map``, applied to
        *future* loaded datasets; already-loaded sets keep their styling until
        ``reset_format()`` re-applies the new defaults. **Figure / per-call
        defaults** (figsize, legend, suppress_legends, legend_scroll, ncols, nrows,
        hspace, vspace, barmode, agg, histfunc, histnorm, points, boxmode) seed the matching argument of the
        plot methods whenever a call doesn't pass its own value; an explicit
        per-call argument always wins. Only the values you pass change; others
        persist — except ``sig_figs`` and ``decimals``, two spellings of one
        knob, which clear each other. Color remains controlled by ``color_map``.

        ``reset=True`` restores *all* of the above — per-dataset styles, figsize,
        and the per-call defaults — to their built-ins, and ignores other args
        (equivalent to ``reset_format('defaults')``).

        Parameters
        ----------
        markersize, linewidth, edgewidth : float (>= 0)
        alpha : float in [0, 1]
            Default opacity for per-dataset styles *and* the histogram bar
            opacity default (the latter overridable per-call via ``histogram(alpha=)``).
        alpha_marker, alpha_line : float in [0, 1], or None
            Default marker-only / line-only opacity for future datasets
            (see :meth:`alpha_marker`, :meth:`alpha_line`). ``None`` means
            "same as alpha".
        sig_figs : int (>= 1)
            Default significant figures for *displayed* values (table cells,
            summary statistics, hover readouts) of future datasets — see
            :meth:`sig_figs`, which also restyles the sets already loaded.
        decimals : int (>= 0)
            The fixed-decimal-places spelling of ``sig_figs`` (see
            :meth:`decimals`). The two are alternatives: passing one clears the
            other, and passing both raises.
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
        legend_scroll : bool
            Whether an interactive plot's legend scrolls when it is too tall for
            the figure (built-in True, Plotly's behavior). ``False`` shows every
            entry instead, making the figure taller so the plot area keeps its
            size. ``save_png`` and static images always show the whole legend.
        ncols, nrows : positive int
            Default subplot grid. Takes precedence over the sticky "remember the
            last grid" memory in ``plot``, but an explicit per-call ncols/nrows
            still wins. Resolved as a pair: setting one leaves the other auto.
        hspace, vspace : number or str
            Default gap between subplot columns / rows for every gridded plot
            method. A value of 1 or more is pixels (``60`` or ``'60px'``); a
            value below 1 is a fraction of the plot area, as Plotly's
            ``horizontal_spacing`` / ``vertical_spacing`` take it. Unset, each
            gap is a fixed pixel budget (80 px between columns, 70 px between
            rows; contour and secondary-axis plots reserve more for their
            colorbars and extra axes) whatever the grid size, instead of
            Plotly's grid-relative fractions.
        barmode : 'group' | 'stack' | 'overlay' | 'relative'
            Default bar mode for ``bar`` (built-in 'group') and ``histogram``
            (built-in 'overlay'). Validated against the union of both; a value
            valid for only one method errors when the other method runs.
        agg : reducer for ``bar`` (built-in 'mean'): a name such as 'sum',
            'max', 'count', any pandas Series reducer, or a callable. Checked
            when ``bar`` runs.
        histfunc, histnorm : for ``histogram`` (built-ins 'sum', '').
        boxmode, points : for ``box`` (built-ins 'group', 'outliers').

        Examples
        --------
        nb.set_default_format(markersize=6, linestyle='--', linewidth=1)
        nb.set_default_format(marker=None)   # turn markers off for future sets
        nb.set_default_format(figsize=(10, 6), legend='right', ncols=2)
        nb.set_default_format(legend_scroll=False)  # never scroll the legend
        nb.set_default_format(vspace=100, hspace=0.05)  # 100 px rows, 5% columns
        nb.set_default_format(reset=True)    # clear styles, figsize, and defaults
        """
        if reset:
            self._reset_defaults()
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
        for _name, _val in (('alpha_marker', alpha_marker), ('alpha_line', alpha_line)):
            if _val is not _UNSET:
                updates[_name] = None if _val is None else _num(_name, _val, 0.0, 1.0)
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
        if hue_palette is not None:
            if not isinstance(hue_palette, str):
                raise TypeError("hue_palette must be a Plotly colorscale name "
                                f"(e.g. 'Viridis'), got {type(hue_palette).__name__}")
            updates['hue_palette'] = hue_palette
        if sig_figs is not None and decimals is not None:
            raise ValueError("Pass either sig_figs or decimals, not both.")
        if sig_figs is not None:
            if isinstance(sig_figs, bool) or not isinstance(sig_figs, int) or sig_figs < 1:
                raise ValueError(f"sig_figs must be a positive integer, got {sig_figs!r}")
            updates['sig_figs'] = sig_figs
            updates['decimals'] = None        # the two are alternatives
        if decimals is not None:
            if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0:
                raise ValueError(f"decimals must be a non-negative integer, got {decimals!r}")
            updates['decimals'] = decimals
            updates['sig_figs'] = None
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
        if legend_scroll is not None:
            if not isinstance(legend_scroll, bool):
                raise TypeError("legend_scroll must be bool, got "
                                f"{type(legend_scroll).__name__}")
            pd_updates['legend_scroll'] = legend_scroll
        for _name, _val in (('ncols', ncols), ('nrows', nrows)):
            if _val is not None:
                if isinstance(_val, bool) or not isinstance(_val, int) or _val < 1:
                    raise ValueError(f"{_name} must be a positive integer, got {_val!r}")
                pd_updates[_name] = _val
        for _name, _val in (('hspace', hspace), ('vspace', vspace)):
            if _val is not None:
                _parse_spacing(_name, _val)   # validates; the raw value is stored
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

    def _template_name(self):
        """Plotly template for the active (plot_style, darkmode) pair."""
        if self.plot_style == 'matplotlib':
            return _MPL_TEMPLATE_DARK if self.darkmode else _MPL_TEMPLATE
        return "plotly_dark" if self.darkmode else "plotly_white"

    def _apply_style_defaults(self, style):
        """Install a plot style's *defaults*: color_map, the per-dataset format
        entries the style owns, and its preferred font sizes.

        Deliberately does not touch already-loaded datasets (that is
        ``set_plot_style``'s ``sets`` argument) or the figure layout (that is
        ``_template_name``, applied in ``_finalize``). Reverting to 'plotly'
        restores only the keys a style owns, so unrelated
        ``set_default_format`` choices survive a style switch.
        """
        self.plot_style = style
        if style == 'matplotlib':
            self.color_map = list(MPL_COLOR_CYCLE)
            self.default_format.update(MPL_DATASET_FORMAT)
            self._style_font_defaults = dict(MPL_FONT_SIZES)
        else:
            self.color_map = px.colors.qualitative.Plotly
            for key in MPL_DATASET_FORMAT:
                self.default_format[key] = _DATASET_FORMAT_DEFAULTS[key]
            self._style_font_defaults = {}

    def set_plot_style(self, style=DEFAULT_PLOT_STYLE, sets=True):
        """Switch the overall look of the figures between Matplotlib's and
        Plotly's.

        ``'matplotlib'`` is what a notebook starts with: plots approximate
        Matplotlib's defaults — white (or black, in dark mode) plot area framed
        by spines on all four sides, outward ticks, no zero lines, a gray grid,
        DejaVu Sans at Matplotlib's point sizes, the tab10 color cycle, and
        viridis for contours. ``'plotly'`` opts into Plotly's own house style
        instead, which is how unichart drew before plot styles existed.

        The style is orthogonal to :meth:`toggle_darkmode` — both styles have a
        light and a dark variant — and to the rest of the formatting API: any
        explicit ``color``/``markersize``/``set_font_sizes``/``var_format``
        value you set afterwards still wins.

        Parameters
        ----------
        style : str
            ``'matplotlib'`` (the default; also ``'mpl'``, ``'plt'``,
            ``'pyplot'``) or ``'plotly'``. ``'default'`` and ``'reset'`` name
            whichever style unichart ships with, currently ``'matplotlib'``.
        sets : bool
            Re-derive the **already loaded** datasets' color, markersize and
            hue palette from the new style, so a switch takes effect on data you
            loaded earlier. This clears manual ``nb.color(...)`` /
            ``nb.markersize(...)`` / ``nb.hue_palette(...)`` overrides on those
            sets — pass ``sets=False`` to keep them (only future loads and the
            layout then follow the new style).

        Notes
        -----
        Two things stay outside the style. Matplotlib draws line plots without
        markers while unichart assigns one per dataset — turn them off with
        ``nb.set_default_format(marker=None)`` if you want that too. And
        ``dashboard`` panels override the figure font with the board's UI font
        so charts and chrome read as one surface, so the DejaVu font (only) is
        not carried into dashboards.

        Examples
        --------
        nb.set_plot_style('plotly')                  # Plotly's native look
        nb.set_plot_style('plotly', sets=False)      # ...keeping hand-set colors
        nb.set_plot_style('matplotlib')              # back to the default look
        """
        key = style.strip().lower() if isinstance(style, str) else style
        resolved = _PLOT_STYLE_ALIASES.get(key)
        if resolved is None:
            valid = ', '.join(sorted(set(_PLOT_STYLE_ALIASES)))
            raise ValueError(f"Unknown plot style {style!r}. Valid names: {valid}")

        self._apply_style_defaults(resolved)
        if sets:
            for ds in self.sets:
                self._reset_set_attrs(ds, ('color', 'markersize', 'hue_palette'))
        print(f"Plot style set to: {resolved}"
              f"{' (existing datasets restyled)' if sets and self.sets else ''}")


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

        The delta set inherits the study set's styling — colour, marker,
        linestyle, markersize, linewidth, edgewidth, alpha, edge colour, fill,
        hue/hue palette/hue order, style, display_parms and sort order — so it
        reads as a visual continuation of the study series it was derived from.
        Only appearance is carried over: ``reg_order`` is not inherited (the
        delta gets no trendline unless you set one), nor is ``plot_type`` (a
        delta always starts as a scatter), and its title and index are its own.
        A column-referencing option is only carried over when that column
        survives into the result.
        Restyle it like any other set (``nb.linestyle(idx, ...)``), or undo an
        inherited attribute with that setter's ``'reset'`` (``nb.linestyle(idx,
        'reset')``). Pass ``name_as_study=True`` to also title it after the
        study set.

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
            default ``'Set {study} rel. to Set {base}'``. The study set's
            formatting is inherited regardless of this flag.
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
            # Inherit the study set's formatting so the delta reads as a visual
            # continuation of the study series it was derived from.
            self.copy_format(study_ds, ds)
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
    def line(self, column, level, color='red', linestyle=None, dash=None,
             label=None, label_size=None, label_position=None, label_color=None):
        """Add a vertical or horizontal line to the next plot.

        Args:
            column (str): The variable the line is keyed to (x-var -> vertical
                line; y-var -> horizontal line). Use 'all' with level='reset'.
            level (float or 'reset'): The line position, or 'reset' (alias
                'clear') to remove the line(s) for ``column`` ('all' removes
                every line; also available as ``reset_format('lines')``).
            color (str): Line color.
            linestyle (str, optional): Line style — Matplotlib-style
                ('-', '--', '-.', ':') or Plotly-style ('solid', 'dash',
                'dashdot', 'dot'). Defaults to 'dash'.
            dash (str, optional): Deprecated alias for ``linestyle``, kept for
                backwards compatibility. Ignored if ``linestyle`` is given.
            label (str, optional): Text drawn on the line, inside the plot area.
            label_size (float or str, optional): Label font size — a number or a
                size name ('small', 'lg', ...). Defaults to the ``axes_tick``
                size from :meth:`set_font_sizes` when one is set, else Plotly's
                default.
            label_position (float or str, optional): Where the label sits **on**
                the line. Either a 0-1 fraction along it (0 = bottom/left,
                1 = top/right) or a string of position tokens; the two can be
                combined, e.g. ``'bottom left'`` or ``'0.25 left'``.
                For a vertical line: 'top'/'middle'/'bottom' slide the label,
                'left'/'right' pick the side of the line the text sits on.
                For a horizontal line: 'left'/'center'/'right' slide it,
                'top'/'above' or 'bottom'/'below' pick the side.
                Defaults to the far end of the line ('top right' for a vertical
                line, 'right' and above the line for a horizontal one).
            label_color (str, optional): Label text color. Defaults to ``color``.

        On a subplot grid the label is repeated on every subplot the line is
        drawn on, matching how the line itself repeats.

        Examples:
            nb.line('rpm', 5000, label='redline')
            nb.line('cht', 400, color='orange', label='limit',
                    label_size='lg', label_position='left')
            nb.line('time', 12.5, label='event', label_position=0.25)
        """
        if level in ('clear', 'reset'):
            if column == 'all':
                self.lines.clear()
            else:
                self.lines.pop(column, None)
            return

        # linestyle is the preferred name; dash is the legacy alias. When
        # neither is supplied, preserve the original default of 'dash'.
        style = linestyle if linestyle is not None else (dash if dash is not None else 'dash')

        if label_size is not None:
            if isinstance(label_size, str):
                resolved = FONT_SIZE_MAP.get(label_size.lower())
                if resolved is None:
                    raise ValueError(f"label_size: unknown size name '{label_size}'. "
                                     f"Valid names: {', '.join(sorted(FONT_SIZE_MAP))}")
                label_size = resolved
            if isinstance(label_size, bool) or not isinstance(label_size, (int, float)):
                raise TypeError("label_size must be numeric or a size name, "
                                f"got {type(label_size).__name__}")
            if label_size <= 0:
                raise ValueError(f"label_size must be positive, got {label_size}")
            label_size = float(label_size)

        # Position tokens can only be checked against an orientation, which is
        # known at draw time; the type and a bare fraction are checked here.
        if label_position is not None:
            if isinstance(label_position, bool) or not isinstance(
                    label_position, (int, float, str, list, tuple)):
                raise TypeError("label_position must be a number (0-1) or a position string, "
                                f"got {type(label_position).__name__}")
            if isinstance(label_position, (int, float)) and not 0.0 <= float(label_position) <= 1.0:
                raise ValueError("label_position fraction must be between 0 and 1, "
                                 f"got {label_position}")

        if column not in self.lines: self.lines[column] = []
        plotly_dash = LINESTYLE_MAP_MPL_TO_PLOTLY.get(style, style)
        self.lines[column].append({'level': level, 'color': color, 'dash': plotly_dash,
                                   'label': label, 'label_size': label_size,
                                   'label_position': label_position,
                                   'label_color': label_color})

    def highlight(self, column, range_tuple, color='yellow', alpha=0.2, opacity=None):
        """Add a highlighted region to the next plot.

        Pass ``'reset'`` (alias ``'clear'``) as ``range_tuple`` to remove the
        highlight(s) for ``column`` ('all' removes every highlight; also
        available as ``reset_format('highlights')``).
        """
        if opacity is not None:
            warnings.warn("'opacity' is deprecated, use 'alpha'", DeprecationWarning, stacklevel=2)
            alpha = opacity
        if range_tuple in ('clear', 'reset'):
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

        Pass ``'reset'`` (alias ``'clear'``, or None) as ``range_tuple`` to
        remove the stored limits; ``column='all'`` then clears every axis
        limit (also available as ``reset_format('scales')``).

        For bar charts, a marker/tick/whisker overlay column (passed via
        ``bar(markers=...)``) is a valid ``column`` here: it shares the bar's
        y-axis, so its range is unioned with the bar variable's own scale.
        """
        clearing = range_tuple in ('clear', 'reset') or range_tuple is None
        if clearing and column == 'all':
            self.axis_limits.clear()
            print("All axis limits cleared.")
            return

        if isinstance(column, (list, tuple)):
            columns = column
        else:
            columns = [column]

        for col in columns:
            if clearing:
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
            Font sizes for plot elements. ``'reset'`` clears that one size back
            to its default.
        table_header : float or str
            Font size for table header row.
        table_cell : float or str
            Font size for table cell content.
        all : float or str
            Set all font sizes at once (overridden by individual parameters).
            ``all='reset'`` clears every size back to the defaults, with
            individual parameters still applied on top.
        reset : bool
            Reset all font sizes to defaults (equivalent to
            ``reset_format('fonts')`` or ``all='reset'``).
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
                if value.lower() == 'reset':
                    return 'reset'
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
            if v == 'reset':
                setattr(self, k, None)
            elif v is not None:
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
    def set_plot_size(self, width=None, height=None, per_subplot=True, reset=False):
        """Pin the plot area so plots come out the same size — and therefore the
        same aspect ratio — regardless of suptitle lines, legend rows, colorbars
        or any other margin change.

        ``width``/``height`` are in inches (same units as ``figsize``). By
        default they size **one subplot panel**, and the figure grows to fit the
        whole grid plus its margins — so ``nb.plot(x, y='A')`` and
        ``nb.plot(x, y=['A', 'B', 'C'])`` draw panels of identical size and
        shape, instead of splitting one fixed area between however many
        variables you asked for. Pass ``per_subplot=False`` to pin the *whole*
        grid instead, letting the panels shrink as it grows.

        Pass only the dimension(s) you want to pin — ``None`` leaves that
        dimension driven by ``figsize``. Each call replaces the previous setting
        (calling with only ``height`` drops a prior ``width`` pin).
        ``reset=True`` (or both ``None``) clears it — equivalent to
        ``reset_format('plot_size')``.

        Parameters
        ----------
        width, height : float, optional
            Plot-area size in inches. Either may be ``None`` to leave that
            dimension to ``figsize``.
        per_subplot : bool
            ``True`` (default) sizes each subplot panel; the figure grows with
            the grid, so a 3-panel plot is roughly three times as wide as a
            1-panel one. ``False`` pins the combined grid area — the behaviour
            this method had before per-panel sizing — which keeps the figure
            size stable but makes each panel shrink as panels are added.
            Like the sizes, this is per call and not remembered: a later
            ``set_plot_size(height=3)`` goes back to per-panel mode unless you
            pass ``per_subplot=False`` again.
        reset : bool
            Clear the pin (equivalent to ``reset_format('plot_size')``).

        Examples
        --------
        nb.set_plot_size(4, 3)                    # every panel 4x3in, always
        nb.set_plot_size(height=3)                # pin height only
        nb.set_plot_size(6, 4, per_subplot=False) # pin the whole grid instead
        nb.set_plot_size(reset=True)

        Notes
        -----
        The figure is sized as ``plot_area + margins``, so the pin only holds
        while Plotly's margin ``autoexpand`` stays inert — any margin it grows
        comes straight out of the plot area. unichart therefore re-reserves the
        top band (suptitle + above-legend) against the figure's final width and
        height whenever a size is pinned, and keeps the right margin a plotting
        method reserved for a colorbar or extra y axes. The stacked y axes of
        ``plot_ymult`` (and ``bar``/``box`` with ``by='dataset_x'``) are re-laid
        the same way, each keeping a fixed pixel slot for its labels rather than
        a share of a plot area the pin may have narrowed. Unusually long tick
        labels can still expand a margin and leave that dimension slightly
        short; widen ``figsize`` or shorten the labels if you hit it.

        In per-subplot mode the figure grows with the grid, so a wide grid can
        get large — five panels at 6in each is a ~22in-wide figure. Use a
        smaller per-panel size, or ``ncols=1``, when that is inconvenient.

        Dashboards are unaffected: each board panel is rendered at the size the
        board gives it (``width``/``height`` on :meth:`dashboard`), so the pin
        governs notebook figures, not board tiles.
        """
        if reset or (width is None and height is None):
            self.plot_size = None
            self.plot_size_per_subplot = True
            return

        def _v(name, val):
            if val is None:
                return None
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be numeric (inches), got {type(val).__name__}")
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")
            return val * 100

        if not isinstance(per_subplot, bool):
            raise TypeError("per_subplot must be True or False, got "
                            f"{type(per_subplot).__name__}")
        self.plot_size = (_v('width', width), _v('height', height))
        self.plot_size_per_subplot = per_subplot

    @staticmethod
    def _panel_fractions(fig):
        """Fraction of the figure's inner width/height that a single subplot
        panel spans, read off the axis domains.

        Overlaying axes are skipped: ``plot_ymult`` and secondary-y plots stack
        extra axes on the same panel with their own domains, and counting those
        as panels would corrupt the arithmetic. The largest remaining domain
        wins, so an uneven grid errs toward the smaller figure rather than an
        enormous one.

        Also picks up the horizontal room ``plot_ymult`` reserves for its extra
        y axes — that space is inside the paper area but outside the panel, so
        the fraction is what turns a paper-width pin into a true panel-width one.
        """
        def _spans(axes):
            out = []
            for ax in axes:
                if ax.overlaying:
                    continue
                dom = ax.domain
                if dom is None or len(dom) != 2:
                    continue
                span = float(dom[1]) - float(dom[0])
                if span > 0:
                    out.append(span)
            return max(out) if out else 1.0

        return _spans(fig.select_xaxes()), _spans(fig.select_yaxes())

    def _pinned_inner_size(self, fig):
        """The pinned inner (paper) size of ``fig`` in px, as ``(w, h)`` with
        ``None`` for an unpinned dimension.

        In per-subplot mode the stored size is one panel, so it is divided by
        that panel's share of the paper area to get the paper size the figure
        must provide. Shared by ``_apply_footer`` (which needs the final plot
        height before the figure is resized) and ``_enforce_plot_size``."""
        if fig is None or self.plot_size is None:
            return None, None
        pw, ph = self.plot_size
        if not self.plot_size_per_subplot:
            return pw, ph
        fx, fy = self._panel_fractions(fig)
        return (None if pw is None else pw / fx,
                None if ph is None else ph / fy)

    @staticmethod
    def _keep_right_margin(fig, default):
        """Right margin for a legend placement, without dropping a larger
        reservation a plotting method already made (a hue/contour colorbar, the
        extra y axes of ``plot_ymult``). Overwriting it would hand the space
        back to Plotly's margin autoexpand, which then takes it out of the plot
        area and shrinks a ``set_plot_size`` width pin."""
        cur = fig.layout.margin.r
        return default if cur is None else max(default, cur)

    @staticmethod
    def _legend_row_estimate(fig):
        """How many rows Plotly will lay the above-legend out in.

        Two layouts, because Plotly treats them differently:

        * A plain horizontal legend flows its entries across the width and
          wraps.
        * A *grouped* one (any trace carrying a ``legendgrouptitle``, as
          ``plot_ymult`` does) becomes a row of **columns**: each group is a
          column headed by its title with its entries stacked beneath, and the
          columns themselves flow across the width and wrap into bands. So the
          row count is ``bands x (1 title + deepest group)``, not the entry
          count — a 3-set x 2-variable ymult legend is 3 rows, not 9.

        Both widths are estimated with the pinned-path constants, calibrated
        against rendered legends: over-estimating only costs whitespace, while
        under-estimating lets Plotly's autoexpand grow the top margin and shrink
        the pinned plot area.
        """
        groups, plain = {}, []
        for tr in fig.data:
            if getattr(tr, 'showlegend', None) is False:
                continue
            name = getattr(tr, 'name', None)
            if not name:
                continue
            gt = getattr(tr, 'legendgrouptitle', None)
            gt = getattr(gt, 'text', None) if gt is not None else None
            if gt:
                if name not in groups.setdefault(gt, []):
                    groups[gt].append(name)
            elif name not in plain:
                plain.append(name)

        usable = max(200, (fig.layout.width or 1200) - 40)
        rows = 0
        if plain:
            rows += _flow_rows(
                [_PINNED_ENTRY_BASE_PX + _PINNED_ENTRY_CHAR_PX * len(n)
                 for n in plain], usable)
        if groups:
            depth = 1 + max(len(names) for names in groups.values())
            widths = [_PINNED_GROUP_BASE_PX + _PINNED_GROUP_CHAR_PX
                      * max([len(title)] + [len(n) for n in names])
                      for title, names in groups.items()]
            rows += _flow_rows(widths, usable) * depth
        return max(1, rows)

    def _pin_top_space(self, fig):
        """Re-reserve the top band (suptitle + above-legend) against the
        figure's *final* width, and return the recomputed top margin.

        ``_base_layout`` sizes that band from ``figsize``: the legend's row
        count is estimated at the figsize width, and the title/legend anchors
        are stored as fractions of the figsize height. ``set_plot_size`` then
        changes both dimensions — by a lot in per-subplot mode, where the
        figure grows with the grid. The stale estimates let the legend wrap to
        more rows than were reserved, and slide the title and legend down the
        taller figure, until Plotly's margin autoexpand pushes the top margin
        out to keep them clear of the plot — quietly taking that space out of
        the plot area and shrinking the pin.

        Only called from ``_enforce_plot_size``, so figures without a pinned
        size keep their previous geometry exactly."""
        title = fig.layout.title.text
        leg = fig.layout.legend
        has_above = (leg.orientation == 'h' and leg.yref == 'container')
        rows = self._legend_row_estimate(fig) if has_above else 1
        top_margin, _, _ = _top_space(
            title, (None, (fig.layout.height or 800) / 100), has_above,
            title_font_size=self._font_size('suptitle_size'), legend_rows=rows)
        # Plotly measures the rendered legend, whose rows run taller than the
        # _LEGEND_ROW_PX estimate (~31px vs 26px) and by a little more with each
        # row. Without this slack autoexpand fires for the difference and the
        # height pin lands a few px short.
        if has_above:
            top_margin += _PINNED_TOP_SLACK + _PINNED_ROW_SLACK * rows
        meta = dict(fig.layout.meta or {})
        meta['uc_legend_rows'] = rows
        fig.update_layout(margin=dict(t=top_margin), meta=meta)
        return top_margin

    def _anchor_top(self, fig):
        """Re-pin the suptitle and above-legend to fixed pixel offsets from the
        top, now that the figure's final height is known. Their stored ``y``
        values are container *fractions*, so they only mean the intended offset
        at the height they were computed for."""
        rows = (fig.layout.meta or {}).get('uc_legend_rows', 1)
        leg = fig.layout.legend
        has_above = (leg.orientation == 'h' and leg.yref == 'container')
        _, title_pos, legend_pos = _top_space(
            fig.layout.title.text, (None, (fig.layout.height or 800) / 100),
            has_above, title_font_size=self._font_size('suptitle_size'),
            legend_rows=rows)
        if fig.layout.title.text:
            fig.update_layout(title=dict(y=title_pos['y']))
        if legend_pos is not None:
            fig.update_layout(legend=dict(y=legend_pos['y']))
        return fig

    @staticmethod
    def _legend_items(fig):
        """The entries Plotly draws in ``fig``'s legend, in trace order, as
        ``(legendgroup, group title, name)`` with repeats dropped."""
        items, seen = [], set()
        for tr in fig.data:
            if getattr(tr, 'showlegend', None) is False:
                continue
            name = getattr(tr, 'name', None)
            if not name:
                continue
            group = getattr(tr, 'legendgroup', None) or None
            if (group, name) in seen:
                continue
            seen.add((group, name))
            gt = getattr(tr, 'legendgrouptitle', None)
            gt = getattr(gt, 'text', None) if gt is not None else None
            items.append((group, gt or None, name))
        return items

    @staticmethod
    def _legend_metrics(fig):
        """``(font px, row px, group gap px)`` of ``fig``'s rendered legend."""
        leg = fig.layout.legend
        font = leg.font.size or fig.layout.font.size or 12
        row = max(_FULL_ROW_MIN_PX, font * _FULL_ROW_LINE_FACTOR) + _FULL_ROW_PAD_PX
        gap = _FULL_GROUP_GAP_PX if leg.tracegroupgap is None else leg.tracegroupgap
        return font, row, gap

    def _side_legend_height(self, fig):
        """Estimated rendered height (px) of a vertical legend showing every
        entry: a row per entry and per group title, and Plotly's
        ``tracegroupgap`` after each legend group."""
        _, row, gap = self._legend_metrics(fig)
        items = self._legend_items(fig)
        groups = {g for g, _, _ in items}
        titles = {(g, t) for g, t, _ in items if t}
        return (_FULL_LEGEND_PAD_PX + (len(items) + len(titles)) * row
                + len(groups) * gap)

    def _above_legend_height(self, fig):
        """Estimated rendered height (px) of a horizontal legend showing every
        entry, wrapped the way Plotly wraps it: across the plot-area width.

        Plain entries flow into rows, each an entry row plus the group gap. A
        grouped legend (traces with a ``legendgrouptitle``) flows whole group
        columns instead, each band as tall as its deepest group (title row plus
        entries) plus the gap. Unlike ``_legend_row_estimate`` — sized for the
        pinned path's top band — this is measured in px against rendered
        legends, so a very long legend doesn't pick up hundreds of px of
        whitespace above the plot."""
        font, row, gap = self._legend_metrics(fig)
        m = fig.layout.margin
        usable = max(200, (fig.layout.width or 1200)
                     - (80 if m.l is None else m.l) - (80 if m.r is None else m.r))

        def entry_w(text):
            return _FULL_ENTRY_BASE_PX + _FULL_ENTRY_CHAR_EM * font * len(str(text))

        columns, plain = {}, []
        for group, title, name in self._legend_items(fig):
            if title:
                columns.setdefault((group, title), []).append(name)
            else:
                plain.append(name)
        height = _FULL_LEGEND_PAD_PX
        if plain:
            height += _flow_rows([entry_w(n) for n in plain], usable) * (row + gap)
        if columns:
            widths = [max([_FULL_GROUP_BASE_PX + _FULL_GROUP_CHAR_EM * font * len(str(title))]
                          + [entry_w(n) for n in names])
                      for (_, title), names in columns.items()]
            depth = 1 + max(len(names) for names in columns.values())
            height += _flow_rows(widths, usable) * (depth * row + gap)
        return height

    def _title_band(self, fig):
        """Height (px) of the suptitle band an above-legend is pinned just
        below (the ``title_band`` of ``_top_space``)."""
        no_legend, _, _ = _top_space(
            fig.layout.title.text, (None, (fig.layout.height or 800) / 100),
            False, title_font_size=self._font_size('suptitle_size'))
        return no_legend - _TITLE_TOP_PAD

    def _fit_full_legend(self, fig):
        """Show every legend entry instead of letting Plotly scroll the legend,
        growing the figure so the plot area keeps its size. Mutates ``fig``.

        Plotly caps a horizontal legend at half the figure height and a side
        legend at the plot height, and scrolls past that — fine on screen, but
        an exported image just shows the clipped part. Short of the cap, an
        above-legend that overflows the band reserved for it isn't clipped but
        Plotly's margin autoexpand takes the overflow out of the plot area. So
        the cap is lifted, and a legend that outgrows its space gets room made:

        * an above-legend running past the top margin: the top margin and the
          height both grow to fit it (``_above_legend_height``), and the title
          and legend are re-anchored;
        * a side legend taller than the plot area: the bottom margin and the
          height grow by the difference, so it continues past the plot's bottom
          edge, beside the x-axis labels.

        A legend that already fits leaves the figure exactly as it was. Runs at
        most once per figure (``meta['uc_legend_fit']`` marks it), and not at
        all while ``_legend_fit_enabled`` is off (the dashboard's fixed-size
        panels)."""
        if fig is None or fig.layout.showlegend is False:
            return fig
        if not getattr(self, '_legend_fit_enabled', True):
            return fig
        meta = dict(fig.layout.meta or {})
        if 'uc_legend_fit' in meta:
            return fig
        try:
            fig.layout.legend.maxheight = _FULL_LEGEND_MAXHEIGHT
        except ValueError:
            return fig  # a Plotly without legend.maxheight: keep its scrolling

        leg, m = fig.layout.legend, fig.layout.margin
        height = fig.layout.height or 800
        top = 100 if m.t is None else m.t
        has_above = (leg.orientation == 'h' and leg.yref == 'container')
        delta = 0
        if has_above:
            legend_bottom = self._title_band(fig) + self._above_legend_height(fig)
            if legend_bottom > top:
                delta = legend_bottom + _LEGEND_GAP + _PINNED_TOP_SLACK - top
                fig.update_layout(margin=dict(t=top + delta),
                                  height=height + delta)
        else:
            bottom = 80 if m.b is None else m.b
            delta = max(0, self._side_legend_height(fig) - (height - top - bottom))
            if delta:
                fig.update_layout(margin=dict(b=bottom + delta),
                                  height=height + delta)
        meta['uc_legend_fit'] = {'delta': delta}
        fig.update_layout(meta=meta)
        if delta:
            # The suptitle (and an above-legend) sit at container fractions of
            # the old height; re-pin them so the growth doesn't slide them down.
            self._anchor_top(fig)
        return fig

    def _full_legend_figure(self, fig):
        """``fig`` with its whole legend showing, for image output
        (``save_png``, static images). Fits a copy, so the interactive figure
        cached as ``last_fig`` keeps its scrolling legend; a figure that is
        already fitted (``legend_scroll=False``) is used as is."""
        if fig is None or 'uc_legend_fit' in dict(fig.layout.meta or {}):
            return fig
        return self._fit_full_legend(go.Figure(fig))

    @staticmethod
    def _extra_yaxes(fig):
        """The free-positioned right-hand y axes of a multi-axis plot
        (``plot_ymult``, ``bar``/``box`` with ``by='dataset_x'``), ordered
        left to right.

        The second y variable is anchored to the x axis's domain end and so
        follows it automatically; only the third and beyond carry an explicit
        paper ``position`` that has to be re-solved when the width changes.

        Ordered by axis number rather than by current ``position``, so the
        caller can overwrite positions as it walks the list without the order
        shifting underneath it."""
        defined = fig.to_plotly_json().get('layout', {})
        numbered = []
        for key in defined:
            if not key.startswith('yaxis'):
                continue
            ax = fig.layout[key]
            if ax.overlaying and ax.anchor == 'free' and ax.position is not None:
                numbered.append((int(key[5:] or 1), ax))
        return [ax for _, ax in sorted(numbered, key=lambda pair: pair[0])]

    def _reflow_extra_yaxes(self, fig, panel_px, inner_px):
        """Re-lay the extra y axes so each gets ``_YAXIS_SLOT_PX`` of real
        estate, given a plot area of ``inner_px`` whose data region is
        ``panel_px`` wide. The last axis lands on paper 1.0 exactly as it does
        at build time, so ``margin.r`` still covers its labels."""
        axes = self._extra_yaxes(fig)
        if not axes:
            return
        # Normally the caller sized inner_px to give every axis a full slot, so
        # this is exactly _YAXIS_SLOT_PX. It only bites in whole-grid mode,
        # where the axes come out of a fixed plot area and the 50% clamp can
        # leave less than they want: share what there is evenly rather than
        # handing the first axes a full slot and stacking the rest on the edge.
        slot = min(_YAXIS_SLOT_PX, (inner_px - panel_px) / len(axes))
        fig.update_layout(xaxis=dict(domain=[0, panel_px / inner_px]))
        for k, ax in enumerate(axes, start=1):
            ax.position = min(1.0, (panel_px + k * slot) / inner_px)

    def _enforce_plot_size(self, fig):
        """Resize the figure so its plot area matches ``self.plot_size`` — each
        panel in the default per-subplot mode, the whole grid otherwise.
        No-op unless a plot size is pinned.

        Runs in three steps, because the pieces depend on each other in one
        direction only: the width follows from the left/right margins, the top
        band follows from the width (how many rows the legend wraps to), and
        the height follows from the top band. The title and legend anchors are
        fixed up last, once the height they are expressed against is final.

        Extra right-hand y axes are the exception to "read the width off the
        domain": their slots are sized in px, so the domain that decides the
        width is itself a function of the width. That one is solved in closed
        form — ``plot area = data region + one slot per extra axis`` — and the
        axes are re-laid against the answer."""
        if fig is None or self.plot_size is None:
            return fig
        m = fig.layout.margin

        def mv(val, default):
            return default if val is None else val

        raw_w, raw_h = self.plot_size
        n_extra = len(self._extra_yaxes(fig))
        if raw_w is not None:
            if n_extra:
                if self.plot_size_per_subplot:
                    # The pin is the data region; the axes get their slots on top.
                    panel_px, inner_w = raw_w, raw_w + n_extra * _YAXIS_SLOT_PX
                else:
                    # The pin is the whole plot area; the axes take their slots
                    # out of it, which is what whole-grid mode asks for.
                    inner_w = raw_w
                    panel_px = max(inner_w * 0.5,
                                   inner_w - n_extra * _YAXIS_SLOT_PX)
                self._reflow_extra_yaxes(fig, panel_px, inner_w)
            else:
                inner_w = self._pinned_inner_size(fig)[0]
            fig.update_layout(width=inner_w + mv(m.l, 80) + mv(m.r, 80))
        top = self._pin_top_space(fig)
        if raw_h is not None:
            inner_h = self._pinned_inner_size(fig)[1]
            fig.update_layout(height=inner_h + top + mv(fig.layout.margin.b, 80))
        return self._anchor_top(fig)

    # ------------------------------------------------------------------
    # Gridline formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_grid_format():
        return {'x': {'major': {}, 'minor': {}},
                'y': {'major': {}, 'minor': {}}}

    def grid(self, visible=None, color=None, width=None, dash=None,
             axis='both', which='major', reset=False):
        """Configure gridline formatting. Settings persist across plots and sit
        on top of the active plot style (they survive ``set_plot_style`` and
        ``toggle_darkmode``).

        Only the options you pass are stored; repeated calls merge, so
        ``nb.grid(color='gray')`` then ``nb.grid(width=2)`` keeps both. Called
        with no arguments, returns the current settings without changing them.

        ``visible=True`` never re-enables grids that a plot deliberately hides
        (a per-call ``grid=False``, or the secondary axes of ``plot_ymult`` /
        secondary-y plots, whose grid is suppressed to avoid double gridlines);
        ``visible=False`` hides gridlines everywhere it targets.

        Parameters
        ----------
        visible : bool, optional
            Show or hide the targeted gridlines. Also accepts ``'reset'``,
            equivalent to ``reset=True``.
        color : str, optional
            Gridline color (named color, hex, or rgb/rgba string).
        width : float, optional
            Gridline width in px.
        dash : str, optional
            Dash pattern: one of GRID_DASH_OPTIONS ('solid', 'dot', 'dash',
            'longdash', 'dashdot', 'longdashdot') or a Matplotlib alias
            ('-', '--', '-.', ':').
        axis : {'both', 'x', 'y'}
            Which axes this call's options apply to (default 'both').
        which : {'major', 'minor', 'both'}
            Major gridlines (default), minor gridlines (off by default in
            Plotly — enable with ``nb.grid(which='minor', visible=True)``), or
            both.
        reset : bool
            Drop all stored gridline settings (equivalent to
            ``reset_format('grid')``). Note this resets *gridline* formatting;
            the subplot-grid defaults (ncols/nrows) live in
            ``set_default_format``.

        Examples
        --------
        nb.grid(False)                            # no gridlines
        nb.grid(color='lightgray', width=0.5, dash='dot')
        nb.grid(axis='x', visible=False)          # vertical gridlines off
        nb.grid(which='minor', visible=True)      # minor gridlines on
        nb.grid(reset=True)                       # back to the style's grid
        """
        if reset or visible == 'reset':
            self.grid_format = self._empty_grid_format()
            return

        if axis not in ('both', 'x', 'y'):
            raise ValueError(f"axis must be 'both', 'x' or 'y', got {axis!r}")
        if which not in ('major', 'minor', 'both'):
            raise ValueError(
                f"which must be 'major', 'minor' or 'both', got {which!r}")

        opts = {}
        if visible is not None:
            if not isinstance(visible, bool):
                raise TypeError(
                    f"visible must be True or False, got {visible!r}")
            opts['visible'] = visible
        if color is not None:
            if not validate_color(color):
                raise TypeError(
                    f"color must be a color string, got {type(color).__name__}")
            opts['color'] = color
        if width is not None:
            if isinstance(width, bool) or not isinstance(width, (int, float)):
                raise TypeError(
                    f"width must be numeric (px), got {type(width).__name__}")
            if width <= 0:
                raise ValueError(f"width must be positive, got {width}")
            opts['width'] = width
        if dash is not None:
            resolved = LINESTYLE_MAP_MPL_TO_PLOTLY.get(dash, dash)
            if resolved not in GRID_DASH_OPTIONS:
                valid = ', '.join(GRID_DASH_OPTIONS)
                raise ValueError(
                    f"dash must be one of {valid} (or a Matplotlib alias "
                    f"'-', '--', '-.', ':'), got {dash!r}")
            opts['dash'] = resolved

        if not opts:
            # Pure query: report what's set (deep copy so callers can't mutate).
            return {dim: {w: dict(o) for w, o in stored.items()}
                    for dim, stored in self.grid_format.items()}

        for dim in ('x', 'y') if axis == 'both' else (axis,):
            for w in ('major', 'minor') if which == 'both' else (which,):
                self.grid_format[dim][w].update(opts)

    def _apply_grid(self, fig):
        """Apply the sticky :meth:`grid` settings to a finished figure. Runs
        right after ``_apply_style`` so the explicit per-axis values land on
        top of the plot-style template.

        ``visible=True`` skips axes that carry an explicit ``showgrid=False``
        or overlay another axis — those grids were hidden deliberately (per-call
        ``grid=False``, secondary y axes) and re-enabling them would draw
        misaligned double grids."""
        if fig is None:
            return fig
        for dim in ('x', 'y'):
            stored = self.grid_format[dim]
            if not (stored['major'] or stored['minor']):
                continue
            axes = list(fig.select_xaxes() if dim == 'x' else fig.select_yaxes())
            for which, opts in stored.items():
                if not opts:
                    continue
                kw = {}
                if 'color' in opts: kw['gridcolor'] = opts['color']
                if 'width' in opts: kw['gridwidth'] = opts['width']
                if 'dash' in opts: kw['griddash'] = opts['dash']
                vis = opts.get('visible')
                for ax in axes:
                    ax_kw = dict(kw)
                    if vis is not None and not (
                            vis and (ax.overlaying or ax.showgrid is False)):
                        ax_kw['showgrid'] = vis
                    if ax_kw:
                        ax.update(**({'minor': ax_kw} if which == 'minor'
                                     else ax_kw))
        return fig

    # ------------------------------------------------------------------
    # Watermarks
    # ------------------------------------------------------------------
    def watermark(self, source=None, opacity=None, position=None, size=None,
                  layer=None, sizing=None, reset=False):
        """Stamp an image (logo, seal, 'DRAFT' graphic) onto every plot.

        Settings persist across plots and sit on top of the active plot style
        (they survive ``set_plot_style`` and ``toggle_darkmode``). Only the
        options you pass are stored; repeated calls merge, so
        ``nb.watermark('logo.png')`` then ``nb.watermark(opacity=0.3)`` keeps
        both. Called with no arguments, returns the current settings without
        changing them.

        Nothing is drawn until a ``source`` has been given. The image is
        inlined into the figure as base64, so it also shows up in
        ``save_png``, ``set_static_images`` mode, the '⧉ copy' button and the
        chart panels of a ``dashboard`` — not just the live notebook plot.
        (Table panels are rendered as HTML tables, not figures, so they carry
        no watermark.)

        Parameters
        ----------
        source : str | Path | bytes | PIL.Image, optional
            The watermark image: a local file path (png, jpg, jpeg, gif, webp,
            bmp, svg), an ``http(s)://`` URL, a ``data:`` URI, raw image bytes,
            or a PIL Image. Local files are read immediately, so a bad path
            raises here rather than silently drawing nothing later. A URL is
            passed through untouched and therefore only renders where the
            viewer can reach it — prefer a local file if you plan to export or
            copy the plot. The image is embedded in every figure, so keep it
            small; a multi-megabyte logo works against ``set_static_images``.
        opacity : float, optional
            0 (invisible) to 1 (fully opaque). Default 0.15.
        position : str | tuple, optional
            Where the image sits in the plot area: any combination of 'top' /
            'upper', 'bottom' / 'lower', 'left', 'right' and 'center' /
            'middle' — e.g. 'center' (default), 'bottom right', 'top',
            'center left'. Alternatively an explicit ``(x, y)`` pair in paper
            coordinates (0-1 across the plot area, ``(0, 0)`` = bottom left),
            which centers the image on that point.
        size : float | tuple, optional
            Size of the box the image is fitted into, as a fraction of the plot
            area. A scalar applies to both dimensions (default 0.3); pass
            ``(width, height)`` to set them separately. With the default
            ``sizing='contain'`` the image keeps its aspect ratio inside that
            box, so it never stretches.
        layer : {'below', 'above'}, optional
            Draw the watermark under the data (default 'below') or on top of
            it. 'above' plus a low opacity is the classic "tint the whole
            plot" look; 'below' keeps the data fully readable.
        sizing : {'contain', 'fill', 'stretch'}, optional
            How the image fills the size box — 'contain' (default) fits it
            whole, 'fill' crops to cover, 'stretch' distorts to fit exactly.
        reset : bool
            Remove the watermark and drop all stored settings (equivalent to
            ``reset_format('watermark')``). ``source='reset'`` does the same.

        Examples
        --------
        nb.watermark('logo.png')                          # faint, centered
        nb.watermark('logo.png', opacity=0.4, position='bottom right', size=0.15)
        nb.watermark('draft.png', layer='above', opacity=0.08)
        nb.watermark(position=(0.5, 0.1))                 # move the current one
        nb.watermark()                                    # show current settings
        nb.watermark(reset=True)                          # remove it
        """
        if reset or (isinstance(source, str) and source.lower() == 'reset'):
            self.watermark_format = {}
            return

        opts = {}
        if source is not None:
            # Resolve now so a bad path/type raises at the call site instead of
            # producing a figure with an image Plotly quietly declines to draw.
            opts['source'] = _watermark_data_uri(source)
            # Remember how the user named it, so the query form can echo
            # 'logo.png' instead of a multi-megabyte data URI.
            shown = os.fspath(source) if isinstance(source, os.PathLike) else source
            opts['source_repr'] = (
                shown if isinstance(shown, str) and not shown.startswith('data:')
                else f"<{type(source).__name__} image>")
        if opacity is not None:
            if isinstance(opacity, bool) or not isinstance(opacity, (int, float)):
                raise TypeError(
                    f"opacity must be numeric (0-1), got {type(opacity).__name__}")
            if not 0.0 <= opacity <= 1.0:
                raise ValueError(f"opacity must be between 0 and 1, got {opacity}")
            opts['opacity'] = float(opacity)
        if position is not None:
            _parse_watermark_position(position)     # validate now, resolve at draw
            opts['position'] = position
        if size is not None:
            if isinstance(size, (list, tuple)):
                if len(size) != 2:
                    raise ValueError("size tuple must be (width, height), got "
                                     f"{len(size)} values")
                dims = tuple(size)
            else:
                dims = (size, size)
            for val in dims:
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise TypeError("size must be numeric (fraction of the plot "
                                    f"area), got {type(val).__name__}")
                if val <= 0:
                    raise ValueError(f"size must be positive, got {val}")
            opts['size'] = (float(dims[0]), float(dims[1]))
        if layer is not None:
            if layer not in ('below', 'above'):
                raise ValueError(f"layer must be 'below' or 'above', got {layer!r}")
            opts['layer'] = layer
        if sizing is not None:
            if sizing not in ('contain', 'fill', 'stretch'):
                raise ValueError("sizing must be 'contain', 'fill' or 'stretch', "
                                 f"got {sizing!r}")
            opts['sizing'] = sizing

        if not opts:
            # Pure query: report the resolved settings, with the source shown as
            # the user typed it rather than as a multi-megabyte data URI.
            current = dict(_WATERMARK_DEFAULTS)
            current['size'] = (current['size'], current['size'])
            current.update(self.watermark_format)
            current['source'] = current.pop('source_repr', None) or current['source']
            return current

        self.watermark_format.update(opts)

    def _apply_watermark(self, fig):
        """Draw the sticky :meth:`watermark` image on a finished figure.

        No-op until a source has been set. Any watermark already on the figure
        is removed first, so re-finalizing a figure replaces the image instead
        of stacking copies (which would compound the opacity)."""
        if fig is None:
            return fig
        keep = tuple(img for img in (fig.layout.images or ())
                     if getattr(img, 'name', None) != _WATERMARK_NAME)
        src = self.watermark_format.get('source')
        if src is None:
            if len(keep) != len(fig.layout.images or ()):
                fig.layout.images = keep
            return fig

        cfg = dict(_WATERMARK_DEFAULTS)
        cfg['size'] = (cfg['size'], cfg['size'])
        cfg.update(self.watermark_format)
        x, y, xanchor, yanchor = _parse_watermark_position(cfg['position'])
        sizex, sizey = cfg['size']
        fig.layout.images = keep + (dict(
            name=_WATERMARK_NAME, source=src,
            xref='paper', yref='paper', x=x, y=y,
            xanchor=xanchor, yanchor=yanchor,
            sizex=sizex, sizey=sizey, sizing=cfg['sizing'],
            opacity=cfg['opacity'], layer=cfg['layer']),)
        return fig

    def _font_size(self, name):
        """Resolved size for one font slot (e.g. ``'axes_tick_size'``): an
        explicit :meth:`set_font_sizes` value wins, else the active plot
        style's preference, else None (leave Plotly's own default).

        Keeping style sizes in ``_style_font_defaults`` rather than writing them
        into the ``*_size`` attributes means ``get_font_sizes`` keeps reporting
        what the *user* set, and ``set_font_sizes(reset=True)`` clears only that.
        """
        value = getattr(self, name, None)
        if value is not None:
            return value
        return self._style_font_defaults.get(name)

    def _apply_style(self, fig):
        """Re-theme a finished figure for the active plot style.

        Layout only: swapping the template restyles backgrounds, spines, ticks,
        grid and fonts, but cannot reach per-trace colors and markers (those are
        written explicitly from each Dataset) — :meth:`set_plot_style` owns that
        half.

        Skipped for the 'plotly' style: ``_base_layout`` already set
        plotly_white / plotly_dark on the way in, so there is nothing to swap
        and the figure comes out exactly as it did before plot styles existed.
        """
        if fig is None or self.plot_style == 'plotly':
            return fig
        fig.update_layout(template=self._template_name())
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
        footer_size = self._font_size('footer_size')
        text = footer.replace('\n', '<br>')
        m = fig.layout.margin
        band = _bottom_space(text, footer_size)
        base_b = m.b if m.b is not None else 80
        new_b = base_b + band
        fig.update_layout(margin=dict(b=new_b))

        # Plot-area height in px (one paper-y unit). If the height is pinned via
        # set_plot_size, resolve it to the *paper* height the figure will be
        # given (in per-subplot mode the stored value is one panel); otherwise
        # derive it from the current figure height.
        pinned_h = self._pinned_inner_size(fig)[1]
        if pinned_h is not None:
            plot_h = pinned_h
        else:
            t = m.t if m.t is not None else 100
            plot_h = max(1.0, (fig.layout.height or 800) - t - new_b)

        # Sit the footer's bottom edge _FOOTER_PAD px above the figure's bottom,
        # below the axis labels (which occupy base_b); extra lines grow upward.
        y = -(new_b - _FOOTER_PAD) / plot_h
        ann = dict(text=text, showarrow=False, align='center',
                   x=0.5, xref='paper', xanchor='center',
                   y=y, yref='paper', yanchor='bottom')
        if footer_size is not None:
            ann['font'] = dict(size=footer_size)
        fig.add_annotation(**ann)
        return fig

    def _apply_fonts(self, fig):
        """Apply the resolved font sizes (user settings over plot-style
        defaults — see :meth:`_font_size`) to a Plotly figure."""
        if fig is None:
            return fig

        suptitle_size = self._font_size('suptitle_size')
        legend_size = self._font_size('legend_size')
        hover_size = self._font_size('hover_size')

        layout_updates = {}
        if suptitle_size is not None:
            layout_updates['title_font'] = dict(size=suptitle_size)
        if legend_size is not None:
            layout_updates['legend'] = dict(font=dict(size=legend_size))
        if hover_size is not None:
            layout_updates['hoverlabel'] = dict(font=dict(size=hover_size))
        if layout_updates:
            fig.update_layout(**layout_updates)

        # With the (possibly larger) custom title font now applied, re-reserve
        # the top space so a bigger suptitle can't collide with an above-legend.
        # _base_layout sized it for the default font; redo it for the real size.
        if suptitle_size is not None and fig.layout.title.text:
            height = fig.layout.height or 800
            leg = fig.layout.legend
            has_above = (leg.orientation == 'h' and leg.yref == 'container')
            legend_rows = (fig.layout.meta or {}).get('uc_legend_rows', 1)
            top_margin, title_pos, legend_pos = _top_space(
                fig.layout.title.text, (None, height / 100), has_above,
                title_font_size=suptitle_size, legend_rows=legend_rows)
            geo = {'margin': dict(t=top_margin), 'title': dict(y=title_pos['y'])}
            if legend_pos is not None:
                geo['legend'] = dict(y=legend_pos['y'])
            fig.update_layout(**geo)

        sp_size = self._font_size('subplot_title_size')
        if sp_size is not None and fig.layout.annotations:
            for ann in fig.layout.annotations:
                # Reference-line labels carry their own size; skip them so the
                # subplot-title sweep doesn't overwrite it.
                if (getattr(ann, 'name', None) or '').startswith(_LINE_LABEL_NAME):
                    continue
                ann.font = dict(size=sp_size)

        axes_title_size = self._font_size('axes_title_size')
        axes_tick_size = self._font_size('axes_tick_size')
        x_updates, y_updates = {}, {}
        if axes_title_size is not None:
            x_updates['title_font'] = dict(size=axes_title_size)
            y_updates['title_font'] = dict(size=axes_title_size)
        if axes_tick_size is not None:
            x_updates['tickfont'] = dict(size=axes_tick_size)
            y_updates['tickfont'] = dict(size=axes_tick_size)
        if x_updates:
            fig.update_xaxes(**x_updates)
        if y_updates:
            fig.update_yaxes(**y_updates)

        cb_size = self._font_size('colorbar_size')
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
        """Shared tail for every plotting method: apply the plot style and font
        sizes, add the optional footer, optionally collapse traces to
        legend-only, and cache the figure as ``last_fig``.

        This is the one step a new plotting method must not forget — keeping the
        ``self.last_fig`` cache contract in a single place. Grid sizing, decorations,
        and axis-limit ranges stay in each method, since those differ per plot type.

        ``footer`` is added after ``_apply_fonts`` (so the subplot-title font loop
        doesn't resize it) and before ``_enforce_plot_size`` (so the reserved
        bottom margin is included when pinning the plot area). The watermark
        follows it: sized in paper coords, it is unaffected by either, and
        running last keeps it out of the margin arithmetic entirely.

        With ``set_default_format(legend_scroll=False)`` the whole legend is
        shown (``_fit_full_legend``) once the plot size is final; static images
        always show it, via a fitted copy.
        """
        fig = self._apply_style(fig)
        fig = self._apply_grid(fig)
        fig = self._apply_fonts(fig)
        fig = self._apply_footer(fig, footer)
        fig = self._apply_watermark(fig)
        fig = self._enforce_plot_size(fig)
        if not self._apply_default('legend_scroll', None, True):
            fig = self._fit_full_legend(fig)
        if fig is not None and suppress_legends:
            fig.update_traces(visible='legendonly')
        self.last_fig = fig
        if self.static_images and fig is not None:
            # An image can't scroll: always render the whole legend.
            return self._render_static(self._full_legend_figure(fig))
        if fig is not None and self.copy_buttons:
            self._display_copy_button()
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
        fall back to interactive automatically. Static images always show the
        whole legend, made taller where it would otherwise scroll.

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

    def set_copy_buttons(self, enabled=True):
        """Show (default) or hide the '⧉ copy' button above interactive plots.

        The button copies the rendered plot to the clipboard as a PNG image,
        entirely in the browser (no kaleido round trip) — handy for pasting
        plots into slides, docs, or chats. It applies to interactive figures
        only; static-image mode (``set_static_images``) has no button because
        the browser's native right-click → Copy Image already works on PNGs.

        Examples
        --------
        nb.set_copy_buttons(False)   # hide the buttons
        nb.set_copy_buttons()        # show them again
        """
        self.copy_buttons = bool(enabled)
        print(f"Plot copy buttons {'on' if self.copy_buttons else 'off'}.")

    def _display_copy_button(self):
        """Display a small right-aligned '⧉ copy' button as its own HTML
        output, immediately before ``_finalize`` returns the figure — so the
        button lands directly above the plot in the notebook.

        The button and the plot are separate outputs, so at click time the
        handler finds its plot geometrically: the nearest ``.js-plotly-plot``
        just below the button (within 300px, so it never grabs a plot from a
        later cell if this figure was assigned instead of displayed).

        Rasterizing prefers ``Plotly.toImage`` when the page exposes a global
        ``Plotly`` (classic notebook renderer); otherwise — e.g. JupyterLab's
        mime extension keeps plotly.js module-scoped — it composites the
        plot's stacked ``svg.main-svg`` layers *and* WebGL ``.gl-canvas``
        layers onto a canvas, in DOM (= paint) order. Two traps the fallback
        handles:

        * WebGL traces (``scattergl``, used past ``WEBGL_POINT_THRESHOLD``)
          live on ``<canvas>``, not in the SVG layers, and their drawing
          buffer may already be cleared at click time — so the handler forces
          a synchronous scene redraw and snapshots the gl canvases in the
          same task, before any async SVG work.
        * When the page has a ``<base>`` tag (JupyterLab does), plotly writes
          ``clip-path``/``fill`` references as absolute page URLs. Those
          can't resolve inside a serialized standalone SVG — Chrome then
          skips the clip, Firefox drops the whole clipped group (no markers /
          no bars) — so the handler rewrites them back to local ``#id``
          fragments before serializing.

        The clipboard ``ClipboardItem`` wraps the PNG *promise* so the write
        starts inside the click gesture (required by Safari)."""
        # Only under a live IPython kernel: in plain scripts the `display`
        # shim degrades to print, which would dump this HTML to stdout.
        try:
            from IPython.core.getipython import get_ipython
        except ImportError:
            return
        if get_ipython() is None:
            return

        import uuid
        uid = f"unichart-copy-{uuid.uuid4().hex}"

        snippet = """
        <div id="__UID__" style="display:flex; justify-content:flex-start; margin:2px 0 0 0;">
            <button type="button" class="uc-plot-copy-btn"
                    title="Copy this plot as a PNG image to the clipboard"
                    style="font: inherit; font-size: 12px; padding: 2px 10px;
                           cursor: pointer; color: inherit; opacity: 0.75;
                           background: transparent; border-radius: 6px;
                           border: 1px solid rgba(128, 128, 128, 0.5);"
            >&#x29C9; copy</button>
        </div>
        <script>
        (function() {
            var btn = document.querySelector("#__UID__ .uc-plot-copy-btn");
            if (!btn) { return; }
            var flash = function(msg) {
                btn.textContent = msg;
                setTimeout(function() { btn.textContent = "\\u29C9 copy"; }, 1400);
            };
            // Fallback rasterizer for pages without a window.Plotly global:
            // composite the plot's stacked SVG layers and WebGL canvases onto
            // one canvas, in document order (= plotly's paint order: plot
            // layer, gl traces, annotations above). See the Python docstring
            // for the two traps handled here (gl buffer readback, absolute
            // url() references under a <base> tag).
            var svgToPng = async function(gd) {
                var layers = gd.querySelectorAll(
                    "svg.main-svg, canvas.gl-canvas-context, canvas.gl-canvas-focus");
                var firstSvg = gd.querySelector("svg.main-svg");
                if (!firstSvg) { throw new Error("no svg layers"); }
                var rect = firstSvg.getBoundingClientRect();
                var scale = 2;
                var canvas = document.createElement("canvas");
                canvas.width = Math.round(rect.width * scale);
                canvas.height = Math.round(rect.height * scale);
                var ctx = canvas.getContext("2d");
                ctx.scale(scale, scale);
                // WebGL drawing buffers may already be cleared (plotly doesn't
                // always preserveDrawingBuffer), so force a synchronous scene
                // redraw and snapshot the gl canvases NOW, in this same task —
                // before the async SVG loads below let the browser composite
                // (and clear) them again.
                try {
                    var plots = (gd._fullLayout || {})._plots || {};
                    for (var k in plots) {
                        var scene = plots[k]._scene;
                        if (scene && scene.draw) { scene.draw(); }
                    }
                } catch (e) { /* internal API — best effort */ }
                var snaps = [];
                for (var i = 0; i < layers.length; i++) {
                    if (layers[i].tagName !== "CANVAS") { snaps.push(null); continue; }
                    var s = null;
                    if (layers[i].width && layers[i].height) {
                        try {
                            s = document.createElement("canvas");
                            s.width = layers[i].width;
                            s.height = layers[i].height;
                            s.getContext("2d").drawImage(layers[i], 0, 0);
                        } catch (e) { s = null; }
                    }
                    snaps.push(s);
                }
                // Under a <base> tag plotly writes url() refs (clip-path,
                // gradient fills) as absolute page URLs, which break inside a
                // standalone serialized SVG; rewrite them to local #fragments.
                var URL_ATTRS = ["clip-path", "fill", "stroke", "filter", "mask"];
                var toLocal = function(v) {
                    return v.replace(/url\((['"]?)[^#)]*#/g, "url($1#");
                };
                var fixRefs = function(root) {
                    var els = root.querySelectorAll("*");
                    for (var i = 0; i < els.length; i++) {
                        for (var j = 0; j < URL_ATTRS.length; j++) {
                            var v = els[i].getAttribute(URL_ATTRS[j]);
                            if (v && v.indexOf("url(") !== -1 && v.indexOf("#") !== -1) {
                                els[i].setAttribute(URL_ATTRS[j], toLocal(v));
                            }
                        }
                        var st = els[i].getAttribute("style");
                        if (st && st.indexOf("url(") !== -1 && st.indexOf("#") !== -1) {
                            els[i].setAttribute("style", toLocal(st));
                        }
                    }
                };
                for (var i = 0; i < layers.length; i++) {
                    var r = layers[i].getBoundingClientRect();
                    var x = r.left - rect.left, y = r.top - rect.top;
                    if (layers[i].tagName === "CANVAS") {
                        if (snaps[i]) { ctx.drawImage(snaps[i], x, y, r.width, r.height); }
                        continue;
                    }
                    var clone = layers[i].cloneNode(true);
                    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
                    clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
                    fixRefs(clone);
                    var url = URL.createObjectURL(new Blob(
                        [new XMLSerializer().serializeToString(clone)],
                        {type: "image/svg+xml;charset=utf-8"}));
                    try {
                        await new Promise(function(res, rej) {
                            var img = new Image();
                            img.onload = function() {
                                ctx.drawImage(img, x, y, r.width, r.height);
                                res();
                            };
                            img.onerror = rej;
                            img.src = url;
                        });
                    } finally { URL.revokeObjectURL(url); }
                }
                return new Promise(function(res, rej) {
                    canvas.toBlob(function(b) {
                        if (b) { res(b); } else { rej(new Error("toBlob failed")); }
                    }, "image/png");
                });
            };
            btn.addEventListener("click", function() {
                // Nearest rendered plot just below the button = this plot.
                var b = btn.getBoundingClientRect();
                var best = null, bestTop = Infinity;
                document.querySelectorAll(".js-plotly-plot").forEach(function(el) {
                    var r = el.getBoundingClientRect();
                    if (r.height > 0 && r.top >= b.bottom - 1 &&
                            r.top - b.bottom < 300 && r.top < bestTop) {
                        best = el; bestTop = r.top;
                    }
                });
                if (!best) { flash("\\u2717 no plot"); return; }
                if (!navigator.clipboard || !window.ClipboardItem) {
                    flash("\\u2717 no clipboard"); return;
                }
                // ClipboardItem wraps the promise so the write starts inside
                // the click gesture even though rasterizing takes a moment.
                var png = (window.Plotly && window.Plotly.toImage)
                    ? window.Plotly.toImage(best, {format: "png", scale: 2})
                          .then(function(u) { return fetch(u); })
                          .then(function(r) { return r.blob(); })
                    : svgToPng(best);
                navigator.clipboard.write([new ClipboardItem({"image/png": png})])
                    .then(function() { flash("\\u2713 copied"); },
                          function() { flash("\\u2717 failed"); });
            });
        })();
        </script>
        """.replace("__UID__", uid)

        display(HTML(snippet))

    # ------------------------------------------------------------------
    # Main Plot Function
    # ------------------------------------------------------------------
    def plot(self, x=None, y=None, by='vars', figsize=None, ncols=None, nrows=None,
                subplot_titles=None, suptitle=None, footer=None, suppress_legends=None,
                legend=None, hspace=None, vspace=None, **kwargs):
        """
        Main plotting wrapper.

        Parameters:
        -----------
        by : str, optional
            'vars'    (default) - Subplot per Y variable.
            'sets' / 'datasets' - Subplot per Dataset.
            'ymult'           - Single plot, multiple Y axes (delegates to plot_ymult).
            'marginal'        - Scatter with marginal distribution strips
                                (delegates to plot_marginal).
        legend : str, optional
            Legend placement, matching ``plot_ymult`` (default 'above', or the
            ``set_default_format(legend=)`` default):
            'above' - horizontal legend above the plot.
            'right' - vertical legend to the right of the plot.
            'off'   - hide the legend.
        hspace, vspace : number or str, optional
            Gap between subplot columns / rows: pixels if 1 or more (``60``,
            ``'60px'``), a fraction of the plot area if below 1. Defaults to
            ``set_default_format(hspace=, vspace=)`` or, unset, a fixed pixel
            budget per gap (80 px / 70 px) whatever the grid size. The bar,
            box, histogram and contour methods take the same two arguments.
        """
        if figsize is None: figsize = self.figsize
        legend = self._apply_default('legend', legend, 'above')
        suppress_legends = self._apply_default('suppress_legends', suppress_legends, False)

        # Delegate to the multi-y wrapper if requested
        if by == 'ymult':
            return self.plot_ymult(x=x, y=y, suptitle=suptitle,
                                     figsize=figsize, legend=legend,
                                     suppress_legends=suppress_legends,
                                     style_by=kwargs.pop('style_by', None))

        if by == 'marginal':
            return self.plot_marginal(x=x, y=y, figsize=figsize, ncols=ncols, nrows=nrows,
                                      subplot_titles=subplot_titles, suptitle=suptitle,
                                      footer=footer, legend=legend,
                                      suppress_legends=suppress_legends,
                                      hspace=hspace, vspace=vspace, **kwargs)

        self._clear_last_fig()

        if x is None: x = self.last_x
        if y is None: y = self.last_y
        self.last_x = x
        self.last_y = y
        self._record_plot_call('plot', locals())

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
                return_axes=True,
                **self._spacing_kwargs(hspace, vspace),
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
                **self._spacing_kwargs(hspace, vspace),
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
                # yref='paper' explicitly: update_layout merges, and an inherited
                # container yref (from the above-legend default) parks this legend
                # in the title band, where autoexpand takes it out of the plot area.
                legend=dict(orientation='v', xanchor='left', x=1.02,
                            yanchor='top', y=1, yref='paper'),
                margin=dict(r=self._keep_right_margin(fig, 160), t=_top),
            )
        else:  # 'above' (default)
            _legend, _top = _above_legend_layout(suptitle or self.suptitle, figsize)
            fig.update_layout(legend=_legend,
                              margin=dict(r=self._keep_right_margin(fig, 80),
                                          t=_top))
        return self._finalize(fig, suppress_legends, footer=footer or self.footer)

    # ------------------------------------------------------------------
    # Multi-Y plot wrapper
    # ------------------------------------------------------------------
    def plot_ymult(self, x=None, y=None, suptitle=None, footer=None, figsize=None,
                     legend=None, legend_group_by='sets', suppress_legends=None,
                     style_by=None):
        """
        Single plot, multiple Y-axes. All selected datasets overlay on the same x-axis.
        Applies all notebook-level formatting: axis_limits, variable_formats, lines, highlights.

        ``legend`` (default 'above') and ``suppress_legends`` (default False) fall
        back to the ``set_default_format`` defaults when not passed.

        ``style_by`` auto-differentiates the y variables by cycling one or more
        attributes per variable: ``'color'``, ``'marker'``, ``'linestyle'``, or
        a combination (``'color+marker'``, ``['marker', 'linestyle']``, ...).
        Colors and markers cycle ``color_map`` / ``marker_map``, so they follow
        the active palette and :meth:`set_plot_style`.
        Any attribute set via :meth:`var_format` still overrides the auto value
        for that variable, and unstyled attributes fall back to the dataset as
        usual. With ``'color'``, each variable's y-axis is tinted to match.
        """
        if figsize is None: figsize = self.figsize
        legend = self._apply_default('legend', legend, 'above')
        suppress_legends = self._apply_default('suppress_legends', suppress_legends, False)
        self._clear_last_fig()
        if x is None: x = self.last_x
        if y is None: y = self.last_y
        self.last_x, self.last_y = x, y
        self._record_plot_call('plot_ymult', locals())

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
            style_by=style_by,
            color_cycle=self.color_map,
            marker_cycle=self.marker_map,
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
                                  line_color=l['color'],
                                  **_prefix_annotation(self._line_label(l, 'vertical')))
            elif col in yref_for:
                yref = yref_for[col]
                for l in lines:
                    fig.add_shape(type='line', x0=0, x1=1,
                                  y0=l['level'], y1=l['level'],
                                  xref='paper', yref=yref,
                                  line=dict(color=l['color'], dash=l['dash'] or 'solid'))
                    # The shape spans paper so it reaches across the stacked
                    # y-axes; the label is placed against the x-axis domain
                    # instead, so it lands inside the plot area, not a margin.
                    ann = self._line_label(l, 'horizontal')
                    if ann:
                        fig.add_annotation(xref='x domain', yref=yref, **ann)

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

    def explore(self, **kwargs):
        """Open the explorer GUI on this notebook — build plots on the fly.

        Where :meth:`dashboard` renders a board you specified in code, this
        opens an empty one you fill in the browser: load more data from the
        header's data bar, hit "+ add panel", and point each panel at whatever
        columns you want. Datasets loaded in the GUI stay on this notebook
        afterwards; panel renders leave its state untouched.

        From a kernel the board has no close button: it would have to end the
        process, and here that process is the kernel. Standalone — the
        ``unichart`` command, or a script — it does, and so does ``exit()`` in
        the terminal pane.

        Thin wrapper around :func:`unichart_dashboard.explore`; imported lazily
        so the optional Dash dependency isn't required to use the rest of the
        toolkit. See that function for the keyword options.
        """
        from unichart_dashboard import explore as _explore
        return _explore(self, **kwargs)

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
    # Marginal distribution plot wrapper
    # ------------------------------------------------------------------
    def plot_marginal(self, x=None, y=None, by='vars', marginal=None, marginal_x=None,
                      marginal_y=None, marginal_size=None, nbins=None, bin_size=None,
                      bin_start=None, bin_end=None, histnorm=None, alpha=None, color=None,
                      subplot_titles=None, suptitle=None, footer=None, figsize=None,
                      ncols=None, nrows=None, legend=None, suppress_legends=None,
                      hspace=None, vspace=None, **kwargs):
        """Scatter of ``y`` against ``x`` with marginal distribution plots: the
        distribution of ``x`` in a strip above the plot and of ``y`` in a strip
        to its right, each sharing the main panel's axis.

        Also reachable as ``plot(x, y, by='marginal')``. The scatter itself is
        styled like ``plot`` (per-dataset color / marker / alpha / hue /
        reg_order, hover parms, lines and highlights); the strips take the
        dataset color.

        Parameters
        ----------
        by : str, optional
            'vars' (default) - one block per (x, y) pair, datasets overlaid.
            'sets' / 'datasets' - one block per dataset, single x / y.
        marginal : str, optional
            Kind of distribution drawn in both strips (default 'histogram'):
            'histogram', 'box', 'violin', 'rug' or 'kde' (Gaussian kernel
            density, via scipy).
        marginal_x, marginal_y : str or False, optional
            Override the kind for one side, or ``False`` to drop that strip.
        marginal_size : float, optional
            The strips' share of each block (default 0.2, must be below 0.6).
        nbins, bin_size, bin_start, bin_end, histnorm : optional
            Histogram binning / normalisation for ``marginal='histogram'``,
            as in ``histogram()``; ``histnorm`` falls back to the
            ``set_default_format`` default.
        alpha : float, optional
            Opacity of the strips (default 0.7); the scatter keeps each
            dataset's own alpha.
        color : str, optional
            One color for every dataset (scatter and strips).
        legend : str, optional
            'above' (default), 'right' or 'off', as in ``plot``.
        hspace, vspace : number or str, optional
            Gap between blocks, as in ``plot``. The gap between a main panel
            and its strips is fixed (a few px).

        Remaining keyword arguments (``hue=``, ``marker=``, ``markersize=``,
        ``display_parms=``) pass through to ``unimarginal``.
        """
        if figsize is None: figsize = self.figsize
        legend = self._apply_default('legend', legend, 'above')
        suppress_legends = self._apply_default('suppress_legends', suppress_legends, False)
        histnorm = self._apply_default('histnorm', histnorm, '')
        alpha = self._apply_default('alpha', alpha, 0.7)
        if marginal is None: marginal = 'histogram'
        if marginal_size is None: marginal_size = 0.2

        self._clear_last_fig()

        if x is None: x = self.last_x
        if y is None: y = self.last_y
        self.last_x = x
        self.last_y = y
        self._record_plot_call('plot_marginal', locals())

        # Grid precedence as in plot: explicit arg > standing default > sticky last grid
        ncols, nrows = self._resolve_grid(ncols, nrows)
        if ncols is None and nrows is None and (
                self.last_ncols is not None or self.last_nrows is not None):
            ncols, nrows = self.last_ncols, self.last_nrows
        self.last_ncols = ncols
        self.last_nrows = nrows

        common = dict(
            list_of_datasets=self.sets, x=x, y=y,
            marginal=marginal, marginal_x=marginal_x, marginal_y=marginal_y,
            marginal_size=marginal_size, nbins=nbins, bin_size=bin_size,
            bin_start=bin_start, bin_end=bin_end, histnorm=histnorm, alpha=alpha,
            color=color, display_parms=self.display_parms,
            suptitle=suptitle or self.suptitle, xlabel=self.x_label, ylabel=self.y_label,
            darkmode=self.darkmode, figsize=figsize, ncols=ncols, nrows=nrows,
            axis_limits=self.axis_limits, legend=legend, return_axes=True,
            **self._spacing_kwargs(hspace, vspace),
        )
        common.update(kwargs)

        if by in ['sets', 'datasets']:
            fig = unimarginal_per_dataset(**common)
            x1 = x[0] if isinstance(x, list) else x
            y1 = y[0] if isinstance(y, list) else y
            pairs = [(x1, y1)] * len([d for d in self.sets if d.select])
        else:
            fig = unimarginal(subplot_titles=subplot_titles, **common)
            pairs = _xy_pairs(x, y)

        if fig is None: return

        x_list = x if isinstance(x, list) else [x]
        y_list = y if isinstance(y, list) else [y]
        calc_ncols = max(1, _calc_grid(len(pairs), nrows, ncols)[1])

        # Decorations go on the main panel and on the strip that shares the
        # decorated axis: an x line is drawn on the main panel and the top
        # strip, a y line on the main panel and the right strip.
        plot_items, refs = [], []
        for idx, (xi, yi) in enumerate(pairs):
            main, top, right = _marginal_block_refs(idx, calc_ncols)
            plot_items += [(xi, yi), (xi, None), (None, yi)]
            refs += [main, top, right]
        fig = self._apply_decorations(fig, x_list, y_list, 'vars', calc_ncols,
                                      plot_items, refs=refs)

        if legend == 'off':
            fig.update_layout(showlegend=False)
        elif legend == 'right':
            _top, _, _ = _top_space(suptitle or self.suptitle, figsize, False)
            fig.update_layout(
                showlegend=True,
                # yref='paper' explicitly: update_layout merges, and an inherited
                # container yref (from the above-legend default) parks this legend
                # in the title band, where autoexpand takes it out of the plot area.
                legend=dict(orientation='v', xanchor='left', x=1.02,
                            yanchor='top', y=1, yref='paper'),
                margin=dict(r=self._keep_right_margin(fig, 160), t=_top),
            )
        else:  # 'above' (default)
            _legend, _top = _above_legend_layout(suptitle or self.suptitle, figsize)
            fig.update_layout(legend=_legend,
                              margin=dict(r=self._keep_right_margin(fig, 80),
                                          t=_top))
        return self._finalize(fig, suppress_legends, footer=footer or self.footer)


    # ------------------------------------------------------------------
    # The bar Command
    # ------------------------------------------------------------------
    def bar(self, x=None, y=None, markers=None, by='vars', barmode=None, agg=None,
            color=None, suptitle=None, footer=None, figsize=None, ncols=None, nrows=None, suppress_legends=None,
            hspace=None, vspace=None):
        """
        Bar chart: one bar per (dataset, ``x`` category), whose height is the
        ``agg`` of ``y`` over the rows in that category.

        Parameters
        ----------
        x : str
            Column holding the categories. Not used for ``by='dataset_x'``,
            where the datasets themselves are the categories.
        y : str or list of str
            Column(s) to reduce and draw as bars.
        markers : str or list of str, optional
            Column(s) overlaid on the bars, reduced with the same ``agg``, as a
            marker symbol, a tick or a whisker (the ``style`` key of
            ``var_format``). See "Overlay columns" below.
        by : {'vars', 'sets', 'dataset_x'}
            'vars'      (default) one subplot per y column, bars colored by dataset.
            'sets'      one subplot per dataset, bars colored by variable
                        ('datasets' is accepted as an alias).
            'dataset_x' one plot whose categories are the datasets, one bar and
                        one y-axis per variable.
            Any other value raises ``ValueError``.
        barmode : {'group', 'stack'}
            Default 'group'. Not meaningful for ``by='dataset_x'`` (each
            variable has its own axis), where it is ignored with a warning.
        agg : str, callable or False
            How the rows of a category are reduced: 'mean' (default), 'sum',
            'min', 'max', 'median', 'std', 'var', 'count', 'first', 'last', any
            other pandas ``Series`` reducer name ('nunique', 'sem', ...), or a
            callable taking a Series. ``agg=False`` draws the rows as they are
            (one bar per row) for data that is already one row per category.
            Unknown values raise ``ValueError``.
        color : str, optional
            Single bar color for ``by='vars'``. The other views color by
            variable and ignore it with a warning.
        suptitle, footer, figsize, ncols, nrows, suppress_legends
            The usual layout options. ``ncols``/``nrows`` do not apply to the
            single-panel ``by='dataset_x'`` view.

        ``barmode``, ``agg`` and ``suppress_legends`` fall back to the
        ``set_default_format`` defaults when not passed. ``x`` and ``y`` fall
        back to the ones used by the previous plotting call.

        Hovering a bar or overlay reports ``<agg> <column>: <value> (n=<rows>)``,
        where n is the number of valid (non-NaN) rows behind that value.

        A column (``x``, ``y`` or ``markers``) missing from every selected
        dataset raises ``ValueError`` naming it; one missing from only some
        datasets is drawn where present and warned about.

        The x-axis is drawn as categories (evenly spaced bars, even for a
        numeric ``x``) unless ``scale()`` has been set on ``x``, in which case
        the axis stays numeric so the range applies.

        Overlay columns
        ~~~~~~~~~~~~~~~
        Formatting comes from ``var_format``::

            nb.var_format('EGT_LIMIT', color='red', marker='*', markersize=18)
            nb.bar(x='PHASE', y='EGT', markers='EGT_LIMIT')

            nb.var_format('EGT_LIMIT', style='tick')     # horizontal dash at the value
            nb.var_format('EGT_LIMIT', style='whisker')  # dash + stem to the bar top

        In the default ``by='vars'`` view each panel names its variable on the
        y-axis; setting ``nb.y_label`` reverts to one shared y title on the
        first column. In a multi-panel chart, overlay columns pair positionally
        with the y variables: the i-th marker column draws only on the i-th y
        variable's panel, attached to that variable's bars (extras fall back to
        the first panel). So ``y=['EGT', 'RU'], markers=['EGT_LIMIT',
        'RU_LIMIT']`` puts each limit on its own variable's panel and scale.
        In ``by='sets'`` the pairing is the same, within each dataset's panel:
        the i-th tick/whisker sits on the i-th y variable's bar.

        ``scale()`` accepts an overlay column too, since the overlay shares the
        bar's y-axis. Its range is unioned with the paired bar variable's own
        scale so both the bars and the limit line stay in frame.
        """
        if figsize is None: figsize = self.figsize

        valid_by = ('vars', 'sets', 'datasets', 'dataset_x')
        if by not in valid_by:
            raise ValueError(f"by must be one of {valid_by}, got {by!r}")

        # Arguments that the chosen view cannot honor: say so instead of
        # silently dropping them. Only explicitly passed values are reported
        # (the stored defaults are not the caller's intent for this call).
        if by == 'dataset_x':
            ignored = [name for name, val in (('x', x), ('barmode', barmode),
                                              ('color', color), ('ncols', ncols),
                                              ('nrows', nrows)) if val is not None]
            if ignored:
                warnings.warn(f"bar(by='dataset_x') ignores {', '.join(ignored)}: "
                              "the datasets are the x categories, each variable has "
                              "its own axis and color, and there is a single panel.",
                              UserWarning, stacklevel=2)
        elif by in ('sets', 'datasets') and color is not None:
            warnings.warn("bar(by='sets') colors bars by variable and ignores color=; "
                          "use var_format(<variable>, color=...) instead.",
                          UserWarning, stacklevel=2)

        barmode = self._apply_default('barmode', barmode, 'group')
        agg = self._apply_default('agg', agg, 'mean')
        _resolve_agg(agg)                      # validate up front, before any drawing
        suppress_legends = self._apply_default('suppress_legends', suppress_legends, False)
        ncols, nrows = self._resolve_grid(ncols, nrows)
        self._clear_last_fig()

        if x is None: x = self.last_x
        if y is None: y = self.last_y
        self.last_x, self.last_y = x, y
        self._record_plot_call('bar', locals())

        needs_x = by != 'dataset_x'
        missing = [name for name, val in (('x', x), ('y', y))
                   if val is None and (needs_x or name != 'x')]
        if missing:
            print(f"Error: bar() needs {' and '.join(missing)} (none given and none "
                  "remembered from a previous plotting call).")
            return None

        y_list = y if isinstance(y, list) else [y]
        markers_list = markers if isinstance(markers, list) else ([markers] if markers else [])

        active_sets = [d for d in self.sets if d.select]
        if not active_sets:
            print("No datasets selected.")
            return None
        self._check_bar_columns(active_sets, ([x] if needs_x else []) + y_list + markers_list)

        categorical_x = x not in self.axis_limits

        if by == 'dataset_x':
            fig = unibar_datasets_as_x(
                list_of_datasets=self.sets, y=y_list, agg=agg, markers=markers,
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
                barmode=barmode, agg=agg, categorical_x=categorical_x,
                suptitle=suptitle or self.suptitle, xlabel=self.x_label, ylabel=self.y_label,
                figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, return_axes=True,
                **self._spacing_kwargs(hspace, vspace),
            )
        else:
            fig = unibar(
                list_of_datasets=self.sets, x=x, y=y, markers=markers,
                variable_formats=self.variable_formats,         # <-- pass through
                barmode=barmode, color=color, agg=agg, categorical_x=categorical_x,
                suptitle=suptitle or self.suptitle, xlabel=self.x_label, ylabel=self.y_label,
                figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, return_axes=True,
                **self._spacing_kwargs(hspace, vspace),
            )

        if fig:
            if x in self.axis_limits:
                fig.update_xaxes(range=self.axis_limits[x])

            n_items = len(active_sets) if by in ['sets', 'datasets'] else len(y_list)
            calc_ncols = max(1, _calc_grid(n_items, nrows, ncols)[1])

            # Overlay columns (markers=) share the bar's y-axis, so a scale()
            # on one of them widens that axis (unioned with the bar variable's
            # own scale) rather than being ignored. In the by='sets' view every
            # overlay rides the shared axis; in by='vars' each overlay pairs
            # positionally with one y variable and widens only that panel.
            if by in ['sets', 'datasets']:
                marker_lims = [self.axis_limits[m] for m in markers_list if m in self.axis_limits]
                primary_y = y_list[0]
                yr = _union_ranges([self.axis_limits.get(primary_y)] + marker_lims)
                if yr is not None:
                    fig.update_yaxes(range=yr)
            else:
                for idx, yi in enumerate(y_list):
                    paired_lims = [self.axis_limits[m]
                                   for m_idx, m in enumerate(markers_list)
                                   if m in self.axis_limits
                                   and (m_idx if m_idx < len(y_list) else 0) == idx]
                    yr = _union_ranges([self.axis_limits.get(yi)] + paired_lims)
                    if yr is not None:
                        r = (idx // calc_ncols) + 1
                        c = (idx % calc_ncols) + 1
                        fig.update_yaxes(range=yr, row=r, col=c)

            dec_mode = 'sets' if by in ['sets', 'datasets'] else 'vars'
            dec_items = [(x, yi) for yi in y_list] if dec_mode == 'vars' else None
            fig = self._apply_decorations(fig, [], y_list, dec_mode, calc_ncols, dec_items)

            fig = self._finalize(fig, suppress_legends, footer=footer or self.footer)

        return fig

    # ------------------------------------------------------------------
    # The box Command
    # ------------------------------------------------------------------
    def box(self, x=None, y=None, by='vars', boxmode=None, points=None, notched=False,
                color=None, suptitle=None, footer=None, figsize=None, ncols=None, nrows=None, suppress_legends=None,
                hspace=None, vspace=None):
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
        self._record_plot_call('box', locals())

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
                fig.update_layout(legend=_legend,
                              margin=dict(r=self._keep_right_margin(fig, 80),
                                          t=_top))
                fig = self._finalize(fig, suppress_legends, footer=footer or self.footer)
            return fig

        elif by in ['sets', 'datasets']:
            primary_y = y_list[0]
            y_limit = self.axis_limits.get(primary_y)
            fig = unibox_per_dataset(
                list_of_datasets=self.sets, x=x, y=y, boxmode=boxmode,
                points=points, notched=notched,
                suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, y_lim=y_limit, return_axes=True,
                **self._spacing_kwargs(hspace, vspace),
            )
        else:
            fig = unibox(
                list_of_datasets=self.sets, x=x, y=y, boxmode=boxmode,
                points=points, notched=notched, color=color,
                suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, y_lim=None, return_axes=True,
                **self._spacing_kwargs(hspace, vspace),
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
            fig.update_layout(legend=_legend,
                              margin=dict(r=self._keep_right_margin(fig, 80),
                                          t=_top))
            fig = self._finalize(fig, suppress_legends, footer=footer or self.footer)

        return fig

    # ------------------------------------------------------------------
    # The histogram Command
    # ------------------------------------------------------------------
    def histogram(self, x=None, y=None, histfunc=None, by='vars', nbins=None,
                    bin_size=None, bin_start=None, bin_end=None,
                    histnorm=None, barmode=None, alpha=None,
                    color=None, suptitle=None, footer=None, figsize=None, ncols=None, nrows=None, suppress_legends=None,
                    opacity=None, hspace=None, vspace=None):
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
        self._record_plot_call('histogram', locals())

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
                darkmode=self.darkmode, x_lim=limit, return_axes=True,
                **self._spacing_kwargs(hspace, vspace),
            )
            if fig:
                fig = self._apply_decorations(fig, x_list, [], 'sets', 1)
        else:
            fig = unihistogram(
                list_of_datasets=self.sets, x=x, y=y, histfunc=histfunc, nbins=nbins,
                bin_size=bin_size, bin_start=bin_start, bin_end=bin_end,
                histnorm=histnorm, barmode=barmode, alpha=alpha, color=color,
                suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, x_lim=limit, return_axes=True,
                **self._spacing_kwargs(hspace, vspace),
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
                    suptitle=None, footer=None, figsize=None, ncols=None, nrows=None, suppress_legends=None,
                    hspace=None, vspace=None):
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

        Contour boundaries are drawn in each set's own ``color``,
        ``linestyle`` and ``linewidth``. ``contours_coloring`` defaults to
        ``'fill'``; when a single plot would stack several filled fields on
        each other (more than one set with ``by='vars'``, or more than one
        ``z`` with ``by='sets'``) it drops to bare boundaries so they stay
        readable. The one mode where a set's color has no effect is an
        explicit ``contours_coloring='lines'``, which is Plotly's
        "color the lines from the colorscale by z" mode.

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
        self._record_plot_call('contour', locals())

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
                darkmode=self.darkmode, axis_limits=self.axis_limits, return_axes=True,
                **self._spacing_kwargs(hspace, vspace),
            )
        else:
            fig = unicontour(
                list_of_datasets=self.sets, x=x, y=y, z=z,
                contours_coloring=contours_coloring, colorscale=colorscale,
                interpolate=interpolate, interp_res=interp_res, interp_method=interp_method,
                ncontours=ncontours, overlay_datasets=overlay_datasets,
                suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, axis_limits=self.axis_limits, return_axes=True,
                **self._spacing_kwargs(hspace, vspace),
            )
            
        if fig:
            z_list = z if isinstance(z, list) else [z]
            if by in ['sets', 'datasets']:
                fig = self._apply_decorations(fig, [x], [y], 'sets', 1,
                                              highlight_layer='above')
            else:
                _nc = max(1, _calc_grid(len(z_list), nrows, ncols)[1])
                fig = self._apply_decorations(fig, [x], [y], 'vars', _nc,
                                              [(x, y) for _ in z_list],
                                              highlight_layer='above')
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
              sig_figs=None, decimals=None, output=None):
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
            Mutually exclusive with ``decimals``. Left out, each set's own
            ``sig_figs`` / ``decimals`` (see :meth:`sig_figs`,
            :meth:`decimals`) formats its rows, and a set carrying neither
            keeps the built-in ``.5g`` display.
        decimals : int, optional
            The fixed-decimal-places alternative to ``sig_figs``: round every
            float column to this many places after the point, keeping trailing
            zeros in the rendered table (``decimals=2`` shows ``1.5`` as
            ``1.50``); use ``decimals=0`` for whole numbers. Affects display
            only, exactly as ``sig_figs`` does, and with the same caveat that
            Markdown output re-renders plain numeric columns without the
            trailing zeros. Mutually exclusive with ``sig_figs``.
        output : {None, 'df', 'md', 'fig'}, optional
            What to return:

            - ``None`` (default): render and display the styled HTML table.
              Column headers are clickable to sort the table by that column
              (click again to reverse the order); numeric columns sort
              numerically, with missing values (``'-'``) always last. A
              ``Copy`` button above the table copies it (in the current sort
              order) to the clipboard as tab-separated text plus an HTML
              table, so it pastes into Excel/Sheets one value per cell. Cell
              text can also be selected and copied directly. Clicking the
              small ``⌕`` icon in a column header opens a filter box under
              that column, hiding non-matching rows as you type: plain text
              is a case-insensitive substring match, ``*``/``?`` wildcards
              match the whole cell (``alt*`` starts-with, ``*ft``
              ends-with, ``?`` any one character), and numeric columns
              also accept ``>10``, ``>=10``, ``<10``, ``<=10``, ``=10``, or
              ``5..20`` (inclusive range). Terms can be combined with ``&``
              (AND) and ``|`` or ``,`` (OR), e.g. ``>=5 & <20`` or
              ``idle, cruise``. :meth:`pandas.DataFrame.query`-style syntax
              also works: ``and``/``or``/``not`` keywords, ``==`` / ``!=``,
              quoted strings for exact (rather than substring) matches,
              ``in ['idle', 'cruise']`` lists, and a redundant leading
              column name — so ``power > 5 and power < 20`` typed in the
              ``power`` box behaves like the equivalent ``df.query``.
              Clicking ``⌕`` again collapses the box but keeps its filter
              applied; a filtered column stays marked with an accent-colored
              ``⌕`` and header underline, and hovering ``⌕`` shows the
              active filter. Esc clears and closes a filter. The Copy button
              copies only the rows currently shown.
            - ``'df'``: return the assembled :class:`pandas.DataFrame`.
            - ``'md'``: return a GitHub-flavored Markdown string.
            - ``'fig'``: return the styled Plotly ``go.Figure`` (a ``go.Table``),
              with ``sig_figs``/``decimals`` and dark-mode already applied.
              Useful for embedding the table alongside other figures (e.g. in a
              dashboard panel) without triggering the HTML display side
              effect.

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

        Show every float to two decimal places instead::

            chart.table(cols=['speed', 'power'], decimals=2)
        """
        if output is not None and output not in ('df', 'md', 'fig'):
            print(f"Unknown output mode '{output}'. Use None, 'df', 'md', or 'fig'.")
            return
        if sig_figs is not None and (not isinstance(sig_figs, int) or
                                     isinstance(sig_figs, bool) or sig_figs < 1):
            print("sig_figs must be a positive integer.")
            return
        if decimals is not None and (not isinstance(decimals, int) or
                                     isinstance(decimals, bool) or decimals < 0):
            print("decimals must be a non-negative integer.")
            return
        if sig_figs is not None and decimals is not None:
            print("Pass either sig_figs or decimals, not both.")
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

        # With neither argument given, each set's own ``sig_figs`` /
        # ``decimals`` (see :meth:`sig_figs`, :meth:`decimals`) decides its
        # rows' precision, so the format can vary row by row — the 'Set' column
        # says which set a row came from.
        set_fmt = ({ds.index: self._set_display_fmt(ds.sig_figs, ds.decimals)
                    for ds in self.sets}
                   if sig_figs is None and decimals is None else {})
        per_set_sig = any(f is not None for f in set_fmt.values())

        # Capture float columns before fillna (which can turn columns
        # containing NaN into object dtype) so the sig_figs / decimals
        # formatting below knows which columns to round.
        float_cols = (list(final_df.select_dtypes(include='float').columns)
                      if sig_figs is not None or decimals is not None or per_set_sig
                      else [])

        final_df = final_df.fillna('-')

        if output == 'df':
            return final_df

        if sig_figs is not None or decimals is not None:
            fmt = ((lambda v: self._sig_fig_str(v, sig_figs))
                   if sig_figs is not None
                   else (lambda v: self._decimals_str(v, decimals)))
            for c in float_cols:
                final_df[c] = final_df[c].map(fmt)
        elif per_set_sig:
            row_fmt = [set_fmt.get(i) for i in final_df['Set']]
            for c in float_cols:
                final_df[c] = [f(v) if f else v
                               for v, f in zip(final_df[c], row_fmt)]

        if output == 'md':
            try:
                return final_df.to_markdown(index=False)
            except ImportError:
                print("Markdown output requires the 'tabulate' package "
                      "(pip install tabulate).")
                return

        fig = self._build_table_figure(final_df, title=title)

        if output == 'fig':
            return fig

        self._display_html_table(self._format_table_display(final_df),
                                 title=title)

    def _set_display_fmt(self, sig_figs, decimals):
        """Formatter for one set's displayed values, from its ``sig_figs`` /
        ``decimals`` attributes — or None when the set carries neither and the
        caller's own built-in precision should stand."""
        if sig_figs:
            return lambda v: self._sig_fig_str(v, sig_figs)
        if decimals is not None:
            return lambda v: self._decimals_str(v, decimals)
        return None

    @staticmethod
    def _sig_fig_str(v, sig_figs):
        """
        Round ``v`` to ``sig_figs`` significant figures, rendered as a plain
        decimal string (never scientific notation). Non-floats (e.g. the ``'-'``
        fill value or string columns) pass through unchanged.
        """
        if not isinstance(v, float) or not np.isfinite(v):
            return v
        if v == 0:
            return f"{0:.{sig_figs - 1}f}"
        digits = sig_figs - int(np.floor(np.log10(abs(v)))) - 1
        if digits <= 0:
            return f"{round(v, digits):.0f}"
        return f"{v:.{digits}f}"

    @staticmethod
    def _decimals_str(v, decimals):
        """
        Round ``v`` to ``decimals`` places after the decimal point, keeping
        trailing zeros (``decimals=0`` gives a whole number). Non-floats (e.g.
        the ``'-'`` fill value or string columns) pass through unchanged, as in
        :meth:`_sig_fig_str`.
        """
        if not isinstance(v, float) or not np.isfinite(v):
            return v
        return f"{v:.{decimals}f}"

    @staticmethod
    def _format_table_display(final_df, fmt='.5g'):
        """
        Copy of ``final_df`` with float columns rendered for display. Values
        already turned into strings (by ``sig_figs``/``decimals``) are left
        alone — including in a column where only *some* rows were formatted,
        which a per-set ``sig_figs`` produces and which pandas stores as an
        object column.
        """
        display_df = final_df.copy()
        for col in display_df.columns:
            if display_df[col].dtype.kind in ('f', 'O'):
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:{fmt}}" if isinstance(x, float) else x)
        return display_df

    def _build_table_figure(self, final_df, title=None):
        """
        Build the Plotly ``go.Table`` figure for ``final_df``, styled to match
        ``self.darkmode`` and the current template/fonts. Shared by
        :meth:`table` and :meth:`summary` (both accept ``output='fig'``).
        """
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
            'template': self._template_name(),
            'margin': dict(l=20, r=20, t=50, b=20),
        }
        fig.update_layout(**layout_args)

        self.last_fig = fig
        return self._apply_fonts(fig)

    def _display_html_table(self, display_df, title=None):
        """
        Render ``display_df`` (values already formatted for display) as the
        shared styled HTML table: click-to-sort headers, per-column ``⌕``
        filter boxes, a Copy-to-clipboard button (TSV + HTML, visible rows
        only), and a light/dark palette following ``self.darkmode``. Used by
        :meth:`table`, :meth:`summary`, :meth:`list_sets` and
        :meth:`list_parms`.
        """
        header_size = self.table_header_size or 22
        cell_size = self.table_cell_size or 20
        title_size = self.suptitle_size or header_size + 4

        import uuid
        table_uid = f"unichart-table-{uuid.uuid4().hex}"

        # Palette for the styled HTML table, following self.darkmode like the
        # Plotly outputs do.
        if self.darkmode:
            pal = dict(
                text='#e8e8e8', wrap_border='#464646',
                th_bg='#2b2b2b', th_hover='#383838', th_sorted='#31405a',
                th_border='#4a4a4a', cell_border='#343434',
                row_odd='#1f1f1f', row_even='#262626', row_hover='#2b3648',
                shadow='0 1px 4px rgba(0,0,0,0.5)', copied='#7bc67e',
                accent='#6b93c9',
            )
        else:
            pal = dict(
                text='#000', wrap_border='#cfcfcf',
                th_bg='#ececec', th_hover='#e0e0e0', th_sorted='#dbe6f5',
                th_border='#c9c9c9', cell_border='#e3e3e3',
                row_odd='#ffffff', row_even='#f6f6f6', row_hover='#edf3fb',
                shadow='0 1px 4px rgba(0,0,0,0.10)', copied='#2e7d32',
                accent='#4a80c4',
            )

        # escape=True so cell text like a dataset query "speed < 20" can't
        # break the markup or inject HTML.
        html_table = display_df.to_html(index=False, escape=True)
        if title:
            caption_html = (
                f'<caption style="caption-side:top;text-align:center;'
                f'font-weight:600;color:{pal["text"]};font-size:{title_size}px;'
                f'padding:6px 10px;background-color:{pal["th_bg"]};'
                f'border-bottom:1px solid {pal["th_border"]};'
                f'font-family:-apple-system, BlinkMacSystemFont, \'Segoe UI\', Arial, sans-serif;">'
                f'{title}</caption>'
            )
            html_table = html_table.replace('<table', '<table', 1)
            html_table = html_table.replace('>', f'>{caption_html}', 1)

        # Click-to-sort behavior for the displayed table. The script is scoped
        # to this render's unique container id so several tables in one
        # notebook sort independently. Numeric columns sort numerically (the
        # '-' fill value always sinks to the bottom); everything else sorts as
        # text. Clicking a header toggles ascending/descending. The sort state
        # is shown purely via CSS classes (a fixed-width ::after arrow slot on
        # every header), so sorting never changes the header text or column
        # widths and the layout stays put.
        sort_script = """
        <script>
        (function() {
            var container = document.getElementById("__UID__");
            if (!container) return;
            var table = container.querySelector("table");
            if (!table) return;
            var tbody = table.querySelector("tbody");
            var headers = table.querySelectorAll("thead th");

            // Zebra striping is applied as classes over the *visible* rows so
            // it stays alternating after any combination of sort and filter
            // (CSS nth-child would keep counting hidden rows).
            function restripe() {
                var vis = 0;
                Array.prototype.forEach.call(tbody.querySelectorAll("tr"), function(tr) {
                    tr.classList.remove("uc-odd", "uc-even");
                    if (tr.style.display === "none") return;
                    tr.classList.add((vis % 2 === 0) ? "uc-odd" : "uc-even");
                    vis++;
                });
            }

            function reEsc(s) {
                return s.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\\\$&");
            }

            // Split on top-level separator chars only — separators inside
            // quotes or brackets don't count, so `in ['a', 'b']` lists and
            // quoted strings survive the OR-split on commas.
            function splitTop(s, seps) {
                var parts = [], cur = "", depth = 0, quote = null;
                for (var i = 0; i < s.length; i++) {
                    var ch = s.charAt(i);
                    if (quote) {
                        cur += ch;
                        if (ch === quote) quote = null;
                    } else if (ch === '"' || ch === "'") {
                        quote = ch; cur += ch;
                    } else if (ch === "[" || ch === "(") {
                        depth++; cur += ch;
                    } else if (ch === "]" || ch === ")") {
                        depth = Math.max(0, depth - 1); cur += ch;
                    } else if (depth === 0 && seps.indexOf(ch) !== -1) {
                        parts.push(cur); cur = "";
                    } else {
                        cur += ch;
                    }
                }
                parts.push(cur);
                return parts;
            }

            function unquote(s) {
                s = s.trim();
                if (s.length >= 2 && (s.charAt(0) === '"' || s.charAt(0) === "'") &&
                        s.charAt(s.length - 1) === s.charAt(0)) {
                    return { text: s.slice(1, -1), quoted: true };
                }
                return { text: s, quoted: false };
            }

            // Equality that compares numerically when both sides are
            // numbers, else as case-insensitive exact text.
            function eqMatch(value, text) {
                var nv = parseFloat(value), nt = parseFloat(text);
                if (!isNaN(nv) && !isNaN(nt) && /^-?[0-9.]+$/.test(value)) {
                    return nt === nv;
                }
                return text.toLowerCase() === value.toLowerCase();
            }

            // A single filter term. Accepts both the shorthand and
            // DataFrame.query-style forms:
            //   >10  >=10  <10  <=10  =10  ==10  !=10  5..20
            //   == "idle"   != 'idle'   "idle" (quoted = exact match)
            //   in ['idle', 'cruise']   not <term>
            // A redundant leading column name (as in `power > 5` typed in
            // the power column's box) is stripped when followed by an
            // operator, `in`, or `not`. Unquoted terms containing `*`/`?`
            // are fnmatch-style wildcards matched against the whole cell
            // (`alt*` = starts with, `*ft` = ends with). Anything else is
            // a case-insensitive substring match.
            function matchesTerm(term, text, label) {
                term = term.trim();
                if (!term) return true;
                if (label) {
                    term = term.replace(new RegExp(
                        "^" + reEsc(label) + "\\\\s*(?=[<>=!]|in\\\\b|not\\\\b)", "i"), "").trim();
                    if (!term) return true;
                }
                var m = term.match(/^not\\s+(.+)$/i);
                if (m) return !matchesTerm(m[1], text, null);
                m = term.match(/^in\\s*[\\[(]([^\\])]*)[\\])]$/i);
                if (m) {
                    return splitTop(m[1], ",").some(function(item) {
                        return eqMatch(unquote(item).text, text);
                    });
                }
                m = term.match(/^(==|!=|>=|<=|>|<|=)\\s*(.+)$/);
                if (m) {
                    var op = m[1];
                    var val = unquote(m[2]).text;
                    if (op === "=" || op === "==") return eqMatch(val, text);
                    if (op === "!=") return !eqMatch(val, text);
                    var num = parseFloat(text), v = parseFloat(val);
                    if (isNaN(num) || isNaN(v)) return false;
                    if (op === ">") return num > v;
                    if (op === ">=") return num >= v;
                    if (op === "<") return num < v;
                    return num <= v;
                }
                m = term.match(/^(-?[0-9.]+)\\s*\\.\\.\\s*(-?[0-9.]+)$/);
                if (m) {
                    var n = parseFloat(text);
                    if (isNaN(n)) return false;
                    return n >= parseFloat(m[1]) && n <= parseFloat(m[2]);
                }
                var uq = unquote(term);
                if (uq.quoted) return eqMatch(uq.text, text);
                if (term.indexOf("*") !== -1 || term.indexOf("?") !== -1) {
                    var wild = new RegExp(
                        "^" + reEsc(term).replace(/\\\\\\*/g, ".*")
                                         .replace(/\\\\\\?/g, ".") + "$", "i");
                    return wild.test(text);
                }
                return text.toLowerCase().indexOf(term.toLowerCase()) !== -1;
            }

            // A column's filter box can combine terms: "|" or "," separate
            // OR alternatives, and "&" joins AND terms within an
            // alternative, e.g. ">=5 & <20" or "idle, cruise". The
            // DataFrame.query keywords `and`/`or` (surrounded by spaces)
            // work as synonyms. Empty alternatives (a trailing comma while
            // typing) are ignored, so an all-empty filter matches
            // everything.
            function matches(filter, text, label) {
                if (!filter.trim()) return true;
                filter = filter.replace(/\\s+and\\s+/gi, " & ")
                               .replace(/\\s+or\\s+/gi, " | ");
                var anyGroup = false, matched = false;
                splitTop(filter, "|,").forEach(function(group) {
                    var terms = splitTop(group, "&").map(function(t) {
                        return t.trim();
                    }).filter(Boolean);
                    if (!terms.length) return;
                    anyGroup = true;
                    if (terms.every(function(t) { return matchesTerm(t, text, label); })) {
                        matched = true;
                    }
                });
                return anyGroup ? matched : true;
            }

            var filterInputs = [];
            var filterBtns = [];
            function applyFilters() {
                Array.prototype.forEach.call(tbody.querySelectorAll("tr"), function(tr) {
                    var show = filterInputs.every(function(inp, i) {
                        return matches(inp.value, tr.cells[i].textContent.trim(),
                                       headers[i].getAttribute("data-uc-label"));
                    });
                    tr.style.display = show ? "" : "none";
                });
                filterInputs.forEach(function(inp, i) {
                    var active = inp.value.trim() !== "";
                    filterBtns[i].classList.toggle("uc-filter-active", active);
                    headers[i].classList.toggle("uc-filtered", active);
                    filterBtns[i].title = active
                        ? "Filtered: " + inp.value.trim() + " (click to edit, Esc to clear)"
                        : "Filter this column";
                });
                restripe();
            }

            headers.forEach(function(th, colIdx) {
                th.addEventListener("click", function() {
                    // Header text is selectable: when the click ends a text
                    // selection inside this header, the user was copying the
                    // column name, not asking for a sort.
                    var sel = window.getSelection();
                    if (sel && !sel.isCollapsed && th.contains(sel.anchorNode)) return;
                    var asc = !th.classList.contains("uc-sort-asc");
                    headers.forEach(function(h) {
                        h.classList.remove("uc-sort-asc", "uc-sort-desc");
                    });
                    th.classList.add(asc ? "uc-sort-asc" : "uc-sort-desc");
                    var rows = Array.prototype.slice.call(tbody.querySelectorAll("tr"));
                    rows.sort(function(a, b) {
                        var av = a.cells[colIdx].textContent.trim();
                        var bv = b.cells[colIdx].textContent.trim();
                        if (av === "-" && bv === "-") return 0;
                        if (av === "-") return 1;
                        if (bv === "-") return -1;
                        var an = parseFloat(av), bn = parseFloat(bv);
                        var cmp;
                        if (!isNaN(an) && !isNaN(bn)) {
                            cmp = an - bn;
                        } else {
                            cmp = av.localeCompare(bv, undefined, {numeric: true});
                        }
                        return asc ? cmp : -cmp;
                    });
                    rows.forEach(function(r) { tbody.appendChild(r); });
                    restripe();
                });
            });

            // Filter row: one text input per column, inserted below the
            // headers. Built here (after the sort listeners are bound to the
            // static `headers` NodeList) so these cells never get sort
            // handlers. The row starts hidden; a small ⌕ toggle in each
            // header opens the filter box for just that column. Toggling a
            // box closed collapses it but KEEPS its filter applied — the
            // column stays clearly marked (accent ⌕ + accent underline on
            // the header, filter text in the ⌕ tooltip) so a collapsed
            // filter is never silently active. Esc clears and closes.
            var filterRow = document.createElement("tr");
            filterRow.className = "uc-filter-row";
            filterRow.style.display = "none";
            function closeFilter(i) {
                filterInputs[i].style.display = "none";
                if (filterInputs.every(function(inp) {
                    return inp.style.display === "none";
                })) {
                    filterRow.style.display = "none";
                }
            }
            headers.forEach(function(th, colIdx) {
                // Keep the pristine column label for the Copy payload, since
                // the toggle glyph below becomes part of th.textContent.
                th.setAttribute("data-uc-label", th.textContent.trim());

                var cell = document.createElement("th");
                var inp = document.createElement("input");
                inp.type = "text";
                inp.placeholder = "filter";
                inp.title = "Text matches as substring; wildcards: alt* (starts with), *ft (ends with), ? = any char; numeric: >10, >=10, <10, <=10, =10, 5..20. Combine: & (and), | or , (or). df.query style works too: power > 5 and power < 20, == \\"idle\\", != 3, in ['a', 'b'], not x";
                inp.style.display = "none";
                inp.addEventListener("input", applyFilters);
                inp.addEventListener("blur", function() {
                    if (inp.value.trim() === "") closeFilter(colIdx);
                });
                inp.addEventListener("keydown", function(e) {
                    if (e.key === "Escape") {
                        inp.value = "";
                        applyFilters();
                        closeFilter(colIdx);
                        inp.blur();
                    }
                });
                cell.appendChild(inp);
                filterRow.appendChild(cell);
                filterInputs.push(inp);

                var fbtn = document.createElement("span");
                fbtn.className = "uc-filter-btn";
                // Glyph comes from CSS ::before so the active state can swap
                // it (⌕ -> filled ●) without touching the DOM or the
                // header's copyable text.
                fbtn.title = "Filter this column";
                fbtn.addEventListener("click", function(e) {
                    e.stopPropagation();   // don't trigger the sort
                    if (inp.style.display === "none") {
                        filterRow.style.display = "";
                        inp.style.display = "";
                        inp.focus();
                    } else {
                        // Collapse only — the filter (if any) stays applied
                        // and the header stays marked.
                        closeFilter(colIdx);
                    }
                });
                th.appendChild(fbtn);
                filterBtns.push(fbtn);
            });
            table.querySelector("thead").appendChild(filterRow);
            restripe();

            // Copy button: puts the table on the clipboard as displayed —
            // current sort order, filtered-out rows skipped — as both TSV
            // (text/plain) and a clean HTML table
            // (text/html), so pasting into Excel/Sheets lands one value per
            // cell with the header row intact. Header text comes from
            // textContent, so the CSS-drawn sort arrows are never included.
            var btn = container.querySelector(".uc-copy-btn");
            if (btn) {
                btn.addEventListener("click", function() {
                    var data = [];
                    data.push(Array.prototype.map.call(headers, function(h) {
                        return h.getAttribute("data-uc-label") || h.textContent.trim();
                    }));
                    Array.prototype.forEach.call(tbody.querySelectorAll("tr"), function(tr) {
                        if (tr.style.display === "none") return;
                        data.push(Array.prototype.map.call(tr.cells, function(c) {
                            return c.textContent.trim();
                        }));
                    });
                    var tsv = data.map(function(r) { return r.join("\\t"); }).join("\\n");
                    var esc = function(s) {
                        return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                    };
                    var htmlTable = "<table>" + data.map(function(r, i) {
                        var tag = i === 0 ? "th" : "td";
                        return "<tr>" + r.map(function(c) {
                            return "<" + tag + ">" + esc(c) + "</" + tag + ">";
                        }).join("") + "</tr>";
                    }).join("") + "</table>";
                    var done = function() {
                        btn.textContent = "Copied \\u2713";
                        btn.classList.add("uc-copied");
                        setTimeout(function() {
                            btn.textContent = "Copy";
                            btn.classList.remove("uc-copied");
                        }, 1500);
                    };
                    if (navigator.clipboard && window.ClipboardItem) {
                        navigator.clipboard.write([new ClipboardItem({
                            "text/plain": new Blob([tsv], {type: "text/plain"}),
                            "text/html": new Blob([htmlTable], {type: "text/html"})
                        })]).then(done, function() {
                            navigator.clipboard.writeText(tsv).then(done);
                        });
                    } else if (navigator.clipboard) {
                        navigator.clipboard.writeText(tsv).then(done);
                    }
                });
            }
        })();
        </script>
        """.replace("__UID__", table_uid)

        # All CSS is scoped under this render's #id so it can't restyle other
        # tables in the notebook (or be restyled by a later chart.table()
        # call). Zebra striping keys off tbody position, so it stays
        # alternating after any sort.
        styled_html = f"""
        <div id="{table_uid}" style="margin-top:8px; margin-bottom:8px; overflow-x:auto;">
            <div class="uc-table-holder">
                <div class="uc-table-toolbar">
                    <button class="uc-copy-btn" type="button"
                            title="Copy table to clipboard (paste into Excel)">Copy</button>
                </div>
                <div class="uc-table-wrap">
                    {html_table}
                </div>
            </div>
        </div>
        <style>
        #{table_uid} .uc-table-holder {{
            display: inline-block;
        }}
        #{table_uid} .uc-table-toolbar {{
            text-align: left;
            margin-bottom: 4px;
        }}
        #{table_uid} .uc-copy-btn {{
            background-color: {pal['th_bg']};
            color: {pal['text']};
            border: 1px solid {pal['wrap_border']};
            border-radius: 6px;
            padding: 2px 10px;
            font-size: 12px;
            min-width: 80px;
            cursor: pointer;
            user-select: none;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
        }}
        #{table_uid} .uc-copy-btn:hover {{ background-color: {pal['th_hover']}; }}
        #{table_uid} .uc-copy-btn.uc-copied {{
            color: {pal['copied']};
            border-color: {pal['copied']};
        }}
        #{table_uid} .uc-table-wrap {{
            display: inline-block;
            border: 1px solid {pal['wrap_border']};
            border-radius: 8px;
            overflow: hidden;
            box-shadow: {pal['shadow']};
        }}
        #{table_uid} table {{
            border-collapse: collapse;
            width: auto;
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
        }}
        #{table_uid} th {{
            background-color: {pal['th_bg']};
            color: {pal['text']};
            font-weight: 600;
            font-size: {header_size}px;
            padding: 4px 10px;
            text-align: center;
            border: none;
            border-bottom: 2px solid {pal['th_border']};
            border-right: 1px solid {pal['cell_border']};
            cursor: pointer;
            white-space: nowrap;
        }}
        #{table_uid} th:last-child {{ border-right: none; }}
        #{table_uid} th::after {{
            content: "⇅";
            display: inline-block;
            width: 1em;
            margin-left: 6px;
            font-size: 0.7em;
            opacity: 0.35;
        }}
        #{table_uid} th.uc-sort-asc::after {{ content: "▲"; opacity: 0.9; }}
        #{table_uid} th.uc-sort-desc::after {{ content: "▼"; opacity: 0.9; }}
        #{table_uid} th:hover {{ background-color: {pal['th_hover']}; }}
        #{table_uid} th.uc-sort-asc,
        #{table_uid} th.uc-sort-desc {{ background-color: {pal['th_sorted']}; }}
        #{table_uid} td {{
            color: {pal['text']};
            font-size: {cell_size}px;
            padding: 3px 10px;
            text-align: center;
            border: none;
            border-bottom: 1px solid {pal['cell_border']};
            border-right: 1px solid {pal['cell_border']};
        }}
        #{table_uid} td:last-child {{ border-right: none; }}
        #{table_uid} tbody tr:last-child td {{ border-bottom: none; }}
        #{table_uid} th .uc-filter-btn {{
            display: inline-block;
            width: 1.5em;
            text-align: center;
            margin-left: 4px;
            font-size: 0.75em;
            line-height: 1;
            opacity: 0.35;
            cursor: pointer;
            user-select: none;
        }}
        #{table_uid} th .uc-filter-btn::before {{ content: "⌕"; }}
        #{table_uid} th .uc-filter-btn:hover {{ opacity: 0.85; }}
        #{table_uid} th .uc-filter-btn.uc-filter-active {{
            color: {pal['accent']};
            opacity: 1;
        }}
        #{table_uid} th .uc-filter-btn.uc-filter-active::before {{
            content: "●";
            font-size: 1.6em;
            vertical-align: middle;
        }}
        #{table_uid} th.uc-filtered {{
            border-bottom: 2px solid {pal['accent']};
        }}
        #{table_uid} tr.uc-filter-row th {{
            padding: 3px 6px;
            cursor: default;
            background-color: {pal['row_even']};
        }}
        #{table_uid} tr.uc-filter-row th::after {{ content: none; }}
        #{table_uid} tr.uc-filter-row input {{
            width: 100%;
            min-width: 4em;
            box-sizing: border-box;
            font-size: 12px;
            font-weight: 400;
            padding: 2px 6px;
            border: 1px solid {pal['wrap_border']};
            border-radius: 4px;
            background-color: {pal['row_odd']};
            color: {pal['text']};
            font-family: inherit;
        }}
        #{table_uid} tr.uc-filter-row input::placeholder {{ opacity: 0.45; }}
        #{table_uid} tr.uc-filter-row input:focus {{
            outline: none;
            border-color: {pal['accent']};
        }}
        /* nth-child striping is the no-JS fallback; once the script runs,
           the uc-odd/uc-even classes (set over visible rows only) take over
           via these later, equal-specificity rules. */
        #{table_uid} tbody tr:nth-child(odd) td {{ background-color: {pal['row_odd']}; }}
        #{table_uid} tbody tr:nth-child(even) td {{ background-color: {pal['row_even']}; }}
        #{table_uid} tbody tr.uc-odd td {{ background-color: {pal['row_odd']}; }}
        #{table_uid} tbody tr.uc-even td {{ background-color: {pal['row_even']}; }}
        #{table_uid} tbody tr:hover td {{ background-color: {pal['row_hover']}; }}
        </style>
        {sort_script}
        """

        display(HTML(styled_html))

    def save_png(self, filename="plot.png", scale=3, width=None, height=None,
                 embed_session='all', parms=None):
        """
        Save the last generated plot to a PNG file.
        Requires the 'kaleido' package to be installed.

        The image always shows the whole legend: where the on-screen legend
        would scroll, the image is made taller instead (the plot area keeps its
        size). Passing ``height`` overrides that and can clip the legend again.

        By default the image also carries everything needed to remake the
        plot: the full plotting session (every set's rows, queries and
        formatting, the notebook-level formatting) plus the plotting call
        that produced the figure, stored as a PNG text chunk. Image viewers
        ignore it; ``nb.load_session('plot.png')`` (or
        ``UnichartNotebook.from_session('plot.png')``) reads it back and
        replots. :func:`read_png_session` shows what an image contains.

        Parameters
        ----------
        embed_session : 'all' | True | False
            ``'all'`` (default): embed every set's rows so the image is fully
            self-contained. ``True``: like :meth:`save_session` — sets loaded
            from a file that still exists are stored as a file reference
            (smaller image, but replotting needs the file). ``False``: plain
            image, no metadata. Ignored when ``filename`` isn't a PNG.
        parms : str | list of str, optional
            Whitelist of columns to embed, as in :meth:`save_session`: only
            these columns are stored for embedded sets (the plotted columns
            and any a set's query/hue/style needs are always kept), so a wide
            table doesn't bloat the image. Default: every column.
        """
        if self.last_fig is None:
            print("No plot to save. Please run .plot() first.")
            return

        try:
            self._suppress_mathjax()

            self._full_legend_figure(self.last_fig).write_image(
                filename, scale=scale, width=width, height=height)
            note = ''
            if embed_session:
                note = self._embed_png_session(filename, embed_session,
                                               dict(scale=scale, width=width, height=height),
                                               parms=parms)
            print(f"Plot saved to {filename}{note}")

        except ValueError as e:
            print(f"Error saving image (ensure 'kaleido' is installed): {e}")
        except Exception as e:
            print(f"Error saving image: {e}")

    def _embed_png_session(self, filename, embed_data, image_kwargs, parms=None):
        """Write the current session + last plot call into the PNG at
        ``filename`` (see :func:`png_embed_text`). Returns a note for the
        save message, empty if the file isn't a PNG."""
        with open(filename, 'rb') as fh:
            data = fh.read()
        if not data.startswith(_PNG_SIGNATURE):
            return ''
        session, stats = self._build_session(embed_data, Path(filename),
                                             plot_call=self._last_plot_call, parms=parms)
        session['image'] = {k: v for k, v in image_kwargs.items() if v is not None}
        text = json.dumps(session, separators=(',', ':'),
                          default=self._session_json_default)
        with open(filename, 'wb') as fh:
            fh.write(png_embed_text(data, _PNG_SESSION_KEYWORD, text))
        return (" (session embedded: " + self._session_summary(stats)
                + self._plot_call_note() + ")")

    def list_sets(self, search=None, output=None):
        """
        Display a table of the loaded datasets (styled HTML: sortable,
        filterable, copyable — same look as :meth:`table`).

        ``search`` doubles as a set selector: pass a dataset selector — an
        int, a ``slice``/``range``, range shorthand such as ``'1:4'`` or
        ``'0,3,7:'``, a :class:`Dataset`, ``'all'``, or a list of those — and
        only those sets are listed. Pass any *other* string and it is taken
        as a parameter search instead, listing the parameters matching it
        across all loaded sets (delegates to :meth:`list_parms`).

        The int/string split follows the rest of the library (see
        :meth:`_get_uset_slice`): titles are never selectors, and a bare
        numeric string stays a parameter name — so ``list_sets(2)`` shows set
        2 while ``list_sets('2')`` searches parameters for '2'.

        Parameters
        ----------
        search : int, str, slice, range, Dataset or list, optional
            A dataset selector narrows the table to those sets; any other
            string searches the parameters of all loaded sets instead (on
            that path ``output`` does not apply — :meth:`list_parms` returns
            its parameter names).
        output : {None, 'df', 'md', 'fig'}, optional
            What to return, following :meth:`table` and :meth:`summary`:

            - ``None`` (default): display the styled HTML table and return
              ``None``.
            - ``'df'``: return the listed sets as a :class:`pandas.DataFrame`
              with raw, unformatted values — ``Selected`` is a real ``bool``
              and the shape is split into integer ``Rows`` / ``Cols`` columns
              (the display path renders these as ``✓``/``✗`` and
              ``"rows x cols"``). Nothing is displayed.
            - ``'md'``: return a GitHub-flavored Markdown string.
            - ``'fig'``: return the styled Plotly ``go.Figure`` (a
              ``go.Table``), with dark-mode already applied.

        Examples
        --------
        Work with the set table as data, or narrow it to a few sets::

            df = chart.list_sets(output='df')
            df[df['Selected']]['Title'].tolist()

            chart.list_sets('0,3:5')        # sets 0, 3 and 4
            chart.list_sets(-1, output='df')
        """
        if output not in (None, 'df', 'md', 'fig'):
            print("output must be one of None, 'df', 'md' or 'fig'.")
            return

        # A selector narrows the sets listed; any other string means the
        # caller wants a parameter search (_var_targets draws the same line
        # the formatting setters do).
        title = "Loaded Datasets"
        if search is None:
            target_sets = list(self.sets)
        elif self._var_targets(search) is not None:
            return self.list_parms(set_number='all', search_string=search)
        else:
            target_sets = self._get_uset_slice(search)
            if target_sets and len(target_sets) < len(self.sets):
                # Name the sets actually listed, not the selector's spelling —
                # a list of Dataset objects would repr badly.
                title = ("Loaded Datasets: set"
                         f"{'' if len(target_sets) == 1 else 's'} "
                         f"{_compact_indices(ds.index for ds in target_sets)}")

        columns = ["Set", "Title", "Selected", "Rows", "Cols", "Query"]

        if not target_sets:
            if output is None:
                print("No datasets loaded." if not self.sets else
                      f"No datasets match selector {search!r}.")
                return
            empty = pd.DataFrame(columns=columns)
            empty = empty.astype({"Selected": bool, "Rows": int, "Cols": int})
            if output == 'df':
                return empty
            return self._list_sets_render(empty, output, title)

        rows = []
        for ds in target_sets:
            # Shape from cached row positions and own-column count — no
            # full-width materialization.
            rows.append([ds.index, ds.title, bool(ds.select),
                         len(ds._masked_positions()), len(ds._own_col_positions()),
                         ds.query])

        df = pd.DataFrame(rows, columns=columns)
        # Keep ``Query`` as plain objects so an unfiltered set stays ``None``
        # rather than being coerced to NaN by the string dtype.
        df["Query"] = pd.Series([r[-1] for r in rows], dtype=object)

        if output == 'df':
            return df

        return self._list_sets_render(df, output, title)

    def _list_sets_render(self, df, output, title="Loaded Datasets"):
        """
        Turn the raw frame built by :meth:`list_sets` into its display form
        (``✓``/``✗``, ``"rows x cols"``) and emit it per ``output``.
        """
        display_df = pd.DataFrame({
            "Set": df["Set"],
            "Title": df["Title"],
            "Selected": df["Selected"].map(lambda s: "✓" if s else "✗"),
            "Shape": [f"{r} x {c}" for r, c in zip(df["Rows"], df["Cols"])],
            "Query": df["Query"].map(str),
        })

        if output == 'md':
            try:
                return display_df.to_markdown(index=False)
            except ImportError:
                print("Markdown output requires the 'tabulate' package "
                      "(pip install tabulate).")
                return

        if output == 'fig':
            return self._build_table_figure(display_df, title=title)

        self._display_html_table(display_df, title=title)

    def list_parms(self, set_number=None, search_string=None, use_regex=False):
        """
        List the parameters (columns) available in the loaded datasets.

        With more than one dataset in scope, each parameter is annotated with
        the sets that actually own it ('in all sets' when every set in scope
        does), so a parameter calculated in only some sets is visible at a
        glance. Displays as the styled HTML table (sortable, filterable,
        copyable — same look as :meth:`table`); the filter boxes are handy
        for narrowing long parameter lists. Returns the parameter names, as
        before.

        A bare string as the first argument is taken as the search
        substring (set selectors are ints/'all'/Datasets, never names), so
        ``list_parms('egt')`` searches all sets in scope for 'egt'.
        """
        import fnmatch

        if (isinstance(set_number, str) and set_number != 'all'
                and search_string is None):
            set_number, search_string = None, set_number

        if set_number is None:
            target_sets = self.selected()
            if not target_sets:
                target_sets = self.sets
        else:
            target_sets = self._get_uset_slice(set_number)

        if not target_sets:
            print("No datasets available to list parameters from.")
            return []

        # ds.columns is the cheap ownership-aware view: no rows materialized,
        # and all-NaN phantom columns introduced by other sets are excluded.
        owners = {}
        for ds in target_sets:
            for col in ds.columns:
                owners.setdefault(col, []).append(ds.index)
        all_cols = set(owners)

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

        title = f"Found {len(filtered_cols)} parameters"
        if search_string:
            title += f" matching '{search_string}'"
        if set_number == 'all':
            title += " across all sets"
        elif set_number is not None:
            title += f" in set(s) {set_number}"
        else:
            title += " in active datasets"

        # The ownership column only earns its space when sets can differ.
        tags = {}
        if len(target_sets) > 1:
            n_scope = len(target_sets)
            tags = {c: ("in all sets" if len(owners[c]) == n_scope
                        else f"in set{'' if len(owners[c]) == 1 else 's'} "
                             f"{_compact_indices(owners[c])}")
                    for c in filtered_cols}

        rows = []
        for col in filtered_cols:
            desc = self.parm_description_dict.get(col, "No description available.")
            if tags:
                rows.append([str(col), tags[col], desc])
            else:
                rows.append([str(col), desc])

        columns = (["Parameter", "Sets", "Description"] if tags
                   else ["Parameter", "Description"])
        self._display_html_table(pd.DataFrame(rows, columns=columns), title=title)

        return filtered_cols

    def summary(self, cols=None, title=None, sig_figs=None, decimals=None,
                output=None, print_table=None):
        """
        Summarize the given columns across the currently selected datasets.

        Reports count / min / mean / max / std per dataset and column. Like
        :meth:`table`, this displays the styled HTML table by default and offers
        the same alternative output modes.

        Parameters
        ----------
        cols : str or list of str, optional
            Column(s) to summarize. Defaults to the columns from the last plot
            (``self.last_x`` + ``self.last_y``).
        title : str, optional
            Table title. Defaults to ``"Statistical Summary: <cols>"``.
        sig_figs : int, optional
            Round every statistic to this many significant figures for display,
            keeping ordinary decimal notation (no scientific notation). Affects
            the rendered HTML table, the Markdown output and the ``'fig'``
            table only; the ``output='df'`` DataFrame keeps its full-precision
            numeric values. Without it (or ``decimals``), each set's own
            ``sig_figs`` / ``decimals`` (see :meth:`sig_figs`,
            :meth:`decimals`) formats its rows, and a set carrying neither
            displays its statistics as ``.4g``. Mutually exclusive with
            ``decimals``.
        decimals : int, optional
            The fixed-decimal-places alternative to ``sig_figs``: round every
            statistic to this many places after the point, keeping trailing
            zeros in the rendered table. Affects display only, exactly as
            ``sig_figs`` does. Mutually exclusive with ``sig_figs``.
        output : {None, 'df', 'md', 'fig'}, optional
            What to return:

            - ``None`` (default): render and display the styled HTML table —
              sortable, filterable and copyable, exactly as :meth:`table` does
              (see that method for the full list of interactions). Returns
              ``None``.
            - ``'df'``: return the summary :class:`pandas.DataFrame` with full
              numeric precision.
            - ``'md'``: return a GitHub-flavored Markdown string.
            - ``'fig'``: return the styled Plotly ``go.Figure`` (a ``go.Table``),
              with ``sig_figs``/``decimals`` and dark-mode already applied.
              Useful for embedding the summary alongside other figures (e.g. in
              a dashboard panel) without triggering the HTML display side
              effect.
        print_table : bool, optional
            Backwards-compatible switch from the older signature, where
            ``summary()`` always returned the DataFrame and only displayed the
            table when asked. When given (and ``output`` is not), the
            DataFrame is still returned: ``True`` also displays the styled HTML
            table, ``False`` displays nothing. Prefer ``output=`` in new code.

        Examples
        --------
        Display statistics for the columns from the last plot::

            chart.summary()

        Summarize specific columns to 3 significant figures::

            chart.summary(cols=['speed', 'power'], sig_figs=3)

        Or to two decimal places::

            chart.summary(cols=['speed', 'power'], decimals=2)

        Get the numbers back instead of displaying them::

            df = chart.summary(cols='power', output='df')
        """
        if output is not None and output not in ('df', 'md', 'fig'):
            print(f"Unknown output mode '{output}'. Use None, 'df', 'md', or 'fig'.")
            return
        if sig_figs is not None and (not isinstance(sig_figs, int) or
                                     isinstance(sig_figs, bool) or sig_figs < 1):
            print("sig_figs must be a positive integer.")
            return
        if decimals is not None and (not isinstance(decimals, int) or
                                     isinstance(decimals, bool) or decimals < 0):
            print("decimals must be a non-negative integer.")
            return
        if sig_figs is not None and decimals is not None:
            print("Pass either sig_figs or decimals, not both.")
            return

        # ``output`` drives the new behavior; ``print_table`` keeps the old
        # "always return the DataFrame" contract alive when it is passed.
        if output is None:
            show = True if print_table is None else bool(print_table)
            return_df = print_table is not None
        else:
            show = False
            return_df = output == 'df'

        headers = ["Set", "Title", "Query", "Variable", "Count", "Min", "Mean", "Max", "Std"]
        empty = pd.DataFrame(columns=headers)

        if cols is None:
            y_part = self.last_y if isinstance(self.last_y, list) else [self.last_y] if self.last_y else []
            x_part = [self.last_x] if self.last_x else []
            target_cols = x_part + y_part
        else:
            target_cols = cols if isinstance(cols, list) else [cols]

        if not target_cols:
            if show:
                print("No columns specified and no previous plot variables defined.")
            return empty if return_df else None

        active_ds = self.selected()
        if not active_ds:
            if show:
                print("No datasets selected. Cannot generate summary.")
            return empty if return_df else None

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

        if df.empty:
            if show:
                print(f"None of the selected datasets contain the specified columns: {target_cols}")
            return df if return_df else None

        if output == 'df':
            return df

        # Build the rendered frame the same way :meth:`table` does: NaN becomes
        # '-', then sig_figs / decimals (or the default .4g) formats the
        # statistics.
        stat_cols = ["Min", "Mean", "Max", "Std"]
        final_df = df.copy()
        final_df["Count"] = final_df["Count"].astype(int)
        final_df = final_df.fillna('-')
        if sig_figs is not None:
            stat_fmt = lambda v, f: self._sig_fig_str(v, sig_figs)
        elif decimals is not None:
            stat_fmt = lambda v, f: self._decimals_str(v, decimals)
        else:
            # No explicit argument: each set's own sig_figs / decimals (see
            # :meth:`sig_figs`, :meth:`decimals`), falling back to .4g.
            stat_fmt = lambda v, f: (f(v) if f else
                                     (f"{v:.4g}" if isinstance(v, float) else v))
        set_fmt = {ds.index: self._set_display_fmt(ds.sig_figs, ds.decimals)
                   for ds in self.sets}
        row_fmt = [set_fmt.get(i) for i in final_df["Set"]]
        for c in stat_cols:
            final_df[c] = [stat_fmt(v, f) for v, f in zip(final_df[c], row_fmt)]

        if output == 'md':
            try:
                return final_df.to_markdown(index=False)
            except ImportError:
                print("Markdown output requires the 'tabulate' package "
                      "(pip install tabulate).")
                return

        table_title = title or f"Statistical Summary: {', '.join(map(str, target_cols))}"

        if output == 'fig':
            # Only built on demand: unlike :meth:`table`, the display path must
            # not overwrite ``self.last_fig`` (which save_png and the dashboard
            # read back), since a summary is informational rather than a plot.
            return self._build_table_figure(final_df, title=table_title)

        if show:
            self._display_html_table(final_df, title=table_title)

        return df if return_df else None
        return df

    # Method groupings for help(). A method left out of every list still shows,
    # under "Other" (help() fills that bucket by set-difference), so a newly
    # added method is never silently hidden; names here that no longer exist are
    # simply skipped.
    _HELP_CATEGORIES = [
        ("Loading & data",   ['load', 'load_df', 'load_clipboard', 'combine_sets',
                              'combine', 'add_column', 'set_column',
                              'save_session', 'load_session']),
        ("Selection",        ['select', 'selected', 'omit', 'query', 'restore',
                              'clear_data']),
        ("Plotting",         ['plot', 'plot_ymult', 'plot_marginal', 'plot_type',
                              'bar', 'box', 'contour', 'histogram', 'line', 'highlight',
                              'save_png', 'dashboard']),
        ("Styling & format", ['color', 'marker', 'markersize', 'alpha',
                              'alpha_marker', 'alpha_line', 'fill',
                              'linestyle', 'linewidth', 'edgewidth', 'hue',
                              'hue_palette', 'reg_order', 'copy_format',
                              'sig_figs', 'decimals',
                              'set_color_palette',
                              'var_format', 'clear_var_format', 'list_var_formats',
                              'set_display_parms', 'set_title', 'set_default_format',
                              'reset_format', 'set_font_sizes', 'get_font_sizes',
                              'toggle_darkmode', 'set_plot_style', 'scale',
                              'set_plot_size', 'grid', 'watermark',
                              'set_static_images', 'set_copy_buttons']),
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
        # The preview is one docstring line, so its ``literals`` get the same
        # treatment as the full docstring in help('<name>').
        preview = _DOC_LITERAL_RE.sub(
            lambda m: _hc(m.group(0)[2:-2], 'lit'), preview)
        print(f"  • {_hc(name, 'name')}{_hc(self._help_sig(name), 'sig')}")
        print(f"      → {preview}")

    @staticmethod
    def _help_attr_line(name, val):
        s = str(val)
        if len(s) > 100:
            s = s[:100] + "..."
        return f"  • {_hc(name, 'name')}: {type(val).__name__} = {s}"

    def _help_topic(self, key):
        """Detailed help for a single method or a category (see :meth:`help`)."""
        cls = self.__class__
        func = getattr(cls, key, None)
        if callable(func) and not key.startswith('_'):
            print("=" * 70)
            print(f"📖 {_hc(key, 'name')}{_hc(self._help_sig(key), 'sig')}")
            print("=" * 70)
            print(_color_docstring(inspect.getdoc(func)
                                  or "No description available."))
            return
        for label, names in self._HELP_CATEGORIES:
            if key.lower() == label.lower():
                print("📂 " + _hc(label, 'head'))
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

        Notes
        -----
        Headings, category labels, method names and signatures are colored with
        ANSI wherever that renders: a terminal, a Jupyter kernel, and the
        explorer's web terminal (which turns the codes into styled spans). A
        method's docstring is painted too — its section headings, parameter
        names and types, and its ``literals``, whose backticks the color
        replaces.
        Output that is piped or redirected stays plain. ``NO_COLOR=1`` turns it
        off; ``unichart._HELP_COLOR = True/False`` forces it either way.
        """
        if topic is not None:
            self._help_topic(str(topic).strip())
            return

        print("=" * 70)
        print(_hc("📚 UnichartNotebook HELP", 'head'))
        print("=" * 70)

        cls = self.__class__
        doc = inspect.getdoc(cls)
        if doc:
            print("\n" + _hc("📋 CLASS DESCRIPTION:", 'head'))
            print(_color_docstring(doc))

        # Methods, grouped. Anything not mapped falls into "Other" via
        # set-difference so nothing is ever hidden.
        public = {n: f for n, f in inspect.getmembers(cls, inspect.isfunction)
                  if not n.startswith('_')}
        print("\n" + _hc("🔍 PUBLIC METHODS  (call nb.help('name') for "
                          "full details):", 'head'))
        print("-" * 70)
        shown = set()
        for label, names in self._HELP_CATEGORIES:
            entries = [n for n in names if n in public]
            if not entries:
                continue
            print("\n" + _hc(label, 'label'))
            for n in entries:
                self._help_method_line(n, public[n])
                shown.add(n)
        leftover = sorted(set(public) - shown)
        if leftover:
            print("\n" + _hc("Other", 'label'))
            for n in leftover:
                self._help_method_line(n, public[n])

        # Attributes, split into user-facing config vs internal plot memory
        # (the volatile ``last_*`` cache). The rule is prefix-based rather than a
        # fixed list, so a new attribute defaults to Config and stays visible.
        print("\n" + _hc("🛠️  ATTRIBUTES:", 'head'))
        print("-" * 70)
        attrs = {a: v for a, v in self.__dict__.items() if not a.startswith('_')}
        if not attrs:
            print("No public instance attributes found.")
        else:
            config = {a: v for a, v in attrs.items() if not a.startswith('last_')}
            state = {a: v for a, v in attrs.items() if a.startswith('last_')}
            if config:
                print("\n" + _hc("Config", 'label'))
                for name in sorted(config):
                    print(self._help_attr_line(name, config[name]))
            if state:
                print("\n" + _hc("State (last-plot memory)", 'label'))
                for name in sorted(state):
                    print(self._help_attr_line(name, state[name]))

        print("\n" + _hc("💡 QUICK START:", 'head'))
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

    def _line_label(self, line_spec, orientation):
        """Annotation kwargs for a reference line's label, or None if unlabeled.
        Falls back to the axes-tick font size so labels track the plot's scale."""
        return _line_label_annotation(line_spec, orientation,
                                      getattr(self, 'axes_tick_size', None))

    def _apply_decorations(self, fig, x_vars, y_vars, mode, calc_ncols, plot_items=None,
                           highlight_layer='below', refs=None):
        """
        Apply stored lines and highlights to a figure.

        ``refs`` optionally gives each ``plot_items`` entry its own
        ``(xref, yref)`` pair, for figures whose panels don't sit on a regular
        ``calc_ncols`` grid (``plot_marginal``'s main panel plus its two
        marginals). Without it, an item's position is computed from its index.

        ``highlight_layer`` controls whether highlight rectangles are drawn below
        or above the traces. It defaults to ``'below'`` so data marks stay on top
        (correct for scatter/line/bar/box/histogram). Contour plots pass
        ``'above'`` because their opaque filled field would otherwise hide a
        highlight; the highlight's ``alpha`` keeps the contours visible through it.
        """
        x_list = x_vars if isinstance(x_vars, list) else ([x_vars] if x_vars else [])
        y_list = y_vars if isinstance(y_vars, list) else ([y_vars] if y_vars else [])

        def _item_refs(idx):
            if refs is not None:
                return refs[idx]
            r, c = (idx // calc_ncols) + 1, (idx % calc_ncols) + 1
            return _subplot_refs(r, c, calc_ncols)

        for col_name, col_lines in self.lines.items():
            if col_name in x_list:
                if mode == 'vars' and plot_items:
                    for idx, (xi, yi) in enumerate(plot_items):
                        if xi == col_name:
                            xref, yref = _item_refs(idx)
                            for l in col_lines:
                                fig.add_shape(
                                    type='line', x0=l['level'], x1=l['level'], y0=0, y1=1,
                                    xref=xref, yref=f'{yref} domain',
                                    line=dict(color=l['color'], dash=l['dash'] or 'solid')
                                )
                                ann = self._line_label(l, 'vertical')
                                if ann:
                                    fig.add_annotation(xref=xref, yref=f'{yref} domain', **ann)
                else:
                    for l in col_lines:
                        fig.add_vline(x=l['level'], line_dash=l['dash'] or 'solid', line_color=l['color'],
                                      **_prefix_annotation(self._line_label(l, 'vertical')))

            if col_name in y_list:
                if mode == 'vars' and plot_items:
                    for idx, (xi, yi) in enumerate(plot_items):
                        if yi == col_name:
                            xref, yref = _item_refs(idx)
                            for l in col_lines:
                                fig.add_shape(
                                    type='line', x0=0, x1=1, y0=l['level'], y1=l['level'],
                                    xref=f'{xref} domain', yref=yref,
                                    line=dict(color=l['color'], dash=l['dash'] or 'solid')
                                )
                                ann = self._line_label(l, 'horizontal')
                                if ann:
                                    fig.add_annotation(xref=f'{xref} domain', yref=yref, **ann)
                else:
                    for l in col_lines:
                        fig.add_hline(y=l['level'], line_dash=l['dash'] or 'solid', line_color=l['color'],
                                      **_prefix_annotation(self._line_label(l, 'horizontal')))

        for col_name, hls in self.highlights.items():
            if col_name in x_list:
                if mode == 'vars' and plot_items:
                    for idx, (xi, yi) in enumerate(plot_items):
                        if xi == col_name:
                            xref, yref = _item_refs(idx)
                            for h in hls:
                                fig.add_shape(
                                    type='rect', x0=h['range'][0], x1=h['range'][1], y0=0, y1=1,
                                    xref=xref, yref=f'{yref} domain',
                                    fillcolor=h['color'], opacity=h['alpha'], layer=highlight_layer, line_width=0
                                )
                else:
                    for h in hls:
                        fig.add_vrect(x0=h['range'][0], x1=h['range'][1], fillcolor=h['color'],
                                      opacity=h['alpha'], layer=highlight_layer, line_width=0)

            if col_name in y_list:
                if mode == 'vars' and plot_items:
                    for idx, (xi, yi) in enumerate(plot_items):
                        if yi == col_name:
                            xref, yref = _item_refs(idx)
                            for h in hls:
                                fig.add_shape(
                                    type='rect', x0=0, x1=1, y0=h['range'][0], y1=h['range'][1],
                                    xref=f'{xref} domain', yref=yref,
                                    fillcolor=h['color'], opacity=h['alpha'], layer=highlight_layer, line_width=0
                                )
                else:
                    for h in hls:
                        fig.add_hrect(y0=h['range'][0], y1=h['range'][1], fillcolor=h['color'],
                                      opacity=h['alpha'], layer=highlight_layer, line_width=0)

        return fig

# ======================================================================
# new_uc — one-call notebook factory
# ======================================================================

class _Auto:
    """Sentinel for a ``new_uc`` knob left alone: the value is decided by the
    plot style (``markersize``, ``hue_palette``, ``color_map``, ``marker_map``)
    or differs per plot method (``alpha``, ``barmode``), so there is no single
    literal that could sit in the signature without being wrong somewhere.
    Renders as ``auto`` in ``help(new_uc)``."""
    __slots__ = ()
    def __repr__(self):
        return 'auto'


AUTO = _Auto()


def new_uc(
    # ---- overall look ------------------------------------------------
    plot_style='matplotlib',
    darkmode=False,
    color_map=AUTO,
    marker_map=AUTO,
    # ---- per-dataset styles (applied to datasets as they load) -------
    # Sourced from _DATASET_FORMAT_DEFAULTS rather than retyped, so these can
    # never drift out of sync with the constructor. Signature defaults are
    # evaluated at def time, so help(new_uc) still shows the literal values.
    marker='map',                                      # _MARKER_BY_INDEX, as the API spells it
    markersize=AUTO,
    linestyle=_DATASET_FORMAT_DEFAULTS['linestyle'],
    linewidth=_DATASET_FORMAT_DEFAULTS['linewidth'],
    edgewidth=_DATASET_FORMAT_DEFAULTS['edgewidth'],
    edge_color=_DATASET_FORMAT_DEFAULTS['edge_color'],
    alpha=AUTO,
    alpha_marker=_DATASET_FORMAT_DEFAULTS['alpha_marker'],
    alpha_line=_DATASET_FORMAT_DEFAULTS['alpha_line'],
    fill=_DATASET_FORMAT_DEFAULTS['fill'],
    hue_palette=AUTO,
    sig_figs=_DATASET_FORMAT_DEFAULTS['sig_figs'],
    decimals=_DATASET_FORMAT_DEFAULTS['decimals'],
    # ---- figure / per-call plot defaults -----------------------------
    figsize=(12, 8),
    legend='above',
    suppress_legends=False,
    legend_scroll=True,
    ncols=None,
    nrows=None,
    hspace=None,
    vspace=None,
    barmode=AUTO,
    agg='mean',
    histfunc='sum',
    histnorm='',
    boxmode='group',
    points='outliers',
    # ---- fonts, decorations, plot area -------------------------------
    font_sizes=None,
    suptitle=None,
    footer=None,
    plot_size=None,
    plot_size_per_subplot=True,
    # ---- output ------------------------------------------------------
    static_images=False,
    static_scale=2,
    copy_buttons=True,
    # ---- data --------------------------------------------------------
    data=None,
    **load_kwargs,
):
    """Return a configured :class:`UnichartNotebook` in one call.

    Every keyword is a *preset*: its default is unichart's current built-in, so
    ``new_uc()`` is equivalent to ``UnichartNotebook()`` and the signature
    doubles as the list of what those built-ins are. Override any of them to
    start a notebook already styled the way you want, instead of following the
    constructor with a run of ``set_*`` calls::

        nb = new_uc()                                  # today's defaults
        nb = new_uc(plot_style='plotly', darkmode=True) # dark plotly look
        nb = new_uc(markersize=5, linewidth=1, figsize=(10, 6), legend='right')
        nb = new_uc(data=df, set_name_column='ENGINE')  # styled, then loaded

    Settings are applied in dependency order — style first (it installs its own
    palette and format defaults), then palettes, then the format and per-call
    defaults, and ``data`` last, so loaded datasets pick up the finished look.

    Parameters
    ----------
    plot_style : 'matplotlib' | 'plotly'
        Overall look. Installs the style's ``color_map``, its ``markersize`` /
        ``hue_palette``, and its font-size fallbacks — which is why those knobs
        default to ``auto`` rather than a literal.
    darkmode : bool
        Dark or light variant of the style.
    color_map, marker_map : list
        Per-index color / marker sequences. ``auto`` keeps the style's own
        (matplotlib: the mpl color cycle; plotly: ``px.colors.qualitative.Plotly``).
    marker, markersize, linestyle, linewidth, edgewidth, edge_color, alpha,
    alpha_marker, alpha_line, fill, hue_palette
        Per-dataset styles, exactly as :meth:`UnichartNotebook.set_default_format`
        takes them. ``markersize`` and ``hue_palette`` are style-owned
        (matplotlib: 8.3 / 'Viridis'; plotly: 10 / 'Jet'). ``alpha`` is ``auto``
        because it is double-booked: it sets the per-dataset opacity (built-in 1)
        *and* the histogram / marginal-strip opacity (built-in 0.7), so passing a
        value changes both.
    figsize, legend, suppress_legends, legend_scroll, ncols, nrows, hspace,
    vspace, barmode, agg, histfunc, histnorm, boxmode, points
        Figure and per-call plot defaults, as in ``set_default_format``. An
        explicit argument to a plot method still wins over any of them.
        ``barmode`` is ``auto`` because its built-in differs per method
        (``bar`` 'group', ``histogram`` 'overlay'); setting it pins both.
    font_sizes : dict, optional
        Forwarded to :meth:`UnichartNotebook.set_font_sizes`, e.g.
        ``{'all': 16, 'suptitle': 24}``. ``None`` keeps the style's fallbacks.
    suptitle, footer : str, optional
        Standing figure title / footnote (the ``suptitle`` and ``footer``
        attributes), used whenever a plot call doesn't pass its own.
    plot_size : (width, height), optional
        Pin the plot area, in inches, via :meth:`set_plot_size`. Either element
        may be ``None`` to leave that dimension free. Distinct from ``figsize``,
        which sizes the whole figure.
    plot_size_per_subplot : bool
        Whether ``plot_size`` sizes one panel (default) or the whole grid.
    static_images : bool
        Render plots as flat PNGs instead of interactive figures (needs kaleido).
    static_scale : float
        Resolution multiplier for those PNGs.
    copy_buttons : bool
        Show the copy-to-clipboard button on interactive plots.
    data : DataFrame, optional
        Loaded with ``nb.load_df(data, **load_kwargs)`` *after* everything above,
        so the datasets are styled by the presets.
    **load_kwargs
        Remaining keywords go to ``load_df`` (``set_name_column=``,
        ``set_idx_column=``, ...). Passing any without ``data`` is an error,
        which is also how a misspelled preset name gets caught.

    Returns
    -------
    UnichartNotebook

    See Also
    --------
    uc_defaults : the same presets as a plain dict.
    UnichartNotebook.set_default_format : change these on an existing notebook.
    """
    if load_kwargs and data is None:
        raise TypeError(
            f"new_uc() got unexpected keyword argument(s) {sorted(load_kwargs)}. "
            f"Load arguments are only accepted alongside data=; for a preset, "
            f"check the spelling against uc_defaults().")

    nb = UnichartNotebook()

    # Style first: _apply_style_defaults overwrites color_map and the
    # style-owned default_format entries, so anything set before it is lost.
    if plot_style != nb.plot_style:
        nb.set_plot_style(plot_style)
    if bool(darkmode) != nb.darkmode:
        nb.toggle_darkmode(bool(darkmode))

    if color_map is not AUTO:
        nb.color_map = list(color_map)
    if marker_map is not AUTO:
        nb.marker_map = list(marker_map)

    # One set_default_format call so a bad value fails before anything else is
    # touched. AUTO knobs are dropped; the rest pass None through harmlessly
    # (set_default_format reads None as "leave unchanged", and for these that
    # is also the built-in).
    fmt = {
        'marker': marker, 'markersize': markersize, 'linestyle': linestyle,
        'linewidth': linewidth, 'edgewidth': edgewidth, 'edge_color': edge_color,
        'alpha': alpha, 'alpha_marker': alpha_marker, 'alpha_line': alpha_line,
        'fill': fill, 'hue_palette': hue_palette, 'sig_figs': sig_figs,
        'decimals': decimals,
        'figsize': figsize, 'legend': legend, 'suppress_legends': suppress_legends,
        'legend_scroll': legend_scroll, 'ncols': ncols, 'nrows': nrows,
        'hspace': hspace, 'vspace': vspace, 'barmode': barmode, 'agg': agg,
        'histfunc': histfunc, 'histnorm': histnorm, 'boxmode': boxmode,
        'points': points,
    }
    nb.set_default_format(**{k: v for k, v in fmt.items() if v is not AUTO})

    if font_sizes:
        nb.set_font_sizes(**font_sizes)
    if suptitle is not None:
        nb.suptitle = suptitle
    if footer is not None:
        nb.footer = footer
    if plot_size is not None:
        if not isinstance(plot_size, (tuple, list)) or len(plot_size) != 2:
            raise ValueError("plot_size must be a (width, height) tuple in "
                             f"inches, got {plot_size!r}")
        nb.set_plot_size(plot_size[0], plot_size[1],
                         per_subplot=plot_size_per_subplot)

    if bool(static_images) != nb.static_images or static_scale != nb.static_scale:
        nb.set_static_images(static_images, scale=static_scale)
    if bool(copy_buttons) != nb.copy_buttons:
        nb.set_copy_buttons(copy_buttons)

    if data is not None:
        nb.load_df(data, **load_kwargs)
    return nb


def uc_defaults():
    """Return ``new_uc``'s presets — every setting and the built-in it defaults
    to — as a plain dict. The values are read off the signature, so this is the
    same list ``new_uc()`` applies, with ``auto`` for the style-dependent ones.

    Handy for seeing what unichart's current defaults are, and for building a
    named preset on top of them::

        report = {**uc_defaults(), 'figsize': (10, 6), 'markersize': 5}
        report.pop('data')
        nb = new_uc(**report)
    """
    return {name: p.default
            for name, p in inspect.signature(new_uc).parameters.items()
            if p.kind is not inspect.Parameter.VAR_KEYWORD}
