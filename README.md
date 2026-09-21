# unichart

A stateful, multi-dataset plotting and dashboard toolkit for Jupyter notebooks,
built on [Plotly](https://plotly.com/python/).

`unichart` is designed for the common real-world workflow of comparing **many
datasets** (test runs, simulation cases, measurement series) that share the same
schema. Instead of hand-assembling Plotly traces, you load your data once,
choose what to show, and get publication-ready interactive figures with a short,
memory-ful API. A companion module, `unichart_dashboard`, wires those same
figures into interactive Dash boards or self-contained HTML files.

- **`unichart`** — the `UnichartNotebook` plotting environment (core).
- **`unichart_dashboard`** — optional Dash dashboards built from notebook panels.
- **`unichart_terminal`** — the `explore()` GUI: a sidebar, a chart pane and a
  Python terminal for plotting on the fly.
- **`unichart_cli`** — the `unichart` command: open that GUI on a data file
  straight from a terminal.

---

## Installation

Install directly from GitHub with pip:

```bash
pip install git+https://github.com/Cunon/unichart.git
```

Optional extras:

```bash
# Trend lines / regressions (LOWESS smoothing via statsmodels)
pip install "unichart[trend] @ git+https://github.com/Cunon/unichart.git"

# Interactive web dashboards (Dash)
pip install "unichart[dashboard] @ git+https://github.com/Cunon/unichart.git"

# Everything
pip install "unichart[all] @ git+https://github.com/Cunon/unichart.git"
```

The `unichart` command comes with the package. Its GUI needs the `dashboard`
extra (`--info` and `--html` work without it):

```bash
pip install "unichart[dashboard] @ git+https://github.com/Cunon/unichart.git"
unichart runs.csv
```

### Requirements

- Python >= 3.9
- `pandas`, `numpy`, `plotly`, `scipy`, `ipywidgets`, `ipython` (installed automatically)
- Optional: `statsmodels` for trend lines, `dash` for dashboards and the
  `unichart` command's GUI, `kaleido` for static-image / PNG export

---

## Core concepts

A `UnichartNotebook` holds any number of **datasets** (`nb.sets`, each a
`Dataset`) backed by a single shared DataFrame. Each dataset carries its own
style state (color, marker, line style, selection flag, query, title). The
typical loop is:

| Step | Methods |
|------|---------|
| **1. Load** | `load_df` · `load` · `load_clipboard` |
| **2. Select** | `select` · `omit` · `query` · `restore` |
| **3. Plot** | `plot` · `plot_ymult` · `plot_marginal` · `bar` · `box` · `histogram` · `contour` · `table` |
| **4. Style** | `color` · `marker` · `var_format` · `set_default_format` · `toggle_darkmode` · `set_plot_style` |
| **5. Analyse** | `delta` · `table` · `summary` · `reg_info` · `combine_sets` |

Two conveniences run through the whole API:

- **Sticky arguments.** Plot calls remember their last `x`/`y` (`nb.last_x`,
  `nb.last_y`, grid, format), so follow-up styling and analysis calls don't need
  to re-specify them.
- **Built-in help.** `nb.help()` prints a categorized method + attribute
  overview; `nb.help('delta')` prints one method's full signature and docstring;
  `nb.help('Plotting')` lists a single category. Headings, method names and
  signatures are colored in a terminal, in Jupyter and in the explorer's
  terminal (plain when piped; `NO_COLOR=1` disables it).

---

## Quick start

```python
from unichart import UnichartNotebook

nb = UnichartNotebook()

# 1. Load — one DataFrame split into one Dataset per unique set id,
#    or a list of DataFrames loaded as separate sets.
nb.load_df(df, set_idx_column='run_id', set_name_column='run_name')

# 2. Select which datasets participate in the next plot
nb.select([0, 1, 2])          # or nb.omit(3), nb.query('all', 'temp > 100')
nb.select('1:10')             # range shorthand — sets 1-9 (exclusive stop,
                              # like a Python slice). Also '0,3,7:', '::2', '-3:'.
                              # Works anywhere a set selector is accepted:
                              # nb.omit('5:8'), nb.color('0:3', 'red'), ...

# 3. Plot — one subplot per Y variable by default
nb.plot(x='time', y=['temperature', 'pressure'])

# 4. Style
nb.color(0, 'red')                        # dataset 0 red
nb.var_format('temperature', linestyle='--')   # every temperature line dashed
nb.toggle_darkmode(True)

# 5. Analyse
nb.plot(x='time', y='temperature', by='ymult')   # shared X, multiple Y axes
nb.delta(base_idx=0, study_indices='all', delta_parms='temperature')
```

### Loading data

- `load_df(df, ...)` — split one DataFrame into one dataset per unique
  `set_idx_column` value, or load a **list** of DataFrames as separate sets
  (`combined=True` concatenates them into one). Auto-detects `SETNUMBER` /
  `INDEX` / `TITLE` columns if you don't name them.
- `load(source, ...)` — load from a file path (CSV/Excel/etc.) or other source.
- `load_clipboard()` — pull a table straight from the system clipboard.
- `add_column`, `set_column`, `combine_sets` / `combine`, `clear_data` — manage
  the shared frame after loading.

### Selecting & querying

Every plot draws only the **selected** datasets, so selection is how you slice:

```python
nb.select([0, 2])          # show only sets 0 and 2
nb.omit(1)                 # hide set 1
nb.restore()               # re-select everything
nb.query(1, 'rpm > 5000')  # row-level filter on set 1 (pandas query syntax)
nb.selected()              # list currently selected sets
```

Most styling/analysis methods accept the same `uset_slice` argument: an int, a
list of ints, `'all'`, or `'selected'`.

---

## Plot types

All plotting methods return an interactive Plotly `go.Figure` (cached in
`nb.last_fig`) and share layout options (`figsize`, `ncols`/`nrows`,
`hspace`/`vspace`, `suptitle`, `footer`, `legend`, `by`).

| Method | What it draws |
|--------|---------------|
| `plot(x, y, by=...)` | Line / scatter. `by='vars'` = one subplot per Y variable (default); `by='sets'` = one subplot per dataset; `by='ymult'` = single plot with multiple Y axes. |
| `plot_ymult(x, y)` | One plot, multiple stacked/overlaid Y axes for differently-scaled variables. |
| `plot_marginal(x, y, marginal=)` | Scatter with marginal distribution strips: `x`'s distribution above the plot, `y`'s to its right (`'histogram'` default, or `'box'`, `'violin'`, `'rug'`, `'kde'`; `marginal_x`/`marginal_y` set one side or drop it with `False`). Also `plot(x, y, by='marginal')`. |
| `bar(x, y, barmode=, agg=)` | Bar charts: one bar per dataset and `x` category, height = `agg` of `y` over that category's rows (`'mean'` default; any pandas reducer name, a callable, or `agg=False` for pre-reduced rows). Grouped or stacked; hover shows the reducer and row count. |
| `box(x, y, points=, notched=)` | Box / distribution plots. |
| `histogram(x, histfunc=, nbins=)` | Histograms with configurable binning and normalization. |
| `contour(x, y, z, overlay_sets=)` | Filled/line contour maps from scattered data, with optional scatter overlays. |
| `table(cols=, x_in=, kind=)` | Rendered data table; optionally interpolate values at given `x_in` points. |

```python
nb.plot(x='time', y='temp', by='sets', ncols=2, suptitle='Per run')
nb.plot_marginal(x='rpm', y='torque', marginal='kde')
nb.bar(x='config', y='efficiency', barmode='group', agg='mean')
nb.histogram(x='error', nbins=40, histnorm='probability')
nb.contour(x='rpm', y='torque', z='efficiency', overlay_sets=[1, 2])
```

---

## Styling & formatting

Style resolves in layers — **per-variable** overrides beat **per-dataset**
style, which beats **notebook defaults** — each on a per-attribute basis.

**Per-dataset** (by `uset_slice`):

```python
nb.color(0, 'red'); nb.marker([1, 2], 's'); nb.linestyle('all', '--')
nb.markersize(0, 12); nb.alpha('selected', 0.5); nb.fill(1, True)
nb.linewidth(0, 3); nb.edgewidth(0, 1); nb.hue(0, 'category')
nb.alpha_marker(0, 0.3); nb.alpha_line(1, 0.2)   # opacity of just the markers / just the line
nb.zorder(2, 1)       # draw set 2 on top of the rest (higher = later = on top)
nb.sig_figs(0, 4)     # set 0's values display to 4 significant figures
nb.decimals(1, 2)     # set 1's to two decimal places instead
```

`sig_figs` and `decimals` are the display-precision knob, in its two
spellings: significant figures, or places after the point (trailing zeros
kept, `0` = whole numbers). They round what is *shown* — `table()` cells,
`summary()` statistics and the plot hover readouts (x/y/z and the
`display_parms` lines) — and never the stored data, so `output='df'` and
`ds.df` keep full precision. A value is rounded one way or the other, so
setting either clears the other. One argument is the notebook-wide form (it
also restyles the sets already loaded), two are selector-then-value, and
`'reset'` restores the built-in precision:

```python
nb.sig_figs(4)          # 4 sig figs everywhere, and for sets loaded later
nb.decimals(2)          # ... or two decimal places everywhere
nb.sig_figs(0, 6)       # just set 0 (use this form when you mean a set index)
nb.sig_figs('reset')    # back to the built-in precision
nb.sig_figs()           # report the current setting
```

**Per-variable** (applies wherever that column is plotted):

```python
nb.var_format('Temperature', linestyle='--')          # all Temp lines dashed
nb.var_format('Pressure', color='blue', marker='s')   # Pressure = blue squares
nb.var_format('Pressure', color='reset')              # drop just the color override
nb.var_format(['CHT1', 'CHT2'], marker='x')           # a list gets the same overrides
nb.var_format(['CHT1', 'CHT2'], reset=True)           # drop all their overrides
```

**Notebook-wide defaults & appearance:**

- `set_default_format(...)` — persistent defaults (markersize, linestyle,
  sig_figs/decimals, legend, grid, subplot spacing, barmode, agg, alpha, …)
  applied to future plots/datasets.
- `set_color_palette` / `color_map` / `marker_map` — the ordered lists assigned
  to datasets by index (integer lookups cycle).
- `toggle_darkmode(True/False)` — dark theme.
- `set_plot_style('matplotlib')` — Matplotlib-look figures (see below).
- `set_font_sizes` / `get_font_sizes` — named sizes (`'sm'`, `'lg'`, `'xl'`, …)
  for title, legend, axes, ticks, table cells, hover, etc.
- `set_plot_size` — pin the plot area so every panel comes out the same size
  and aspect ratio regardless of titles, legends or subplot count (see below).
- `grid(...)` — gridline formatting: visibility, color, width, dash pattern,
  per axis (`axis='x'/'y'/'both'`) and major/minor (`which=`), e.g.
  `nb.grid(color='lightgray', dash=':')` or `nb.grid(which='minor', visible=True)`.
- `watermark(...)` — stamp a logo or seal onto every plot, with control over
  opacity, position and size (see below).
- `set_static_images(True)` / `save_png(...)` — render flat PNGs inline (keeps
  notebook file size down) or export a high-resolution PNG (needs `kaleido`:
  `pip install unichart[png]`).
  Every saved PNG also carries the full plotting session (data, queries,
  formatting and the plot call) in a metadata chunk, so
  `UnichartNotebook.from_session('plot.png')` or `nb.load_session('plot.png')`
  remakes the plot from the image alone; `read_png_session('plot.png')` shows
  what is embedded, `save_png(..., embed_session=False)` writes a plain image.
  `parms=['x', 'y']` whitelists the columns embedded (plotted, query and hue
  columns are always kept) so a wide table doesn't bloat the file.
- `save_session(path)` / `load_session(path)` — the same session as a
  standalone `.json` (file references or embedded rows, `embed_data=`; the
  same `parms=` whitelist applies to embedded sets, and it likewise keeps the
  columns the recorded plot needs). A `.json` session records the last plotting
  call just as a PNG does, so `load_session` brings the figure back with the
  data; pass `replay=False` for the data and formatting alone. A call that
  won't replay — a `figsize` JSON could only store as text, say — warns and
  leaves everything else restored.

Both formats are openable from the `unichart` command and droppable on the
explorer's sidebar; see [Command line](#command-line) below.

Marker and line-style strings are **Matplotlib-compatible** (`'o'`, `'s'`,
`'^'`, `'--'`, `'-.'`, `':'`) and translated to Plotly automatically.

### Matplotlib look

Plots are drawn with Plotly and look like it. If your figures need to sit next
to Matplotlib output — a paper, a report, a deck already full of `pyplot` —
switch the whole environment over:

```python
nb.set_plot_style('matplotlib')   # or 'mpl' / 'plt'
nb.set_plot_style('plotly')       # back to the default look
```

That restyles plots to approximate Matplotlib's defaults: a white (or black, in
dark mode) plot area framed by spines on all four sides, outward ticks, no zero
lines, a gray grid, DejaVu Sans at Matplotlib's point sizes, the **tab10** color
cycle, and **viridis** for contours and hue-colored scatters.

- It is **orthogonal to `toggle_darkmode`** — each style has a light and a dark
  variant — and to the rest of the formatting API: `color`, `markersize`,
  `var_format`, `set_font_sizes` and friends still win wherever you set them.
- Existing datasets are restyled too. That clears manual `color()` /
  `markersize()` / `hue_palette()` overrides on them; pass `sets=False` to keep
  those and apply the style only to the layout and to future loads.
- Two Matplotlib habits stay opt-in: it draws lines *without* markers
  (`nb.set_default_format(marker=None)` if you want that), and dashboards keep
  the board's UI font so charts and chrome read as one surface.
- `reset_format('all')` (or `'defaults'`) returns to the `'plotly'` style along
  with the other stored defaults.

### Decorations

```python
nb.line('rpm', level=5000, color='red', linestyle='--')  # reference line
nb.highlight('time', (10, 20), color='yellow', alpha=0.2) # shaded band
nb.scale('pressure', (0, 100))                            # fix an axis range
nb.suptitle = 'Overview'; nb.footer = 'source: rig A'      # figure text
```

Reference lines can carry a **label**, drawn on the line inside the plot area:

```python
nb.line('rpm', 5000, label='redline')                     # default: far end of the line
nb.line('cht', 400, color='orange', label='limit',
        label_size='lg', label_position='left')           # named size + position
nb.line('time', 12.5, label='event', label_position=0.25) # 0-1 fraction along the line
nb.line('cht', 350, label='target', label_position='center below', label_color='gray')
```

- `label_size` — a number or a size name (`'small'`, `'lg'`, …, same vocabulary
  as `set_font_sizes`). Defaults to the `axes_tick` size when one is set.
- `label_position` — a `0`–`1` fraction along the line, or position tokens
  (which may be combined with a fraction, e.g. `'0.25 left'`). A vertical line
  slides with `'top'`/`'middle'`/`'bottom'` and picks its side with
  `'left'`/`'right'`; a horizontal line slides with `'left'`/`'center'`/`'right'`
  and picks its side with `'above'`/`'below'` (`'top'`/`'bottom'`).
- `label_color` — defaults to the line's `color`.

### Fixed plot size / aspect ratio

`figsize` sets the size of the whole *figure*, so the actual drawing area moves
around as titles wrap, legends grow or a plot splits into more subplots — and
two plots in the same notebook end up different shapes. `set_plot_size` pins
the **plot area** instead:

```python
nb.set_plot_size(4, 3)          # every panel exactly 4x3in, in every plot
nb.plot(x='t', y='CHT')                   # 1 panel,  4x3in
nb.plot(x='t', y=['CHT', 'EGT', 'RPM'])   # 3 panels, 4x3in each
nb.set_plot_size(height=3)      # pin height only; width follows figsize
nb.set_plot_size(reset=True)    # back to figsize-driven sizing
```

The size applies to **one subplot panel** by default, and the figure grows to
fit the grid plus its margins — so panels keep the same size and aspect ratio no
matter how many variables you plot or how tall the title and legend get. Pass
`per_subplot=False` to pin the combined grid area instead (the behaviour this
method had before per-panel sizing), which holds the figure size steady but
shrinks each panel as the grid grows.

Multi-axis plots (`plot_ymult`, or `bar`/`box` with `by='dataset_x'`) are
handled too: each stacked right-hand Y axis keeps a fixed pixel slot for its
tick labels and title, so a narrow or pinned plot pushes the axes further out
instead of crushing them into each other.

Each call replaces the previous setting, `per_subplot` included — a later
`set_plot_size(height=3)` returns to per-panel mode unless you pass
`per_subplot=False` again.

Note that in per-panel mode a wide grid makes a wide figure — five 6in panels
side by side is a ~22in figure. Use a smaller per-panel size or `ncols=1` when
that is inconvenient. Dashboard panels are rendered at the size the board gives
them, so the pin governs notebook figures, not board tiles.

### Subplot spacing

The gap between subplots is a fixed pixel budget — 80px between columns, 70px
between rows (enough for the neighbouring panel's tick labels, axis title and
subplot title) — whatever the grid size, so a 6-row plot doesn't overlap and a
2-row plot doesn't waste a third of the figure on the gap. Contour plots and
secondary-axis plots reserve more for their colorbars and extra axes. When a
grid is too crowded for its figure the gaps are clamped and a warning says so;
a bigger `figsize` or `set_plot_size` is the cure.

To change it, every gridded plot method (`plot`, `bar`, `box`, `histogram`,
`contour`) takes `hspace` (columns) and `vspace` (rows), and
`set_default_format` sets the standing default:

```python
nb.plot(x='t', y=['CHT', 'EGT', 'RPM'], ncols=1, vspace=100)   # 100px rows
nb.plot(x='t', y=['CHT', 'EGT'], hspace=0.05)     # 5% of the plot width
nb.set_default_format(hspace='60px', vspace=40)   # for every plot from now on
```

A value of 1 or more is pixels (`60` or `'60px'`); below 1 it is a fraction of
the plot area, the way Plotly's `horizontal_spacing`/`vertical_spacing` take
it. With `set_plot_size` pinned, the pixel gap is exact and the figure grows to
hold it.

### Watermarks

Stamp a logo, seal or "DRAFT" graphic onto every plot:

```python
nb.watermark('logo.png')                                   # faint, centered
nb.watermark('logo.png', opacity=0.4,
             position='bottom right', size=0.15)           # corner logo
nb.watermark('draft.png', opacity=0.08, layer='above')     # tint over the data
nb.watermark(opacity=0.3)                                  # tweak; other settings kept
nb.watermark()                                             # show current settings
nb.watermark(reset=True)                                   # remove it
```

- **`opacity`** — `0`–`1` (default `0.15`).
- **`position`** — any combination of `'top'`/`'upper'`, `'bottom'`/`'lower'`,
  `'left'`, `'right'`, `'center'`/`'middle'` — e.g. `'center'` (default),
  `'bottom right'`, `'top'`, `'center left'`. Or an explicit `(x, y)` pair in
  paper coordinates (`0`–`1` across the plot area, `(0, 0)` = bottom left),
  which centers the image on that point.
- **`size`** — fraction of the plot area the image is fitted into (default
  `0.3`); pass `(width, height)` to set the two separately. `sizing='contain'`
  (default) preserves the image's aspect ratio inside that box; `'fill'` crops
  to cover, `'stretch'` distorts to fit.
- **`layer`** — `'below'` (default) draws under the data, `'above'` over it.
- **`source`** — a file path (png, jpg, gif, webp, bmp, svg), an `http(s)` URL,
  a `data:` URI, raw bytes, or a PIL image. Local files are read and inlined as
  base64 at call time, so the watermark also appears in `save_png`,
  `set_static_images` mode, the ⧉ copy button and a dashboard's chart panels —
  and a bad path raises immediately instead of silently drawing nothing. A URL
  is passed through as-is and only renders where the viewer can reach it. The
  image rides along in every figure, so keep the file small.

Settings persist across plots and merge across calls, like `grid()`.

### Resetting formatting

Two consistent rules cover every reset:

1. **Any formatting setter accepts `'reset'` as its value** to restore the
   default for whatever it targets — a dataset attribute, a variable override,
   or a decoration:

   ```python
   nb.color(0, 'reset')             # dataset 0 back to its color_map color
   nb.marker('all', 'reset')        # every set back to its marker_map marker
   nb.color('Pressure', 'reset')    # drop the Pressure color override
   nb.var_format('CHT', color='reset')  # same, per attribute
   nb.line('all', 'reset')          # remove reference lines ('clear' also works)
   nb.highlight('rpm', 'reset')     # remove highlights on one column
   nb.scale('all', 'reset')         # clear every fixed axis range
   ```

2. **`reset_format()` is the single bulk-reset hub**, with optional scopes
   `'sets'`, `'vars'`, `'lines'`, `'highlights'`, `'scales'`, `'fonts'`,
   `'plot_size'`, `'grid'`, `'watermark'`, `'defaults'`, `'all'`:

   ```python
   nb.reset_format()                     # all applied formatting
   nb.reset_format('lines', 'scales')    # just those
   nb.reset_format([0, 1])               # just datasets 0 and 1
   nb.reset_format(vars='CHT')           # just one variable's overrides
   nb.reset_format('all')                # everything, incl. set_default_format state
   ```

   `set_font_sizes(reset=True)`, `set_plot_size(reset=True)`,
   `set_default_format(reset=True)`, and `clear_var_format()` still work and
   are equivalent to the matching `reset_format` scope.

---

## Analysis & stats

- **`delta(base_idx, study_indices, delta_parms=...)`** — compute absolute
  (`DL_<P>`) and percentage (`DLPCT_<P>`) differences of each study dataset
  against a baseline, aligned by nearest-match merge on a chosen column (with
  optional interpolation at specified `x_ins`). The delta set inherits the study
  set's color/marker so it reads as a continuation of that series.
- **`table(...)` / `table_read(...)`** — tabulate columns, or interpolate a
  Y column at arbitrary X inputs (`kind='linear'`, extrapolation controllable).
  `sig_figs=` / `decimals=` round the displayed cells; without them each set's
  own `sig_figs` / `decimals` (see `nb.sig_figs`, `nb.decimals`) applies.
- **`reg_info(...)`** / `reg_order` — fit and report regressions / trend lines
  (polynomial or LOWESS; LOWESS needs `statsmodels`).
- **`summary(cols=...)`** — per-dataset descriptive statistics (count / min /
  mean / max / std), displayed as the same sortable, filterable, copyable
  table as `table(...)`; `output='df' | 'md' | 'fig'`, `sig_figs=` and
  `decimals=` work the same way too.
- **`min` / `max` / `mean` / `median`** — quick per-column aggregates.

---

## Dashboards (`unichart_dashboard`)

The same figures compose into an interactive **Dash** board with one shared data
context: a header bar owns the dataset selection and the light/dark theme for
*every* panel, and each panel keeps only the controls that are genuinely its own
(plot type, x / y / z variables, title, legend position).

```python
# Inline in a Jupyter notebook:
nb.dashboard(panels=[
    {'method': 'plot', 'x': 'time', 'y': 'temp'},
    {'method': 'bar',  'x': 'config', 'y': 'efficiency',
     'kwargs': {'barmode': 'group', 'agg': 'mean'}},
    {'method': 'contour', 'x': 'rpm', 'y': 'torque', 'z': 'eff',
     'datasets': [0], 'kwargs': {'overlay_sets': [1, 2]}},
], ncols=2, title='Test-rig overview')
```

Panel spec keys:

- `method` — one of `plot`, `plot_ymult`, `plot_marginal`, `bar`, `box`,
  `histogram`, `contour`, `table` (default `plot`).
- `x`, `y`, `z` — variables (the `z` / legend controls appear only for the
  methods that use them).
- `suptitle` — the card title (also names CSV exports).
- `datasets=[...]` — **pin** the panel to specific datasets; it then ignores the
  header picker and carries a "pinned" badge.
- `kwargs={...}` — method-specific passthrough (e.g. `{'nbins': 20}`,
  `{'barmode': 'stack'}`, `{'overlay_sets': [1, 2]}`); each is applied only when
  the active method accepts it, so it survives plot-type switches.

Key options:

- `controls=False` — render a locked **presentation** board: a clean grid of
  titled figure cards with all editing chrome hidden.
- `jupyter_mode` — `'inline'` (default), or `'external'` / `'tab'` to open a
  browser. Ports are auto-selected if the preferred one is busy.

### Explorer — a plotting terminal

`dashboard()` renders a board you specified in code. `explore()` opens a
workspace you drive by **typing**: a sidebar with a drop zone, the loaded
datasets and a clickable cheat sheet; a chart pane showing the latest figure;
and a Python terminal underneath.

```python
from unichart_dashboard import explore

explore(data='runs.csv')   # standalone — serves the board and opens your browser
explore()                  # empty; drop a file on the sidebar, or hit "Load demo data"
nb.explore()               # on a notebook you already have (inline in Jupyter)
nb.explore(app_window=True)  # in its own desktop window instead of a browser tab
```

The notebook's methods are bound as bare names in the terminal, so the cheat
sheet reads the way the library does, and `nb` covers everything else:

```python
>>> nb.load('runs.csv')
>>> plot(x='time', y=['temperature', 'pressure'])
>>> select([0, 1]); color(0, 'red')
>>> summary()
```

- **Enter** runs, **Shift+Enter** adds a line, **↑ / ↓** walks history.
- **⧉ copy chart** in the top bar puts the current figure on the clipboard as a
  2× PNG, rasterized from what's on screen (no `kaleido` needed). The same
  `⧉ copy` affordance unichart shows under plots in a notebook — `nb.plot()` in
  Jupyter still has its own; this is the board's.
- **Syntax highlighting** everywhere Python appears: the cheat-sheet snippets,
  every command in the transcript, and the input line as you type. The
  transcript is coloured by Python's own tokenizer, so f-strings, comments and
  nested quotes are handled properly; half-typed input degrades to plain text
  rather than breaking.
- **Drag the pane edges** to resize: the sidebar's right edge and the divider
  between the chart and the terminal. Double-click an edge to reset it. Sizes
  are remembered per browser, and the chart reflows as you drag.
- A trailing expression echoes its value, like any REPL — `1 + 1` prints `2`.
- Plots go to the chart pane; `table()` / `summary()` / `list_parms()` render as
  their real sortable, filterable HTML tables inline in the transcript.
- Errors show a traceback trimmed to the line you typed.
- Dropping a file loads it through a visible `nb.load(...)` command, so the
  transcript is a real record of the session. Drop a **saved session** —
  `.json` or a `save_png` image — and it restores through an equally visible
  `nb.load_session(...)`, bringing its datasets, formatting and plot back. The
  file's contents decide, not its extension, so a data `.json` still loads as
  data.
- **✕ close** in the top bar shuts the board down: it asks first, then stops
  the server and ends the process — so `unichart runs.csv` in a shell returns
  to the prompt without a `Ctrl-C`. Typing `exit()` in the terminal pane does
  the same. Inside a Jupyter kernel neither is offered, because the process
  they would end is the kernel.
- **💾 save session** in the top bar downloads the board as a session file:
  everything loaded, however it is styled, and whatever is currently plotted.
  Reopen it with `unichart that-file.json`, by dropping it back on the sidebar,
  or with `nb.load_session(...)` from Python.
- The board runs against **your** notebook: what you load or restyle there is
  on `nb` afterwards.

The board is dark, and `explore()` switches the notebook to dark mode to match
unless it already is. Options: `data=`, `panels=` (dashboard-style specs,
replayed as startup commands), `title=`, `port=`, `open_browser=False` for
headless hosts, and `jupyter_mode=`.

> **The terminal executes real Python in your process.** It can do anything you
> could do at a Python prompt, so the server binds to `127.0.0.1` only. It is
> not a sandbox — don't expose it to a network.

### Command line

Installing the package puts a `unichart` command on your PATH, so a quick look
at a data file never needs a Python session at all:

```bash
unichart                                # empty explorer; load from the data bar
unichart runs.csv                       # open the explorer on one file
unichart runs.csv --app                 # ...in its own window, like a desktop app
unichart a.csv b.csv --combine          # several files, merged into one dataset
unichart runs.csv --set-col run_id      # split into one dataset per run
unichart runs.csv --info                # print datasets + columns, then exit
unichart runs.csv --html board.html     # write a static board instead of serving
```

A `FILE` can also be a **saved session** — a `.json` from `save_session`, or a
PNG from `save_png`, which carries its session in a metadata chunk. Those are
restored rather than read as data, so the datasets, queries, formatting *and the
plot* come back:

```bash
unichart plot.png                       # reopen the plot that PNG came from
unichart board.json --info              # inspect a session without serving
unichart runs.csv board.json            # data first, then the session on top
unichart runs.csv --panel plot:time:temp --save-session board.json
```

Data files load first (as one batch, so `--combine` still means what it says),
then sessions in the order given — so a session's own formatting is the one that
sticks, and any `--panel` draws on top of the restored state. A session records
its own theme, which the explorer honours; `--dark` overrides it. `--combine`,
`--set-col` and `--name-col` describe how to *read* a data file, so passing one
with nothing but a session on the line is an error rather than a no-op.

Seed the board with panels (repeat `--panel`, as `method:x:y[,y2][:z]`). For
the GUI each one is replayed as a startup command in the transcript; for
`--html` each becomes a card:

```bash
unichart runs.csv --panel plot:time:temp,press --panel histogram:temp
unichart map.csv  --panel contour:rpm:torque:eff --html map.html --embed-js inline
```

`unichart --help` is colored when it's printing to a terminal — flags in cyan,
metavars and choices in green, section headings in bold — and plain the moment
it's piped or redirected. `NO_COLOR=1` turns it off; `FORCE_COLOR=1` forces it
back on.

#### A window of its own (`--app`)

By default the explorer opens in a browser tab, next to everything else you had
open. `--app` opens it as a **standalone window** instead — no tabs, no address
bar, no bookmarks: just the board, titled and iconed as itself, with its own
entry in the taskbar.

```bash
unichart runs.csv --app
```

```python
explore(data='runs.csv', app_window=True)   # the same thing from Python
nb.explore(app_window=True)
```

It works by handing the URL to a Chromium-family browser's `--app` mode (Chrome,
Chromium, Brave or Edge — whichever is found first; `UNICHART_APP_BROWSER=/path/to/browser`
names one the search misses). There is no extra dependency and no packaging
step: it's the browser you already have, wearing a different window. On a
machine with none of them — a Firefox-only box — the board opens in an ordinary
tab and says why.

Two things to know. `--no-browser` still wins, so a headless host is unaffected.
And closing the *window* does not stop the server — the board is still running
in the terminal you launched it from. To end it, use the top bar's **✕ close**
(or type `exit()` in the terminal pane), which asks first and then stops the
server and the program; `Ctrl-C` in the launching terminal still works too.
Inside a Jupyter kernel there is no close button, because the process it would
end is the kernel.

A note on the icon. The board serves its own — a line over three bars, in the
board's palette — and the browser paints it in the window's title bar, the tab
and the page. Whether it also reaches the *taskbar* is the desktop's call, not
the browser's: X11 and Windows take the icon from the window, so it follows;
Wayland matches windows to `.desktop` files instead, so an `--app` window there
shows the browser's generic icon. The fix on those desktops is to install the
board, which the served web app manifest exists for: Chrome's *Install page as
app* (in the ⋮ menu) writes a real launcher entry carrying the unichart icon,
and starting the board from it gets you the icon everywhere.

Other flags: `--title`, `--port`, `--no-browser` (serve without opening a
browser — headless or remote hosts), `--save-session`, and `--version`;
`--ncols` `--width` `--height` apply to `--html`, and `--dark` to `--html` and
to a session's stored theme (the terminal board's own chrome is always dark).
`unichart --help` lists them all.

Panel options beyond `method` / `x` / `y` / `z` — `kwargs` like `nbins` or
`barmode`, dataset pins — aren't expressible as a flag; use `nb.dashboard` /
`nb.explore` from Python, where the spec dict is clearer than any encoding
would be. A missing file or malformed `--panel` prints one line and exits
non-zero rather than raising.

If `unichart` isn't found after installing, the module is runnable directly:
`python -m unichart_cli runs.csv`.

#### Tab completion

Completion is built in — no extra package to install. Add one line to your
shell's rc file:

```bash
eval "$(unichart --completion bash)"    # ~/.bashrc
eval "$(unichart --completion zsh)"     # ~/.zshrc
```

Then `--<TAB>` lists the flags, and `--panel` completes field by field against
the **real column names of the files already on the command line** — so a board
can be built without opening the data first:

```
$ unichart runs.csv --panel <TAB>
plot:  plot_ymult:  plot_marginal:  bar:  box:  histogram:  contour:  table

$ unichart runs.csv --panel plot:<TAB>
time   temp   press   rpm

$ unichart runs.csv --panel plot:time:te<TAB>
$ unichart runs.csv --panel plot:time:temp,<TAB>     # y takes a list
```

`--set-col` / `--name-col` complete column names the same way, `--embed-js` and
`--completion` complete their choices, and filenames fall through to the
shell's own completion. Only the column lookups read a file (header row only),
so completing a flag costs nothing.

### Export to standalone HTML

```python
nb.dashboard_to_html(panels, path='board.html')
```

Renders each panel once and writes a self-contained HTML file (the frozen
presentation view). The Plotly charts stay fully interactive (hover, zoom, pan,
modebar), a global dataset filter is recreated as offline chips
(`global_select=True`), and `table` panels become real HTML tables with
click-to-sort headers. `embed_js` controls how plotly.js is included:
`'cdn'` (small, needs internet), `True` (embeds the full library for a truly
offline file), or `'directory'`.

Dash is imported lazily, so the core toolkit never requires it.

---

## Utility & design notes

- **One shared frame, many views.** All datasets live in a single backing
  DataFrame with per-set style/selection state, so cross-dataset operations
  (deltas, combined sets, consistent color/marker assignment) are cheap and
  consistent.
- **Notebook-friendly memory management.** Static-image mode and last-figure
  clearing keep notebook file sizes manageable even with many large plots, while
  `nb.last_fig` still caches the real interactive figure for re-styling or PNG
  export.
- **Discoverable.** `nb.help()`, `nb.list_sets()`, `nb.list_parms()`, and
  `nb.summary()` let you inspect the environment without leaving the notebook.

---

## Learning more

- **`nb.help()`** — live, categorized API reference inside the notebook.
- **[`PLOTTING_STYLE_GUIDE.md`](PLOTTING_STYLE_GUIDE.md)** — conventions for
  producing clean, consistent figures.
- **[`demo_notebooks/`](demo_notebooks/)** — runnable examples covering the main
  features:
  - `UnichartNotebook_Tutorial.ipynb`, `unichart_data_model_tutorial.ipynb` — start here.
  - `dashboard_demo.ipynb`, `dashboard_contour_demo.ipynb`,
    `dashboard_progression_demo.ipynb` — dashboards.
  - `delta_demo.ipynb`, `interpolation_table_tests.ipynb` — analysis.
  - `variable_color_formatting_demo.ipynb`, `marker_map_tests.ipynb`,
    `color_map_tests.ipynb`, `environment_presets_demo.ipynb` — styling.
  - `plot_style_demo.ipynb` — the Matplotlib look (`set_plot_style`).
  - `contour_overlay_demo.ipynb`, `static_images_demo.ipynb`,
    `large_data_showcase.ipynb` — specialized plotting.

## Usage

```python
from unichart import UnichartNotebook
from unichart_dashboard import dashboard, explore   # optional, needs `dash`

nb = UnichartNotebook()
```
