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

### Requirements

- Python >= 3.9
- `pandas`, `numpy`, `plotly`, `scipy`, `ipywidgets`, `ipython` (installed automatically)
- Optional: `statsmodels` for trend lines, `dash` for dashboards, `kaleido` for static-image / PNG export

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
| **3. Plot** | `plot` · `plot_ymult` · `bar` · `box` · `histogram` · `contour` · `table` |
| **4. Style** | `color` · `marker` · `var_format` · `set_default_format` · `toggle_darkmode` |
| **5. Analyse** | `delta` · `table` · `summary` · `reg_info` · `combine_sets` |

Two conveniences run through the whole API:

- **Sticky arguments.** Plot calls remember their last `x`/`y` (`nb.last_x`,
  `nb.last_y`, grid, format), so follow-up styling and analysis calls don't need
  to re-specify them.
- **Built-in help.** `nb.help()` prints a categorized method + attribute
  overview; `nb.help('delta')` prints one method's full signature and docstring;
  `nb.help('Plotting')` lists a single category.

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
`suptitle`, `footer`, `legend`, `by`).

| Method | What it draws |
|--------|---------------|
| `plot(x, y, by=...)` | Line / scatter. `by='vars'` = one subplot per Y variable (default); `by='sets'` = one subplot per dataset; `by='ymult'` = single plot with multiple Y axes. |
| `plot_ymult(x, y)` | One plot, multiple stacked/overlaid Y axes for differently-scaled variables. |
| `bar(x, y, barmode=, agg=)` | Bar charts, grouped or stacked, with aggregation. |
| `box(x, y, points=, notched=)` | Box / distribution plots. |
| `histogram(x, histfunc=, nbins=)` | Histograms with configurable binning and normalization. |
| `contour(x, y, z, overlay_sets=)` | Filled/line contour maps from scattered data, with optional scatter overlays. |
| `table(cols=, x_in=, kind=)` | Rendered data table; optionally interpolate values at given `x_in` points. |

```python
nb.plot(x='time', y='temp', by='sets', ncols=2, suptitle='Per run')
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
```

**Per-variable** (applies wherever that column is plotted):

```python
nb.var_format('Temperature', linestyle='--')          # all Temp lines dashed
nb.var_format('Pressure', color='blue', marker='s')   # Pressure = blue squares
nb.var_format('Pressure', color='reset')              # drop just the color override
```

**Notebook-wide defaults & appearance:**

- `set_default_format(...)` — persistent defaults (markersize, linestyle,
  legend, grid, barmode, agg, alpha, …) applied to future plots/datasets.
- `set_color_palette` / `color_map` / `marker_map` — the ordered lists assigned
  to datasets by index (integer lookups cycle).
- `toggle_darkmode(True/False)` — dark theme.
- `set_font_sizes` / `get_font_sizes` — named sizes (`'sm'`, `'lg'`, `'xl'`, …)
  for title, legend, axes, ticks, table cells, hover, etc.
- `set_plot_size` — pin the inner plot area so plots stay the same size
  regardless of titles/legends.
- `set_static_images(True)` / `save_png(...)` — render flat PNGs inline (keeps
  notebook file size down) or export a high-resolution PNG (needs `kaleido`).

Marker and line-style strings are **Matplotlib-compatible** (`'o'`, `'s'`,
`'^'`, `'--'`, `'-.'`, `':'`) and translated to Plotly automatically.

### Decorations

```python
nb.line('rpm', level=5000, color='red', linestyle='--')  # reference line
nb.highlight('time', (10, 20), color='yellow', alpha=0.2) # shaded band
nb.scale('pressure', (0, 100))                            # fix an axis range
nb.suptitle = 'Overview'; nb.footer = 'source: rig A'      # figure text
```

---

## Analysis & stats

- **`delta(base_idx, study_indices, delta_parms=...)`** — compute absolute
  (`DL_<P>`) and percentage (`DLPCT_<P>`) differences of each study dataset
  against a baseline, aligned by nearest-match merge on a chosen column (with
  optional interpolation at specified `x_ins`). The delta set inherits the study
  set's color/marker so it reads as a continuation of that series.
- **`table(...)` / `table_read(...)`** — tabulate columns, or interpolate a
  Y column at arbitrary X inputs (`kind='linear'`, extrapolation controllable).
- **`reg_info(...)`** / `reg_order` — fit and report regressions / trend lines
  (polynomial or LOWESS; LOWESS needs `statsmodels`).
- **`summary(cols=...)`** — per-dataset descriptive statistics.
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

- `method` — one of `plot`, `plot_ymult`, `bar`, `box`, `histogram`,
  `contour`, `table` (default `plot`).
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
  - `contour_overlay_demo.ipynb`, `static_images_demo.ipynb`,
    `large_data_showcase.ipynb` — specialized plotting.

## Usage

```python
from unichart import UnichartNotebook
from unichart_dashboard import dashboard   # optional, needs `dash`

nb = UnichartNotebook()
```
