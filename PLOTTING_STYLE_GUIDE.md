# `unichart` Plotting API — Style Guide

How to add a new plotting capability to `unichart.py` so its inputs and behavior
match every existing plot. Read this before copy-pasting an existing method:
the existing methods have drifted in a few places, and copying the wrong one
propagates the drift. Known drift is catalogued in [§7](#7-standardization-recommendations).

All line references are to `unichart.py` as of this writing — treat them as
"look near here," not exact addresses, since the file changes.

---

## 1. Architecture: two layers

Every plot is split into a **pure render function** and a **notebook dispatch
method**. Keep them separate; do not put notebook state into a render function or
plotting logic into a method.

### Layer A — module-level render functions (pure)
- Take an explicit `list_of_datasets` plus explicit arguments. **Hold no notebook
  state** — no `self`, no reading of `self.axis_limits`, etc.
- Return a `go.Figure` through `_show_or_return(fig, return_axes)` (`:165`) — the
  single exit point for every render function.
- Naming convention encodes the layout:

  | Suffix | Layout | Example |
  |--------|--------|---------|
  | `uniXXX` | base — one subplot per Y variable | `uniplot`, `unibar`, `unibox` |
  | `uniXXX_per_dataset` | one subplot per dataset | `unibar_per_dataset` |
  | `uniXXX_datasets_as_x` | datasets on the X axis, variables as series | `unibar_datasets_as_x` |

### Layer B — `UnichartNotebook` methods (thin dispatchers)
- Pull state off `self` (`suptitle`, `darkmode`, `axis_limits`,
  `variable_formats`, `display_parms`, `lines`, `highlights`, fonts).
- Choose the right render function based on `by`.
- Post-process (decorations, fonts, legends) and store `self.last_fig`.
- Public methods: `plot`, `plot_ymult`, `bar`, `box`, `histogram`, `contour`.

---

## 2. Canonical signature order

Match the argument **order and names** below. Plot-specific arguments slot into the
`<plot-specific>` position; everything else is fixed so callers build muscle memory.

### Render function
```python
def uniXXX(list_of_datasets, x, y, [z,]
           <plot-specific>,                 # e.g. barmode, points, histfunc
           variable_formats=None, color=None,
           suptitle=None, xlabel=None, ylabel=None, subplot_titles=None,
           darkmode=False, figsize=(12, 8), ncols=None, nrows=None,
           x_lim=None, y_lim=None, axis_limits=None, return_axes=False):
```
`unibar` (`:1215`) is the reference shape. The trailing layout block
(`suptitle … return_axes`) is identical across functions — copy it verbatim.

### Notebook method
```python
def XXX(self, x=None, y=None, [z=None,] [markers=None,]
        by='vars',
        <plot-specific>,
        figsize=(12, 8), ncols=None, nrows=None,
        suptitle=None, suppress_legends=False):
```
`x`/`y`/`z` default to `None` and fall back to `self.last_*`. `figsize`,
`ncols`, `nrows`, `suptitle`, `suppress_legends` are the common tail.

---

## 3. The `by` vocabulary (fixed values)

`by` selects the layout. Use exactly these strings:

| `by` value | Dispatches to | Meaning |
|------------|---------------|---------|
| `'vars'` *(default)* | `uniXXX` | one subplot per Y variable |
| `'sets'` / `'datasets'` | `uniXXX_per_dataset` | one subplot per dataset (synonyms) |
| `'dataset_x'` | `uniXXX_datasets_as_x` | datasets on X, variables as series (bar/box) |
| `'ymult'` | `plot_ymult` | single plot, multiple Y axes (line only) |

Branch on the per-dataset case with `if by in ['sets', 'datasets']:` — both spellings
must work (see `:4291`, `:4439`, `:4502`). Do not invent new `by` values without
adding the matching `uniXXX_<layout>` render function.

---

## 4. Standard method-body skeleton

Every notebook method follows this contract. Deviate only with a comment saying why.

```python
def XXX(self, x=None, y=None, by='vars', ..., suptitle=None, suppress_legends=False):
    self._clear_last_fig()                          # 1. reset cached figure  (:5236)

    if x is None: x = self.last_x                   # 2. resolve + remember inputs
    if y is None: y = self.last_y
    self.last_x, self.last_y = x, y

    y_list = y if isinstance(y, list) else [y]       # 3. normalize, read axis_limits

    if by in ['sets', 'datasets']:                  # 4. dispatch, threading state
        fig = uniXXX_per_dataset(
            list_of_datasets=self.sets, x=x, y=y, ...,
            variable_formats=self.variable_formats,  #    (where the render fn supports it)
            suptitle=suptitle or self.suptitle,
            darkmode=self.darkmode,
            axis_limits=self.axis_limits,
            display_parms=self.display_parms,
            return_axes=True,
        )
    else:
        fig = uniXXX(... same state passthrough ...)

    if fig:                                          # 5. post-process
        calc_ncols = max(1, _calc_grid(n_items, nrows, ncols)[1])
        fig = self._apply_decorations(fig, x_list, y_list, mode, calc_ncols, plot_items)
        # ... apply axis-limit ranges per subplot (per-plot-type) ...
        fig = self._finalize(fig, suppress_legends)  # fonts + suppress + last_fig  (:4058)

    return fig                                        # 6. always return the figure
```

State passthrough rules:
- `suptitle=suptitle or self.suptitle` — explicit arg wins, else notebook default.
- `darkmode=self.darkmode`, `axis_limits=self.axis_limits`, `return_axes=True` —
  always.
- `variable_formats=self.variable_formats` — pass when the render function accepts it.
- `display_parms=self.display_parms` — for line/hover-bearing plots.

---

## 5. Helpers to reuse (don't reinvent)

| Helper | Location | Use for |
|--------|----------|---------|
| `_calc_grid(n, nrows, ncols)` | `:124` | rows/cols for a subplot grid |
| `_base_layout(darkmode, suptitle, figsize, **extra)` | `:134` | template, title, margin, size |
| `_build_xbins(size, start, end)` | `:158` | histogram bin spec |
| `_show_or_return(fig, return_axes)` | `:165` | the single render-function exit |
| `_subplot_refs(row, col, ncols)` | `:171` | `(xref, yref)` for shapes |
| `_resolve_var_format(dataset, variable, variable_formats)` | `:183` | per-(dataset, variable) style |
| `_scatter_cls(n_points)` | `:217` | SVG vs WebGL scatter class |
| `_get_uset_slice(selector)` | `:2788` | resolve a selector to a dataset list |
| `_apply_decorations` | `:5243` | draw stored `lines` / `highlights` |
| `_apply_fonts(fig)` | `:4012` | apply font-size settings |
| `_finalize(fig, suppress_legends)` | `:4058` | shared tail: fonts + suppress + cache `last_fig` |
| `_clear_last_fig()` | `:5236` | reset cached figure at method start |

If you need grid math or layout scaffolding and one of these fits, call it — do not
inline a second copy.

---

## 6. Naming & convention rules

- **Selector args end in `_sets`.** A user-facing argument that picks datasets
  (`overlay_sets`, `uset_slice`, …) accepts the selector vocabulary — `int` / `list`
  / `'all'` / a `Dataset`. Resolve it with `_get_uset_slice` to a concrete list
  *before* handing it to a render function, where the parameter is named
  `_datasets` (e.g. `overlay_sets` → `overlay_datasets`, `:4496`).
- **Opacity is `alpha`.** New code uses `alpha`. `opacity` survives only as a
  deprecated alias that warns and copies into `alpha` (pattern at `:1496`, `:4424`).
  Do not add a fresh `opacity` parameter.
- **`return_axes` lives on render functions only.** Render functions expose it;
  notebook methods always call them with `return_axes=True` and themselves return
  the figure (the notebook never calls `fig.show()` — Jupyter's repr displays it).
- **Per-variable formatting goes through `_resolve_var_format`** so
  `variable_formats[var]` overrides the dataset attribute per-attribute. Don't read
  `dataset.color` directly when a `variable_formats` override could apply.

---

## 7. Standardization recommendations

Real inconsistencies found in the current code. Except where marked **DONE**, these
are recommendations not yet applied — listed so new methods follow the *intended*
convention and so a future cleanup has a checklist. Priority high → low.

1. ~~**Add `suptitle=None` to `bar()`.**~~ **DONE.** `bar` now accepts `suptitle`
   (placed before `figsize`, matching `box`/`histogram`) and threads
   `suptitle or self.suptitle` to all three render branches. New methods must take
   `suptitle`.

2. ~~**Normalize `darkmode` default to `False`.**~~ **DONE.** `uniplot_per_dataset`
   (`:1076`) was the lone render function defaulting `darkmode=True`; now `False`
   like all the others. Behavior-preserving (the notebook always passes
   `self.darkmode`); only fixes the surprise for direct callers. New render
   functions default `darkmode=False`.

3. **Thread `variable_formats` through `uniplot` / `uniplot_per_dataset`**
   (`:859`, `:1076`). Today per-variable color is honored only by `plot_ymult`,
   bar/box `by='dataset_x'`, and bar marker overlays — **not** by ordinary
   line/scatter plots. Wiring it in (via `_resolve_var_format`, already used by
   `uniplot_ymultaxis`) would make variable color universal and remove a confusing
   gap. *Recommended direction.*

4. ~~**Expose `color=None` on `bar()`**~~ **DONE.** `bar` now takes `color` and
   threads it to the `by='vars'` branch (`unibar`, which already accepts it),
   mirroring `box` exactly — the `by='sets'` / `by='dataset_x'` branches color per
   dataset / per variable and ignore `color`, as their render functions take no
   `color` arg.

5. ~~**Extract a shared finalize helper.**~~ **DONE (scoped down).** A close read
   showed the full block could *not* be one helper without flattening real
   per-plot-type differences (each passes different `x_vars`/`y_vars`/`plot_items`
   to `_apply_decorations`; axis-limit strategy differs four ways; only `plot` adds
   a legend/margin update). So two genuinely-safe extractions were applied instead,
   verified behavior-preserving by a full `fig.to_json()` snapshot diff across all
   methods/modes:
   - **`_finalize(fig, suppress_legends)`** — the identical tail (`_apply_fonts` →
     `suppress_legends` → `self.last_fig`), now used by every grid plotting method
     (`plot`, `plot_ymult`, `bar`, `box`, `histogram`, `contour`). This is where the
     `last_fig` cache contract lives.
   - **`_calc_grid` reuse** — every hand-reimplemented grid recomputation (across
     `plot`, `bar`, `box`, `histogram`, `contour`) now calls the existing
     `_calc_grid` helper (`:124`); the only remaining `np.sqrt` grid math is inside
     `_calc_grid` itself.

   Decorations and axis-limit ranges deliberately **stay in each method** — they are
   per-plot-type semantics, not boilerplate. A new method's post-processing is:
   compute `calc_ncols = max(1, _calc_grid(n, nrows, ncols)[1])`, call
   `_apply_decorations`, apply any axis-limit ranges, then `return self._finalize(fig, suppress_legends)`.

---

## 8. Worked template

A complete, convention-correct skeleton for a hypothetical violin plot. Adapt the
plot-specific bits; keep everything else.

### Render functions (Layer A)
```python
def univiolin(list_of_datasets, x, y, points='outliers',
              variable_formats=None, color=None,
              suptitle=None, xlabel=None, ylabel=None, subplot_titles=None,
              darkmode=False, figsize=(12, 8), ncols=None, nrows=None,
              y_lim=None, return_axes=False):
    y_list = y if isinstance(y, list) else [y]
    nrows, ncols = _calc_grid(len(y_list), nrows, ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles or y_list)
    fig.update_layout(**_base_layout(
        darkmode, suptitle or f"Violin: {x}", figsize, showlegend=True,
    ))

    for ds in list_of_datasets:
        if not ds.select:
            continue
        df = ds.cols([c for c in dict.fromkeys([x] + y_list) if c in ds.columns])
        for idx_y, yi in enumerate(y_list):
            if yi not in df.columns:
                continue
            row, col = (idx_y // ncols) + 1, (idx_y % ncols) + 1
            fig.add_trace(go.Violin(
                x=df[x], y=df[yi], name=f"{ds.index}: {ds.title}",
                marker_color=color or ds.color, opacity=ds.alpha, points=points,
            ), row=row, col=col)

    fig.update_xaxes(title_text=xlabel or x)
    fig.update_yaxes(title_text=ylabel or "Value")
    if y_lim:
        fig.update_yaxes(range=y_lim)
    return _show_or_return(fig, return_axes)


def univiolin_per_dataset(list_of_datasets, x, y, points='outliers',
                          variable_formats=None,
                          suptitle=None, figsize=(12, 8), ncols=None, nrows=None,
                          darkmode=False, y_lim=None, return_axes=False):
    ...  # one subplot per dataset; same trailing block, return via _show_or_return
```

### Notebook method (Layer B)
```python
def violin(self, x=None, y=None, by='vars', points='outliers',
           figsize=(12, 8), ncols=None, nrows=None,
           suptitle=None, suppress_legends=False):
    """Unified interface for Violin Plots."""
    self._clear_last_fig()

    if x is None: x = self.last_x
    if y is None: y = self.last_y
    self.last_x, self.last_y = x, y

    y_list = y if isinstance(y, list) else [y]

    if by in ['sets', 'datasets']:
        fig = univiolin_per_dataset(
            list_of_datasets=self.sets, x=x, y=y, points=points,
            variable_formats=self.variable_formats,
            suptitle=suptitle or self.suptitle, figsize=figsize,
            ncols=ncols, nrows=nrows, darkmode=self.darkmode, return_axes=True,
        )
        mode = 'sets'
    else:
        fig = univiolin(
            list_of_datasets=self.sets, x=x, y=y, points=points,
            variable_formats=self.variable_formats,
            suptitle=suptitle or self.suptitle, figsize=figsize,
            ncols=ncols, nrows=nrows, darkmode=self.darkmode, return_axes=True,
        )
        mode = 'vars'

    if fig:
        n_items = len([d for d in self.sets if d.select]) if mode == 'sets' else len(y_list)
        calc_ncols = max(1, _calc_grid(n_items, nrows, ncols)[1])
        plot_items = [(x, yi) for yi in y_list] if mode == 'vars' else None
        fig = self._apply_decorations(fig, [x], y_list, mode, calc_ncols, plot_items)
        fig = self._finalize(fig, suppress_legends)   # fonts + suppress + last_fig

    return fig
```

This skeleton already obeys: the canonical signature order (§2), the `by`
vocabulary (§3), the body contract (§4), helper reuse (§5), and the naming rules
(§6) — including taking `suptitle` as an argument per recommendation §7.1.
