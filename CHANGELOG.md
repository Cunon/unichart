# Changelog

## 0.1.0 — unreleased

First release on PyPI.

- `UnichartNotebook`: stateful, multi-dataset Plotly plotting for Jupyter.
- `unichart.dashboard`: Dash boards and self-contained HTML from notebook panels.
- `unichart.terminal`: the `explore()` GUI — sidebar, chart pane and Python terminal.
  Its example datasets and cheat sheet only show when the notebook starts
  empty; `explore(cheat_sheet=True)` brings them back. Its sidebar lists the
  sets and the parameters in searchable tables; clicking a name puts it in the
  terminal.
- `unichart` command (also `python -m unichart`): open the explorer on a data or
  session file, or print `--info` / write `--html` without serving.
- Sessions save to JSON or embed in PNGs, and restore with `load_session`.
- `plot_defaults()` and its siblings (`bar_defaults()`, `table_defaults()`,
  `save_png_defaults()`, ...) store default arguments for their method; a later
  call uses them for whatever it leaves out. `uc.plot_defaults` used to be the
  dict `set_default_format` writes, which is now internal (older sessions still
  load).
- `legend_entry()` adds custom legend entries that no dataset draws, with your
  own color, marker and line style. Call it again by name to edit an entry, or
  with `'reset'` to remove it. The entries are saved with the session and
  cleared by `reset_format('legend_entries')`.
- `polar()` draws on polar axes, with `x` as the angle and `y` as the radius,
  styled like `plot`. Given a `z`, it draws a polar contour map instead,
  interpolated all the way round the circle, or over the sector the data
  covers. `scale`, `line`, `highlight` and `grid` work on it as angle and
  radius. It has a `polar_defaults()` sister, and works as a dashboard panel and
  a CLI `--panel` method. The explorer has a new "Inlet survey" example for it,
  and the gallery an "In the round" section.
- Subplot titles no longer land on the legend. The top row's titles get room
  of their own above the plot, sized from their font and line count, and the
  above-legend is sized by its measured height. Without an above-legend the
  titles clear the suptitle instead. `title_gap` and `legend_gap` (px, default
  12) set the space under the suptitle and under the legend:
  `uc.legend_gap = 24`, `set_default_format(title_gap=, legend_gap=)` or
  `new_uc(...)`. See `demo_notebooks/subplot_title_spacing_demo.ipynb`.
- `markersize(..., 0)` now turns that set's (or variable's) markers off, as
  `marker('none')` does. Before, Plotly still drew each size-0 marker's outline.
- `bar(..., overlay_sets=)` compares bars against other datasets' values of
  the same variable: the chosen sets draw one marker per category (the `agg`
  of their own rows) in their own color and symbol instead of bars, selected
  or not. In `by='sets'` they appear on every panel over the matching bar.
  A dict pairs them with particular bar sets instead,
  `overlay_sets={0: 1, 3: [2, 4]}`: set 1's markers sit over set 0's bars and
  sets 2 and 4 over set 3's (in `by='sets'`, on those sets' panels only).
  The markers follow the `y` variable's `var_format` `style`, so
  `var_format('EGT', style='whisker')` draws each overlay set as a dash with a
  stem to the bar it sits on (`linestyle` / `linewidth` style the stem),
  still in the set's own color.
  `by='dataset_x'` ignores them with a warning.
