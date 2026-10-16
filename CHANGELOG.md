# Changelog

## 0.1.0 — unreleased

First release on PyPI.

- `UnichartNotebook`: stateful, multi-dataset Plotly plotting for Jupyter.
- `unichart.dashboard`: Dash boards and self-contained HTML from notebook panels.
- `unichart.terminal`: the `explore()` GUI — sidebar, chart pane and Python terminal.
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
