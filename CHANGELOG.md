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
