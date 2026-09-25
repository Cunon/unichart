# Changelog

## 0.1.0 — unreleased

First release on PyPI.

- `UnichartNotebook`: stateful, multi-dataset Plotly plotting for Jupyter.
- `unichart.dashboard`: Dash boards and self-contained HTML from notebook panels.
- `unichart.terminal`: the `explore()` GUI — sidebar, chart pane and Python terminal.
- `unichart` command (also `python -m unichart`): open the explorer on a data or
  session file, or print `--info` / write `--html` without serving.
- Sessions save to JSON or embed in PNGs, and restore with `load_session`.
