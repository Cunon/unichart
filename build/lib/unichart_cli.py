"""Command line for unichart: open the explorer GUI, or export a board.

The default action is the GUI — ``unichart runs.csv`` loads the file and opens
the explorer in a browser, so a quick look at a data file never needs a Python
session::

    unichart                              # empty explorer; load from the data bar
    unichart runs.csv                     # open the explorer on one file
    unichart a.csv b.csv --combine        # several files, merged into one set
    unichart runs.csv --info              # print what's in it and exit
    unichart runs.csv --html board.html   # write a static board instead of serving

Panels are optional. For the GUI each ``--panel`` is replayed as a terminal
command at startup, so the board opens with those plots already drawn and the
commands visible in the transcript; for ``--html`` each one becomes a card.

This module is also runnable directly (``python -m unichart_cli ...``) for when
the installed console script isn't on PATH.
"""

import argparse
import sys

# unichart / unichart_dashboard are imported lazily inside main() so that
# `unichart --help` and `--version` stay instant and work even if the optional
# Dash dependency is missing.

__version__ = '0.1.0'

# One --panel is `method:x:y1,y2[:z]`. Everything the panel spec dict supports
# beyond that (kwargs like nbins / barmode / overlay_sets, dataset pins) is
# deliberately not expressible here — those belong in Python, where the dict
# form is clearer than any flag encoding would be.
PANEL_HELP = (
    "add a panel, as method:x:y[,y2][:z] (repeatable). method is one of "
    "plot, plot_ymult, bar, box, histogram, contour, table. "
    "Examples: plot:time:temp  |  plot:time:temp,press  |  "
    "contour:rpm:torque:eff  |  histogram:temp"
)


class CliError(Exception):
    """A user-facing error: reported as one line, never a traceback."""


def parse_panel(spec, methods):
    """Parse one ``method:x:y1,y2[:z]`` panel spec into a panel dict.

    Only ``method`` is required, so ``histogram:temp`` (which needs just an x)
    and a bare ``table`` both work.
    """
    parts = spec.split(':')
    method = parts[0].strip()
    if not method:
        raise CliError(f"--panel {spec!r}: no plot method given")
    if method not in methods:
        raise CliError(f"--panel {spec!r}: unknown method {method!r} — "
                       f"choose from {', '.join(methods)}")
    if len(parts) > 4:
        raise CliError(f"--panel {spec!r}: too many ':' fields — expected "
                       f"method:x:y[,y2][:z]")

    panel = {'method': method}
    x = parts[1].strip() if len(parts) > 1 else ''
    if x:
        panel['x'] = x
    if len(parts) > 2:
        ys = [y.strip() for y in parts[2].split(',') if y.strip()]
        if ys:
            panel['y'] = ys
    z = parts[3].strip() if len(parts) > 3 else ''
    if z:
        panel['z'] = z
    return panel


def build_parser():
    p = argparse.ArgumentParser(
        prog='unichart',
        description='Open the unichart explorer on a data file, or export a '
                    'static board.',
        epilog='Panels beyond method/x/y/z (kwargs, dataset pins) are '
               'expressible from Python: see nb.dashboard / nb.explore.',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('files', nargs='*', metavar='FILE',
                   help='data file(s) to load: .csv .tsv .txt .xlsx .xls '
                        '.json .parquet. Omit to open an empty explorer.')
    p.add_argument('--version', action='version',
                   version=f'unichart {__version__}')

    load = p.add_argument_group('loading')
    load.add_argument('--combine', action='store_true',
                      help='merge every FILE into one dataset instead of '
                           'loading each separately')
    load.add_argument('--set-col', metavar='COL', dest='set_col',
                      help='split the data into one dataset per unique value '
                           'of this column')
    load.add_argument('--name-col', metavar='COL', dest='name_col',
                      help='take dataset names from this column')

    board = p.add_argument_group('board')
    board.add_argument('--panel', action='append', default=[], metavar='SPEC',
                       help=PANEL_HELP)
    board.add_argument('--ncols', type=int, default=2, metavar='N',
                       help='--html only: panels per row (default: 2)')
    board.add_argument('--width', type=int, default=600, metavar='PX',
                       help='--html only: panel width in px (default: 600)')
    board.add_argument('--height', type=int, default=420, metavar='PX',
                       help='--html only: panel height in px (default: 420)')
    board.add_argument('--title', metavar='TEXT', help='board title')
    board.add_argument('--dark', action='store_true',
                       help='--html only: export in dark mode (the terminal '
                            'board is always dark)')

    out = p.add_argument_group('output')
    out.add_argument('--info', action='store_true',
                     help='print the datasets and columns, then exit')
    out.add_argument('--html', metavar='PATH',
                     help='write a self-contained static HTML board to PATH '
                          'instead of serving the explorer')
    out.add_argument('--embed-js', default='cdn',
                     choices=['cdn', 'inline', 'directory'], dest='embed_js',
                     help="--html only: how to include plotly.js. 'cdn' "
                          "(default, small, needs internet), 'inline' (fully "
                          "offline), 'directory'")
    out.add_argument('--port', type=int, default=8050, metavar='N',
                     help='preferred port (default: 8050; a free one is '
                          'chosen if it is busy)')
    out.add_argument('--no-browser', action='store_true', dest='no_browser',
                     help="serve, but don't open a browser (headless / remote)")
    return p


def _load(nb, args):
    """Load every FILE onto the notebook, reporting missing paths up front."""
    from pathlib import Path

    missing = [f for f in args.files if not Path(f).exists()]
    if missing:
        raise CliError(f"no such file: {', '.join(missing)}")
    if not args.files:
        return
    sources = args.files if len(args.files) > 1 else args.files[0]
    try:
        nb.load(sources, set_idx_column=args.set_col,
                set_name_column=args.name_col, combined=args.combine)
    except Exception as exc:                              # noqa: BLE001
        raise CliError(f'could not read the data: {exc}') from exc


def _print_info(nb):
    """Summarize what got loaded — the non-GUI way to check a file parsed."""
    from unichart_dashboard import _all_columns, _numeric_columns

    if not nb.sets:
        print('No datasets loaded.')
        return
    print(f'\n{len(nb.sets)} dataset(s):')
    for ds in nb.sets:
        print(f'  [{ds.index}] {ds.title_format} — {len(ds.df):,} rows')
    numeric = set(_numeric_columns(nb))
    print(f'\n{len(_all_columns(nb))} column(s) '
          f'({len(numeric)} numeric, marked *):')
    for name in _all_columns(nb):
        print(f"  {'*' if name in numeric else ' '} {name}")


def main(argv=None):
    """Entry point. Returns a process exit code rather than raising."""
    args = build_parser().parse_args(argv)

    try:
        from unichart import UnichartNotebook
        from unichart_dashboard import (PLOT_METHODS, _default_panel_spec,
                                        explore, to_html)
    except ImportError as exc:
        print(f'unichart: {exc}', file=sys.stderr)
        return 1

    try:
        panels = [parse_panel(spec, PLOT_METHODS) for spec in args.panel]

        nb = UnichartNotebook()
        _load(nb, args)
        if args.dark:
            nb.toggle_darkmode(True)

        if args.info:
            _print_info(nb)
            return 0

        if args.html:
            if not nb.sets:
                raise CliError('--html needs data: pass at least one FILE')
            # to_html rejects an empty panel list, and a board with no panels
            # would be a blank page anyway; fall back to the same auto-seeded
            # plot the explorer opens with.
            to_html(nb, panels or [_default_panel_spec(nb)], args.html,
                    ncols=args.ncols, width=args.width, height=args.height,
                    title=args.title, embed_js=args.embed_js)
            print(f'Wrote {args.html}')
            return 0

        # Default: serve the explorer. This blocks until interrupted.
        # ncols/width/height size a static --html grid; the terminal board has
        # one chart pane that fills its own space, so they are not forwarded.
        explore(nb, panels=panels or None, title=args.title, port=args.port,
                open_browser=not args.no_browser)
        return 0

    except ImportError as exc:
        # Dash is an optional extra. --info and --html don't need it, so this
        # only fires on the serving path — report it as one actionable line
        # rather than a traceback out of _require_dash.
        print(f'unichart: {exc}\n'
              '  the explorer needs Dash:  pip install dash',
              file=sys.stderr)
        return 1
    except CliError as exc:
        print(f'unichart: {exc}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == '__main__':
    sys.exit(main())
