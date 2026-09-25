"""Command line for unichart: open the explorer GUI, or export a board.

The default action is the GUI — ``unichart runs.csv`` loads the file and opens
the explorer in a browser, so a quick look at a data file never needs a Python
session::

    unichart                              # empty explorer; load from the data bar
    unichart runs.csv                     # open the explorer on one file
    unichart runs.csv --app               # ...in its own window, like an app
    unichart a.csv b.csv --combine        # several files, merged into one set
    unichart runs.csv --info              # print what's in it and exit
    unichart runs.csv --html board.html   # write a static board instead of serving
    unichart --gallery                    # open the example gallery and exit

A FILE can also be a saved session — a ``.json`` from ``nb.save_session`` or a
PNG from ``nb.save_png``, which carries its session in a metadata chunk. Those
are restored rather than read as data, bringing back the datasets, the queries,
the formatting and the plot itself::

    unichart plot.png                     # reopen the plot that PNG came from
    unichart board.json --info            # inspect a session without serving
    unichart runs.csv --panel plot:time:temp --save-session board.json

Tab completion is built in — ``eval "$(unichart --completion bash)"`` (or
``zsh``) in your shell rc file completes the flags, and completes ``--panel``
method / x / y / z fields against the real column names of the data files
already on the command line.

The explorer serves until you close it: the board's ✕ close button (or
``exit()`` in its terminal pane) stops the server and returns the shell, and
Ctrl-C here does the same.

Panels are optional. For the GUI each ``--panel`` is replayed as a terminal
command at startup, so the board opens with those plots already drawn and the
commands visible in the transcript; for ``--html`` each one becomes a card.

This module is also runnable directly (``python -m unichart ...``) for when
the installed console script isn't on PATH.
"""

import argparse
import os
import re
import sys

# unichart._core / unichart.dashboard are imported lazily inside main() so that
# `unichart --help` and `--version` stay instant (pandas + Plotly + Dash take
# about a second to import).

from . import __version__

# The example gallery is a built page in the source tree (gallery/index.html).
# A pip install does not carry it — the wheel ships no data files — so
# --gallery falls back to pointing at the repository.
GALLERY_URL = 'https://github.com/Cunon/unichart/tree/main/gallery'

# One --panel is `method:x:y1,y2[:z]`. Everything the panel spec dict supports
# beyond that (kwargs like nbins / barmode / overlay_sets, dataset pins) is
# deliberately not expressible here — those belong in Python, where the dict
# form is clearer than any flag encoding would be.
PANEL_HELP = (
    "add a panel, as method:x:y[,y2][:z] (repeatable). method is one of "
    "plot, plot_ymult, plot_marginal, bar, box, histogram, contour, table. "
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


# ---------------------------------------------------------------------------
# Shell completion
# ---------------------------------------------------------------------------
#
# Hand-rolled rather than argcomplete, for two reasons: no extra dependency to
# install before the first tab press works, and --panel needs colon-field
# completion that a generic completer can't express.
#
# The shell hands us the whole line (COMP_LINE/COMP_POINT) and we hand back
# candidates, so all of the logic lives here in Python where it can be tested
# without a shell.

# Bash breaks the word under the cursor on these (its default COMP_WORDBREAKS),
# so candidates must be returned as the tail after the last one — otherwise
# bash re-prefixes what it already has and you get `plot:time:plot:time:temp`.
_WORDBREAKS = ':='

# The panel methods, spelled out rather than imported: unichart.dashboard pulls
# in Dash, which is both slow on every tab press and missing entirely for the
# from-scratch user who hasn't installed the extra. Kept honest by a check
# against PLOT_METHODS in the test suite.
COMPLETION_METHODS = ('plot', 'plot_ymult', 'plot_marginal', 'bar', 'box',
                      'histogram', 'contour', 'table')

_DATA_SUFFIXES = ('.csv', '.tsv', '.txt', '.xlsx', '.xls', '.json', '.parquet')

# Extensions a saved session can have, as a cheap pre-filter before the real
# content sniff. Deliberately *not* folded into _DATA_SUFFIXES: that tuple feeds
# _header_columns, and a tab press must never try to read a PNG as a table.
_SESSION_SUFFIXES = ('.json', '.png')


def _takes_value(action):
    """True for options that consume a following value (not store_true etc.)."""
    return action.nargs != 0


def _option_actions():
    """``{flag: action}`` for every option the parser defines."""
    return {flag: action for action in build_parser()._actions
            for flag in action.option_strings}


def _split_line(line, point):
    """Split the line at the cursor into (words before, word being typed)."""
    head = line[:point]
    words = head.split()
    if not words:
        return [], ''
    if head[-1:].isspace():
        return words, ''
    return words[:-1], words[-1]


def _match(items, partial):
    """Prefix-match ``items`` case-insensitively, order-preserving and deduped."""
    low = partial.lower()
    seen, out = set(), []
    for item in items:
        text = str(item)
        if text.lower().startswith(low) and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _trim(word, candidates):
    """Cut each candidate down to what the shell will actually substitute."""
    cut = max(word.rfind(ch) for ch in _WORDBREAKS)
    return [c[cut + 1:] for c in candidates] if cut >= 0 else list(candidates)


def _header_columns(path):
    """Column names of one data file, read as cheaply as the format allows.

    Header-only reads (``nrows=0``): a tab press must not pay for a 2 GB csv.
    pandas is imported here, not at module scope, so completing a flag or a
    method name stays a pure-stdlib round trip. Any failure — unreadable file,
    missing optional engine, half-typed path — completes nothing rather than
    spilling a traceback into the shell's candidate list.
    """
    suffix = os.path.splitext(path)[1].lower()
    try:
        import pandas as pd

        if suffix == '.csv':
            frame = pd.read_csv(path, nrows=0)
        elif suffix in ('.tsv', '.txt'):
            frame = pd.read_csv(path, sep='\t', nrows=0)
        elif suffix in ('.xlsx', '.xls'):
            frame = pd.read_excel(path, nrows=0)
        elif suffix == '.parquet':
            frame = pd.read_parquet(path)[:0]
        else:
            # .json has no cheap header read — not worth parsing the file.
            return []
        return [str(c) for c in frame.columns]
    except Exception:                                     # noqa: BLE001
        return []


def _columns_on_line(words):
    """Columns available from the data files already typed on the line."""
    options = _option_actions()
    columns, skip = [], False
    for word in words[1:]:
        if skip:
            skip = False
            continue
        if word.startswith('-'):
            action = options.get(word)
            skip = action is not None and _takes_value(action)
            continue
        if word.lower().endswith(_DATA_SUFFIXES) and os.path.isfile(word):
            columns.extend(_header_columns(word))
    # Header names only: these are exactly the columns the loaded datasets end
    # up with (unichart's synthesized TITLE/SETNUMBER only appear on loads the
    # CLI doesn't make), so nothing is offered that the board can't plot.
    return columns


def _complete_panel(spec, columns):
    """Complete one ``method:x:y[,y2][:z]`` spec, field by field."""
    fields = spec.split(':')
    head, last = fields[:-1], fields[-1]

    if len(fields) == 1:
        # A trailing ':' on every method but `table` (the one spec that is
        # complete on its own) says "keep typing" — paired with -o nospace.
        return [m if m == 'table' else m + ':'
                for m in COMPLETION_METHODS if m.startswith(last)]
    if len(fields) == 2:                                   # x
        values = _match(columns, last)
    elif len(fields) == 3:                                 # y, possibly a list
        done, sep, partial = last.rpartition(',')
        values = [done + sep + c for c in _match(columns, partial)]
    elif len(fields) == 4:                                 # z — contour only
        values = _match(columns, last) if head[0] == 'contour' else []
    else:
        return []
    return [':'.join(head + [value]) for value in values]


def _option_values(action, words, partial):
    """Candidate values for ``--flag <here>``."""
    if action.choices:
        return _match([str(c) for c in action.choices], partial)
    if action.dest in ('set_col', 'name_col'):
        return _match(_columns_on_line(words), partial)
    if action.dest == 'panel':
        return _complete_panel(partial, _columns_on_line(words))
    # Paths (FILE, --html) and free text (--title): returning nothing lets the
    # shell fall back to its own filename completion, which handles spaces,
    # directories and ~ far better than we would.
    return []


def complete(line, point=None):
    """Candidates for the word under the cursor in ``line``.

    Returned already trimmed to what the shell will substitute; the caller
    prints them one per line.
    """
    point = len(line) if point is None else point
    words, word = _split_line(line, point)
    options = _option_actions()

    if word.startswith('-') and '=' in word:               # --flag=value form
        flag, _, partial = word.partition('=')
        action = options.get(flag)
        if action is not None and _takes_value(action):
            return _trim(word, [f'{flag}={v}'
                                for v in _option_values(action, words, partial)])
        return []

    previous = options.get(words[-1] if words else '')
    if previous is not None and _takes_value(previous):
        return _trim(word, _option_values(previous, words, word))

    if word.startswith('-'):
        return _trim(word, sorted(f for f in options if f.startswith(word)))

    # A bare word is a FILE: leave it to the shell's filename completion.
    return []


# Raw: the `$'\n'` IFS and the line continuation are bash syntax, not
# Python escapes.
_COMPLETION_FUNCTION = r"""_unichart_completion() {
    local IFS=$'\n' candidate nospace=1
    COMPREPLY=($(COMP_LINE="$COMP_LINE" COMP_POINT="$COMP_POINT" \
                 unichart --_complete 2>/dev/null))
    [ ${#COMPREPLY[@]} -eq 0 ] && return 1
    for candidate in "${COMPREPLY[@]}"; do
        case "$candidate" in *:) ;; *) nospace=0 ;; esac
    done
    [ $nospace -eq 1 ] && compopt -o nospace 2>/dev/null
    return 0
}
complete -o default -o bashdefault -F _unichart_completion unichart"""

COMPLETION_SCRIPTS = {
    'bash': f"""# unichart bash completion.
# Install: add this line to ~/.bashrc
#     eval "$(unichart --completion bash)"
{_COMPLETION_FUNCTION}
""",
    'zsh': f"""# unichart zsh completion.
# Install: add this line to ~/.zshrc
#     eval "$(unichart --completion zsh)"
# bashcompinit lets zsh run the bash-style function below. compinit is its
# prerequisite, so run it only if the rc file hasn't already.
if (( ! $+functions[compdef] )); then autoload -U +X compinit && compinit -u; fi
autoload -U +X bashcompinit && bashcompinit
{_COMPLETION_FUNCTION}
""",
}


def _run_completion(rest):
    """The ``--_complete`` hook the shell scripts call. Never fails loudly."""
    try:
        line = rest[0] if rest else os.environ.get('COMP_LINE', '')
        if len(rest) > 1:
            point = int(rest[1])
        else:
            point = int(os.environ.get('COMP_POINT') or len(line))
        for candidate in complete(line, point):
            print(candidate)
    except Exception:                                     # noqa: BLE001
        pass                       # a tab press must never print an error
    return 0


# ---------------------------------------------------------------------------
# Colored help
# ---------------------------------------------------------------------------
#
# The coloring is a post-pass over the text argparse has already laid out, not
# an override of _format_action_invocation: argparse sizes its help column with
# len() of the invocation string, so escape codes added *during* formatting
# inflate that width and knock every help column out of alignment. Painting the
# finished string keeps the layout byte-identical to the uncolored output.

_ANSI = {
    'bold': '\033[1m',
    'head': '\033[1;33m',    # section headings
    'flag': '\033[36m',      # -h, --panel
    'meta': '\033[32m',      # FILE, PATH, {cdn,inline,directory}
    'err': '\033[1;31m',
    'off': '\033[0m',
}

# A flag anywhere in the text (including inside help sentences like
# "--html only: ..."), and the metavars / choice lists that follow one.
_FLAG_RE = re.compile(r'(?<![\w.-])(--?[A-Za-z][\w-]*)')
# Single letters count (--ncols N), which is safe because metavars are only
# painted in the usage block and the invocation column — never in help prose.
_META_RE = re.compile(r'(?<![\w-])([A-Z][A-Z_]*)(?![\w-])|(\{[^{}]*\})')

# Option and positional rows are indented two spaces; wrapped help text sits
# out at the help column. The gap between the two columns is 2+ spaces.
_ROW_RE = re.compile(r'^( {2,3})(\S.*?)(\s{2,}|$)(.*)$')
_WRAP_RE = re.compile(r'^ {4,}\S')
_HEADING_RE = re.compile(r'^[a-z][a-z \-]*:$')


def _color_enabled(stream):
    """Whether to emit ANSI on ``stream``. ``NO_COLOR`` always wins."""
    if os.environ.get('NO_COLOR'):     # set and non-empty, per no-color.org
        return False
    if os.environ.get('FORCE_COLOR'):
        return True
    if os.environ.get('TERM') == 'dumb':
        return False
    try:
        if not stream.isatty():
            return False
    except Exception:                                     # noqa: BLE001
        return False
    if os.name == 'nt' and not (os.environ.get('WT_SESSION')
                                or os.environ.get('ANSICON')
                                or os.environ.get('TERM')):
        return False
    return True


def _paint(text, key):
    return f"{_ANSI[key]}{text}{_ANSI['off']}"


def _paint_flags(text):
    return _FLAG_RE.sub(lambda m: _paint(m.group(1), 'flag'), text)


def _paint_values(text):
    """Metavars and choice lists. Run after _paint_flags — the escape codes it
    inserts are lowercase, so neither pattern here can match inside one."""
    return _META_RE.sub(lambda m: _paint(m.group(0), 'meta'), text)


def _colorize_help(text):
    """Paint an already-formatted help/usage string."""
    lines, in_usage = [], False
    for line in text.split('\n'):
        if line.startswith('usage:'):
            in_usage = True
            prog, _, rest = line[6:].strip().partition(' ')
            lines.append(_paint('usage:', 'bold') + ' ' + _paint(prog, 'bold')
                         + (' ' + _paint_values(_paint_flags(rest)) if rest else ''))
        elif in_usage:
            # The usage block runs to the first blank line, and its wrapped
            # lines start with '[' or '-' — classifying them as option rows
            # would split them on their internal whitespace.
            in_usage = bool(line.strip())
            lines.append(_paint_values(_paint_flags(line)) if in_usage else line)
        elif _HEADING_RE.match(line):
            # 'options:' on 3.10+, 'optional arguments:' on 3.9, plus the
            # argument groups this parser defines.
            lines.append(_paint(line, 'head'))
        elif _WRAP_RE.match(line):
            lines.append(_paint_flags(line))              # wrapped help text
        elif _ROW_RE.match(line):
            indent, invocation, gap, help_text = _ROW_RE.match(line).groups()
            lines.append(indent + _paint_values(_paint_flags(invocation))
                         + gap + _paint_flags(help_text))
        else:
            lines.append(_paint_flags(line))              # description, epilog
    return '\n'.join(lines)


class ColorHelpParser(argparse.ArgumentParser):
    """``ArgumentParser`` that colors help, usage and errors on a terminal.

    The decision is made per stream at print time — help goes to stdout, an
    argument error to stderr — so piping one never strips color from the other.
    """

    def print_help(self, file=None):
        file = file or sys.stdout
        text = self.format_help()
        file.write(_colorize_help(text) if _color_enabled(file) else text)

    def print_usage(self, file=None):
        file = file or sys.stdout
        text = self.format_usage()
        file.write(_colorize_help(text) if _color_enabled(file) else text)

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, _error_line(f'{self.prog}: error: {message}'))


def _error_line(message):
    """One-line error text, with the leading ``unichart:`` marked on a tty."""
    prefix, sep, rest = message.partition(':')
    if _color_enabled(sys.stderr) and sep:
        return f"{_paint(prefix + sep, 'err')}{rest}\n"
    return f'{message}\n'


def _gallery_paths():
    """Where gallery/index.html would live, nearest layout first."""
    here = os.path.dirname(os.path.abspath(__file__))
    return [os.path.join(here, 'gallery'),
            os.path.join(os.path.dirname(here), 'gallery')]


def _open_gallery(no_browser=False):
    """Open the example gallery, or say where to get it. Returns an exit code."""
    for folder in _gallery_paths():
        page = os.path.join(folder, 'index.html')
        if os.path.isfile(page):
            if no_browser:
                print(page)                     # script-friendly: the path alone
                return 0
            import webbrowser
            from pathlib import Path
            print(f'Opening {page}')
            webbrowser.open(Path(page).as_uri())
            return 0

    # A checkout that simply hasn't built the page yet can build it; an
    # installed copy has no gallery directory at all, so point at the repo.
    for folder in _gallery_paths():
        builder = os.path.join(folder, 'make_gallery.py')
        if os.path.isfile(builder):
            sys.stderr.write(_error_line(
                'unichart: the gallery has not been built yet. Build it with:\n'
                f'    python {builder}'))
            return 1

    sys.stderr.write(_error_line(
        'unichart: the example gallery ships in the source tree, which this '
        'install does not carry. Browse it at:\n'
        f'    {GALLERY_URL}'))
    return 1


def build_parser():
    p = ColorHelpParser(
        prog='unichart',
        description='Open the unichart explorer on a data file, or export a '
                    'static board.',
        epilog='Panels beyond method/x/y/z (kwargs, dataset pins) are '
               'expressible from Python: see nb.dashboard / nb.explore.\n'
               'Help is colored on a terminal; set NO_COLOR=1 to turn that off.',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('files', nargs='*', metavar='FILE',
                   help='data file(s) to load: .csv .tsv .txt .xlsx .xls '
                        '.json .parquet. Omit to open an empty explorer.')
    p.add_argument('--version', action='version',
                   version=f'unichart {__version__}')
    p.add_argument('--gallery', action='store_true',
                   help='open the example gallery — every plot type on one '
                        'page, each with the code that drew it — and exit. '
                        'With --no-browser, print its path instead')

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
                       help='export in dark mode (--html); with a session, '
                            'override the theme it was saved in')

    out = p.add_argument_group('output')
    out.add_argument('--info', action='store_true',
                     help='print the datasets and columns, then exit')
    out.add_argument('--html', metavar='PATH',
                     help='write a self-contained static HTML board to PATH '
                          'instead of serving the explorer')
    out.add_argument('--save-session', metavar='PATH', dest='save_session',
                     help='write a session file to PATH and exit, instead of '
                          'serving. Any --panel is drawn first, so the session '
                          'reopens on that plot')
    out.add_argument('--embed-js', default='cdn',
                     choices=['cdn', 'inline', 'directory'], dest='embed_js',
                     help="--html only: how to include plotly.js. 'cdn' "
                          "(default, small, needs internet), 'inline' (fully "
                          "offline), 'directory'")
    out.add_argument('--port', type=int, default=8050, metavar='N',
                     help='preferred port (default: 8050; a free one is '
                          'chosen if it is busy)')
    out.add_argument('--app', action='store_true', dest='app_window',
                     help='open the explorer as a standalone desktop window '
                          '(no tabs, no address bar, its own taskbar entry) '
                          'instead of a browser tab. Needs Chrome, Chromium, '
                          'Brave or Edge — without one it falls back to a tab '
                          'and says so. --no-browser wins over it')
    out.add_argument('--no-browser', action='store_true', dest='no_browser',
                     help="serve, but don't open a browser (headless / remote)")
    out.add_argument('--completion', metavar='SHELL',
                     choices=sorted(COMPLETION_SCRIPTS),
                     help='print a tab-completion script and exit. Install it '
                          'with:  eval "$(unichart --completion bash)"  in '
                          '~/.bashrc (or --completion zsh in ~/.zshrc)')
    return p


def _split_files(args):
    """Partition FILEs into (data files, session files), missing paths first.

    A session is identified by its contents, not its extension — ``.json`` is
    both a data format and the session format — so only files that could
    plausibly be one are opened for the check.
    """
    from pathlib import Path

    from ._core import sniff_session

    missing = [f for f in args.files if not Path(f).exists()]
    if missing:
        raise CliError(f"no such file: {', '.join(missing)}")

    data, sessions = [], []
    for name in args.files:
        if (name.lower().endswith(_SESSION_SUFFIXES)
                and sniff_session(name) is not None):
            sessions.append(name)
        else:
            data.append(name)

    # --combine / --set-col / --name-col describe how to read a data file. A
    # session already records how its sets were built, so with nothing else on
    # the line these flags have nothing to act on — say so rather than accepting
    # them and quietly doing nothing.
    if sessions and not data:
        for flag, value in (('--combine', args.combine),
                            ('--set-col', args.set_col),
                            ('--name-col', args.name_col)):
            if value:
                raise CliError(f'{flag} applies to data files, not sessions')
    return data, sessions


def _load(nb, files, args):
    """Load the data FILEs onto the notebook."""
    if not files:
        return
    sources = files if len(files) > 1 else files[0]
    try:
        nb.load(sources, set_idx_column=args.set_col,
                set_name_column=args.name_col, combined=args.combine)
    except Exception as exc:                              # noqa: BLE001
        raise CliError(f'could not read the data: {exc}') from exc


def _load_sessions(nb, paths, replay=True):
    """Restore each session file onto the notebook, in command-line order.

    Only for the paths with no board — ``--info``, ``--html``,
    ``--save-session``. The serving path hands them to ``explore(sessions=...)``
    instead, so the restore shows up in the transcript and paints the chart.

    ``replay=False`` skips redrawing the session's plot, for the caller that
    only wants to look at what loaded.
    """
    for path in paths:
        try:
            nb.load_session(path, replay=replay)
        except Exception as exc:                          # noqa: BLE001
            raise CliError(f'could not load the session {path}: {exc}') from exc


def _draw_panels(nb, panels):
    """Draw each --panel through its real plot method.

    ``--save-session`` needs this: a session records the last plotting call, and
    on the serving path the panels are replayed inside the board rather than
    here, so nothing would have plotted in this process.
    """
    from .dashboard import _TABLE_METHODS, _Z_METHODS

    for panel in panels:
        method = panel['method']
        kwargs = dict(panel.get('kwargs') or {})
        x, y = panel.get('x'), panel.get('y')
        # The spec's y is always a list; the methods that take a scalar y
        # (histogram, box) would choke on a one-item one.
        if isinstance(y, list) and len(y) == 1:
            y = y[0]

        if method in _TABLE_METHODS:
            # table() does not take x/y: the columns to show are cols, and x is
            # the interpolation axis. Same mapping the board's render_panel uses.
            if y:
                kwargs['cols'] = y
            if x:
                kwargs['x_col'] = x
        else:
            if x is not None:
                kwargs['x'] = x
            if y is not None:
                kwargs['y'] = y
            if panel.get('z') is not None and method in _Z_METHODS:
                kwargs['z'] = panel['z']
        try:
            getattr(nb, method)(**kwargs)
        except Exception as exc:                          # noqa: BLE001
            raise CliError(f'--panel {method}: {exc}') from exc


def _session_plot_calls(paths):
    """The plot call recorded in each session file, straight from the file.

    Read rather than taken from ``nb._last_plot_call`` after the restore: that
    attribute is only set when the replay *worked*, and --info exists to say
    what a file contains — including a call that would not replay.
    """
    import json

    from ._core import read_png_session, sniff_session

    calls = []
    for path in paths:
        try:
            if sniff_session(path) == 'png':
                session = read_png_session(path) or {}
            else:
                with open(path, encoding='utf-8') as fh:
                    session = json.load(fh)
        except Exception:                                 # noqa: BLE001
            continue
        call = session.get('plot_call')
        if isinstance(call, dict) and call.get('method'):
            calls.append((path, call))
    return calls


def _print_info(nb, sessions=()):
    """Summarize what got loaded — the non-GUI way to check a file parsed."""
    from .dashboard import _all_columns, _numeric_columns

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
    # Only a session brings a plot along, so this is the answer to "will opening
    # this file give me my chart back?".
    calls = _session_plot_calls(sessions)
    if calls:
        print()
        for path, call in calls:
            fields = ', '.join(
                f'{k}={v!r}' for k, v in (call.get('kwargs') or {}).items()
                if k in ('x', 'y', 'z'))
            label = f'{path}: ' if len(calls) > 1 else ''
            print(f"{label}plot call: {call['method']}({fields})")


def main(argv=None):
    """Entry point. Returns a process exit code rather than raising."""
    argv = list(sys.argv[1:] if argv is None else argv)

    # The completion hook comes first and never reaches argparse: the line
    # being completed is half-typed by definition, and argparse would exit(2)
    # on it. It is deliberately not a registered argument — it is a private
    # protocol between the shell script and this module, not a user-facing flag.
    if argv and argv[0] == '--_complete':
        return _run_completion(argv[1:])

    args = build_parser().parse_args(argv)

    if args.completion:
        print(COMPLETION_SCRIPTS[args.completion], end='')
        return 0

    # Before the unichart/Dash imports: the gallery is a static page, so it
    # opens instantly without loading pandas, Plotly or Dash.
    if args.gallery:
        return _open_gallery(no_browser=args.no_browser)

    try:
        from ._core import UnichartNotebook
        from .dashboard import (PLOT_METHODS, _default_panel_spec,
                                        explore, to_html)
    except ImportError as exc:
        sys.stderr.write(_error_line(f'unichart: {exc}'))
        return 1

    try:
        panels = [parse_panel(spec, PLOT_METHODS) for spec in args.panel]
        data_files, session_files = _split_files(args)

        nb = UnichartNotebook()
        _load(nb, data_files, args)

        # Serving defers the sessions to the board (see _load_sessions); every
        # other action has no board to defer to and restores them here, after
        # the data so the session's own formatting is the one that sticks.
        serving = not (args.info or args.html or args.save_session)
        if not serving:
            # --info reads the recorded call out of the file itself, so redrawing
            # the plot here would be work nothing looks at.
            _load_sessions(nb, session_files, replay=not args.info)
        if args.dark:
            nb.toggle_darkmode(True)

        if args.info:
            _print_info(nb, session_files)
            return 0

        if args.save_session:
            if not nb.sets:
                raise CliError('--save-session needs data: pass at least one FILE')
            _draw_panels(nb, panels)
            nb.save_session(args.save_session)
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

        # Default: serve the explorer. This blocks until the board is
        # closed (its ✕ ends the process from inside) or Ctrl-C.
        # ncols/width/height size a static --html grid; the terminal board has
        # one chart pane that fills its own space, so they are not forwarded.
        # --dark is passed through rather than applied above: a restored session
        # carries its own theme and would otherwise overwrite the flag.
        explore(nb, sessions=session_files or None, panels=panels or None,
                title=args.title, port=args.port,
                dark=True if args.dark else None,
                open_browser=not args.no_browser,
                app_window=args.app_window)
        return 0

    except ImportError as exc:
        # Dash is a dependency, but a hand-built environment can still lack
        # it. --info and --html don't need it, so this only fires on the
        # serving path — report _require_dash's message as one line, not a
        # traceback.
        sys.stderr.write(_error_line(f'unichart: {exc}'))
        return 1
    except CliError as exc:
        sys.stderr.write(_error_line(f'unichart: {exc}'))
        return 2
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == '__main__':
    sys.exit(main())
