"""A terminal-style explorer: sidebar, live chart, and a Python REPL.

This is the GUI behind :func:`unichart.dashboard.explore` — a dark, three-part
board that mirrors the web version at plot.thomas-emmons.com:

* a **sidebar** with a drop zone for data, tables of the loaded sets and
  their parameters (searchable; click a name to put it in the terminal), and
  a cheat sheet of clickable snippets,
* a **chart pane** showing the most recent figure, and
* a **terminal** you drive with real Python: ``plot(x='time', y='temp')``.

The notebook's own methods are bound as bare names in the terminal's namespace,
so the cheat sheet reads the way the library does. ``uc`` is there too, for
anything the shortcuts don't cover.

The top bar's ✕ closes the board: outside a Jupyter kernel the board owns the
process it is served from, so closing it (or typing ``exit()``) stops the
server and ends the program, after asking.

Every command runs in-process against one ``UnichartNotebook``, so the board is
a view onto a live notebook rather than a copy of one — and because the terminal
executes arbitrary Python, the server binds to 127.0.0.1 only. It has exactly
the rights the person running it already has at a Python prompt; it is not a
sandbox, and it must not be exposed to a network.

Dash is an optional dependency, imported lazily.
"""

import ast
import base64
import builtins
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tokenize
import tempfile
import threading
import traceback
import webbrowser
from html import escape as _escape
from pathlib import Path

import pandas as pd

from ._core import (_READER_STUB_BODY, _compact_indices, _reader_name,
                    sniff_session)
from .dashboard import (_UPLOAD_READERS, _df_from_upload, _pick_port,
                        _require_dash, _upload_bytes)

# ---------------------------------------------------------------------------
# Palette — sampled from the web version. Dark only, on purpose: this board
# does not carry the dashboard's light/dark token machinery, so restyling it
# leaves uc.dashboard() and to_html() untouched.
# ---------------------------------------------------------------------------

BG = '#0b0d11'          # page + terminal
SURFACE = '#171a21'     # top bar, sidebar, chips
HAIRLINE = '#262a33'
ACCENT = '#3b82f6'
INK = '#e8eaed'
MUTED = '#888e9a'
ERROR = '#f87171'

# Syntax colors: One Dark, which is what the web version uses (its string green
# #98c379 is pixel-identical in the reference). Keyed by the short class names
# the tokenizer emits.
SYNTAX = {
    'kw':  '#c678dd',   # keywords: def, for, in, not, None ...
    'str': '#98c379',   # strings, including f-string literal parts
    'num': '#d19a66',   # numbers
    'fn':  '#61afef',   # a name being called
    'bi':  '#e5c07b',   # builtins
    'op':  '#abb2bf',   # operators and punctuation
    'com': '#7f848e',   # comments
    'nm':  '#e8eaed',   # everything else
}

UI_FONT = ('system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", '
           'Arial, sans-serif')
MONO_FONT = ('ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, '
             '"Liberation Mono", monospace')

# Methods that would launch another server from inside this one's callback.
_RECURSIVE = {'explore', 'dashboard', 'dashboard_to_html', 'terminal'}

# Executing user code touches process-global state (sys.stdout, sys.displayhook)
# and the shared notebook, so commands are serialized.
_RUN_LOCK = threading.Lock()

# Uploaded and demo files land here so they load through the same
# uc.load(path) the transcript shows, rather than a hidden side channel. One
# directory per process, not per board: re-running an `explore()` cell in a
# notebook would otherwise mint a new one every time.
_UPLOADS = None


def _uploads_dir():
    global _UPLOADS
    if _UPLOADS is None:
        _UPLOADS = Path(tempfile.mkdtemp(prefix='unichart-'))
    return _UPLOADS


def _slug(text):
    """A filename-safe stub of a board title, for the session it downloads."""
    return re.sub(r'[^A-Za-z0-9._-]+', '-', str(text)).strip('-').lower() or 'unichart'


def _reader_template(paths):
    """Terminal input that reads files unichart has no reader for.

    One stub reader per extension, registered so the next file of that type
    loads with no questions, then one load per file. It is staged in the input
    rather than run: the stub is only a guess at the format, so the user edits
    it until it returns the right DataFrame, then presses Enter. A file with no
    extension can't be registered for, so its reader is passed to its load.
    """
    defs, registers, loads = [], [], []
    for path in paths:
        suffix = Path(path).suffix.lower()
        name = _reader_name(suffix)
        stub = (f"def {name}(path):\n"
                f"    return {_READER_STUB_BODY}  # edit to fit the file")
        if stub not in defs:
            defs.append(stub)
            if suffix:
                registers.append(f"uc.register_reader({suffix!r}, {name})")
        loads.append(f"uc.load({str(path)!r})" if suffix
                     else f"uc.load({str(path)!r}, reader={name})")
    return '\n\n'.join(defs) + '\n\n' + '\n'.join(registers + loads)


def _staged_note(path):
    """The transcript line that points at a reader staged for ``path``."""
    name, suffix = Path(path).name, Path(path).suffix.lower()
    what = (f"unichart has no reader for {suffix} files" if suffix
            else "unichart can't tell what kind of file this is")
    return (f"{name}: {what}. A reader is staged below — a function that "
            f"takes the file's path and returns a DataFrame. Edit it to parse "
            f"the file, then press Enter.")


# ---------------------------------------------------------------------------
# Closing the board
# ---------------------------------------------------------------------------
#
# "Close" means "end this process". A Dash server has no supported way to
# unbind its port and leave the process running — werkzeug dropped the shutdown
# hook in 2.1 — and there would be nothing left to run anyway: the board owns
# the process it serves from (the `unichart` command, or a script that calls
# explore()). So the control is only offered where that is true, which is
# everywhere except inside a Jupyter kernel, where closing the board would take
# the kernel and the user's variables with it.
#
# os._exit rather than sys.exit because the serving thread is not this one: a
# SystemExit raised in a timer thread ends the timer and nothing else.

def _shutdown(delay=0.4, code=0):
    """End the process, once the response that asked for it is on the wire.

    The delay is what lets the browser finish the round trip that triggered
    this — without it the socket dies mid-response and the board is left
    showing a network error instead of its goodbye.
    """
    def stop():
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(int(code or 0))

    threading.Timer(delay, stop).start()


def _ends_with_semicolon(source):
    """True when the last token of ``source``, comments aside, is a ``;``."""
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
            tokenize.DEDENT, tokenize.ENDMARKER}
    try:
        tokens = [t for t in tokenize.generate_tokens(io.StringIO(source).readline)
                  if t.type not in skip]
    except (tokenize.TokenError, SyntaxError):
        return False
    return bool(tokens) and tokens[-1].string == ';'


class _Quitter:
    """``exit`` / ``quit`` in the terminal's namespace.

    Python's own are :class:`site.Quitter`, which raises ``SystemExit`` — here
    that is caught by the command runner and painted into the transcript as a
    traceback. Someone will type ``exit()``; it should close the board.
    """

    def __init__(self, allowed=True):
        self.allowed = allowed

    def __call__(self, code=0):
        if not self.allowed:
            # "the board", not "the tab": from a kernel it may be a window.
            print('unichart: this board runs inside a notebook kernel, so '
                  'closing it would close the kernel too. Close the board '
                  'instead; the notebook keeps the data.')
            return
        print('unichart: closing — the server is stopping.')
        # Longer than the button's: this one is answering a command, and the
        # transcript it prints into has to reach the browser first.
        _shutdown(0.8, code)

    def __repr__(self):
        return ('Use exit() to close the board.' if self.allowed
                else 'This board is in a notebook kernel; close the tab.')

# The cheat sheet: (snippet, is_wide). Clicking one drops it into the input.
CHEAT_SHEET = [
    ("plot(x='time', y=['temperature', 'pressure'])", True),
    ("plot(x='time', y='rpm', by='sets')", True),
    ("bar(x='TITLE', y='pressure', agg='mean')", True),
    ("histogram(x='temperature', nbins=30)", True),
    ("contour(x='rpm', y='pressure', z='temperature')", True),
    ("select([0, 1])", False),
    ("select('1:10')  # sets 1-9", False),
    ("omit(2)", False),
    ("restore()", False),
    ("color(0, 'red')", False),
    ("marker(1, 's')", False),
    ("var_format('temperature', linestyle='--')", True),
    ("uc.table()", False),
    ("summary()", False),
    ("list_parms()", False),
    ("help()", False),
    ("help('plot')", False),
    ("save_session('session.json')", True),
    ("load_session('session.json')", True),
]

# The transcript opens with what the notebook itself prints on construction,
# so the board reads as a session you are joining rather than a blank slate.
BANNER = [
    'UniChart Notebook Environment Initialized.',
    'Plot theme set to: Dark Mode',
    'Unichart ready. Load data on the left, then plot(x=..., y=...).',
    "Type help() for the full API, help('plot') for one method.",
]


# ---------------------------------------------------------------------------
# The REPL
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _scoped_output(uc, sink):
    """Route unichart's rich output into ``sink`` for the duration of one command.

    ``unichart._core.display`` is a module-level name (bound from IPython at import,
    with a print fallback), so rebinding it captures every ``display(...)`` the
    library makes — HTML tables from ``table``/``summary``/``list_parms``, and
    the plot copy button. Restored in ``finally`` because the notebook driving
    this board may well be a live Jupyter session: leaving the shim installed
    would break the user's next ``uc.plot()`` cell.

    The copy button is a Jupyter affordance and would just be noise here (this
    board has its own chart pane), so it is off for the duration too.

    ``help()`` colors itself with ANSI, which _ansi_markup turns into spans. Its
    auto-detection can't help us here — commands run under redirect_stdout, so
    it would always see "not a tty" — so the override is set explicitly and
    restored with the rest. NO_COLOR still wins: it is the user's switch, and
    this board is not special enough to override it.
    """
    from . import _core

    real_display = _core.display
    real_copy = uc.copy_buttons
    real_help_color = _core._HELP_COLOR

    def capture(*objs, **kwargs):
        for obj in objs:
            sink.append(obj)

    _core.display = capture
    uc.copy_buttons = False
    _core._HELP_COLOR = not os.environ.get('NO_COLOR')
    try:
        yield
    finally:
        _core.display = real_display
        uc.copy_buttons = real_copy
        _core._HELP_COLOR = real_help_color


def _format_exception(exc):
    """Render a traceback showing only the user's own frames.

    Everything above ``<terminal>`` is this module's exec wrapper; showing it
    would put library plumbing in front of the line the user actually typed.
    """
    frames = traceback.extract_tb(exc.__traceback__)
    user = [f for f in frames if f.filename == '<terminal>']
    parts = ['Traceback (most recent call last):\n'] if user else []
    parts += traceback.format_list(user)
    parts += traceback.format_exception_only(type(exc), exc)
    return ''.join(parts).rstrip()


def _token_class(tok_type, text, next_text):
    """Map one Python token to a highlight class."""
    import keyword
    import token as _t

    if tok_type == _t.STRING or tok_type == getattr(_t, 'FSTRING_START', -1):
        return 'str'
    if tok_type in (getattr(_t, 'FSTRING_MIDDLE', -1),
                    getattr(_t, 'FSTRING_END', -1)):
        return 'str'
    if tok_type == _t.NUMBER:
        return 'num'
    if tok_type == _t.COMMENT:
        return 'com'
    if tok_type == _t.OP:
        return 'op'
    if tok_type == _t.NAME:
        if keyword.iskeyword(text) or keyword.issoftkeyword(text):
            return 'kw'
        # A name immediately followed by '(' reads as a call — which is most of
        # what this terminal is used for, so it is worth distinguishing.
        if next_text == '(':
            return 'fn'
        if text in dir(builtins):
            return 'bi'
    return 'nm'


def highlight(source):
    """Tokenize Python into per-line ``[(text, class), ...]`` runs.

    Uses the stdlib tokenizer rather than a regex, so f-strings, nested quotes
    and comments are handled properly. Half-typed or invalid input raises in
    ``tokenize``, and a transcript entry must render regardless — so any failure
    falls back to unhighlighted text rather than losing the line.
    """
    lines = source.split('\n')
    try:
        raw = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return [[(line, 'nm')] for line in lines]

    # Positions are (row, col) 1-indexed by row; walk them against the source
    # so the whitespace *between* tokens is preserved verbatim.
    out = [[] for _ in lines]
    row, col = 1, 0
    for i, tk in enumerate(raw):
        if tk.type in (tokenize.ENCODING, tokenize.ENDMARKER):
            continue
        (srow, scol), (erow, ecol) = tk.start, tk.end
        # Gap before this token (indentation, spaces between operators).
        while (row, col) < (srow, scol):
            line = lines[row - 1]
            stop = scol if row == srow else len(line)
            if stop > col:
                out[row - 1].append((line[col:stop], 'nm'))
            if row == srow:
                col = scol
            else:
                row, col = row + 1, 0
        if tk.type in (tokenize.NEWLINE, tokenize.NL):
            row, col = erow, ecol
            continue
        nxt = next((t.string for t in raw[i + 1:]
                    if t.type not in (tokenize.NL, tokenize.NEWLINE,
                                      tokenize.COMMENT)), '')
        cls = _token_class(tk.type, tk.string, nxt)
        # A token may span lines (triple-quoted strings); split it per line so
        # each transcript line renders independently.
        for offset, piece in enumerate(tk.string.split('\n')):
            if piece:
                out[srow - 1 + offset].append((piece, cls))
        row, col = erow, ecol
    return out


def _code_spans(html, source, base=''):
    """Render highlighted source as a list of ``html.Span`` per line."""
    return [
        [html.Span(text, className=f'tk-{cls}') for text, cls in line]
        or [html.Span('', className='tk-nm')]
        for line in highlight(source)
    ]


class Session:
    """One terminal bound to one notebook.

    The namespace exposes the notebook's public methods as bare names — the
    cheat sheet's ``plot(...)`` / ``select(...)`` are those, not globals of our
    own — plus ``uc`` itself for anything else, and ``exit`` / ``quit``, which
    close the board when it owns its process.
    """

    def __init__(self, uc, allow_quit=True):
        self.uc = uc
        self.ns = {'__name__': '__console__', '__builtins__': __builtins__,
                   'uc': uc, 'nb': uc, 'pd': pd}   # nb: the old name, still answered
        # Shadowing the builtins: a bare exit() should close the board rather
        # than raise SystemExit into the transcript.
        self.ns['exit'] = self.ns['quit'] = _Quitter(allow_quit)
        for name in dir(uc):
            if name.startswith('_') or name in _RECURSIVE:
                continue
            attr = getattr(uc, name, None)
            if callable(attr):
                self.ns[name] = attr

    def run(self, source):
        """Execute one command. Returns a dict the transcript renders.

        Keys: ``text`` (captured stdout/stderr), ``error`` (a traceback, or
        None), ``html`` (rich blocks from ``display``), ``figure`` (a new
        figure, or None).
        """
        source = (source or '').strip()
        if not source:
            return None

        sink, out = [], io.StringIO()
        error = None
        with _RUN_LOCK:
            before = self.uc.last_fig
            with _scoped_output(self.uc, sink):
                try:
                    with contextlib.redirect_stdout(out), \
                            contextlib.redirect_stderr(out):
                        self._exec(source)
                except SyntaxError as exc:
                    error = ''.join(
                        traceback.format_exception_only(type(exc), exc)).rstrip()
                except BaseException as exc:            # noqa: BLE001
                    error = _format_exception(exc)
            after = self.uc.last_fig

        figure = after if (after is not None and after is not before) else None
        html_blocks = []
        for obj in sink:
            # A figure handed to display() (rather than returned) still counts.
            if hasattr(obj, 'to_plotly_json'):
                figure = obj
            elif hasattr(obj, 'data') and isinstance(obj.data, str):
                html_blocks.append(obj.data)
            else:
                out.write(f'{obj}\n')
        # table()/summary()/list_parms() cache a go.Table in last_fig *and*
        # display the richer sortable HTML. Showing both would put the same
        # data on screen twice, so the rich block wins and the chart is left
        # on whatever was plotted last — which is what a notebook does.
        if html_blocks:
            figure = None

        return {'text': out.getvalue().rstrip('\n'), 'error': error,
                'html': html_blocks, 'figure': figure}

    def _exec(self, source):
        """Run a block, echoing the value of a trailing expression.

        ``compile(..., 'single')`` handles only one statement, so a multi-line
        entry (Shift+Enter) is parsed instead: everything up to the last node is
        exec'd, and a trailing expression is eval'd and echoed the way a REPL
        would. That is what makes both ``1 + 1`` and a pasted several-line block
        behave as expected.
        """
        tree = ast.parse(source, filename='<terminal>', mode='exec')
        if not tree.body:
            return
        *head, last = tree.body
        if head:
            exec(compile(ast.Module(body=head, type_ignores=[]),
                         '<terminal>', 'exec'), self.ns)
        if isinstance(last, ast.Expr):
            value = eval(compile(ast.Expression(body=last.value),
                                 '<terminal>', 'eval'), self.ns)
            # A closing ';' hides the value, as in Jupyter — list_parms();
            if not _ends_with_semicolon(source):
                self._echo(value)
        else:
            exec(compile(ast.Module(body=[last], type_ignores=[]),
                         '<terminal>', 'exec'), self.ns)

    def _echo(self, value):
        """Display a trailing expression's value, REPL-style.

        A figure is routed to the chart pane rather than repr'd — printing
        ``Figure({...})`` into the transcript is what a notebook would never do.
        """
        if value is None:
            return
        if hasattr(value, 'to_plotly_json'):
            self.uc.last_fig = value
            return
        if isinstance(value, (pd.DataFrame, pd.Series)):
            print(value.to_string())
            return
        print(repr(value))


def demo_frame(n=90, seed=0):
    """The demo dataset: three runs that exercise every cheat-sheet snippet.

    Columns are chosen so the snippets are runnable as written — ``time``,
    ``temperature``, ``pressure``, ``rpm``, and a ``TITLE`` naming each run.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    runs = [('Baseline run', 185.0, 27.0, 1.00),
            ('High boost', 205.0, 31.0, 1.18),
            ('Lean mixture', 172.0, 25.5, 0.90)]
    frames = []
    for idx, (name, t_max, p_max, scale) in enumerate(runs):
        t = np.linspace(0, 30, n)
        rise = 1 - np.exp(-t / 7.5)
        frames.append(pd.DataFrame({
            'SETNUMBER': idx,
            'TITLE': name,
            'time': t,
            'temperature': 55 + (t_max - 55) * rise + rng.normal(0, 1.6, n),
            'pressure': 16 + (p_max - 16) * rise + rng.normal(0, 0.28, n),
            'rpm': 900 + 4200 * rise * scale + rng.normal(0, 55, n),
        }))
    return pd.concat(frames, ignore_index=True)



def battery_frame(seed=1):
    """Four cells discharged at different C-rates, each logged every 20 s.

    The runs end at different times, so they only line up against state of
    charge — which is what ``table(x_in=...)`` and ``delta(align_on=...)`` are for.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    frames = []
    for idx, c_rate in enumerate((0.5, 1.0, 2.0, 3.0)):
        capacity = 5.0 * (1 - 0.035 * c_rate)              # Ah delivered
        current = 5.0 * c_rate
        t = np.arange(0.0, capacity / current * 3600, 20.0)
        soc = 100 * (1 - current * t / 3600 / 5.0)
        s = np.clip(soc / 100, 0, 1)
        ocv = 3.35 + 0.75 * s - 0.25 * np.exp(-s / 0.06) + 0.1 * np.exp((s - 1) / 0.05)
        temp = 25 + 6 * c_rate ** 1.5 * (1 - np.exp(-t / 900)) * (1 + 0.4 * (1 - s))
        frames.append(pd.DataFrame({
            'SETNUMBER': idx,
            'TITLE': f'{c_rate:g}C discharge',
            'time_s': t,
            'soc_pct': soc.round(2),
            'voltage_v': (ocv - current * 0.018 + rng.normal(0, 0.004, t.size)).round(4),
            'current_a': np.full(t.size, current),
            'temp_c': (temp + rng.normal(0, 0.15, t.size)).round(2),
        }))
    return pd.concat(frames, ignore_index=True)


def compressor_frame(seed=2):
    """A compressor map: four speed lines, each swept from surge to choke.

    Scattered (flow, pressure ratio) points with an efficiency at each — the
    shape ``contour``, ``hue`` and ``plot_marginal`` are built for.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    frames = []
    for idx, speed in enumerate((70, 80, 90, 100)):
        n = speed / 100
        flow = np.linspace(4.0 * n ** 1.3, 7.2 * n ** 1.1, 22)
        x = (flow - flow[0]) / (flow[-1] - flow[0])          # 0 surge, 1 choke
        pr = 1 + 2.6 * n ** 2.2 * (1 - 0.55 * x ** 2.5)
        eff = 86 - 4 * (1 - n) * 10 - 30 * (x - 0.45) ** 2 - 25 * (n - 0.88) ** 2
        frames.append(pd.DataFrame({
            'SETNUMBER': idx,
            'TITLE': f'{speed}% speed',
            'flow_kgs': (flow + rng.normal(0, 0.02, flow.size)).round(3),
            'pressure_ratio': (pr + rng.normal(0, 0.01, flow.size)).round(3),
            'efficiency_pct': (eff + rng.normal(0, 0.35, flow.size)).round(2),
            'speed_pct': speed,
        }))
    return pd.concat(frames, ignore_index=True)


def weather_frame(seed=3):
    """A year of daily weather at four stations: one set per station, with
    ``month`` and ``season`` columns to group by."""
    import numpy as np

    rng = np.random.default_rng(seed)
    day = np.arange(1, 366)
    month = pd.to_datetime(day - 1, unit='D', origin='2025-01-01').month
    season = np.array(['winter', 'spring', 'summer', 'autumn'])[(month % 12) // 3]
    stations = [('Coastal', 14, 6, 3.2, 22), ('Inland', 12, 13, 1.8, 14),
                ('Mountain', 4, 10, 2.6, 26), ('Desert', 22, 11, 0.3, 18)]
    frames = []
    for idx, (name, mean, swing, rain, wind) in enumerate(stations):
        temp = mean - swing * np.cos(2 * np.pi * (day - 15) / 365)
        wet = rng.random(day.size) < 0.25 + 0.15 * np.cos(2 * np.pi * day / 365)
        frames.append(pd.DataFrame({
            'SETNUMBER': idx,
            'TITLE': name,
            'day': day,
            'month': month,
            'season': season,
            'temp_c': (temp + rng.normal(0, 2.5, day.size)).round(1),
            'rain_mm': np.where(wet, rng.gamma(1.2, rain * 3, day.size), 0).round(1),
            'wind_kmh': rng.gamma(4, wind / 4, day.size).round(1),
        }))
    return pd.concat(frames, ignore_index=True)


def survey_frame(seed=4):
    """An inlet total-pressure survey at three throttle settings: rakes every
    30 degrees round the duct, five probes from hub to tip. A strut at 210
    degrees leaves a wake that deepens with the airflow — the shape ``polar``
    and its contour map are built for."""
    import numpy as np

    rng = np.random.default_rng(seed)
    angle = np.repeat(np.arange(0.0, 360.0, 30.0), 5)        # 12 rakes ...
    radius = np.tile(np.linspace(40.0, 90.0, 5), 12)          # ... x 5 probes, mm
    span = (radius - 40.0) / 50.0                             # 0 at the hub, 1 at the tip
    off = (angle - 210.0 + 180.0) % 360.0 - 180.0             # degrees from the strut
    frames = []
    for idx, (name, flow) in enumerate((('Idle', 0.35), ('Cruise', 0.75),
                                        ('Max power', 1.0))):
        loss = flow ** 2 * (3.2 * np.exp(-(off / 28.0) ** 2) * (0.5 + 0.5 * span)
                            + 1.6 * (span - 0.45) ** 2
                            + 0.35 * np.cos(np.radians(2 * (angle - 40.0))))
        frames.append(pd.DataFrame({
            'SETNUMBER': idx,
            'TITLE': name,
            'angle_deg': angle,
            'radius_mm': radius,
            'pt_kpa': (101.3 - loss + rng.normal(0, 0.03, angle.size)).round(3),
            'tt_c': (15 + 6 * flow + 4 * flow * span * np.exp(-((angle - 110.0) / 35.0) ** 2)
                     + rng.normal(0, 0.15, angle.size)).round(2),
        }))
    return pd.concat(frames, ignore_index=True)


# Example datasets offered on the sidebar. Each brings its own cheat sheet,
# since snippets are only worth clicking if they run on the columns at hand.
# Snippets are templates: {sets} is the range of set indices this example
# landed on ('3:7' after three sets already loaded) and {s0}, {s1}, ... its
# sets one by one. Sets a snippet creates (a delta, a combine) are selected by
# what the call returns, not by guessing the index they will get.
# A load appends, so whatever was loaded before is kept, just deselected.
EXAMPLES = [
    dict(key='engine', label='Engine warm-up', frame=demo_frame,
         about='Three engine runs warming up over 30 s: time, temperature, pressure, rpm',
         blurb="plot · by='sets' · plot_ymult · delta",
         start=["plot(x='time', y=['temperature', 'pressure'])"],
         snippets=[
             ("plot(x='time', y=['temperature', 'pressure'])", True),
             ("plot(x='time', y='rpm', by='sets')", True),
             ("plot_ymult(x='time', y=['temperature', 'pressure', 'rpm'])", True),
             ("select(delta(base_idx={s0}, study_indices=[{s1}, {s2}], "
              "align_on='time', delta_parms=['temperature'])); "
              "plot(x='time', y='DL_temperature')", True),
             ("select([{s0}, {s1}])", False),
             ("omit({s2})", False),
             ("restore('{sets}')", False),
             ("color({s0}, 'red')", False),
             ("var_format('pressure', linestyle='--'); "
              "plot(x='time', y='pressure')", True),
             ("summary()", False),
             ("list_sets()", False),
         ]),
    dict(key='battery', label='Battery discharge', frame=battery_frame,
         about='Four cells discharged at 0.5C to 3C: time_s, soc_pct, voltage_v, temp_c',
         blurb='table(x_in=) · plot_ymult · line · delta',
         start=["plot(x='soc_pct', y='voltage_v')"],
         snippets=[
             ("plot(x='soc_pct', y='voltage_v')", True),
             ("table(cols='voltage_v', x_col='soc_pct', x_in=[80, 50, 20], "
              "sig_figs=4)", True),
             ("plot_ymult(x='time_s', y=['voltage_v', 'temp_c'])", True),
             ("plot(x='time_s', y=['voltage_v', 'temp_c'], by='sets')", True),
             ("line('voltage_v', 3.2, label='cutoff'); "
              "plot(x='soc_pct', y='voltage_v')", True),
             ("select(delta(base_idx={s0}, study_indices=[{s3}], "
              "align_on='soc_pct', delta_parms=['voltage_v'])); "
              "plot(x='soc_pct', y='DL_voltage_v')", True),
             ("summary(cols=['voltage_v', 'temp_c'])", True),
             ("list_parms();", False),
         ]),
    dict(key='compressor', label='Compressor map', frame=compressor_frame,
         about='Four compressor speed lines, surge to choke: flow_kgs, pressure_ratio, efficiency_pct',
         blurb='contour · hue · reg_order · plot_marginal',
         start=["marker('{sets}', 'o')",
                "plot(x='flow_kgs', y='pressure_ratio')"],
         snippets=[
             ("plot(x='flow_kgs', y='pressure_ratio')", True),
             ("select(combine_sets('{sets}', title='whole map')); "
              "contour(x='flow_kgs', y='pressure_ratio', z='efficiency_pct')", True),
             ("hue('{sets}', 'efficiency_pct'); "
              "plot(x='flow_kgs', y='pressure_ratio')", True),
             ("reg_order('{sets}', 2); plot(x='flow_kgs', y='efficiency_pct')", True),
             ("plot_marginal(x='flow_kgs', y='efficiency_pct', marginal='box')", True),
             ("linestyle('{sets}', '-')", False),
             ("list_sets()", False),
         ]),
    dict(key='weather', label='Weather stations', frame=weather_frame,
         about='A year of daily weather at four stations: day, month, season, temp_c, rain_mm, wind_kmh',
         blurb='box · bar(agg=) · histogram · query',
         start=["plot(x='day', y='temp_c')"],
         snippets=[
             ("plot(x='day', y='temp_c')", True),
             ("box(x='season', y='temp_c')", True),
             ("bar(x='month', y='rain_mm', agg='sum')", True),
             ("histogram(x='wind_kmh', nbins=30)", True),
             ("query('{sets}', 'season == \"summer\"'); plot(x='day', y='temp_c')", True),
             ("query('{sets}', None)  # back to every row", True),
             ("summary(cols=['temp_c', 'rain_mm'])", True),
             ("list_sets()", False),
         ]),
    dict(key='survey', label='Inlet survey', frame=survey_frame,
         about='An inlet pressure survey, rakes round a duct at three throttle settings: '
               'angle_deg, radius_mm, pt_kpa, tt_c',
         blurb='polar · contour map · spokes · query',
         start=["polar(x='angle_deg', y='radius_mm', z='pt_kpa', by='sets', ncols=3)"],
         snippets=[
             ("polar(x='angle_deg', y='radius_mm', z='pt_kpa', by='sets', ncols=3)", True),
             ("select({s2}); polar(x='angle_deg', y='radius_mm', z='pt_kpa', "
              "interp_method='cubic')", True),
             ("line('angle_deg', 210, label='strut'); polar(x='angle_deg', "
              "y='radius_mm', z='pt_kpa', by='sets', ncols=3, "
              "direction='clockwise')", True),
             ("query('{sets}', 'radius_mm == 90'); scale('pt_kpa', (97, 101.5)); "
              "polar(x='angle_deg', y='pt_kpa')", True),
             ("linestyle('{sets}', '-')", False),
             ("query('{sets}', None); scale('pt_kpa', None)  # every row, auto range", True),
             ("polar(x='angle_deg', y='radius_mm', z=['pt_kpa', 'tt_c'], by='sets', "
              "ncols=3)", True),
             ("summary(cols=['pt_kpa', 'tt_c'])", True),
             ("list_sets()", False),
         ]),
]


def _example_fields(first, count):
    """Template fields for an example whose ``count`` sets start at ``first``."""
    fields = {f's{i}': first + i for i in range(count)}
    fields['sets'] = f'{first}:{first + count}'
    return fields


def example_commands(example, first, count):
    """What loading ``example`` runs after its ``uc.load``, and its cheat sheet,
    both with the set indices filled in."""
    fields = _example_fields(first, count)
    start = [f"select('{first}:{first + count}')"] if first else []
    start += [cmd.format(**fields) for cmd in example['start']]
    sheet = [(snippet.format(**fields), wide) for snippet, wide in example['snippets']]
    return start, sheet


# ---------------------------------------------------------------------------
# Chrome
# ---------------------------------------------------------------------------

# The app icon: a line over three bars, in the board's own palette. Written as
# SVG rather than shipped as a .png because unichart installs as flat top-level
# modules with no package data — an icon that is source can't go missing — and
# because one vector file covers the tab, the standalone window and the taskbar
# at every size.
_ICON_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="14" fill="{BG}"/>
  <g fill="{ACCENT}">
    <rect x="11.5" y="34.6" width="8.3" height="16.6" rx="2.3" opacity=".35"/>
    <rect x="26.2" y="25.6" width="8.3" height="25.6" rx="2.3" opacity=".55"/>
    <rect x="41.0" y="15.4" width="8.3" height="35.8" rx="2.3" opacity=".75"/>
  </g>
  <polyline points="10.2,39.7 24.3,28.2 37.1,33.3 53.8,14.1" fill="none"
            stroke="{INK}" stroke-width="4.8" stroke-linecap="round"
            stroke-linejoin="round"/>
  <circle cx="10.2" cy="39.7" r="3" fill="{INK}"/>
  <circle cx="53.8" cy="14.1" r="3" fill="{INK}"/>
</svg>"""

# Served off the Dash server rather than out of an ``assets/`` folder, for the
# same no-package-data reason, and because the manifest has to carry the board's
# title, which only exists once build_terminal_app has one.
_ICON_PATH = '_unichart/icon.svg'
_MANIFEST_PATH = '_unichart/manifest.webmanifest'
_GALLERY_PATH = '_unichart/gallery.html'
_OPEN_PATH = '_unichart/open'


def _gallery_file():
    """The built example gallery page, if this is a source checkout."""
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (here, os.path.dirname(here)):
        page = os.path.join(folder, 'gallery', 'index.html')
        if os.path.isfile(page):
            return page
    return None


def _serve_gallery(app):
    """Serve the example gallery off the board, and return its href.

    The board is an http page, and browsers refuse to follow a ``file://``
    link from one — so the local page has to come off this server or not at
    all. Returns ``None`` when the page isn't in this install, and the board
    then shows no gallery link rather than one to the online copy.
    """
    page = _gallery_file()
    if page is None:
        return None
    server = app.server
    prefix = app.config.routes_pathname_prefix or '/'

    @server.route(prefix + _GALLERY_PATH)
    def _unichart_gallery():
        # Read per request, so rebuilding the gallery shows up on reload
        # instead of needing the board restarted.
        with open(page, encoding='utf-8') as fh:
            return server.response_class(fh.read(), mimetype='text/html')

    return app.get_relative_path('/' + _GALLERY_PATH)


def _serve_opener(app):
    """Let the board ask its server to open one of its pages, and return the
    route's href.

    For a page that can't open a tab itself: a notebook in VS Code shows the
    board in a sandboxed iframe that allows no popups, so a ``target=_blank``
    link there does nothing at all. The kernel runs on the same machine, so
    it opens the link in the system browser instead. Only paths on this
    server are taken — the route opens the board's own pages, nothing else.
    """
    server = app.server
    prefix = app.config.routes_pathname_prefix or '/'

    @server.route(prefix + _OPEN_PATH, methods=['POST'])
    def _unichart_open():
        from flask import request
        path = request.get_data(as_text=True)
        if not path.startswith('/') or path.startswith('//'):
            return server.response_class(status=400)
        webbrowser.open(request.host_url.rstrip('/') + path)
        return server.response_class(status=204)

    return app.get_relative_path('/' + _OPEN_PATH)


def _serve_icon(app, board_title):
    """Add the icon and web app manifest routes to a built app."""
    server = app.server
    prefix = app.config.routes_pathname_prefix or '/'
    manifest = json.dumps({
        'name': board_title,
        'short_name': board_title,
        # Relative, so an installed window comes back to whichever host and
        # port this board happens to be on.
        'start_url': '.',
        'scope': '.',
        # What makes Chrome's "Install" give the board a real window of its
        # own — the same chrome-less shape --app opens directly.
        'display': 'standalone',
        'background_color': BG,
        'theme_color': BG,
        'icons': [{'src': app.get_relative_path('/' + _ICON_PATH),
                   'type': 'image/svg+xml', 'sizes': 'any'}],
    })

    @server.route(prefix + _ICON_PATH)
    def _unichart_icon():
        return server.response_class(_ICON_SVG, mimetype='image/svg+xml')

    @server.route(prefix + _MANIFEST_PATH)
    def _unichart_manifest():
        return server.response_class(manifest,
                                     mimetype='application/manifest+json')


def _index_string(requests_prefix='/'):
    """Dark-only page shell.

    Deliberately not ``unichart.dashboard._index_string``: that one stamps the
    light/dark ``data-theme`` tokens the seeded boards run on, and this board
    must not drag them along.

    Dash's ``{{%favicon%}}`` is replaced rather than joined: it points at the
    plotly mark Dash ships, and two icon links would leave which one the
    standalone window picks up to the browser.
    """
    icon = requests_prefix + _ICON_PATH
    return f"""<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}<title>{{%title%}}</title>
        <link rel="icon" type="image/svg+xml" href="{icon}">
        <link rel="apple-touch-icon" href="{icon}">
        <link rel="manifest" href="{requests_prefix + _MANIFEST_PATH}">
        <meta name="theme-color" content="{BG}">
        {{%css%}}
        <style>{_CSS}</style>
    </head>
    <body>{{%app_entry%}}
        <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
    </body>
</html>"""


SYNTAX_KW, SYNTAX_STR, SYNTAX_NUM, SYNTAX_FN = (
    SYNTAX['kw'], SYNTAX['str'], SYNTAX['num'], SYNTAX['fn'])
SYNTAX_BI, SYNTAX_OP, SYNTAX_COM, SYNTAX_NM = (
    SYNTAX['bi'], SYNTAX['op'], SYNTAX['com'], SYNTAX['nm'])

_CSS = f"""
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; height: 100%; }}
body {{
  background: {BG}; color: {INK};
  font-family: {UI_FONT}; font-size: 13px;
  -webkit-font-smoothing: antialiased;
}}
#react-entry-point, .term-app {{ height: 100%; }}

/* Three regions: a full-width bar, then sidebar | main. The two pane sizes
   are variables so the drag handles can move them without touching layout. */
.term-app {{
  display: flex; flex-direction: column;
  --side-w: 216px; --chart-h: 58%;
}}
.term-body {{ display: flex; flex: 1 1 auto; min-height: 0; }}

/* Drag handles. Each is a thin strip with a wider invisible hit area, so it
   is easy to grab without drawing a heavy divider. */
.term-grip {{ flex: none; position: relative; background: {HAIRLINE}; }}
.term-grip::after {{
  content: ''; position: absolute; z-index: 5;
}}
.term-grip:hover, .term-grip.dragging {{ background: {ACCENT}; }}
.term-grip-v {{ width: 1px; cursor: col-resize; }}
.term-grip-v::after {{ top: 0; bottom: 0; left: -3px; right: -3px; }}
.term-grip-h {{ height: 1px; cursor: row-resize; }}
.term-grip-h::after {{ left: 0; right: 0; top: -3px; bottom: -3px; }}
/* While dragging, don't let the pointer select text or land on the iframe. */
body.term-dragging {{ user-select: none; cursor: inherit; }}
body.term-dragging iframe {{ pointer-events: none; }}

/* ---- top bar ---- */
.term-top {{
  display: flex; align-items: center; gap: 12px; flex: none;
  background: {SURFACE}; border-bottom: 1px solid {HAIRLINE};
  padding: 10px 16px;
}}
.term-badge {{
  width: 28px; height: 28px; border-radius: 8px; flex: none;
  background: {ACCENT}; color: #fff; font-weight: 700; font-size: 12px;
  display: flex; align-items: center; justify-content: center;
}}
.term-name {{ font-size: 15px; font-weight: 700; letter-spacing: -0.01em; }}
.term-tagline {{ color: {MUTED}; font-size: 12.5px; }}
.term-top-actions {{ margin-left: auto; display: flex; align-items: center; gap: 10px; }}
/* ---- sidebar ---- */
.term-side {{
  width: var(--side-w); flex: none; overflow-y: auto;
  background: {SURFACE};
  padding: 14px 12px 20px;
}}
.term-label {{
  font-size: 10px; font-weight: 700; letter-spacing: 0.09em;
  text-transform: uppercase; color: {MUTED};
  margin: 0 0 8px; display: block;
}}
.term-section {{ margin-bottom: 22px; }}
.term-drop {{
  border: 1px dashed {HAIRLINE}; border-radius: 8px;
  padding: 16px 10px; text-align: center; cursor: pointer;
  color: {MUTED}; font-size: 12px; line-height: 1.5;
  transition: border-color 120ms, background 120ms;
}}
.term-drop:hover {{ border-color: {ACCENT}; background: rgba(59,130,246,0.06); }}
.term-drop b {{ color: {INK}; font-weight: 600; }}
.term-drop .exts {{ font-size: 10.5px; margin-top: 6px; display: block; }}

/* Example datasets: one button each, name over the methods it shows off. */
.term-examples {{ display: flex; flex-direction: column; gap: 5px; margin-top: 10px; }}
.term-examples-note {{ color: {MUTED}; font-size: 11px; }}
.term-example {{
  text-align: left; background: {BG}; color: {INK}; cursor: pointer;
  border: 1px solid {HAIRLINE}; border-radius: 6px; padding: 6px 8px;
  font-size: 12px; line-height: 1.35;
  transition: border-color 120ms, background 120ms;
}}
.term-example:hover {{ border-color: {ACCENT}; background: rgba(59,130,246,0.10); }}
.term-example b {{ font-weight: 600; display: block; }}
.term-example span {{ color: {MUTED}; font-size: 10.5px; font-family: {MONO_FONT};
                      display: block; margin-top: 2px; }}

/* Sets and parameters: compact tables sized for the sidebar's default width.
   Fixed layout, so a long name ellipsizes (the full text is its tooltip)
   instead of pushing the sidebar wider. */
.term-tablebox {{
  max-height: 42vh; overflow-y: auto;
  border: 1px solid {HAIRLINE}; border-radius: 6px;
}}
.term-table {{
  width: 100%; table-layout: fixed; border-collapse: collapse;
  font-size: 12px;
}}
.term-table th {{
  position: sticky; top: 0; z-index: 1; background: {SURFACE};
  color: {MUTED}; font-size: 10px; font-weight: 600; letter-spacing: 0.05em;
  text-transform: uppercase; text-align: left;
  padding: 4px 5px; border-bottom: 1px solid {HAIRLINE};
}}
.term-table td {{
  padding: 3px 5px; border-bottom: 1px solid rgba(38,42,51,0.55);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.term-table tbody tr:last-child td {{ border-bottom: none; }}
.term-table tr.term-pick {{ cursor: pointer; }}
.term-table tr.term-pick:hover td {{ background: rgba(59,130,246,0.12); }}
/* An unselected set: still listed, still pickable, visibly not plotted. */
.term-table tr.off td {{ opacity: 0.45; }}
.term-table .num {{ text-align: right; font-variant-numeric: tabular-nums; color: {MUTED}; }}
.term-table .mono {{ font-family: {MONO_FONT}; font-size: 11.5px; }}
.term-table td.sw {{ padding: 0 0 0 7px; text-overflow: clip; }}
.term-swatch {{
  display: inline-block; width: 10px; height: 10px; border-radius: 3px;
  vertical-align: middle;
}}
/* dcc.Input: Dash 4 puts the class on its wrapper and colors it from its own
   tokens, so the board's palette goes in through those. */
.term-search.dash-input-container {{
  --Dash-Fill-Inverse-Strong: {BG}; --Dash-Stroke-Strong: {HAIRLINE};
  --Dash-Fill-Interactive-Strong: {ACCENT};
  height: 28px; margin: 0 0 6px; border-radius: 6px;
  font-size: 12px; color: {INK};
}}
.term-search.dash-input-container:has(:focus-visible) {{ outline: none; }}
.term-search .dash-input-element {{ padding: 4px 8px; }}
.term-search .dash-input-element::placeholder {{ color: {MUTED}; }}
.term-empty {{ color: {MUTED}; font-size: 12px; font-style: italic; }}
/* A collapsible section's heading: the label behind a chevron that points
   down while the section is open, right while it is folded away. */
.term-fold {{
  list-style: none; cursor: pointer; user-select: none;
  display: flex; align-items: center; gap: 6px; margin: 0 0 8px;
}}
.term-fold::-webkit-details-marker {{ display: none; }}
.term-fold::before {{
  content: ''; flex: none; width: 4px; height: 4px; margin: 0 1px 0 2px;
  border: solid {MUTED}; border-width: 0 1.5px 1.5px 0;
  transform: rotate(-45deg); transition: transform 120ms;
}}
details[open] > .term-fold::before {{ transform: rotate(45deg) translate(-1px, -1px); }}
.term-fold .term-label {{ margin: 0; }}
.term-fold:hover .term-label {{ color: {INK}; }}
.term-fold:hover::before {{ border-color: {INK}; }}
.term-fold:focus-visible {{ outline: 1px solid {ACCENT}; outline-offset: 2px;
                            border-radius: 3px; }}

/* Cheat-sheet snippets: clickable, monospace, wrapped like code. */
.term-chips {{ display: flex; flex-wrap: wrap; gap: 5px; }}
.term-chip {{
  font-family: {MONO_FONT}; font-size: 11px; line-height: 1.45;
  text-align: left; white-space: pre-wrap; word-break: break-word;
  background: {BG}; color: {INK};
  border: 1px solid {HAIRLINE}; border-radius: 5px;
  padding: 4px 7px; cursor: pointer;
  transition: border-color 120ms, background 120ms;
}}
.term-chip:hover {{ border-color: {ACCENT}; background: rgba(59,130,246,0.10); }}
.term-chip.wide {{ width: 100%; }}
.term-note {{ color: {MUTED}; font-size: 11px; line-height: 1.6; }}
.term-note code {{ font-family: {MONO_FONT}; font-size: 10.5px; }}

/* ---- main: chart over terminal ---- */
.term-main {{ flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; }}
.term-chart {{
  flex: 0 0 auto; height: var(--chart-h); min-height: 120px;
  overflow: hidden;
}}
.term-chart .js-plotly-plot {{ width: 100% !important; }}
.term-console {{ flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }}
.term-scroll {{
  flex: 1 1 auto; overflow-y: auto; padding: 12px 16px;
  font-family: {MONO_FONT}; font-size: 12.5px; line-height: 1.55;
  white-space: pre-wrap; word-break: break-word;
}}
.term-in {{ color: {INK}; }}
.tk-kw {{ color: {SYNTAX_KW}; }}
.tk-str {{ color: {SYNTAX_STR}; }}
.tk-num {{ color: {SYNTAX_NUM}; }}
.tk-fn {{ color: {SYNTAX_FN}; }}
.tk-bi {{ color: {SYNTAX_BI}; }}
.tk-op {{ color: {SYNTAX_OP}; }}
.tk-com {{ color: {SYNTAX_COM}; }}
.tk-nm {{ color: {SYNTAX_NM}; }}
.tk-com {{ font-style: italic; }}
.term-in::before {{ content: '>>> '; color: {ACCENT}; }}
.term-cont::before {{ content: '... '; color: {ACCENT}; }}
.term-out {{ color: {INK}; opacity: 0.88; }}
.term-err {{ color: {ERROR}; }}
.term-rich {{ border: 0; width: 100%; background: transparent; margin: 4px 0 10px; }}

/* ---- input ---- */
.term-inputrow {{
  flex: none; display: flex; align-items: flex-start; gap: 8px;
  border-top: 1px solid {HAIRLINE}; padding: 10px 16px;
}}
.term-prompt {{
  font-family: {MONO_FONT}; font-size: 12.5px; color: {ACCENT};
  padding-top: 3px; flex: none; user-select: none;
}}
/* The live-highlight layer and the textarea are stacked in the same box and
   must agree on every metric that affects wrapping, or the colored text will
   drift out from under the caret. */
.term-inputwrap {{ position: relative; flex: 1 1 auto; min-width: 0; }}
.term-input, .term-mirror {{
  font-family: {MONO_FONT}; font-size: 12.5px; line-height: 1.55;
  padding: 0; border: 0; margin: 0;
  white-space: pre-wrap; word-break: break-word;
  min-height: 22px; max-height: 160px;
}}
.term-input {{
  position: relative; z-index: 1; display: block; width: 100%;
  resize: none; overflow-y: auto;
  background: transparent; outline: none;
  /* Transparent text with a visible caret: the color comes from the mirror
     underneath, but selection and the caret stay native. */
  color: transparent; caret-color: {INK};
}}
.term-input::selection {{ background: rgba(59,130,246,0.32); }}
.term-input::placeholder {{ color: {MUTED}; }}
.term-mirror {{
  position: absolute; inset: 0; z-index: 0;
  overflow: hidden; pointer-events: none; color: {INK};
}}

.term-btn {{
  font: inherit; font-size: 12px; color: {INK}; background: transparent;
  border: 1px solid {HAIRLINE}; border-radius: 7px; padding: 5px 12px;
  cursor: pointer; transition: border-color 120ms, background 120ms;
}}
.term-btn:hover {{ border-color: {ACCENT}; background: rgba(59,130,246,0.10); }}
/* The one button that ends things reads red on hover, not blue. */
.term-btn.danger:hover {{ border-color: {ERROR}; background: rgba(248,113,113,0.12); }}
.term-link {{ color: {MUTED}; text-decoration: none; font-size: 12.5px; }}
.term-link:hover {{ color: {INK}; }}
.hidden {{ display: none !important; }}

/* ---- the close dialog ----
   In-page rather than window.confirm(): a native box is the one unstyled
   thing that could land on this board, and it cannot say what is being lost.
   The same card carries the goodbye — once the server is going there is no
   round trip left to swap children with, so both states ship with the page
   and a class on the backdrop chooses between them. */
.term-modal {{
  position: fixed; inset: 0; z-index: 50;
  display: flex; align-items: center; justify-content: center;
  background: rgba(11,13,17,0.72);
}}
.term-modal-card {{
  background: {SURFACE}; border: 1px solid {HAIRLINE}; border-radius: 12px;
  padding: 22px 24px; width: 380px; max-width: calc(100vw - 32px);
  box-shadow: 0 18px 50px rgba(0,0,0,0.55);
}}
.term-modal-title {{ font-size: 15px; font-weight: 700; margin-bottom: 8px; }}
.term-modal-note {{ color: {MUTED}; font-size: 12.5px; line-height: 1.6; }}
.term-modal-row {{
  display: flex; justify-content: flex-end; gap: 8px; margin-top: 18px;
}}
.term-modal-row .term-btn {{ padding: 6px 14px; }}
.term-modal-row .danger {{ border-color: {ERROR}; color: {ERROR}; }}
.term-modal-row .danger:hover {{ background: rgba(248,113,113,0.12); }}
.term-modal .term-modal-bye {{ display: none; }}
.term-modal.closing .term-modal-ask {{ display: none; }}
.term-modal.closing .term-modal-bye {{ display: block; }}

::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{ background: {HAIRLINE}; border-radius: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
"""


def _fit(fig):
    """A copy of ``fig`` sized to the pane instead of its baked-in dimensions.

    unichart pins width/height on every figure (``_enforce_plot_size``) so
    notebook output is reproducible. The chart pane wants the opposite — fill
    the available width — so those are cleared on a *copy*: mutating the
    original would corrupt ``uc.last_fig`` for ``save_png`` and re-styling.
    """
    if fig is None:
        return _blank()
    clone = fig.to_dict()
    _promote_to_webgl(fig, clone.get('data') or [])
    layout = clone.setdefault('layout', {})
    layout.pop('width', None)
    layout.pop('height', None)
    layout['autosize'] = True
    layout['margin'] = {'l': 70, 'r': 30, 't': 50, 'b': 55}
    return clone


def _trace_points(trace):
    """How many points a scatter trace draws."""
    for values in (trace.x, trace.y):
        if values is not None:
            return len(values)
    return 0


def _promote_to_webgl(fig, traces):
    """Draw a figure's SVG scatters with WebGL once together they are large.

    The plotters choose Scattergl per trace (``_scatter_cls``), so twenty sets
    of 5000 rows stay SVG — 100k points per y column, each its own DOM node.
    In this pane that stalls the whole page, typing included, so here the
    threshold is judged over the figure. Points are counted on ``fig``, whose
    arrays are real arrays (its ``to_dict`` hands them back base64-encoded);
    ``traces`` — that dict's, in the same order — are the ones retyped. They
    belong to the pane's copy, never to ``uc.last_fig``.
    """
    from . import _core

    threshold = _core.WEBGL_POINT_THRESHOLD
    if threshold is None:
        return
    svg = [i for i, tr in enumerate(fig.data) if tr.type == 'scatter']
    if sum(_trace_points(fig.data[i]) for i in svg) > threshold:
        for i in svg:
            traces[i]['type'] = 'scattergl'


def _blank():
    """The empty chart pane: no axes, just a hint."""
    return {
        'data': [],
        'layout': {
            'paper_bgcolor': BG, 'plot_bgcolor': BG, 'autosize': True,
            'xaxis': {'visible': False}, 'yaxis': {'visible': False},
            'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
            'annotations': [{
                'text': 'Load data, then plot(x=…, y=…)',
                'xref': 'paper', 'yref': 'paper', 'x': 0.5, 'y': 0.5,
                'showarrow': False,
                'font': {'color': MUTED, 'size': 13, 'family': UI_FONT},
            }],
        },
    }


def _table(html, widths, head, rows):
    """A compact sidebar table. ``widths`` are CSS widths per column (None
    takes the rest); ``rows`` are ``html.Tr``."""
    return html.Div(html.Table([
        html.Colgroup([html.Col(style={'width': w} if w else None)
                       for w in widths]),
        html.Thead(html.Tr([html.Th(h) for h in head])),
        html.Tbody(rows),
    ], className='term-table'), className='term-tablebox')


def _dataset_rows(html, uc):
    """The sidebar's set table, swatches colored to match the chart's traces.

    The same facts as ``list_sets()``, narrowed for the sidebar: the full
    title, shape and query ride in the tooltip, and an unselected set is
    dimmed rather than given a column. Clicking a row puts its set number in
    the terminal.
    """
    if not uc.sets:
        return [html.Div('No data loaded.', className='term-empty')]
    rows = []
    for ds in uc.sets:
        n_rows = len(ds._masked_positions())
        tip = (f'{ds.title_format} · {n_rows} rows × '
               f'{len(ds._own_col_positions())} cols')
        if not ds.select:
            tip += ' · not selected'
        if ds.query:
            tip += f' · query: {ds.query}'
        rows.append(html.Tr([
            html.Td(html.Span(className='term-swatch',
                              style={'background': ds.color}), className='sw'),
            html.Td(str(ds.index), className='num'),
            # title_format is '0: Baseline run' — the index is already its own
            # column here, so use the bare title and don't print it twice.
            html.Td(ds.title, className='title'),
            html.Td(str(n_rows), className='num'),
        ], className='term-pick' if ds.select else 'term-pick off',
            title=tip + ' — click to put its number in the terminal',
            **{'data-insert': str(ds.index), 'data-kind': 'set'}))
    return [_table(html, ['20px', '24px', None, '44px'],
                   ['', '#', 'Title', 'Rows'], rows)]


def _parm_listing(uc):
    """The sidebar's parameter table as data, its length, and a signature.

    ``list_parms('all')`` in sidebar form: every parameter of every set, and —
    with more than one set — which sets own it. Each row is ``[name, sets,
    tooltip]``, and the browser draws the table from them (``renderParms``
    in :func:`_register`) rather than the server building it from Dash
    components: a wide file runs to thousands of parameters, and the page
    pays for every Dash component on each keystroke anywhere on the board.
    The signature lets the refresh after each command skip resending a list
    that hasn't changed.
    """
    owners = {}
    for ds in uc.sets:
        for col in ds.columns:
            owners.setdefault(col, []).append(ds.index)
    names = sorted(owners, key=lambda c: str(c).lower())
    many = len(uc.sets) > 1
    descs = uc.parm_description_dict
    rows = []
    for col in names:
        name, desc = str(col), descs.get(col)
        own = owners[col]
        sets = None
        if many:
            sets = 'all' if len(own) == len(uc.sets) else _compact_indices(own)
        rows.append([name, sets, f'{name} — {desc}' if desc else name])
    sig = str(hash(repr(rows)))
    return {'many': many, 'rows': rows}, len(rows), sig


def _chips(html, sheet):
    """Cheat-sheet buttons for ``sheet``, a list of (snippet, is_wide)."""
    return [
        html.Button(
            # Highlighted the same way the transcript is, so a snippet
            # looks identical before and after you run it.
            [s for line in _code_spans(html, snippet) for s in line],
            id={'type': 'term-chip', 'index': i}, n_clicks=0,
            className='term-chip wide' if wide else 'term-chip',
            title='Click to put this in the terminal')
        for i, (snippet, wide) in enumerate(sheet)
    ]


def _count_label(noun, count):
    """A table's heading: 'Sets · 3', or just 'Sets' when there are none.
    The browser rewrites it as a search narrows the table ('Sets · 1 of 3')."""
    return f'{noun} · {count}' if count else noun


def _fold(html, label):
    """The clickable heading of a collapsible sidebar section. The label stays
    a child of its own, since the browser rewrites its text as a search
    narrows the table."""
    return html.Summary(label, className='term-fold',
                        title='Click to show or hide this list')


def _sidebar(html, dcc, uc, cheat_sheet=True):
    parm_data, parm_count, parm_sig = _parm_listing(uc)
    # Hidden rather than left out: the dispatch callback outputs to the chips
    # and takes the example buttons as inputs, so they stay in the layout.
    hide = None if cheat_sheet else {'display': 'none'}
    return html.Div([
        html.Div([
            html.Span('Data', className='term-label'),
            dcc.Upload(
                id='term-upload', multiple=True, className='term-drop',
                children=html.Div([
                    html.B('Drop a file'), ' or click to browse',
                    html.Span('.csv · .tsv · .txt · .xlsx · .xls · .json',
                              className='exts'),
                    html.Span('or a saved session — .json · .png',
                              className='exts'),
                    html.Span('any other type: write its reader',
                              className='exts'),
                ])),
            html.Div([
                html.Span('…or try an example, with a cheat sheet to match:',
                          className='term-examples-note'),
            ] + [
                html.Button([html.B(ex['label']), html.Span(ex['blurb'])],
                            id={'type': 'term-example', 'index': i}, n_clicks=0,
                            className='term-example',
                            title=ex['about'])
                for i, ex in enumerate(EXAMPLES)
            ], className='term-examples', style=hide),
        ], className='term-section'),

        # Sets and parameters fold away under their headings, for when a
        # long list crowds out the rest of the sidebar.
        html.Details([
            _fold(html, html.Span(_count_label('Sets', len(uc.sets)),
                                  id='term-sets-label', className='term-label')),
            # A title, or set numbers in the library's own shorthand.
            dcc.Input(id='term-set-search', type='search', debounce=True,
                      placeholder='search · boost  ·  0:3  ·  -1',
                      className='term-search', autoComplete='off',
                      spellCheck=False),
            html.Div(_dataset_rows(html, uc), id='term-datasets'),
        ], open=True, className='term-section'),

        html.Details([
            _fold(html, html.Span(_count_label('Parameters', parm_count),
                                  id='term-parms-label', className='term-label')),
            # Filtered in the browser as you type — no round trip per key.
            dcc.Input(id='term-parm-search', type='search', debounce=True,
                      placeholder='search · egt  ·  *_in  ·  n?',
                      className='term-search', autoComplete='off',
                      spellCheck=False),
            # Left empty here: the browser draws the table into it from
            # term-parms-data, so no callback may ever output to its children.
            html.Div(id='term-parms'),
            dcc.Store(id='term-parms-data', data=parm_data),
            dcc.Store(id='term-parms-sig', data=parm_sig),
        ], open=True, className='term-section'),

        html.Div([
            html.Span('Cheat sheet', id='term-cheat-label', className='term-label'),
            html.Div(_chips(html, CHEAT_SHEET), id='term-chips',
                     className='term-chips'),
        ], className='term-section', style=hide),

        html.Div([
            'Click any snippet to put it in the terminal. The full ',
            html.Code('UnichartNotebook'), ' API is available as ',
            html.Code('uc'), '.',
        ], className='term-note', style=hide),
    ], className='term-side')


# ANSI (from uc.help()) rendered as spans, in this board's palette. Only the
# codes unichart actually emits are mapped; anything else is stripped rather
# than left to show up as a literal "[36m" in the transcript.
_ANSI_RE = re.compile(r'\x1b\[([0-9;]*)m')
_ANSI_STYLE = {
    '1': {'font-weight': '600'},
    '2': {'color': MUTED},
    '31': {'color': ERROR},
    '32': {'color': SYNTAX['str']},
    '33': {'color': SYNTAX['bi']},
    '36': {'color': SYNTAX['fn']},
}


def _ansi_markup(text):
    """``text`` as escaped HTML, with ANSI runs wrapped in styled spans."""
    if '\x1b' not in text:
        return _escape(text)

    def run(chunk, style):
        if not style:
            return _escape(chunk)
        css = '; '.join(f'{k}: {v}' for k, v in style.items())
        return f'<span style="{css}">{_escape(chunk)}</span>'

    out, style, pos = [], {}, 0
    for match in _ANSI_RE.finditer(text):
        chunk = text[pos:match.start()]
        if chunk:
            out.append(run(chunk, style))
        for code in (c for c in match.group(1).split(';') if c) or ['0']:
            style = {} if code == '0' else {**style, **_ANSI_STYLE.get(code, {})}
        pos = match.end()
    if text[pos:]:
        out.append(run(text[pos:], style))
    return ''.join(out)


def _entry_markup(entry):
    """One transcript entry as HTML. Rich HTML rides in an iframe.

    The transcript is drawn in the browser from these strings rather than
    sent as Dash components: highlighting makes every token of every command
    its own span, and the page pays for each Dash component on every
    keystroke, so a long session would slowly make typing lag.

    ``table``/``summary`` emit self-contained HTML carrying their own
    click-to-sort and filter scripts; ``dcc.Markdown`` would strip those, so the
    only faithful container is an iframe with its own document.
    """
    kind, text = entry.get('kind'), entry.get('text', '')
    if kind == 'in':
        lines = [''.join(f'<span class="tk-{cls}">{_escape(piece)}</span>'
                         for piece, cls in line) or '<span class="tk-nm"></span>'
                 for line in highlight(text)]
        return ''.join(
            f'<div class="{"term-in term-cont" if i else "term-in"}">{line}</div>'
            for i, line in enumerate(lines))
    if kind == 'err':
        return f'<div class="term-err">{_escape(text)}</div>'
    if kind == 'html':
        return (f'<iframe class="term-rich" '
                f'style="height: {int(entry.get("height", 320))}px" '
                f'srcdoc="{_escape(text, quote=True)}"></iframe>')
    if text:
        return f'<div class="term-out">{_ansi_markup(text)}</div>'
    return ''


def _with_markup(entries):
    """Entries as the browser receives them: each carries its ``markup``.

    A rich block's text is its markup already, so it isn't sent twice."""
    out = []
    for e in entries:
        sent = {**e, 'markup': _entry_markup(e)}
        if e.get('kind') == 'html':
            sent.pop('text', None)
        out.append(sent)
    return out


def build_terminal_app(uc, title=None, banner=True, startup=(),
                       allow_quit=None, staged=(), cheat_sheet=None):
    """Build (but do not run) the terminal explorer.

    ``startup`` is a sequence of commands run before the first paint — how the
    CLI's ``--panel`` specs and ``explore(panels=...)`` are honored, so they
    land in the transcript exactly as if they had been typed.

    ``staged`` are files unichart has no reader for. The board opens with a
    reader for them waiting in the input (see :func:`_reader_template`), the
    same as when such a file is dropped.

    ``allow_quit`` offers the top bar's close button, and makes ``exit()`` in
    the terminal mean it. It defaults to "unless we are in a Jupyter kernel",
    which is the same question :func:`_shutdown` is documented against: closing
    the board ends the process, and in a kernel that process is the user's.

    ``cheat_sheet`` shows the sidebar's example datasets and cheat sheet. It
    defaults to "only if the notebook has no data yet": they are a way in for
    a first look, and noise for someone who arrives with their data loaded.
    """
    (Dash, dcc, html, dash_table, no_update,
     Input, Output, State, MATCH, ALL) = _require_dash()
    from dash import ctx

    board_title = title or 'Unichart'
    if cheat_sheet is None:
        cheat_sheet = not uc.sets
    if allow_quit is None:
        from .dashboard import _in_notebook
        allow_quit = not _in_notebook()
    session = Session(uc, allow_quit=allow_quit)
    uploads = _uploads_dir()

    entries = []
    if banner:
        entries += [{'kind': 'out', 'text': line} for line in BANNER]
    figure = _blank()
    for command in startup:
        entries.append({'kind': 'in', 'text': command})
        result = session.run(command)
        entries.extend(_result_entries(result))
        if result and result.get('figure') is not None:
            figure = _fit(result['figure'])
    staged = [str(p) for p in staged]
    entries += [{'kind': 'out', 'text': _staged_note(p)} for p in staged]

    app = Dash(__name__, title=board_title,
               update_title=None, suppress_callback_exceptions=True)
    app.index_string = _index_string(app.config.requests_pathname_prefix)
    _serve_icon(app, board_title)
    gallery_href = _serve_gallery(app)
    open_href = _serve_opener(app)
    app.layout = html.Div([
        html.Div([
            html.Div('UC', className='term-badge'),
            html.Span(board_title, className='term-name'),
            html.Span('plotting terminal — Python at your prompt',
                      className='term-tagline'),
            html.Div([
                html.Button('⧉ copy chart', id='term-copy', n_clicks=0,
                            className='term-btn',
                            title='Copy the current chart to the clipboard '
                                  'as a PNG'),
                html.Button('💾 save session', id='term-save', n_clicks=0,
                            className='term-btn',
                            title='Download the data, formatting and current '
                                  'plot as a session file you can reopen'),
                html.A('Gallery', href=gallery_href,
                       target='_blank', className='term-link',
                       **{'data-fallback': open_href},
                       title='Example gallery — every plot type on one page, '
                             'each with the code that drew it')
                if gallery_href else None,
                html.Button('✕ close', id='term-quit', n_clicks=0,
                            className='term-btn danger',
                            title='Close unichart — stops the server and ends '
                                  'the program')
                if allow_quit else None,
            ], className='term-top-actions'),
        ], className='term-top'),

        html.Div([
            _sidebar(html, dcc, uc, cheat_sheet),
            html.Div(id='term-grip-v', className='term-grip term-grip-v',
                     title='Drag to resize · double-click to reset'),
            html.Div([
                html.Div(
                    dcc.Graph(id='term-chart', figure=figure,
                              config={'displaylogo': False, 'responsive': True,
                                      'displayModeBar': 'hover'},
                              style={'height': '100%', 'width': '100%'}),
                    className='term-chart'),
                html.Div(id='term-grip-h', className='term-grip term-grip-h',
                         title='Drag to resize · double-click to reset'),
                html.Div([
                    # Filled in the browser from term-entries (see
                    # _entry_markup); no callback outputs its children.
                    html.Div(id='term-scroll', className='term-scroll'),
                    html.Div([
                        html.Span('>>>', className='term-prompt'),
                        html.Div([
                            html.Div(id='term-mirror', className='term-mirror'),
                            dcc.Textarea(
                                id='term-input',
                                value=_reader_template(staged) if staged else '',
                                className='term-input',
                                placeholder='uc.help()  ·  Enter runs · '
                                            'Shift+Enter for a new line · '
                                            '↑ history',
                                spellCheck=False),
                        ], className='term-inputwrap'),
                        html.Button('run', id='term-submit', n_clicks=0,
                                    className='hidden'),
                    ], className='term-inputrow'),
                ], className='term-console'),
            ], className='term-main'),
        ], className='term-body'),

        # The entries the latest action added — not the whole transcript,
        # which only the page keeps, so a long session doesn't make every
        # command resend everything before it.
        dcc.Store(id='term-entries', data=_with_markup(entries)),
        dcc.Store(id='term-history', data=list(startup)),
        # The cheat sheet on screen, so a chip click resolves to its own text
        # after an example has swapped the sheet out.
        dcc.Store(id='term-cheats', data=[list(c) for c in CHEAT_SHEET]),
        dcc.Download(id='term-save-download'),
        html.Div(id='term-side-sink', className='hidden'),
        html.Div(id='term-parms-sink', className='hidden'),
        html.Div(id='term-resize-sink', className='hidden'),
        _quit_dialog(html) if allow_quit else None,
    ], className='term-app')

    _register(app, uc, session, uploads, board_title, dcc, html, ctx,
              Input, Output, State, ALL, no_update, allow_quit)
    return app


def _quit_dialog(html):
    """The close confirmation, and the goodbye it turns into.

    Both states are in the page from the start; see the ``.term-modal`` rules
    for why. It ships hidden and is shown by the callback.
    """
    return html.Div(
        html.Div([
            html.Div([
                html.Div('Close unichart?', className='term-modal-title'),
                html.Div(
                    'This stops the server and ends the program. Loaded data, '
                    'formatting and the current plot go with it — save a '
                    'session first if you want them back.',
                    className='term-modal-note'),
                html.Div([
                    html.Button('Cancel', id='term-quit-cancel', n_clicks=0,
                                className='term-btn'),
                    html.Button('Close unichart', id='term-quit-yes',
                                n_clicks=0, className='term-btn danger'),
                ], className='term-modal-row'),
            ], className='term-modal-ask'),
            html.Div([
                html.Div('Unichart closed.', className='term-modal-title'),
                html.Div('The server has stopped. You can close this window.',
                         className='term-modal-note'),
            ], className='term-modal-bye'),
            html.Div(id='term-quit-sink', className='hidden'),
        ], className='term-modal-card'),
        id='term-quit-modal', className='term-modal hidden')


def _result_entries(result):
    """Turn one Session.run() result into transcript entries."""
    if not result:
        return []
    out = []
    if result.get('text'):
        out.append({'kind': 'out', 'text': result['text']})
    for block in result.get('html') or []:
        # Rich blocks carry their own <style>/<script>; give the iframe a
        # height that suits a table without swallowing the whole console.
        out.append({'kind': 'html', 'text': block, 'height': 340})
    if result.get('error'):
        out.append({'kind': 'err', 'text': result['error']})
    return out


def _register(app, uc, session, uploads, board_title, dcc, html, ctx,
              Input, Output, State, ALL, no_update, allow_quit=False):
    """Wire the board up.

    One callback owns the transcript, the chart, the dataset list and the input
    box, because every trigger — submitting a command, clicking a cheat-sheet
    chip, dropping a file, loading an example, saving the session — wants to write
    some subset of those four. Splitting them would mean two callbacks with the
    same Output, which Dash rejects at construction. Saving could have been a
    sibling callback, since ``dcc.Download`` is an Output of its own — but then
    it would be the one action on this board that leaves no trace in the
    transcript.
    """

    @app.callback(
        Output('term-entries', 'data'),
        Output('term-chart', 'figure'),
        Output('term-datasets', 'children'),
        Output('term-input', 'value'),
        Output('term-history', 'data'),
        Output('term-upload', 'contents'),
        Output('term-save-download', 'data'),
        Output('term-chips', 'children'),
        Output('term-cheat-label', 'children'),
        Output('term-cheats', 'data'),
        Input('term-submit', 'n_clicks'),
        Input('term-upload', 'contents'),
        Input({'type': 'term-example', 'index': ALL}, 'n_clicks'),
        Input('term-save', 'n_clicks'),
        Input({'type': 'term-chip', 'index': ALL}, 'n_clicks'),
        State('term-cheats', 'data'),
        State('term-input', 'value'),
        State('term-history', 'data'),
        State('term-upload', 'filename'),
        prevent_initial_call=True,
    )
    def _dispatch(submit_n, upload_contents, example_clicks, save_n, chip_clicks,
                  cheats, source, history, upload_names):
        trigger = ctx.triggered_id
        entries = []               # what this action adds to the transcript
        history = list(history or [])
        download = no_update
        sheet = no_update          # the cheat sheet changes only with an example
        nothing = (no_update,) * 10

        # A chip only stages text in the input — the user still presses Enter,
        # so a mis-click is editable rather than immediately executed.
        if isinstance(trigger, dict) and trigger.get('type') == 'term-chip':
            if not any(chip_clicks or []):
                return nothing
            snippet = (cheats or CHEAT_SHEET)[trigger['index']][0]
            return (no_update, no_update, no_update,
                    snippet) + (no_update,) * 6

        commands = []
        staged = []                # dropped files waiting on a reader
        if trigger == 'term-upload':
            if not upload_contents:
                return nothing
            for contents, name in zip(upload_contents, upload_names or []):
                try:
                    raw = _upload_bytes(contents)
                except Exception as exc:                  # noqa: BLE001
                    entries.append({'kind': 'err', 'text': f'{name}: {exc}'})
                    continue
                path = uploads / Path(name).name
                # A drop is not always data: a session .json or a save_png image
                # is a whole board to restore, not a frame to read. Sniff the
                # bytes rather than the extension — .json is both formats — and
                # write them through untouched, since a session is not
                # frame-shaped and a PNG would not survive a to_csv.
                if sniff_session(raw) is not None:
                    path.write_bytes(raw)
                    commands.append(f'uc.load_session({str(path)!r})')
                    continue
                # A type with a reader registered on the notebook (a built-in
                # one it overrides included), or one the board can't read in
                # memory, goes to disk as it came. It loads through the
                # notebook's reader, or with none, a reader is staged in the
                # input for the user to finish — the board asking for one.
                suffix = path.suffix.lower()
                if suffix in uc._custom_readers or suffix not in _UPLOAD_READERS:
                    path.write_bytes(raw)
                    if uc._reader_for(path)[0] is not None:
                        commands.append(f'uc.load({str(path)!r})')
                    else:
                        staged.append(str(path))
                        entries.append({'kind': 'out', 'text': _staged_note(path)})
                    continue
                try:
                    frame = _df_from_upload(contents, name)
                except Exception as exc:                  # noqa: BLE001
                    entries.append({'kind': 'err', 'text': f'{name}: {exc}'})
                    continue
                # Land the upload on disk so the command reads like a real
                # load — and so re-running it from history actually works.
                frame.to_csv(path, index=False)
                commands.append(f'uc.load({str(path)!r})')
        elif isinstance(trigger, dict) and trigger.get('type') == 'term-example':
            if not any(example_clicks or []):
                return nothing
            example = EXAMPLES[trigger['index']]
            frame = example['frame']()
            path = uploads / f"example-{example['key']}.csv"
            frame.to_csv(path, index=False)
            # Appended, not swapped in: what was loaded stays, deselected, and
            # the example's snippets name the set indices it actually got.
            first = len(uc.sets)
            start, cheat_list = example_commands(example, first,
                                                 frame['SETNUMBER'].nunique())
            commands.append(f'uc.load({str(path)!r})')
            commands.extend(start)
            sheet = cheat_list
        elif trigger == 'term-save':
            if not save_n or not uc.sets:
                return nothing
            # Written to the uploads dir and then handed to the browser: the
            # board is served to a browser, so "save" has to mean a download,
            # not a file left on whichever machine is running the server. Named
            # after the board because the uploads dir is per *process*, and two
            # explore() calls in one notebook share it.
            save_path = uploads / f'{_slug(board_title)}-session.json'
            commands.append(f'uc.save_session({str(save_path)!r})')
        else:
            if not (source or '').strip():
                return nothing
            commands.append(source.strip())

        # A command that draws nothing leaves the chart as it is. Not read
        # back as State: that would upload the whole drawn figure — megabytes
        # for a few sets of a few thousand rows — on every command.
        figure = no_update
        for command in commands:
            entries.append({'kind': 'in', 'text': command})
            result = session.run(command)
            entries.extend(_result_entries(result))
            if result and result.get('figure') is not None:
                figure = _fit(result['figure'])
            if command != (history[-1] if history else None):
                history.append(command)

        if trigger == 'term-save' and save_path.exists():
            download = dcc.send_file(str(save_path))

        if sheet is no_update:
            sheet_out = (no_update,) * 3
        else:
            sheet_out = (_chips(html, sheet), f"Cheat sheet · {example['label']}",
                         [list(c) for c in sheet])
        return (_with_markup(entries), figure,
                _dataset_rows(html, uc),
                _reader_template(staged) if staged else '',
                history, None, download) + sheet_out

    # The parameter table follows the data. term-datasets is rewritten after
    # every command, so that is the cue; the signature keeps a command that
    # changed nothing listed from resending the list (and resetting the
    # table's scroll position).
    @app.callback(
        Output('term-parms-data', 'data'),
        Output('term-parms-sig', 'data'),
        Input('term-datasets', 'children'),
        State('term-parms-sig', 'data'),
        prevent_initial_call=True,
    )
    def _parms(_datasets, sig):
        with _RUN_LOCK:
            data, _count, new_sig = _parm_listing(uc)
        if new_sig == sig:
            return no_update, no_update
        return data, new_sig

    # The sidebar's search and click-to-insert. Wired once, on first paint;
    # the search runs on every keystroke and again whenever the table is
    # replaced, all in the browser. Raw string: the regexes below are
    # backslashes all the way down.
    app.clientside_callback(
        r"""
        function(_) {
            if (window._uniSide) { return window.dash_clientside.no_update; }
            window._uniSide = true;
            // Is the caret inside a string literal? Python-ish: quotes, escapes,
            // comments; a newline ends a single-quoted string.
            const openQuote = function(src) {
                let q = null;
                for (let i = 0; i < src.length; i++) {
                    const c = src[i];
                    if (q) {
                        if (c === '\\') { i++; }
                        else if (c === q || c === '\n') { q = null; }
                    } else if (c === '#') {
                        const nl = src.indexOf('\n', i);
                        if (nl === -1) { return null; }
                        i = nl;
                    } else if (c === "'" || c === '"') {
                        q = c;
                    }
                }
                return q;
            };
            const insert = function(text, kind) {
                const ta = document.getElementById('term-input');
                if (!ta) { return; }
                const v = ta.value || '';
                let before = v.slice(0, ta.selectionStart);
                const after = v.slice(ta.selectionEnd);
                // A parameter goes in as a string, unless the caret is
                // already inside one; a set goes in as its number.
                let piece = text;
                const inString = openQuote(before) !== null;
                if (kind === 'parm' && !inString) {
                    // JSON's escapes are Python's, for names a plain
                    // '...' would mangle.
                    piece = /['\\]/.test(text) ? JSON.stringify(text)
                                                : "'" + text + "'";
                }
                // Picking several in a row builds a list: after a closed
                // string or a number, add the comma.
                if (!inString) {
                    const trimmed = before.replace(/\s+$/, '');
                    if (/['"]$/.test(trimmed) || /(^|[^\w.])\d+$/.test(trimmed)) {
                        before = trimmed + ', ';
                    }
                }
                const proto = Object.getPrototypeOf(ta);
                Object.getOwnPropertyDescriptor(proto, 'value').set
                    .call(ta, before + piece + after);
                ta.dispatchEvent(new Event('input', {bubbles: true}));
                const caret = before.length + piece.length;
                ta.focus();
                setTimeout(function() {
                    ta.selectionStart = ta.selectionEnd = caret;
                }, 0);
            };
            // A name search, as list_parms does it: a substring, or with
            // * / ? a wildcard pattern for the whole name, ignoring case.
            const nameTest = function(q) {
                const pat = q.replace(/[.+^${}()|[\]\\]/g, '\\$&')
                             .replace(/\*/g, '.*').replace(/\?/g, '.');
                const re = new RegExp(/[*?]/.test(q) ? '^' + pat + '$' : pat, 'i');
                return function(text) { return re.test(text || ''); };
            };
            // Set numbers in the library's range shorthand ("2", "0:3",
            // "0,3,7:", "::2", "-1"), with Python's slice rules, as
            // _get_uset_slice reads them. Anything else is not a selector —
            // nor is one that picks no set, so "2024" still finds a title.
            const TOKEN = /^\s*-?\d*\s*(?::\s*-?\d*\s*){0,2}$/;
            const setSelector = function(q, n) {
                const tokens = q.split(',');
                if (!/\d/.test(q) || !tokens.every(function(t) {
                    return t.trim() && TOKEN.test(t);
                })) { return null; }
                const picked = new Set();
                for (const token of tokens) {
                    const parts = token.split(':').map(function(x) {
                        x = x.trim();
                        return x === '' || x === '-' ? null : parseInt(x, 10);
                    });
                    if (parts.length === 1) {
                        const k = parts[0] === null ? NaN
                            : (parts[0] < 0 ? parts[0] + n : parts[0]);
                        if (k >= 0 && k < n) { picked.add(k); }
                        continue;
                    }
                    const step = parts[2] == null ? 1 : parts[2];
                    if (step === 0) { continue; }
                    const fix = function(v, dflt, lo, hi) {
                        if (v == null) { return dflt; }
                        if (v < 0) { v += n; }
                        return Math.min(Math.max(v, lo), hi);
                    };
                    let a, b;
                    if (step > 0) { a = fix(parts[0], 0, 0, n); b = fix(parts[1], n, 0, n); }
                    else { a = fix(parts[0], n - 1, -1, n - 1); b = fix(parts[1], -1, -1, n - 1); }
                    for (let k = a; step > 0 ? k < b : k > b; k += step) { picked.add(k); }
                }
                if (!picked.size) { return null; }
                return function(index) { return picked.has(index); };
            };
            const filterTable = function(boxId, inputId, labelId, noun, makeTest) {
                const box = document.getElementById(boxId);
                const input = document.getElementById(inputId);
                const label = document.getElementById(labelId);
                if (!box || !input) { return; }
                const q = input.value.trim();
                const rows = box.querySelectorAll('tbody tr');
                const test = q ? makeTest(q, rows.length) : null;
                let shown = 0;
                rows.forEach(function(tr) {
                    const ok = !test || test(tr);
                    tr.style.display = ok ? '' : 'none';
                    if (ok) { shown++; }
                });
                if (label) {
                    label.textContent = !rows.length ? noun : noun + ' · '
                        + (q ? shown + ' of ' + rows.length : rows.length);
                }
            };
            // The parameter table, drawn from term-parms-data rather than
            // shipped as Dash components (see _parm_listing). A wide file has
            // thousands of names, so only the first rows matching the search
            // go in the DOM, and more are appended as the table scrolls.
            const PARM_CHUNK = 300;
            const escHtml = function(s) {
                return String(s == null ? '' : s).replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;').replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;');
            };
            const parmRow = function(r, many) {
                return '<tr class="term-pick" data-kind="parm" data-insert="'
                    + escHtml(r[0]) + '" title="' + escHtml(r[2]) + '">'
                    + '<td class="mono">' + escHtml(r[0]) + '</td>'
                    + (many ? '<td class="num">' + escHtml(r[1]) + '</td>' : '')
                    + '</tr>';
            };
            const renderParms = function() {
                const box = document.getElementById('term-parms');
                const data = window._uniParmData;
                if (!box || !data) { return; }
                const input = document.getElementById('term-parm-search');
                const label = document.getElementById('term-parms-label');
                const rows = data.rows || [];
                const q = input ? input.value.trim() : '';
                let matches = rows;
                if (q) {
                    const t = nameTest(q);
                    matches = rows.filter(function(r) { return t(r[0]); });
                }
                if (label) {
                    label.textContent = !rows.length ? 'Parameters' : 'Parameters · '
                        + (q ? matches.length + ' of ' + rows.length : rows.length);
                }
                if (!rows.length) {
                    box.innerHTML = '<div class="term-empty">No parameters yet.</div>';
                    return;
                }
                const many = !!data.many;
                box.innerHTML = '<div class="term-tablebox"><table class="term-table">'
                    + '<colgroup><col>' + (many ? '<col style="width: 62px">' : '')
                    + '</colgroup><thead><tr><th>Parameter</th>'
                    + (many ? '<th>Sets</th>' : '') + '</tr></thead>'
                    + '<tbody></tbody></table></div>';
                const scroller = box.firstChild;
                const body = scroller.querySelector('tbody');
                let drawn = 0;
                const more = function() {
                    if (drawn >= matches.length) { return; }
                    const next = matches.slice(drawn, drawn + PARM_CHUNK);
                    body.insertAdjacentHTML('beforeend', next.map(function(r) {
                        return parmRow(r, many);
                    }).join(''));
                    drawn += next.length;
                };
                more();
                scroller.addEventListener('scroll', function() {
                    if (scroller.scrollTop + scroller.clientHeight
                            > scroller.scrollHeight - 400) { more(); }
                });
            };
            window._uniFilter = function() {
                renderParms();
                filterTable('term-datasets', 'term-set-search', 'term-sets-label',
                            'Sets', function(q, n) {
                    const sel = setSelector(q, n);
                    if (sel) {
                        return function(tr) { return sel(Number(tr.dataset.insert)); };
                    }
                    const t = nameTest(q);
                    return function(tr) {
                        const cell = tr.querySelector('td.title');
                        return t(cell ? cell.textContent : '');
                    };
                });
            };
            document.addEventListener('input', function(e) {
                const id = e.target && e.target.id;
                if (id === 'term-parm-search' || id === 'term-set-search') {
                    window._uniFilter();
                }
            });
            // A row is a pick, not a place to put the focus: keeping the
            // mousedown from landing leaves the caret in the terminal input,
            // where the click then inserts.
            const pickOf = function(e) {
                return e.target.closest && e.target.closest('.term-side [data-insert]');
            };
            document.addEventListener('mousedown', function(e) {
                if (pickOf(e)) { e.preventDefault(); }
            });
            document.addEventListener('click', function(e) {
                const row = pickOf(e);
                if (row) { insert(row.dataset.insert, row.dataset.kind); }
            });
            // A new-tab link opens through window.open, which comes back
            // null where the page may not open one (VS Code's notebook
            // iframe); the server then opens it in the system browser.
            document.addEventListener('click', function(e) {
                const a = e.target.closest && e.target.closest('a[data-fallback]');
                if (!a) { return; }
                e.preventDefault();
                let tab = null;
                try { tab = window.open(a.href, '_blank'); } catch (err) {}
                if (!tab) {
                    fetch(a.dataset.fallback,
                          {method: 'POST', body: a.getAttribute('href')});
                }
            });
            // The parameter list may have landed before this ran.
            window._uniFilter();
            return window.dash_clientside.no_update;
        }
        """,
        Output('term-side-sink', 'className'),
        Input('term-side-sink', 'id'),
    )

    # Draw the parameter table when its list arrives, and re-run both
    # searches when either table changes. Called on load too: that is how the
    # parameter list first reaches the page.
    app.clientside_callback(
        """
        function(parms, _sets) {
            window._uniParmData = parms;
            if (window._uniFilter) { window._uniFilter(); }
            return window.dash_clientside.no_update;
        }
        """,
        Output('term-parms-sink', 'className'),
        Input('term-parms-data', 'data'),
        Input('term-datasets', 'children'),
    )

    # Enter runs, Shift+Enter adds a line, up/down walks history. All of it is
    # clientside: an arrow key must never cost a server round trip, and the
    # textarea needs a real keydown listener to tell Enter from Shift+Enter.
    # Raw string: the highlighter's regex is all backslashes, and Python must
    # not touch them on the way to the browser.
    app.clientside_callback(
        r"""
        function(history) {
            // Live highlighter for the input line. It mirrors the Python
            // tokenizer that paints the transcript: one line of input doesn't
            // need the stdlib's precision, and both sides emit the same tk-*
            // classes, so the two can never disagree on color.
            const KW = new Set(['False','None','True','and','as','assert','async',
                'await','break','class','continue','def','del','elif','else',
                'except','finally','for','from','global','if','import','in','is',
                'lambda','nonlocal','not','or','pass','raise','return','try',
                'while','with','yield','match','case']);
            const BI = new Set(['abs','all','any','bool','dict','dir','enumerate',
                'filter','float','format','getattr','hasattr','help','id','input',
                'int','isinstance','len','list','map','max','min','next','object',
                'open','ord','print','range','repr','reversed','round','set',
                'sorted','str','sum','tuple','type','zip']);
            const esc = function(s) {
                return s.replace(/&/g, '&amp;').replace(/</g, '&lt;')
                        .replace(/>/g, '&gt;');
            };
            const paint = function(ta) {
                const mirror = document.getElementById('term-mirror');
                if (!mirror) { return; }
                const src = ta.value || '';
                // Ordered alternatives: comment, string, number, name, gap,
                // operator. Unterminated quotes match to end-of-line on
                // purpose so a half-typed string still colors as one.
                const RE = /(#[^\n]*)|([rRbBuUfF]{0,3}(?:'{3}[\s\S]*?'{3}|"{3}[\s\S]*?"{3}|'(?:\\.|[^'\\\n])*'?|"(?:\\.|[^"\\\n])*"?))|(\b\d[\w.]*)|([A-Za-z_]\w*)|(\s+)|([^\s\w])/g;
                let html = '', m;
                while ((m = RE.exec(src)) !== null) {
                    if (m[0] === '') { RE.lastIndex++; continue; }
                    let cls = 'nm';
                    if (m[1]) { cls = 'com'; }
                    else if (m[2]) { cls = 'str'; }
                    else if (m[3]) { cls = 'num'; }
                    else if (m[4]) {
                        const gap = src.slice(RE.lastIndex).match(/^\s*/)[0].length;
                        const after = src[RE.lastIndex + gap];
                        cls = KW.has(m[4]) ? 'kw'
                            : (after === '(' ? 'fn'
                            : (BI.has(m[4]) ? 'bi' : 'nm'));
                    }
                    else if (m[6]) { cls = 'op'; }
                    html += '<span class="tk-' + cls + '">' + esc(m[0]) + '</span>';
                }
                // A trailing newline needs a spacer or the last line collapses.
                mirror.innerHTML = html + '\n';
                mirror.scrollTop = ta.scrollTop;
            };

            const ta = document.getElementById('term-input');
            const btn = document.getElementById('term-submit');
            if (!ta || !btn) { return window.dash_clientside.no_update; }
            window._uniHist = history || [];
            if (!ta.dataset.wired) {
                ta.dataset.wired = '1';
                window._uniPos = null;
                const setValue = function(v) {
                    // React owns this input, so write through the native
                    // setter and fire `input` or Dash never sees the change.
                    const proto = Object.getPrototypeOf(ta);
                    const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
                    setter.call(ta, v);
                    ta.dispatchEvent(new Event('input', {bubbles: true}));
                    setTimeout(function() {
                        ta.selectionStart = ta.selectionEnd = ta.value.length;
                    }, 0);
                };
                // Keep the colored layer aligned when the box scrolls.
                ta.addEventListener('scroll', function() {
                    const mirror = document.getElementById('term-mirror');
                    if (mirror) { mirror.scrollTop = ta.scrollTop; }
                });
                ta.addEventListener('keydown', function(e) {
                    const hist = window._uniHist || [];
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        window._uniPos = null;
                        if (ta.value.trim()) { btn.click(); }
                        return;
                    }
                    // Only walk history from the edges, so the arrows still
                    // move the caret inside a multi-line command.
                    if (e.key === 'ArrowUp' && hist.length) {
                        if (ta.selectionStart !== 0) { return; }
                        e.preventDefault();
                        window._uniPos = (window._uniPos === null)
                            ? hist.length - 1 : Math.max(0, window._uniPos - 1);
                        setValue(hist[window._uniPos]);
                    } else if (e.key === 'ArrowDown' && window._uniPos !== null) {
                        if (ta.selectionStart !== ta.value.length) { return; }
                        e.preventDefault();
                        window._uniPos += 1;
                        if (window._uniPos >= hist.length) {
                            window._uniPos = null; setValue('');
                        } else {
                            setValue(hist[window._uniPos]);
                        }
                    }
                });
            }
            // Grow the box with its content, up to the CSS max-height.
            ta.style.height = 'auto';
            ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
            paint(ta);
            return window.dash_clientside.no_update;
        }
        """,
        Output('term-input', 'placeholder'),
        Input('term-history', 'data'),
        Input('term-input', 'value'),
    )

    # Pane resizing. Entirely clientside: a drag fires dozens of events a
    # second and none of them should cost a server round trip. The two sizes
    # live as CSS variables on the app root, so a drag is one style write.
    app.clientside_callback(
        """
        function(_) {
            const root = document.querySelector('.term-app');
            if (!root || root.dataset.grips) {
                return window.dash_clientside.no_update;
            }
            root.dataset.grips = '1';

            const DEFAULTS = {'--side-w': '216px', '--chart-h': '58%'};
            const KEY = 'unichart.panes';

            const save = function() {
                try {
                    const v = {};
                    for (const k in DEFAULTS) { v[k] = root.style.getPropertyValue(k) || ''; }
                    localStorage.setItem(KEY, JSON.stringify(v));
                } catch (e) { /* private window, blocked storage — not fatal */ }
            };
            try {
                const saved = JSON.parse(localStorage.getItem(KEY) || '{}');
                for (const k in DEFAULTS) {
                    if (saved[k]) { root.style.setProperty(k, saved[k]); }
                }
            } catch (e) { /* ignore malformed or unreadable state */ }

            // Plotly only reflows on a window resize event, and the graph is
            // configured responsive, so nudge it as the pane changes.
            const reflow = function() { window.dispatchEvent(new Event('resize')); };

            const wire = function(grip, apply, reset) {
                if (!grip) { return; }
                grip.addEventListener('pointerdown', function(e) {
                    e.preventDefault();
                    grip.setPointerCapture(e.pointerId);
                    grip.classList.add('dragging');
                    document.body.classList.add('term-dragging');
                });
                grip.addEventListener('pointermove', function(e) {
                    if (!grip.classList.contains('dragging')) { return; }
                    apply(e);
                    reflow();
                });
                const stop = function(e) {
                    if (!grip.classList.contains('dragging')) { return; }
                    grip.classList.remove('dragging');
                    document.body.classList.remove('term-dragging');
                    try { grip.releasePointerCapture(e.pointerId); } catch (err) {}
                    reflow(); save();
                };
                grip.addEventListener('pointerup', stop);
                grip.addEventListener('pointercancel', stop);
                grip.addEventListener('dblclick', function() {
                    reset(); reflow(); save();
                });
            };

            const clamp = function(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); };

            wire(document.getElementById('term-grip-v'),
                 function(e) {
                     // Sidebar spans from the viewport's left edge, so the
                     // pointer's x is the width directly.
                     const w = clamp(e.clientX, 150, window.innerWidth - 320);
                     root.style.setProperty('--side-w', w + 'px');
                 },
                 function() { root.style.setProperty('--side-w', DEFAULTS['--side-w']); });

            wire(document.getElementById('term-grip-h'),
                 function(e) {
                     const main = document.querySelector('.term-main');
                     if (!main) { return; }
                     const top = main.getBoundingClientRect().top;
                     // Leave room for the input row and a couple of lines of
                     // scrollback, so the console can never be dragged away.
                     const h = clamp(e.clientY - top, 120,
                                     main.clientHeight - 140);
                     root.style.setProperty('--chart-h', h + 'px');
                 },
                 function() { root.style.setProperty('--chart-h', DEFAULTS['--chart-h']); });

            return window.dash_clientside.no_update;
        }
        """,
        Output('term-resize-sink', 'className'),
        Input('term-resize-sink', 'id'),
    )

    # Copy the chart to the clipboard as a PNG. Clientside because both halves
    # have to happen in the browser: Plotly.toImage rasterizes the figure that
    # is already rendered (no kaleido round trip, and WebGL traces come out
    # right because plotly does its own scene capture), and the Clipboard API
    # only permits a write from inside a user gesture.
    app.clientside_callback(
        r"""
        function(n) {
            const nu = window.dash_clientside.no_update;
            if (!n) { return nu; }
            const btn = document.getElementById('term-copy');
            const flash = function(msg) {
                if (!btn) { return; }
                btn.textContent = msg;
                setTimeout(function() { btn.textContent = '⧉ copy chart'; }, 1500);
            };
            const box = document.getElementById('term-chart');
            const gd = box && box.querySelector('.js-plotly-plot');
            // An empty board still has a graph, but with no traces there is
            // nothing worth putting on the clipboard.
            if (!gd || !gd.data || !gd.data.length) { flash('✗ no chart'); return nu; }
            if (!navigator.clipboard || !window.ClipboardItem) {
                flash('✗ no clipboard'); return nu;
            }
            // Rasterize at 2x against the board's own background, so the PNG
            // matches what is on screen instead of landing transparent.
            const rect = gd.getBoundingClientRect();
            const png = window.Plotly.toImage(gd, {
                    format: 'png', scale: 2,
                    width: Math.round(rect.width), height: Math.round(rect.height)})
                .then(function(url) { return fetch(url); })
                .then(function(r) { return r.blob(); });
            // ClipboardItem wraps the *promise* so the write begins inside the
            // click gesture even though rasterizing takes a moment — Safari
            // requires this, and it is harmless elsewhere.
            navigator.clipboard.write([new ClipboardItem({'image/png': png})])
                .then(function() { flash('✓ copied'); },
                      function() { flash('✗ failed'); });
            return nu;
        }
        """,
        Output('term-copy', 'children'),
        Input('term-copy', 'n_clicks'),
    )

    # Append what the latest action added to the transcript, and keep the
    # newest output in view. Appended, never redrawn: earlier rich blocks are
    # iframes, and redrawing would reload every one of them.
    app.clientside_callback(
        """
        function(entries) {
            const el = document.getElementById('term-scroll');
            if (el && entries && entries.length) {
                el.insertAdjacentHTML('beforeend', entries.map(function(e) {
                    return e.markup || '';
                }).join(''));
                setTimeout(function() { el.scrollTop = el.scrollHeight; }, 30);
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('term-scroll', 'className'),
        Input('term-entries', 'data'),
    )

    if not allow_quit:
        return

    # Close the board. Deliberately its own callback rather than another branch
    # of _dispatch: this one has to answer *before* the process goes away, and
    # it shares none of that callback's outputs.
    @app.callback(
        Output('term-quit-modal', 'className'),
        Input('term-quit', 'n_clicks'),
        Input('term-quit-cancel', 'n_clicks'),
        Input('term-quit-yes', 'n_clicks'),
        prevent_initial_call=True,
    )
    def _quit(open_n, cancel_n, confirm_n):
        trigger = ctx.triggered_id
        if trigger == 'term-quit-yes':
            _shutdown()
            # The card's goodbye state: this is the last thing the board says,
            # and it has to be on screen before the server stops answering.
            return 'term-modal closing'
        if trigger == 'term-quit':
            return 'term-modal'
        # Canceled — and anything unexpected is read as a cancel, because the
        # one branch that ends the process should only ever be reached by the
        # button that says so.
        return 'term-modal hidden'

    # Try to take the window with it. A window opened by --app (or a tab the
    # user opened themselves) was not opened by script, so Chrome is entitled
    # to refuse — which is why the goodbye card above is the real answer and
    # this is only a courtesy for the windows where it works.
    app.clientside_callback(
        """
        function(n) {
            if (n) { setTimeout(function() { window.close(); }, 900); }
            return window.dash_clientside.no_update;
        }
        """,
        Output('term-quit-sink', 'className'),
        Input('term-quit-yes', 'n_clicks'),
    )


# ---------------------------------------------------------------------------
# Standalone window
# ---------------------------------------------------------------------------
#
# ``--app=URL`` is Chromium's standalone mode: no tabs, no address bar, no
# bookmarks — a window that belongs to this board, with its own taskbar entry
# and the page's icon on it. It is the whole desktop-app illusion for the price
# of a subprocess, which is why the explorer reaches for it rather than for a
# webview toolkit nobody has installed.
#
# Chromium-family only: Firefox dropped -app years ago, so a machine with only
# Firefox falls back to an ordinary tab.

_APP_BROWSER_COMMANDS = (
    'google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser',
    'brave-browser', 'microsoft-edge', 'microsoft-edge-stable', 'chrome',
    'msedge',
)

# Where the same browsers live when they aren't on PATH, which on macOS and
# Windows is the normal case.
_APP_BROWSER_PATHS = (
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe',
)

APP_WINDOW_SIZE = (1440, 920)


def _app_browser():
    """Path to a browser that can open a standalone window, or None.

    ``UNICHART_APP_BROWSER`` overrides the search — a name on PATH or a full
    path — for the Chromium build this list doesn't know about.
    """
    override = os.environ.get('UNICHART_APP_BROWSER')
    if override:
        return shutil.which(override) or (override if os.path.exists(override)
                                          else None)
    for name in _APP_BROWSER_COMMANDS:
        found = shutil.which(name)
        if found:
            return found
    for path in _APP_BROWSER_PATHS:
        if os.path.exists(path):
            return path
    return None


def _browser_launcher(url, app_window):
    """A no-arg callable that opens ``url`` once the server is up.

    The browser is resolved here rather than inside the callable so that the
    "no Chromium found" note prints next to the URL, at launch, instead of
    surfacing a second later out of a timer thread.
    """
    if app_window:
        browser = _app_browser()
        if browser is None:
            # Told which browser to use and it isn't there: naming the value
            # back is the whole answer, where the generic note would advise
            # setting the variable that is already set.
            override = os.environ.get('UNICHART_APP_BROWSER')
            if override:
                print(f'unichart: UNICHART_APP_BROWSER={override!r} is not a '
                      'browser I can start; opening a normal browser tab '
                      'instead')
            else:
                print('unichart: no Chromium-family browser found for the app '
                      'window (Chrome, Chromium, Brave or Edge — or set '
                      'UNICHART_APP_BROWSER); opening a normal browser tab '
                      'instead')
        else:
            # No --user-data-dir on purpose: reusing the running browser opens
            # the window instantly and skips the first-run chrome a cold
            # profile drags in. The cost is that this process can't tell when
            # the window closes — closing the *window* leaves the server up,
            # so the way out is the board's ✕ close button, or Ctrl-C.
            argv = [browser, f'--app={url}',
                    '--window-size={},{}'.format(*APP_WINDOW_SIZE)]
            return lambda: _spawn(argv, url)
    return lambda: webbrowser.open(url)


def _spawn(argv, url):
    """Start the browser detached, falling back to a normal tab if it won't."""
    try:
        subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    except OSError:                                       # noqa: BLE001
        webbrowser.open(url)


def terminal(uc=None, data=None, sessions=None, panels=None, title=None,
             port=8050, debug=False, open_browser=None, app_window=False,
             jupyter_mode=None, dark=None, cheat_sheet=None, nb=None,
             **run_kwargs):
    """Launch the terminal explorer. See :func:`unichart.dashboard.explore`.

    ``nb=`` is the old name of ``uc=``, still accepted."""
    from .dashboard import _in_notebook

    if uc is None:
        uc = nb
    if uc is None:
        from ._core import UnichartNotebook
        uc = UnichartNotebook()
    # Judged before data= loads: it is the notebook you brought that says
    # whether you are new here, not the file you asked the board to open.
    if cheat_sheet is None:
        cheat_sheet = not uc.sets
    # The board is dark; match the plots to it unless the caller already chose.
    if not getattr(uc, 'darkmode', False):
        uc.toggle_darkmode(True)

    startup = []
    staged = []
    if data is not None:
        # A file unichart has no reader for doesn't fail the launch: the board
        # opens with a reader for it staged in the input, as a drop would.
        items = list(data) if isinstance(data, (list, tuple)) else [data]
        unread = [isinstance(d, (str, Path)) and Path(d).is_file()
                  and uc._reader_for(d)[0] is None for d in items]
        staged = [str(d) for d, u in zip(items, unread) if u]
        readable = [d for d, u in zip(items, unread) if not u]
        if readable:
            uc.load(readable)
    # Sessions restore as startup commands rather than here, for three reasons:
    # the restore lands in the transcript like every other action on this board;
    # the plot call it replays paints the chart pane, which otherwise opens blank
    # because it is only ever seeded from a startup result; and it runs after the
    # dark flip above, so a session saved in light mode keeps its theme.
    for path in ([sessions] if isinstance(sessions, (str, Path)) else sessions or []):
        startup.append(f'uc.load_session({str(path)!r})')
    # ...which is also why an explicit dark=True has to be re-asserted here: the
    # session would otherwise overwrite it on its way back in.
    if dark is not None and sessions:
        startup.append(f'uc.toggle_darkmode({bool(dark)!r})')
    for panel in panels or []:
        startup.append(_panel_command(panel))

    # Asked before the board is built: whether closing it may end the process
    # is the one layout decision that depends on where we are running.
    in_notebook = _in_notebook()
    app = build_terminal_app(uc, title=title, startup=[c for c in startup if c],
                             allow_quit=not in_notebook, staged=staged,
                             cheat_sheet=cheat_sheet)
    # A standalone window is a browser window either way, so asking for one
    # from a kernel means the external board rather than the inline iframe.
    standalone = app_window or not in_notebook
    if jupyter_mode is None:
        jupyter_mode = 'external' if standalone else 'inline'
    if jupyter_mode == 'inline':
        # An explicit inline mode wins: the iframe resolves its own URL, so
        # nothing below applies to it — app_window included.
        standalone = False
        run_kwargs.setdefault('jupyter_height', 860)
    if open_browser is None:
        open_browser = standalone

    port = _pick_port(port)
    if standalone:
        url = f'http://127.0.0.1:{port}/'
        print(f'unichart terminal: {url}')
        if open_browser:
            threading.Timer(1.0, _browser_launcher(url, app_window)).start()
        # Pinned explicitly off the inline path: the terminal executes arbitrary
        # Python, so the board is for this machine only. Inside a kernel the
        # host is left to Dash — its inline mode resolves the iframe's URL
        # itself, and forcing a host there yields a blank frame. Dash already
        # defaults to 127.0.0.1, so the binding is the same either way.
        run_kwargs['host'] = '127.0.0.1'
    app.run(port=port, debug=debug, jupyter_mode=jupyter_mode, **run_kwargs)
    return app


def _panel_command(panel):
    """Render a dashboard panel spec as the terminal command that draws it.

    Keeps ``--panel`` / ``explore(panels=...)`` meaningful now that the board is
    a REPL: each spec is replayed as a line of the transcript instead of
    becoming a card.
    """
    if not isinstance(panel, dict):
        return None
    method = panel.get('method', 'plot')
    args = []
    if panel.get('x') is not None:
        args.append(f'x={panel["x"]!r}')
    y = panel.get('y')
    if y is not None:
        args.append(f'y={y[0]!r}' if isinstance(y, (list, tuple)) and len(y) == 1
                    else f'y={y!r}')
    if panel.get('z') is not None:
        args.append(f'z={panel["z"]!r}')
    for key, value in (panel.get('kwargs') or {}).items():
        args.append(f'{key}={value!r}')
    return f'{method}({", ".join(args)})'
