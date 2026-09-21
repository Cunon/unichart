#!/usr/bin/env python3
"""Build the unichart example gallery.

Renders every snippet in ``EXAMPLES`` with a real ``UnichartNotebook``,
exports each figure to PNG (kaleido) and writes a single self-contained
``gallery/index.html`` with the images embedded as data URIs.

    python gallery/make_gallery.py            # -> gallery/index.html
    python gallery/make_gallery.py --only bar # render one card, for iterating

Nothing here is hand-transcribed: the code shown on a card is the exact
string that was executed to make the figure above it.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import contextlib
import html
import io
import json
import os
import sys
import re
import warnings

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:                       # pragma: no cover
    raise SystemExit(
        f"{exc.name!r} is not installed for this interpreter:\n"
        f"    {sys.executable}  (Python {sys.version.split()[0]})\n\n"
        "The gallery needs numpy, pandas, plotly, kaleido and pillow. A venv whose\n"
        "python3 resolves to a different minor version than the one it was built\n"
        "with — a sandboxed terminal, say — sees none of its own packages. Check\n"
        "with:  python3 -c 'import sys; print(sys.version, sys.prefix)'"
    ) from exc

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from unichart import UnichartNotebook  # noqa: E402

DATA_CSV = os.path.join(HERE, "dyno_runs.csv")
OUT_HTML = os.path.join(HERE, "index.html")
TEMPLATE = os.path.join(HERE, "template.html")

HEAD_OPEN, HEAD_CLOSE = "<!--HEAD-->", "<!--/HEAD-->"
PAGE_DESCRIPTION = ("A skimmable gallery of unichart figures, each shown with the "
                    "few lines of code that drew it.")

PANEL_W, PANEL_H = 4.6, 3.0     # inches, per subplot panel
IMAGE_SCALE = 2                 # kaleido device pixel ratio
EAGER_CHARTS = 2                # mounted at load; the rest mount as you approach
THUMB_W, THUMB_H = 300, 190     # px, the chart-kind previews at the top

# The plotly.js that matches the plotly.py that wrote the figures.
try:
    from plotly.offline import get_plotlyjs_version
    PLOTLY_VERSION = get_plotlyjs_version()
except Exception:                                    # pragma: no cover
    PLOTLY_VERSION = "3.5.0"
PLOTLY_CDN = (f"https://cdnjs.cloudflare.com/ajax/libs/plotly.js/"
              f"{PLOTLY_VERSION}/plotly.min.js")


# --------------------------------------------------------------------------
# Demo data: four runs of a piston engine on a dynamometer test stand.
# One tidy frame, one row per 2.5 s sample, one dataset per run_id.
# --------------------------------------------------------------------------

PHASES = [("idle", 0, 90), ("cruise", 90, 300), ("climb", 300, 450), ("max", 450, 600)]

RUNS = [
    # name,          rpm bias, load bias, oat_c, eta peak, eta offset
    ("Baseline",     1.00, 1.00, 15.0, 2400.0, 0.0),
    ("Cold day",     1.02, 1.03, -5.0, 2400.0, 0.6),
    ("Hot day",      0.98, 0.95, 35.0, 2350.0, -1.1),
    ("Tuned map",    1.01, 1.06, 15.0, 2550.0, 2.2),
]


def build_frame(seed: int = 7) -> pd.DataFrame:
    """Four dyno runs sharing one schema, saved alongside this script."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 600.0, 2.5)

    rpm_target = np.empty_like(t)
    load_target = np.empty_like(t)
    phase = np.empty(t.shape, dtype=object)
    for name, lo, hi in PHASES:
        m = (t >= lo) & (t < hi)
        phase[m] = name
        rpm_target[m] = {"idle": 900, "cruise": 2300, "climb": 2600, "max": 2850}[name]
        load_target[m] = {"idle": 25, "cruise": 185, "climb": 235, "max": 265}[name]

    frames = []
    for i, (name, rpm_bias, load_bias, oat, eta_peak, eta_off) in enumerate(RUNS):
        ph = 0.7 * i
        # Two slow, out-of-phase sweeps so the runs cover the rpm/torque plane
        # instead of tracing one line through it.
        rpm_set = _lag(rpm_target, 0.10)          # the stand ramps between phases
        load_set = _lag(load_target, 0.10)
        rpm = rpm_bias * (rpm_set + 0.14 * rpm_set * np.sin(2 * np.pi * t / 170 + ph))
        rpm += rng.normal(0, 10, t.size)

        torque = load_bias * (load_set
                              + (0.25 * load_set + 4) * np.sin(2 * np.pi * t / 110 + 1.3 * ph))
        torque += rng.normal(0, 2.0, t.size)

        power = torque * rpm * 2 * np.pi / 60 / 1000.0            # kW
        eta = 7.0 + (26.0 + eta_off) * np.exp(
            -((rpm - eta_peak) / 1500.0) ** 2 - ((torque - 235) / 210.0) ** 2
        ) + rng.normal(0, 0.3, t.size)
        fuel = power / (eta / 100.0 * 43.0) * 3.6   # kg/h, LHV 43 MJ/kg

        cht = _lag(oat + 165 * power / power.max() + rng.normal(0, 1.2, t.size), 0.08)
        egt = _lag(oat + 300 + 430 * power / power.max() + rng.normal(0, 4.0, t.size), 0.15)

        frames.append(pd.DataFrame({
            "time_s": t,
            "rpm": rpm.round(1),
            "torque_nm": torque.round(2),
            "power_kw": power.round(3),
            "fuel_kgh": fuel.round(3),
            "cht_c": cht.round(2),
            "egt_c": egt.round(1),
            "eta_pct": eta.round(3),
            "phase": phase,
            "run_id": i,
            "run_name": name,
        }))
    return pd.concat(frames, ignore_index=True)


def _lag(x: np.ndarray, alpha: float) -> np.ndarray:
    """First-order lag, so temperatures trail the power they follow."""
    out = np.empty_like(x)
    out[0] = x[0]
    for k in range(1, x.size):
        out[k] = out[k - 1] + alpha * (x[k] - out[k - 1])
    return out


# The setup shown once at the top of the page. `fresh_nb` runs exactly this.
SETUP_CODE = """\
from unichart import UnichartNotebook
import pandas as pd

# 4 runs x 240 samples, one tidy frame
df = pd.read_csv('dyno_runs.csv')

nb = UnichartNotebook()
nb.set_default_format(marker=None, linestyle='-',
                      linewidth=1.8)      # traces, not markers
nb.load_df(df, set_idx_column='run_id',
               set_name_column='run_name')
nb.set_plot_size(4.6, 3.0)   # one panel size, every plot
"""


def fresh_nb(df: pd.DataFrame) -> UnichartNotebook:
    """A notebook loaded with the demo frame, with no state from other cards."""
    with contextlib.redirect_stdout(io.StringIO()):
        nb = UnichartNotebook()
        nb.set_default_format(marker=None, linestyle='-', linewidth=1.8)
        nb.load_df(df, set_idx_column="run_id", set_name_column="run_name")
        nb.set_plot_size(PANEL_W, PANEL_H)
    return nb


# --------------------------------------------------------------------------
# The gallery itself. Each entry's `code` is executed verbatim and shown
# verbatim; `nb` is a notebook freshly loaded with the frame above.
# --------------------------------------------------------------------------

SECTIONS = [
    ("start", "Start here", "The four layouts that cover most of a test campaign."),
    ("shape", "Slice and shape", "Choose what is drawn before you worry about how it looks."),
    ("dist", "Distributions", "When the spread matters more than the trace."),
    ("fields", "Fields and tables", "Scattered data as a surface, and numbers as numbers."),
    ("analysis", "Analysis", "Differences against a baseline, and fits through the cloud."),
    ("looks", "Looks", "The same figures, restyled."),
]

EXAMPLES = [
    dict(
        section="start", id="plot", thumb="Line", api="nb.plot",
        title="One line per run",
        blurb="Every selected dataset is drawn on the same axes, each with its own "
              "color and marker from the palette. Nothing to loop over.",
        code="nb.plot(x='time_s', y='cht_c')",
    ),
    dict(
        section="start", id="plot-vars", thumb="Subplot grid", api="nb.plot(by='vars')",
        title="A subplot per variable",
        blurb="Pass a list of Y columns and you get a grid — the default layout. "
              "The runs stay color-matched across the panels.",
        code="nb.plot(x='time_s', y=['rpm', 'torque_nm', 'cht_c', 'eta_pct'], ncols=2)",
    ),
    dict(
        section="start", id="plot-sets", thumb="Per-run panels", api="nb.plot(by='sets')",
        title="A subplot per run",
        blurb="Same data, transposed: one panel per dataset with every variable in it. "
              "ncols sets the grid; hspace and vspace set the gaps between panels, "
              "in pixels.",
        code="nb.select([0,3])\n"
             "nb.plot(x='time_s', y=['cht_c', 'egt_c'], by='sets', vspace=110)",
    ),
    dict(
        section="start", id="ymult", thumb="Multi-axis", api="nb.plot_ymult",
        title="Several scales, one panel",
        blurb="Stacked Y axes for quantities that share an X but nothing else — "
              "rpm in thousands next to a fuel flow in single digits.",
        code="nb.select([0, 2])\n"
             "nb.linestyle(2, ':')\n"
             "nb.color('rpm', 'blue')\n"
             "nb.color('cht_c', 'red')\n"
             "nb.color('fuel_kgh', 'orange')\n"
             "nb.plot_ymult(x='time_s', y=['rpm', 'cht_c', 'fuel_kgh'])",
    ),

    dict(
        section="shape", id="select", api="nb.select · nb.query",
        title="Select runs, filter rows",
        blurb="Selection decides which datasets are drawn; a query filters rows "
              "inside them. Both stick until you change them.",
        code="nb.select([0, 3])                       # draw these two runs\n"
             "nb.query('all', 'phase == \"climb\"')      # keep these rows\n"
             "nb.plot(x='time_s', y=['rpm', 'eta_pct'])",
    ),
    dict(
        section="shape", id="decorations", api="nb.line · nb.highlight",
        title="Limits and events",
        blurb="Reference lines carry a label drawn inside the plot area; highlights "
              "shade a band of the X axis. Both persist across plots.",
        code="nb.line('cht_c', 150, color='firebrick', linestyle='--',\n        label='CHT limit', label_position='left above')\n"
             "nb.highlight('time_s', (300, 450), color='orange', alpha=0.12)\n"
             "nb.plot(x='time_s', y='cht_c')",
    ),
    dict(
        section="shape", id="varformat", api="nb.var_format",
        title="Format by variable, not by call",
        blurb="Where the variables are the series rather than the runs, a "
              "per-variable override gives each column its own identity — and it "
              "outranks the per-dataset style, so CHT is dashed red everywhere.",
        code="nb.select(0)\n"
             "nb.var_format('cht_c', color='firebrick', linestyle='--')\n"
             "nb.var_format('egt_c', color='darkorange')\n"
             "nb.var_format('fuel_kgh', color='steelblue')\n"
             "nb.plot_ymult(x='time_s', y=['cht_c', 'egt_c', 'fuel_kgh'])",
    ),

    dict(
        section="dist", id="marginal", thumb="Marginal", api="nb.plot_marginal",
        title="Scatter with its margins",
        blurb="The operating points, with the distribution of each axis in a strip "
              "beside it. Swap in 'box', 'violin', 'rug' or 'kde'.",
        code="nb.marker('all', 'o'); nb.linestyle('all', ''); nb.markersize('all', 5)\n"
             "nb.plot_marginal(x='rpm', y='torque_nm', marginal='box')",
    ),
    dict(
        section="dist", id="hist", thumb="Histogram", api="nb.histogram",
        title="Where the run spent its time",
        blurb="Overlaid histograms per dataset, with the binning, normalization and "
              "opacity you would expect to be able to set.",
        code="nb.histogram(x='cht_c', nbins=40, alpha=0.6)",
    ),
    dict(
        section="dist", id="box", thumb="Box", api="nb.box",
        title="Spread by category",
        blurb="One box per run per category, grouped along a categorical X column.",
        code="nb.box(x='phase', y='eta_pct', points='outliers')",
    ),
    dict(
        section="dist", id="bar", thumb="Bar", api="nb.bar",
        title="Reduced to one number each",
        blurb="Bar height is an aggregate of Y over the rows in each category — "
              "'mean' by default, or any pandas reducer.",
        code="nb.bar(x='phase', y=['fuel_kgh', 'eta_pct'], agg='mean', barmode='group')",
    ),

    dict(
        section="fields", id="contour", thumb="Contour", api="nb.contour",
        title="A map from scattered points",
        blurb="Efficiency over the rpm/torque plane, interpolated from the samples "
              "the runs happened to visit, with one run's track drawn over it.",
        code="nb.combine_sets('all', title='all runs')\n"
             "nb.select(4)\n"
             "nb.color(0, 'black')\n"
             "nb.linestyle(0, '--')\n"
             "nb.contour(x='rpm', y='torque_nm', z='eta_pct', overlay_sets=0)",
    ),
    dict(
        section="fields", id="table", thumb="Table", api="nb.table",
        title="Values at the X you ask for",
        blurb="Interpolation mode reads each run's Y at your X inputs, so runs "
              "sampled at different times line up in one table.",
        code="nb.table(cols=['rpm', 'eta_pct'], x_col='time_s',\n"
             "         x_in=[150, 300, 450], sig_figs=4, output='fig')",
    ),
    dict(
        section="fields", id="summary", thumb="Summary", api="nb.summary",
        title="Descriptive stats per run",
        blurb="Count, min, mean, max and std for the columns you name — the same "
              "sortable table widget as nb.table.",
        code="nb.summary(cols=['rpm', 'torque_nm', 'eta_pct'], sig_figs=4, output='fig')",
    ),

    dict(
        section="analysis", id="delta", thumb="Delta", api="nb.delta",
        title="Against a baseline",
        blurb="Each study run is aligned to the base run and differenced, producing "
              "DL_<parm> and DLPCT_<parm> columns in new datasets that keep the "
              "study run's color.",
        code="nb.delta(base_idx=0, study_indices=[1, 2, 3],\n"
             "         align_on='time_s', delta_parms=['cht_c', 'eta_pct'])\n"
             "nb.select([4, 5, 6])\n"
             "nb.plot(x='time_s', y=['DL_cht_c', 'DLPCT_eta_pct'], hspace=100)",
    ),
    dict(
        section="analysis", id="trend", thumb="Trend fit", api="nb.reg_order",
        title="Fits through the cloud",
        blurb="Give a dataset a regression order and its line becomes the fit, "
              "labelled in the legend. LOWESS needs statsmodels.",
        code="nb.marker('all', 'o'); nb.markersize('all', 4); nb.alpha_marker('all', 0.7)\n"
             "nb.reg_order('all', 2)\n"
             "nb.plot(x='rpm', y='eta_pct')",
    ),
    dict(
        section="analysis", id="hue", thumb="Color by value", api="nb.hue",
        title="Color by a third column",
        blurb="A dataset can take its point colors from any column, turning a "
              "scatter into a three-variable read.",
        code="nb.select(3)\n"
             "nb.marker(3, 'o'); nb.linestyle(3, ''); nb.markersize(3, 6)\n"
             "nb.hue(3, 'eta_pct')\n"
             "nb.plot(x='rpm', y='torque_nm')",
    ),

    dict(
        section="looks", id="plotly-style", api="nb.set_plot_style",
        title="Plotly look",
        blurb="Every figure above is drawn in the default Matplotlib-like style — "
              "four spines, outward ticks, DejaVu Sans, the tab10 cycle. One call "
              "switches the whole environment to Plotly's own look instead.",
        code="nb.set_plot_style('plotly')\n"
             "nb.plot(x='time_s', y=['torque_nm', 'eta_pct'], hspace=200)",
    ),
    dict(
        section="looks", id="dark", api="nb.toggle_darkmode",
        title="Dark mode",
        blurb="One switch, applied to the whole environment — plots, tables and "
              "dashboards alike.",
        code="nb.toggle_darkmode(True)\n"
             "nb.plot(x='time_s', y=['rpm', 'cht_c'], hspace=200)",
        dark=True,
    ),
]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render_example(ex: dict, df: pd.DataFrame, static: bool = False) -> dict:
    """Execute one card's code and capture its figure. Returns the card dict."""
    import plotly.io as pio

    nb = fresh_nb(df)
    ns = {"nb": nb, "np": np, "pd": pd}

    chatter = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(chatter):
            exec(ex["code"], ns)          # noqa: S102 - the snippet IS the content
        messages = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
    # unichart reports bad arguments by printing, not by raising, so a card can
    # look fine while quietly ignoring half its snippet. Surface those too.
    messages += [ln.strip() for ln in chatter.getvalue().splitlines()
                 if ln.strip().lower().startswith(("warning", "error", "unknown"))]

    fig = ns.get("fig") or nb.last_fig
    if fig is None:
        raise RuntimeError(f"{ex['id']}: snippet produced no figure")

    card = dict(
        ex,
        warnings=messages,
        width=int(fig.layout.width or 900),
        height=int(fig.layout.height or 600),
    )

    if ex.get("thumb"):
        card["thumb_png"] = _thumbnail(fig)

    if static:
        png = _shrink(pio.to_image(fig, format="png", scale=IMAGE_SCALE))
        if len(png) < 5000:
            raise RuntimeError(f"{ex['id']}: suspiciously small PNG ({len(png)} bytes)")
        card["png"] = png
    else:
        spec = json.loads(pio.to_json(fig, remove_uids=True))
        layout = spec.setdefault("layout", {})
        # The card decides the width; the figure keeps the height it was drawn at.
        layout.pop("width", None)
        layout.pop("autosize", None)
        card["template"] = layout.pop("template", None)
        # A table figure keeps the plot's height, which leaves half the card
        # empty. Size it to its rows instead.
        if all(t.get("type") == "table" for t in spec.get("data", [])):
            rows = max(len(c) for c in spec["data"][0]["cells"]["values"])
            layout["height"] = card["height"] = 108 + 26 * (rows + 1)
        card["spec"] = spec
        if not spec.get("data"):
            raise RuntimeError(f"{ex['id']}: figure has no traces")
    return card


_TOKEN = re.compile(r"""
      (?P<cm>\#[^\n]*)
    | (?P<st>'[^'\n]*'|"[^"\n]*")
    | (?P<fn>(?<=\.)[A-Za-z_]\w*(?=\())
    | (?P<nu>\b\d+\.?\d*\b)
""", re.X)


def highlight(code: str) -> str:
    """Minimal Python colouring: comments, strings, numbers, method names."""
    out, pos = [], 0
    for m in _TOKEN.finditer(code):
        out.append(html.escape(code[pos:m.start()]))
        out.append(f'<span class="{m.lastgroup}">{html.escape(m.group())}</span>')
        pos = m.end()
    out.append(html.escape(code[pos:]))
    return "".join(out)


def _thumbnail(fig) -> bytes:
    """A small, wordless preview of a figure: the shape of the chart with the
    titles, legend, colorbars and tick labels taken off."""
    import plotly.graph_objects as go
    import plotly.io as pio

    t = go.Figure(fig)
    t.update_layout(showlegend=False, title=None,
                    width=THUMB_W, height=THUMB_H,
                    margin=dict(l=4, r=4, t=4, b=4))
    # update_layout merges array properties element-wise, so subplot titles
    # survive `annotations=[]`. Assigning the property outright drops them.
    t.layout.annotations = []
    t.update_xaxes(title=None, showticklabels=False, ticks="")
    t.update_yaxes(title=None, showticklabels=False, ticks="")
    for tr in t.data:
        if "showscale" in tr:
            tr.showscale = False
    if t.layout.coloraxis is not None:        # hue-colored scatters share one
        t.layout.coloraxis.showscale = False
    return _shrink(pio.to_image(t, format="png", scale=2))


def _shrink(png: bytes) -> bytes:
    """Palette-quantize the export. Plot figures use few colors, so 256 is
    visually lossless here and cuts the embedded page to a third."""
    try:
        from PIL import Image
    except ImportError:
        return png
    im = Image.open(io.BytesIO(png)).convert("RGB")
    buf = io.BytesIO()
    im.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.NONE).save(
        buf, "PNG", optimize=True)
    return buf.getvalue() if len(buf.getvalue()) < len(png) else png


def _json_for_html(obj) -> str:
    """JSON safe to drop inside a <script> element."""
    return (json.dumps(obj, separators=(",", ":"))
            .replace("</", "<\\/").replace("<!--", "<\\u0021--"))


def card_html(card: dict, index: int) -> str:
    plate = "plate plate--dark" if card.get("dark") else "plate"

    if "png" in card:
        uri = "data:image/png;base64," + base64.b64encode(card["png"]).decode("ascii")
        figure = (f'<figure class="{plate}">\n'
                  f'          <img src="{uri}" alt="{html.escape(card["title"])}"\n'
                  f'               width="{card["width"]}" height="{card["height"]}"'
                  f' loading="lazy">\n        </figure>')
    else:
        eager = " data-eager" if index < EAGER_CHARTS else ""
        figure = (f'<figure class="{plate}">\n'
                  f'          <div class="chart" id="fig-{card["id"]}"'
                  f' style="height:{card["height"]}px"'
                  f' data-tpl="{card["tpl"]}"{eager}></div>\n'
                  f'          <script type="application/json"'
                  f' data-for="fig-{card["id"]}">{_json_for_html(card["spec"])}</script>\n'
                  f'        </figure>')

    return f"""      <article class="card" id="{card['id']}">
        <header class="card-head">
          <p class="api">{html.escape(card['api'])}</p>
          <h3>{html.escape(card['title'])}</h3>
          <p class="blurb">{html.escape(card['blurb'])}</p>
        </header>
        <div class="code">
          <button class="copy" type="button" data-copy="{html.escape(card['code'])}">Copy</button>
          <pre><code>{highlight(card['code'])}</code></pre>
        </div>
        {figure}
      </article>"""


def atlas_html(cards: list[dict]) -> str:
    """The grid of chart-kind previews that opens the page."""
    tiles = []
    for card in cards:
        if "thumb_png" not in card:
            continue
        uri = "data:image/png;base64," + base64.b64encode(card["thumb_png"]).decode("ascii")
        tiles.append(
            f'        <a class="tile" href="#{card["id"]}">\n'
            f'          <img src="{uri}" alt="" width="{THUMB_W}" height="{THUMB_H}">\n'
            f'          <span class="tile-name">{html.escape(card["thumb"])}</span>\n'
            f'          <span class="tile-api">{html.escape(card["api"])}</span>\n'
            f'        </a>'
        )
    if not tiles:
        return ""
    return ('    <section class="atlas" id="kinds">\n'
            '      <h2>What you can draw</h2>\n'
            f'      <p>{len(tiles)} kinds of chart. Pick one to jump to its '
            'example.</p>\n'
            '      <div class="tiles">\n'
            + "\n".join(tiles)
            + '\n      </div>\n    </section>')


RUNTIME_JS = """
<script type="application/json" id="figure-templates">%(templates)s</script>
<script>
  (function () {
    var slots = [].slice.call(document.querySelectorAll('.chart'));
    if (!slots.length) return;
    var templates = JSON.parse(document.getElementById('figure-templates').textContent);

    function mount(div) {
      if (div.dataset.mounted) return;
      div.dataset.mounted = '1';
      var holder = document.querySelector('script[data-for="' + div.id + '"]');
      var spec = JSON.parse(holder.textContent);
      spec.layout.template = templates[div.dataset.tpl];
      Plotly.newPlot(div, spec.data, spec.layout, {
        responsive: true, displaylogo: false, scrollZoom: false
      });
    }

    function start() {
      if (typeof Plotly === 'undefined') {
        document.documentElement.classList.add('noplotly');
        return;
      }
      slots.filter(function (d) { return 'eager' in d.dataset; }).forEach(mount);
      var rest = slots.filter(function (d) { return !('eager' in d.dataset); });
      if (!('IntersectionObserver' in window)) { rest.forEach(mount); return; }
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) { mount(e.target); io.unobserve(e.target); }
        });
      }, { rootMargin: '900px 0px' });
      rest.forEach(function (d) { io.observe(d); });
    }

    if (document.readyState === 'complete') start();
    else window.addEventListener('load', start);
  })();
</script>
"""


def _plotly_script(embed_js: bool) -> str:
    """The <script> that provides window.Plotly."""
    if not embed_js:
        return (f'<script src="{PLOTLY_CDN}" charset="utf-8"></script>')
    from plotly.offline import get_plotlyjs
    return f"<script>{get_plotlyjs()}</script>"


def as_document(page: str) -> str:
    """The template, wrapped as a standalone HTML document for the repo."""
    head = page[page.index(HEAD_OPEN) + len(HEAD_OPEN):page.index(HEAD_CLOSE)].strip()
    body = page[page.index(HEAD_CLOSE) + len(HEAD_CLOSE):].strip()
    return (
        '<!doctype html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" '
        'content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f'<meta name="description" content="{html.escape(PAGE_DESCRIPTION)}">\n'
        f'{head}\n'
        '</head>\n'
        f'<body>\n{body}\n</body>\n</html>\n'
    )


def as_fragment(page: str) -> str:
    """The same page without the document shell, for hosts that supply their own."""
    return page.replace(HEAD_OPEN, "").replace(HEAD_CLOSE, "").strip() + "\n"


def build(only: str | None = None, dump_dir: str | None = None,
          fragment: str | None = None, static: bool = False,
          embed_js: bool = False) -> None:
    df = build_frame()
    df.to_csv(DATA_CSV, index=False)

    wanted = [e for e in EXAMPLES if only is None or only in e["id"]]
    cards, problems = [], []
    for ex in wanted:
        card = render_example(ex, df, static=static)
        cards.append(card)
        for msg in card["warnings"]:
            problems.append(f"{ex['id']}: {msg}")
        size = len(card.get("png") or _json_for_html(card["spec"])) / 1024
        print(f"  {ex['id']:<12} {card['width']:>5}x{card['height']:<5} {size:7.0f} KB")
        if dump_dir and "png" in card:
            os.makedirs(dump_dir, exist_ok=True)
            with open(os.path.join(dump_dir, ex["id"] + ".png"), "wb") as fh:
                fh.write(card["png"])

    # Figures mostly share one layout template; emit each distinct one once.
    templates: dict[str, dict] = {}
    for card in cards:
        if card.get("template") is not None:
            blob = json.dumps(card["template"], separators=(",", ":"))
            key = hashlib.sha1(blob.encode()).hexdigest()[:8]
            templates[key] = card["template"]
            card["tpl"] = key

    nav, body, n = [], [], 0
    for sid, stitle, sblurb in SECTIONS:
        in_section = [c for c in cards if c["section"] == sid]
        if not in_section:
            continue
        nav.append(f'<a href="#{sid}">{html.escape(stitle)}</a>')
        rendered = []
        for c in in_section:
            rendered.append(card_html(c, n))
            n += 1
        body.append(
            f'    <section class="group" id="{sid}">\n'
            f'      <div class="group-head">\n'
            f'        <h2>{html.escape(stitle)}</h2>\n'
            f'        <p>{html.escape(sblurb)}</p>\n'
            f'      </div>\n'
            + "\n".join(rendered)
            + "\n    </section>"
        )

    plotly_tag, runtime = "", ""
    if not static:
        plotly_tag = _plotly_script(embed_js)
        runtime = RUNTIME_JS % {"templates": _json_for_html(templates)}

    with open(TEMPLATE, encoding="utf-8") as fh:
        page = fh.read()
    page = (page
            .replace("<!--PLOTLY-->", plotly_tag)
            .replace("<!--RUNTIME-->", runtime)
            .replace("<!--ATLAS-->", atlas_html(cards))
            .replace("<!--NAV-->", "\n        ".join(nav))
            .replace("<!--SETUP_RAW-->", html.escape(SETUP_CODE.strip()))
            .replace("<!--SETUP-->", highlight(SETUP_CODE.strip()))
            .replace("<!--CARDS-->", "\n".join(body))
            .replace("<!--COUNT-->", str(len(cards))))
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(as_document(page))

    kb = os.path.getsize(OUT_HTML) / 1024
    print(f"\n{OUT_HTML}  ({kb:.0f} KB, {len(cards)} cards)")
    if fragment:
        with open(fragment, "w", encoding="utf-8") as fh:
            fh.write(as_fragment(page))
        print(f"{fragment}  (no document shell)")
    if problems:
        print("\nWarnings raised while rendering:")
        for p in problems:
            print("  ! " + p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="substring of a card id, to render just that one")
    ap.add_argument("--dump", help="also write each PNG into this directory")
    ap.add_argument("--static", action="store_true",
                    help="embed PNG stills instead of live Plotly charts")
    ap.add_argument("--embed-js", action="store_true",
                    help="inline plotly.js instead of loading it from the CDN, "
                         "so the page works with no network")
    ap.add_argument("--fragment", metavar="PATH",
                    help="also write a copy with no <html>/<head> shell, for hosts "
                         "that wrap the page themselves")
    args = ap.parse_args()
    build(only=args.only, dump_dir=args.dump, fragment=args.fragment,
          static=args.static, embed_js=args.embed_js)
