# Publishing unichart to PyPI — plan

Status as of 2026-09-24. Work top to bottom; phase 0 is a decision gate because
it fixes the public import API for good once 0.1.0 is on PyPI.

## Layout change (2026-09-22)

The flat modules moved into the package: `unichart.py` → `unichart/_core.py`,
`unichart_cli.py` → `unichart/cli.py`, `unichart_dashboard.py` →
`unichart/dashboard.py`, `unichart_terminal.py` → `unichart/terminal.py`.
Existing editable installs need `pip install -e .` re-run, since the old
`unichart_cli:main` entry point is gone.

## Where things stand (verified)

- `python -m build` produces a clean sdist + wheel; `twine check --strict` passes.
- The wheel installs into a fresh venv (py3.12: pandas 3.0, numpy 2.5, plotly 7.1,
  dash 4.4); `import unichart`, `unichart --version` and `unichart FILE --info` work,
  and all 53 tests pass against the *installed* wheel.
- The name `unichart` is free on both PyPI and TestPyPI.
- The GitHub repo is public, with no license detected.

## Phase 0 — decisions (one-way doors) — decided 2026-09-22

- [x] **Module layout: package it.** `unichart/` package; `from unichart import
      UnichartNotebook` unchanged; submodules `unichart.dashboard`,
      `unichart.terminal`, `unichart.cli`; `python -m unichart`. No shims for the
      old flat names. The core lives in private `unichart/_core.py` behind a lazy
      `__init__` so `unichart --version/--help` stay instant.
- [x] **License: MIT.**
- [x] **Dash is core.** `pip install unichart && unichart data.csv` just works.
- [x] **IPython/ipywidgets stay core** (Jupyter-first; code still degrades without them).
- [x] **Author: name only** — `Cunon`, no email in PyPI metadata.
- [x] **Python `>=3.10`.**

## Phase 1 — packaging metadata (`pyproject.toml`)

- [x] Add `LICENSE`, and declare it in `pyproject.toml`. Use the SPDX form
      `license = "MIT"` + `license-files = ["LICENSE"]`, which needs
      `setuptools>=77`, so bump `build-system.requires`.
- [x] Classifiers: Development Status :: 3 - Alpha, License, one Programming
      Language :: Python :: 3.x per tested version, Framework :: Jupyter,
      Intended Audience :: Science/Research, Topic :: Scientific/Engineering ::
      Visualization.
- [x] Dependency floors: `pandas>=2.0`, `numpy>=1.24`, `plotly>=5.16`,
      `scipy>=1.10`, `ipywidgets>=8.0`, `ipython>=8.0`, `dash>=4` (2026-09-22).
      The CI `oldest-deps` job installs exactly these (`uv --resolution
      lowest-direct`) and runs the tests and the gallery on them. plotly 5.15
      fails (`Shape.showlegend`, used by reference-line legend entries);
      `test_full_legend_with_reference_line` pins that. The `png`/`all` extras
      add `plotly>=6.1.1`: kaleido 1.x refuses older plotly (verified: 5.16 and
      6.0.1 fail, 6.1 works); plotly 5.x users need kaleido 0.2.1.
- [x] Single-source the version. It is duplicated in `pyproject.toml` and
      `unichart_cli.py:52`. Read it from `importlib.metadata.version("unichart")`
      and expose `unichart.__version__`.
- [x] More `project.urls`: Documentation, Issues, Changelog.
- [ ] Optional: `[tool.pytest.ini_options]` so the tests don't need their
      `sys.path.insert` hacks.

## Phase 2 — repo hygiene (the repo is public)

- [~] Build output is untracked and ignored (checked on GitHub's `pypi` tree
      2026-09-24: no `__pycache__/`, `build/`, `*.egg-info`). `.gitignore`'s `.claude/`
      already matches at any depth, but `demo_notebooks/.claude/settings.local.json`
      is still tracked: `git rm --cached` it.
- [ ] Remove the unrelated `demo_notebooks/space_invaders_*.html` files.
- [ ] Consider stripping notebook outputs or moving the heavy files out:
      `large_data_showcase.ipynb` is 11 MB, `dashboard_progression_demo.html` is 5 MB,
      and five others are over 1 MB. They are not in the sdist, but they make
      cloning slow.
- [ ] Decide what happens to the `unichart_gui` branch.

## Phase 3 — README / user-facing text

- [x] Change the install instructions from `git+https://…` to `pip install unichart` /
      `pip install "unichart[all]"`. Keep the git URL as a "latest dev" option.
      Quote extras everywhere; line ~240 has a bare `pip install unichart[png]`,
      which zsh breaks.
- [x] Make relative links absolute. On PyPI, `PLOTTING_STYLE_GUIDE.md` and
      `demo_notebooks/` resolve against pypi.org and 404. Use
      `https://github.com/Cunon/unichart/blob/main/...`.
- [ ] Add a screenshot or gallery image near the top, using an absolute `raw.githubusercontent`
      URL. The PyPI page currently has no visuals.
- [x] Fix the stray trailing "## Usage" section, which repeats the Quick start.
- [x] Fix the Dash error message. It says "add it to requirements.txt"; it should say
      `pip install "unichart[dashboard]"`. It is printed twice by the CLI
      (`unichart_cli.py:828` plus the `_require_dash` text).
- [ ] `--gallery` points at `github.com/.../tree/main/gallery`, which shows the HTML
      source rather than the page. Host the gallery on GitHub Pages and point there
      (or ship it as package data under the package layout).
- [x] Add `CHANGELOG.md` with a 0.1.0 entry.

## Phase 4 — CI and testing

- [x] GitHub Actions, `.github/workflows/ci.yml` (2026-09-22). Four jobs:
      - `test`: pytest on Linux, macOS and Windows × py3.10–3.13, with `fail-fast: false`.
      - `oldest-deps`: py3.10 at the dependency floors, running the tests plus the
        gallery through kaleido 0.2.1.
      - `gallery`: py3.13 at the newest dependencies, with Chrome via
        `plotly_get_chrome`; `index.html` is uploaded as an artifact.
      - `package`: build, `twine check --strict`, a wheel-contents check, then
        tests against the installed wheel run from outside the source tree.

      Everything on Linux was simulated locally and passes. macOS and Windows have
      never run; the first push will tell.
- [x] Core smoke test, `tests/test_core_smoke.py` (19 tests, 72 in the suite).
      - One test per `PLOT_METHODS` entry, with a guard so new methods can't
        silently go untested.
      - `render_panel` for every panel type (it returns error figures rather than
        raising, so the test checks for those).
      - `dashboard_to_html` with no empty panels, a session round trip, delta,
        formatting, hue/trend, the full-legend fit, and styles/dark mode.
      - Mutation-checked: a bad column fails the relevant tests.
- [ ] Out of scope for 0.1.0: splitting up `unichart/_core.py`, or silencing the
      `UnichartNotebook()` "Initialized" print (polish, noted for later).
- [ ] Later: add py3.14 to the matrix once its wheels are settled for scipy/pandas.

## Phase 5 — release

- [ ] **Run CI first — it has never run.** `ci.yml` triggers on pushes to `main`
      and on PRs, and `workflow_dispatch` only works from the default branch, so
      the `pypi` branch has no runs (checked 2026-09-24). Open a PR `pypi` → `main`,
      get all four jobs green (macOS/Windows are untried), and merge. `release.yml`
      goes in the same merge: a tag runs the workflow file at the tagged commit.
- [ ] Create a PyPI account (with 2FA) and a separate TestPyPI account.
- [ ] Add a **pending** Trusted Publisher on each index (Account → Publishing,
      since the project doesn't exist yet). The names must match exactly, or the
      upload fails with `invalid-publisher`:

      | field       | PyPI       | TestPyPI   |
      |-------------|------------|------------|
      | PyPI project name | `unichart` | `unichart` |
      | Owner       | `Cunon`    | `Cunon`    |
      | Repository  | `unichart` | `unichart` |
      | Workflow    | `release.yml` | `release.yml` |
      | Environment | `pypi`     | `testpypi` |
- [ ] GitHub → Settings → Environments: create `testpypi` and `pypi`. On `pypi`,
      add yourself as a required reviewer and restrict deployments to tags `v*`.
- [x] `.github/workflows/release.yml` (2026-09-24). It runs on a `v*` tag. The `build`
      job runs `python -m build` and `twine check --strict`, fails unless the tag is
      `v` + the `pyproject.toml` version, and uploads `dist/`. `testpypi` publishes
      to TestPyPI (`skip-existing`, so re-runs pass). `pypi` waits for approval on
      the `pypi` environment, then publishes the same files. Only the publish jobs get
      `id-token: write`. The YAML parses, and build + `twine check` + the tag check
      were run locally.
- [ ] Tag `v0.1.0` on `main` after the merge and push the tag. While the `pypi`
      job waits for approval, rehearse from TestPyPI in a fresh venv:
      `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "unichart[all]"`,
      then run `unichart gallery/dyno_runs.csv --info`, the explorer and the test suite
      from that install. Check that `pip list` shows sane pandas/plotly/dash versions,
      because TestPyPI has junk uploads under popular names. If it doesn't, install
      `unichart` from TestPyPI with `--no-deps` and the rest from PyPI. Check the
      rendered project page, including links and the license. Then approve.
- [ ] Verify `pip install unichart` from real PyPI.
- [ ] Afterwards: create a GitHub Release from the tag and add PyPI/Python-version badges to the README.
