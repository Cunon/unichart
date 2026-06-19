# unichart

A plotting and notebook dashboard tool built on [Plotly](https://plotly.com/python/).

## Installation

Install directly from GitHub with pip:

```bash
pip install git+https://github.com/Cunon/unichart.git
```

Optional extras:

```bash
# Trend lines (LOWESS smoothing via statsmodels)
pip install "unichart[trend] @ git+https://github.com/Cunon/unichart.git"

# Interactive web dashboards (Dash)
pip install "unichart[dashboard] @ git+https://github.com/Cunon/unichart.git"

# Everything
pip install "unichart[all] @ git+https://github.com/Cunon/unichart.git"
```

## Usage

```python
from unichart import UnichartNotebook

nb = UnichartNotebook()
```

## Requirements

- Python >= 3.9
- pandas, numpy, plotly, scipy, ipywidgets, ipython (installed automatically)
- Optional: `statsmodels` for trend lines, `dash` for dashboards
