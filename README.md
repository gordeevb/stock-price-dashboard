# Stock Price Dashboard

Fetch historical market data, run core financial analyses, and generate publication-ready charts (price with moving averages, volume, rolling volatility, and daily returns distribution).

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![Build](https://img.shields.io/badge/build-pytest-informational.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

---

## Features
- **Data acquisition** via `yfinance` with **validation**, **retries**, and **on-disk caching**
- **Analysis**: simple/log returns, cumulative return, moving averages, rolling volatility, volume & money-flow metrics
- **Visualization**: consistent matplotlib style. 4 figures (price+MAs, volume, volatility, returns distribution)
- **Modular design** with clear separation of concerns
- **Config** defaults (paths, windows, plotting style)


| Area            | What you get                                       |
|-----------------|----------------------------------------------------|
| Fetch           | Validated parameters, retry, local cache           |
| Analyze         | Returns (simple/log), MAs, rolling volatility, CMF |
| Plot            | Unified style, saved PNG/PDF files                 |
| Orchestrate     | One command to fetch, analyze, and plot            |

---

## Quickstart

```bash
# 1) Clone
git clone https://github.com/gordeevb/stock-price-dashboard.git
cd stock-price-dashboard

# 2) Create env (pick ONE of the below)

#   a) venv + pip
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

#   b) conda
conda create -n spdash python=3.10 -y
conda activate spdash
pip install -r requirements.txt

#   c) poetry
poetry install
poetry shell

# 3) Run example
python src/main.py --ticker AAPL --period 1y --interval 1d --outdir artifacts/AAPL_1y_1d
