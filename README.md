# Stock Analysis Dashboard

A comprehensive Python-based stock market analysis tool that fetches real-time data, performs technical analysis, and generates interactive visualizations. Built as a course project for EECE 2140 (Computing Fundamentals for Engineers).

## Project Goal

Create a production-ready stock analysis dashboard that enables users to:
- **Fetch historical stock data** from Yahoo Finance with intelligent caching
- **Analyze financial metrics** including returns, volatility, and technical indicators
- **Visualize trends** through professional charts (price, volume, volatility, returns distribution)
- **Assess market sentiment** using the Chaikin Money Flow (CMF) indicator
- **Export results** to CSV for further analysis

The project demonstrates software engineering best practices including modular design, testing, error handling, and both CLI and web interfaces.

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for fetching stock data)

### Installation

1. **Clone or download the project:**
```bash
git clone <your-repo-url>
cd stock-price-dashboard
```

2. **Create a virtual environment (recommended):**
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
yfinance>=0.2.28
streamlit>=1.28.0
pytest>=7.4.0
```

---

## How to Run

### Option 1: Command-Line Interface (CLI)

The CLI provides an interactive terminal-based experience:

```bash
python src/main.py
```

**Workflow:**

1. **Enter stock ticker** (e.g., AAPL, MSFT, GOOGL)
2. **Choose time period** (1d, 1mo, 1y, max)
3. **Select data interval** (1m, 1h, 1d, 1wk)
4. **View analysis report** in terminal
5. **Save report** (optional) as CSV
6. **View charts** - 4 windows open automatically

**Example session:**
```
==============================================================================
STOCK ANALYSIS DASHBOARD
==============================================================================
Enter ticker symbol: AAPL
Enter period (1d, 1mo, 1y, max): 1y
Enter interval (1m, 1h, 1d, 1wk): 1d

Fetching data for AAPL...
  Fetched 252 rows
  Date range: 2024-01-01 to 2024-12-31

Generating analysis...

==============================================================================
ANALYSIS REPORT - AAPL
==============================================================================
Metric                    Value
------------------------  -----------------
Ticker                    AAPL
Data Points               252
Date Range                2024-01-01 to 2024-12-31
Current Price             $185.23
Period High               $198.45
Period Low                $165.32
Cumulative Return         15.67%
Avg Daily Return          0.0623%
Volatility (30d)          24.35%
MA_20                     $183.12
MA_50                     $180.45
MA_200                    $175.89
Current Volume            45,234,567
Avg Volume (20d)          42,156,789
Chaikin Money Flow        0.145
==============================================================================

Save report to file? (y/n): y
  Saved to: outputs/AAPL_analysis_20241205_143022.csv

Creating charts...
  All charts created!
```

### Option 2: Web Interface (Streamlit)

The web interface provides an interactive, browser-based dashboard:

```bash
streamlit run src/streamlit_app.py
```

This opens your browser at `http://localhost:8501` with:
- **Sidebar controls** for ticker, period, interval, and analysis options
- **Key metrics cards** showing price, returns, and volatility
- **Market sentiment analysis** with CMF indicator interpretation
- **Interactive charts** in tabbed interface
- **Real-time updates** when parameters change

**Features:**
- Clean, modern UI with custom CSS styling
- Automatic data caching for performance
- Popular tickers quick reference
- Color-coded sentiment indicators (bullish/bearish/neutral)


---

## Methodology & Approach

### 1. Data Acquisition (`fetch_data.py`)

**Challenge:** Yahoo Finance API can be unreliable.

**Solution:**
- **Smart caching system** with different expiry times:
  - Intraday data (1m, 5m): 15-minute cache
  - Daily and more data (1d, 1wk): 24-hour cache
- **Retry logic** with exponential backoff
- **Fallback mechanism** uses expired cache if network fails
- **Input validation** prevents invalid API calls

**Key Functions:**
- `fetch_stock_data()` - Main entry point with comprehensive error handling
- `validate_ticker()` - Checks ticker format and existence
- `download_stock_data()` - Downloads with automatic retry
- Cache management: `get_cached_data()`, `save_to_cache()`, `is_cache_valid()`

### 2. Statistical Analysis (`analyze.py`)

**Metrics Calculated:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Daily Returns** | `(P_today - P_yesterday) / P_yesterday × 100` | Daily % change |
| **Cumulative Return** | `(P_end - P_start) / P_start × 100` | Total % gain/loss |
| **Volatility (30d)** | `σ(returns) × √252` | Annualized risk measure |
| **Moving Averages** | `mean(prices[-N:])` | Trend indicators (20, 50, 200 day) |
| **Chaikin Money Flow** | `Σ(MFV) / Σ(Volume)` | Buying/selling pressure (-1 to +1) |

**CMF Interpretation:**
- CMF > +0.15: Strong buying pressure (bullish)
- CMF > +0.05: Moderate buying
- -0.05 to +0.05: Balanced (neutral)
- CMF < -0.05: Moderate selling
- CMF < -0.15: Strong selling pressure (bearish)

**Key Functions:**
- `generate_summary_statistics()` - Analysis pipeline
- `calculate_returns()` - Percentage returns calculation
- `calculate_volatility()` - Risk measurement with annualization
- `calculate_chaikin_money_flow()` - Market sentiment indicator

### 3. Visualization (`plot.py`)

**Four Chart Types:**

1. **Price Chart with Moving Averages**
   - Main price line (blue)
   - 20-day MA (green) - short-term trend
   - 50-day MA (orange) - medium-term trend
   - 200-day MA (purple) - long-term trend

2. **Volume Chart**
   - Bars colored by price direction (green=up, red=down)
   - 20-day average volume line
   - Helps confirm price movements

3. **Volatility Chart**
   - 30-day rolling volatility (annualized)
   - Mean line for reference
   - Shows risk changes over time

4. **Returns Distribution**
   - Histogram of daily returns
   - Green for positive, red for negative
   - Shows return patterns and outliers


### 4. Testing Strategy

- **Unit Tests** - Test individual functions in isolation
- **Integration Tests** - Test module interactions
- **Mocking** - Avoid actual network calls during testing
- **Fixtures** - Reusable test data
- **Edge Cases** - Handle insufficient data, missing columns, etc.

**Run tests:**
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_analyze.py

# Run with verbose output
pytest -v
```
---

## Results & Outputs

### CLI Output Example

**Terminal Report:**
```
==============================================================================
ANALYSIS REPORT - TSLA
==============================================================================
Ticker                    TSLA
Data Points               252
Date Range                2024-01-01 to 2024-12-31
Current Price             $245.67
Period High               $299.89
Period Low                $189.45
Cumulative Return         28.34%
Avg Daily Return          0.1123%
Volatility (30d)          45.67%
MA_20                     $242.34
MA_50                     $235.12
MA_200                    $220.45
Current Volume            125,456,789
Avg Volume (20d)          98,234,567
Chaikin Money Flow        0.234
==============================================================================
```

**Saved CSV Format:**
```csv
Metric,Value
Ticker,TSLA
Data Points,252
Current Price,$245.67
Cumulative Return,28.34%
...
```

---

## Testing

Run the test suite to verify everything works:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest tests/test_fetch_data.py      # Data fetching tests
pytest tests/test_analyze.py         # Analysis tests
pytest tests/test_plot.py            # Plotting tests
pytest tests/test_integration.py     # Integration tests
```

**Expected output:**
```
========================= test session starts ==========================
collected 48 items

tests/test_fetch_data.py ............                           [ 25%]
tests/test_analyze.py ..............                            [ 54%]
tests/test_plot.py ...........                                  [ 77%]
tests/test_integration.py ...........                           [100%]

========================== 48 passed in 3.42s ==========================
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **pandas** | ≥2.0.0 | Data manipulation and analysis |
| **numpy** | ≥1.24.0 | Numerical computations |
| **matplotlib** | ≥3.7.0 | Chart creation |
| **yfinance** | ≥0.2.28 | Yahoo Finance API wrapper |
| **streamlit** | ≥1.28.0 | Web interface framework |
| **pytest** | ≥7.4.0 | Testing framework |

## Acknowledgments

- **Yahoo Finance** for providing free stock data API
- **yfinance library** for making data access easy
- **Streamlit** for the web framework
- **Dr. Fatema Nafa** for course instruction