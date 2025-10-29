"""
Module: config.py
Description: Central configuration file for stock dashboard project.
"""

import os
from pathlib import Path


# PROJECT PATHS

# Base directory (project root)
BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / 'src'
DATA_DIR = BASE_DIR / 'data'
CACHE_DIR = DATA_DIR / 'cache'
LOG_DIR = DATA_DIR / 'logs'
OUTPUT_DIR = DATA_DIR / 'outputs'
TEST_DIR = BASE_DIR / 'tests'

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DATA FETCHING CONFIGURATION

# Retry settings
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds
RETRY_BACKOFF = 2  # exponential multiplier (2 means 2s, 4s, 8s...)

# Cache settings
CACHE_EXPIRY_MINUTES_INTRADAY = 15  # for 1m, 5m, 15m, 1h intervals
CACHE_EXPIRY_HOURS_DAILY = 24  # for 1d, 1wk, 1mo intervals
MAX_CACHE_SIZE_MB = 500  # maximum cache size before cleanup
CACHE_CLEANUP_DAYS = 7  # delete cache files older than N days

# Yahoo Finance parameters
VALID_PERIODS = [
    '1d',  # 1 day
    '5d',  # 5 days
    '1mo',  # 1 month
    '3mo',  # 3 months
    '6mo',  # 6 months
    '1y',  # 1 year
    '2y',  # 2 years
    '5y',  # 5 years
    '10y',  # 10 years
    'ytd',  # year to date
    'max'  # maximum available
]

VALID_INTERVALS = [
    '1m',  # 1 minute
    '2m',  # 2 minutes
    '5m',  # 5 minutes
    '15m',  # 15 minutes
    '30m',  # 30 minutes
    '60m',  # 60 minutes (1 hour)
    '90m',  # 90 minutes
    '1h',  # 1 hour
    '1d',  # 1 day
    '5d',  # 5 days
    '1wk',  # 1 week
    '1mo',  # 1 month
    '3mo'  # 3 months
]

# Default parameters
DEFAULT_PERIOD = '1mo'
DEFAULT_INTERVAL = '1d'
DEFAULT_TICKER = 'AAPL'

# ANALYSIS CONFIGURATION

# Moving average windows (days)
DEFAULT_MA_WINDOWS = [20, 50, 200]
AVAILABLE_MA_WINDOWS = [5, 10, 20, 30, 50, 100, 200]

# Volatility settings
DEFAULT_VOLATILITY_WINDOW = 30
VOLATILITY_WINDOWS = [7, 14, 30, 60, 90]

# Financial constants
TRADING_DAYS_PER_YEAR = 252
TRADING_HOURS_PER_DAY = 6.5
RISK_FREE_RATE = 0.04

# Timeframe definitions
TIMEFRAMES = {
    '1d': {'days': 1, 'label': '1 Day'},
    '1m': {'days': 30, 'label': '1 Month'},
    '1y': {'days': 365, 'label': '1 Year'}
}

# VISUALIZATION CONFIGURATION

# Chart style
CHART_STYLE = 'seaborn-v0_8-whitegrid'
FALLBACK_STYLE = 'default'

# Color palette
COLOR_PALETTE = {
    'primary': '#2E86AB',  # Blue
    'secondary': '#A23B72',  # Purple
    'accent': '#F18F01',  # Orange
    'danger': '#C73E1D',  # Red
    'success': '#6A994E',  # Green
    'neutral': '#555555',  # Gray
    'background': '#FFFFFF',  # White
    'grid': '#CCCCCC'  # Light gray
}


# Figure dimensions
FIGURE_SIZE_SMALL = (10, 5)
FIGURE_SIZE_MEDIUM = (12, 6)
FIGURE_SIZE_LARGE = (14, 8)
FIGURE_SIZE_DASHBOARD = (14, 10)
FIGURE_SIZE_FULL = (16, 12)

# Resolution
DPI_SCREEN = 100
DPI_PRINT = 300
DPI_DEFAULT = DPI_SCREEN

# Font sizes
FONT_SIZE_SMALL = 8
FONT_SIZE_NORMAL = 10
FONT_SIZE_LARGE = 12
FONT_SIZE_TITLE = 14
FONT_SIZE_HEADER = 16

# Font configuration
FONT_FAMILY = 'sans-serif'
FONT_WEIGHT_NORMAL = 'normal'
FONT_WEIGHT_BOLD = 'bold'

# Grid settings
GRID_ALPHA = 0.3
GRID_LINESTYLE = '-'
GRID_LINEWIDTH = 0.5

# Line widths
LINEWIDTH_THIN = 1
LINEWIDTH_NORMAL = 1.5
LINEWIDTH_THICK = 2
LINEWIDTH_VERY_THICK = 3

# Chart-specific settings
CANDLESTICK_WIDTH = 0.6
VOLUME_BAR_WIDTH = 0.8
HISTOGRAM_BINS = 50
HISTOGRAM_ALPHA = 0.7

# LOGGING CONFIGURATION

# Log levels
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Log file settings
LOG_FILE = LOG_DIR / 'stock_dashboard.log'
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

# Enable logging to file
LOG_TO_FILE = True
LOG_TO_CONSOLE = True


# POPULAR TICKERS

POPULAR_TICKERS = {
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD'],
    'finance': ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'BLK', 'V', 'MA'],
    'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK', 'ABT', 'CVS'],
    'consumer': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'NKE', 'MCD', 'SBUX'],
    'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO'],
}

ALL_POPULAR_TICKERS = []
for category, tickers in POPULAR_TICKERS.items():
    ALL_POPULAR_TICKERS.extend(tickers)

# Default tickers for quick demo
DEFAULT_DEMO_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# ERROR MESSAGES

ERROR_MESSAGES = {
    # Data fetching errors
    'invalid_ticker': "'{ticker}' is not a valid stock symbol. Try: {suggestions}",
    'network_error': "Cannot connect to data source. {details}",
    'no_data': "No data available for {ticker} with period={period} and interval={interval}",
    'cache_not_found': "No cached data found for {ticker}",
    'cache_expired': "Cached data is outdated. Attempting to fetch fresh data...",

    # Analysis errors
    'insufficient_data': "Insufficient data for {calculation}. Need at least {required} data points",
    'calculation_error': "Error calculating {metric}: {details}",
    'missing_column': "Required column '{column}' not found in data",

    # Plotting errors
    'plot_error': "Error creating {chart_type} chart: {details}",
    'save_error': "Cannot save chart to {path}: {details}",

    # Input validation errors
    'invalid_period': "Invalid period '{period}'. Valid options: {valid_periods}",
    'invalid_interval': "Invalid interval '{interval}'. Valid options: {valid_intervals}",
    'invalid_parameter': "Invalid parameter {param}={value}",

    # General errors
    'unknown_error': "An unexpected error occurred: {details}",
    'permission_error': "Permission denied: {details}",
    'file_not_found': "File not found: {path}"
}

# Success messages
SUCCESS_MESSAGES = {
    'data_fetched': "Successfully fetched {rows} rows of data for {ticker}",
    'data_cached': "Data saved to cache: {filename}",
    'chart_created': "Created {chart_type} chart",
    'chart_saved': "Chart saved to {path}",
    'cache_cleared': "Cleared {count} cached files"
}

# Warning messages
WARNING_MESSAGES = {
    'using_cache': "Using cached data from {timestamp}",
    'partial_data': "Some data points are missing or invalid",
    'calculation_skipped': "Skipping {calculation} due to insufficient data"
}

# FEATURE FLAGS

# Enable/disable features
FEATURES = {
    'enable_caching': True,
    'enable_retry': True,
    'enable_logging': True,
    'enable_validation': True,
    'enable_export': True,
}

# TESTING CONFIGURATION

# Test settings
TEST_TICKER = 'AAPL'
TEST_PERIOD = '3mo'
TEST_INTERVAL = '1d'
TEST_OUTPUT_DIR = BASE_DIR / 'test_outputs'

# Create test output directory
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# ENVIRONMENT VARIABLES

# Load from environment if available
ENABLE_DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
CUSTOM_CACHE_DIR = os.getenv('CACHE_DIR', None)

if CUSTOM_CACHE_DIR:
    CACHE_DIR = Path(CUSTOM_CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)




