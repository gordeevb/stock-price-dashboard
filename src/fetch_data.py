"""
Module: fetch_data.py
Description: Handles stock data acquisition from Yahoo Finance with retry logic,
             caching, and comprehensive error handling.
"""

import logging         # Track events & errors
import time            # Retry delays
from datetime import datetime, timedelta  # Cache management
from pathlib import Path                  # File paths
from typing import Optional, Tuple        # Type hints
import pandas as pd                       # Data structure
import yfinance as yf                     # Download stock data
from functools import wraps               # Decorator metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
CACHE_DIR = Path('data/cache')
CACHE_EXPIRY_MINUTES_INTRADAY = 15
CACHE_EXPIRY_HOURS_DAILY = 24
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds
RETRY_BACKOFF = 2  # exponential multiplier

# Valid parameters
VALID_PERIODS = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
VALID_INTERVALS = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class DataFetchError(Exception):
    """Custom exception for data fetching errors"""
    pass


class InvalidTickerError(DataFetchError):
    """Exception raised for invalid ticker symbols"""
    pass


class NetworkError(DataFetchError):
    """Exception raised for network-related issues"""
    pass


def retry_on_failure(max_attempts: int = RETRY_ATTEMPTS, delay: int = RETRY_DELAY,
                     backoff: int = RETRY_BACKOFF):
    """
    Decorator to retry a function on failure with exponential backoff.

    Args:
        max_attempts (int): Maximum number of retry attempts
        delay (int): Initial delay between retries in seconds
        backoff (int): Multiplier for exponential backoff

    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {str(e)}"
                        )
                        logger.info(f"Retrying in {current_delay} seconds...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )

            raise last_exception
        return wrapper
    return decorator


def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """
    Validate ticker symbol format and existence.

    Args:
        ticker (str): Stock ticker symbol

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not ticker:
        return False, "Ticker symbol cannot be empty"

    # Remove whitespace and convert to uppercase
    ticker = ticker.strip().upper()

    # Check basic format (alphanumeric with possible dots and dashes)
    if not all(c.isalnum() or c in ['.', '-'] for c in ticker):
        return False, f"Invalid ticker format: '{ticker}'. Use only letters, numbers, dots, and dashes"

    # Check if ticker exists by trying to get info
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # yfinance returns a dict with just 'symbol' if ticker doesn't exist
        if 'regularMarketPrice' not in info and 'currentPrice' not in info:
            return False, f"Ticker '{ticker}' not found. Please verify the symbol"

        return True, ""
    except Exception as e:
        logger.warning(f"Could not validate ticker {ticker}: {str(e)}")
        # If validation fails, let the download attempt proceed
        return True, ""


def validate_parameters(period: str, interval: str) -> Tuple[bool, str]:
    """
    Validate period and interval parameters.

    Args:
        period (str): Time period for data ('1mo', '1y')
        interval (str): Data interval ('1d', '1h')

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    errors = []

    if period not in VALID_PERIODS:
        errors.append(
            f"Invalid period '{period}'. Valid options: {', '.join(VALID_PERIODS)}"
        )

    if interval not in VALID_INTERVALS:
        errors.append(
            f"Invalid interval '{interval}'. Valid options: {', '.join(VALID_INTERVALS)}"
        )

    # Check compatibility (intraday intervals need short periods)
    if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m'] and period not in ['1d', '5d']:
        errors.append(
            f"Interval '{interval}' requires period of '1d' or '5d' (Yahoo Finance limitation)"
        )

    if errors:
        return False, "; ".join(errors)

    return True, ""


def get_cache_filename(ticker: str, period: str, interval: str) -> Path:
    """
    Generate cache filename based on parameters.

    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period
        interval (str): Data interval

    Returns:
        Path: Full path to cache file
    """
    filename = f"{ticker.upper()}_{period}_{interval}.csv"
    return CACHE_DIR / filename


def is_cache_valid(cache_file: Path, interval: str) -> bool:
    """
    Check if cached data is still valid based on age.

    Args:
        cache_file (Path): Path to cache file
        interval (str): Data interval (determines expiry time)

    Returns:
        bool: True if cache is valid, False otherwise
    """
    if not cache_file.exists():
        return False

    # Get file modification time
    file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
    current_time = datetime.now()
    age = current_time - file_time

    # Determine expiry based on interval
    if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']:
        # Intraday data: 15 minutes expiry
        return age < timedelta(minutes=CACHE_EXPIRY_MINUTES_INTRADAY)
    else:
        # Daily or longer: 24 hours expiry
        return age < timedelta(hours=CACHE_EXPIRY_HOURS_DAILY)


def get_cached_data(ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """
    Retrieve data from cache if available and valid.

    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period
        interval (str): Data interval

    Returns:
        Optional[pd.DataFrame]: Cached data or None if not available
    """
    cache_file = get_cache_filename(ticker, period, interval)

    if not is_cache_valid(cache_file, interval):
        logger.info(f"Cache expired or not found for {ticker}")
        return None

    try:
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded cached data for {ticker} from {cache_file}")
        return df
    except Exception as e:
        logger.error(f"Error reading cache file {cache_file}: {str(e)}")
        return None


def save_to_cache(ticker: str, period: str, interval: str, data: pd.DataFrame) -> None:
    """
    Save data to cache file.

    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period
        interval (str): Data interval
        data (pd.DataFrame): Stock data to cache
    """
    cache_file = get_cache_filename(ticker, period, interval)

    try:
        data.to_csv(cache_file)
        logger.info(f"Saved data to cache: {cache_file}")
    except Exception as e:
        logger.error(f"Error saving to cache {cache_file}: {str(e)}")


def validate_dataframe(df: pd.DataFrame, ticker: str) -> Tuple[bool, str]:
    """
    Validate that downloaded DataFrame contains expected data.

    Args:
        df (pd.DataFrame): Downloaded stock data
        ticker (str): Stock ticker symbol

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if df is None:
        return False, "Received None instead of DataFrame"

    if df.empty:
        return False, f"No data available for {ticker}. Check ticker symbol and date range"

    # Check for required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"

    # Check for all NaN columns
    if df[required_columns].isna().all().any():
        nan_cols = df[required_columns].columns[df[required_columns].isna().all()].tolist()
        return False, f"Columns contain only NaN values: {', '.join(nan_cols)}"

    return True, ""


@retry_on_failure(max_attempts=RETRY_ATTEMPTS, delay=RETRY_DELAY, backoff=RETRY_BACKOFF)
def download_stock_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance with retry logic.

    This function is decorated with retry logic and will attempt to download
    data multiple times before failing.

    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period (e.g., '1mo', '1y', 'max')
        interval (str): Data interval (e.g., '1d', '1h', '1wk')

    Returns:
        pd.DataFrame: Stock data with OHLCV columns

    Raises:
        NetworkError: If download fails after all retries
        InvalidTickerError: If ticker is invalid
    """
    logger.info(f"Downloading data for {ticker} (period={period}, interval={interval})")

    try:
        # Download data
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        # Validate data
        is_valid, error_msg = validate_dataframe(df, ticker)
        if not is_valid:
            raise InvalidTickerError(error_msg)

        logger.info(f"Successfully downloaded {len(df)} rows for {ticker}")
        return df

    except InvalidTickerError:
        raise
    except Exception as e:
        error_msg = f"Failed to download data for {ticker}: {str(e)}"
        logger.error(error_msg)
        raise NetworkError(error_msg)


def fetch_stock_data(ticker: str, period: str = '1mo', interval: str = '1d',
                     use_cache: bool = True) -> Tuple[pd.DataFrame, dict]:
    """
    Main function to fetch stock data with caching and error handling.

    This is the primary function to use for getting stock data. It handles:
    - Input validation
    - Cache checking and retrieval
    - Data downloading with retries
    - Cache saving
    - Comprehensive error handling

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        period (str): Time period - '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'
        interval (str): Data interval - '1m', '5m', '15m', '1h', '1d', '1wk', '1mo'
        use_cache (bool): Whether to use cached data if available

    Returns:
        Tuple[pd.DataFrame, dict]: (stock_data, metadata)
            metadata contains: {
                'ticker': str,
                'period': str,
                'interval': str,
                'from_cache': bool,
                'cached_time': str or None,
                'rows': int,
                'date_range': tuple(str, str),
                'warnings': list of str
            }

    Raises:
        InvalidTickerError: If ticker validation fails
        DataFetchError: If data cannot be fetched and no cache is available
    """
    # Initialize metadata
    metadata = {
        'ticker': ticker.upper(),
        'period': period,
        'interval': interval,
        'from_cache': False,
        'cached_time': None,
        'rows': 0,
        'date_range': (None, None),
        'warnings': []
    }

    # Validate ticker
    is_valid, error_msg = validate_ticker(ticker)
    if not is_valid:
        logger.error(error_msg)
        raise InvalidTickerError(error_msg)

    # Validate parameters
    is_valid, error_msg = validate_parameters(period, interval)
    if not is_valid:
        logger.error(error_msg)
        raise DataFetchError(error_msg)

    ticker = ticker.upper()

    # Try to get cached data first
    if use_cache:
        cached_df = get_cached_data(ticker, period, interval)
        if cached_df is not None:
            cache_file = get_cache_filename(ticker, period, interval)
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)

            metadata['from_cache'] = True
            metadata['cached_time'] = cache_time.strftime('%Y-%m-%d %H:%M:%S')
            metadata['rows'] = len(cached_df)
            metadata['date_range'] = (
                cached_df.index[0].strftime('%Y-%m-%d'),
                cached_df.index[-1].strftime('%Y-%m-%d')
            )
            metadata['warnings'].append(f"Using cached data from {metadata['cached_time']}")

            logger.info(f"Returning cached data for {ticker}")
            return cached_df, metadata

    # Download fresh data
    try:
        df = download_stock_data(ticker, period, interval)

        # Save to cache
        save_to_cache(ticker, period, interval, df)

        # Update metadata
        metadata['rows'] = len(df)
        metadata['date_range'] = (
            df.index[0].strftime('%Y-%m-%d'),
            df.index[-1].strftime('%Y-%m-%d')
        )

        logger.info(f"Successfully fetched {len(df)} rows for {ticker}")
        return df, metadata

    except NetworkError as e:
        # Network error: try to use cached data even if expired
        logger.warning(f"Network error occurred: {str(e)}")

        cached_df = get_cached_data(ticker, period, interval)
        if cached_df is None:
            # Try to load any cache even if expired
            cache_file = get_cache_filename(ticker, period, interval)
            if cache_file.exists():
                try:
                    cached_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    logger.info(f"Loaded expired cache as fallback for {ticker}")
                except:
                    pass

        if cached_df is not None:
            cache_file = get_cache_filename(ticker, period, interval)
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)

            metadata['from_cache'] = True
            metadata['cached_time'] = cache_time.strftime('%Y-%m-%d %H:%M:%S')
            metadata['rows'] = len(cached_df)
            metadata['date_range'] = (
                cached_df.index[0].strftime('%Y-%m-%d'),
                cached_df.index[-1].strftime('%Y-%m-%d')
            )
            metadata['warnings'].append(
                f"Network error: Using cached data from {metadata['cached_time']}"
            )

            logger.info(f"Returning cached data as fallback for {ticker}")
            return cached_df, metadata
        else:
            error_msg = f"Cannot fetch data for {ticker} and no cache available"
            logger.error(error_msg)
            raise DataFetchError(error_msg)

    except InvalidTickerError:
        raise

    except Exception as e:
        error_msg = f"Unexpected error fetching data for {ticker}: {str(e)}"
        logger.error(error_msg)
        raise DataFetchError(error_msg)