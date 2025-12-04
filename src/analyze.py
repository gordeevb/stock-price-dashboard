"""
Stock Analysis Module - Updated with CMF Window Parameter

This module provides functions for analyzing stock market data including
returns calculation, volatility analysis, and technical indicators.
Now includes configurable Chaikin Money Flow calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import configuration
try:
    from config import (
        DEFAULT_MA_WINDOWS,
        DEFAULT_VOLATILITY_WINDOW,
        TRADING_DAYS_PER_YEAR,
        RISK_FREE_RATE
    )
except ImportError:
    DEFAULT_MA_WINDOWS = [20, 50, 200]
    DEFAULT_VOLATILITY_WINDOW = 30
    TRADING_DAYS_PER_YEAR = 252
    RISK_FREE_RATE = 0.02

class AnalysisError(Exception):
    """Custom exception for analysis errors"""
    pass


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, str]:
    """
    Validate DataFrame has required columns and data

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None:
        return False, "DataFrame is None"

    if df.empty:
        return False, "DataFrame is empty"

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"

    for col in required_columns:
        if df[col].isna().all():
            return False, f"Column '{col}' contains only NaN values"

    return True, ""


def calculate_returns(df: pd.DataFrame, column: str = 'Close') -> pd.Series:
    """
    Calculate percentage returns

    Args:
        df: DataFrame with price data
        column: Column name to calculate returns for

    Returns:
        Series of percentage returns
    """
    is_valid, error_msg = validate_dataframe(df, [column])
    if not is_valid:
        raise AnalysisError(f"Invalid data for returns calculation: {error_msg}")

    returns = df[column].pct_change() * 100
    return returns.dropna()


def calculate_cumulative_return(df: pd.DataFrame, column: str = 'Close') -> float:
    """
    Calculate cumulative return over the entire period

    Args:
        df: DataFrame with price data
        column: Column name to calculate return for

    Returns:
        Cumulative return as percentage
    """
    is_valid, error_msg = validate_dataframe(df, [column])
    if not is_valid:
        raise AnalysisError(f"Invalid data for cumulative return: {error_msg}")

    if len(df) < 2:
        return 0.0

    start_price = df[column].iloc[0]
    end_price = df[column].iloc[-1]

    cumulative_return = ((end_price - start_price) / start_price) * 100
    return cumulative_return


def calculate_volatility(df: pd.DataFrame, window: int = DEFAULT_VOLATILITY_WINDOW,
                        annualize: bool = True, column: str = 'Close') -> float:
    """
    Calculate rolling volatility (standard deviation of returns)

    Args:
        df: DataFrame with price data
        window: Rolling window size
        annualize: Whether to annualize the volatility
        column: Column name to calculate volatility for

    Returns:
        Volatility as percentage
    """
    is_valid, error_msg = validate_dataframe(df, [column])
    if not is_valid:
        raise AnalysisError(f"Invalid data for volatility calculation: {error_msg}")

    if len(df) < window:
        logger.warning(f"Not enough data for {window}-day volatility. Using available data.")
        window = max(2, len(df))

    returns = calculate_returns(df, column)
    volatility = returns.rolling(window=window).std().iloc[-1]

    if annualize:
        volatility = volatility * np.sqrt(TRADING_DAYS_PER_YEAR)

    return volatility


def calculate_moving_average(df: pd.DataFrame, window: int, column: str = 'Close') -> pd.Series:
    """
    Calculate simple moving average

    Args:
        df: DataFrame with price data
        window: Window size for moving average
        column: Column name to calculate MA for

    Returns:
        Series of moving average values
    """
    is_valid, error_msg = validate_dataframe(df, [column])
    if not is_valid:
        raise AnalysisError(f"Invalid data for moving average: {error_msg}")

    return df[column].rolling(window=window).mean()


def get_price_change(df: pd.DataFrame, timeframe: str = '1d',
                     column: str = 'Close') -> Dict[str, float]:
    """
    Calculate price change over specified timeframe

    Args:
        df: DataFrame with price data
        timeframe: Timeframe for price change ('1d', '1w', '1mo', etc.)
        column: Column name to calculate change for

    Returns:
        Dictionary with absolute and percentage change
    """
    is_valid, error_msg = validate_dataframe(df, [column])
    if not is_valid:
        raise AnalysisError(f"Invalid data for price change: {error_msg}")

    periods_map = {
        '1d': 1,
        '1w': 5,
        '1mo': 21,
        '3mo': 63,
        '6mo': 126,
        '1y': 252
    }

    periods = periods_map.get(timeframe, 1)

    if len(df) < periods + 1:
        periods = max(1, len(df) - 1)

    if periods == 0:
        return {'absolute_change': 0.0, 'percentage_change': 0.0}

    current_price = df[column].iloc[-1]
    previous_price = df[column].iloc[-periods-1]

    absolute_change = current_price - previous_price
    percentage_change = (absolute_change / previous_price) * 100

    return {
        'absolute_change': absolute_change,
        'percentage_change': percentage_change
    }


def calculate_volume_metrics(df: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
    """
    Calculate volume-related metrics

    Args:
        df: DataFrame with volume data
        window: Window for average volume calculation

    Returns:
        Dictionary with volume metrics
    """
    is_valid, error_msg = validate_dataframe(df, ['Volume'])
    if not is_valid:
        raise AnalysisError(f"Invalid data for volume metrics: {error_msg}")

    current_volume = df['Volume'].iloc[-1]

    if len(df) >= window:
        average_volume = df['Volume'].rolling(window=window).mean().iloc[-1]
    else:
        average_volume = df['Volume'].mean()

    volume_ratio = current_volume / average_volume if average_volume > 0 else 1.0
    is_spike = volume_ratio > 1.5

    return {
        'current_volume': int(current_volume),
        'average_volume': int(average_volume),
        'volume_ratio': volume_ratio,
        'is_spike': is_spike
    }


def calculate_chaikin_money_flow(df: pd.DataFrame, window: int = 20) -> Optional[float]:
    """
    Calculate Chaikin Money Flow (CMF)

    CMF measures buying and selling pressure. Positive values indicate buying
    pressure, negative values indicate selling pressure.

    Args:
        df: DataFrame with OHLCV data
        window: Window for CMF calculation

    Returns:
        CMF value between -1 and 1, or None if insufficient data
    """
    required_columns = ['High', 'Low', 'Close', 'Volume']
    is_valid, error_msg = validate_dataframe(df, required_columns)
    if not is_valid:
        logger.warning(f"Cannot calculate CMF: {error_msg}")
        return None

    if len(df) < window:
        logger.warning(f"Not enough data for {window}-period CMF. Need at least {window} rows.")
        return None

    try:
        # Money Flow Multiplier
        high_low_diff = df['High'] - df['Low']

        # Avoid division by zero
        high_low_diff = high_low_diff.replace(0, np.nan)

        mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / high_low_diff

        # Money Flow Volume
        mfv = mfm * df['Volume']

        # CMF calculation
        mfv_sum = mfv.rolling(window=window).sum()
        volume_sum = df['Volume'].rolling(window=window).sum()

        # Avoid division by zero
        volume_sum = volume_sum.replace(0, np.nan)

        cmf = mfv_sum / volume_sum

        # Get the most recent valid CMF value
        cmf_value = cmf.iloc[-1]

        # Return None if NaN
        if pd.isna(cmf_value):
            return None

        # Ensure CMF is within valid range
        cmf_value = max(-1.0, min(1.0, float(cmf_value)))

        return cmf_value

    except Exception as e:
        logger.error(f"Error calculating CMF: {str(e)}")
        return None


def generate_summary_statistics(df: pd.DataFrame, ticker: str = "UNKNOWN",
                                ma_windows: Optional[List[int]] = None,
                                cmf_window: int = 20) -> Dict[str, Any]:
    """
    Generate comprehensive summary statistics for stock data

    Args:
        df: DataFrame with stock data
        ticker: Stock ticker symbol
        ma_windows: List of moving average windows to calculate
        cmf_window: Window for Chaikin Money Flow calculation

    Returns:
        Dictionary containing various statistics
    """
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    is_valid, error_msg = validate_dataframe(df, required_columns)
    if not is_valid:
        raise AnalysisError(f"Invalid data for summary statistics: {error_msg}")

    if ma_windows is None:
        ma_windows = DEFAULT_MA_WINDOWS

    stats = {
        'ticker': ticker,
        'data_points': len(df),
        'start_date': df.index[0].strftime('%Y-%m-%d'),
        'end_date': df.index[-1].strftime('%Y-%m-%d'),
    }

    # Price statistics
    stats['current_price'] = float(df['Close'].iloc[-1])
    stats['period_high'] = float(df['High'].max())
    stats['period_low'] = float(df['Low'].min())
    stats['period_open'] = float(df['Open'].iloc[0])

    # Returns
    try:
        returns = calculate_returns(df)
        stats['avg_daily_return'] = float(returns.mean())
        stats['cumulative_return'] = float(calculate_cumulative_return(df))
    except Exception as e:
        logger.warning(f"Could not calculate returns: {str(e)}")
        stats['avg_daily_return'] = None
        stats['cumulative_return'] = None

    # Volatility
    try:
        stats['volatility_30d'] = float(calculate_volatility(df, window=30))
    except Exception as e:
        logger.warning(f"Could not calculate volatility: {str(e)}")
        stats['volatility_30d'] = None

    # Moving averages
    stats['moving_averages'] = {}
    for window in ma_windows:
        try:
            if len(df) >= window:
                ma = calculate_moving_average(df, window)
                stats['moving_averages'][f'MA_{window}'] = float(ma.iloc[-1])
        except Exception as e:
            logger.warning(f"Could not calculate MA{window}: {str(e)}")

    # Price changes
    stats['price_changes'] = {}
    for timeframe in ['1d', '1w', '1mo']:
        try:
            change = get_price_change(df, timeframe)
            stats['price_changes'][timeframe] = change
        except Exception as e:
            logger.warning(f"Could not calculate {timeframe} price change: {str(e)}")

    # Volume metrics
    try:
        volume_metrics = calculate_volume_metrics(df)
        stats['volume_metrics'] = volume_metrics
    except Exception as e:
        logger.warning(f"Could not calculate volume metrics: {str(e)}")
        stats['volume_metrics'] = None

    # Chaikin Money Flow
    try:
        cmf = calculate_chaikin_money_flow(df, window=cmf_window)
        stats['cmf'] = cmf
    except Exception as e:
        logger.warning(f"Could not calculate CMF: {str(e)}")
        stats['cmf'] = None

    return stats


def create_statistics_table(stats: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a formatted table from statistics dictionary

    Args:
        stats: Dictionary of statistics from generate_summary_statistics

    Returns:
        DataFrame with formatted statistics
    """
    rows = []

    # Basic info
    rows.append(('Ticker', stats['ticker']))
    rows.append(('Data Points', f"{stats['data_points']:,}"))
    rows.append(('Date Range', f"{stats['start_date']} to {stats['end_date']}"))

    # Price metrics
    rows.append(('Current Price', f"${stats['current_price']:.2f}"))
    rows.append(('Period High', f"${stats['period_high']:.2f}"))
    rows.append(('Period Low', f"${stats['period_low']:.2f}"))

    # Returns
    if stats['cumulative_return'] is not None:
        rows.append(('Cumulative Return', f"{stats['cumulative_return']:.2f}%"))
    if stats['avg_daily_return'] is not None:
        rows.append(('Avg Daily Return', f"{stats['avg_daily_return']:.4f}%"))

    # Volatility
    if stats['volatility_30d'] is not None:
        rows.append(('Volatility (30d)', f"{stats['volatility_30d']:.2f}%"))

    # Moving averages
    for ma_name, ma_value in stats.get('moving_averages', {}).items():
        rows.append((ma_name, f"${ma_value:.2f}"))

    # Volume
    if stats.get('volume_metrics'):
        vm = stats['volume_metrics']
        rows.append(('Current Volume', f"{vm['current_volume']:,}"))
        rows.append(('Avg Volume (20d)', f"{vm['average_volume']:,}"))

    # CMF
    if stats.get('cmf') is not None:
        rows.append(('Chaikin Money Flow', f"{stats['cmf']:.3f}"))

    df = pd.DataFrame(rows, columns=['Metric', 'Value'])
    return df


if __name__ == "__main__":
    logger.info("Analysis module loaded successfully")