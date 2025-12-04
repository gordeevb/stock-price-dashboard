"""
Module: analyze.py
Description: Financial analysis and metrics calculation for stock data.
             All calculations include clear mathematical definitions.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Analysis constants
DEFAULT_VOLATILITY_WINDOW = 30  # days
DEFAULT_MA_WINDOWS = [20, 50, 200]  # moving average periods
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.04  # 4% default risk-free rate


class AnalysisError(Exception):
    """Custom exception for analysis errors"""
    pass


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, str]:
    """
    Validate that DataFrame has required columns and data.

    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (List[str]): List of required column names

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty or None"

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"

    return True, ""


def calculate_returns(df: pd.DataFrame, column: str = 'Close') -> pd.Series:
    """
    Calculate daily percentage returns.

    Mathematical Definition:
        Daily Return (R_t) = (P_t - P_(t-1)) / P_(t-1) × 100%

        Where:
        - P_t = Price at time t
        - P_(t-1) = Price at time t-1 (previous period)

    Args:
        df (pd.DataFrame): Stock data with price column
        column (str): Column name to calculate returns from (default: 'Close')

    Returns:
        pd.Series: Daily percentage returns

    Raises:
        AnalysisError: If required column is missing
    """
    is_valid, error_msg = validate_dataframe(df, [column])
    if not is_valid:
        raise AnalysisError(error_msg)

    try:
        # Calculate percentage change: (price_today - price_yesterday) / price_yesterday * 100
        returns = df[column].pct_change() * 100

        # Remove NaN from first value
        returns = returns.dropna()

        logger.info(f"Calculated {len(returns)} daily returns from '{column}' column")
        return returns

    except Exception as e:
        error_msg = f"Error calculating returns: {str(e)}"
        logger.error(error_msg)
        raise AnalysisError(error_msg)


def calculate_cumulative_return(df: pd.DataFrame, column: str = 'Close') -> float:
    """
    Calculate total cumulative return over the entire period.

    Mathematical Definition:
        Cumulative Return = (P_end - P_start) / P_start × 100%

        Where:
        - P_start = First price in period
        - P_end = Last price in period

    Args:
        df (pd.DataFrame): Stock data with price column
        column (str): Column name to calculate from (default: 'Close')

    Returns:
        float: Cumulative return as percentage
    """
    is_valid, error_msg = validate_dataframe(df, [column])
    if not is_valid:
        raise AnalysisError(error_msg)

    try:
        start_price = df[column].iloc[0]
        end_price = df[column].iloc[-1]

        cum_return = ((end_price - start_price) / start_price) * 100

        logger.info(f"Cumulative return: {cum_return:.2f}%")
        return cum_return

    except Exception as e:
        error_msg = f"Error calculating cumulative return: {str(e)}"
        logger.error(error_msg)
        raise AnalysisError(error_msg)


def calculate_volatility(df: pd.DataFrame, window: int = DEFAULT_VOLATILITY_WINDOW,
                        column: str = 'Close', annualize: bool = True) -> float:
    """
    Calculate volatility using standard deviation of returns.

    Mathematical Definition:
        1. Calculate daily returns: R_t = (P_t - P_(t-1)) / P_(t-1)
        2. Volatility = σ(R) = sqrt(Σ(R_i - μ)² / (n-1))
        3. If annualize=True: σ_annual = σ_daily × sqrt(252)

        Where:
        - σ = standard deviation
        - μ = mean of returns
        - n = number of observations
        - 252 = typical trading days per year

    Args:
        df (pd.DataFrame): Stock data with price column
        window (int): Number of days to calculate volatility over (default: 30)
        column (str): Column name to use (default: 'Close')
        annualize (bool): Whether to annualize the volatility (default: True)

    Returns:
        float: Volatility as percentage

    """
    is_valid, error_msg = validate_dataframe(df, [column])
    if not is_valid:
        raise AnalysisError(error_msg)

    try:
        # Calculate daily returns
        returns = calculate_returns(df, column)

        # Use only the most recent 'window' days if data is longer
        if len(returns) > window:
            returns = returns.tail(window)

        # Calculate standard deviation (volatility)
        volatility = returns.std()

        # Annualize if requested
        if annualize:
            volatility = volatility * np.sqrt(TRADING_DAYS_PER_YEAR)

        logger.info(f"Calculated volatility: {volatility:.2f}% (window={window}, annualized={annualize})")
        return volatility

    except Exception as e:
        error_msg = f"Error calculating volatility: {str(e)}"
        logger.error(error_msg)
        raise AnalysisError(error_msg)


def calculate_moving_average(df: pd.DataFrame, window: int, column: str = 'Close') -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).

    Mathematical Definition:
        SMA_t = (P_t + P_(t-1) + ... + P_(t-window+1)) / window

        Where:
        - P_t = Price at time t
        - window = Number of periods to average

    Args:
        df (pd.DataFrame): Stock data
        window (int): Number of periods for moving average
        column (str): Column to calculate MA from (default: 'Close')

    Returns:
        pd.Series: Simple moving average

    """
    is_valid, error_msg = validate_dataframe(df, [column])
    if not is_valid:
        raise AnalysisError(error_msg)

    try:
        sma = df[column].rolling(window=window).mean()
        logger.info(f"Calculated {window}-period moving average")
        return sma

    except Exception as e:
        error_msg = f"Error calculating moving average: {str(e)}"
        logger.error(error_msg)
        raise AnalysisError(error_msg)


def get_price_change(df: pd.DataFrame, timeframe: str = '1d', column: str = 'Close') -> Dict:
    """
    Calculate price change over specified timeframe.

    Mathematical Definition:
        Change = (P_current - P_past) / P_past × 100%

        Where:
        - P_current = Most recent price
        - P_past = Price N days ago

    Args:
        df (pd.DataFrame): Stock data
        timeframe (str): '1d', '1m', or '1y'
        column (str): Column to use (default: 'Close')

    Returns:
        Dict: {
            'absolute_change': float,
            'percentage_change': float,
            'start_price': float,
            'end_price': float,
            'start_date': str,
            'end_date': str
        }
    """
    is_valid, error_msg = validate_dataframe(df, [column])
    if not is_valid:
        raise AnalysisError(error_msg)

    try:
        current_price = df[column].iloc[-1]
        current_date = df.index[-1]

        # Determine lookback period (simplified to 1d, 1m, 1y)
        if timeframe == '1d':
            lookback_days = 1
        elif timeframe == '1m':
            lookback_days = 30
        elif timeframe == '1y':
            lookback_days = 365
        else:
            raise AnalysisError(f"Invalid timeframe: {timeframe}. Valid options: '1d', '1m', '1y'")

        # Get past price
        target_date = current_date - timedelta(days=lookback_days)
        # Find closest date in DataFrame
        past_idx = df.index.get_indexer([target_date], method='nearest')[0]
        past_price = df[column].iloc[past_idx]
        past_date = df.index[past_idx]

        # Calculate changes
        absolute_change = current_price - past_price
        percentage_change = (absolute_change / past_price) * 100

        result = {
            'absolute_change': float(absolute_change),
            'percentage_change': float(percentage_change),
            'start_price': float(past_price),
            'end_price': float(current_price),
            'start_date': past_date.strftime('%Y-%m-%d'),
            'end_date': current_date.strftime('%Y-%m-%d'),
            'timeframe': timeframe
        }

        logger.info(f"Price change ({timeframe}): {percentage_change:.2f}%")
        return result

    except AnalysisError:
        raise
    except Exception as e:
        error_msg = f"Error calculating price change: {str(e)}"
        logger.error(error_msg)
        raise AnalysisError(error_msg)


def calculate_volume_metrics(df: pd.DataFrame, window: int = 20) -> Dict:
    """
    Calculate volume-related metrics.

    Mathematical Definitions:
        Average Volume = mean(Volume) over window
        Current vs Avg = (Current Volume / Avg Volume) × 100%
        Volume Spike = Current Volume > 2 × Avg Volume

    Args:
        df (pd.DataFrame): Stock data with Volume column
        window (int): Window for average calculation (default: 20)

    Returns:
        Dict: {
            'current_volume': int,
            'average_volume': float,
            'volume_ratio': float,
            'is_spike': bool
        }
    """
    is_valid, error_msg = validate_dataframe(df, ['Volume'])
    if not is_valid:
        raise AnalysisError(error_msg)

    try:
        current_volume = int(df['Volume'].iloc[-1])
        avg_volume = df['Volume'].tail(window).mean()
        volume_ratio = (current_volume / avg_volume) * 100
        is_spike = current_volume > (2 * avg_volume)

        return {
            'current_volume': current_volume,
            'average_volume': float(avg_volume),
            'volume_ratio': float(volume_ratio),
            'is_spike': bool(is_spike)
        }

    except Exception as e:
        error_msg = f"Error calculating volume metrics: {str(e)}"
        logger.error(error_msg)
        raise AnalysisError(error_msg)


def calculate_chaikin_money_flow(df: pd.DataFrame, window: int = 20) -> float:
    """
    Calculate Chaikin Money Flow (CMF) indicator.

    Mathematical Definition:
        CMF = Σ(Money Flow Volume) / Σ(Volume) over n periods

        Where:
        Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
        Money Flow Volume = Money Flow Multiplier × Volume

        CMF ranges from -1 to +1:
        - Positive CMF: Buying pressure (accumulation)
        - Negative CMF: Selling pressure (distribution)
        - CMF near 0: Neutral or balanced pressure

    Args:
        df (pd.DataFrame): Stock data with High, Low, Close, Volume
        window (int): Number of periods for CMF calculation (default: 20)

    Returns:
        float: Chaikin Money Flow value (between -1 and +1)
    """
    required_cols = ['High', 'Low', 'Close', 'Volume']
    is_valid, error_msg = validate_dataframe(df, required_cols)
    if not is_valid:
        raise AnalysisError(error_msg)

    # Adjust window if not enough data
    original_window = window
    if len(df) < window:
        window = max(5, len(df) // 2)  # Use smaller window, minimum 5 days
        logger.warning(f"Not enough data for {original_window}-day CMF, using {window}-day instead")

    if len(df) < 5:
        raise AnalysisError(f"Not enough data points ({len(df)}) for CMF calculation (minimum 5)")

    try:
        # Calculate Money Flow Multiplier
        # MFM = [(Close - Low) - (High - Close)] / (High - Low)
        high_low = df['High'] - df['Low']

        # Avoid division by zero
        high_low = high_low.replace(0, np.nan)

        mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / high_low

        # Replace NaN with 0 (happens when High == Low)
        mfm = mfm.fillna(0)

        # Calculate Money Flow Volume
        # MFV = MFM × Volume
        mfv = mfm * df['Volume']

        # Calculate CMF over the window period
        # CMF = Sum(MFV over n periods) / Sum(Volume over n periods)
        sum_mfv = mfv.rolling(window=window).sum()
        sum_volume = df['Volume'].rolling(window=window).sum()

        cmf = sum_mfv / sum_volume

        # Get the most recent CMF value
        current_cmf = float(cmf.iloc[-1])

        logger.info(f"Calculated Chaikin Money Flow ({window}-day): {current_cmf:.4f}")
        return current_cmf

    except Exception as e:
        error_msg = f"Error calculating Chaikin Money Flow: {str(e)}"
        logger.error(error_msg)
        raise AnalysisError(error_msg)


def generate_summary_statistics(df: pd.DataFrame, ticker: str = "",
                                ma_windows: List[int] = None) -> Dict:
    """
    Generate comprehensive summary statistics for stock data.

    Focuses on essential metrics:
    - Daily returns
    - Volatility (30-day)
    - SMA (20, 50, 200-day)
    - Chaikin Money Flow (CMF)
    - Price changes (1d, 1m, 1y)

    Args:
        df (pd.DataFrame): Stock data
        ticker (str): Stock ticker symbol (for labeling)
        ma_windows (List[int]): Moving average windows (default: [20, 50, 200])

    Returns:
        Dict: Comprehensive statistics dictionary with essential metrics
    """
    if ma_windows is None:
        ma_windows = DEFAULT_MA_WINDOWS

    is_valid, error_msg = validate_dataframe(df, ['Open', 'High', 'Low', 'Close', 'Volume'])
    if not is_valid:
        raise AnalysisError(error_msg)

    try:
        stats = {
            'ticker': ticker,
            'data_points': len(df),
            'start_date': df.index[0].strftime('%Y-%m-%d'),
            'end_date': df.index[-1].strftime('%Y-%m-%d'),
        }

        # Current price and basic info
        stats['current_price'] = float(df['Close'].iloc[-1])
        stats['previous_close'] = float(df['Close'].iloc[-2]) if len(df) > 1 else None

        # Price extremes
        stats['period_high'] = float(df['High'].max())
        stats['period_low'] = float(df['Low'].min())
        stats['period_high_date'] = df['High'].idxmax().strftime('%Y-%m-%d')
        stats['period_low_date'] = df['Low'].idxmin().strftime('%Y-%m-%d')

        # Price changes for 1d, 1m, 1y only
        timeframes = ['1d', '1m', '1y']
        stats['price_changes'] = {}

        for tf in timeframes:
            try:
                change = get_price_change(df, tf)
                stats['price_changes'][tf] = {
                    'absolute': change['absolute_change'],
                    'percentage': change['percentage_change']
                }
            except:
                # If timeframe data not available, skip
                pass

        # Daily returns
        try:
            returns = calculate_returns(df)
            stats['avg_daily_return'] = float(returns.mean())
            stats['cumulative_return'] = float(calculate_cumulative_return(df))
        except Exception as e:
            logger.warning(f"Could not calculate returns: {str(e)}")
            stats['avg_daily_return'] = None
            stats['cumulative_return'] = None

        # Volatility (30-day only)
        try:
            stats['volatility_30d'] = float(calculate_volatility(df, window=30))
        except Exception as e:
            logger.warning(f"Could not calculate volatility: {str(e)}")
            stats['volatility_30d'] = None

        # Simple Moving Averages (SMA)
        stats['moving_averages'] = {}
        for window in ma_windows:
            try:
                if len(df) >= window:
                    ma = calculate_moving_average(df, window)
                    stats['moving_averages'][f'sma_{window}'] = float(ma.iloc[-1])
                    # Price vs MA (percentage above/below)
                    stats['moving_averages'][f'price_vs_sma_{window}'] = \
                        float((stats['current_price'] - ma.iloc[-1]) / ma.iloc[-1] * 100)
            except Exception as e:
                logger.warning(f"Could not calculate SMA({window}): {str(e)}")

        # Volume metrics
        try:
            vol_metrics = calculate_volume_metrics(df)
            stats['volume'] = vol_metrics
        except Exception as e:
            logger.warning(f"Could not calculate volume metrics: {str(e)}")
            stats['volume'] = None

        # Chaikin Money Flow (CMF)
        try:
            stats['cmf'] = float(calculate_chaikin_money_flow(df, window=20))
        except Exception as e:
            logger.warning(f"Could not calculate Chaikin Money Flow: {str(e)}")
            stats['cmf'] = None

        logger.info(f"Generated summary statistics for {ticker}")
        return stats

    except Exception as e:
        error_msg = f"Error generating summary statistics: {str(e)}"
        logger.error(error_msg)
        raise AnalysisError(error_msg)


def create_statistics_table(stats: Dict) -> pd.DataFrame:
    """
    Create a Pandas DataFrame table from statistics dictionary.

    Args:
        stats (Dict): Statistics dictionary from generate_summary_statistics()

    Returns:
        pd.DataFrame: Formatted table with metrics and values
    """
    rows = []

    # Header info
    rows.append(['Ticker', stats['ticker']])
    rows.append(['Period', f"{stats['start_date']} to {stats['end_date']}"])
    rows.append(['Data Points', stats['data_points']])
    rows.append(['', ''])  # Blank row

    # Current Price Section
    rows.append(['CURRENT PRICE', ''])
    rows.append(['Current Price', f"${stats['current_price']:.2f}"])
    if stats['previous_close']:
        rows.append(['Previous Close', f"${stats['previous_close']:.2f}"])
        daily_change = stats['current_price'] - stats['previous_close']
        daily_pct = (daily_change / stats['previous_close']) * 100
        rows.append(['Daily Change', f"${daily_change:+.2f} ({daily_pct:+.2f}%)"])
    rows.append(['', ''])  # Blank row

    # Price Extremes Section
    rows.append(['PRICE EXTREMES', ''])
    rows.append(['Period High', f"${stats['period_high']:.2f} ({stats['period_high_date']})"])
    rows.append(['Period Low', f"${stats['period_low']:.2f} ({stats['period_low_date']})"])
    rows.append(['', ''])  # Blank row

    # Price Changes Section (1d, 1m, 1y)
    if stats['price_changes']:
        rows.append(['PRICE CHANGES', ''])
        for tf in ['1d', '1m', '1y']:
            if tf in stats['price_changes']:
                change = stats['price_changes'][tf]
                rows.append([f'{tf.upper()} Change', f"${change['absolute']:+.2f} ({change['percentage']:+.2f}%)"])
        rows.append(['', ''])  # Blank row

    # Returns Section
    rows.append(['RETURNS', ''])
    if stats['avg_daily_return'] is not None:
        rows.append(['Avg Daily Return', f"{stats['avg_daily_return']:.4f}%"])
    if stats['cumulative_return'] is not None:
        rows.append(['Cumulative Return', f"{stats['cumulative_return']:.2f}%"])
    rows.append(['', ''])  # Blank row

    # Volatility Section (30-day only)
    if stats['volatility_30d'] is not None:
        rows.append(['VOLATILITY', ''])
        rows.append(['30-Day (Annualized)', f"{stats['volatility_30d']:.2f}%"])
        rows.append(['', ''])  # Blank row

    # Moving Averages Section (SMA and EMA)
    if stats['moving_averages']:
        rows.append(['MOVING AVERAGES', ''])

        # SMA
        for key in ['sma_20', 'sma_50', 'sma_200']:
            if key in stats['moving_averages']:
                value = stats['moving_averages'][key]
                ma_period = key.split('_')[1]
                price_vs_key = f'price_vs_sma_{ma_period}'
                vs_pct = stats['moving_averages'].get(price_vs_key, 0)
                rows.append([f'SMA-{ma_period}', f"${value:.2f} (price {vs_pct:+.2f}% vs MA)"])

        rows.append(['', ''])  # Blank row

    # Volume Section
    if stats['volume']:
        rows.append(['VOLUME', ''])
        vol = stats['volume']
        rows.append(['Current Volume', f"{vol['current_volume']:,}"])
        rows.append(['20-Day Avg Volume', f"{vol['average_volume']:,.0f}"])
        rows.append(['Current vs Avg', f"{vol['volume_ratio']:.1f}%"])
        if vol['is_spike']:
            rows.append(['Volume Alert', 'Volume Spike Detected!'])
        rows.append(['', ''])  # Blank row

    # Money Flow Metrics Section
    rows.append(['MONEY FLOW', ''])
    if stats['cmf'] is not None:
        cmf_value = stats['cmf']
        rows.append(['Chaikin Money Flow (20-day)', f"{cmf_value:.4f}"])

        # Interpretation
        if cmf_value > 0.05:
            interpretation = "Strong buying pressure"
        elif cmf_value > 0:
            interpretation = "Mild buying pressure"
        elif cmf_value > -0.05:
            interpretation = "Mild selling pressure"
        else:
            interpretation = "Strong selling pressure"

        rows.append(['CMF Interpretation', interpretation])
    else:
        # Add message when CMF not available
        rows.append(['Chaikin Money Flow', 'Not available (insufficient data)'])
    rows.append(['', ''])  # Blank row

    # Create DataFrame
    df_table = pd.DataFrame(rows, columns=['Metric', 'Value'])

    logger.info(f"Created statistics table for {stats['ticker']}")
    return df_table