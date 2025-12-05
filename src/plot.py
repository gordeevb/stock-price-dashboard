"""
Module: plot.py
Description: Visualization functions for stock data analysis using Matplotlib and Plotly.
             Creates professional, publication-ready charts with consistent styling.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Visualization configuration
CHART_STYLE = 'seaborn-v0_8-whitegrid'
COLOR_PALETTE = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple
    'accent': '#F18F01',       # Orange
    'danger': '#C73E1D',       # Red
    'success': '#6A994E',      # Green
    'neutral': '#555555'       # Gray
}
FIGURE_SIZE = (14, 10)
DPI = 100
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 9

# Set matplotlib style
try:
    plt.style.use(CHART_STYLE)
except:
    plt.style.use('default')


class PlotError(Exception):
    """Custom exception for plotting errors"""
    pass


def setup_plot_style():
    """Configure matplotlib global settings for styling."""
    plt.rcParams['figure.figsize'] = FIGURE_SIZE
    plt.rcParams['figure.dpi'] = DPI
    plt.rcParams['font.size'] = FONT_SIZE_TICK
    plt.rcParams['axes.titlesize'] = FONT_SIZE_TITLE
    plt.rcParams['axes.labelsize'] = FONT_SIZE_LABEL
    plt.rcParams['xtick.labelsize'] = FONT_SIZE_TICK
    plt.rcParams['ytick.labelsize'] = FONT_SIZE_TICK
    plt.rcParams['legend.fontsize'] = FONT_SIZE_TICK
    plt.rcParams['figure.autolayout'] = True


def validate_plot_data(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, str]:
    """
    Validate that DataFrame has required columns for plotting.

    Args:
        df (pd.DataFrame): Data to validate
        required_columns (List[str]): Required column names

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty or None"

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"

    return True, ""


def format_large_number(num: float) -> str:
    """
    Format large numbers with K, M, B suffixes.

    Args:
        num (float): Number to format

    Returns:
        str: Formatted string
    """
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return f"{num:.0f}"


def plot_price_chart(df: pd.DataFrame, ticker: str = "",
                    show_ma: bool = True, ma_windows: List[int] = None,
                    figsize: Tuple[int, int] = None, save_path: Optional[str] = None) -> Figure:
    """
    Create a price chart with optional moving averages.

    Args:
        df (pd.DataFrame): Stock data with Close prices
        ticker (str): Stock ticker symbol for title
        show_ma (bool): Whether to show moving averages
        ma_windows (List[int]): Moving average periods (default: [20, 50, 200])
        figsize (Tuple[int, int]): Figure size (default: from config)
        save_path (str, optional): Path to save figure

    Returns:
        Figure: Matplotlib figure object
    """
    is_valid, error_msg = validate_plot_data(df, ['Close'])
    if not is_valid:
        raise PlotError(error_msg)

    if ma_windows is None:
        ma_windows = [20, 50, 200]

    if figsize is None:
        figsize = (12, 6)

    try:
        fig, ax = plt.subplots(figsize=figsize)

        # Plot closing price
        ax.plot(df.index, df['Close'].values, linewidth=2,
                label='Close Price', color=COLOR_PALETTE['primary'])

        # Add moving averages if requested
        if show_ma:
            colors = [COLOR_PALETTE['success'], COLOR_PALETTE['accent'], COLOR_PALETTE['secondary']]
            for i, window in enumerate(ma_windows):
                if len(df) >= window:
                    ma = df['Close'].rolling(window=window).mean()
                    # Plot MA values
                    ax.plot(df.index, ma.values, linewidth=1.5, alpha=0.7,
                           label=f'{window}-day MA',
                           color=colors[i % len(colors)],
                           linestyle='--')

        # Formatting
        title = f'{ticker} Stock Price' if ticker else 'Stock Price'
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Price ($)', fontsize=FONT_SIZE_LABEL)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        if isinstance(df.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Saved price chart to {save_path}")

        logger.info("Created price chart")
        return fig

    except Exception as e:
        error_msg = f"Error creating price chart: {str(e)}"
        logger.error(error_msg)
        raise PlotError(error_msg)


def plot_volume_chart(df: pd.DataFrame, ticker: str = "",
                      show_average: bool = True, avg_window: int = 20,
                      figsize: Tuple[int, int] = None, save_path: Optional[str] = None) -> Figure:
    """
    Create a volume bar chart with optional average line.

    Args:
        df (pd.DataFrame): Stock data with Volume column
        ticker (str): Stock ticker symbol
        show_average (bool): Show average volume line
        avg_window (int): Window for average calculation
        figsize (Tuple[int, int]): Figure size
        save_path (str, optional): Path to save figure

    Returns:
        Figure: Matplotlib figure object
    """
    is_valid, error_msg = validate_plot_data(df, ['Volume'])
    if not is_valid:
        raise PlotError(error_msg)

    if figsize is None:
        figsize = (12, 4)

    try:
        fig, ax = plt.subplots(figsize=figsize)

        # Color bars based on price change (if Close is available)
        if 'Close' in df.columns:
            colors = [COLOR_PALETTE['success'] if close >= open_price
                     else COLOR_PALETTE['danger']
                     for close, open_price in zip(df['Close'], df['Open'])]
        else:
            colors = COLOR_PALETTE['primary']

        # Plot volume bars
        ax.bar(df.index, df['Volume'], color=colors, alpha=0.6, width=0.8)

        # Add average line if requested
        if show_average and len(df) >= avg_window:
            avg_volume = df['Volume'].rolling(window=avg_window).mean()
            ax.plot(df.index, avg_volume, color=COLOR_PALETTE['neutral'],
                   linewidth=2, label=f'{avg_window}-day Avg',
                   linestyle='--', alpha=0.8)
            ax.legend(loc='best')

        # Formatting
        title = f'{ticker} Trading Volume' if ticker else 'Trading Volume'
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Volume', fontsize=FONT_SIZE_LABEL)
        ax.grid(True, alpha=0.3, axis='y')

        # Format y-axis for large numbers
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_large_number(x)))

        # Format x-axis dates and ensure datetime handling
        if isinstance(df.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Saved volume chart to {save_path}")

        logger.info("Created volume chart")
        return fig

    except Exception as e:
        error_msg = f"Error creating volume chart: {str(e)}"
        logger.error(error_msg)
        raise PlotError(error_msg)


def plot_volatility(df: pd.DataFrame, ticker: str = "", window: int = 30,
                   figsize: Tuple[int, int] = None, save_path: Optional[str] = None) -> Figure:
    """
    Create rolling volatility chart.

    Args:
        df (pd.DataFrame): Stock data
        ticker (str): Stock ticker symbol
        window (int): Rolling window for volatility calculation
        figsize (Tuple[int, int]): Figure size
        save_path (str, optional): Path to save figure

    Returns:
        Figure: Matplotlib figure object
    """
    is_valid, error_msg = validate_plot_data(df, ['Close'])
    if not is_valid:
        raise PlotError(error_msg)

    if figsize is None:
        figsize = (12, 5)

    try:
        # Adjust window
        if len(df) < window:
            window = max(5, len(df) // 2)  # Use smaller window
            logger.warning(f"Not enough data for {window}-day volatility, using {window}-day instead")

        # Calculate returns and rolling volatility
        returns = df['Close'].pct_change() * 100
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized

        # Remove NaN values
        rolling_vol = rolling_vol.dropna()

        if len(rolling_vol) == 0:
            raise PlotError("Not enough data to calculate volatility")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot volatility
        ax.plot(rolling_vol.index, rolling_vol.values,
               color=COLOR_PALETTE['danger'], linewidth=2)
        ax.fill_between(rolling_vol.index, rolling_vol.values, alpha=0.3,
                        color=COLOR_PALETTE['danger'])

        # Add mean line
        mean_vol = rolling_vol.mean()
        ax.axhline(y=mean_vol, color=COLOR_PALETTE['neutral'],
                  linestyle='--', linewidth=1.5,
                  label=f'Mean: {mean_vol:.2f}%', alpha=0.7)

        # Formatting
        title = f'{ticker} Rolling Volatility ({window}-day)' if ticker else f'Rolling Volatility ({window}-day)'
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Annualized Volatility (%)', fontsize=FONT_SIZE_LABEL)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Format x-axis - use actual datetime index
        if isinstance(rolling_vol.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Saved volatility chart to {save_path}")

        logger.info("Created volatility chart")
        return fig

    except Exception as e:
        error_msg = f"Error creating volatility chart: {str(e)}"
        logger.error(error_msg)
        raise PlotError(error_msg)


def plot_returns_distribution(df: pd.DataFrame, ticker: str = "",
                              bins: int = 50, figsize: Tuple[int, int] = None,
                              save_path: Optional[str] = None) -> Figure:
    """
    Create histogram of daily returns distribution.

    Args:
        df (pd.DataFrame): Stock data
        ticker (str): Stock ticker symbol
        bins (int): Number of histogram bins
        figsize (Tuple[int, int]): Figure size
        save_path (str, optional): Path to save figure

    Returns:
        Figure: Matplotlib figure object
    """
    is_valid, error_msg = validate_plot_data(df, ['Close'])
    if not is_valid:
        raise PlotError(error_msg)

    if figsize is None:
        figsize = (10, 6)

    try:
        # Calculate returns
        returns = df['Close'].pct_change() * 100
        returns = returns.dropna()

        fig, ax = plt.subplots(figsize=figsize)

        # Create histogram
        n, bins_edges, patches = ax.hist(returns, bins=bins,
                                         color=COLOR_PALETTE['primary'],
                                         alpha=0.7, edgecolor='black', linewidth=0.5)

        # Color bars: green for positive, red for negative
        for i, patch in enumerate(patches):
            if bins_edges[i] < 0:
                patch.set_facecolor(COLOR_PALETTE['danger'])
            else:
                patch.set_facecolor(COLOR_PALETTE['success'])

        # Formatting
        title = f'{ticker} Daily Returns Distribution' if ticker else 'Daily Returns Distribution'
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=20)
        ax.set_xlabel('Daily Return (%)', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Frequency', fontsize=FONT_SIZE_LABEL)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Saved returns distribution to {save_path}")

        logger.info("Created returns distribution chart")
        return fig

    except Exception as e:
        error_msg = f"Error creating returns distribution: {str(e)}"
        logger.error(error_msg)
        raise PlotError(error_msg)