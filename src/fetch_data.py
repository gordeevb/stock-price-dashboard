import pandas as pd
import yfinance as yf


def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical stock price data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format

    Returns:
        DataFrame with OHLCV data (Open, High, Low, Close, Adj Close, Volume)

    Raises:
        RuntimeError: If no data is returned from Yahoo Finance
    """
    # Download daily stock data from Yahoo Finance
    # progress=False and threads=False suppress download status messages
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, threads=False)

    # Validate that data was successfully retrieved
    if df is None or df.empty:
        raise RuntimeError("No data returned (bad ticker or date range).")

    # Remove timezone information from the datetime index for consistency
    # This ensures all timestamps are timezone-naive
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # Sort data chronologically by date using an in-place operation
    df.sort_index(inplace=True)

    # Ensure Adj Close column exists
    # If no Adj Close, use Close as fallback
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    # Return only the standard OHLCV columns in a consistent order
    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]


print(fetch("AAPL", "2023-01-01", "2025-10-20"))