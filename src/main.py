"""
Module: main.py
Description: Main entry point for the Stock Price Dashboard.
             Fetches data, generates analysis, and displays 4 charts.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Import other modules from repo
try:
    from fetch_data import fetch_stock_data
    from analyze import generate_summary_statistics, create_statistics_table
    from plot import (
        plot_price_chart,
        plot_volume_chart,
        plot_volatility,
        plot_returns_distribution
    )
    import config
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure fetch_data.py, analyze.py, plot.py, and config.py are in the same directory")
    sys.exit(1)

import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_user_input(prompt: str, default: str = None) -> str:
    """Get user input with optional default value."""
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "

    value = input(prompt).strip()
    return value if value else default


def run_interactive_mode():
    """Run the application - directly analyze a stock and display charts."""
    print("==============================================================================")
    print("STOCK ANALYSIS DASHBOARD")
    print("==============================================================================")

    # Get user input for stock analysis
    ticker = get_user_input("Enter ticker symbol").upper()
    period = get_user_input("Enter period (1d, 1mo, 1y, max)")
    interval = get_user_input("Enter interval (1m, 1h, 1d, 1wk)")

    try:
        # Fetch data
        print(f"\nFetching data for {ticker}...")
        data, metadata = fetch_stock_data(ticker, period, interval)

        print(f"  Fetched {metadata['rows']} rows")
        print(f"  Date range: {metadata['date_range'][0]} to {metadata['date_range'][1]}")

        if metadata['from_cache']:
            print(f"  Using cached data from {metadata['cached_time']}")

        # Generate analysis
        print("\nGenerating analysis...")
        stats = generate_summary_statistics(data, ticker)

        # Create and display Pandas table
        table = create_statistics_table(stats)

        print("\n" + "==============================================================================")
        print(f"ANALYSIS REPORT - {ticker}")
        print("==============================================================================")
        print(table.to_string(index=False))
        print("==============================================================================")

        # Ask if user wants to save to file
        save_report = get_user_input("\nSave report to file? (y/n)").lower()

        if save_report == 'y':
            output_dir = Path('outputs')
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save as CSV
            csv_file = output_dir / f"{ticker}_analysis_{timestamp}.csv"
            table.to_csv(csv_file, index=False)
            print(f"  Saved to: {csv_file}")


        # Automatically create and display multiple charts
        print("\nCreating charts...")
        display_multiple_charts(data, ticker)

        print("\n" + "==============================================================================")
        print("Analysis complete. Close chart windows to exit.")
        print("==============================================================================")

    except Exception as e:
        print(f"\n Error: {str(e)}")
        logger.error(f"Error in interactive mode: {str(e)}")


def display_multiple_charts(data: pd.DataFrame, ticker: str):
    """
    Create and display multiple charts

    Args:
        data (pd.DataFrame): Stock data
        ticker (str): Stock ticker symbol
    """
    print("  Creating individual charts...")

    # Chart 1: Price with SMA
    print("    1/4 Price chart with moving averages...")
    fig1 = plot_price_chart(data, ticker, show_ma=True)

    # Chart 2: Volume
    print("    2/4 Volume chart...")
    fig2 = plot_volume_chart(data, ticker)

    # Chart 3: Volatility
    print("    3/4 Volatility chart...")
    fig3 = plot_volatility(data, ticker, window=30)

    # Chart 4: Returns Distribution
    print("    4/4 Returns distribution...")
    fig4 = plot_returns_distribution(data, ticker)

    print("    All charts created!")
    print("\n  Displaying charts in separate windows...")
    print("  (Close windows to exit)")

    # Display all charts
    plt.show()


def main():
    """Main entry point - runs interactive stock analysis."""
    try:
        run_interactive_mode()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n  Unexpected error: {str(e)}")
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()