"""
Streamlit Web Interface for Stock Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import required modules
try:
    from fetch_data import fetch_stock_data, VALID_PERIODS, VALID_INTERVALS
    from analyze import generate_summary_statistics, create_statistics_table
    from plot import (
        plot_price_chart,
        plot_volume_chart,
        plot_volatility,
        plot_returns_distribution
    )
    import config
except ImportError:
    class config:
        DEFAULT_MA_WINDOWS = [20, 50, 200]

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .stMetric label {
        color: #000000 !important;
        font-weight: 600;
    }
    .stMetric .metric-value {
        color: #000000 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }
    h1 {
        color: #2E86AB;
    }
    h2 {
        color: #A23B72;
    }
    h3 {
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'metadata' not in st.session_state:
        st.session_state.metadata = None
    if 'stats' not in st.session_state:
        st.session_state.stats = None
    if 'last_ticker' not in st.session_state:
        st.session_state.last_ticker = ""


def display_header():
    """Display application header"""
    st.title("Stock Analysis Dashboard")
    st.markdown("---")


def display_sidebar():
    """Display sidebar with input controls"""
    st.sidebar.header("Settings")

    # Ticker input
    ticker = st.sidebar.text_input(
        "Stock Ticker Symbol",
        value="AAPL",
        help="Enter a valid stock ticker symbol"
    ).upper()

    # Period selection
    period = st.sidebar.selectbox(
        "Time Period",
        options=VALID_PERIODS,
        index=VALID_PERIODS.index('1mo'),
        help="Select the time period for historical data"
    )

    # Interval selection
    if period in ['1d', '5d']:
        available_intervals = VALID_INTERVALS
    else:
        available_intervals = [i for i in VALID_INTERVALS
                             if i not in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']]

    interval = st.sidebar.selectbox(
        "Data Interval",
        options=available_intervals,
        index=available_intervals.index('1d') if '1d' in available_intervals else 0,
        help="Select the data interval/granularity"
    )

    # Cache option
    use_cache = st.sidebar.checkbox(
        "Use Cached Data",
        value=True,
        help="Use cached data if available to speed up loading"
    )

    # Analysis options
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Options")

    show_ma = st.sidebar.checkbox("Show Moving Averages", value=True)
    ma_periods = st.sidebar.multiselect(
        "MA Periods",
        options=[5, 10, 20, 50, 100, 200],
        default=[20, 50, 200]
    )

    volatility_window = st.sidebar.slider(
        "Volatility Window (days)",
        min_value=7,
        max_value=90,
        value=30,
        step=1
    )

    # Fetch button
    st.sidebar.markdown("---")
    fetch_button = st.sidebar.button("Fetch Data", type="primary", use_container_width=True)

    return ticker, period, interval, use_cache, show_ma, ma_periods, volatility_window, fetch_button


def fetch_data(ticker, period, interval, use_cache):
    """Fetch stock data and handle errors"""
    try:
        with st.spinner(f"Fetching data for {ticker}..."):
            data, metadata = fetch_stock_data(ticker, period, interval, use_cache)
            return data, metadata, None
    except Exception as e:
        return None, None, str(e)


def display_metadata(metadata):
    """Display metadata information"""
    if metadata is None:
        return

    cols = st.columns(4)

    with cols[0]:
        st.metric("Ticker", metadata['ticker'])

    with cols[1]:
        st.metric("Data Points", f"{metadata['rows']:,}")

    with cols[2]:
        st.metric("Period", metadata['period'].upper())

    with cols[3]:
        st.metric("Interval", metadata['interval'].upper())

    # Show cache info
    if metadata['from_cache']:
        st.info(f"Using cached data from {metadata['cached_time']}")

    # Show date range
    st.caption(f"**Date Range:** {metadata['date_range'][0]} to {metadata['date_range'][1]}")


def display_key_metrics(stats):
    """Display key metrics in a prominent way"""
    if stats is None:
        return

    st.subheader("Key Metrics")

    cols = st.columns(5)

    with cols[0]:
        st.metric(
            "Current Price",
            f"${stats['current_price']:.2f}",
            delta=f"{stats['price_changes'].get('1d', {}).get('percentage', 0):.2f}%"
                  if '1d' in stats.get('price_changes', {}) else None
        )

    with cols[1]:
        if stats['cumulative_return'] is not None:
            st.metric(
                "Cumulative Return",
                f"{stats['cumulative_return']:.2f}%"
            )

    with cols[2]:
        if stats['volatility_30d'] is not None:
            st.metric(
                "Volatility (30d)",
                f"{stats['volatility_30d']:.2f}%"
            )

    with cols[3]:
        st.metric(
            "Period High",
            f"${stats['period_high']:.2f}"
        )

    with cols[4]:
        st.metric(
            "Period Low",
            f"${stats['period_low']:.2f}"
        )


def display_charts(data, ticker, show_ma, ma_periods, volatility_window):
    """Display all charts in tabs"""
    if data is None:
        return

    st.subheader("Interactive Charts")

    # Create tabs for different charts
    tab1, tab2, tab3, tab4 = st.tabs([
        "Price Chart",
        "Volume",
        "Volatility",
        "Returns Distribution"
    ])

    with tab1:
        try:
            fig = plot_price_chart(
                data,
                ticker=ticker,
                show_ma=show_ma,
                ma_windows=ma_periods if show_ma else []
            )
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error creating price chart: {str(e)}")

    with tab2:
        try:
            fig = plot_volume_chart(data, ticker=ticker)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error creating volume chart: {str(e)}")

    with tab3:
        try:
            fig = plot_volatility(data, ticker=ticker, window=volatility_window)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error creating volatility chart: {str(e)}")

    with tab4:
        try:
            fig = plot_returns_distribution(data, ticker=ticker)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error creating returns distribution: {str(e)}")


def main():
    """Main application function"""
    # Initialize session state
    init_session_state()

    # Display header
    display_header()

    # Display sidebar and get inputs
    ticker, period, interval, use_cache, show_ma, ma_periods, volatility_window, fetch_button = display_sidebar()

    # Fetch data when button is clicked or ticker changes
    if fetch_button or (ticker != st.session_state.last_ticker and ticker):
        data, metadata, error = fetch_data(ticker, period, interval, use_cache)

        if error:
            st.error(f"Error: {error}")
            st.session_state.data = None
            st.session_state.metadata = None
            st.session_state.stats = None
        else:
            st.session_state.data = data
            st.session_state.metadata = metadata
            st.session_state.last_ticker = ticker

            # Generate statistics
            try:
                with st.spinner("Analyzing data..."):
                    st.session_state.stats = generate_summary_statistics(
                        data,
                        ticker=ticker,
                        ma_windows=ma_periods if show_ma else None
                    )
            except Exception as e:
                st.error(f"Error generating statistics: {str(e)}")
                st.session_state.stats = None

    # Display results if data is available
    if st.session_state.data is not None:
        # Display metadata
        display_metadata(st.session_state.metadata)

        st.markdown("---")

        # Display key metrics
        display_key_metrics(st.session_state.stats)

        st.markdown("---")

        # Display charts
        display_charts(
            st.session_state.data,
            ticker,
            show_ma,
            ma_periods,
            volatility_window
        )

    else:
        # Show welcome message when no data is loaded
        st.info("Enter a stock ticker and click 'Fetch Data' to begin analysis")

        # Show popular tickers
        st.subheader("Popular Tickers")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**Tech:**")
            st.markdown("AAPL\n\nMSFT\n\nGOOGL\n\nAMZN\n\nMETA")

        with col2:
            st.markdown("**Finance:**")
            st.markdown("JPM\n\nBAC\n\nGS\n\nV\n\nMA")

        with col3:
            st.markdown("**Healthcare:**")
            st.markdown("JNJ\n\nUNH\n\nPFE\n\nABBV\n\nTMO")

        with col4:
            st.markdown("**Energy:**")
            st.markdown("XOM\n\nCVX\n\nCOP\n\nSLB\n\nEOG")


if __name__ == "__main__":
    main()