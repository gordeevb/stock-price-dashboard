"""
Streamlit Web Interface for Stock Analysis Dashboard

Fixed: White backgrounds for metrics and proper NaN/volatility handling
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src directory to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

from fetch_data import fetch_stock_data, VALID_PERIODS, VALID_INTERVALS
from analyze import generate_summary_statistics
from plot import plot_price_chart, plot_volume_chart, plot_volatility, plot_returns_distribution

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - White backgrounds and proper styling
st.markdown("""
    <style>
    .stMetric {
        background-color: #ffffff !important;
        padding: 15px !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        border: 1px solid #e0e0e0 !important;
    }
    .stMetric label {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    .sentiment-positive {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'metadata' not in st.session_state:
        st.session_state.metadata = None
    if 'stats' not in st.session_state:
        st.session_state.stats = None
    if 'last_ticker' not in st.session_state:
        st.session_state.last_ticker = ""


def format_value(value, value_type='currency'):
    """Format values, handling None/NaN"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    if value_type == 'currency':
        return f"${value:.2f}"
    elif value_type == 'percentage':
        return f"{value:.2f}%"
    return str(value)


def get_sentiment_from_cmf(cmf_value):
    if cmf_value is None or pd.isna(cmf_value):
        return "Unknown", "neutral"
    if cmf_value > 0.15:
        return "Strong Buy", "positive"
    elif cmf_value > 0.05:
        return "Buy", "positive"
    elif cmf_value > -0.05:
        return "Neutral", "neutral"
    elif cmf_value > -0.15:
        return "Sell", "negative"
    else:
        return "Strong Sell", "negative"


def display_cmf_and_sentiment(stats):
    if stats is None or 'cmf' not in stats:
        return

    cmf_value = stats['cmf']
    sentiment, sentiment_type = get_sentiment_from_cmf(cmf_value)

    st.subheader("Market Sentiment Analysis")

    col1, col2 = st.columns(2)

    with col1:
        if cmf_value is not None and not pd.isna(cmf_value):
            st.metric("Chaikin Money Flow (20d)", f"{cmf_value:.3f}")
        else:
            st.metric("Chaikin Money Flow (20d)", "N/A")

    with col2:
        st.metric("Market Sentiment", sentiment)

    if cmf_value is not None and not pd.isna(cmf_value):
        if sentiment_type == "positive":
            st.markdown(f"""
            <div class="sentiment-positive">
                <strong>Bullish Signal:</strong> CMF is positive ({cmf_value:.3f}), indicating buying pressure.
            </div>
            """, unsafe_allow_html=True)
        elif sentiment_type == "negative":
            st.markdown(f"""
            <div class="sentiment-negative">
                <strong>Bearish Signal:</strong> CMF is negative ({cmf_value:.3f}), indicating selling pressure.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="sentiment-neutral">
                <strong>Neutral Signal:</strong> CMF is near zero ({cmf_value:.3f}), balanced pressure.
            </div>
            """, unsafe_allow_html=True)


def display_sidebar():
    st.sidebar.header("Settings")

    ticker = st.sidebar.text_input("Stock Ticker Symbol", value="AAPL").upper()
    period = st.sidebar.selectbox("Time Period", options=VALID_PERIODS, index=VALID_PERIODS.index('1mo'))

    if period in ['1d', '5d']:
        available_intervals = VALID_INTERVALS
    else:
        available_intervals = [i for i in VALID_INTERVALS
                             if i not in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']]

    interval = st.sidebar.selectbox("Data Interval", options=available_intervals,
                                   index=available_intervals.index('1d') if '1d' in available_intervals else 0)

    use_cache = st.sidebar.checkbox("Use Cached Data", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Options")

    show_ma = st.sidebar.checkbox("Show Moving Averages", value=True)
    ma_periods = st.sidebar.multiselect("MA Periods", options=[5, 10, 20, 50, 100, 200], default=[20, 50, 200])
    volatility_window = st.sidebar.slider("Volatility Window (days)", min_value=7, max_value=90, value=30)
    cmf_window = st.sidebar.slider("CMF Window (days)", min_value=10, max_value=30, value=20, step=5)

    st.sidebar.markdown("---")
    fetch_button = st.sidebar.button("Fetch Data", type="primary", use_container_width=True)

    return ticker, period, interval, use_cache, show_ma, ma_periods, volatility_window, cmf_window, fetch_button


def display_key_metrics(stats):
    if stats is None:
        return

    st.subheader("Key Metrics")

    cols = st.columns(5)

    with cols[0]:
        price_change = None
        if '1d' in stats.get('price_changes', {}):
            price_change = stats['price_changes']['1d'].get('percentage')

        st.metric("Current Price", format_value(stats['current_price'], 'currency'),
                 delta=f"{price_change:.2f}%" if price_change is not None else None)

    with cols[1]:
        st.metric("Cumulative Return", format_value(stats.get('cumulative_return'), 'percentage'))

    with cols[2]:
        st.metric("Volatility (30d)", format_value(stats.get('volatility_30d'), 'percentage'))

    with cols[3]:
        st.metric("Period High", format_value(stats.get('period_high'), 'currency'))

    with cols[4]:
        st.metric("Period Low", format_value(stats.get('period_low'), 'currency'))


def display_charts(data, ticker, show_ma, ma_periods, volatility_window):
    if data is None:
        return

    st.subheader("Interactive Charts")

    tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Volume", "Volatility", "Returns Distribution"])

    with tab1:
        try:
            fig = plot_price_chart(data, ticker=ticker, show_ma=show_ma,
                                  ma_windows=ma_periods if show_ma else [])
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error: {str(e)}")

    with tab2:
        try:
            fig = plot_volume_chart(data, ticker=ticker)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error: {str(e)}")

    with tab3:
        try:
            adjusted_window = min(volatility_window, max(7, len(data) - 1))
            fig = plot_volatility(data, ticker=ticker, window=adjusted_window)
            if adjusted_window < volatility_window:
                st.info(f"Using {adjusted_window}-day window (insufficient data for {volatility_window}-day)")
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error: {str(e)}")

    with tab4:
        try:
            fig = plot_returns_distribution(data, ticker=ticker)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error: {str(e)}")


def main():
    init_session_state()

    st.title("Stock Analysis Dashboard")
    st.markdown("---")

    ticker, period, interval, use_cache, show_ma, ma_periods, volatility_window, cmf_window, fetch_button = display_sidebar()

    if fetch_button or (ticker != st.session_state.last_ticker and ticker):
        try:
            with st.spinner(f"Fetching data for {ticker}..."):
                data, metadata = fetch_stock_data(ticker, period, interval, use_cache)
                st.session_state.data = data
                st.session_state.metadata = metadata
                st.session_state.last_ticker = ticker

                # Auto-adjust windows based on available data
                data_points = len(data)
                if data_points < 30:
                    st.warning(f"Only {data_points} data points. Some calculations may be limited.")

                st.session_state.stats = generate_summary_statistics(
                    data, ticker=ticker,
                    ma_windows=ma_periods if show_ma else None,
                    cmf_window=min(cmf_window, max(10, data_points - 1))
                )
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.data = None

    if st.session_state.data is not None:
        if st.session_state.metadata:
            cols = st.columns(4)
            with cols[0]:
                st.metric("Ticker", st.session_state.metadata['ticker'])
            with cols[1]:
                st.metric("Data Points", f"{st.session_state.metadata['rows']:,}")
            with cols[2]:
                st.metric("Period", st.session_state.metadata['period'].upper())
            with cols[3]:
                st.metric("Interval", st.session_state.metadata['interval'].upper())

        st.markdown("---")
        display_key_metrics(st.session_state.stats)

        st.markdown("---")
        display_cmf_and_sentiment(st.session_state.stats)

        st.markdown("---")
        display_charts(st.session_state.data, ticker, show_ma, ma_periods, volatility_window)
    else:
        st.info("Enter a stock ticker and click 'Fetch Data' to begin analysis")

        st.subheader("Popular Tickers")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**Tech:**\n\nAAPL\n\nMSFT\n\nGOOGL")
        with col2:
            st.markdown("**Finance:**\n\nJPM\n\nBAC\n\nGS")
        with col3:
            st.markdown("**Healthcare:**\n\nJNJ\n\nUNH\n\nPFE")
        with col4:
            st.markdown("**Energy:**\n\nXOM\n\nCVX\n\nCOP")


if __name__ == "__main__":
    main()