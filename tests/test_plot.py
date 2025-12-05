"""Test suite for plot.py module"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from plot import (
    validate_plot_data,
    format_large_number,
    plot_price_chart,
    plot_volume_chart,
    plot_volatility,
    plot_returns_distribution,
    PlotError,
    COLOR_PALETTE
)


@pytest.fixture
def sample_stock_data():
    """Create sample data for plotting - 1 year of daily data"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    n = len(dates)
    prices = 100 + np.cumsum(np.random.randn(n) * 2)

    return pd.DataFrame({
        'Open': prices + np.random.randn(n) * 0.5,
        'High': prices + np.abs(np.random.randn(n) * 1.5),
        'Low': prices - np.abs(np.random.randn(n) * 1.5),
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, n)
    }, index=dates)


class TestValidatePlotData:
    def test_validate_valid_data(self, sample_stock_data):
        is_valid, error_msg = validate_plot_data(sample_stock_data, ['Close', 'Volume'])
        assert is_valid is True

    def test_validate_none(self):
        is_valid, error_msg = validate_plot_data(None, ['Close'])
        assert is_valid is False


class TestFormatLargeNumber:
    def test_format_billions(self):
        assert format_large_number(2_500_000_000) == "2.5B"

    def test_format_millions(self):
        assert format_large_number(1_500_000) == "1.5M"

    def test_format_thousands(self):
        assert format_large_number(1_500) == "1.5K"

    def test_format_small_numbers(self):
        assert format_large_number(500) == "500"


class TestPlotPriceChart:
    def test_create_price_chart_basic(self, sample_stock_data):
        fig = plot_price_chart(sample_stock_data, ticker="AAPL", show_ma=False)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_create_price_chart_with_ma(self, sample_stock_data):
        fig = plot_price_chart(sample_stock_data, ticker="AAPL", show_ma=True, ma_windows=[20, 50])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_create_price_chart_missing_data(self):
        df = pd.DataFrame({'Open': [100, 101]})
        with pytest.raises(PlotError):
            plot_price_chart(df)


class TestPlotVolumeChart:
    def test_create_volume_chart_basic(self, sample_stock_data):
        fig = plot_volume_chart(sample_stock_data, ticker="AAPL")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_create_volume_chart_missing_data(self):
        df = pd.DataFrame({'Close': [100, 101]})
        with pytest.raises(PlotError):
            plot_volume_chart(df)


class TestPlotVolatility:
    def test_create_volatility_chart_basic(self, sample_stock_data):
        fig = plot_volatility(sample_stock_data, ticker="AAPL", window=30)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_create_volatility_chart_missing_data(self):
        df = pd.DataFrame({'Volume': [100, 101]})
        with pytest.raises(PlotError):
            plot_volatility(df)


class TestPlotReturnsDistribution:
    def test_create_returns_distribution_basic(self, sample_stock_data):
        fig = plot_returns_distribution(sample_stock_data, ticker="AAPL")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_create_returns_distribution_missing_data(self):
        df = pd.DataFrame({'Volume': [100, 101]})
        with pytest.raises(PlotError):
            plot_returns_distribution(df)


class TestPlotError:
    def test_plot_error_raised(self):
        with pytest.raises(PlotError):
            raise PlotError("Test error")


class TestColorPalette:
    def test_color_palette_exists(self):
        assert isinstance(COLOR_PALETTE, dict)
        assert len(COLOR_PALETTE) > 0


class TestChartIntegration:
    def test_create_all_charts(self, sample_stock_data):
        """Test creating all chart types together"""
        charts = []
        charts.append(plot_price_chart(sample_stock_data, "AAPL"))
        charts.append(plot_volume_chart(sample_stock_data, "AAPL"))
        charts.append(plot_volatility(sample_stock_data, "AAPL"))
        charts.append(plot_returns_distribution(sample_stock_data, "AAPL"))

        for chart in charts:
            assert isinstance(chart, Figure)
            plt.close(chart)