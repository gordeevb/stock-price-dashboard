"""Integration Test Suite for Stock Analysis Dashboard"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import os
from unittest.mock import patch

import config
from fetch_data import fetch_stock_data, save_to_cache, get_cached_data
from analyze import generate_summary_statistics, create_statistics_table
from plot import (
    plot_price_chart,
    plot_volume_chart,
    plot_volatility,
    plot_returns_distribution
)


class TestEndToEndWorkflow:
    """Test complete workflows from fetch through analysis to plotting"""

    @pytest.fixture
    def mock_stock_data(self):
        """Create realistic mock data - 1 year"""
        dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
        np.random.seed(42)
        n = len(dates)
        base_price = 100
        prices = base_price + np.cumsum(np.random.randn(n) * 2)

        return pd.DataFrame({
            'Open': prices + np.random.randn(n) * 0.5,
            'High': prices + np.abs(np.random.randn(n) * 1.5),
            'Low': prices - np.abs(np.random.randn(n) * 1.5),
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, n)
        }, index=dates)

    @patch('fetch_data.download_stock_data')
    def test_complete_analysis_workflow(self, mock_download, mock_stock_data):
        """Test fetch → analyze → plot workflow"""
        mock_download.return_value = mock_stock_data

        # Fetch data
        data, metadata = fetch_stock_data('AAPL', '1y', '1d', use_cache=False)
        assert data is not None
        assert not data.empty
        assert metadata['ticker'] == 'AAPL'

        # Generate statistics
        stats = generate_summary_statistics(data, ticker='AAPL')
        assert stats is not None
        assert 'current_price' in stats

        # Create table
        table = create_statistics_table(stats)
        assert isinstance(table, pd.DataFrame)

        # Create all charts
        fig1 = plot_price_chart(data, ticker='AAPL', show_ma=True)
        fig2 = plot_volume_chart(data, ticker='AAPL')
        fig3 = plot_volatility(data, ticker='AAPL', window=30)
        fig4 = plot_returns_distribution(data, ticker='AAPL')

        assert all([fig1, fig2, fig3, fig4])

        import matplotlib.pyplot as plt
        plt.close('all')

    @patch('fetch_data.download_stock_data')
    def test_workflow_with_caching(self, mock_download, mock_stock_data):
        """Test that caching works correctly"""
        mock_download.return_value = mock_stock_data

        # First fetch - not from cache
        data1, metadata1 = fetch_stock_data('MSFT', '1mo', '1d', use_cache=True)
        assert not metadata1['from_cache']

        # Second fetch - from cache
        data2, metadata2 = fetch_stock_data('MSFT', '1mo', '1d', use_cache=True)
        assert metadata2['from_cache']

        pd.testing.assert_frame_equal(data1, data2)


class TestModuleIntegration:
    """Test integration between module pairs"""

    def test_fetch_to_analyze_integration(self):
        """Test fetched data works with analysis"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Open': np.random.randn(100) + 100,
            'High': np.random.randn(100) + 101,
            'Low': np.random.randn(100) + 99,
            'Close': np.random.randn(100) + 100,
            'Volume': np.random.randint(1000000, 2000000, 100)
        }, index=dates)

        stats = generate_summary_statistics(data, ticker='TEST')
        assert stats['ticker'] == 'TEST'
        assert stats['data_points'] == 100

    def test_analyze_to_plot_integration(self):
        """Test data works with both analysis and plotting"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Open': np.random.randn(100) + 100,
            'High': np.random.randn(100) + 101,
            'Low': np.random.randn(100) + 99,
            'Close': np.random.randn(100) + 100,
            'Volume': np.random.randint(1000000, 2000000, 100)
        }, index=dates)

        # Generate statistics
        stats = generate_summary_statistics(data)

        # Create all charts
        fig1 = plot_price_chart(data)
        fig2 = plot_volume_chart(data)
        fig3 = plot_volatility(data)
        fig4 = plot_returns_distribution(data)

        assert all([fig1, fig2, fig3, fig4])

        import matplotlib.pyplot as plt
        plt.close('all')

    def test_config_integration(self):
        """Test config values are consistent"""
        assert config.DEFAULT_PERIOD in config.VALID_PERIODS
        assert config.DEFAULT_INTERVAL in config.VALID_INTERVALS


class TestErrorPropagation:
    """Test error handling across modules"""

    @patch('fetch_data.download_stock_data')
    def test_fetch_error_stops_workflow(self, mock_download):
        mock_download.side_effect = Exception("Network error")
        with pytest.raises(Exception):
            fetch_stock_data('INVALID', '1mo', '1d', use_cache=False)

    def test_analysis_error_with_bad_data(self):
        bad_data = pd.DataFrame()
        with pytest.raises(Exception):
            generate_summary_statistics(bad_data)

    def test_plot_error_with_missing_columns(self):
        bad_data = pd.DataFrame({'SomeColumn': [1, 2, 3]})
        with pytest.raises(Exception):
            plot_price_chart(bad_data)


class TestDataConsistency:
    """Test data format consistency across modules"""

    def test_data_format_consistency(self):
        """Test same data works in all modules"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'Open': np.random.randn(50) + 100,
            'High': np.random.randn(50) + 101,
            'Low': np.random.randn(50) + 99,
            'Close': np.random.randn(50) + 100,
            'Volume': np.random.randint(1000000, 2000000, 50)
        }, index=dates)

        # Use in analysis
        stats = generate_summary_statistics(data)

        # Use in plotting
        fig = plot_price_chart(data)
        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close(fig)


class TestCachingIntegration:
    """Test cache operations with other modules"""

    def setup_method(self):
        """Create temp cache directory"""
        self.temp_cache = tempfile.mkdtemp()
        self.original_cache = config.CACHE_DIR
        import fetch_data
        fetch_data.CACHE_DIR = Path(self.temp_cache)

    def teardown_method(self):
        """Clean up temp cache"""
        if os.path.exists(self.temp_cache):
            shutil.rmtree(self.temp_cache)
        import fetch_data
        fetch_data.CACHE_DIR = self.original_cache

    def test_cache_save_and_load(self):
        """Test cache save/load cycle"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'Open': np.random.randn(50) + 100,
            'High': np.random.randn(50) + 101,
            'Low': np.random.randn(50) + 99,
            'Close': np.random.randn(50) + 100,
            'Volume': np.random.randint(1000000, 2000000, 50)
        }, index=dates)

        # Save and load
        save_to_cache('TEST', '1mo', '1d', data)
        cached_data = get_cached_data('TEST', '1mo', '1d')

        assert cached_data is not None
        pd.testing.assert_frame_equal(data, cached_data)