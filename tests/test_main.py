"""Test suite for main.py module"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from main import (
    get_user_input,
    display_multiple_charts
)


@pytest.fixture
def sample_stock_data():
    """Create sample data - 100 days"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
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


class TestGetUserInput:
    @patch('builtins.input', return_value='AAPL')
    def test_get_user_input_with_value(self, mock_input):
        result = get_user_input("Enter ticker")
        assert result == 'AAPL'

    @patch('builtins.input', return_value='')
    def test_get_user_input_with_default(self, mock_input):
        result = get_user_input("Enter ticker", default='MSFT')
        assert result == 'MSFT'

    @patch('builtins.input', return_value='  GOOGL  ')
    def test_get_user_input_strips_whitespace(self, mock_input):
        result = get_user_input("Enter ticker")
        assert result == 'GOOGL'


class TestDisplayMultipleCharts:
    @patch('main.plt.show')
    @patch('main.plot_returns_distribution')
    @patch('main.plot_volatility')
    @patch('main.plot_volume_chart')
    @patch('main.plot_price_chart')
    def test_display_multiple_charts_creates_all_charts(
        self,
        mock_price,
        mock_volume,
        mock_volatility,
        mock_returns,
        mock_show,
        sample_stock_data
    ):
        """Test that all four chart functions are called"""
        mock_fig = MagicMock()
        mock_price.return_value = mock_fig
        mock_volume.return_value = mock_fig
        mock_volatility.return_value = mock_fig
        mock_returns.return_value = mock_fig

        display_multiple_charts(sample_stock_data, "AAPL")

        mock_price.assert_called_once()
        mock_volume.assert_called_once()
        mock_volatility.assert_called_once()
        mock_returns.assert_called_once()


class TestRunInteractiveMode:
    @patch('main.display_multiple_charts')
    @patch('main.create_statistics_table')
    @patch('main.generate_summary_statistics')
    @patch('main.fetch_stock_data')
    @patch('main.get_user_input')
    @patch('builtins.print')
    def test_run_interactive_mode_basic_flow(
        self,
        mock_print,
        mock_input,
        mock_fetch,
        mock_stats,
        mock_table,
        mock_charts,
        sample_stock_data
    ):
        """Test the complete interactive workflow"""
        # Mock user inputs: ticker, period, interval, save report
        mock_input.side_effect = ['AAPL', '1mo', '1d', 'n']

        # Mock fetch_stock_data return
        metadata = {
            'ticker': 'AAPL',
            'rows': 100,
            'date_range': ('2024-01-01', '2024-12-31'),
            'from_cache': False,
            'cached_time': None
        }
        mock_fetch.return_value = (sample_stock_data, metadata)

        # Mock stats and table
        stats = {'ticker': 'AAPL', 'current_price': 150.0}
        mock_stats.return_value = stats
        table = pd.DataFrame({'Metric': ['Price'], 'Value': ['$150.00']})
        mock_table.return_value = table

        from main import run_interactive_mode
        run_interactive_mode()

        mock_fetch.assert_called_once_with('AAPL', '1mo', '1d')


class TestMainFunction:
    @patch('main.run_interactive_mode')
    def test_main_calls_interactive_mode(self, mock_interactive):
        from main import main
        main()
        mock_interactive.assert_called_once()

    @patch('main.run_interactive_mode')
    @patch('sys.exit')
    def test_main_handles_keyboard_interrupt(self, mock_exit, mock_interactive):
        """Test that Ctrl+C is handled"""
        mock_interactive.side_effect = KeyboardInterrupt()
        from main import main
        main()
        mock_exit.assert_called_once_with(0)


class TestLogging:
    def test_logger_configured(self):
        from main import logger
        assert logger is not None
        assert logger.name == 'main'