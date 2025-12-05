"""Test suite for analyze.py module"""

import pytest
import pandas as pd
import numpy as np

from analyze import (
    validate_dataframe,
    calculate_returns,
    calculate_cumulative_return,
    calculate_volatility,
    calculate_moving_average,
    get_price_change,
    calculate_volume_metrics,
    calculate_chaikin_money_flow,
    generate_summary_statistics,
    create_statistics_table,
    AnalysisError
)


@pytest.fixture
def sample_stock_data():
    """Create sample data for testing - 1 year of daily data"""
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


@pytest.fixture
def minimal_stock_data():
    """Create minimal data for edge case testing - 5 days"""
    dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
    return pd.DataFrame({
        'Open': [100, 101, 99, 102, 103],
        'High': [102, 103, 101, 104, 105],
        'Low': [99, 100, 98, 101, 102],
        'Close': [101, 102, 100, 103, 104],
        'Volume': [1000000, 1100000, 900000, 1200000, 1300000]
    }, index=dates)


class TestValidateDataFrame:
    def test_validate_valid_dataframe(self, sample_stock_data):
        is_valid, error_msg = validate_dataframe(sample_stock_data, ['Close', 'Volume'])
        assert is_valid is True

    def test_validate_none(self):
        is_valid, error_msg = validate_dataframe(None, ['Close'])
        assert is_valid is False


class TestCalculateReturns:
    def test_calculate_returns_basic(self, minimal_stock_data):
        returns = calculate_returns(minimal_stock_data)
        assert len(returns) == len(minimal_stock_data) - 1
        # Verify first return: (102 - 101) / 101 * 100
        expected_first = ((102 - 101) / 101) * 100
        assert np.isclose(returns.iloc[0], expected_first, rtol=1e-5)

    def test_calculate_returns_no_nans(self, sample_stock_data):
        returns = calculate_returns(sample_stock_data)
        assert not returns.isna().any()

    def test_calculate_returns_missing_column(self, minimal_stock_data):
        with pytest.raises(AnalysisError):
            calculate_returns(minimal_stock_data, column='NonExistent')


class TestCalculateCumulativeReturn:
    def test_cumulative_return_positive(self):
        df = pd.DataFrame({'Close': [100, 110]})
        cum_return = calculate_cumulative_return(df)
        assert np.isclose(cum_return, 10.0, rtol=1e-5)

    def test_cumulative_return_negative(self):
        df = pd.DataFrame({'Close': [100, 90]})
        cum_return = calculate_cumulative_return(df)
        assert np.isclose(cum_return, -10.0, rtol=1e-5)


class TestCalculateVolatility:
    def test_volatility_basic(self, sample_stock_data):
        vol = calculate_volatility(sample_stock_data, window=30, annualize=True)
        assert vol > 0
        assert isinstance(vol, float)


class TestCalculateMovingAverage:
    def test_moving_average_basic(self, minimal_stock_data):
        ma = calculate_moving_average(minimal_stock_data, window=3)
        assert len(ma) == len(minimal_stock_data)
        # Verify 3rd MA: average of [101, 102, 100]
        expected = (101 + 102 + 100) / 3
        assert np.isclose(ma.iloc[2], expected, rtol=1e-5)


class TestGetPriceChange:
    def test_price_change_1d(self, minimal_stock_data):
        change = get_price_change(minimal_stock_data, timeframe='1d')
        assert 'absolute_change' in change
        assert 'percentage_change' in change


class TestCalculateVolumeMetrics:
    def test_volume_metrics_basic(self, sample_stock_data):
        metrics = calculate_volume_metrics(sample_stock_data)
        assert 'current_volume' in metrics
        assert 'average_volume' in metrics


class TestCalculateChaikinMoneyFlow:
    def test_cmf_basic(self, sample_stock_data):
        cmf = calculate_chaikin_money_flow(sample_stock_data, window=20)
        assert isinstance(cmf, float)
        assert -1 <= cmf <= 1


class TestGenerateSummaryStatistics:
    def test_summary_stats_basic(self, sample_stock_data):
        stats = generate_summary_statistics(sample_stock_data, ticker="AAPL")
        # Check all required keys exist
        required_keys = [
            'ticker', 'data_points', 'start_date', 'end_date',
            'current_price', 'period_high', 'period_low',
            'avg_daily_return', 'cumulative_return', 'volatility_30d'
        ]
        for key in required_keys:
            assert key in stats


class TestCreateStatisticsTable:
    def test_create_table_basic(self, sample_stock_data):
        stats = generate_summary_statistics(sample_stock_data, ticker="AAPL")
        table = create_statistics_table(stats)
        assert isinstance(table, pd.DataFrame)
        assert 'Metric' in table.columns
        assert 'Value' in table.columns