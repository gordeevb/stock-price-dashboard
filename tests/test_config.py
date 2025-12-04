"""Test suite for config.py module"""

import pytest
from pathlib import Path

import config


class TestPathConfiguration:
    def test_base_dir_exists(self):
        assert hasattr(config, 'BASE_DIR')
        assert isinstance(config.BASE_DIR, Path)
    
    def test_cache_dir_created(self):
        assert hasattr(config, 'CACHE_DIR')
        assert config.CACHE_DIR.exists()
        assert config.CACHE_DIR.is_dir()
    
    def test_log_dir_created(self):
        assert hasattr(config, 'LOG_DIR')
        assert config.LOG_DIR.exists()
        assert config.LOG_DIR.is_dir()
    
    def test_output_dir_created(self):
        assert hasattr(config, 'OUTPUT_DIR')
        assert config.OUTPUT_DIR.exists()
        assert config.OUTPUT_DIR.is_dir()


class TestDataFetchingConfiguration:
    def test_retry_attempts(self):
        assert hasattr(config, 'RETRY_ATTEMPTS')
        assert isinstance(config.RETRY_ATTEMPTS, int)
        assert config.RETRY_ATTEMPTS > 0
    
    def test_retry_delay(self):
        assert hasattr(config, 'RETRY_DELAY')
        assert isinstance(config.RETRY_DELAY, int)
        assert config.RETRY_DELAY > 0
    
    def test_cache_expiry_intraday(self):
        assert hasattr(config, 'CACHE_EXPIRY_MINUTES_INTRADAY')
        assert isinstance(config.CACHE_EXPIRY_MINUTES_INTRADAY, int)
        assert config.CACHE_EXPIRY_MINUTES_INTRADAY > 0
    
    def test_cache_expiry_daily(self):
        assert hasattr(config, 'CACHE_EXPIRY_HOURS_DAILY')
        assert isinstance(config.CACHE_EXPIRY_HOURS_DAILY, int)
        assert config.CACHE_EXPIRY_HOURS_DAILY > 0


class TestValidParameters:
    def test_valid_periods_list(self):
        assert hasattr(config, 'VALID_PERIODS')
        assert isinstance(config.VALID_PERIODS, list)
        assert len(config.VALID_PERIODS) > 0
        assert '1mo' in config.VALID_PERIODS
    
    def test_valid_intervals_list(self):
        assert hasattr(config, 'VALID_INTERVALS')
        assert isinstance(config.VALID_INTERVALS, list)
        assert len(config.VALID_INTERVALS) > 0
        assert '1d' in config.VALID_INTERVALS


class TestDefaultValues:
    def test_default_period(self):
        assert hasattr(config, 'DEFAULT_PERIOD')
        assert config.DEFAULT_PERIOD in config.VALID_PERIODS
    
    def test_default_interval(self):
        assert hasattr(config, 'DEFAULT_INTERVAL')
        assert config.DEFAULT_INTERVAL in config.VALID_INTERVALS
    
    def test_default_ticker(self):
        assert hasattr(config, 'DEFAULT_TICKER')
        assert isinstance(config.DEFAULT_TICKER, str)
        assert len(config.DEFAULT_TICKER) > 0


class TestAnalysisConfiguration:
    def test_default_ma_windows(self):
        assert hasattr(config, 'DEFAULT_MA_WINDOWS')
        assert isinstance(config.DEFAULT_MA_WINDOWS, list)
        assert len(config.DEFAULT_MA_WINDOWS) > 0
        
        for window in config.DEFAULT_MA_WINDOWS:
            assert isinstance(window, int)
            assert window > 0
    
    def test_default_volatility_window(self):
        assert hasattr(config, 'DEFAULT_VOLATILITY_WINDOW')
        assert isinstance(config.DEFAULT_VOLATILITY_WINDOW, int)
        assert config.DEFAULT_VOLATILITY_WINDOW > 0
    
    def test_trading_days_per_year(self):
        assert hasattr(config, 'TRADING_DAYS_PER_YEAR')
        assert isinstance(config.TRADING_DAYS_PER_YEAR, int)
        assert config.TRADING_DAYS_PER_YEAR == 252
    
    def test_risk_free_rate(self):
        assert hasattr(config, 'RISK_FREE_RATE')
        assert isinstance(config.RISK_FREE_RATE, (int, float))
        assert 0 <= config.RISK_FREE_RATE <= 1


class TestVisualizationConfiguration:
    def test_chart_style(self):
        assert hasattr(config, 'CHART_STYLE')
        assert isinstance(config.CHART_STYLE, str)
    
    def test_color_palette(self):
        assert hasattr(config, 'COLOR_PALETTE')
        assert isinstance(config.COLOR_PALETTE, dict)
        
        for color in ['primary', 'secondary', 'accent']:
            if color in config.COLOR_PALETTE:
                assert isinstance(config.COLOR_PALETTE[color], str)
                assert config.COLOR_PALETTE[color].startswith('#')


class TestPopularTickers:
    def test_popular_tickers_dict(self):
        assert hasattr(config, 'POPULAR_TICKERS')
        assert isinstance(config.POPULAR_TICKERS, dict)
        assert len(config.POPULAR_TICKERS) > 0
    
    def test_popular_tickers_categories(self):
        for category, tickers in config.POPULAR_TICKERS.items():
            assert isinstance(tickers, list)
            for ticker in tickers:
                assert isinstance(ticker, str)


class TestErrorMessages:
    def test_error_messages_dict(self):
        assert hasattr(config, 'ERROR_MESSAGES')
        assert isinstance(config.ERROR_MESSAGES, dict)
        assert len(config.ERROR_MESSAGES) > 0
    
    def test_error_messages_are_strings(self):
        for key, message in config.ERROR_MESSAGES.items():
            assert isinstance(message, str)
            assert len(message) > 0


class TestColorValidation:
    def test_all_colors_valid_hex(self):
        import re
        hex_pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')
        
        for color_name, color_value in config.COLOR_PALETTE.items():
            assert hex_pattern.match(color_value), \
                f"Color '{color_name}' has invalid hex value: {color_value}"
