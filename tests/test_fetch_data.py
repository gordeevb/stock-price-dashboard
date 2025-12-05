"""Test suite for fetch_data.py module"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
import os

from fetch_data import (
    validate_ticker,
    validate_parameters,
    get_cache_filename,
    is_cache_valid,
    validate_dataframe,
    DataFetchError,
    InvalidTickerError,
    NetworkError,
    VALID_PERIODS,
    VALID_INTERVALS,
    CACHE_DIR
)


# Ticker validation tests
class TestValidateTicker:
    def test_valid_ticker(self):
        is_valid, error_msg = validate_ticker("AAPL")
        assert is_valid is True
        assert error_msg == ""

    def test_empty_ticker(self):
        is_valid, error_msg = validate_ticker("")
        assert is_valid is False
        assert "empty" in error_msg.lower()

    def test_invalid_characters(self):
        is_valid, error_msg = validate_ticker("ABC@#$")
        assert is_valid is False


# Parameter validation tests
class TestValidateParameters:
    def test_valid_parameters(self):
        is_valid, error_msg = validate_parameters("1mo", "1d")
        assert is_valid is True
        assert error_msg == ""

    def test_invalid_period(self):
        is_valid, error_msg = validate_parameters("invalid", "1d")
        assert is_valid is False
        assert "period" in error_msg.lower()

    def test_invalid_interval(self):
        is_valid, error_msg = validate_parameters("1mo", "invalid")
        assert is_valid is False
        assert "interval" in error_msg.lower()


# Cache management tests
class TestCacheManagement:
    def setup_method(self):
        """Create temp directory for cache testing"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temp directory"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_get_cache_filename(self):
        filename = get_cache_filename("AAPL", "1mo", "1d")
        assert filename.name == "AAPL_1mo_1d.csv"

    def test_is_cache_valid_nonexistent(self):
        fake_path = Path(self.temp_dir) / "nonexistent.csv"
        assert is_cache_valid(fake_path, "1d") is False


# DataFrame validation tests
class TestDataFrameValidation:
    def test_validate_valid_dataframe(self):
        # Create valid OHLCV data
        df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        })
        is_valid, error_msg = validate_dataframe(df, "AAPL")
        assert is_valid is True

    def test_validate_none_dataframe(self):
        is_valid, error_msg = validate_dataframe(None, "AAPL")
        assert is_valid is False

    def test_validate_empty_dataframe(self):
        df = pd.DataFrame()
        is_valid, error_msg = validate_dataframe(df, "AAPL")
        assert is_valid is False


# Exception tests
class TestExceptions:
    def test_data_fetch_error(self):
        with pytest.raises(DataFetchError):
            raise DataFetchError("Test error")

    def test_invalid_ticker_error(self):
        with pytest.raises(InvalidTickerError):
            raise InvalidTickerError("Invalid ticker")

    def test_network_error(self):
        with pytest.raises(NetworkError):
            raise NetworkError("Network error")


# Constants validation tests
class TestConstants:
    def test_valid_periods_list(self):
        assert isinstance(VALID_PERIODS, list)
        assert len(VALID_PERIODS) > 0
        assert '1mo' in VALID_PERIODS

    def test_valid_intervals_list(self):
        assert isinstance(VALID_INTERVALS, list)
        assert len(VALID_INTERVALS) > 0
        assert '1d' in VALID_INTERVALS

    def test_cache_dir_exists(self):
        assert CACHE_DIR.exists()
        assert CACHE_DIR.is_dir()