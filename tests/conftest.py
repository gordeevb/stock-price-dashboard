"""
conftest.py - Pytest configuration

This file adds the src/ directory to Python's path so tests can import modules.
"""

import sys
from pathlib import Path

# Get directories
tests_dir = Path(__file__).parent          # tests/
project_root = tests_dir.parent            # stock-price-dashboard/
src_dir = project_root / "src"             # stock-price-dashboard/src/

# Add src/ to Python path
sys.path.insert(0, str(src_dir))