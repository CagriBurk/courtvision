import time
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.player_stats import _cache_fresh, CACHE_TTL_HOURS

def test_cache_fresh_file_does_not_exist():
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = False

    result = _cache_fresh(mock_path)

    assert result is False
    mock_path.exists.assert_called_once()

@patch('src.player_stats.time.time')
@patch('src.player_stats.os.path.getmtime')
def test_cache_fresh_file_is_fresh(mock_getmtime, mock_time):
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True

    current_time = 100000.0
    mock_time.return_value = current_time

    # Age is CACHE_TTL_HOURS - 1 hours (fresh)
    mock_getmtime.return_value = current_time - ((CACHE_TTL_HOURS - 1) * 3600)

    result = _cache_fresh(mock_path)

    assert result is True
    mock_path.exists.assert_called_once()
    mock_getmtime.assert_called_once_with(mock_path)

@patch('src.player_stats.time.time')
@patch('src.player_stats.os.path.getmtime')
def test_cache_fresh_file_is_stale(mock_getmtime, mock_time):
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True

    current_time = 100000.0
    mock_time.return_value = current_time

    # Age is CACHE_TTL_HOURS + 1 hours (stale)
    mock_getmtime.return_value = current_time - ((CACHE_TTL_HOURS + 1) * 3600)

    result = _cache_fresh(mock_path)

    assert result is False
    mock_path.exists.assert_called_once()
    mock_getmtime.assert_called_once_with(mock_path)

@patch('src.player_stats.time.time')
@patch('src.player_stats.os.path.getmtime')
def test_cache_fresh_file_is_exactly_ttl(mock_getmtime, mock_time):
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True

    current_time = 100000.0
    mock_time.return_value = current_time

    # Age is exactly CACHE_TTL_HOURS hours
    mock_getmtime.return_value = current_time - (CACHE_TTL_HOURS * 3600)

    result = _cache_fresh(mock_path)

    assert result is False
    mock_path.exists.assert_called_once()
    mock_getmtime.assert_called_once_with(mock_path)
