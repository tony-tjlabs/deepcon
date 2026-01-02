"""
Data Formatting and Sanitization Utilities
==========================================
Common formatters for numbers, time, and data cleaning
"""
import numpy as np
import pandas as pd
from typing import Any, Union


def format_number(value: Union[int, float], precision: int = 0) -> str:
    """
    Format number with thousands separator
    
    Args:
        value: Number to format
        precision: Decimal places (default 0)
        
    Returns:
        Formatted string (e.g., "1,234" or "1,234.56")
    """
    if precision == 0:
        return f"{int(value):,}"
    return f"{value:,.{precision}f}"


def time_index_to_time_str(time_index: int) -> str:
    """
    Convert time index (0-95) to HH:MM format
    
    Args:
        time_index: Time index from 0 (00:00) to 95 (23:45)
        
    Returns:
        Time string in HH:MM format
    """
    hour = time_index // 4
    minute = (time_index % 4) * 15
    return f"{hour:02d}:{minute:02d}"


def bin_index_to_time_str(bin_index: int) -> str:
    """
    Convert bin index (0-47) to HH:MM format
    
    Args:
        bin_index: Bin index from 0 (00:00) to 47 (23:30)
        
    Returns:
        Time string in HH:MM format
    """
    hour = bin_index // 2
    minute = (bin_index % 2) * 30
    return f"{hour:02d}:{minute:02d}"


def _sanitize_data(data: Any) -> Any:
    """
    Sanitize data for JSON serialization
    Converts NaN, Inf to None
    
    Args:
        data: Any Python object
        
    Returns:
        Sanitized data safe for JSON
    """
    if isinstance(data, (np.floating, float)):
        if np.isnan(data) or np.isinf(data):
            return None
    elif isinstance(data, (np.integer, np.int64)):
        return int(data)
    elif isinstance(data, np.ndarray):
        return [_sanitize_data(item) for item in data.tolist()]
    elif isinstance(data, list):
        return [_sanitize_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: _sanitize_data(value) for key, value in data.items()}
    return data


def _deterministic_jitter(values: np.ndarray, seed: int = 42, scale: float = 0.001) -> np.ndarray:
    """
    Add deterministic jitter to avoid overlap in scatter plots
    
    Args:
        values: Array of values
        seed: Random seed for reproducibility
        scale: Jitter scale factor
        
    Returns:
        Array with added jitter
    """
    rng = np.random.default_rng(seed)
    return values + rng.uniform(-scale, scale, size=len(values))


def _clean_figure_for_json(fig_dict: dict) -> dict:
    """
    Clean Plotly figure dictionary for JSON serialization
    Removes NaN/Inf values from figure data
    
    Args:
        fig_dict: Plotly figure dictionary
        
    Returns:
        Cleaned figure dictionary
    """
    if 'data' in fig_dict:
        for trace in fig_dict['data']:
            for key in list(trace.keys()):
                if key in ('x', 'y', 'z', 'text', 'marker'):
                    trace[key] = _sanitize_data(trace[key])
    
    if 'layout' in fig_dict:
        fig_dict['layout'] = _sanitize_data(fig_dict['layout'])
    
    return fig_dict


def _deep_sanitize(obj: Any) -> Any:
    """
    Deep sanitization for nested structures
    More aggressive than _sanitize_data
    
    Args:
        obj: Any Python object
        
    Returns:
        Deeply sanitized object
    """
    if isinstance(obj, dict):
        return {k: _deep_sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_deep_sanitize(item) for item in obj]
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return [_deep_sanitize(item) for item in obj.tolist()]
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    return obj


def percentage(value: float, total: float, precision: int = 1) -> str:
    """
    Calculate and format percentage
    
    Args:
        value: Numerator value
        total: Denominator value
        precision: Decimal places
        
    Returns:
        Formatted percentage string (e.g., "25.0%")
    """
    if total == 0:
        return "0.0%"
    pct = (value / total) * 100
    return f"{pct:.{precision}f}%"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)
