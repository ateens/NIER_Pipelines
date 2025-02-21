import numpy as np
import pandas as pd
from fastdtw import fastdtw
from typing import List



def str_to_float_list(values_str: str) -> list:
    """
    returns a list of floats from a comma-separated string of values.
    """
    return [float(val) for val in values_str.split(',')]

def sliding_fast_dtw(x, y, w=6, policy="zero") -> float:
    """
    Returns the Average FastDTW distance between two time-series using sliding window method
    
    Args:
        x (list): The first time-series to compare
        y (list): The second time-series to compare
        w (int): The window size for the sliding window
        policy (str): The policy to handle missing values. Options: "mean", "zero", "interpolate"
    Returns:
        float: The average FastDTW distance between the two time-series
    """
    
    def handle_missing_values(values, policy):
        """Handles missing values in a time-series based on the specified policy."""
        if policy == "mean":
            return np.nan_to_num(values, nan=np.nanmean(values))
        elif policy == "zero":
            return np.nan_to_num(values, nan=0)
        elif policy == "interpolate":
            # Use Pandas for interpolation
            return pd.Series(values).interpolate(method='linear', limit_direction='both').to_numpy()
        else:
            raise ValueError(f"Unsupported missing_value_policy: {policy}")
        
    # Handle missing values
    x = handle_missing_values(x, policy)
    y = handle_missing_values(y, policy)
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    if n < w:
        raise ValueError(
            "Time-series length must be greater than or equal to window size (w).")
    distances = []
    
    for start in range(0, n - w + 1, w):
        # Extract window slices
        x_window = x[start:start + w]
        y_window = y[start:start + w]
        # Compute FastDTW distance for the current window
        distance, local_path = fastdtw(x_window, y_window)
        distances.append(distance)
        # visualize_dtw_path(x_window, y_window, local_path)
    avg_distance = float(np.mean(distances))
    
    return avg_distance

def compare_time_series(original_data: dict, related_data_list: List[dict], comparison_type: str) -> List[dict]:
    """
    Compare time-series data using FastDTW.
    
    Args:
        original_data (dict): Original time-series data.
        related_data_list (List[dict]): List of related time-series data.
        comparison_type (str): 'station' or 'element'.
        
    Returns:
        List[dict]: A list of comparison results with distance scores.
    """
    
    results = []
    try:
        # Ensure original values are float lists
        original_values = original_data.get("values", [])
        # print("###COMPARE_TIME_SERIES### original_values: ", original_values)
        
        if isinstance(original_values, str):
            original_values = str_to_float_list(original_values)
            
        # Replace 999999.0 with NaN
        original_values = [
            float('nan') if v == 999999.0 else v for v in original_values]
        
        if not original_values:
            raise ValueError("Original data is empty or invalid.")
        
        for related_data in related_data_list:
            related_values = related_data.get("values", [])
            
            if isinstance(related_values, str):
                related_values = str_to_float_list(related_values)
                
            if not related_values:
                continue  # Skip if related data is invalid
            
            distance = sliding_fast_dtw(
                original_values, related_values)
            
            # Determine the comparison target based on comparison type
            compared_with = related_data.get(
                "element") if comparison_type == "element" else related_data.get("region")
            
            results.append({
                "comparison_type": comparison_type,
                "compared_with": compared_with,
                "distance": distance
            })
            
    except Exception as e:
        print(f"###COMPARE_ERROR### {comparison_type} comparison failed: {e}")
        
    return results


def calculate_z_score(x, mean, std):
    """
    Calculate the Z-score of a value based on the mean and standard deviation.
    """
    
    return (x - mean) / std

def sigmoid(x, k=1, d0=5):
    """
    Calculate the sigmoid function with parameters k and d0 for Reliability-based Fusion 
    """
    
    return 1 / (1 + np.exp(-k * (x - d0)))