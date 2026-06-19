'''
Accelera Consulting
Test file for auto_binning_woe.py
'''

from auto_binning_woe import (
    assign_bins_from_intervals,
    auto_woe_binning_numeric, 
    calculate_woe_iv_table, 
    create_initial_bins,
    find_monotonicity_violation_pair,
    find_most_similar_adjacent_pair, merge_intervals, 
    assign_bins_from_intervals,
    find_closest_neighbor,
    find_most_similar_adjacent_pair,
    find_monotonicity_violation_pair,
    check_binning_quality,
    )
from auto_binning_woe_categorical import auto_woe_binning_categorical

import pandas as pd
import numpy as np
import json
from pandas.api.types import is_numeric_dtype

if __name__ == "__main__":
    test_parameter = [
        "Test with UCI Credit Card Dataset"
    ]

    if "calculate_woe_iv_table" in test_parameter:
        # Generate Sample Data to test calculate_woe_iv_table function
        data = {
            "feature": [18, 22, 25, 30, 35, 40, 45, 50, 55, 60],
            "target": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        }
        df = pd.DataFrame(data)
        df["bin"] = pd.qcut(df["feature"], q=3)
        woe_iv_table, metadata = calculate_woe_iv_table(
            df, 
            bin_col="bin", 
            target="target", 
            eps=0.5, 
            output_dir="./outputs/auto_binning_woe"
            )

    if "create_initial_bins" in test_parameter:
        # Generate Sample Data to test create_initial_bins function
        df_bins = pd.DataFrame({
            "feature": [18, 22, 25, 30, 35, 40, 45, 50, 55, 60]
        })
        initial_bins = create_initial_bins(df_bins, feature="feature", initial_bins=3)

        # Generate Sample Data Series to test create_initial_bins function with series input
        series_bins = pd.Series([18, 22, 25, 30, 35, 40, 45, 50, 55, 60], name="feature_series")
        initial_bins_from_series = create_initial_bins(df_bins, feature=None, series=series_bins, initial_bins=3)
    
    if "merge_intervals" in test_parameter:
        # Generate Sample Data to test merge_intervals function
        intervals = [
            pd.Interval(18, 25, closed="right"),
            pd.Interval(25, 35, closed="right"),
            pd.Interval(35, 50, closed="right"),
            pd.Interval(50, 80, closed="right"), # We send this index to be merged with the next one
            pd.Interval(80, 100, closed="right"),
            pd.Interval(100, 150, closed="right")
        ]
        # Saying that merge intervals at index 3, which means we want to merge intervals[3] and intervals[4], which are (50, 80] and (80, 100]
        merged_intervals = merge_intervals(intervals, i=3)
    
    if "assign_bins_from_intervals" in test_parameter:
        # Generate Sample Data to test assign_bins_from_intervals function
        x = pd.Series([18, 22, 25, 30, 35, 40, 45, 50, 55, 60])
        intervals_for_assignment = [
            pd.Interval(18, 25, closed="right"),
            pd.Interval(25, 50, closed="right"),
            pd.Interval(50, 80, closed="right"),
            pd.Interval(80, 100, closed="right")
        ]
        assigned_bins = assign_bins_from_intervals(x, intervals_for_assignment)
    
    if "find_closest_neighbor" in test_parameter:
        # Generate Sample Data to test find_closest_neighbor function
        summary = pd.DataFrame({
            "bin": ["Bin A", "Bin B", "Bin C", "Bin D"],
            "bad_rate": [0.1, 0.11, 0.3, 0.025]
        })
        closest_neighbor_idx = find_closest_neighbor(summary, idx=1, metric="bad_rate")

    if "find_most_similar_adjacent_pair" in test_parameter:
        # Generate Sample Data to test find_most_similar_adjacent_pair function
        summary_for_similarity = pd.DataFrame({
            "bin": ["Bin A", "Bin B", "Bin C", "Bin D"],
            "bad_rate": [0.1, 0.11, 0.3, 0.3]
        })

        most_similar_adjacent_pair_idx = find_most_similar_adjacent_pair(summary_for_similarity, metric="bad_rate")

    if "find_monotonicity_violation_pair" in test_parameter:
        # Generate Sample Data to test find_monotonicity_violation_pair function
        summary_for_monotonicity = pd.DataFrame({
            "bin": ["Bin A", "Bin B", "Bin C", "Bin D"],
            "bad_rate": [0.05, 0.1, 0.08, 0.2]
        })

        monotonicity_violation_pair_idx = find_monotonicity_violation_pair(summary_for_monotonicity, metric="bad_rate")
    
    if "check_binning_quality" in test_parameter:
        # Generate Sample Data to test check_binning_quality function
        summary_for_quality_check = pd.DataFrame({
            "bin": ["Bin A", "Bin B", "Bin C"],
            "total": [100, 100, 100],
            "good": [90, 89, 70],
            "bad": [10, 11, 30],
            "bad_rate": [0.1, 0.11, 0.3],
            "bin_pct": [0.33, 0.33, 0.34],
            "woe": [-0.405465, -0.182322, 0.405465],
            "iv": [0.004054, 0.000328, 0.016218]
        })

        quality_checks, quality_checks_formatted = check_binning_quality(summary_for_quality_check, min_bin_pct=0.05, max_bins=5)
    
    if "auto_woe_binning_numeric" in test_parameter:
        # Generate Sample Data (100 Rows) to test auto_woe_binning_numeric function
        # Feature and target are logically related: as feature increases, probability of target=1 increases
        np.random.seed(42)
        feature_data = np.random.uniform(18, 250, 1000)
        # Use sigmoid function to create logical relationship: higher feature → higher probability of bad (1)
        probabilities = 1 / (1 + np.exp(-(feature_data - 59) / 15))  # Sigmoid centered around 59
        target_data = np.array([np.random.choice([0, 1], p=[1-p, p]) for p in probabilities])
        
        df_for_auto_binning = pd.DataFrame({
            "feature": feature_data,
            "target": target_data
        })

        auto_binning_result = auto_woe_binning_numeric(
            df=df_for_auto_binning,
            feature="feature",
            target="target",
            initial_bins=20, # We start with 20 initial bins to allow the algorithm to have enough granularity to work with and merge as needed based on the defined rules.
            min_bin_pct=0.05, # We set the minimum bin percentage to 5% to ensure that each bin has a sufficient number of observations to be statistically meaningful.
            max_final_bins=10, # We set the maximum final bins to 10 to align with common scorecard binning practices and to ensure that the final output is interpretable and actionable.
            min_final_bins=3, # We set the minimum final bins to 3 to ensure that the final binning output has enough granularity to capture meaningful patterns in the data while avoiding oversimplification.
            min_iv=0.02, # We set the minimum IV to 0.02 to flag features that may be too weak for predictive modeling and may require further review or exclusion from the model.
            max_iv=0.50, # We set the maximum IV to 0.50 to flag features that may have too strong of a relationship with the target variable, which could indicate potential data leakage or overfitting issues that require further investigation.
            max_iter=25 # We set the maximum iterations to 25 to allow the algorithm to perform enough merging steps to satisfy the defined rules while preventing infinite loops in cases where the rules cannot be satisfied within a reasonable number of merges.
        )

    if "Test with UCI Credit Card Dataset" in test_parameter:
        # Test with UCI Credit Card Dataset
        metadata_path = "inputs/sample/datatypes.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        column_dtypes = metadata.get("column_dtypes", {})
        # Get the sample data
        input_csv_path = "inputs/sample/uci_credit_card_dataset.csv"
        df = pd.read_csv(input_csv_path, dtype=column_dtypes)

        for col in df.columns:
            if col not in ["TARGET"]:
                if is_numeric_dtype(df[col]):
                    result = auto_woe_binning_numeric(
                        df=df,
                        feature=col,
                        target="TARGET",
                        initial_bins=20,
                        min_bin_pct=0.05,
                        max_final_bins=10,
                        min_final_bins=3,
                        min_iv=0.02,
                        max_iv=0.50,
                        max_iter=25
                    )
                # if data type is object, we will use auto_woe_binning_categorical
                elif df[col].dtype == 'object':
                    result = auto_woe_binning_categorical(
                        df=df,
                        feature=col,
                        target="TARGET",
                        min_bin_pct=0.05,
                        max_final_bins=6,
                        min_final_bins=2,
                        min_iv=0.02,
                        max_iv=0.50,
                        max_iter=20
                    )

