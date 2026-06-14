import pandas as pd
from helper import io_check_dataframe_quality, io_save_json
from helper import convert_binary_columns


if __name__ == "__main__":
    # Generate Sample DataFrame and test all the checks in the check_dataframe_quality function
    sample_data = {
        "numeric_col": [1, 2, 3, 4, 5], # Numeric column with no issues
        "categorical_col": pd.Categorical(["A", "B", "A", "C", "B"]), # Categorical column with 3 categories
        "object_col": ["foo", "bar", "baz", "qux", "quux"], # Object column with no issues
        "string_col": pd.Series(["alpha", "beta", "gamma", "delta", "epsilon"], dtype="string"), # String column with no issues
        "missing_col": [1, 2, None, 4, 5], # Column with a missing value
        "inf_col": [1, 2, float('inf'), 4, 5], # Column with an Inf value
        "special_char_col": ["hello!", "world@", "test#", "data$", "mlops%"], # Column with special characters
        "constant_numeric_col": [1, 1, 1, 1, 1], # Constant column with zero variance
        'constant_object_col': ['constant', 'constant', 'constant', 'constant', 'constant'], # Constant object column
        "high_cardinality_col": [f"category_{i}" for i in range(5)], # Column with high cardinality
        "numeric_like_object_col": ["1", "2", "3", "4", "5"], # Column with numeric-like strings
        "zero_variance_col": [1, 1, 1, 1, 1], # Column with zero variance
        "high_corr_col1": [1, 2, 3, 4, 5], # Column with high correlation
        "high_corr_col2": [2, 4, 6, 8, 10], # Column with high correlation
        "secret_binary_col": [0, 1, 0, 1, 0], # Binary column with values 0 and 1
        "secret_binary_col_2": [10, 10, 10, 20, 20], # Binary column with values 10 and 20
    }
    sample_df = pd.DataFrame(sample_data)
    result = io_check_dataframe_quality(sample_df)
    _ = io_save_json(result, "./outputs/io_operations/data_quality_report.json")
    sample_df, binary_cols = convert_binary_columns(sample_df)
    _ = io_save_json(binary_cols, "./outputs/column_operations/binary_columns_metadata.json")   
