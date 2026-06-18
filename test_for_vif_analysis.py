'''
Accelera Consulting

Test for vif_analysis.py
'''
import pandas as pd
import json
import numpy as np
from vif_analysis import variance_inflation_factor, iterative_feature_selector_with_vif
from logging_config.logger_config import get_logger

logger_name = "mlops.vif_analysis"
logger_file_name = "vif_analysis.log"
logger = get_logger(logger_name, logger_file_name)

if __name__ == "__main__":
        # Test with UCI Credit Card Dataset
    metadata_path = "inputs/sample/datatypes.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    column_dtypes = metadata.get("column_dtypes", {})
    # Get the sample data
    input_csv_path = "inputs/sample/uci_credit_card_dataset.csv"
    df = pd.read_csv(input_csv_path, dtype=column_dtypes)

    # Get numerical columns only
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    X_train = df[numerical_columns].drop(columns=["TARGET"])
    y_train = df["TARGET"]

    # Calculate VIF for each feature in the training data and log the results.
    # vif_results = {}
    # for i in range(X_train.shape[1]):
    #     vif, interpretation = variance_inflation_factor(X_train.values, i)
    #     vif_results[X_train.columns[i]] = {
    #         "vif": vif,
    #         "interpretation": interpretation["interpretation"]
    #     }
    
    # Select features iteratively based on VIF threshold and log the remaining features after the selection process.
    selected_features, steps = iterative_feature_selector_with_vif(
        X_train, 
        vif_treshold=5.0, 
        maxiterations=100, 
        output_json_path="outputs/vif_analysis/vif_feature_selection_results.json",
        output_csv_path="outputs/vif_analysis/vif_feature_selection_steps.csv")