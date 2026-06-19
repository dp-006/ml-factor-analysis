'''
Accelera Consulting

Test for ml_logistic.py
'''
from datetime import datetime, timezone, timedelta
import uuid
from logging_config.logger_config import setup_multiple_loggers
import pandas as pd
import json
from factor_analysis import prepare_factor_analysis_data
from helper import io_load_model
from ml_logistic import LogisticRegression

if __name__ == "__main__":

    # With the approach we log under timestamp and runid, we can easily track and compare logs across different runs and modules.
    timestamp = datetime.now(timezone(timedelta(hours=3))).strftime("%Y%m%d_%H%M%S")
    runid = uuid.uuid4().hex[:8]

    # Set up loggers for different modules with consistent configuration
    loggers = setup_multiple_loggers(
        level="info",
        log_mode="w",
        timestamp=timestamp,
        runid=runid,
        propagate=False
)


    # Test with UCI Credit Card Dataset
    metadata_path = "inputs/sample/datatypes.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    column_dtypes = metadata.get("column_dtypes", {})
    # Get the sample data
    input_csv_path = "inputs/sample/uci_credit_card_dataset.csv"
    df = pd.read_csv(input_csv_path, dtype=column_dtypes)
    
    # Prepare data for factor analysis and logistic regression
    df_prepared, metadata = prepare_factor_analysis_data(
        df=df,
        target_variable="TARGET",
    )
    X_train = df_prepared.drop(columns=["TARGET"])
    y_train = df_prepared["TARGET"]

    log_reg = LogisticRegression(model_dir=f"logitmodel/{timestamp}/{runid}")
    fitted_model = log_reg.fit(X_train, y_train)

    # Load model
    loaded_model = io_load_model(f"./outputs/logitmodel/{timestamp}/{runid}/logistic_model.pkl")

    # Check results with fitted model and loaded model
    # Important: For statsmodels predict(), we must add constant term manually as sm.add_constant() 
    # doesn't work reliably in all contexts
    one_sample = X_train.iloc[0:1]
    one_sample_with_const = one_sample.copy()
    one_sample_with_const.insert(0, 'const', 1.0)
    
    # Make predictions using three methods
    fitted_prediction = fitted_model.predict(one_sample_with_const)
    fitted_prediction_via_class = log_reg.predict(one_sample)
    loaded_prediction = loaded_model.predict(one_sample_with_const)
    
    # Display results
    print("\n" + "="*60)
    print("MODEL PREDICTION TEST RESULTS")
    print("="*60)
    print(f"Sample prediction: {fitted_prediction.values[0]:.6f}")
    print(f"Predictions match (fitted vs loaded): {(fitted_prediction - loaded_prediction).abs().max() < 1e-10}")
    print(f"Predictions match (class method vs loaded): {(fitted_prediction_via_class - loaded_prediction).abs().max() < 1e-10}")
    print("="*60)
