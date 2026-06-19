"""
Test script for WOETransformerNumeric and WOETransformerCategorical classes
Demonstrates both numeric and categorical feature transformation
"""

import os
import pandas as pd
from sklearn_wrapper_woe import WOETransformerNumeric, WOETransformerCategorical
from get_sample_data import get_sample_data
from logging_config.logger_config import get_logger

logger_name = "test_woe_transformers"
logger_file_name = "test_woe_transformers.log"
logger = get_logger(logger_name, logger_file_name)

def test_numeric_transformer():
    """Test WOETransformerNumeric with numeric features"""
    logger.info("=" * 80)
    logger.info("Testing WOETransformerNumeric")
    logger.info("=" * 80)
    
    # Load sample data
    df = get_sample_data()
    logger.info(f"Loaded sample data with shape: {df.shape}")
    
    # Select numeric features for testing (e.g., 'age', 'bill_amount_apr_2005')
    numeric_features = ['age', 'bill_amount_apr_2005', 'credit_limit']
    numeric_features = [f for f in numeric_features if f in df.columns]
    
    logger.info(f"Selected numeric features for transformation: {numeric_features}")
    X_numeric = df[numeric_features].head(100)
    
    logger.info(f"Input numeric data shape: {X_numeric.shape}")
    logger.info(f"Input numeric data:\n{X_numeric.head()}")
    
    # Fit and transform
    transformer = WOETransformerNumeric()
    X_numeric_woe = transformer.fit_transform(X_numeric)
    
    logger.info(f"Transformed numeric data shape: {X_numeric_woe.shape}")
    logger.info(f"Transformed numeric features: {list(X_numeric_woe.columns)}")
    logger.info(f"Transformed numeric data:\n{X_numeric_woe.head()}")
    
    return X_numeric_woe

def test_categorical_transformer():
    """Test WOETransformerCategorical with categorical features"""
    logger.info("=" * 80)
    logger.info("Testing WOETransformerCategorical")
    logger.info("=" * 80)
    
    # Load sample data
    df = get_sample_data()
    logger.info(f"Loaded sample data with shape: {df.shape}")
    
    # Select categorical features for testing
    categorical_features = ['education', 'gender', 'marital_status']
    categorical_features = [f for f in categorical_features if f in df.columns]
    
    logger.info(f"Selected categorical features for transformation: {categorical_features}")
    X_categorical = df[categorical_features].head(100)
    
    logger.info(f"Input categorical data shape: {X_categorical.shape}")
    logger.info(f"Input categorical data:\n{X_categorical.head()}")
    logger.info(f"Input categorical data dtypes:\n{X_categorical.dtypes}")
    
    # Fit and transform
    transformer = WOETransformerCategorical()
    X_categorical_woe = transformer.fit_transform(X_categorical)
    
    logger.info(f"Transformed categorical data shape: {X_categorical_woe.shape}")
    logger.info(f"Transformed categorical features: {list(X_categorical_woe.columns)}")
    logger.info(f"Transformed categorical data:\n{X_categorical_woe.head()}")
    
    return X_categorical_woe

def test_combined_pipeline():
    """Test combining both numeric and categorical transformers"""
    logger.info("=" * 80)
    logger.info("Testing Combined Pipeline (Numeric + Categorical)")
    logger.info("=" * 80)
    
    # Load sample data
    df = get_sample_data()
    
    # Select both types of features
    numeric_features = ['age', 'bill_amount_apr_2005', 'credit_limit']
    categorical_features = ['education', 'gender', 'marital_status']
    
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]
    
    sample_data = df.head(100)
    
    # Transform numeric features
    X_numeric = sample_data[numeric_features]
    transformer_numeric = WOETransformerNumeric()
    X_numeric_woe = transformer_numeric.fit_transform(X_numeric)
    logger.info(f"Numeric WOE transformed features: {list(X_numeric_woe.columns)}")
    
    # Transform categorical features
    X_categorical = sample_data[categorical_features]
    transformer_categorical = WOETransformerCategorical()
    X_categorical_woe = transformer_categorical.fit_transform(X_categorical)
    logger.info(f"Categorical WOE transformed features: {list(X_categorical_woe.columns)}")
    
    # Combine results
    X_combined = pd.concat([X_numeric_woe, X_categorical_woe], axis=1)
    logger.info(f"Combined WOE data shape: {X_combined.shape}")
    logger.info(f"Combined WOE features: {list(X_combined.columns)}")
    logger.info(f"Combined WOE data:\n{X_combined.head()}")
    
    return X_combined

if __name__ == "__main__":
    try:
        logger.info("Starting WOE Transformers Tests")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # Test individual transformers
        X_numeric_woe = test_numeric_transformer()
        logger.info("\n✓ Numeric transformer test passed")
        
        X_categorical_woe = test_categorical_transformer()
        logger.info("\n✓ Categorical transformer test passed")
        
        # Test combined pipeline
        X_combined = test_combined_pipeline()
        logger.info("\n✓ Combined pipeline test passed")
        
        logger.info("\n" + "=" * 80)
        logger.info("All tests completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}", exc_info=True)
        raise
