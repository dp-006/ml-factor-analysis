'''
Accelera Consulting
Author: Accelera Team

Sklearn-compatible wrapper for Logistic Regression using Statsmodels backend.
Provides a standard sklearn estimator interface for the LogisticRegression model.
'''

import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from logging_config.logger_config import get_logger
from ml_logistic import LogisticRegression as StatsmodelsLogisticRegression

logger_name = "mlops.sklearn_wrapper_logistic"
logger_file_name = "sklearn_wrapper_logistic.log"
logger = get_logger(logger_name, logger_file_name)


class LogisticRegressionWrapper(BaseEstimator, ClassifierMixin):
    """
    Purpose
    -------
    A sklearn-compatible wrapper for Logistic Regression using Statsmodels backend.
    This wrapper provides a standard sklearn estimator interface, allowing the
    LogisticRegression model to be used in sklearn pipelines and cross-validation.
    
    The wrapper delegates to the internal LogisticRegression class from ml_logistic module
    which implements logistic regression using statsmodels.api.Logit.
    
    Parameters
    ----------
    disp : bool, optional
        Display convergence messages during model fitting. Default is False.
    
    method : str, optional
        Optimization method to use for fitting. Default is 'bfgs'.
        Options include: 'newton', 'lbfgs', 'powell', 'cg', 'ncg', 'basinhopping', 'minimize'.
    
    maxiter : int, optional
        Maximum number of iterations for the optimization algorithm. Default is 500.
    
    model_dir : str, optional
        Directory where model results and artifacts will be saved. Default is 'model'.
    
    metrics_dir : str, optional
        Directory where evaluation metrics and plots will be saved.
        If not provided, defaults to '<model_dir>/metrics'.
    
    save_results : bool, optional
        Whether to save model results to JSON files. Default is True.
    
    save_model : bool, optional
        Whether to save the fitted model to a pickle file. Default is True.
    
    evaluate_model : bool, optional
        Whether to evaluate the model after fitting. Default is True.
    
    evaluate_bins : int, optional
        Number of bins to use for model evaluation metrics. Default is 20.
    
    threshold : float, optional
        Probability threshold for binary classification (used in predict). Default is 0.5.
        Values >= threshold are classified as 1, values < threshold as 0.
    
    Attributes
    ----------
    model_ : LogisticRegression
        The internal LogisticRegression instance from ml_logistic module.
    
    model_fit_ : statsmodels.discrete.discrete_model.LogitResults
        The fitted model results object from statsmodels after calling fit().
    
    classes_ : np.ndarray
        The unique class labels [0, 1] for binary classification.
    
    n_features_in_ : int
        Number of features seen during fit (excluding constant term).
    
    Examples
    --------
    >>> from sklearn_wrapper_logistic import LogisticRegressionWrapper
    >>> import pandas as pd
    >>> 
    >>> # Create wrapper instance
    >>> model = LogisticRegressionWrapper(method='bfgs', maxiter=500)
    >>> 
    >>> # Fit model
    >>> X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    >>> y = pd.Series([0, 1, 1])
    >>> model.fit(X, y)
    >>> 
    >>> # Make predictions
    >>> predictions = model.predict(X)  # Returns class labels: [0, 1, 1]
    >>> probabilities = model.predict_proba(X)  # Returns probabilities: [[0.4, 0.6], ...]
    
    """
    
    def __init__(
        self, 
        disp=False, 
        method="bfgs", 
        maxiter=500, 
        model_dir="model",
        metrics_dir=None,
        save_results=True,
        save_model=True,
        evaluate_model=True,
        evaluate_bins=20,
        threshold=0.5
    ):
        self.disp = disp
        self.method = method
        self.maxiter = maxiter
        self.model_dir = model_dir
        self.metrics_dir = metrics_dir
        self.save_results = save_results
        self.save_model = save_model
        self.evaluate_model = evaluate_model
        self.evaluate_bins = evaluate_bins
        self.threshold = threshold
        
        # These will be set during fit()
        self.model_ = None
        self.model_fit_ = None
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = None
        self.evaluation_metrics_ = None
        
        logger.info("-" * 50)
        logger.info("LogisticRegressionWrapper initialized with parameters:")
        logger.info(f"\tDisplay convergence messages: {self.disp}")
        logger.info(f"\tOptimization method: {self.method}")
        logger.info(f"\tMaximum iterations: {self.maxiter}")
        logger.info(f"\tModel directory: {self.model_dir}")
        logger.info(f"\tMetrics directory: {self.metrics_dir if self.metrics_dir is not None else f'{self.model_dir}/metrics (default)'}")
        logger.info(f"\tSave results: {self.save_results}")
        logger.info(f"\tSave model: {self.save_model}")
        logger.info(f"\tEvaluate model: {self.evaluate_model}")
        logger.info(f"\tEvaluation bins: {self.evaluate_bins}")
        logger.info(f"\tPrediction threshold: {self.threshold}")
        logger.info("-" * 50)
    
    def fit(self, X, y):
        """
        Purpose
        -------
        Fit the Logistic Regression model to training data using sklearn interface.
        Internally delegates to LogisticRegression.fit() from ml_logistic module.
        
        Parameters
        ----------
        X : array-like or pd.DataFrame
            Training features. If array-like, will be converted to DataFrame.
            Shape: (n_samples, n_features)
        
        y : array-like or pd.Series
            Target variable with binary values (0 and 1).
            Shape: (n_samples,)
        
        Returns
        -------
        self
            Returns self for method chaining in sklearn pipelines.
        
        Raises
        ------
        ValueError
            If target variable is not binary (0 and 1).
            If X or y contains missing values.
            If X and y have different number of samples.
        
        Exception
            For any errors during model fitting.
        """
        logger.info("Starting fit method")
        
        # Convert to DataFrame if necessary
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            else:
                X = pd.DataFrame(X)
            logger.info(f"Converted input X to DataFrame with shape: {X.shape}")
        
        # Convert y to Series if necessary
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            logger.info(f"Converted input y to Series with shape: {y.shape}")
        
        # Validate input shapes match
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples. Got X: {X.shape[0]}, y: {y.shape[0]}")
        
        # Store number of features
        self.n_features_in_ = X.shape[1]
        logger.info(f"Number of input features: {self.n_features_in_}")
        
        try:
            # Create internal LogisticRegression instance
            self.model_ = StatsmodelsLogisticRegression(
                disp=self.disp,
                method=self.method,
                maxiter=self.maxiter,
                model_dir=self.model_dir,
                metrics_dir=self.metrics_dir,
                save_results=self.save_results,
                save_model=self.save_model,
                evaluate_model=self.evaluate_model,
                evaluate_bins=self.evaluate_bins
            )
            
            logger.info("Created internal LogisticRegression instance")
            
            # Fit the internal model
            self.model_fit_ = self.model_.fit(X, y)
            logger.info("Internal model fitted successfully")
            
            # Expose evaluation metrics from internal model
            self.evaluation_metrics_ = self.model_.evaluation_metrics_
            
        except Exception as e:
            error_message = f"Error during model fitting: {str(e)}"
            logger.error(error_message)
            raise Exception(error_message) from e
        
        return self
    
    def predict(self, X):
        """
        Purpose
        -------
        Predict binary class labels using the fitted logistic regression model.
        Uses the threshold parameter to convert probabilities to class labels.
        
        Parameters
        ----------
        X : array-like or pd.DataFrame
            Features to make predictions on. If array-like, will be converted to DataFrame.
            Shape: (n_samples, n_features)
            Must have the same number of features as training data.
        
        Returns
        -------
        np.ndarray
            Predicted class labels (0 or 1).
            Shape: (n_samples,)
        
        Raises
        ------
        ValueError
            If model has not been fitted yet.
            If number of features does not match training data.
        """
        logger.info(f"Starting predict with threshold={self.threshold:.4f}")
        
        if self.model_ is None or self.model_fit_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Convert to DataFrame if necessary
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            else:
                X = pd.DataFrame(X)
            logger.info(f"Converted input X to DataFrame with shape: {X.shape}")
        
        # Validate number of features
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Number of features mismatch. Expected {self.n_features_in_}, got {X.shape[1]}")
        
        # Get probabilities and convert to class labels
        probabilities = self.model_.predict_proba(X)
        predictions = np.array([int(prob >= self.threshold) for prob in probabilities])
        
        logger.info(f"Predictions completed for {len(predictions)} samples")
        logger.info(f"Class 0: {(predictions == 0).sum()}, Class 1: {(predictions == 1).sum()}")
        
        return predictions
    
    def predict_proba(self, X):
        """
        Purpose
        -------
        Predict probabilities of both classes using the fitted logistic regression model.
        
        Parameters
        ----------
        X : array-like or pd.DataFrame
            Features to make predictions on. If array-like, will be converted to DataFrame.
            Shape: (n_samples, n_features)
            Must have the same number of features as training data.
        
        Returns
        -------
        np.ndarray
            Predicted class probabilities.
            Shape: (n_samples, 2)
            Column 0: Probability of class 0
            Column 1: Probability of class 1
        
        Raises
        ------
        ValueError
            If model has not been fitted yet.
            If number of features does not match training data.
        
        Examples
        --------
        >>> probs = model.predict_proba(X)
        >>> # Output shape: (n_samples, 2)
        >>> # Example: [[0.7, 0.3], [0.2, 0.8], [0.5, 0.5]]
        """
        logger.info("Starting predict_proba")
        
        if self.model_ is None or self.model_fit_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Convert to DataFrame if necessary
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            else:
                X = pd.DataFrame(X)
            logger.info(f"Converted input X to DataFrame with shape: {X.shape}")
        
        # Validate number of features
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Number of features mismatch. Expected {self.n_features_in_}, got {X.shape[1]}")
        
        # Get probabilities for positive class from internal model
        probs_positive = np.array(self.model_.predict_proba(X))
        
        # Create probability matrix for both classes (sklearn convention)
        # Column 0: probability of class 0, Column 1: probability of class 1
        probs_negative = 1 - probs_positive
        proba = np.column_stack([probs_negative, probs_positive])
        
        logger.info(f"Probabilities predicted for {len(proba)} samples")
        logger.info(f"Output shape: {proba.shape}")
        
        return proba
    
    def score(self, X, y):
        """
        Purpose
        -------
        Return the mean accuracy on the given test data and labels.
        This method is required for sklearn compatibility.
        
        Parameters
        ----------
        X : array-like or pd.DataFrame
            Test features.
            Shape: (n_samples, n_features)
        
        y : array-like or pd.Series
            True binary labels.
            Shape: (n_samples,)
        
        Returns
        -------
        float
            Accuracy score in the range [0, 1].
            Calculated as: (correct predictions) / (total predictions)
        
        Examples
        --------
        >>> accuracy = model.score(X_test, y_test)
        >>> print(f"Model accuracy: {accuracy:.4f}")  # Output: Model accuracy: 0.8234
        """
        logger.info("Calculating accuracy score")
        
        predictions = self.predict(X)
        
        # Convert y to numpy array if necessary
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        accuracy = np.mean(predictions == y_array)
        logger.info(f"Accuracy score: {accuracy:.4f}")
        
        return accuracy
    
    def get_params(self, deep=True):
        """
        Purpose
        -------
        Get parameters for this estimator. Required for sklearn compatibility,
        especially for GridSearchCV and other hyperparameter tuning utilities.
        
        Parameters
        ----------
        deep : bool, optional
            If True, will return parameters for this estimator and
            contained subobjects that are estimators. Default is True.
        
        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        return {
            'disp': self.disp,
            'method': self.method,
            'maxiter': self.maxiter,
            'model_dir': self.model_dir,
            'metrics_dir': self.metrics_dir,
            'save_results': self.save_results,
            'save_model': self.save_model,
            'evaluate_model': self.evaluate_model,
            'evaluate_bins': self.evaluate_bins,
            'threshold': self.threshold
        }
    
    def set_params(self, **params):
        """
        Purpose
        -------
        Set the parameters of this estimator. Required for sklearn compatibility,
        especially for GridSearchCV and other hyperparameter tuning utilities.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        
        Returns
        -------
        self
            Returns self for method chaining.
        
        Examples
        --------
        >>> model.set_params(maxiter=1000, threshold=0.6)
        >>> # Now model has maxiter=1000 and threshold=0.6
        """
        if not params:
            return self
        
        valid_params = self.get_params(deep=False)
        nested_params = {}
        
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter {key} for estimator {type(self).__name__}. "
                               f"Valid parameters are: {list(valid_params.keys())}")
            setattr(self, key, value)
        
        return self


if __name__ == "__main__":
    """
    Example usage of LogisticRegressionWrapper
    """
    import json
    from factor_analysis import prepare_factor_analysis_data
    
    logger.info("Starting example usage of LogisticRegressionWrapper")
    
    # Load sample data
    metadata_path = "inputs/sample/datatypes.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    column_dtypes = metadata.get("column_dtypes", {})
    
    input_csv_path = "inputs/sample/uci_credit_card_dataset.csv"
    df = pd.read_csv(input_csv_path, dtype=column_dtypes)
    logger.info(f"Loaded sample data with shape: {df.shape}")
    
    # Prepare data for factor analysis and logistic regression
    df_prepared, metadata = prepare_factor_analysis_data(
        df=df,
        target_variable="TARGET",
    )
    
    logger.info(f"Prepared data shape: {df_prepared.shape}")
    
    # Split features and target
    X = df_prepared.drop(columns=["TARGET"])
    y = df_prepared["TARGET"]
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Create and fit the wrapper model
    model = LogisticRegressionWrapper(
        method='bfgs',
        maxiter=500,
        threshold=0.5
    )
    
    logger.info("Fitting LogisticRegressionWrapper...")
    model.fit(X, y)
    logger.info("Model fitted successfully")
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Probabilities shape: {probabilities.shape}")
    
    # Calculate accuracy
    accuracy = model.score(X, y)
    logger.info(f"Model accuracy: {accuracy:.4f}")
