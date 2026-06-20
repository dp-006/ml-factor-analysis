'''
Accelera Consulting
Author: Accelera Team
Created on: 2024-06-17

This module implements Logistic Regression using Statsmodels. It includes detailed logging, model summary extraction, and saving results to a model directory. The class `LogisticRegression` provides methods for fitting the model, making predictions, and extracting summaries with interpretations of coefficients and fit quality metrics.

For Logit Model:
https://www.statsmodels.org/stable/examples/notebooks/generated/discrete_choice_overview.html
https://www.statsmodels.org/stable/examples/notebooks/generated/discrete_choice_overview.html#Logit-Model
https://www.statsmodels.org/stable/examples/notebooks/generated/discrete_choice_example.html


API:
statsmodels.api: Cross-sectional models and methods. Canonically imported using import statsmodels.api as sm
https://www.statsmodels.org/stable/api.html#statsmodels-api

For logit fit:
https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit.html#statsmodels.discrete.discrete_model.Logit.fit

For logit results:
https://www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.LogitResults.html

For Marginal Effects:
https://www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.LogitResults.get_margeff.html#statsmodels.discrete.discrete_model.LogitResults.get_margeff


'''
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from logging_config.logger_config import get_logger
from factor_analysis import prepare_factor_analysis_data
from helper import (
    io_save_json, 
    io_save_dataframe_as_csv, 
    io_save_model, 
    io_load_model, 
    io_save_figure,

    interpret_p_value, 
    interpret_odds_ratio,
    interpret_marginal_effect,

    calculate_confusion_matrix,
    calculate_accuracy,
    calculate_precision,
    calculate_recall,
    calculate_f1_score,

    calculate_roc_auc,
    calculate_gini,

    interpret_roc_auc,
    interpret_gini,

    calculate_ks,
    interpret_ks,

    calculate_decile_analysis,
    calculate_threshold_analysis,

    plot_ks,
    plot_lift,
    plot_lorenz,
    plot_roc_auc
    )

logger_name = "mlops.ml_logistic"
logger_file_name = "ml_logistic.log"
logger = get_logger(logger_name, logger_file_name)


class LogisticRegression:
    '''
    Purpose
    -------
    This class implements Logistic Regression using Statsmodels. 
    It provides methods for fitting the model, making predictions, and extracting 
    summaries with interpretations of coefficients and fit quality metrics.

    Methods:
    -------
    fit(X, y): Fit the Logistic Regression model to the training data.
    get_model_info(model_fit): Extract key model fit metrics from the fitted model.
    get_coefficients(model_fit): Extract coefficients, standard errors, z-statistics, p-values, and confidence intervals.
    get_odds_ratios(model_fit): Extract odds ratios and their interpretations.
    get_marginal_effects(model_fit, at="overall"): Extract marginal effects and their interpretations.
    evaluate_model(probs, labels, actuals): Evaluate model performance using various metrics.
    predict_proba(X): Predict probabilities of the positive class for given features.
    predict(X, threshold=0.5): Predict binary class labels based on a specified probability threshold.

    '''
    def __init__(
            self, 
            disp=False, 
            method="bfgs", 
            maxiter=500, 
            model_dir="outputs/logitmodel",
            metrics_dir=None,
            save_results=True,
            save_model=True,
            evaluate_model=True,
            evaluate_bins=20):
        self.model = None
        self.model_fit = None
        self.evaluation_metrics_ = None
        self.disp = disp
        self.method = method
        self.maxiter = maxiter
        self.model_dir = model_dir
        self.metrics_dir = metrics_dir if metrics_dir is not None else f"{model_dir}/metrics"
        self.save_results = save_results
        self.save_model = save_model
        self.perform_evaluation = evaluate_model
        self.evaluate_bins = evaluate_bins
        logger.info("-" * 50)
        logger.info("LogisticRegression instance initialized following parameters:")
        logger.info(f"\tModel directory set to: {self.model_dir}")
        logger.info(f"\tMetrics directory set to: {self.metrics_dir}")
        logger.info(f"\tDisplay convergence messages: {self.disp}")
        logger.info(f"\tOptimization method: {self.method}")
        logger.info(f"\tMaximum number of iterations: {self.maxiter}")
        logger.info(f"\tSave results to JSON: {self.save_results}")
        logger.info(f"\tSave fitted model: {self.save_model}")
        logger.info(f"\tEvaluate model after fitting: {self.perform_evaluation}")
        logger.info(f"\tNumber of bins for evaluation: {self.evaluate_bins}")
        logger.info("-" * 50)
    
    def fit(self, X, y):
        """
        Purpose
        -------
        Train Logistic Regression model using statsmodels Logit.
        Fits the model, extracts summaries, and saves results to output directory.

        Parameters
        ----------
        X : pd.DataFrame
            Features DataFrame containing independent variables.
        y : pd.Series
            Target Series containing binary dependent variable (0 or 1).

        Returns
        -------
        statsmodels.discrete.discrete_model.LogitResults
            Result object containing fitted model statistics, coefficients,
            and all model fit information.
        Raises
        ------
        ValueError
            If target variable is not binary (0 and 1) or if input data contains missing values.
            If X or y contains missing values, a ValueError is raised with details.
        Exception
            For any errors during model fitting, an exception is raised with details.
        """
        # Raise errors for invalid input data
        if not set(y.unique()).issubset({0, 1}):
            raise ValueError("Target must be binary with values 0 and 1.")

        # Raise error if X or y contains missing values
        if X.isnull().any().any():
            raise ValueError("X contains missing values.")

        # Raise error if y contains missing values
        if y.isnull().any():
            raise ValueError("y contains missing values.")
        
        try:
            logger.info("Starting Logistic Regression model training")
            logger.info(f"Input data shape - Features: {X.shape}, Target: {y.shape}")
            logger.info(f"Number of features: {X.shape[1]}")
            logger.info(f"Number of observations: {X.shape[0]}")
            logger.info(f"Target distribution - Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")
            
            # Add constant term to DataFrame for intercept
            X_with_const = sm.add_constant(X)
            logger.info(f"Added constant term to features. Constant column added with name: const")
            logger.info(f"Added Value: {X_with_const['const'].iloc[0]} for all observations")
            
            # Check for unique values in target variable
            unique_values = y.unique()
            logger.info(f"Unique values in target variable: {unique_values}")
            class_counts = y.value_counts().sort_index()
            class_pct = y.value_counts(normalize=True).sort_index() * 100
            distribution = " | ".join(
                f"Class {int(label)}: {int(count)} ({class_pct[label]:.2f}%)"
                for label, count in class_counts.items()
            )
            logger.info(f"Target variable distribution -> {distribution}")
            
            # Initialize the Logistic Regression model using Statsmodels
            self.model = sm.Logit(y, X_with_const)
            # Log equation for beautiful display
            formula_terms = "\n\t+ ".join(
                [f"(beta{i} * {col})" for i, col in enumerate(X.columns, start=1)]
            )
            logger.info(f"Logistic Regression model initialized with formula:\n\tlogit(P(Y=1)) = \u03b20\n\t+ {formula_terms}")
            
            # Fit the model
            logger.info("Fitting model...")
            # disp: Set to True to print convergence messages. Set to False to suppress output.
            # method: Optimization method to use. Default is 'bfgs' which is a quasi-Newton method. 
            # Other options include: newton, lbfgs, powell, cg, ncg, basinhopping, minimize.
            # maxiter: Maximum number of iterations for the optimization algorithm.
            # For details: https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit.html#statsmodels.discrete.discrete_model.Logit.fit
            self.model_fit = self.model.fit(disp=self.disp, method=self.method, maxiter=self.maxiter) 
            logger.info(f"Model training completed successfully")

            logger.info("Extracting model summary and metrics")
            summary = self.model_fit.summary()
            print(summary)

            # Get Model Info
            model_info = self.get_model_info(self.model_fit)

            # Get Coefficients
            coefficients = self.get_coefficients(self.model_fit)

            # Get Odds Ratios
            odds_ratios = self.get_odds_ratios(self.model_fit)

            # Get Marginal Effects
            marginal_effects = self.get_marginal_effects(self.model_fit, at="overall")

            # Combine all results into a single dictionary for saving
            results = {
                "modelInfo": model_info,
                "coefficients": coefficients,
                "oddsRatios": odds_ratios,
                "marginalEffects": marginal_effects
            }

            # Save results to JSON file in model directory
            if self.save_results:
                saved_path = io_save_json(results, f"{self.model_dir}/logit_model_results.json")
                logger.info(f"Model results saved to: {saved_path}")

            # Save fitted model to file for future use
            if self.save_model:
                model_file_path = f"{self.model_dir}/logit_model_fit.pkl"
                io_save_model(self.model_fit, model_file_path)
                logger.info(f"Fitted model saved to: {model_file_path}")
            
            # Evaluate model performance if requested
            if self.perform_evaluation:
                labels = self.predict(X)
                probs = self.predict_proba(X)
                actuals = y.values
                evaluation_metrics = self.evaluate_model(probs=probs, labels=labels, actuals=actuals, bins=self.evaluate_bins, output_dir=self.metrics_dir)

            return self.model_fit
        
        except Exception as e:
            error_message = f"Error during model training: {str(e)}"
            logger.error(error_message)
            raise Exception(error_message) from e
    
    @staticmethod
    def get_model_info(model_fit):
        """
        Extracts key model fit metrics from the fitted model results.

        Parameters
        ----------
        model_fit : statsmodels.discrete.discrete_model.LogitResults
            The fitted model results object returned by the fit() method.

        Returns
        -------
        dict
            A dictionary containing key model fit metrics such as:
            - AIC: Akaike Information Criterion
            - BIC: Bayesian Information Criterion
            - Pseudo R-squared: McFadden's R-squared
        """
        model_info = {
            "dependentVariable": str(model_fit.model.endog_names),
            "numberOfObservations": int(model_fit.nobs),
            "degreesOfFreedomResiduals": int(model_fit.df_resid),
            "degreesOfFreedomModel": int(model_fit.df_model),
            "pseudoRSquared": round(float(model_fit.prsquared), 6),
            "logLikelihood": round(float(model_fit.llf), 6),
            "nullLogLikelihood": round(float(model_fit.llnull), 6),
            "likelihoodRatioPValue": float(model_fit.llr_pvalue),
            "likelihoodRatioSignificance": interpret_p_value(model_fit.llr_pvalue).get("significanceLevel", "not available"),
            "converged": bool(model_fit.mle_retvals.get("converged", False)),
            "covarianceType": str(model_fit.cov_type),
            "aic": round(float(model_fit.aic), 6),
            "bic": round(float(model_fit.bic), 6)
            }
        logger.info("Extracted model metrics:")
        for key, value in model_info.items():
            logger.info(f"\t{key}: {value}")
        return model_info

    @staticmethod
    def get_coefficients(model_fit) -> dict:

        params = model_fit.params
        std_err = model_fit.bse
        # In statsmodels, LogitResults stores Wald z-statistics under the attribute
        # name `tvalues`. Although the attribute is called tvalues, the Logit summary
        # displays these values as z-statistics.
        z_stats = model_fit.tvalues
        p_values = model_fit.pvalues
        conf_int = model_fit.conf_int()

        coefficient_summary = {}

        for variable in params.index:

            coefficient_summary[variable] = {
                "coefficient": float(params[variable]),
                "standardError": float(std_err[variable]),
                "zStatistic": float(z_stats[variable]),
                "pValue": float(p_values[variable]),
                "significanceOfCoeff": interpret_p_value(p_values[variable]).get("significanceLevel", "not available"),
                "confidenceInterval95": {
                    "lower": float(conf_int.loc[variable, 0]),
                    "upper": float(conf_int.loc[variable, 1])
                }
            }
        
        sorted_features = sorted(
            params.drop("const").items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        logger.info("Top 25 strongest coefficients by absolute value:")

        for feature, coef in sorted_features[:25]:
            logger.info(f"\t{feature:<40} {coef:+.4f}")

        return coefficient_summary    

    @staticmethod
    def get_odds_ratios(model_fit) -> dict:
        """
        Extract odds ratios from fitted Logistic Regression model.

        Odds Ratio = exp(coefficient)

        Returns
        -------
        dict
            Dictionary containing odds ratio values for each variable.
        """
        # =============================================================================
        # ODDS RATIO INTERPRETATION
        # =============================================================================
        #
        # Definition
        # ----------
        # Odds Ratio (OR) measures the multiplicative change in the odds of the
        # positive class resulting from a one-unit increase in a predictor variable,
        # while holding all other variables constant.
        #
        # Odds Ratio is calculated as:
        #
        #     Odds Ratio = exp(coefficient)
        #
        # Interpretation
        # --------------
        # Odds Ratio > 1
        #     A one-unit increase in the variable increases the odds of the
        #     positive class.
        #
        # Odds Ratio < 1
        #     A one-unit increase in the variable decreases the odds of the
        #     positive class.
        #
        # Odds Ratio = 1
        #     No effect on the odds of the positive class.
        #
        # Example
        # -------
        # Assume:
        #
        #     Odds Ratio = 0.7836
        #
        # Interpretation:
        #
        #     A one-unit increase in the variable decreases the odds of the
        #     positive class by 21.64%.
        #
        # Example Using Probabilities
        # ---------------------------
        # Assume:
        #
        #     P(Y=1) = 90%
        #     P(Y=0) = 10%
        #
        # Initial Odds:
        #
        #     Odds = 0.90 / 0.10 = 9.0
        #
        # Apply Odds Ratio:
        #
        #     New Odds = 9.0 × 0.7836 = 7.0524
        #
        # Convert back to probability:
        #
        #     P(Y=1) = 7.0524 / (1 + 7.0524)
        #            = 87.58%
        #
        # Result:
        #
        #     Odds decrease by 21.64%
        #     Probability decreases from 90.00% to 87.58%
        #
        # Note
        # ----
        # Odds Ratio describes the relative change in odds, NOT the absolute change
        # in probability.
        #
        # Therefore:
        #
        #     Odds decrease by 21.64%
        #
        # does NOT mean:
        #
        #     Probability decreases by 21.64 percentage points.
        #
        # The relationship between odds and probability is non-linear.
        #
        # Common Interpretation Examples
        # ------------------------------
        #
        # Odds Ratio = 2.00
        #     Odds increase by 100% (double).
        #
        # Odds Ratio = 1.20
        #     Odds increase by 20%.
        #
        # Odds Ratio = 1.00
        #     No effect.
        #
        # Odds Ratio = 0.80
        #     Odds decrease by 20%.
        #
        # Odds Ratio = 0.50
        #     Odds decrease by 50% (half).
        #
        # Business Interpretation
        # -----------------------
        # Odds Ratios are useful for understanding the direction and relative strength
        # of a predictor's effect.
        #
        # For direct interpretation in probability terms (percentage-point change),
        # use Marginal Effects instead.
        # =============================================================================

        params = model_fit.params
        odds_ratios_exp = np.exp(params)

        odds_ratios = {}

        for variable in params.index:
            
            if variable == "const":
                continue  # Skip the constant term for odds ratio interpretation
            
            interpreation_dict =  interpret_odds_ratio(odds_ratios_exp[variable])

            odds_ratios[variable] = {
                "coefficient": float(params[variable]),
                "oddsRatio": float(odds_ratios_exp[variable]),
                "absOddsRatioChange": abs(odds_ratios_exp[variable] - 1),
                "direction": interpreation_dict.get("direction", "not available"),
                "percentChangeInOdds": interpreation_dict.get("percentChangeInOdds", 0.0),
                "magnitude": interpreation_dict.get("magnitude", "not available"),
                "interpretation": interpreation_dict.get("interpretation", "not available")
            }

        sorted_features = sorted(
            odds_ratios.items(),
            key=lambda x: x[1]["absOddsRatioChange"],
            reverse=True
        )

        logger.info("Top 25 strongest odds ratio effects:")

        for feature, values in sorted_features[:25]:
            logger.info(
                f"\t{feature:<40} "
                f"coef={values['coefficient']:+.4f} "
                f"oddsRatio={values['oddsRatio']:.4f} "
                f"absOddsRatioChange={values['absOddsRatioChange']:.4f} "
                f"direction={values['direction']} "
                f"percentChangeInOdds={values['percentChangeInOdds']:.2f}% "
                f"magnitude={values['magnitude']} "
                f"interpretation={values['interpretation']}"
            )

        return odds_ratios

    @staticmethod
    def get_marginal_effects(model_fit, at="overall") -> dict:
        """
        Extract marginal effects from fitted Logistic Regression model.

        Parameters
        ----------
        model_fit : statsmodels LogitResults
            Fitted statsmodels logistic regression result.

        at : str, optional
            Location where marginal effects are calculated.
            Default is "overall", which means Average Marginal Effects (AME).

        Returns
        -------
        dict
            Dictionary containing marginal effect statistics for each variable.
        """
        # =============================================================================
        # MARGINAL EFFECT INTERPRETATION
        # =============================================================================
        #
        # Definition
        # ----------
        # Marginal Effect measures the approximate change in the predicted probability
        # of the positive class resulting from a one-unit increase in a predictor
        # variable, while holding all other variables constant.
        #
        # Logistic Regression coefficients operate on the log-odds scale and Odds
        # Ratios operate on the odds scale. Marginal Effects convert the impact into
        # probability terms, making the result easier to interpret from a business
        # perspective.
        #
        # Interpretation
        # --------------
        # Marginal Effect > 0
        #     A one-unit increase in the variable increases the predicted probability
        #     of the positive class.
        #
        # Marginal Effect < 0
        #     A one-unit increase in the variable decreases the predicted probability
        #     of the positive class.
        #
        # Example
        # -------
        # Assume:
        #
        #     Marginal Effect = -0.0372
        #
        # Interpretation:
        #
        #     A one-unit increase in the variable decreases the predicted probability
        #     of the positive class by approximately 3.72 percentage points.
        #
        # If the baseline probability is:
        #
        #     P(Y=1) = 20.00%
        #
        # Then after a one-unit increase:
        #
        #     P(Y=1) ≈ 16.28%
        #
        # Note
        # ----
        # Marginal Effects represent changes in probability (percentage points),
        # not percentage changes.
        #
        # Example:
        #
        #     Marginal Effect = -0.05
        #
        # means:
        #
        #     Probability decreases by 5 percentage points
        #
        # not:
        #
        #     Probability decreases by 5%
        #
        # Average Marginal Effects (AME)
        # ------------------------------
        # Statsmodels get_margeff(at="overall") returns Average Marginal Effects (AME).
        #
        # AME is calculated by:
        #
        #     1. Computing the marginal effect for each observation.
        #     2. Averaging those effects across the entire dataset.
        #
        # Therefore, the reported value represents the average probability impact
        # across all observations in the sample.
        # =============================================================================

        marginal_effects = model_fit.get_margeff(at=at)

        feature_names = marginal_effects.results.model.exog_names

        # Statsmodels marginal effects do not include const
        feature_names = [f for f in feature_names if f != "const"]

        margeff = marginal_effects.margeff
        std_err = marginal_effects.margeff_se
        z_stats = marginal_effects.tvalues
        p_values = marginal_effects.pvalues
        conf_int = marginal_effects.conf_int()

        marginal_effects_dict = {}

        for variable, me, se, z, p, ci in zip(
            feature_names,
            margeff,
            std_err,
            z_stats,
            p_values,
            conf_int
        ):
            marginal_effects_dict[variable] = {
                "marginalEffect": float(me),
                "interpretationOfMarginalEffect": interpret_marginal_effect(me).get("interpretation", "not available"),
                "magnitude": interpret_marginal_effect(me).get("magnitude", "not available"),
                "direction": interpret_marginal_effect(me).get("direction", "not available"),
                "standardError": float(se),
                "zStatistic": float(z),
                "pValue": float(p),
                "significanceOfMarginalEffect": interpret_p_value(p).get("significanceLevel", "not available"),
                "confidenceInterval95": {
                    "lower": float(ci[0]),
                    "upper": float(ci[1])
                }
            }

        sorted_features = sorted(
            marginal_effects_dict.items(),
            key=lambda x: abs(x[1]["marginalEffect"]),
            reverse=True
        )

        logger.info(f"Marginal effects extracted with at='{at}'")

        logger.info("Top 25 strongest marginal effects by absolute value:")

        for feature, values in sorted_features[:25]:
            logger.info(
                f"{feature} --> "
                f"marginalEffect={values['marginalEffect']:+.6f} "
                f"pValue={values['pValue']:.6f}"
            )

        return marginal_effects_dict

    @staticmethod
    def evaluate_model(probs: list, labels: list, actuals: list, bins: int = 20, output_dir: str = "outputs/logitmodel/metrics") -> dict:
        """
        Evaluate model performance using various metrics.

        Parameters
        ----------
        model_fit : statsmodels LogitResults
            Fitted statsmodels logistic regression result.
        probs : list
            List of predicted probabilities for the positive class (1).
        labels : list
            List of predicted class labels (0 or 1) based on a threshold.
        actuals : list
            List of actual class labels (0 or 1) from the dataset.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics such as:
            - Accuracy
            - Precision
            - Recall
            - F1 Score
            - AUC-ROC
            - Gini Coefficient
            - KS Statistic
            - Decile Analysis
            - KS Plot
            - Lift Plot
            - Lorenz Curve Plot
            - ROC AUC Plot
            - Threshold Analysis
        """
        conf_matrix = calculate_confusion_matrix(actuals, labels)
        accuracy = calculate_accuracy(actuals, labels, confusion_matrix_dict=conf_matrix)
        precision = calculate_precision(actuals, labels, confusion_matrix_dict=conf_matrix)
        recall = calculate_recall(actuals, labels, confusion_matrix_dict=conf_matrix)
        f1_score = calculate_f1_score(actuals, labels, confusion_matrix_dict=conf_matrix)

        roc_auc = calculate_roc_auc(actuals, probs) # works with probabilities
        roc_auc_interpretation = interpret_roc_auc(roc_auc.get("rocAuc", 0.0))

        gini = calculate_gini(actuals, probs, roc_auc=roc_auc.get("rocAuc")) # works with probabilities
        gini_interpretation = interpret_gini(gini.get("gini", 0.0))

        ks = calculate_ks(actuals, probs) # works with probabilities
        ks_interpretation = interpret_ks(ks.get("ks", 0.0))

        decile_analysis = calculate_decile_analysis(actuals, probs, bins) # works with probabilities
        io_save_dataframe_as_csv(pd.DataFrame(decile_analysis.get("deciles")), f"{output_dir}/decile_analysis.csv")

        ks_fig = plot_ks(decile_analysis)
        ks_fig_saved_path = io_save_figure(ks_fig, f"{output_dir}/ks_plot.png")
        logger.info(f"KS plot saved to: {ks_fig_saved_path}")

        lift_fig = plot_lift(decile_analysis)
        lift_fig_saved_path = io_save_figure(lift_fig, f"{output_dir}/lift_plot.png")
        logger.info(f"Lift plot saved to: {lift_fig_saved_path}")

        lorenz_fig = plot_lorenz(decile_analysis)
        lorenz_fig_saved_path = io_save_figure(lorenz_fig, f"{output_dir}/lorenz_curve_plot.png")
        logger.info(f"Lorenz curve plot saved to: {lorenz_fig_saved_path}")

        roc_auc_fig = plot_roc_auc(actuals, probs)
        roc_auc_fig_saved_path = io_save_figure(roc_auc_fig, f"{output_dir}/roc_auc_plot.png")
        logger.info(f"ROC AUC plot saved to: {roc_auc_fig_saved_path}")

        threshold_analysis = calculate_threshold_analysis(actuals, probs, thresholds=[0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]) # works with probabilities
        io_save_dataframe_as_csv(pd.DataFrame(threshold_analysis.get("rows")), f"{output_dir}/threshold_analysis.csv")
        logger.info(f"Threshold analysis saved to: {output_dir}/threshold_analysis.csv")

        return {
            "confusionMatrix": conf_matrix,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1Score": f1_score,
            "rocAuc": roc_auc,
            "rocAucInterpretation": roc_auc_interpretation,
            "gini": gini,
            "giniInterpretation": gini_interpretation,
            "ks": ks,
            "ksInterpretation": ks_interpretation,
            "decileAnalysis": decile_analysis,
            "thresholdAnalysis": threshold_analysis,
            "savedPaths": {
                "ksPlot": ks_fig_saved_path,
                "liftPlot": lift_fig_saved_path,
                "lorenzCurvePlot": lorenz_fig_saved_path,
                "rocAucPlot": roc_auc_fig_saved_path,
                "decileAnalysisCsv": f"{output_dir}/decile_analysis.csv",
                "thresholdAnalysisCsv": f"{output_dir}/threshold_analysis.csv"
            }
        }

    def predict_proba(self, X: pd.DataFrame) -> list:
        """
        Predict probability of positive class.

        Parameters
        ----------
        X : pd.DataFrame
            Features DataFrame containing independent variables. 
            Must have same columns as training data (excluding target variable). 
            If constant term was added during training, it will be added automatically.

        Returns
        -------
        list
            List of predicted probabilities for the positive class (1).
            Example: [0.123, 0.456, 0.789] for three observations.
        """

        # Copy of input DataFrame to avoid modifying original
        X_temp = X.copy()

        if self.model_fit is None:
            raise ValueError("Model has not been fitted yet.")

        logger.info(f"Input Type: {type(X_temp).__name__}, Input Shape: {X_temp.shape}")

        # Add intercept if not exists
        if "const" not in X_temp.columns:
            logger.info("Adding constant column to prediction dataset")
            X_temp.insert(0, "const", 1.0)

        # Raise Error if const is not first column
        if X_temp.columns[0] != "const":
            raise ValueError("Constant column must be the first column in the DataFrame.")
        
        # Raise Error if columns do not match model features
        model_features = self.model_fit.model.exog_names
        if list(X_temp.columns) != model_features:
            # Log pairs of expected vs actual columns for debugging
            for expected, actual in zip(model_features, X_temp.columns):
                logger.debug(f"Expected column: {expected}, Actual column: {actual}, Match: {expected == actual}")
            error_message = (f"Input features do not match model features. "
                             f"Expected: {model_features}, Got: {list(X_temp.columns)}")
            raise ValueError(error_message)

        probabilities = self.model_fit.predict(X_temp).tolist()
        logger.info("Probabilities converted to list format for output")
        logger.info(f"Probabilities predicted for {len(probabilities)}\n"
                    f"Output Type: {type(probabilities).__name__}")

        logger.info("Probability prediction completed")

        return probabilities

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5
    ) -> list:
        """
        Predict binary class labels.

        Parameters
        ----------
        X : pd.DataFrame
            Features DataFrame containing independent variables.
            Must have same columns as training data
            (excluding target variable).

        threshold : float, optional
            Probability threshold used to convert probabilities
            into class labels.

            Example:
            - threshold = 0.50
            - probability >= 0.50 -> class 1
            - probability < 0.50 -> class 0

        Returns
        -------
        list
            Predicted class labels.

            Example:
            [0, 1, 0, 1]
        """

        logger.info(
            f"Starting class prediction "
            f"(threshold={threshold:.4f})"
        )

        # Validate threshold
        if not (0 <= threshold <= 1):
            error_message = (
                f"Invalid threshold: {threshold}. "
                f"Threshold must be between 0 and 1."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        # Get probabilities
        probabilities = self.predict_proba(X)

        # Convert probabilities to class labels
        predictions = [int(prob >= threshold) for prob in probabilities]

        class_0_count = predictions.count(0)
        class_1_count = predictions.count(1)

        logger.info(
            f"Class prediction completed\n"
            f"Total Observations: {len(predictions)}\n"
            f"Predicted Class 0: {class_0_count}\n"
            f"Predicted Class 1: {class_1_count}"
        )

        return predictions

if __name__ == "__main__":
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

    # Fitted Model
    log_reg = LogisticRegression(model_dir="model")
    fitted_model = log_reg.fit(X_train, y_train)

    # Get one sample from data to predict with fitted model
    X_sample = X_train.iloc[[0]]  

    # Make predictions with the fitted model to verify it works before testing persistence
    predicted_proba_with_fitted_model = log_reg.predict_proba(X_sample)
    predicted_class_with_fitted_model = log_reg.predict(X_sample)

    # Load the saved model and make predictions to verify persistence
    saved_model_path = "model/logit_model_fit.pkl"
    loaded_model_fit = io_load_model(saved_model_path)
    predicted_proba_with_loaded_model = log_reg.predict_proba(X_sample)
    predicted_class_with_loaded_model = log_reg.predict(X_sample)

    # Verify that predictions from fitted and loaded models are the same
    if predicted_class_with_loaded_model == predicted_class_with_fitted_model:
        logger.info("Predictions from fitted and loaded models match.")
    else:
        logger.error("Predictions from fitted and loaded models do NOT match.")

    if predicted_proba_with_loaded_model == predicted_proba_with_fitted_model:
        logger.info("Predicted probabilities from fitted and loaded models match.")
    else:
        logger.error("Predicted probabilities from fitted and loaded models do NOT match.")