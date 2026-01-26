"""Tests for frame() format parameter across all report types and ML tasks."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils._testing import _convert_container

from skore import (
    ComparisonReport,
    CrossValidationReport,
    EstimatorReport,
)


class TestFrameFormatEstimatorReport:
    """Test frame() format parameter for EstimatorReport."""

    def test_binary_classification_wide_format(
        self, pyplot, logistic_binary_classification_with_train_test
    ):
        """Binary classification: wide format has feature index with coefficients."""
        estimator, X_train, X_test, y_train, y_test = (
            logistic_binary_classification_with_train_test
        )
        columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
        X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
        X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

        report = EstimatorReport(
            clone(estimator),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        display = report.feature_importance.coefficients()

        df_wide = display.frame(format="wide")
        assert df_wide.index.name == "feature"
        assert "coefficients" in df_wide.columns

        df_long = display.frame(format="long")
        assert "feature" in df_long.columns
        assert "coefficients" in df_long.columns

    def test_multiclass_classification_wide_format(
        self, pyplot, logistic_multiclass_classification_with_train_test
    ):
        """Multiclass classification: wide format has labels as columns."""
        estimator, X_train, X_test, y_train, y_test = (
            logistic_multiclass_classification_with_train_test
        )
        columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
        X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
        X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

        report = EstimatorReport(
            clone(estimator),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        display = report.feature_importance.coefficients()

        df_wide = display.frame(format="wide")
        assert df_wide.index.name == "feature"
        assert df_wide.columns.name == "label"

        df_long = display.frame(format="long")
        assert "feature" in df_long.columns
        assert "label" in df_long.columns
        assert "coefficients" in df_long.columns

    def test_single_output_regression_wide_format(
        self, pyplot, linear_regression_with_train_test
    ):
        """Single output regression: wide format returns feature index."""
        estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
        columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
        X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
        X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

        report = EstimatorReport(
            clone(estimator),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        display = report.feature_importance.coefficients()

        df_wide = display.frame(format="wide")
        assert df_wide.index.name == "feature"
        assert "coefficients" in df_wide.columns

        df_long = display.frame(format="long")
        assert "feature" in df_long.columns
        assert "coefficients" in df_long.columns

    def test_multi_output_regression_wide_format(
        self, pyplot, linear_regression_multioutput_with_train_test
    ):
        """Multi-output regression: wide format has outputs as columns."""
        estimator, X_train, X_test, y_train, y_test = (
            linear_regression_multioutput_with_train_test
        )
        columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
        X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
        X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

        report = EstimatorReport(
            clone(estimator),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        display = report.feature_importance.coefficients()

        df_wide = display.frame(format="wide")
        assert df_wide.index.name == "feature"
        assert df_wide.columns.name == "output"

        df_long = display.frame(format="long")
        assert "feature" in df_long.columns
        assert "output" in df_long.columns
        assert "coefficients" in df_long.columns


class TestFrameFormatCrossValidationReport:
    """Test frame() format parameter for CrossValidationReport."""

    def test_binary_classification_wide_format(
        self, pyplot, logistic_binary_classification_data
    ):
        """Binary classification CV: wide format has splits as columns."""
        estimator, X, y = logistic_binary_classification_data
        columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
        X = _convert_container(X, "dataframe", columns_name=columns_names)

        report = CrossValidationReport(clone(estimator), X, y, splitter=2)
        display = report.feature_importance.coefficients()

        df_wide = display.frame(format="wide")
        assert df_wide.index.name == "feature"
        assert df_wide.columns.name == "split"

        df_long = display.frame(format="long")
        assert "feature" in df_long.columns
        assert "split" in df_long.columns
        assert "coefficients" in df_long.columns

    def test_multiclass_classification_wide_format(
        self, pyplot, logistic_multiclass_classification_data
    ):
        """Multiclass classification CV: wide format has (label, split) columns."""
        estimator, X, y = logistic_multiclass_classification_data
        columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
        X = _convert_container(X, "dataframe", columns_name=columns_names)

        report = CrossValidationReport(clone(estimator), X, y, splitter=2)
        display = report.feature_importance.coefficients()

        df_wide = display.frame(format="wide")
        assert df_wide.index.name == "feature"
        assert isinstance(df_wide.columns, pd.MultiIndex)

        df_long = display.frame(format="long")
        assert "feature" in df_long.columns
        assert "label" in df_long.columns
        assert "split" in df_long.columns
        assert "coefficients" in df_long.columns

    def test_single_output_regression_wide_format(
        self, pyplot, linear_regression_data
    ):
        """Single output regression CV: wide format has splits as columns."""
        estimator, X, y = linear_regression_data
        columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
        X = _convert_container(X, "dataframe", columns_name=columns_names)

        report = CrossValidationReport(clone(estimator), X, y, splitter=2)
        display = report.feature_importance.coefficients()

        df_wide = display.frame(format="wide")
        assert df_wide.index.name == "feature"
        assert df_wide.columns.name == "split"

        df_long = display.frame(format="long")
        assert "feature" in df_long.columns
        assert "split" in df_long.columns
        assert "coefficients" in df_long.columns

    def test_multi_output_regression_wide_format(
        self, pyplot, linear_regression_multioutput_data
    ):
        """Multi-output regression CV: wide format has (output, split) columns."""
        estimator, X, y = linear_regression_multioutput_data
        columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
        X = _convert_container(X, "dataframe", columns_name=columns_names)

        report = CrossValidationReport(clone(estimator), X, y, splitter=2)
        display = report.feature_importance.coefficients()

        df_wide = display.frame(format="wide")
        assert df_wide.index.name == "feature"
        assert isinstance(df_wide.columns, pd.MultiIndex)

        df_long = display.frame(format="long")
        assert "feature" in df_long.columns
        assert "output" in df_long.columns
        assert "split" in df_long.columns
        assert "coefficients" in df_long.columns

    def test_aggregate_with_splits(self, pyplot, logistic_binary_classification_data):
        """Test aggregate parameter combines CV splits."""
        estimator, X, y = logistic_binary_classification_data
        columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
        X = _convert_container(X, "dataframe", columns_name=columns_names)

        report = CrossValidationReport(clone(estimator), X, y, splitter=3)
        display = report.feature_importance.coefficients()

        df_agg = display.frame(format="wide", aggregate=True)
        for val in df_agg["coefficients"]:
            assert "±" in str(val)


class TestFrameFormatComparisonReport:
    """Test frame() format parameter for ComparisonReport."""

    def test_comparison_estimator_binary_classification(
        self, pyplot, logistic_binary_classification_with_train_test
    ):
        """Comparison of EstimatorReports: wide format has estimators as columns."""
        estimator, X_train, X_test, y_train, y_test = (
            logistic_binary_classification_with_train_test
        )
        columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
        X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
        X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

        report1 = EstimatorReport(
            clone(estimator),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        report2 = EstimatorReport(
            clone(estimator),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        comparison = ComparisonReport({"model_1": report1, "model_2": report2})
        display = comparison.feature_importance.coefficients()

        df_wide = display.frame(format="wide")
        assert df_wide.index.name == "feature"
        assert df_wide.columns.name == "estimator"

        df_long = display.frame(format="long")
        assert "feature" in df_long.columns
        assert "estimator" in df_long.columns
        assert "coefficients" in df_long.columns

    def test_comparison_estimator_single_output_regression(
        self, pyplot, linear_regression_with_train_test
    ):
        """Comparison of regression EstimatorReports."""
        estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
        columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
        X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
        X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

        report1 = EstimatorReport(
            clone(estimator),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        report2 = EstimatorReport(
            clone(estimator),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        comparison = ComparisonReport({"model_1": report1, "model_2": report2})
        display = comparison.feature_importance.coefficients()

        df_wide = display.frame(format="wide")
        assert df_wide.index.name == "feature"
        assert df_wide.columns.name == "estimator"

        df_long = display.frame(format="long")
        assert "feature" in df_long.columns
        assert "estimator" in df_long.columns
        assert "coefficients" in df_long.columns

    def test_comparison_cross_validation_binary_classification(
        self, pyplot, logistic_binary_classification_data
    ):
        """Comparison of CrossValidationReports."""
        estimator, X, y = logistic_binary_classification_data
        columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
        X = _convert_container(X, "dataframe", columns_name=columns_names)

        report1 = CrossValidationReport(clone(estimator), X, y, splitter=2)
        report2 = CrossValidationReport(clone(estimator), X, y, splitter=2)
        comparison = ComparisonReport({"model_1": report1, "model_2": report2})
        display = comparison.feature_importance.coefficients()

        df_wide = display.frame(format="wide")
        assert df_wide.index.name == "feature"
        assert isinstance(df_wide.columns, pd.MultiIndex)

        df_long = display.frame(format="long")
        assert "feature" in df_long.columns
        assert "estimator" in df_long.columns
        assert "split" in df_long.columns
        assert "coefficients" in df_long.columns


class TestQueryParameter:
    """Test query parameter across all report types."""

    def test_query_estimator_report(
        self, pyplot, logistic_multiclass_classification_with_train_test
    ):
        """Test query parameter for EstimatorReport."""
        estimator, X_train, X_test, y_train, y_test = (
            logistic_multiclass_classification_with_train_test
        )
        columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
        X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
        X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

        report = EstimatorReport(
            clone(estimator),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        display = report.feature_importance.coefficients()

        labels = np.unique(y_train)
        target_label = int(labels[0])
        df = display.frame(format="long", query={"label": target_label})
        assert set(df["label"]) == {target_label}

    def test_query_cross_validation_report(
        self, pyplot, logistic_multiclass_classification_data
    ):
        """Test query parameter for CrossValidationReport."""
        estimator, X, y = logistic_multiclass_classification_data
        columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
        X = _convert_container(X, "dataframe", columns_name=columns_names)

        report = CrossValidationReport(clone(estimator), X, y, splitter=3)
        display = report.feature_importance.coefficients()

        df = display.frame(format="long", query={"split": 0})
        assert set(df["split"]) == {0}

        labels = np.unique(y)
        target_label = int(labels[0])
        df = display.frame(format="long", query={"label": target_label})
        assert set(df["label"]) == {target_label}

    def test_query_comparison_report(
        self, pyplot, logistic_binary_classification_with_train_test
    ):
        """Test query parameter for ComparisonReport."""
        estimator, X_train, X_test, y_train, y_test = (
            logistic_binary_classification_with_train_test
        )
        columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
        X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
        X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

        report1 = EstimatorReport(
            clone(estimator),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        report2 = EstimatorReport(
            clone(estimator),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        comparison = ComparisonReport({"model_1": report1, "model_2": report2})
        display = comparison.feature_importance.coefficients()

        df = display.frame(format="long", query={"estimator": "model_1"})
        assert set(df["estimator"]) == {"model_1"}

        df = display.frame(format="long", query={"feature": "Intercept"})
        assert set(df["feature"]) == {"Intercept"}
