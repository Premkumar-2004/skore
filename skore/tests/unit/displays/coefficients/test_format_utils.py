"""Tests for the format utilities in CoefficientsDisplay.frame()."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold

from skore import (
    ComparisonReport,
    CrossValidationReport,
    EstimatorReport,
    train_test_split,
)


@pytest.fixture
def iris_data():
    """Load iris dataset for classification tests."""
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    y = iris.target_names[y]
    return X, y


@pytest.fixture
def regression_data():
    """Create regression dataset."""
    X, y = make_regression(n_samples=100, n_features=4, random_state=42)
    X = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    return X, y


@pytest.fixture
def estimator_report(iris_data):
    """Create an EstimatorReport for testing."""
    X, y = iris_data
    split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True, shuffle=True)
    return EstimatorReport(LogisticRegression(), **split_data)


@pytest.fixture
def cv_report(iris_data):
    """Create a CrossValidationReport for testing."""
    X, y = iris_data
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    return CrossValidationReport(LogisticRegression(), X=X, y=y, splitter=cv)


@pytest.fixture
def comparison_report(iris_data):
    """Create a ComparisonReport for testing."""
    from sklearn.linear_model import RidgeClassifier

    X, y = iris_data
    split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True, shuffle=True)
    report1 = EstimatorReport(LogisticRegression(), **split_data)
    report2 = EstimatorReport(RidgeClassifier(), **split_data)
    return ComparisonReport([report1, report2])


class TestEstimatorReportFormat:
    """Tests for EstimatorReport.feature_importance.coefficients().frame() format."""

    def test_default_format_is_long(self, estimator_report):
        """Test that default format is long (backward compatible)."""
        display = estimator_report.feature_importance.coefficients()
        df = display.frame()
        # Long format has 'feature', 'label', 'coefficients' columns
        assert "feature" in df.columns
        assert "coefficients" in df.columns
        assert df.index.name != "feature"  # Not pivoted

    def test_long_format_explicit(self, estimator_report):
        """Test explicit long format."""
        display = estimator_report.feature_importance.coefficients()
        df = display.frame(format="long")
        assert "feature" in df.columns
        assert "coefficients" in df.columns

    def test_wide_format(self, estimator_report):
        """Test wide format pivots data correctly."""
        display = estimator_report.feature_importance.coefficients()
        df = display.frame(format="wide")
        # Wide format has features as index, labels as columns
        assert df.index.name == "feature"
        assert "setosa" in df.columns
        assert "versicolor" in df.columns
        assert "virginica" in df.columns

    def test_auto_format_resolves_to_wide(self, estimator_report):
        """Test that auto format resolves to wide for EstimatorReport."""
        display = estimator_report.feature_importance.coefficients()
        df_auto = display.frame(format="auto")
        df_wide = display.frame(format="wide")
        pd.testing.assert_frame_equal(df_auto, df_wide)

    def test_query_filter(self, estimator_report):
        """Test query filter works correctly."""
        display = estimator_report.feature_importance.coefficients()
        df = display.frame(format="wide", query={"label": "setosa"})
        assert df.columns.tolist() == ["setosa"]

    def test_query_filter_invalid_column(self, estimator_report):
        """Test query filter raises error for invalid column."""
        display = estimator_report.feature_importance.coefficients()
        with pytest.raises(ValueError, match="Column 'invalid' not found"):
            display.frame(query={"invalid": "value"})

    def test_query_filter_no_results(self, estimator_report):
        """Test query filter raises error when no results match."""
        display = estimator_report.feature_importance.coefficients()
        with pytest.raises(ValueError, match="returned no results"):
            display.frame(query={"label": "nonexistent"})

    def test_include_intercept_false(self, estimator_report):
        """Test include_intercept=False removes intercept."""
        display = estimator_report.feature_importance.coefficients()
        df = display.frame(format="wide", include_intercept=False)
        assert "Intercept" not in df.index


class TestCrossValidationReportFormat:
    """Tests for CrossValidationReport.feature_importance.coefficients().frame()."""

    def test_default_format_is_long(self, cv_report):
        """Test that default format is long (backward compatible)."""
        display = cv_report.feature_importance.coefficients()
        df = display.frame()
        assert "split" in df.columns
        assert "feature" in df.columns
        assert "coefficients" in df.columns

    def test_wide_format_without_aggregation(self, cv_report):
        """Test wide format shows splits as columns."""
        display = cv_report.feature_importance.coefficients()
        df = display.frame(format="wide")
        # Should have hierarchical index (label, feature) and split columns
        assert isinstance(df.index, pd.MultiIndex)
        assert "Split #0" in df.columns
        assert "Split #1" in df.columns
        assert "Split #2" in df.columns

    def test_wide_format_with_aggregation(self, cv_report):
        """Test wide format with aggregation shows mean ± std."""
        display = cv_report.feature_importance.coefficients()
        df = display.frame(format="wide", aggregate=True)
        # Aggregated format should have labels as columns
        assert "setosa" in df.columns
        assert "versicolor" in df.columns
        assert "virginica" in df.columns
        # Values should be strings with ± format
        sample_value = df.iloc[0, 0]
        assert isinstance(sample_value, str)
        assert "±" in sample_value


class TestComparisonReportFormat:
    """Tests for ComparisonReport.feature_importance.coefficients().frame()."""

    def test_default_format_is_long(self, comparison_report):
        """Test that default format is long."""
        display = comparison_report.feature_importance.coefficients()
        df = display.frame()
        assert "estimator" in df.columns
        assert "feature" in df.columns
        assert "coefficients" in df.columns

    def test_auto_format_resolves_to_long(self, comparison_report):
        """Test that auto format resolves to long for ComparisonReport."""
        display = comparison_report.feature_importance.coefficients()
        df_auto = display.frame(format="auto")
        df_long = display.frame(format="long")
        pd.testing.assert_frame_equal(df_auto, df_long)

    def test_wide_format_raises_error(self, comparison_report):
        """Test that wide format raises ValueError for ComparisonReport."""
        display = comparison_report.feature_importance.coefficients()
        with pytest.raises(ValueError, match="Wide format is not supported"):
            display.frame(format="wide")


class TestRegressionFormat:
    """Tests for regression problems (no label column)."""

    def test_estimator_report_regression_wide(self, regression_data):
        """Test wide format for regression EstimatorReport."""
        X, y = regression_data
        split_data = train_test_split(
            X=X, y=y, random_state=0, as_dict=True, shuffle=True
        )
        report = EstimatorReport(LinearRegression(), **split_data)
        display = report.feature_importance.coefficients()

        df = display.frame(format="wide")
        assert df.index.name == "feature"
        # Regression has single coefficient column
        assert "coefficient" in df.columns or len(df.columns) == 1
