"""Utilities for converting dataframes between long and wide formats."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from skore._sklearn.types import ReportType

FormatType = Literal["long", "wide", "auto"]


def _get_default_format(report_type: ReportType) -> Literal["long", "wide"]:
    """Get the default format based on report type.

    Parameters
    ----------
    report_type : ReportType
        The type of report.

    Returns
    -------
    format : {"long", "wide"}
        The default format for the report type.
    """
    if report_type in ("comparison-estimator", "comparison-cross-validation"):
        return "long"
    return "wide"


def _validate_format_for_report_type(
    format: FormatType,
    report_type: ReportType,
) -> Literal["long", "wide"]:
    """Validate and resolve the format parameter.

    Parameters
    ----------
    format : {"long", "wide", "auto"}
        The requested format.
    report_type : ReportType
        The type of report.

    Returns
    -------
    resolved_format : {"long", "wide"}
        The resolved format.

    Raises
    ------
    ValueError
        If format="wide" is requested for a comparison report.
    """
    if format == "auto":
        return _get_default_format(report_type)

    if format == "wide" and report_type in (
        "comparison-estimator",
        "comparison-cross-validation",
    ):
        raise ValueError(
            "Wide format is not supported for comparison reports because they contain "
            "data from multiple estimators with potentially different features. "
            "Use format='long' instead, or filter to a single estimator using the "
            "`query` parameter."
        )

    return format


def _apply_query_filter(
    df: pd.DataFrame,
    query: dict[str, Any] | None,
) -> pd.DataFrame:
    """Apply query filters to a dataframe.

    Parameters
    ----------
    df : DataFrame
        The dataframe to filter.
    query : dict or None
        Dictionary of column names to values for filtering.
        Example: {"label": "setosa", "split": 0}

    Returns
    -------
    DataFrame
        The filtered dataframe.
    """
    if query is None:
        return df

    filtered_df = df.copy()
    for column, value in query.items():
        if column not in filtered_df.columns:
            available_cols = [c for c in filtered_df.columns if not c.startswith("_")]
            raise ValueError(
                f"Column '{column}' not found in dataframe. "
                f"Available columns: {available_cols}"
            )
        filtered_df = filtered_df[filtered_df[column] == value]

    if filtered_df.empty:
        raise ValueError(
            f"Query {query} returned no results. Check that the filter values exist."
        )

    return filtered_df


def _aggregate_cv_splits(
    df: pd.DataFrame,
    value_column: str = "coefficients",
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate cross-validation splits by computing mean ± std.

    Parameters
    ----------
    df : DataFrame
        The dataframe with CV split data.
    value_column : str, default="coefficients"
        The column containing values to aggregate.
    group_columns : list of str or None
        Columns to group by. If None, auto-detect from available columns.

    Returns
    -------
    DataFrame
        Aggregated dataframe with mean and std columns.
    """
    if "split" not in df.columns:
        return df

    if group_columns is None:
        # Auto-detect grouping columns (everything except split and value)
        group_columns = [
            col
            for col in df.columns
            if col not in ("split", value_column) and not df[col].isna().all()
        ]

    if not group_columns:
        # No grouping columns, aggregate entire dataframe
        return pd.DataFrame(
            {
                f"{value_column}_mean": [df[value_column].mean()],
                f"{value_column}_std": [df[value_column].std()],
            }
        )

    aggregated = df.groupby(group_columns, as_index=False, sort=False).agg(
        **{
            f"{value_column}_mean": (value_column, "mean"),
            f"{value_column}_std": (value_column, "std"),
        }
    )

    return aggregated


def _convert_coefficients_to_wide(
    df: pd.DataFrame,
    report_type: ReportType,
    aggregate: bool = False,
) -> pd.DataFrame:
    """Convert coefficients dataframe from long to wide format.

    Parameters
    ----------
    df : DataFrame
        The coefficients dataframe in long format.
    report_type : ReportType
        The type of report.
    aggregate : bool, default=False
        Whether to aggregate CV splits (mean ± std).

    Returns
    -------
    DataFrame
        The coefficients in wide format.
    """
    # Determine which columns are present and meaningful
    has_label = "label" in df.columns and not df["label"].isna().all()
    has_output = "output" in df.columns and not df["output"].isna().all()
    has_split = "split" in df.columns and not df["split"].isna().all()
    has_estimator = "estimator" in df.columns

    # Handle aggregation for CV reports
    if aggregate and has_split:
        group_cols = ["feature"]
        if has_label:
            group_cols.append("label")
        if has_output:
            group_cols.append("output")
        if has_estimator:
            group_cols.append("estimator")

        df = _aggregate_cv_splits(df, value_column="coefficients", group_columns=group_cols)
        value_column = "coefficients_mean"
        # Add formatted column with mean ± std
        df["coefficients_formatted"] = df.apply(
            lambda row: f"{row['coefficients_mean']:.4f} ± {row['coefficients_std']:.4f}",
            axis=1,
        )
    else:
        value_column = "coefficients"

    # Build index and columns for pivot
    if report_type == "estimator":
        # Simple case: features as rows, labels/outputs as columns (if multiclass)
        if has_label:
            return df.pivot_table(
                index="feature",
                columns="label",
                values=value_column,
                aggfunc="first",
                sort=False,
            )
        elif has_output:
            return df.pivot_table(
                index="feature",
                columns="output",
                values=value_column,
                aggfunc="first",
                sort=False,
            )
        else:
            # Binary classification or single-output regression
            return df.set_index("feature")[[value_column]].rename(
                columns={value_column: "coefficient"}
            )

    elif report_type == "cross-validation":
        if aggregate:
            # Aggregated: features as rows, labels as columns
            if has_label:
                result = df.pivot_table(
                    index="feature",
                    columns="label",
                    values="coefficients_formatted",
                    aggfunc="first",
                    sort=False,
                )
            elif has_output:
                result = df.pivot_table(
                    index="feature",
                    columns="output",
                    values="coefficients_formatted",
                    aggfunc="first",
                    sort=False,
                )
            else:
                result = df.set_index("feature")[["coefficients_formatted"]].rename(
                    columns={"coefficients_formatted": "coefficient (mean ± std)"}
                )
            return result
        else:
            # Non-aggregated: features as rows, splits as columns
            if has_label:
                # Multi-class: hierarchical index (label, feature)
                df_sorted = df.sort_values(["label", "feature"])
                return df_sorted.pivot_table(
                    index=["label", "feature"],
                    columns="split",
                    values=value_column,
                    aggfunc="first",
                    sort=False,
                ).rename(columns=lambda x: f"Split #{int(x)}" if pd.notna(x) else x)
            elif has_output:
                df_sorted = df.sort_values(["output", "feature"])
                return df_sorted.pivot_table(
                    index=["output", "feature"],
                    columns="split",
                    values=value_column,
                    aggfunc="first",
                    sort=False,
                ).rename(columns=lambda x: f"Split #{int(x)}" if pd.notna(x) else x)
            else:
                # Binary/single-output: features as rows, splits as columns
                return df.pivot_table(
                    index="feature",
                    columns="split",
                    values=value_column,
                    aggfunc="first",
                    sort=False,
                ).rename(columns=lambda x: f"Split #{int(x)}" if pd.notna(x) else x)

    # For comparison reports, we shouldn't reach here due to validation
    raise ValueError(f"Wide format not supported for report type: {report_type}")
