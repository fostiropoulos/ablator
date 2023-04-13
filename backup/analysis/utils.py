from typing import Dict, Literal, Optional
import pandas as pd


def parse_numerical_table(
    numerical_corr_table: pd.DataFrame,
):
    if numerical_corr_table.shape[0] == 0:
        return None

    return numerical_corr_table


def parse_cat_table(
    cat_corr_table: pd.DataFrame,
):
    if cat_corr_table.shape[0] == 0:
        return None

    # cat_report = cat_corr_table.pivot(index="attr", columns="metric")
    # # , values=["best", "worst"]
    # # )
    _table = cat_corr_table.sort_values(["attr", "mean"]).reset_index(drop=True)

    return _table[
        [
            "attr",
            "attr_value",
            "metric",
            "mean",
            "std",
            "corr",
            "best_setting",
            "worst_setting",
        ]
    ]


def table_to_format(
    report: pd.DataFrame,
    is_categorical: bool,
    table_format: Literal["md", "latex"] = "latex",
):
    if is_categorical:
        _table = parse_cat_table(report)
    else:
        _table = parse_numerical_table(report)
    format_fn = "markdown" if table_format == "md" else "latex"
    return getattr(_table, f"to_{format_fn}")(index=False, escape=False)
