import io

import numpy as np
import pandas as pd
import streamlit as st

from forecasting_engine import (
    build_assumption_table,
    build_forecast,
    derive_default_assumptions,
)

from scenario_analysis import (
    build_all_scenarios,
    build_final_year_summary,
    build_scenario_assumptions,
    build_sensitivity_table,
)

from management_review import (
    build_forecast_challenges,
    build_management_questions,
    build_review_points,
)


# ============================================================
# REPORT THEME
# ============================================================

DARK_BROWN = "#6E4A3A"
MEDIUM_BROWN = "#8B6A58"
BEIGE = "#F6F0E7"
TAUPE = "#E8DED3"
WHITE = "#FFFFFF"
BODY_TEXT = "#2F2926"
LIGHT_BORDER = "#D7C7B8"


# ============================================================
# DISPLAY NAMES
# ============================================================

DISPLAY_NAMES = {
    "year": "Year",
    "revenue": "Revenue",
    "cost_of_sales": "Cost of Sales",
    "gross_profit": "Gross Profit",
    "operating_expenses": "Operating Expenses",
    "operating_profit": "Operating Profit",
    "interest_expense": "Interest Expense",
    "profit_before_tax": "Profit Before Tax",
    "tax_expense": "Tax Expense",
    "net_profit": "Net Profit",
    "cash": "Cash",
    "trade_receivables": "Trade Receivables",
    "inventory": "Inventory",
    "other_current_assets": "Other Current Assets",
    "current_assets": "Current Assets",
    "property_plant_equipment": "Property, Plant & Equipment",
    "total_assets": "Total Assets",
    "trade_payables": "Trade Payables",
    "other_current_liabilities": "Other Current Liabilities",
    "current_liabilities": "Current Liabilities",
    "long_term_debt": "Long-Term Debt",
    "total_liabilities": "Total Liabilities",
    "equity": "Equity",
    "working_capital": "Working Capital",
    "operating_cash_flow": "Operating Cash Flow",
    "capital_expenditure": "Capital Expenditure",
    "depreciation": "Depreciation / Runoff",
    "change_in_working_capital": "Change in Working Capital",
    "dividends": "Dividends",
    "free_cash_flow": "Free Cash Flow",
    "gross_margin": "Gross Margin",
    "operating_margin": "Operating Margin",
    "net_margin": "Net Margin",
    "current_ratio": "Current Ratio",
    "quick_ratio": "Quick Ratio",
    "debt_to_equity": "Debt to Equity",
    "interest_coverage": "Interest Coverage",
    "return_on_assets": "Return on Assets",
    "return_on_equity": "Return on Equity",
    "receivable_days": "Receivable Days",
    "inventory_days": "Inventory Days",
    "payable_days": "Payable Days",
    "cash_conversion_cycle": "Cash Conversion Cycle",
}


CURRENCY_COLUMNS = {
    "Revenue",
    "Cost of Sales",
    "Gross Profit",
    "Operating Expenses",
    "Operating Profit",
    "Interest Expense",
    "Profit Before Tax",
    "Tax Expense",
    "Net Profit",
    "Cash",
    "Trade Receivables",
    "Inventory",
    "Other Current Assets",
    "Current Assets",
    "Property, Plant & Equipment",
    "Total Assets",
    "Trade Payables",
    "Other Current Liabilities",
    "Current Liabilities",
    "Long-Term Debt",
    "Total Liabilities",
    "Equity",
    "Working Capital",
    "Operating Cash Flow",
    "Capital Expenditure",
    "Depreciation / Runoff",
    "Change in Working Capital",
    "Dividends",
    "Free Cash Flow",
    "Previous Value",
    "Current Value",
    "£ Movement",
    "Closing Cash",
    "Maximum Difference",
}


PERCENTAGE_COLUMNS = {
    "% Movement",
    "% of Revenue",
    "% of Total Assets",
    "Gross Margin",
    "Operating Margin",
    "Net Margin",
    "Return on Assets",
    "Return on Equity",
    "Revenue Growth",
}


RATIO_COLUMNS = {
    "Current Ratio",
    "Quick Ratio",
    "Debt to Equity",
    "Interest Coverage",
}


DAY_COLUMNS = {
    "Receivable Days",
    "Inventory Days",
    "Payable Days",
    "Cash Conversion Cycle",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_divide(
    numerator,
    denominator,
):
    if (
        pd.isna(numerator)
        or pd.isna(denominator)
        or denominator == 0
    ):
        return np.nan

    return numerator / denominator


def get_state(
    key,
    default,
):
    """
    Read a Streamlit widget value where available.
    """

    return st.session_state.get(
        key,
        default,
    )


# ============================================================
# CURRENT FORECAST SETTINGS
# ============================================================

def get_current_forecast_settings(
    analysis_df,
):
    defaults = (
        derive_default_assumptions(
            analysis_df
        )
    )

    assumptions = (
        defaults.copy()
    )

    assumptions[
        "revenue_growth"
    ] = (
        float(
            get_state(
                "forecast_revenue_growth",
                defaults[
                    "revenue_growth"
                ]
                * 100,
            )
        )
        / 100
    )

    assumptions[
        "gross_margin"
    ] = (
        float(
            get_state(
                "forecast_gross_margin",
                defaults[
                    "gross_margin"
                ]
                * 100,
            )
        )
        / 100
    )

    assumptions[
        "operating_expense_ratio"
    ] = (
        float(
            get_state(
                "forecast_opex_ratio",
                defaults[
                    "operating_expense_ratio"
                ]
                * 100,
            )
        )
        / 100
    )

    assumptions[
        "interest_growth"
    ] = (
        float(
            get_state(
                "forecast_interest_growth",
                0.0,
            )
        )
        / 100
    )

    assumptions[
        "tax_rate"
    ] = (
        float(
            get_state(
                "forecast_tax_rate",
                defaults[
                    "tax_rate"
                ]
                * 100,
            )
        )
        / 100
    )

    assumptions[
        "receivable_days"
    ] = float(
        get_state(
            "forecast_receivable_days",
            defaults[
                "receivable_days"
            ],
        )
    )

    assumptions[
        "inventory_days"
    ] = float(
        get_state(
            "forecast_inventory_days",
            defaults[
                "inventory_days"
            ],
        )
    )

    assumptions[
        "payable_days"
    ] = float(
        get_state(
            "forecast_payable_days",
            defaults[
                "payable_days"
            ],
        )
    )

    assumptions[
        "other_current_asset_ratio"
    ] = (
        float(
            get_state(
                "forecast_other_ca",
                defaults[
                    "other_current_asset_ratio"
                ]
                * 100,
            )
        )
        / 100
    )

    assumptions[
        "other_current_liability_ratio"
    ] = (
        float(
            get_state(
                "forecast_other_cl",
                defaults[
                    "other_current_liability_ratio"
                ]
                * 100,
            )
        )
        / 100
    )

    assumptions[
        "capex_ratio"
    ] = (
        float(
            get_state(
                "forecast_capex",
                defaults[
                    "capex_ratio"
                ]
                * 100,
            )
        )
        / 100
    )

    assumptions[
        "depreciation_rate"
    ] = (
        float(
            get_state(
                "forecast_depreciation",
                10.0,
            )
        )
        / 100
    )

    assumptions[
        "long_term_debt_growth"
    ] = (
        float(
            get_state(
                "forecast_debt_growth",
                0.0,
            )
        )
        / 100
    )

    assumptions[
        "dividend_payout_ratio"
    ] = (
        float(
            get_state(
                "forecast_dividend_payout",
                0.0,
            )
        )
        / 100
    )

    years = int(
        get_state(
            "forecast_years",
            3,
        )
    )

    return (
        assumptions,
        years,
    )


# ============================================================
# CURRENT SCENARIO SETTINGS
# ============================================================

def get_current_scenario_settings(
    analysis_df,
):
    defaults = (
        derive_default_assumptions(
            analysis_df
        )
    )

    base_inputs = {
        "revenue_growth": (
            float(
                get_state(
                    "scenario_base_revenue",
                    defaults[
                        "revenue_growth"
                    ]
                    * 100,
                )
            )
            / 100
        ),

        "gross_margin": (
            float(
                get_state(
                    "scenario_base_margin",
                    defaults[
                        "gross_margin"
                    ]
                    * 100,
                )
            )
            / 100
        ),

        "operating_expense_ratio": (
            float(
                get_state(
                    "scenario_base_opex",
                    defaults[
                        "operating_expense_ratio"
                    ]
                    * 100,
                )
            )
            / 100
        ),

        "receivable_days": float(
            get_state(
                "scenario_base_receivables",
                defaults[
                    "receivable_days"
                ],
            )
        ),

        "inventory_days": float(
            get_state(
                "scenario_base_inventory",
                defaults[
                    "inventory_days"
                ],
            )
        ),

        "payable_days": float(
            get_state(
                "scenario_base_payables",
                defaults[
                    "payable_days"
                ],
            )
        ),
    }

    upside_adjustments = {
        "revenue_growth": (
            float(
                get_state(
                    "scenario_up_revenue",
                    3.0,
                )
            )
            / 100
        ),

        "gross_margin": (
            float(
                get_state(
                    "scenario_up_margin",
                    1.5,
                )
            )
            / 100
        ),

        "operating_expense_ratio": (
            float(
                get_state(
                    "scenario_up_opex",
                    -1.0,
                )
            )
            / 100
        ),

        "receivable_days": float(
            get_state(
                "scenario_up_receivables",
                -5.0,
            )
        ),

        "inventory_days": float(
            get_state(
                "scenario_up_inventory",
                -5.0,
            )
        ),

        "payable_days": float(
            get_state(
                "scenario_up_payables",
                0.0,
            )
        ),
    }

    downside_adjustments = {
        "revenue_growth": (
            float(
                get_state(
                    "scenario_down_revenue",
                    -5.0,
                )
            )
            / 100
        ),

        "gross_margin": (
            float(
                get_state(
                    "scenario_down_margin",
                    -2.5,
                )
            )
            / 100
        ),

        "operating_expense_ratio": (
            float(
                get_state(
                    "scenario_down_opex",
                    1.5,
                )
            )
            / 100
        ),

        "receivable_days": float(
            get_state(
                "scenario_down_receivables",
                10.0,
            )
        ),

        "inventory_days": float(
            get_state(
                "scenario_down_inventory",
                10.0,
            )
        ),

        "payable_days": float(
            get_state(
                "scenario_down_payables",
                0.0,
            )
        ),
    }

    assumptions = (
        build_scenario_assumptions(
            defaults,
            base_inputs,
            upside_adjustments,
            downside_adjustments,
        )
    )

    years = int(
        get_state(
            "scenario_forecast_horizon",
            3,
        )
    )

    return (
        assumptions,
        years,
    )


# ============================================================
# CURRENT REVIEW SETTINGS
# ============================================================

def get_current_review_settings():
    movement_threshold = (
        float(
            get_state(
                "management_movement_threshold",
                10.0,
            )
        )
        / 100
    )

    current_ratio = float(
        get_state(
            "management_current_ratio",
            1.20,
        )
    )

    quick_ratio = float(
        get_state(
            "management_quick_ratio",
            1.00,
        )
    )

    debt_ratio = float(
        get_state(
            "management_debt_ratio",
            1.50,
        )
    )

    interest_cover = float(
        get_state(
            "management_interest_cover",
            3.00,
        )
    )

    return {
        "movement_threshold": (
            movement_threshold
        ),
        "current_ratio": (
            current_ratio
        ),
        "quick_ratio": (
            quick_ratio
        ),
        "debt_ratio": (
            debt_ratio
        ),
        "interest_cover": (
            interest_cover
        ),
    }


# ============================================================
# HISTORICAL TABLES
# ============================================================

def historical_income_statement(
    analysis_df,
):
    columns = [
        "year",
        "revenue",
        "cost_of_sales",
        "gross_profit",
        "operating_expenses",
        "operating_profit",
        "interest_expense",
        "profit_before_tax",
        "tax_expense",
        "net_profit",
    ]

    return (
        analysis_df[
            columns
        ]
        .copy()
        .rename(
            columns=DISPLAY_NAMES
        )
    )


def historical_balance_sheet(
    analysis_df,
):
    columns = [
        "year",
        "cash",
        "trade_receivables",
        "inventory",
        "current_assets",
        "property_plant_equipment",
        "total_assets",
        "trade_payables",
        "current_liabilities",
        "long_term_debt",
        "total_liabilities",
        "equity",
        "working_capital",
    ]

    return (
        analysis_df[
            columns
        ]
        .copy()
        .rename(
            columns=DISPLAY_NAMES
        )
    )


def historical_cash_flow(
    analysis_df,
):
    columns = [
        "year",
        "operating_cash_flow",
        "capital_expenditure",
        "free_cash_flow",
    ]

    return (
        analysis_df[
            columns
        ]
        .copy()
        .rename(
            columns=DISPLAY_NAMES
        )
    )


# ============================================================
# HORIZONTAL ANALYSIS
# ============================================================

def build_horizontal_analysis(
    analysis_df,
):
    columns = [
        "revenue",
        "cost_of_sales",
        "gross_profit",
        "operating_expenses",
        "operating_profit",
        "interest_expense",
        "profit_before_tax",
        "tax_expense",
        "net_profit",
        "cash",
        "trade_receivables",
        "inventory",
        "current_assets",
        "total_assets",
        "trade_payables",
        "current_liabilities",
        "long_term_debt",
        "total_liabilities",
        "equity",
        "operating_cash_flow",
        "capital_expenditure",
        "free_cash_flow",
    ]

    rows = []

    for column in columns:

        for index in range(
            1,
            len(
                analysis_df
            ),
        ):
            current = (
                analysis_df.iloc[
                    index
                ]
            )

            previous = (
                analysis_df.iloc[
                    index - 1
                ]
            )

            change = (
                current[column]
                - previous[column]
            )

            percentage = (
                safe_divide(
                    change,
                    abs(
                        previous[column]
                    ),
                )
            )

            rows.append(
                {
                    "Financial Statement Line": (
                        DISPLAY_NAMES[
                            column
                        ]
                    ),

                    "From Year": int(
                        previous[
                            "year"
                        ]
                    ),

                    "To Year": int(
                        current[
                            "year"
                        ]
                    ),

                    "Previous Value": (
                        previous[
                            column
                        ]
                    ),

                    "Current Value": (
                        current[
                            column
                        ]
                    ),

                    "£ Movement": (
                        change
                    ),

                    "% Movement": (
                        percentage
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# COMMON-SIZE ANALYSIS
# ============================================================

def build_common_size_analysis(
    analysis_df,
):
    rows = []

    income_columns = [
        "revenue",
        "cost_of_sales",
        "gross_profit",
        "operating_expenses",
        "operating_profit",
        "interest_expense",
        "profit_before_tax",
        "tax_expense",
        "net_profit",
    ]

    for _, row in (
        analysis_df.iterrows()
    ):
        for column in (
            income_columns
        ):
            rows.append(
                {
                    "Statement": (
                        "Income Statement"
                    ),
                    "Year": int(
                        row["year"]
                    ),
                    "Line Item": (
                        DISPLAY_NAMES[
                            column
                        ]
                    ),
                    "Amount": (
                        row[column]
                    ),
                    "% of Revenue": (
                        safe_divide(
                            row[column],
                            row["revenue"],
                        )
                    ),
                    "% of Total Assets": (
                        np.nan
                    ),
                }
            )

    balance_columns = [
        "cash",
        "trade_receivables",
        "inventory",
        "current_assets",
        "property_plant_equipment",
        "total_assets",
        "trade_payables",
        "current_liabilities",
        "long_term_debt",
        "total_liabilities",
        "equity",
    ]

    for _, row in (
        analysis_df.iterrows()
    ):
        for column in (
            balance_columns
        ):
            rows.append(
                {
                    "Statement": (
                        "Balance Sheet"
                    ),
                    "Year": int(
                        row["year"]
                    ),
                    "Line Item": (
                        DISPLAY_NAMES[
                            column
                        ]
                    ),
                    "Amount": (
                        row[column]
                    ),
                    "% of Revenue": (
                        np.nan
                    ),
                    "% of Total Assets": (
                        safe_divide(
                            row[column],
                            row[
                                "total_assets"
                            ],
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# RATIO TABLE
# ============================================================

def build_ratio_table(
    analysis_df,
):
    columns = [
        "year",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "current_ratio",
        "quick_ratio",
        "debt_to_equity",
        "interest_coverage",
        "return_on_assets",
        "return_on_equity",
        "receivable_days",
        "inventory_days",
        "payable_days",
        "cash_conversion_cycle",
    ]

    return (
        analysis_df[
            columns
        ]
        .copy()
        .rename(
            columns=DISPLAY_NAMES
        )
    )


# ============================================================
# FORECAST TABLES
# ============================================================

def forecast_income_statement(
    forecast_df,
):
    columns = [
        "year",
        "revenue",
        "cost_of_sales",
        "gross_profit",
        "operating_expenses",
        "operating_profit",
        "interest_expense",
        "profit_before_tax",
        "tax_expense",
        "net_profit",
    ]

    return (
        forecast_df[
            columns
        ]
        .copy()
        .rename(
            columns=DISPLAY_NAMES
        )
    )


def forecast_balance_sheet(
    forecast_df,
):
    columns = [
        "year",
        "cash",
        "trade_receivables",
        "inventory",
        "other_current_assets",
        "current_assets",
        "property_plant_equipment",
        "total_assets",
        "trade_payables",
        "other_current_liabilities",
        "current_liabilities",
        "long_term_debt",
        "total_liabilities",
        "equity",
        "working_capital",
    ]

    return (
        forecast_df[
            columns
        ]
        .copy()
        .rename(
            columns=DISPLAY_NAMES
        )
    )


def forecast_cash_flow(
    forecast_df,
):
    columns = [
        "year",
        "net_profit",
        "depreciation",
        "change_in_working_capital",
        "operating_cash_flow",
        "capital_expenditure",
        "free_cash_flow",
        "dividends",
    ]

    return (
        forecast_df[
            columns
        ]
        .copy()
        .rename(
            columns=DISPLAY_NAMES
        )
    )


def forecast_ratio_table(
    forecast_df,
):
    columns = [
        "year",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "current_ratio",
        "quick_ratio",
        "debt_to_equity",
        "interest_coverage",
        "return_on_assets",
        "return_on_equity",
        "receivable_days",
        "inventory_days",
        "payable_days",
        "cash_conversion_cycle",
    ]

    return (
        forecast_df[
            columns
        ]
        .copy()
        .rename(
            columns=DISPLAY_NAMES
        )
    )


# ============================================================
# MODEL CHECKS
# ============================================================

def build_model_checks(
    analysis_df,
    forecast_df,
    scenario_forecasts,
):
    rows = []

    for _, row in (
        analysis_df.iterrows()
    ):
        difference = (
            row["total_assets"]
            - row["total_liabilities"]
            - row["equity"]
        )

        rows.append(
            {
                "Model": (
                    "Historical"
                ),
                "Period / Scenario": int(
                    row["year"]
                ),
                "Maximum Difference": (
                    difference
                ),
                "Status": (
                    "Passed"
                    if abs(
                        difference
                    ) <= 1
                    else "Review"
                ),
            }
        )

    for _, row in (
        forecast_df.iterrows()
    ):
        difference = (
            row[
                "balance_check"
            ]
        )

        rows.append(
            {
                "Model": (
                    "Forecast"
                ),
                "Period / Scenario": int(
                    row["year"]
                ),
                "Maximum Difference": (
                    difference
                ),
                "Status": (
                    "Passed"
                    if abs(
                        difference
                    ) <= 1
                    else "Review"
                ),
            }
        )

    for (
        scenario_name,
        scenario_df,
    ) in scenario_forecasts.items():

        maximum_difference = float(
            scenario_df[
                "balance_check"
            ]
            .abs()
            .max()
        )

        rows.append(
            {
                "Model": (
                    "Scenario"
                ),
                "Period / Scenario": (
                    scenario_name
                ),
                "Maximum Difference": (
                    maximum_difference
                ),
                "Status": (
                    "Passed"
                    if maximum_difference <= 1
                    else "Review"
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# EXCEL FORMATS
# ============================================================

def create_formats(
    workbook,
):
    formats = {}

    formats[
        "title"
    ] = workbook.add_format(
        {
            "bold": True,
            "font_size": 18,
            "font_color": DARK_BROWN,
            "align": "left",
            "valign": "vcenter",
        }
    )

    formats[
        "subtitle"
    ] = workbook.add_format(
        {
            "font_size": 10,
            "font_color": MEDIUM_BROWN,
            "italic": True,
            "align": "left",
            "valign": "vcenter",
        }
    )

    formats[
        "section"
    ] = workbook.add_format(
        {
            "bold": True,
            "font_size": 11,
            "font_color": WHITE,
            "bg_color": DARK_BROWN,
            "align": "left",
            "valign": "vcenter",
        }
    )

    formats[
        "header"
    ] = workbook.add_format(
        {
            "bold": True,
            "font_color": WHITE,
            "bg_color": DARK_BROWN,
            "border": 1,
            "border_color": LIGHT_BORDER,
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
        }
    )

    formats[
        "body"
    ] = workbook.add_format(
        {
            "font_color": BODY_TEXT,
            "border": 1,
            "border_color": LIGHT_BORDER,
            "valign": "top",
        }
    )

    formats[
        "alt_fill"
    ] = workbook.add_format(
        {
            "bg_color": BEIGE,
        }
    )

    formats[
        "currency"
    ] = workbook.add_format(
        {
            "font_color": BODY_TEXT,
            "border": 1,
            "border_color": LIGHT_BORDER,
            "num_format": '£#,##0.00;[Red]-£#,##0.00',
        }
    )

    formats[
        "percentage"
    ] = workbook.add_format(
        {
            "font_color": BODY_TEXT,
            "border": 1,
            "border_color": LIGHT_BORDER,
            "num_format": "0.0%;[Red]-0.0%",
        }
    )

    formats[
        "ratio"
    ] = workbook.add_format(
        {
            "font_color": BODY_TEXT,
            "border": 1,
            "border_color": LIGHT_BORDER,
            "num_format": '0.00"x"',
        }
    )

    formats[
        "days"
    ] = workbook.add_format(
        {
            "font_color": BODY_TEXT,
            "border": 1,
            "border_color": LIGHT_BORDER,
            "num_format": '0.0 "days"',
        }
    )

    formats[
        "integer"
    ] = workbook.add_format(
        {
            "font_color": BODY_TEXT,
            "border": 1,
            "border_color": LIGHT_BORDER,
            "num_format": "0",
        }
    )

    formats[
        "metric_label"
    ] = workbook.add_format(
        {
            "bold": True,
            "font_color": DARK_BROWN,
            "bg_color": BEIGE,
            "border": 1,
            "border_color": LIGHT_BORDER,
            "align": "center",
            "valign": "vcenter",
        }
    )

    formats[
        "metric_currency"
    ] = workbook.add_format(
        {
            "bold": True,
            "font_size": 14,
            "font_color": DARK_BROWN,
            "border": 1,
            "border_color": LIGHT_BORDER,
            "align": "center",
            "num_format": '£#,##0',
        }
    )

    formats[
        "metric_value"
    ] = workbook.add_format(
        {
            "bold": True,
            "font_size": 14,
            "font_color": DARK_BROWN,
            "border": 1,
            "border_color": LIGHT_BORDER,
            "align": "center",
        }
    )

    formats[
        "note"
    ] = workbook.add_format(
        {
            "font_color": BODY_TEXT,
            "bg_color": BEIGE,
            "text_wrap": True,
            "valign": "top",
            "border": 1,
            "border_color": LIGHT_BORDER,
        }
    )

    formats[
        "pass"
    ] = workbook.add_format(
        {
            "bold": True,
            "font_color": DARK_BROWN,
            "bg_color": TAUPE,
            "border": 1,
            "border_color": LIGHT_BORDER,
        }
    )

    return formats


# ============================================================
# GENERIC DATAFRAME SHEET
# ============================================================

def write_dataframe_sheet(
    writer,
    sheet_name,
    title,
    subtitle,
    dataframe,
    formats,
):
    dataframe.to_excel(
        writer,
        sheet_name=sheet_name,
        startrow=4,
        index=False,
    )

    worksheet = (
        writer.sheets[
            sheet_name
        ]
    )

    worksheet.hide_gridlines(
        2
    )

    worksheet.set_tab_color(
        MEDIUM_BROWN
    )

    last_column = max(
        len(
            dataframe.columns
        )
        - 1,
        0,
    )

    worksheet.merge_range(
        0,
        0,
        0,
        last_column,
        title,
        formats[
            "title"
        ],
    )

    worksheet.merge_range(
        1,
        0,
        1,
        last_column,
        subtitle,
        formats[
            "subtitle"
        ],
    )

    for (
        column_index,
        column_name,
    ) in enumerate(
        dataframe.columns
    ):
        worksheet.write(
            4,
            column_index,
            column_name,
            formats[
                "header"
            ],
        )

    worksheet.set_row(
        4,
        30,
    )

    if len(
        dataframe
    ) > 0:
        worksheet.conditional_format(
            5,
            0,
            (
                4
                + len(
                    dataframe
                )
            ),
            last_column,
            {
                "type": "formula",
                "criteria": (
                    "=MOD(ROW(),2)=0"
                ),
                "format": (
                    formats[
                        "alt_fill"
                    ]
                ),
            },
        )

        worksheet.autofilter(
            4,
            0,
            (
                4
                + len(
                    dataframe
                )
            ),
            last_column,
        )

    worksheet.freeze_panes(
        5,
        0,
    )

    for (
        column_index,
        column_name,
    ) in enumerate(
        dataframe.columns
    ):
        values = (
            dataframe[
                column_name
            ]
            .astype(str)
            .head(100)
            .tolist()
        )

        content_lengths = [
            len(
                str(
                    value
                )
            )
            for value in values
        ]

        maximum_content = max(
            content_lengths
            + [
                len(
                    str(
                        column_name
                    )
                )
            ]
        )

        width = min(
            max(
                maximum_content
                + 2,
                12,
            ),
            45,
        )

        if (
            column_name
            in CURRENCY_COLUMNS
        ):
            column_format = (
                formats[
                    "currency"
                ]
            )

        elif (
            column_name
            in PERCENTAGE_COLUMNS
        ):
            column_format = (
                formats[
                    "percentage"
                ]
            )

        elif (
            column_name
            in RATIO_COLUMNS
        ):
            column_format = (
                formats[
                    "ratio"
                ]
            )

        elif (
            column_name
            in DAY_COLUMNS
        ):
            column_format = (
                formats[
                    "days"
                ]
            )

        elif column_name in [
            "Year",
            "From Year",
            "To Year",
        ]:
            column_format = (
                formats[
                    "integer"
                ]
            )

        else:
            column_format = (
                formats[
                    "body"
                ]
            )

        worksheet.set_column(
            column_index,
            column_index,
            width,
            column_format,
        )

    worksheet.set_landscape()

    worksheet.fit_to_pages(
        1,
        0,
    )

    worksheet.set_margins(
        0.3,
        0.3,
        0.5,
        0.5,
    )

    return worksheet


# ============================================================
# EXECUTIVE SUMMARY SHEET
# ============================================================

def write_executive_summary(
    writer,
    analysis_df,
    forecast_df,
    review_df,
    model_checks,
    formats,
):
    workbook = (
        writer.book
    )

    worksheet = (
        workbook.add_worksheet(
            "Executive Summary"
        )
    )

    writer.sheets[
        "Executive Summary"
    ] = worksheet

    worksheet.hide_gridlines(
        2
    )

    worksheet.set_tab_color(
        DARK_BROWN
    )

    worksheet.set_column(
        "A:H",
        17,
    )

    worksheet.merge_range(
        "A1:H1",
        (
            "FINANCIAL STATEMENT ANALYSIS "
            "& FORECASTING"
        ),
        formats[
            "title"
        ],
    )

    worksheet.merge_range(
        "A2:H2",
        (
            "Professional analysis pack — "
            "portfolio simulation"
        ),
        formats[
            "subtitle"
        ],
    )

    latest = (
        analysis_df.iloc[-1]
    )

    final = (
        forecast_df.iloc[-1]
    )

    worksheet.merge_range(
        "A4:H4",
        "Latest Historical Performance",
        formats[
            "section"
        ],
    )

    labels = [
        "Historical Year",
        "Revenue",
        "Net Profit",
        "Free Cash Flow",
    ]

    values = [
        int(
            latest["year"]
        ),
        latest["revenue"],
        latest["net_profit"],
        latest[
            "free_cash_flow"
        ],
    ]

    columns = [
        0,
        2,
        4,
        6,
    ]

    for (
        label,
        value,
        column,
    ) in zip(
        labels,
        values,
        columns,
    ):
        worksheet.merge_range(
            4,
            column,
            4,
            column + 1,
            label,
            formats[
                "metric_label"
            ],
        )

        metric_format = (
            formats[
                "metric_value"
            ]
            if label
            == "Historical Year"
            else formats[
                "metric_currency"
            ]
        )

        worksheet.merge_range(
            5,
            column,
            5,
            column + 1,
            value,
            metric_format,
        )

    worksheet.merge_range(
        "A8:H8",
        "Final Forecast Period",
        formats[
            "section"
        ],
    )

    forecast_labels = [
        "Forecast Year",
        "Revenue",
        "Net Profit",
        "Closing Cash",
    ]

    forecast_values = [
        int(
            final["year"]
        ),
        final["revenue"],
        final["net_profit"],
        final["cash"],
    ]

    for (
        label,
        value,
        column,
    ) in zip(
        forecast_labels,
        forecast_values,
        columns,
    ):
        worksheet.merge_range(
            8,
            column,
            8,
            column + 1,
            label,
            formats[
                "metric_label"
            ],
        )

        metric_format = (
            formats[
                "metric_value"
            ]
            if label
            == "Forecast Year"
            else formats[
                "metric_currency"
            ]
        )

        worksheet.merge_range(
            9,
            column,
            9,
            column + 1,
            value,
            metric_format,
        )

    worksheet.merge_range(
        "A12:H12",
        "Review Status",
        formats[
            "section"
        ],
    )

    worksheet.write(
        "A13",
        "Generated Review Points",
        formats[
            "metric_label"
        ],
    )

    worksheet.write(
        "B13",
        len(
            review_df
        ),
        formats[
            "metric_value"
        ],
    )

    passed_checks = int(
        model_checks[
            "Status"
        ]
        .eq(
            "Passed"
        )
        .sum()
    )

    worksheet.write(
        "D13",
        "Model Checks Passed",
        formats[
            "metric_label"
        ],
    )

    worksheet.write(
        "E13",
        passed_checks,
        formats[
            "metric_value"
        ],
    )

    worksheet.write(
        "G13",
        "Total Model Checks",
        formats[
            "metric_label"
        ],
    )

    worksheet.write(
        "H13",
        len(
            model_checks
        ),
        formats[
            "metric_value"
        ],
    )

    worksheet.merge_range(
        "A15:H16",
        (
            "This workbook separates historical financial "
            "information, calculated analysis, assumptions "
            "and forecast outputs. Forecasts and scenarios are "
            "assumption-driven estimates rather than statements "
            "of future fact."
        ),
        formats[
            "note"
        ],
    )

    # --------------------------------------------------------
    # CHART DATA
    # --------------------------------------------------------

    combined = pd.concat(
        [
            analysis_df[
                [
                    "year",
                    "revenue",
                    "net_profit",
                ]
            ].assign(
                Period="Historical"
            ),

            forecast_df[
                [
                    "year",
                    "revenue",
                    "net_profit",
                ]
            ].assign(
                Period="Forecast"
            ),
        ],
        ignore_index=True,
    )

    worksheet.write(
        "J2",
        "Year",
        formats[
            "header"
        ],
    )

    worksheet.write(
        "K2",
        "Revenue",
        formats[
            "header"
        ],
    )

    worksheet.write(
        "L2",
        "Net Profit",
        formats[
            "header"
        ],
    )

    worksheet.write(
        "M2",
        "Period",
        formats[
            "header"
        ],
    )

    for (
        row_index,
        row,
    ) in enumerate(
        combined.itertuples(
            index=False
        ),
        start=2,
    ):
        worksheet.write(
            row_index,
            9,
            int(
                row.year
            ),
        )

        worksheet.write(
            row_index,
            10,
            float(
                row.revenue
            ),
        )

        worksheet.write(
            row_index,
            11,
            float(
                row.net_profit
            ),
        )

        worksheet.write(
            row_index,
            12,
            row.Period,
        )

    chart = workbook.add_chart(
        {
            "type": "line"
        }
    )

    last_chart_row = (
        2
        + len(
            combined
        )
        - 1
    )

    chart.add_series(
        {
            "name": "Revenue",
            "categories": [
                "Executive Summary",
                2,
                9,
                last_chart_row,
                9,
            ],
            "values": [
                "Executive Summary",
                2,
                10,
                last_chart_row,
                10,
            ],
            "line": {
                "color": (
                    DARK_BROWN
                ),
                "width": 2.25,
            },
        }
    )

    chart.add_series(
        {
            "name": "Net Profit",
            "categories": [
                "Executive Summary",
                2,
                9,
                last_chart_row,
                9,
            ],
            "values": [
                "Executive Summary",
                2,
                11,
                last_chart_row,
                11,
            ],
            "y2_axis": True,
            "line": {
                "color": (
                    MEDIUM_BROWN
                ),
                "width": 2.25,
            },
        }
    )

    chart.set_title(
        {
            "name": (
                "Historical + Forecast Trend"
            )
        }
    )

    chart.set_x_axis(
        {
            "name": "Year"
        }
    )

    chart.set_y_axis(
        {
            "name": "Revenue (£)",
            "num_format": (
                '£#,##0,,"m"'
            ),
        }
    )

    chart.set_y2_axis(
        {
            "name": "Net Profit (£)",
            "num_format": (
                '£#,##0,"k"'
            ),
        }
    )

    chart.set_legend(
        {
            "position": "bottom"
        }
    )

    chart.set_style(
        10
    )

    worksheet.insert_chart(
        "A19",
        chart,
        {
            "x_scale": 1.55,
            "y_scale": 1.35,
        },
    )

    worksheet.set_column(
        "J:M",
        None,
        None,
        {
            "hidden": True
        },
    )

    worksheet.set_landscape()

    worksheet.fit_to_pages(
        1,
        1,
    )


# ============================================================
# BUILD WORKBOOK
# ============================================================

def create_excel_report(
    analysis_df,
    forecast_assumptions,
    forecast_years,
    scenario_assumptions,
    scenario_years,
    review_settings,
):
    output = io.BytesIO()

    forecast_df = (
        build_forecast(
            analysis_df,
            forecast_assumptions,
            forecast_years,
        )
    )

    scenario_forecasts = (
        build_all_scenarios(
            analysis_df,
            scenario_assumptions,
            scenario_years,
        )
    )

    scenario_summary = (
        build_final_year_summary(
            scenario_forecasts
        )
    )

    sensitivity_df = (
        build_sensitivity_table(
            analysis_df,
            scenario_assumptions[
                "Base"
            ],
            scenario_years,
            "Net Profit",
        )
        .reset_index()
    )

    review_df = (
        build_review_points(
            analysis_df,
            review_settings[
                "movement_threshold"
            ],
            review_settings[
                "current_ratio"
            ],
            review_settings[
                "quick_ratio"
            ],
            review_settings[
                "debt_ratio"
            ],
            review_settings[
                "interest_cover"
            ],
        )
    )

    questions_df = (
        build_management_questions(
            analysis_df
        )
    )

    challenges_df = (
        build_forecast_challenges(
            analysis_df
        )
    )

    model_checks = (
        build_model_checks(
            analysis_df,
            forecast_df,
            scenario_forecasts,
        )
    )

    with pd.ExcelWriter(
        output,
        engine="xlsxwriter",
    ) as writer:

        workbook = (
            writer.book
        )

        workbook.set_properties(
            {
                "title": (
                    "Financial Statement Analysis "
                    "& Forecasting"
                ),
                "subject": (
                    "Portfolio financial analysis "
                    "and forecasting workbook"
                ),
                "author": (
                    "Portfolio Project"
                ),
                "company": (
                    "Portfolio Simulation"
                ),
                "comments": (
                    "Synthetic portfolio simulation. "
                    "Not audit assurance or investment advice."
                ),
            }
        )

        formats = (
            create_formats(
                workbook
            )
        )

        # ----------------------------------------------------
        # EXECUTIVE SUMMARY
        # ----------------------------------------------------

        write_executive_summary(
            writer,
            analysis_df,
            forecast_df,
            review_df,
            model_checks,
            formats,
        )

        # ----------------------------------------------------
        # HISTORICAL STATEMENTS
        # ----------------------------------------------------

        write_dataframe_sheet(
            writer,
            "Historical IS",
            "Historical Income Statement",
            (
                "Historical financial performance "
                "from the supplied dataset."
            ),
            historical_income_statement(
                analysis_df
            ),
            formats,
        )

        write_dataframe_sheet(
            writer,
            "Historical BS",
            "Historical Balance Sheet",
            (
                "Historical financial position "
                "from the supplied dataset."
            ),
            historical_balance_sheet(
                analysis_df
            ),
            formats,
        )

        write_dataframe_sheet(
            writer,
            "Historical CF",
            "Historical Cash Flow Summary",
            (
                "Simplified historical operating "
                "cash-flow and capital-expenditure analysis."
            ),
            historical_cash_flow(
                analysis_df
            ),
            formats,
        )

        # ----------------------------------------------------
        # HISTORICAL ANALYSIS
        # ----------------------------------------------------

        write_dataframe_sheet(
            writer,
            "Horizontal Analysis",
            "Horizontal / Year-on-Year Analysis",
            (
                "Absolute and percentage movements "
                "between financial years."
            ),
            build_horizontal_analysis(
                analysis_df
            ),
            formats,
        )

        write_dataframe_sheet(
            writer,
            "Common Size",
            "Common-Size Financial Analysis",
            (
                "Income statement items relative to revenue "
                "and balance-sheet items relative to total assets."
            ),
            build_common_size_analysis(
                analysis_df
            ),
            formats,
        )

        write_dataframe_sheet(
            writer,
            "Ratios & KPIs",
            "Financial Ratios & KPIs",
            (
                "Profitability, liquidity, leverage "
                "and working-capital indicators."
            ),
            build_ratio_table(
                analysis_df
            ),
            formats,
        )

        # ----------------------------------------------------
        # FORECAST
        # ----------------------------------------------------

        write_dataframe_sheet(
            writer,
            "Forecast Assumptions",
            "Forecast Assumptions",
            (
                "Current assumption-driven settings "
                "used by the Streamlit forecasting model."
            ),
            build_assumption_table(
                forecast_assumptions
            ),
            formats,
        )

        write_dataframe_sheet(
            writer,
            "Forecast IS",
            "Forecast Income Statement",
            (
                "Projected financial performance "
                "under the selected forecast assumptions."
            ),
            forecast_income_statement(
                forecast_df
            ),
            formats,
        )

        write_dataframe_sheet(
            writer,
            "Forecast BS",
            "Forecast Balance Sheet",
            (
                "Projected financial position "
                "under the selected forecast assumptions."
            ),
            forecast_balance_sheet(
                forecast_df
            ),
            formats,
        )

        write_dataframe_sheet(
            writer,
            "Forecast CF",
            "Forecast Cash Flow",
            (
                "Simplified forecast cash-flow bridge. "
                "Not a statutory cash-flow statement."
            ),
            forecast_cash_flow(
                forecast_df
            ),
            formats,
        )

        write_dataframe_sheet(
            writer,
            "Forecast Ratios",
            "Forecast Financial Ratios",
            (
                "Projected profitability, liquidity, "
                "leverage and working-capital indicators."
            ),
            forecast_ratio_table(
                forecast_df
            ),
            formats,
        )

        # ----------------------------------------------------
        # SCENARIOS
        # ----------------------------------------------------

        scenario_summary_export = (
            scenario_summary.rename(
                columns={
                    "Operating Margin": (
                        "Operating Margin"
                    ),
                    "Current Ratio": (
                        "Current Ratio"
                    ),
                    "Debt to Equity": (
                        "Debt to Equity"
                    ),
                    "Cash Conversion Cycle": (
                        "Cash Conversion Cycle"
                    ),
                }
            )
        )

        write_dataframe_sheet(
            writer,
            "Scenario Comparison",
            "Base / Upside / Downside Comparison",
            (
                "Final-year scenario outputs "
                "using the current scenario settings."
            ),
            scenario_summary_export,
            formats,
        )

        write_dataframe_sheet(
            writer,
            "Sensitivity",
            "Revenue Growth & Gross Margin Sensitivity",
            (
                "Final-year net profit sensitivity. "
                "Rows represent gross margin; columns "
                "represent revenue growth."
            ),
            sensitivity_df,
            formats,
        )

        # ----------------------------------------------------
        # REVIEW
        # ----------------------------------------------------

        write_dataframe_sheet(
            writer,
            "Review Register",
            "Financial Review Register",
            (
                "Rule-based analytical review points "
                "and suggested evidence / follow-up."
            ),
            review_df,
            formats,
        )

        write_dataframe_sheet(
            writer,
            "Forecast Challenge",
            "Forecast Assumption Challenge",
            (
                "Questions and evidence considerations "
                "for key forecast assumptions."
            ),
            challenges_df,
            formats,
        )

        write_dataframe_sheet(
            writer,
            "Management Questions",
            "Questions for Management",
            (
                "Potential follow-up questions generated "
                "from the analytical review framework."
            ),
            questions_df,
            formats,
        )

        write_dataframe_sheet(
            writer,
            "Model Checks",
            "Accounting Model Checks",
            (
                "Historical, forecast and scenario "
                "balance-sheet validation."
            ),
            model_checks,
            formats,
        )

        # ----------------------------------------------------
        # CONDITIONAL FORMATTING
        # ----------------------------------------------------

        checks_sheet = (
            writer.sheets[
                "Model Checks"
            ]
        )

        checks_sheet.conditional_format(
            5,
            3,
            (
                4
                + len(
                    model_checks
                )
            ),
            3,
            {
                "type": "text",
                "criteria": (
                    "containing"
                ),
                "value": "Passed",
                "format": (
                    formats[
                        "pass"
                    ]
                ),
            },
        )

    output.seek(
        0
    )

    return output.getvalue()


# ============================================================
# STREAMLIT EXPORT TAB
# ============================================================

def render_reporting_export_tab(
    analysis_df,
):
    st.header(
        "Professional Excel Reporting & Export"
    )

    st.write(
        """
        Generate a structured Excel analysis pack containing the
        historical analysis, current forecast model, scenario
        analysis, sensitivity testing and financial-review
        schedules.
        """
    )

    st.info(
        """
        The workbook uses the current settings from the
        Forecasting, Scenario Analysis and Management Review
        sections of this application. If you change those
        settings, the downloadable workbook updates with them.
        """
    )

    (
        forecast_assumptions,
        forecast_years,
    ) = (
        get_current_forecast_settings(
            analysis_df
        )
    )

    (
        scenario_assumptions,
        scenario_years,
    ) = (
        get_current_scenario_settings(
            analysis_df
        )
    )

    review_settings = (
        get_current_review_settings()
    )

    st.subheader(
        "Workbook Scope"
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:
        st.metric(
            "Historical Years",
            len(
                analysis_df
            ),
        )

    with col2:
        st.metric(
            "Forecast Horizon",
            (
                f"{forecast_years} "
                f"{'year' if forecast_years == 1 else 'years'}"
            ),
        )

    with col3:
        st.metric(
            "Scenario Horizon",
            (
                f"{scenario_years} "
                f"{'year' if scenario_years == 1 else 'years'}"
            ),
        )

    with st.expander(
        "Sheets included in the workbook",
        expanded=False,
    ):
        st.markdown(
            """
            **Executive Summary**

            **Historical analysis**
            - Historical IS
            - Historical BS
            - Historical CF
            - Horizontal Analysis
            - Common Size
            - Ratios & KPIs

            **Forecast**
            - Forecast Assumptions
            - Forecast IS
            - Forecast BS
            - Forecast CF
            - Forecast Ratios

            **Scenario & sensitivity**
            - Scenario Comparison
            - Sensitivity

            **Professional review**
            - Review Register
            - Forecast Challenge
            - Management Questions
            - Model Checks
            """
        )

    st.subheader(
        "Current Export Basis"
    )

    latest = (
        analysis_df.iloc[-1]
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:
        st.metric(
            "Latest Actual Year",
            int(
                latest["year"]
            ),
        )

    with col2:
        st.metric(
            "Revenue Growth Assumption",
            (
                f"{forecast_assumptions['revenue_growth']:.1%}"
            ),
        )

    with col3:
        st.metric(
            "Gross Margin Assumption",
            (
                f"{forecast_assumptions['gross_margin']:.1%}"
            ),
        )

    with col4:
        st.metric(
            "Movement Review Threshold",
            (
                f"{review_settings['movement_threshold']:.0%}"
            ),
        )

    st.caption(
        """
        Review-register statuses are regenerated from the current
        analytical settings when the workbook is produced. A
        status selected interactively in the Management Review
        table does not represent evidence obtained and is not
        treated as a professional clearance.
        """
    )

    st.divider()

    st.subheader(
        "Generate Workbook"
    )

    st.write(
        """
        The report will use the same brown, beige and white visual
        theme as the portfolio working papers.
        """
    )

    try:
        report_bytes = (
            create_excel_report(
                analysis_df,
                forecast_assumptions,
                forecast_years,
                scenario_assumptions,
                scenario_years,
                review_settings,
            )
        )

        latest_year = int(
            analysis_df.iloc[-1][
                "year"
            ]
        )

        file_name = (
            "Financial_Statement_Analysis_"
            f"Forecasting_{latest_year}.xlsx"
        )

        st.download_button(
            label=(
                "Download Professional Excel Report"
            ),
            data=report_bytes,
            file_name=file_name,
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
            type="primary",
            use_container_width=True,
        )

        st.success(
            """
            Workbook generated successfully. The download contains
            the historical analysis, forecasting model, scenario
            analysis, sensitivity testing and review schedules.
            """
        )

    except Exception as error:
        st.error(
            (
                "The Excel report could not be generated: "
                f"{error}"
            )
        )

    st.warning(
        """
        Portfolio simulation only. The exported workbook contains
        synthetic financial analysis and assumption-driven
        forecasts. It is not a statutory financial statement,
        audit opinion, assurance report, investment recommendation
        or guarantee of future performance.
        """
    )