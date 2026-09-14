import numpy as np
import pandas as pd
import streamlit as st

from forecasting_engine import render_forecasting_tab
from scenario_analysis import render_scenario_analysis_tab
from management_review import render_management_review_tab
from reporting_export import render_reporting_export_tab
# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial Statement Analysis & Forecasting",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

REQUIRED_COLUMNS = [
    "year",
    "revenue",
    "cost_of_sales",
    "operating_expenses",
    "interest_expense",
    "tax_expense",
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
    "operating_cash_flow",
    "capital_expenditure",
]


NUMERIC_COLUMNS = [
    column
    for column in REQUIRED_COLUMNS
    if column != "year"
]


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
    "current_assets": "Current Assets",
    "property_plant_equipment": "Property, Plant & Equipment",
    "total_assets": "Total Assets",
    "trade_payables": "Trade Payables",
    "current_liabilities": "Current Liabilities",
    "long_term_debt": "Long-Term Debt",
    "total_liabilities": "Total Liabilities",
    "equity": "Equity",
    "working_capital": "Working Capital",
    "operating_cash_flow": "Operating Cash Flow",
    "capital_expenditure": "Capital Expenditure",
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
    "revenue_growth": "Revenue Growth",
    "net_profit_growth": "Net Profit Growth",
}


INCOME_STATEMENT_COLUMNS = [
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


BALANCE_SHEET_COLUMNS = [
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


RATIO_COLUMNS = [
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


# ============================================================
# FORMATTING HELPERS
# ============================================================

def format_currency(value):
    try:
        if pd.isna(value):
            return "N/A"

        return f"£{float(value):,.0f}"

    except (TypeError, ValueError):
        return "N/A"


def format_percentage(value):
    try:
        if pd.isna(value):
            return "N/A"

        return f"{float(value):.1%}"

    except (TypeError, ValueError):
        return "N/A"


def format_ratio(value):
    try:
        if pd.isna(value):
            return "N/A"

        return f"{float(value):.2f}x"

    except (TypeError, ValueError):
        return "N/A"


def format_days(value):
    try:
        if pd.isna(value):
            return "N/A"

        return f"{float(value):.1f} days"

    except (TypeError, ValueError):
        return "N/A"


def format_signed_days(value):
    try:
        if pd.isna(value):
            return None

        return f"{float(value):+.1f} days"

    except (TypeError, ValueError):
        return None


def safe_divide(numerator, denominator):
    if (
        pd.isna(numerator)
        or pd.isna(denominator)
        or denominator == 0
    ):
        return np.nan

    return numerator / denominator


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_data(df):
    errors = []

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        errors.append(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

        return errors

    if df.empty:
        errors.append(
            "The dataset contains no usable rows."
        )

    if df["year"].duplicated().any():
        errors.append(
            "Each financial year must appear only once."
        )

    if len(df) < 2:
        errors.append(
            """
            At least two years of financial data are required
            for historical movement analysis.
            """
        )

    return errors


def clean_data(df):
    df = df.copy()

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    if "year" in df.columns:
        df["year"] = pd.to_numeric(
            df["year"],
            errors="coerce",
        )

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    if "year" in df.columns:
        df = df.dropna(
            subset=["year"]
        )

        df["year"] = (
            df["year"]
            .astype(int)
        )

        df = df.sort_values(
            "year"
        )

    return df.reset_index(
        drop=True
    )


# ============================================================
# FINANCIAL CALCULATIONS
# ============================================================

def calculate_financial_metrics(df):
    result = df.copy()

    result["gross_profit"] = (
        result["revenue"]
        - result["cost_of_sales"]
    )

    result["operating_profit"] = (
        result["gross_profit"]
        - result["operating_expenses"]
    )

    result["profit_before_tax"] = (
        result["operating_profit"]
        - result["interest_expense"]
    )

    result["net_profit"] = (
        result["profit_before_tax"]
        - result["tax_expense"]
    )

    result["working_capital"] = (
        result["current_assets"]
        - result["current_liabilities"]
    )

    result["free_cash_flow"] = (
        result["operating_cash_flow"]
        - result["capital_expenditure"]
    )

    result["gross_margin"] = result.apply(
        lambda row: safe_divide(
            row["gross_profit"],
            row["revenue"],
        ),
        axis=1,
    )

    result["operating_margin"] = result.apply(
        lambda row: safe_divide(
            row["operating_profit"],
            row["revenue"],
        ),
        axis=1,
    )

    result["net_margin"] = result.apply(
        lambda row: safe_divide(
            row["net_profit"],
            row["revenue"],
        ),
        axis=1,
    )

    result["current_ratio"] = result.apply(
        lambda row: safe_divide(
            row["current_assets"],
            row["current_liabilities"],
        ),
        axis=1,
    )

    result["quick_ratio"] = result.apply(
        lambda row: safe_divide(
            (
                row["current_assets"]
                - row["inventory"]
            ),
            row["current_liabilities"],
        ),
        axis=1,
    )

    result["debt_to_equity"] = result.apply(
        lambda row: safe_divide(
            row["total_liabilities"],
            row["equity"],
        ),
        axis=1,
    )

    result["interest_coverage"] = result.apply(
        lambda row: safe_divide(
            row["operating_profit"],
            row["interest_expense"],
        ),
        axis=1,
    )

    result["return_on_assets"] = result.apply(
        lambda row: safe_divide(
            row["net_profit"],
            row["total_assets"],
        ),
        axis=1,
    )

    result["return_on_equity"] = result.apply(
        lambda row: safe_divide(
            row["net_profit"],
            row["equity"],
        ),
        axis=1,
    )

    result["receivable_days"] = result.apply(
        lambda row: (
            safe_divide(
                row["trade_receivables"],
                row["revenue"],
            )
            * 365
        ),
        axis=1,
    )

    result["inventory_days"] = result.apply(
        lambda row: (
            safe_divide(
                row["inventory"],
                row["cost_of_sales"],
            )
            * 365
        ),
        axis=1,
    )

    result["payable_days"] = result.apply(
        lambda row: (
            safe_divide(
                row["trade_payables"],
                row["cost_of_sales"],
            )
            * 365
        ),
        axis=1,
    )

    result["cash_conversion_cycle"] = (
        result["receivable_days"]
        + result["inventory_days"]
        - result["payable_days"]
    )

    result["revenue_growth"] = (
        result["revenue"]
        .pct_change()
    )

    result["net_profit_growth"] = (
        result["net_profit"]
        .pct_change()
    )

    return result


# ============================================================
# HORIZONTAL ANALYSIS
# ============================================================

def build_horizontal_analysis(
    analysis_df,
    columns,
    movement_threshold,
):
    records = []

    for column in columns:
        for index in range(
            1,
            len(analysis_df),
        ):
            current = analysis_df.iloc[index]
            previous = analysis_df.iloc[index - 1]

            current_value = current[column]
            previous_value = previous[column]

            absolute_change = (
                current_value
                - previous_value
            )

            percentage_change = safe_divide(
                absolute_change,
                abs(previous_value),
            )

            material_flag = False

            if not pd.isna(
                percentage_change
            ):
                material_flag = (
                    abs(percentage_change)
                    >= movement_threshold
                )

            records.append(
                {
                    "Financial Statement Line": (
                        DISPLAY_NAMES.get(
                            column,
                            column,
                        )
                    ),
                    "From Year": int(
                        previous["year"]
                    ),
                    "To Year": int(
                        current["year"]
                    ),
                    "Previous Value": (
                        previous_value
                    ),
                    "Current Value": (
                        current_value
                    ),
                    "£ Movement": (
                        absolute_change
                    ),
                    "% Movement": (
                        percentage_change
                    ),
                    "Review Flag": (
                        "Review"
                        if material_flag
                        else "No flag"
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# COMMON-SIZE ANALYSIS
# ============================================================

def build_common_size_income_statement(
    analysis_df,
):
    records = []

    for _, row in analysis_df.iterrows():
        for column in INCOME_STATEMENT_COLUMNS:
            records.append(
                {
                    "Year": int(
                        row["year"]
                    ),
                    "Line Item": (
                        DISPLAY_NAMES[column]
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
                }
            )

    return pd.DataFrame(
        records
    )


def build_common_size_balance_sheet(
    analysis_df,
):
    columns = [
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

    records = []

    for _, row in analysis_df.iterrows():
        for column in columns:
            records.append(
                {
                    "Year": int(
                        row["year"]
                    ),
                    "Line Item": (
                        DISPLAY_NAMES[column]
                    ),
                    "Amount": (
                        row[column]
                    ),
                    "% of Total Assets": (
                        safe_divide(
                            row[column],
                            row["total_assets"],
                        )
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# CAGR
# ============================================================

def calculate_cagr(
    start_value,
    end_value,
    periods,
):
    if (
        pd.isna(start_value)
        or pd.isna(end_value)
        or start_value <= 0
        or end_value <= 0
        or periods <= 0
    ):
        return np.nan

    return (
        (end_value / start_value)
        ** (1 / periods)
        - 1
    )


# ============================================================
# DUPONT ANALYSIS
# ============================================================

def build_dupont_analysis(
    analysis_df,
):
    records = []

    for index in range(
        len(analysis_df)
    ):
        row = analysis_df.iloc[index]

        if index == 0:
            average_assets = (
                row["total_assets"]
            )

            average_equity = (
                row["equity"]
            )

            balance_basis = (
                "Closing balance used"
            )

        else:
            previous = (
                analysis_df.iloc[index - 1]
            )

            average_assets = (
                (
                    previous["total_assets"]
                    + row["total_assets"]
                )
                / 2
            )

            average_equity = (
                (
                    previous["equity"]
                    + row["equity"]
                )
                / 2
            )

            balance_basis = (
                "Average opening/closing balance"
            )

        net_profit_margin = (
            safe_divide(
                row["net_profit"],
                row["revenue"],
            )
        )

        asset_turnover = (
            safe_divide(
                row["revenue"],
                average_assets,
            )
        )

        equity_multiplier = (
            safe_divide(
                average_assets,
                average_equity,
            )
        )

        dupont_roe = (
            net_profit_margin
            * asset_turnover
            * equity_multiplier
        )

        records.append(
            {
                "Year": int(
                    row["year"]
                ),
                "Net Profit Margin": (
                    net_profit_margin
                ),
                "Asset Turnover": (
                    asset_turnover
                ),
                "Equity Multiplier": (
                    equity_multiplier
                ),
                "DuPont ROE": (
                    dupont_roe
                ),
                "Balance Basis": (
                    balance_basis
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# AUTOMATED ANALYTICAL OBSERVATIONS
# ============================================================

def generate_observations(
    analysis_df,
):
    observations = []

    if len(analysis_df) < 2:
        return observations

    latest = analysis_df.iloc[-1]
    previous = analysis_df.iloc[-2]

    revenue_growth = safe_divide(
        (
            latest["revenue"]
            - previous["revenue"]
        ),
        previous["revenue"],
    )

    if not pd.isna(
        revenue_growth
    ):
        if revenue_growth > 0:
            observations.append(
                (
                    "Revenue growth",
                    (
                        f"Revenue increased by "
                        f"{revenue_growth:.1%} from "
                        f"{int(previous['year'])} to "
                        f"{int(latest['year'])}."
                    ),
                    (
                        "Growth should be considered alongside "
                        "profitability, working capital and "
                        "cash conversion rather than in isolation."
                    ),
                )
            )

        elif revenue_growth < 0:
            observations.append(
                (
                    "Revenue decline",
                    (
                        f"Revenue decreased by "
                        f"{abs(revenue_growth):.1%} from "
                        f"{int(previous['year'])} to "
                        f"{int(latest['year'])}."
                    ),
                    (
                        "The underlying causes of the decline "
                        "would require investigation."
                    ),
                )
            )

    gross_margin_change = (
        latest["gross_margin"]
        - previous["gross_margin"]
    )

    if not pd.isna(
        gross_margin_change
    ):
        if gross_margin_change > 0:
            observations.append(
                (
                    "Gross margin",
                    (
                        "Gross margin improved by "
                        f"{gross_margin_change * 100:.1f} "
                        "percentage points."
                    ),
                    (
                        "This may indicate improved pricing, "
                        "sales mix or cost control, but the "
                        "underlying cause cannot be determined "
                        "from the financial statements alone."
                    ),
                )
            )

        elif gross_margin_change < 0:
            observations.append(
                (
                    "Gross margin",
                    (
                        "Gross margin declined by "
                        f"{abs(gross_margin_change) * 100:.1f} "
                        "percentage points."
                    ),
                    (
                        "This may warrant review of input costs, "
                        "pricing and product or customer mix."
                    ),
                )
            )

    receivable_change = (
        latest["receivable_days"]
        - previous["receivable_days"]
    )

    if not pd.isna(
        receivable_change
    ):
        if receivable_change > 3:
            observations.append(
                (
                    "Receivable days",
                    (
                        "Receivable days increased by "
                        f"{receivable_change:.1f} days."
                    ),
                    (
                        "Sales are taking longer to convert into "
                        "cash, which may warrant review of credit "
                        "control, customer mix and overdue balances."
                    ),
                )
            )

        elif receivable_change < -3:
            observations.append(
                (
                    "Receivable days",
                    (
                        "Receivable days decreased by "
                        f"{abs(receivable_change):.1f} days."
                    ),
                    (
                        "This may indicate faster cash collection, "
                        "subject to verification of the underlying "
                        "receivables population."
                    ),
                )
            )

    inventory_change = (
        latest["inventory_days"]
        - previous["inventory_days"]
    )

    if not pd.isna(
        inventory_change
    ):
        if inventory_change > 5:
            observations.append(
                (
                    "Inventory days",
                    (
                        "Inventory days increased by "
                        f"{inventory_change:.1f} days."
                    ),
                    (
                        "Inventory is being held for longer, which "
                        "may affect working capital and could warrant "
                        "review of demand, purchasing and obsolete "
                        "or slow-moving inventory."
                    ),
                )
            )

    debt_change = (
        latest["debt_to_equity"]
        - previous["debt_to_equity"]
    )

    if not pd.isna(
        debt_change
    ):
        if debt_change > 0.10:
            observations.append(
                (
                    "Financial leverage",
                    (
                        "Debt-to-equity increased from "
                        f"{previous['debt_to_equity']:.2f}x to "
                        f"{latest['debt_to_equity']:.2f}x."
                    ),
                    (
                        "The business is using more liabilities "
                        "relative to equity. This should be considered "
                        "alongside financing terms, cash generation "
                        "and debt-servicing capacity."
                    ),
                )
            )

    free_cash_flow_change = (
        latest["free_cash_flow"]
        - previous["free_cash_flow"]
    )

    if free_cash_flow_change > 0:
        observations.append(
            (
                "Free cash flow",
                (
                    "Free cash flow increased from "
                    f"{format_currency(previous['free_cash_flow'])} "
                    "to "
                    f"{format_currency(latest['free_cash_flow'])}."
                ),
                (
                    "This indicates stronger internally generated "
                    "cash after capital expenditure, although cash "
                    "quality and sustainability still require "
                    "interpretation."
                ),
            )
        )

    return observations


# ============================================================
# HEALTH SCORE
# ============================================================

def calculate_health_score(
    latest,
    previous,
    minimum_current_ratio,
    minimum_quick_ratio,
    maximum_debt_to_equity,
    minimum_interest_coverage,
    efficiency_tolerance_days,
):
    results = []

    def add_test(
        category,
        test,
        passed,
        points,
        maximum_points,
        result_text,
    ):
        results.append(
            {
                "Category": category,
                "Test": test,
                "Result": result_text,
                "Points": (
                    points
                    if passed
                    else 0
                ),
                "Maximum Points": (
                    maximum_points
                ),
                "Status": (
                    "Pass"
                    if passed
                    else "Review"
                ),
            }
        )

    add_test(
        "Profitability",
        "Positive net profit",
        latest["net_profit"] > 0,
        5,
        5,
        format_currency(
            latest["net_profit"]
        ),
    )

    add_test(
        "Profitability",
        "Positive operating margin",
        latest["operating_margin"] > 0,
        5,
        5,
        format_percentage(
            latest["operating_margin"]
        ),
    )

    add_test(
        "Profitability",
        "Gross margin not materially lower",
        (
            latest["gross_margin"]
            >= (
                previous["gross_margin"]
                - 0.02
            )
        ),
        5,
        5,
        (
            f"{latest['gross_margin']:.1%} "
            f"vs {previous['gross_margin']:.1%}"
        ),
    )

    add_test(
        "Profitability",
        "Net margin not materially lower",
        (
            latest["net_margin"]
            >= (
                previous["net_margin"]
                - 0.02
            )
        ),
        5,
        5,
        (
            f"{latest['net_margin']:.1%} "
            f"vs {previous['net_margin']:.1%}"
        ),
    )

    add_test(
        "Liquidity",
        "Current ratio benchmark",
        (
            latest["current_ratio"]
            >= minimum_current_ratio
        ),
        10,
        10,
        (
            f"{latest['current_ratio']:.2f}x "
            f"vs benchmark "
            f"{minimum_current_ratio:.2f}x"
        ),
    )

    add_test(
        "Liquidity",
        "Quick ratio benchmark",
        (
            latest["quick_ratio"]
            >= minimum_quick_ratio
        ),
        10,
        10,
        (
            f"{latest['quick_ratio']:.2f}x "
            f"vs benchmark "
            f"{minimum_quick_ratio:.2f}x"
        ),
    )

    add_test(
        "Efficiency",
        "Receivable-days movement",
        (
            latest["receivable_days"]
            <= (
                previous["receivable_days"]
                + efficiency_tolerance_days
            )
        ),
        7,
        7,
        (
            f"{latest['receivable_days']:.1f} days "
            f"vs {previous['receivable_days']:.1f} days"
        ),
    )

    add_test(
        "Efficiency",
        "Inventory-days movement",
        (
            latest["inventory_days"]
            <= (
                previous["inventory_days"]
                + efficiency_tolerance_days
            )
        ),
        7,
        7,
        (
            f"{latest['inventory_days']:.1f} days "
            f"vs {previous['inventory_days']:.1f} days"
        ),
    )

    add_test(
        "Efficiency",
        "Cash-conversion-cycle movement",
        (
            latest["cash_conversion_cycle"]
            <= (
                previous["cash_conversion_cycle"]
                + efficiency_tolerance_days
            )
        ),
        6,
        6,
        (
            f"{latest['cash_conversion_cycle']:.1f} days "
            f"vs "
            f"{previous['cash_conversion_cycle']:.1f} days"
        ),
    )

    add_test(
        "Leverage",
        "Debt-to-equity benchmark",
        (
            latest["debt_to_equity"]
            <= maximum_debt_to_equity
        ),
        10,
        10,
        (
            f"{latest['debt_to_equity']:.2f}x "
            f"vs maximum "
            f"{maximum_debt_to_equity:.2f}x"
        ),
    )

    add_test(
        "Leverage",
        "Interest coverage benchmark",
        (
            latest["interest_coverage"]
            >= minimum_interest_coverage
        ),
        10,
        10,
        (
            f"{latest['interest_coverage']:.2f}x "
            f"vs minimum "
            f"{minimum_interest_coverage:.2f}x"
        ),
    )

    add_test(
        "Cash Flow",
        "Positive operating cash flow",
        latest["operating_cash_flow"] > 0,
        10,
        10,
        format_currency(
            latest["operating_cash_flow"]
        ),
    )

    add_test(
        "Cash Flow",
        "Positive free cash flow",
        latest["free_cash_flow"] > 0,
        10,
        10,
        format_currency(
            latest["free_cash_flow"]
        ),
    )

    score_df = pd.DataFrame(
        results
    )

    total_score = int(
        score_df["Points"].sum()
    )

    maximum_score = int(
        score_df[
            "Maximum Points"
        ].sum()
    )

    return (
        total_score,
        maximum_score,
        score_df,
    )


def health_score_band(
    score,
):
    if score >= 80:
        return (
            "Strong result under selected analytical rules"
        )

    if score >= 60:
        return (
            "Generally positive with areas for review"
        )

    if score >= 40:
        return (
            "Mixed result — further review warranted"
        )

    return (
        "Multiple areas require analytical review"
    )


# ============================================================
# HEALTH REVIEW FLAGS
# ============================================================

def generate_health_flags(
    latest,
    previous,
    minimum_current_ratio,
    minimum_quick_ratio,
    maximum_debt_to_equity,
    minimum_interest_coverage,
    efficiency_tolerance_days,
):
    flags = []

    def add_flag(
        area,
        observation,
        why_it_matters,
        priority,
    ):
        flags.append(
            {
                "Area": area,
                "Observation": observation,
                "Why It Matters": why_it_matters,
                "Priority": priority,
            }
        )

    if latest["net_profit"] <= 0:
        add_flag(
            "Profitability",
            (
                "Latest-period net profit "
                "is zero or negative."
            ),
            (
                "Persistent losses may affect "
                "cash generation, capital and "
                "financial resilience."
            ),
            "Higher priority",
        )

    operating_margin_change = (
        latest["operating_margin"]
        - previous["operating_margin"]
    )

    if operating_margin_change < -0.02:
        add_flag(
            "Profitability",
            (
                "Operating margin decreased by "
                f"{abs(operating_margin_change) * 100:.1f} "
                "percentage points."
            ),
            (
                "A material margin decline may warrant "
                "review of pricing, sales mix and "
                "operating cost growth."
            ),
            "Review",
        )

    if (
        latest["current_ratio"]
        < minimum_current_ratio
    ):
        add_flag(
            "Liquidity",
            (
                "Current ratio of "
                f"{latest['current_ratio']:.2f}x is below "
                f"the selected {minimum_current_ratio:.2f}x "
                "benchmark."
            ),
            (
                "The relationship between current assets "
                "and short-term liabilities may warrant "
                "closer liquidity review."
            ),
            "Review",
        )

    if (
        latest["quick_ratio"]
        < minimum_quick_ratio
    ):
        add_flag(
            "Liquidity",
            (
                "Quick ratio of "
                f"{latest['quick_ratio']:.2f}x is below "
                f"the selected {minimum_quick_ratio:.2f}x "
                "benchmark."
            ),
            (
                "Liquidity excluding inventory may be "
                "weaker than the selected analytical "
                "benchmark suggests."
            ),
            "Review",
        )

    if (
        latest["debt_to_equity"]
        > maximum_debt_to_equity
    ):
        add_flag(
            "Leverage",
            (
                "Debt-to-equity of "
                f"{latest['debt_to_equity']:.2f}x exceeds "
                f"the selected {maximum_debt_to_equity:.2f}x "
                "benchmark."
            ),
            (
                "Higher financial leverage can increase "
                "financing and repayment exposure."
            ),
            "Review",
        )

    if (
        latest["interest_coverage"]
        < minimum_interest_coverage
    ):
        add_flag(
            "Debt servicing",
            (
                "Interest coverage of "
                f"{latest['interest_coverage']:.2f}x is below "
                f"the selected "
                f"{minimum_interest_coverage:.2f}x benchmark."
            ),
            (
                "Operating profit provides less coverage "
                "of finance costs under the selected "
                "benchmark."
            ),
            "Higher priority",
        )

    receivable_change = (
        latest["receivable_days"]
        - previous["receivable_days"]
    )

    if (
        receivable_change
        > efficiency_tolerance_days
    ):
        add_flag(
            "Receivables",
            (
                "Receivable days increased by "
                f"{receivable_change:.1f} days."
            ),
            (
                "Slower collection can absorb cash and "
                "may warrant customer-level credit review."
            ),
            "Review",
        )

    inventory_change = (
        latest["inventory_days"]
        - previous["inventory_days"]
    )

    if (
        inventory_change
        > efficiency_tolerance_days
    ):
        add_flag(
            "Inventory",
            (
                "Inventory days increased by "
                f"{inventory_change:.1f} days."
            ),
            (
                "Longer inventory holding periods can "
                "absorb working capital and may warrant "
                "review of slow-moving inventory."
            ),
            "Review",
        )

    ccc_change = (
        latest["cash_conversion_cycle"]
        - previous["cash_conversion_cycle"]
    )

    if (
        ccc_change
        > efficiency_tolerance_days
    ):
        add_flag(
            "Working capital",
            (
                "Cash conversion cycle increased by "
                f"{ccc_change:.1f} days."
            ),
            (
                "A longer operating cash cycle can increase "
                "the amount of funding tied up in working "
                "capital."
            ),
            "Review",
        )

    if latest["free_cash_flow"] < 0:
        add_flag(
            "Cash flow",
            (
                "Free cash flow is negative at "
                f"{format_currency(latest['free_cash_flow'])}."
            ),
            (
                "Operating cash generation is insufficient "
                "to cover capital expenditure in the "
                "calculated period."
            ),
            "Higher priority",
        )

    if (
        latest["net_profit"] > 0
        and latest["operating_cash_flow"]
        < latest["net_profit"]
    ):
        add_flag(
            "Cash conversion",
            (
                "Operating cash flow is below "
                "reported net profit."
            ),
            (
                "Profit is not converting into operating "
                "cash at the same level, which may warrant "
                "review of working-capital movements and "
                "non-cash items."
            ),
            "Monitor",
        )

    return pd.DataFrame(
        flags
    )


# ============================================================
# DISPLAY HELPERS
# ============================================================

def prepare_currency_table(
    df,
):
    display_df = df.copy()

    for column in display_df.columns:
        if column != "year":
            display_df[column] = (
                display_df[column]
                .map(format_currency)
            )

    display_df = display_df.rename(
        columns=DISPLAY_NAMES
    )

    return display_df


def prepare_ratio_table(
    analysis_df,
):
    ratio_table = analysis_df[
        ["year"] + RATIO_COLUMNS
    ].copy()

    percentage_columns = [
        "gross_margin",
        "operating_margin",
        "net_margin",
        "return_on_assets",
        "return_on_equity",
    ]

    for column in percentage_columns:
        ratio_table[column] = (
            ratio_table[column]
            .map(format_percentage)
        )

    ratio_columns = [
        "current_ratio",
        "quick_ratio",
        "debt_to_equity",
        "interest_coverage",
    ]

    for column in ratio_columns:
        ratio_table[column] = (
            ratio_table[column]
            .map(format_ratio)
        )

    day_columns = [
        "receivable_days",
        "inventory_days",
        "payable_days",
        "cash_conversion_cycle",
    ]

    for column in day_columns:
        ratio_table[column] = (
            ratio_table[column]
            .map(format_days)
        )

    ratio_table = ratio_table.rename(
        columns=DISPLAY_NAMES
    )

    return ratio_table


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "Financial Statement Analysis & Forecasting"
)

st.write(
    """
    Analyse historical financial statements, calculate financial
    ratios, assess key performance indicators and build
    assumption-driven forecasts.
    """
)

st.info(
    """
    Portfolio simulation only. Historical calculations are based
    on supplied data. KPI benchmarks and financial-health scores
    are transparent analytical rules for comparison purposes;
    they are not universal industry benchmarks, audit assurance,
    investment advice or substitutes for professional judgement.
    """
)


# ============================================================
# DATA INPUT
# ============================================================

st.subheader(
    "Financial Statement Data"
)

uploaded_file = st.file_uploader(
    "Upload financial statements CSV",
    type=["csv"],
)


with st.expander(
    "Required CSV structure"
):
    st.write(
        """
        The dataset must contain one row per financial year and
        the following fields:
        """
    )

    for column in REQUIRED_COLUMNS:
        st.write(
            f"• {column}"
        )


if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(
            uploaded_file
        )

        source_message = (
            "Using uploaded financial statement data."
        )

    except Exception as error:
        st.error(
            f"Unable to read the uploaded CSV: {error}"
        )

        st.stop()

else:
    try:
        raw_df = pd.read_csv(
            "financial_statements.csv"
        )

        source_message = (
            "Using the included synthetic example dataset."
        )

    except FileNotFoundError:
        st.error(
            """
            No uploaded file was provided and
            financial_statements.csv could not be found.
            """
        )

        st.stop()


st.caption(
    source_message
)


# ============================================================
# CLEAN AND VALIDATE
# ============================================================

df = clean_data(
    raw_df
)

validation_errors = validate_data(
    df
)


if validation_errors:
    st.error(
        "The dataset could not be analysed."
    )

    for error in validation_errors:
        st.write(
            f"• {error}"
        )

    st.stop()


if (
    df[NUMERIC_COLUMNS]
    .isna()
    .any()
    .any()
):
    st.warning(
        """
        Some financial values could not be interpreted as numbers.
        Review the source data before relying on the analysis.
        """
    )


analysis_df = (
    calculate_financial_metrics(
        df
    )
)


latest = analysis_df.iloc[-1]
previous = analysis_df.iloc[-2]
first = analysis_df.iloc[0]

latest_year = int(
    latest["year"]
)

previous_year = int(
    previous["year"]
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header(
    "Analysis Settings"
)


movement_threshold_percent = (
    st.sidebar.slider(
        "Movement review threshold (%)",
        min_value=5,
        max_value=50,
        value=10,
        step=1,
    )
)


movement_threshold = (
    movement_threshold_percent
    / 100
)


st.sidebar.caption(
    """
    The movement threshold is an analytical review setting only.
    It is not audit materiality.
    """
)


st.sidebar.divider()


st.sidebar.subheader(
    "Illustrative KPI Benchmarks"
)


minimum_current_ratio = (
    st.sidebar.number_input(
        "Minimum current ratio",
        min_value=0.0,
        max_value=5.0,
        value=1.20,
        step=0.10,
    )
)


minimum_quick_ratio = (
    st.sidebar.number_input(
        "Minimum quick ratio",
        min_value=0.0,
        max_value=5.0,
        value=1.00,
        step=0.10,
    )
)


maximum_debt_to_equity = (
    st.sidebar.number_input(
        "Maximum debt-to-equity",
        min_value=0.0,
        max_value=10.0,
        value=1.50,
        step=0.10,
    )
)


minimum_interest_coverage = (
    st.sidebar.number_input(
        "Minimum interest coverage",
        min_value=0.0,
        max_value=20.0,
        value=3.00,
        step=0.50,
    )
)


efficiency_tolerance_days = (
    st.sidebar.number_input(
        "Efficiency deterioration tolerance (days)",
        min_value=0.0,
        max_value=60.0,
        value=5.0,
        step=1.0,
    )
)


st.sidebar.caption(
    """
    These benchmarks are editable analytical assumptions for
    portfolio demonstration purposes. Appropriate benchmarks
    depend on industry, business model and circumstances.
    """
)


# ============================================================
# HEALTH SCORE CALCULATION
# ============================================================

(
    health_score,
    health_score_maximum,
    health_score_detail,
) = calculate_health_score(
    latest,
    previous,
    minimum_current_ratio,
    minimum_quick_ratio,
    maximum_debt_to_equity,
    minimum_interest_coverage,
    efficiency_tolerance_days,
)


health_percentage = (
    safe_divide(
        health_score,
        health_score_maximum,
    )
    * 100
)


health_band = (
    health_score_band(
        health_percentage
    )
)


health_flags = (
    generate_health_flags(
        latest,
        previous,
        minimum_current_ratio,
        minimum_quick_ratio,
        maximum_debt_to_equity,
        minimum_interest_coverage,
        efficiency_tolerance_days,
    )
)


# ============================================================
# PAGE TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "KPI Dashboard",
        "Overview",
        "Movement Analysis",
        "Common-Size Analysis",
        "Ratios & Interpretation",
        "Forecasting",
        "Scenario Analysis",
        "Management Review",
        "Excel Export",
    ]
)


# ============================================================
# TAB 1 — KPI DASHBOARD
# ============================================================

with tab1:

    st.header(
        f"Financial Health Dashboard — {latest_year}"
    )

    st.caption(
        """
        The score below applies transparent rules using the
        benchmark settings in the sidebar. It is an analytical
        comparison tool rather than a credit rating, audit
        conclusion or prediction of financial viability.
        """
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:
        st.metric(
            "Analytical Score",
            (
                f"{health_percentage:.0f}/100"
            ),
        )


    with col2:
        st.metric(
            "Latest Financial Year",
            latest_year,
        )


    with col3:
        st.metric(
            "Review Flags",
            len(
                health_flags
            ),
        )


    with col4:
        st.metric(
            "Free Cash Flow",
            format_currency(
                latest["free_cash_flow"]
            ),
        )


    st.info(
        f"**Score interpretation:** {health_band}"
    )


    # --------------------------------------------------------
    # PROFITABILITY
    # --------------------------------------------------------

    st.subheader(
        "Profitability"
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:
        st.metric(
            "Gross Margin",
            format_percentage(
                latest["gross_margin"]
            ),
            (
                f"{(
                    latest['gross_margin']
                    - previous['gross_margin']
                ) * 100:+.1f} pp"
            ),
        )


    with col2:
        st.metric(
            "Operating Margin",
            format_percentage(
                latest["operating_margin"]
            ),
            (
                f"{(
                    latest['operating_margin']
                    - previous['operating_margin']
                ) * 100:+.1f} pp"
            ),
        )


    with col3:
        st.metric(
            "Net Margin",
            format_percentage(
                latest["net_margin"]
            ),
            (
                f"{(
                    latest['net_margin']
                    - previous['net_margin']
                ) * 100:+.1f} pp"
            ),
        )


    with col4:
        st.metric(
            "Return on Equity",
            format_percentage(
                latest["return_on_equity"]
            ),
            (
                f"{(
                    latest['return_on_equity']
                    - previous['return_on_equity']
                ) * 100:+.1f} pp"
            ),
        )


    # --------------------------------------------------------
    # LIQUIDITY & LEVERAGE
    # --------------------------------------------------------

    st.subheader(
        "Liquidity & Leverage"
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:
        st.metric(
            "Current Ratio",
            format_ratio(
                latest["current_ratio"]
            ),
            (
                f"{(
                    latest['current_ratio']
                    - previous['current_ratio']
                ):+.2f}x"
            ),
            delta_color="off",
        )


    with col2:
        st.metric(
            "Quick Ratio",
            format_ratio(
                latest["quick_ratio"]
            ),
            (
                f"{(
                    latest['quick_ratio']
                    - previous['quick_ratio']
                ):+.2f}x"
            ),
            delta_color="off",
        )


    with col3:
        st.metric(
            "Debt to Equity",
            format_ratio(
                latest["debt_to_equity"]
            ),
            (
                f"{(
                    latest['debt_to_equity']
                    - previous['debt_to_equity']
                ):+.2f}x"
            ),
            delta_color="off",
        )


    with col4:
        st.metric(
            "Interest Coverage",
            format_ratio(
                latest["interest_coverage"]
            ),
            (
                f"{(
                    latest['interest_coverage']
                    - previous['interest_coverage']
                ):+.2f}x"
            ),
            delta_color="off",
        )


    # --------------------------------------------------------
    # EFFICIENCY
    # --------------------------------------------------------

    st.subheader(
        "Working-Capital Efficiency"
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:
        st.metric(
            "Receivable Days",
            format_days(
                latest["receivable_days"]
            ),
            format_signed_days(
                latest["receivable_days"]
                - previous["receivable_days"]
            ),
            delta_color="off",
        )


    with col2:
        st.metric(
            "Inventory Days",
            format_days(
                latest["inventory_days"]
            ),
            format_signed_days(
                latest["inventory_days"]
                - previous["inventory_days"]
            ),
            delta_color="off",
        )


    with col3:
        st.metric(
            "Payable Days",
            format_days(
                latest["payable_days"]
            ),
            format_signed_days(
                latest["payable_days"]
                - previous["payable_days"]
            ),
            delta_color="off",
        )


    with col4:
        st.metric(
            "Cash Conversion Cycle",
            format_days(
                latest[
                    "cash_conversion_cycle"
                ]
            ),
            format_signed_days(
                latest[
                    "cash_conversion_cycle"
                ]
                - previous[
                    "cash_conversion_cycle"
                ]
            ),
            delta_color="off",
        )


    st.caption(
        """
        Cash conversion cycle = receivable days + inventory days
        − payable days. It estimates how long operating cash is
        tied up in the working-capital cycle.
        """
    )


    # --------------------------------------------------------
    # CASH GENERATION
    # --------------------------------------------------------

    st.subheader(
        "Cash Generation"
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:
        st.metric(
            "Operating Cash Flow",
            format_currency(
                latest["operating_cash_flow"]
            ),
            format_currency(
                latest["operating_cash_flow"]
                - previous[
                    "operating_cash_flow"
                ]
            ),
        )


    with col2:
        st.metric(
            "Capital Expenditure",
            format_currency(
                latest["capital_expenditure"]
            ),
            format_currency(
                latest["capital_expenditure"]
                - previous[
                    "capital_expenditure"
                ]
            ),
            delta_color="off",
        )


    with col3:
        st.metric(
            "Free Cash Flow",
            format_currency(
                latest["free_cash_flow"]
            ),
            format_currency(
                latest["free_cash_flow"]
                - previous[
                    "free_cash_flow"
                ]
            ),
        )


    with col4:
        st.metric(
            "Working Capital",
            format_currency(
                latest["working_capital"]
            ),
            format_currency(
                latest["working_capital"]
                - previous[
                    "working_capital"
                ]
            ),
        )


    # --------------------------------------------------------
    # CATEGORY SCORE
    # --------------------------------------------------------

    st.subheader(
        "Analytical Score by Category"
    )


    category_score = (
        health_score_detail
        .groupby(
            "Category",
            as_index=False,
        )
        .agg(
            {
                "Points": "sum",
                "Maximum Points": "sum",
            }
        )
    )


    category_score[
        "Score %"
    ] = (
        category_score["Points"]
        / category_score[
            "Maximum Points"
        ]
        * 100
    )


    category_chart = (
        category_score[
            [
                "Category",
                "Score %",
            ]
        ]
        .set_index(
            "Category"
        )
    )


    st.bar_chart(
        category_chart
    )


    category_display = (
        category_score.copy()
    )

    category_display[
        "Score %"
    ] = (
        category_display[
            "Score %"
        ]
        .map(
            lambda value: (
                f"{value:.0f}%"
            )
        )
    )


    st.dataframe(
        category_display,
        hide_index=True,
        width="stretch",
    )


    with st.expander(
        "View score calculation"
    ):
        st.dataframe(
            health_score_detail,
            hide_index=True,
            width="stretch",
        )


    # --------------------------------------------------------
    # REVIEW FLAGS
    # --------------------------------------------------------

    st.subheader(
        "Financial Health Review Flags"
    )


    if health_flags.empty:
        st.success(
            """
            No review flags were generated using the selected
            analytical benchmarks.
            """
        )

    else:
        st.dataframe(
            health_flags,
            hide_index=True,
            width="stretch",
        )


    st.caption(
        """
        A review flag identifies an area for investigation.
        It does not establish that an accounting error,
        impairment, going-concern issue or misstatement exists.
        """
    )


    # --------------------------------------------------------
    # MULTI-YEAR KPI TREND
    # --------------------------------------------------------

    st.subheader(
        "Multi-Year KPI Explorer"
    )


    kpi_options = {
        "Revenue": (
            "revenue",
            "currency",
        ),
        "Net Profit": (
            "net_profit",
            "currency",
        ),
        "Free Cash Flow": (
            "free_cash_flow",
            "currency",
        ),
        "Gross Margin": (
            "gross_margin",
            "percentage",
        ),
        "Operating Margin": (
            "operating_margin",
            "percentage",
        ),
        "Net Margin": (
            "net_margin",
            "percentage",
        ),
        "Current Ratio": (
            "current_ratio",
            "ratio",
        ),
        "Quick Ratio": (
            "quick_ratio",
            "ratio",
        ),
        "Debt to Equity": (
            "debt_to_equity",
            "ratio",
        ),
        "Interest Coverage": (
            "interest_coverage",
            "ratio",
        ),
        "Receivable Days": (
            "receivable_days",
            "days",
        ),
        "Inventory Days": (
            "inventory_days",
            "days",
        ),
        "Payable Days": (
            "payable_days",
            "days",
        ),
        "Cash Conversion Cycle": (
            "cash_conversion_cycle",
            "days",
        ),
    }


    selected_kpi = st.selectbox(
        "Choose KPI",
        options=list(
            kpi_options.keys()
        ),
    )


    selected_column, selected_type = (
        kpi_options[
            selected_kpi
        ]
    )


    kpi_trend = analysis_df[
        [
            "year",
            selected_column,
        ]
    ].copy()


    if selected_type == "percentage":
        kpi_trend[
            selected_column
        ] = (
            kpi_trend[
                selected_column
            ]
            * 100
        )


    kpi_trend = (
        kpi_trend
        .rename(
            columns={
                selected_column: selected_kpi
            }
        )
        .set_index(
            "year"
        )
    )


    st.line_chart(
        kpi_trend
    )


    # --------------------------------------------------------
    # DUPONT
    # --------------------------------------------------------

    st.subheader(
        "DuPont Return on Equity Analysis"
    )


    st.write(
        """
        DuPont analysis separates return on equity into three
        components: net profit margin, asset turnover and the
        equity multiplier. This helps distinguish whether ROE is
        being driven primarily by profitability, asset utilisation
        or financial leverage.
        """
    )


    dupont_df = (
        build_dupont_analysis(
            analysis_df
        )
    )


    dupont_display = (
        dupont_df.copy()
    )


    dupont_display[
        "Net Profit Margin"
    ] = dupont_display[
        "Net Profit Margin"
    ].map(
        format_percentage
    )


    dupont_display[
        "Asset Turnover"
    ] = dupont_display[
        "Asset Turnover"
    ].map(
        format_ratio
    )


    dupont_display[
        "Equity Multiplier"
    ] = dupont_display[
        "Equity Multiplier"
    ].map(
        format_ratio
    )


    dupont_display[
        "DuPont ROE"
    ] = dupont_display[
        "DuPont ROE"
    ].map(
        format_percentage
    )


    st.dataframe(
        dupont_display,
        hide_index=True,
        width="stretch",
    )


    st.caption(
        """
        For the first year, closing assets and equity are used
        because no opening balance is supplied. Later years use
        average opening and closing balances. DuPont ROE can
        therefore differ from the closing-balance ROE shown in
        the standard ratio table.
        """
    )


# ============================================================
# TAB 2 — OVERVIEW
# ============================================================

with tab2:

    st.header(
        "Historical Financial Summary"
    )


    revenue_delta = (
        latest["revenue"]
        - previous["revenue"]
    )

    net_profit_delta = (
        latest["net_profit"]
        - previous["net_profit"]
    )

    operating_margin_delta = (
        latest["operating_margin"]
        - previous["operating_margin"]
    )

    free_cash_flow_delta = (
        latest["free_cash_flow"]
        - previous["free_cash_flow"]
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:
        st.metric(
            f"Revenue — {latest_year}",
            format_currency(
                latest["revenue"]
            ),
            format_currency(
                revenue_delta
            ),
        )


    with col2:
        st.metric(
            f"Net Profit — {latest_year}",
            format_currency(
                latest["net_profit"]
            ),
            format_currency(
                net_profit_delta
            ),
        )


    with col3:
        st.metric(
            "Operating Margin",
            format_percentage(
                latest["operating_margin"]
            ),
            (
                f"{operating_margin_delta * 100:+.1f} pp"
            ),
        )


    with col4:
        st.metric(
            "Free Cash Flow",
            format_currency(
                latest["free_cash_flow"]
            ),
            format_currency(
                free_cash_flow_delta
            ),
        )


    st.subheader(
        "Growth Since First Historical Year"
    )


    periods = (
        int(latest["year"])
        - int(first["year"])
    )


    revenue_cagr = calculate_cagr(
        first["revenue"],
        latest["revenue"],
        periods,
    )

    net_profit_cagr = calculate_cagr(
        first["net_profit"],
        latest["net_profit"],
        periods,
    )

    operating_cash_flow_cagr = (
        calculate_cagr(
            first["operating_cash_flow"],
            latest["operating_cash_flow"],
            periods,
        )
    )


    col1, col2, col3 = (
        st.columns(3)
    )


    with col1:
        st.metric(
            "Revenue CAGR",
            format_percentage(
                revenue_cagr
            ),
        )


    with col2:
        st.metric(
            "Net Profit CAGR",
            format_percentage(
                net_profit_cagr
            ),
        )


    with col3:
        st.metric(
            "Operating Cash Flow CAGR",
            format_percentage(
                operating_cash_flow_cagr
            ),
        )


    st.caption(
        f"""
        CAGR measures compound annual growth from
        {int(first['year'])} to {latest_year}.
        """
    )


    st.subheader(
        "Income Statement Summary"
    )


    income_statement = (
        analysis_df[
            ["year"]
            + INCOME_STATEMENT_COLUMNS
        ]
        .copy()
    )


    st.dataframe(
        prepare_currency_table(
            income_statement
        ),
        hide_index=True,
        width="stretch",
    )


    st.subheader(
        "Balance Sheet Summary"
    )


    balance_sheet = (
        analysis_df[
            ["year"]
            + BALANCE_SHEET_COLUMNS
        ]
        .copy()
    )


    st.dataframe(
        prepare_currency_table(
            balance_sheet
        ),
        hide_index=True,
        width="stretch",
    )


    st.subheader(
        "Cash Flow Summary"
    )


    cash_flow = analysis_df[
        [
            "year",
            "operating_cash_flow",
            "capital_expenditure",
            "free_cash_flow",
        ]
    ].copy()


    st.dataframe(
        prepare_currency_table(
            cash_flow
        ),
        hide_index=True,
        width="stretch",
    )


    st.subheader(
        "Revenue & Profit Trend"
    )


    trend_data = analysis_df[
        [
            "year",
            "revenue",
            "gross_profit",
            "operating_profit",
            "net_profit",
        ]
    ].copy()


    trend_data = (
        trend_data
        .set_index(
            "year"
        )
        .rename(
            columns=DISPLAY_NAMES
        )
    )


    st.line_chart(
        trend_data
    )


# ============================================================
# TAB 3 — MOVEMENT ANALYSIS
# ============================================================

with tab3:

    st.header(
        "Horizontal / Year-on-Year Analysis"
    )

    st.write(
        """
        Horizontal analysis compares each financial statement
        line with the previous year to identify absolute and
        percentage movements.
        """
    )

    st.warning(
        """
        A movement flag means that the percentage change exceeds
        the analytical threshold selected in the sidebar. It does
        not mean that the movement is an error or an audit
        misstatement.
        """
    )


    movement_columns = (
        INCOME_STATEMENT_COLUMNS
        + [
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
    )


    horizontal_df = (
        build_horizontal_analysis(
            analysis_df,
            movement_columns,
            movement_threshold,
        )
    )


    selected_year = st.selectbox(
        "Select year to review",
        options=sorted(
            horizontal_df[
                "To Year"
            ]
            .unique(),
            reverse=True,
        ),
        key="movement_year",
    )


    selected_movements = (
        horizontal_df[
            horizontal_df[
                "To Year"
            ]
            == selected_year
        ]
        .copy()
    )


    flagged_movements = (
        selected_movements[
            selected_movements[
                "Review Flag"
            ]
            == "Review"
        ]
        .copy()
    )


    col1, col2, col3 = (
        st.columns(3)
    )


    with col1:
        st.metric(
            "Lines Reviewed",
            len(
                selected_movements
            ),
        )


    with col2:
        st.metric(
            "Movement Flags",
            len(
                flagged_movements
            ),
        )


    with col3:
        st.metric(
            "Review Threshold",
            (
                f"{movement_threshold_percent}%"
            ),
        )


    display_movements = (
        selected_movements.copy()
    )


    display_movements[
        "Previous Value"
    ] = display_movements[
        "Previous Value"
    ].map(
        format_currency
    )


    display_movements[
        "Current Value"
    ] = display_movements[
        "Current Value"
    ].map(
        format_currency
    )


    display_movements[
        "£ Movement"
    ] = display_movements[
        "£ Movement"
    ].map(
        format_currency
    )


    display_movements[
        "% Movement"
    ] = display_movements[
        "% Movement"
    ].map(
        format_percentage
    )


    st.subheader(
        f"Movement Schedule — {selected_year}"
    )


    st.dataframe(
        display_movements,
        hide_index=True,
        width="stretch",
    )


    st.subheader(
        "Items Flagged for Review"
    )


    if flagged_movements.empty:
        st.success(
            """
            No financial statement lines exceeded the
            selected analytical threshold.
            """
        )

    else:
        flagged_display = (
            flagged_movements[
                [
                    "Financial Statement Line",
                    "From Year",
                    "To Year",
                    "£ Movement",
                    "% Movement",
                ]
            ]
            .copy()
        )


        flagged_display[
            "£ Movement"
        ] = flagged_display[
            "£ Movement"
        ].map(
            format_currency
        )


        flagged_display[
            "% Movement"
        ] = flagged_display[
            "% Movement"
        ].map(
            format_percentage
        )


        st.dataframe(
            flagged_display,
            hide_index=True,
            width="stretch",
        )


# ============================================================
# TAB 4 — COMMON-SIZE ANALYSIS
# ============================================================

with tab4:

    st.header(
        "Common-Size / Vertical Analysis"
    )

    st.write(
        """
        Common-size analysis expresses financial statement lines
        relative to a common base, allowing changes in financial
        structure to be compared across years.
        """
    )


    st.subheader(
        "Common-Size Income Statement"
    )

    st.caption(
        """
        Income statement lines are shown as a percentage of
        revenue.
        """
    )


    common_income = (
        build_common_size_income_statement(
            analysis_df
        )
    )


    common_income_display = (
        common_income.copy()
    )


    common_income_display[
        "Amount"
    ] = common_income_display[
        "Amount"
    ].map(
        format_currency
    )


    common_income_display[
        "% of Revenue"
    ] = common_income_display[
        "% of Revenue"
    ].map(
        format_percentage
    )


    st.dataframe(
        common_income_display,
        hide_index=True,
        width="stretch",
    )


    st.subheader(
        "Margin Trend"
    )


    margin_chart = analysis_df[
        [
            "year",
            "gross_margin",
            "operating_margin",
            "net_margin",
        ]
    ].copy()


    margin_chart[
        [
            "gross_margin",
            "operating_margin",
            "net_margin",
        ]
    ] = (
        margin_chart[
            [
                "gross_margin",
                "operating_margin",
                "net_margin",
            ]
        ]
        * 100
    )


    margin_chart = (
        margin_chart
        .set_index(
            "year"
        )
        .rename(
            columns={
                "gross_margin": (
                    "Gross Margin %"
                ),
                "operating_margin": (
                    "Operating Margin %"
                ),
                "net_margin": (
                    "Net Margin %"
                ),
            }
        )
    )


    st.line_chart(
        margin_chart
    )


    st.subheader(
        "Common-Size Balance Sheet"
    )

    st.caption(
        """
        Balance sheet lines are shown as a percentage of
        total assets.
        """
    )


    common_balance = (
        build_common_size_balance_sheet(
            analysis_df
        )
    )


    common_balance_display = (
        common_balance.copy()
    )


    common_balance_display[
        "Amount"
    ] = common_balance_display[
        "Amount"
    ].map(
        format_currency
    )


    common_balance_display[
        "% of Total Assets"
    ] = common_balance_display[
        "% of Total Assets"
    ].map(
        format_percentage
    )


    st.dataframe(
        common_balance_display,
        hide_index=True,
        width="stretch",
    )


# ============================================================
# TAB 5 — RATIOS & INTERPRETATION
# ============================================================

with tab5:

    st.header(
        "Financial Ratios & Analytical Interpretation"
    )


    st.subheader(
        "Key Financial Ratios"
    )


    st.dataframe(
        prepare_ratio_table(
            analysis_df
        ),
        hide_index=True,
        width="stretch",
    )


    st.subheader(
        "Working-Capital Efficiency"
    )


    working_capital_chart = (
        analysis_df[
            [
                "year",
                "receivable_days",
                "inventory_days",
                "payable_days",
                "cash_conversion_cycle",
            ]
        ]
        .copy()
        .set_index(
            "year"
        )
        .rename(
            columns={
                "receivable_days": (
                    "Receivable Days"
                ),
                "inventory_days": (
                    "Inventory Days"
                ),
                "payable_days": (
                    "Payable Days"
                ),
                "cash_conversion_cycle": (
                    "Cash Conversion Cycle"
                ),
            }
        )
    )


    st.line_chart(
        working_capital_chart
    )


    st.subheader(
        "Liquidity & Leverage Trend"
    )


    liquidity_chart = (
        analysis_df[
            [
                "year",
                "current_ratio",
                "quick_ratio",
                "debt_to_equity",
            ]
        ]
        .copy()
        .set_index(
            "year"
        )
        .rename(
            columns={
                "current_ratio": (
                    "Current Ratio"
                ),
                "quick_ratio": (
                    "Quick Ratio"
                ),
                "debt_to_equity": (
                    "Debt to Equity"
                ),
            }
        )
    )


    st.line_chart(
        liquidity_chart
    )


    st.subheader(
        "Interest Coverage Trend"
    )


    interest_chart = (
        analysis_df[
            [
                "year",
                "interest_coverage",
            ]
        ]
        .copy()
        .set_index(
            "year"
        )
        .rename(
            columns={
                "interest_coverage": (
                    "Interest Coverage"
                ),
            }
        )
    )


    st.line_chart(
        interest_chart
    )


    st.subheader(
        "Automated Analytical Observations"
    )


    observations = (
        generate_observations(
            analysis_df
        )
    )


    if not observations:
        st.info(
            """
            No automated observations are available for the
            supplied dataset.
            """
        )

    else:
        for (
            title,
            observation,
            interpretation,
        ) in observations:

            with st.expander(
                title,
                expanded=True,
            ):
                st.write(
                    f"**Observed:** {observation}"
                )

                st.write(
                    f"**Interpretation:** {interpretation}"
                )


    st.warning(
        """
        These observations are analytical prompts only. They
        identify relationships or movements that may warrant
        investigation but do not establish their underlying
        cause.
        """
    )

# ============================================================
# TAB 6 — FORECASTING
# ============================================================

with tab6:

    render_forecasting_tab(
        analysis_df
    )

# ============================================================
# TAB 7 — SCENARIO ANALYSIS
# ============================================================

with tab7:

    render_scenario_analysis_tab(
        analysis_df
    )

# ============================================================
# TAB 8 — MANAGEMENT REVIEW
# ============================================================

with tab8:

    render_management_review_tab(
        analysis_df
    )

# ============================================================
# TAB 9 — EXCEL EXPORT
# ============================================================

with tab9:

    render_reporting_export_tab(
        analysis_df
    )



# ============================================================
# DEVELOPMENT STATUS
# ============================================================

st.divider()

st.caption(
    """
    Part 7 complete: professional Excel reporting and export,
    including historical statements, analytical schedules,
    forecast assumptions, projected statements, scenario
    comparison, sensitivity testing, review points, management
    questions and accounting model checks.

    Final deployment and portfolio release will be completed next.
    """
)