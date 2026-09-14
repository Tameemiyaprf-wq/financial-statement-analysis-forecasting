import numpy as np
import pandas as pd
import streamlit as st


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


def clamp(
    value,
    minimum,
    maximum,
):
    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


# ============================================================
# DEFAULT FORECAST ASSUMPTIONS
# ============================================================

def derive_default_assumptions(
    analysis_df,
):
    latest = analysis_df.iloc[-1]

    revenue_growth_history = (
        analysis_df["revenue"]
        .pct_change()
        .dropna()
    )

    if revenue_growth_history.empty:
        revenue_growth = 0.05

    else:
        revenue_growth = float(
            revenue_growth_history.mean()
        )

    revenue_growth = clamp(
        revenue_growth,
        -0.20,
        0.40,
    )

    gross_margin = safe_divide(
        latest["gross_profit"],
        latest["revenue"],
    )

    operating_expense_ratio = (
        safe_divide(
            latest["operating_expenses"],
            latest["revenue"],
        )
    )

    tax_rate = safe_divide(
        latest["tax_expense"],
        latest["profit_before_tax"],
    )

    if pd.isna(tax_rate):
        tax_rate = 0.20

    other_current_assets = max(
        0.0,
        (
            latest["current_assets"]
            - latest["cash"]
            - latest["trade_receivables"]
            - latest["inventory"]
        ),
    )

    other_current_asset_ratio = (
        safe_divide(
            other_current_assets,
            latest["revenue"],
        )
    )

    other_current_liabilities = max(
        0.0,
        (
            latest["current_liabilities"]
            - latest["trade_payables"]
        ),
    )

    other_current_liability_ratio = (
        safe_divide(
            other_current_liabilities,
            latest["revenue"],
        )
    )

    capex_ratio = safe_divide(
        latest["capital_expenditure"],
        latest["revenue"],
    )

    return {
        "revenue_growth": (
            0.05
            if pd.isna(revenue_growth)
            else revenue_growth
        ),

        "gross_margin": (
            0.40
            if pd.isna(gross_margin)
            else float(gross_margin)
        ),

        "operating_expense_ratio": (
            0.25
            if pd.isna(
                operating_expense_ratio
            )
            else float(
                operating_expense_ratio
            )
        ),

        "interest_growth": 0.00,

        "tax_rate": float(
            clamp(
                tax_rate,
                0.00,
                0.50,
            )
        ),

        "receivable_days": float(
            latest["receivable_days"]
        ),

        "inventory_days": float(
            latest["inventory_days"]
        ),

        "payable_days": float(
            latest["payable_days"]
        ),

        "other_current_asset_ratio": (
            0.03
            if pd.isna(
                other_current_asset_ratio
            )
            else max(
                0.0,
                float(
                    other_current_asset_ratio
                ),
            )
        ),

        "other_current_liability_ratio": (
            0.05
            if pd.isna(
                other_current_liability_ratio
            )
            else max(
                0.0,
                float(
                    other_current_liability_ratio
                ),
            )
        ),

        "capex_ratio": (
            0.05
            if pd.isna(capex_ratio)
            else max(
                0.0,
                float(capex_ratio),
            )
        ),

        "depreciation_rate": 0.10,

        "long_term_debt_growth": 0.00,

        "dividend_payout_ratio": 0.00,
    }


# ============================================================
# WORKING CAPITAL HELPERS
# ============================================================

def previous_operating_working_capital(
    previous,
):
    previous_other_current_assets = (
        previous.get(
            "other_current_assets",
            (
                previous["current_assets"]
                - previous["cash"]
                - previous[
                    "trade_receivables"
                ]
                - previous["inventory"]
            ),
        )
    )

    previous_other_current_liabilities = (
        previous.get(
            "other_current_liabilities",
            (
                previous[
                    "current_liabilities"
                ]
                - previous[
                    "trade_payables"
                ]
            ),
        )
    )

    return (
        previous[
            "trade_receivables"
        ]
        + previous["inventory"]
        + previous_other_current_assets
        - previous[
            "trade_payables"
        ]
        - previous_other_current_liabilities
    )


# ============================================================
# FORECAST ENGINE
# ============================================================

def build_forecast(
    analysis_df,
    assumptions,
    forecast_years,
):
    previous = (
        analysis_df
        .iloc[-1]
        .to_dict()
    )

    forecast_rows = []

    for _ in range(
        forecast_years
    ):
        year = (
            int(previous["year"])
            + 1
        )

        # ----------------------------------------------------
        # INCOME STATEMENT
        # ----------------------------------------------------

        revenue = (
            previous["revenue"]
            * (
                1
                + assumptions[
                    "revenue_growth"
                ]
            )
        )

        cost_of_sales = (
            revenue
            * (
                1
                - assumptions[
                    "gross_margin"
                ]
            )
        )

        gross_profit = (
            revenue
            - cost_of_sales
        )

        operating_expenses = (
            revenue
            * assumptions[
                "operating_expense_ratio"
            ]
        )

        operating_profit = (
            gross_profit
            - operating_expenses
        )

        interest_expense = max(
            0.0,
            (
                previous[
                    "interest_expense"
                ]
                * (
                    1
                    + assumptions[
                        "interest_growth"
                    ]
                )
            ),
        )

        profit_before_tax = (
            operating_profit
            - interest_expense
        )

        tax_expense = (
            max(
                0.0,
                profit_before_tax,
            )
            * assumptions[
                "tax_rate"
            ]
        )

        net_profit = (
            profit_before_tax
            - tax_expense
        )

        # ----------------------------------------------------
        # WORKING CAPITAL
        # ----------------------------------------------------

        trade_receivables = (
            revenue
            * assumptions[
                "receivable_days"
            ]
            / 365
        )

        inventory = (
            cost_of_sales
            * assumptions[
                "inventory_days"
            ]
            / 365
        )

        trade_payables = (
            cost_of_sales
            * assumptions[
                "payable_days"
            ]
            / 365
        )

        other_current_assets = (
            revenue
            * assumptions[
                "other_current_asset_ratio"
            ]
        )

        other_current_liabilities = (
            revenue
            * assumptions[
                "other_current_liability_ratio"
            ]
        )

        # ----------------------------------------------------
        # PPE / CAPITAL EXPENDITURE
        # ----------------------------------------------------

        capital_expenditure = (
            revenue
            * assumptions[
                "capex_ratio"
            ]
        )

        depreciation = (
            previous[
                "property_plant_equipment"
            ]
            * assumptions[
                "depreciation_rate"
            ]
        )

        property_plant_equipment = max(
            0.0,
            (
                previous[
                    "property_plant_equipment"
                ]
                + capital_expenditure
                - depreciation
            ),
        )

        # ----------------------------------------------------
        # LIABILITIES / DEBT
        # ----------------------------------------------------

        current_liabilities = (
            trade_payables
            + other_current_liabilities
        )

        long_term_debt = max(
            0.0,
            (
                previous[
                    "long_term_debt"
                ]
                * (
                    1
                    + assumptions[
                        "long_term_debt_growth"
                    ]
                )
            ),
        )

        change_in_debt = (
            long_term_debt
            - previous[
                "long_term_debt"
            ]
        )

        # ----------------------------------------------------
        # DIVIDENDS / EQUITY
        # ----------------------------------------------------

        dividends = (
            max(
                0.0,
                net_profit,
            )
            * assumptions[
                "dividend_payout_ratio"
            ]
        )

        # ----------------------------------------------------
        # OPERATING WORKING CAPITAL
        # ----------------------------------------------------

        operating_working_capital = (
            trade_receivables
            + inventory
            + other_current_assets
            - trade_payables
            - other_current_liabilities
        )

        prior_operating_working_capital = (
            previous_operating_working_capital(
                previous
            )
        )

        change_in_working_capital = (
            operating_working_capital
            - prior_operating_working_capital
        )

        # ----------------------------------------------------
        # CASH FLOW
        # ----------------------------------------------------

        operating_cash_flow = (
            net_profit
            + depreciation
            - change_in_working_capital
        )

        free_cash_flow = (
            operating_cash_flow
            - capital_expenditure
        )

        cash = (
            previous["cash"]
            + operating_cash_flow
            - capital_expenditure
            + change_in_debt
            - dividends
        )

        # ----------------------------------------------------
        # BALANCE SHEET
        # ----------------------------------------------------

        current_assets = (
            cash
            + trade_receivables
            + inventory
            + other_current_assets
        )

        total_assets = (
            current_assets
            + property_plant_equipment
        )

        total_liabilities = (
            current_liabilities
            + long_term_debt
        )

        equity = (
            previous["equity"]
            + net_profit
            - dividends
        )

        balance_check = (
            total_assets
            - total_liabilities
            - equity
        )

        working_capital = (
            current_assets
            - current_liabilities
        )

        # ----------------------------------------------------
        # FORECAST RATIOS
        # ----------------------------------------------------

        gross_margin = safe_divide(
            gross_profit,
            revenue,
        )

        operating_margin = (
            safe_divide(
                operating_profit,
                revenue,
            )
        )

        net_margin = safe_divide(
            net_profit,
            revenue,
        )

        current_ratio = safe_divide(
            current_assets,
            current_liabilities,
        )

        quick_ratio = safe_divide(
            (
                current_assets
                - inventory
            ),
            current_liabilities,
        )

        debt_to_equity = safe_divide(
            total_liabilities,
            equity,
        )

        interest_coverage = (
            safe_divide(
                operating_profit,
                interest_expense,
            )
        )

        return_on_assets = (
            safe_divide(
                net_profit,
                total_assets,
            )
        )

        return_on_equity = (
            safe_divide(
                net_profit,
                equity,
            )
        )

        receivable_days = (
            safe_divide(
                trade_receivables,
                revenue,
            )
            * 365
        )

        inventory_days = (
            safe_divide(
                inventory,
                cost_of_sales,
            )
            * 365
        )

        payable_days = (
            safe_divide(
                trade_payables,
                cost_of_sales,
            )
            * 365
        )

        cash_conversion_cycle = (
            receivable_days
            + inventory_days
            - payable_days
        )

        # ----------------------------------------------------
        # SAVE FORECAST YEAR
        # ----------------------------------------------------

        row = {
            "year": year,
            "period_type": "Forecast",

            "revenue": revenue,
            "cost_of_sales": (
                cost_of_sales
            ),
            "gross_profit": gross_profit,
            "operating_expenses": (
                operating_expenses
            ),
            "operating_profit": (
                operating_profit
            ),
            "interest_expense": (
                interest_expense
            ),
            "profit_before_tax": (
                profit_before_tax
            ),
            "tax_expense": tax_expense,
            "net_profit": net_profit,

            "cash": cash,
            "trade_receivables": (
                trade_receivables
            ),
            "inventory": inventory,
            "other_current_assets": (
                other_current_assets
            ),
            "current_assets": (
                current_assets
            ),
            "property_plant_equipment": (
                property_plant_equipment
            ),
            "total_assets": total_assets,

            "trade_payables": (
                trade_payables
            ),
            "other_current_liabilities": (
                other_current_liabilities
            ),
            "current_liabilities": (
                current_liabilities
            ),
            "long_term_debt": (
                long_term_debt
            ),
            "total_liabilities": (
                total_liabilities
            ),

            "equity": equity,

            "working_capital": (
                working_capital
            ),

            "operating_cash_flow": (
                operating_cash_flow
            ),
            "capital_expenditure": (
                capital_expenditure
            ),
            "depreciation": (
                depreciation
            ),
            "change_in_working_capital": (
                change_in_working_capital
            ),
            "dividends": dividends,
            "free_cash_flow": (
                free_cash_flow
            ),

            "gross_margin": (
                gross_margin
            ),
            "operating_margin": (
                operating_margin
            ),
            "net_margin": net_margin,

            "current_ratio": (
                current_ratio
            ),
            "quick_ratio": quick_ratio,
            "debt_to_equity": (
                debt_to_equity
            ),
            "interest_coverage": (
                interest_coverage
            ),

            "return_on_assets": (
                return_on_assets
            ),
            "return_on_equity": (
                return_on_equity
            ),

            "receivable_days": (
                receivable_days
            ),
            "inventory_days": (
                inventory_days
            ),
            "payable_days": (
                payable_days
            ),
            "cash_conversion_cycle": (
                cash_conversion_cycle
            ),

            "balance_check": (
                balance_check
            ),
        }

        forecast_rows.append(
            row
        )

        previous = row

    return pd.DataFrame(
        forecast_rows
    )


# ============================================================
# ASSUMPTION TABLE
# ============================================================

def build_assumption_table(
    assumptions,
):
    rows = [
        [
            "Revenue growth",
            format_percentage(
                assumptions[
                    "revenue_growth"
                ]
            ),
            "Applied to prior-year revenue",
        ],

        [
            "Gross margin",
            format_percentage(
                assumptions[
                    "gross_margin"
                ]
            ),
            "Gross profit as % of revenue",
        ],

        [
            "Operating expenses",
            format_percentage(
                assumptions[
                    "operating_expense_ratio"
                ]
            ),
            "Operating expenses as % of revenue",
        ],

        [
            "Interest expense growth",
            format_percentage(
                assumptions[
                    "interest_growth"
                ]
            ),
            "Annual movement in interest expense",
        ],

        [
            "Tax rate",
            format_percentage(
                assumptions[
                    "tax_rate"
                ]
            ),
            "Applied to positive profit before tax",
        ],

        [
            "Receivable days",
            format_days(
                assumptions[
                    "receivable_days"
                ]
            ),
            "Used to forecast trade receivables",
        ],

        [
            "Inventory days",
            format_days(
                assumptions[
                    "inventory_days"
                ]
            ),
            "Used to forecast inventory",
        ],

        [
            "Payable days",
            format_days(
                assumptions[
                    "payable_days"
                ]
            ),
            "Used to forecast trade payables",
        ],

        [
            "Other current assets",
            format_percentage(
                assumptions[
                    "other_current_asset_ratio"
                ]
            ),
            "Other current assets as % of revenue",
        ],

        [
            "Other current liabilities",
            format_percentage(
                assumptions[
                    "other_current_liability_ratio"
                ]
            ),
            "Other current liabilities as % of revenue",
        ],

        [
            "Capital expenditure",
            format_percentage(
                assumptions[
                    "capex_ratio"
                ]
            ),
            "Capital expenditure as % of revenue",
        ],

        [
            "PPE depreciation / runoff",
            format_percentage(
                assumptions[
                    "depreciation_rate"
                ]
            ),
            "Applied to opening PPE",
        ],

        [
            "Long-term debt growth",
            format_percentage(
                assumptions[
                    "long_term_debt_growth"
                ]
            ),
            "Annual movement in long-term debt",
        ],

        [
            "Dividend payout",
            format_percentage(
                assumptions[
                    "dividend_payout_ratio"
                ]
            ),
            "Share of positive net profit paid as dividends",
        ],
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "Assumption",
            "Value",
            "How It Is Used",
        ],
    )


# ============================================================
# DISPLAY TABLE HELPERS
# ============================================================

def prepare_currency_table(
    df,
    currency_columns,
):
    output = df.copy()

    for column in currency_columns:
        if column in output.columns:
            output[column] = (
                output[column]
                .map(format_currency)
            )

    return output


def prepare_forecast_ratio_table(
    forecast_df,
):
    output = forecast_df[
        [
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
    ].copy()

    for column in [
        "gross_margin",
        "operating_margin",
        "net_margin",
        "return_on_assets",
        "return_on_equity",
    ]:
        output[column] = (
            output[column]
            .map(format_percentage)
        )

    for column in [
        "current_ratio",
        "quick_ratio",
        "debt_to_equity",
        "interest_coverage",
    ]:
        output[column] = (
            output[column]
            .map(format_ratio)
        )

    for column in [
        "receivable_days",
        "inventory_days",
        "payable_days",
        "cash_conversion_cycle",
    ]:
        output[column] = (
            output[column]
            .map(format_days)
        )

    return output.rename(
        columns={
            "year": "Year",
            "gross_margin": (
                "Gross Margin"
            ),
            "operating_margin": (
                "Operating Margin"
            ),
            "net_margin": (
                "Net Margin"
            ),
            "current_ratio": (
                "Current Ratio"
            ),
            "quick_ratio": (
                "Quick Ratio"
            ),
            "debt_to_equity": (
                "Debt to Equity"
            ),
            "interest_coverage": (
                "Interest Coverage"
            ),
            "return_on_assets": (
                "Return on Assets"
            ),
            "return_on_equity": (
                "Return on Equity"
            ),
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


# ============================================================
# FORECAST INTERPRETATION
# ============================================================

def generate_forecast_prompts(
    analysis_df,
    forecast_df,
):
    latest = analysis_df.iloc[-1]
    final = forecast_df.iloc[-1]

    prompts = []

    revenue_change = (
        safe_divide(
            (
                final["revenue"]
                - latest["revenue"]
            ),
            latest["revenue"],
        )
    )

    if not pd.isna(
        revenue_change
    ):
        prompts.append(
            (
                "Revenue",
                (
                    f"Revenue reaches "
                    f"{format_currency(final['revenue'])} "
                    f"by {int(final['year'])}, a cumulative "
                    f"change of {revenue_change:.1%} from "
                    f"{int(latest['year'])}."
                ),
            )
        )

    prompts.append(
        (
            "Net margin",
            (
                f"Net margin changes from "
                f"{latest['net_margin']:.1%} "
                f"to {final['net_margin']:.1%} "
                "under the selected assumptions."
            ),
        )
    )

    cash_cycle_change = (
        final[
            "cash_conversion_cycle"
        ]
        - latest[
            "cash_conversion_cycle"
        ]
    )

    prompts.append(
        (
            "Working capital",
            (
                "Cash conversion cycle changes by "
                f"{cash_cycle_change:+.1f} days "
                "from the latest historical year."
            ),
        )
    )

    prompts.append(
        (
            "Cash",
            (
                f"Forecast closing cash is "
                f"{format_currency(final['cash'])} "
                f"in {int(final['year'])}."
            ),
        )
    )

    return prompts


# ============================================================
# STREAMLIT FORECASTING TAB
# ============================================================

def render_forecasting_tab(
    analysis_df,
):
    st.header(
        "Assumption-Driven Financial Forecast"
    )

    st.write(
        """
        Build a simplified forward-looking model from the latest
        historical financial statements. The forecast uses
        explicit assumptions rather than treating historical
        trends as certain predictions.
        """
    )

    st.warning(
        """
        Forecasts are estimates, not facts. The model does not
        predict future performance, assess going concern, provide
        audit assurance or replace a detailed budget and cash-flow
        model.
        """
    )

    defaults = (
        derive_default_assumptions(
            analysis_df
        )
    )

    # --------------------------------------------------------
    # FORECAST HORIZON
    # --------------------------------------------------------

    forecast_years = st.slider(
        "Forecast horizon (years)",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        key="forecast_years",
    )

    # --------------------------------------------------------
    # ASSUMPTIONS
    # --------------------------------------------------------

    st.subheader(
        "Forecast Assumptions"
    )

    st.caption(
        """
        Starting values are derived from the latest historical
        data where appropriate. Every assumption can be edited.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        revenue_growth = (
            st.number_input(
                "Annual revenue growth (%)",
                min_value=-50.0,
                max_value=100.0,
                value=float(
                    round(
                        defaults[
                            "revenue_growth"
                        ]
                        * 100,
                        1,
                    )
                ),
                step=0.5,
                key="forecast_revenue_growth",
            )
            / 100
        )

        gross_margin = (
            st.number_input(
                "Gross margin (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(
                    round(
                        defaults[
                            "gross_margin"
                        ]
                        * 100,
                        1,
                    )
                ),
                step=0.5,
                key="forecast_gross_margin",
            )
            / 100
        )

        operating_expense_ratio = (
            st.number_input(
                "Operating expenses (% of revenue)",
                min_value=0.0,
                max_value=100.0,
                value=float(
                    round(
                        defaults[
                            "operating_expense_ratio"
                        ]
                        * 100,
                        1,
                    )
                ),
                step=0.5,
                key="forecast_opex_ratio",
            )
            / 100
        )

        interest_growth = (
            st.number_input(
                "Annual interest expense growth (%)",
                min_value=-100.0,
                max_value=100.0,
                value=0.0,
                step=0.5,
                key="forecast_interest_growth",
            )
            / 100
        )

        tax_rate = (
            st.number_input(
                "Tax rate (%)",
                min_value=0.0,
                max_value=60.0,
                value=float(
                    round(
                        defaults[
                            "tax_rate"
                        ]
                        * 100,
                        1,
                    )
                ),
                step=0.5,
                key="forecast_tax_rate",
            )
            / 100
        )

        receivable_days = (
            st.number_input(
                "Receivable days",
                min_value=0.0,
                max_value=365.0,
                value=float(
                    round(
                        defaults[
                            "receivable_days"
                        ],
                        1,
                    )
                ),
                step=1.0,
                key="forecast_receivable_days",
            )
        )

        inventory_days = (
            st.number_input(
                "Inventory days",
                min_value=0.0,
                max_value=365.0,
                value=float(
                    round(
                        defaults[
                            "inventory_days"
                        ],
                        1,
                    )
                ),
                step=1.0,
                key="forecast_inventory_days",
            )
        )

    with col2:

        payable_days = (
            st.number_input(
                "Payable days",
                min_value=0.0,
                max_value=365.0,
                value=float(
                    round(
                        defaults[
                            "payable_days"
                        ],
                        1,
                    )
                ),
                step=1.0,
                key="forecast_payable_days",
            )
        )

        other_current_asset_ratio = (
            st.number_input(
                "Other current assets (% of revenue)",
                min_value=0.0,
                max_value=50.0,
                value=float(
                    round(
                        defaults[
                            "other_current_asset_ratio"
                        ]
                        * 100,
                        1,
                    )
                ),
                step=0.5,
                key="forecast_other_ca",
            )
            / 100
        )

        other_current_liability_ratio = (
            st.number_input(
                "Other current liabilities (% of revenue)",
                min_value=0.0,
                max_value=50.0,
                value=float(
                    round(
                        defaults[
                            "other_current_liability_ratio"
                        ]
                        * 100,
                        1,
                    )
                ),
                step=0.5,
                key="forecast_other_cl",
            )
            / 100
        )

        capex_ratio = (
            st.number_input(
                "Capital expenditure (% of revenue)",
                min_value=0.0,
                max_value=50.0,
                value=float(
                    round(
                        defaults[
                            "capex_ratio"
                        ]
                        * 100,
                        1,
                    )
                ),
                step=0.5,
                key="forecast_capex",
            )
            / 100
        )

        depreciation_rate = (
            st.number_input(
                "PPE depreciation / runoff (% of opening PPE)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=0.5,
                key="forecast_depreciation",
            )
            / 100
        )

        long_term_debt_growth = (
            st.number_input(
                "Annual long-term debt growth (%)",
                min_value=-100.0,
                max_value=100.0,
                value=0.0,
                step=0.5,
                key="forecast_debt_growth",
            )
            / 100
        )

        dividend_payout_ratio = (
            st.number_input(
                "Dividend payout (% of positive net profit)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                key="forecast_dividend_payout",
            )
            / 100
        )

    assumptions = {
        "revenue_growth": (
            revenue_growth
        ),
        "gross_margin": (
            gross_margin
        ),
        "operating_expense_ratio": (
            operating_expense_ratio
        ),
        "interest_growth": (
            interest_growth
        ),
        "tax_rate": (
            tax_rate
        ),
        "receivable_days": (
            receivable_days
        ),
        "inventory_days": (
            inventory_days
        ),
        "payable_days": (
            payable_days
        ),
        "other_current_asset_ratio": (
            other_current_asset_ratio
        ),
        "other_current_liability_ratio": (
            other_current_liability_ratio
        ),
        "capex_ratio": (
            capex_ratio
        ),
        "depreciation_rate": (
            depreciation_rate
        ),
        "long_term_debt_growth": (
            long_term_debt_growth
        ),
        "dividend_payout_ratio": (
            dividend_payout_ratio
        ),
    }

    with st.expander(
        "View assumption summary"
    ):
        st.dataframe(
            build_assumption_table(
                assumptions
            ),
            hide_index=True,
            width="stretch",
        )

    # --------------------------------------------------------
    # BUILD FORECAST
    # --------------------------------------------------------

    forecast_df = build_forecast(
        analysis_df,
        assumptions,
        forecast_years,
    )

    latest = analysis_df.iloc[-1]
    final = forecast_df.iloc[-1]

    revenue_change = safe_divide(
        (
            final["revenue"]
            - latest["revenue"]
        ),
        latest["revenue"],
    )

    if latest["net_profit"] != 0:
        profit_change = safe_divide(
            (
                final["net_profit"]
                - latest["net_profit"]
            ),
            abs(
                latest["net_profit"]
            ),
        )

    else:
        profit_change = np.nan

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        f"Forecast Summary — {int(final['year'])}"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:
        st.metric(
            "Revenue",
            format_currency(
                final["revenue"]
            ),
            (
                f"{revenue_change:+.1%} "
                "vs latest actual"
            ),
        )

    with col2:
        st.metric(
            "Net Profit",
            format_currency(
                final["net_profit"]
            ),
            (
                None
                if pd.isna(
                    profit_change
                )
                else (
                    f"{profit_change:+.1%} "
                    "vs latest actual"
                )
            ),
        )

    with col3:
        st.metric(
            "Closing Cash",
            format_currency(
                final["cash"]
            ),
            format_currency(
                final["cash"]
                - latest["cash"]
            ),
        )

    with col4:
        st.metric(
            "Current Ratio",
            format_ratio(
                final["current_ratio"]
            ),
            (
                f"{(
                    final['current_ratio']
                    - latest['current_ratio']
                ):+.2f}x"
            ),
            delta_color="off",
        )

    # --------------------------------------------------------
    # COMBINED HISTORICAL / FORECAST TREND
    # --------------------------------------------------------

    st.subheader(
        "Historical + Forecast Trend"
    )

    historical_chart = (
        analysis_df[
            [
                "year",
                "revenue",
                "net_profit",
                "free_cash_flow",
            ]
        ]
        .copy()
    )

    forecast_chart = (
        forecast_df[
            [
                "year",
                "revenue",
                "net_profit",
                "free_cash_flow",
            ]
        ]
        .copy()
    )

    combined_chart = pd.concat(
        [
            historical_chart,
            forecast_chart,
        ],
        ignore_index=True,
    )

    combined_chart = (
        combined_chart
        .set_index(
            "year"
        )
        .rename(
            columns={
                "revenue": "Revenue",
                "net_profit": "Net Profit",
                "free_cash_flow": "Free Cash Flow",
            }
        )
    )

    st.line_chart(
        combined_chart
    )

    st.caption(
        """
        Historical and forecast periods are displayed together
        for continuity. Forecast points remain estimates driven
        by the assumptions above.
        """
    )

    # --------------------------------------------------------
    # FORECAST INCOME STATEMENT
    # --------------------------------------------------------

    st.subheader(
        "Forecast Income Statement"
    )

    income_columns = [
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

    income_display = (
        prepare_currency_table(
            forecast_df[
                income_columns
            ].copy(),
            [
                column
                for column
                in income_columns
                if column != "year"
            ],
        )
        .rename(
            columns={
                "year": "Year",
                "revenue": "Revenue",
                "cost_of_sales": (
                    "Cost of Sales"
                ),
                "gross_profit": (
                    "Gross Profit"
                ),
                "operating_expenses": (
                    "Operating Expenses"
                ),
                "operating_profit": (
                    "Operating Profit"
                ),
                "interest_expense": (
                    "Interest Expense"
                ),
                "profit_before_tax": (
                    "Profit Before Tax"
                ),
                "tax_expense": (
                    "Tax Expense"
                ),
                "net_profit": (
                    "Net Profit"
                ),
            }
        )
    )

    st.dataframe(
        income_display,
        hide_index=True,
        width="stretch",
    )

    # --------------------------------------------------------
    # FORECAST BALANCE SHEET
    # --------------------------------------------------------

    st.subheader(
        "Forecast Balance Sheet"
    )

    balance_columns = [
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

    balance_display = (
        prepare_currency_table(
            forecast_df[
                balance_columns
            ].copy(),
            [
                column
                for column
                in balance_columns
                if column != "year"
            ],
        )
        .rename(
            columns={
                "year": "Year",
                "cash": "Cash",
                "trade_receivables": (
                    "Trade Receivables"
                ),
                "inventory": (
                    "Inventory"
                ),
                "other_current_assets": (
                    "Other Current Assets"
                ),
                "current_assets": (
                    "Current Assets"
                ),
                "property_plant_equipment": (
                    "Property, Plant & Equipment"
                ),
                "total_assets": (
                    "Total Assets"
                ),
                "trade_payables": (
                    "Trade Payables"
                ),
                "other_current_liabilities": (
                    "Other Current Liabilities"
                ),
                "current_liabilities": (
                    "Current Liabilities"
                ),
                "long_term_debt": (
                    "Long-Term Debt"
                ),
                "total_liabilities": (
                    "Total Liabilities"
                ),
                "equity": (
                    "Equity"
                ),
                "working_capital": (
                    "Working Capital"
                ),
            }
        )
    )

    st.dataframe(
        balance_display,
        hide_index=True,
        width="stretch",
    )

    # --------------------------------------------------------
    # FORECAST CASH FLOW
    # --------------------------------------------------------

    st.subheader(
        "Forecast Cash Flow"
    )

    cash_columns = [
        "year",
        "net_profit",
        "depreciation",
        "change_in_working_capital",
        "operating_cash_flow",
        "capital_expenditure",
        "free_cash_flow",
        "dividends",
    ]

    cash_display = (
        prepare_currency_table(
            forecast_df[
                cash_columns
            ].copy(),
            [
                column
                for column
                in cash_columns
                if column != "year"
            ],
        )
        .rename(
            columns={
                "year": "Year",
                "net_profit": (
                    "Net Profit"
                ),
                "depreciation": (
                    "Depreciation / Runoff"
                ),
                "change_in_working_capital": (
                    "Change in Operating Working Capital"
                ),
                "operating_cash_flow": (
                    "Operating Cash Flow"
                ),
                "capital_expenditure": (
                    "Capital Expenditure"
                ),
                "free_cash_flow": (
                    "Free Cash Flow"
                ),
                "dividends": (
                    "Dividends"
                ),
            }
        )
    )

    st.dataframe(
        cash_display,
        hide_index=True,
        width="stretch",
    )

    st.caption(
        """
        Operating cash flow is modelled using net profit plus
        depreciation/runoff less the increase in operating
        working capital. This is a simplified forecast bridge,
        not a statutory cash-flow statement.
        """
    )

    # --------------------------------------------------------
    # FORECAST RATIOS
    # --------------------------------------------------------

    st.subheader(
        "Forecast Ratios"
    )

    st.dataframe(
        prepare_forecast_ratio_table(
            forecast_df
        ),
        hide_index=True,
        width="stretch",
    )

    # --------------------------------------------------------
    # CASH TREND
    # --------------------------------------------------------

    st.subheader(
        "Cash & Free Cash Flow Trend"
    )

    cash_chart = (
        forecast_df[
            [
                "year",
                "cash",
                "operating_cash_flow",
                "free_cash_flow",
            ]
        ]
        .copy()
        .set_index(
            "year"
        )
        .rename(
            columns={
                "cash": (
                    "Closing Cash"
                ),
                "operating_cash_flow": (
                    "Operating Cash Flow"
                ),
                "free_cash_flow": (
                    "Free Cash Flow"
                ),
            }
        )
    )

    st.line_chart(
        cash_chart
    )

    # --------------------------------------------------------
    # BALANCE SHEET MODEL CHECK
    # --------------------------------------------------------

    st.subheader(
        "Forecast Model Check"
    )

    model_check = (
        forecast_df[
            [
                "year",
                "total_assets",
                "total_liabilities",
                "equity",
                "balance_check",
            ]
        ]
        .copy()
    )

    max_difference = float(
        model_check[
            "balance_check"
        ]
        .abs()
        .max()
    )

    if max_difference <= 1:
        st.success(
            """
            Balance-sheet check passed: forecast assets equal
            liabilities plus equity within the £1 rounding
            tolerance.
            """
        )

    else:
        st.error(
            """
            Balance-sheet check failed. Review forecast
            assumptions and calculations before relying on
            the model.
            """
        )

    model_check[
        "total_assets"
    ] = model_check[
        "total_assets"
    ].map(
        format_currency
    )

    model_check[
        "total_liabilities"
    ] = model_check[
        "total_liabilities"
    ].map(
        format_currency
    )

    model_check[
        "equity"
    ] = model_check[
        "equity"
    ].map(
        format_currency
    )

    model_check[
        "balance_check"
    ] = model_check[
        "balance_check"
    ].map(
        lambda value: (
            f"£{value:,.2f}"
        )
    )

    model_check = (
        model_check.rename(
            columns={
                "year": "Year",
                "total_assets": (
                    "Total Assets"
                ),
                "total_liabilities": (
                    "Total Liabilities"
                ),
                "equity": "Equity",
                "balance_check": (
                    "Assets - Liabilities - Equity"
                ),
            }
        )
    )

    st.dataframe(
        model_check,
        hide_index=True,
        width="stretch",
    )

    # --------------------------------------------------------
    # FORECAST INTERPRETATION
    # --------------------------------------------------------

    st.subheader(
        "Forecast Interpretation Prompts"
    )

    for title, text in (
        generate_forecast_prompts(
            analysis_df,
            forecast_df,
        )
    ):
        st.write(
            f"**{title}:** {text}"
        )

    st.info(
        """
        Interpretation should focus on the assumptions driving
        the outputs. A favourable forecast does not provide
        evidence that those assumptions will occur.
        """
    )