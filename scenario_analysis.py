import numpy as np
import pandas as pd
import streamlit as st

from forecasting_engine import (
    build_forecast,
    derive_default_assumptions,
    format_currency,
    format_percentage,
    format_ratio,
)


# ============================================================
# HELPERS
# ============================================================

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


def build_scenario_assumptions(
    defaults,
    base_inputs,
    upside_adjustments,
    downside_adjustments,
):
    """
    Create Base, Upside and Downside assumption dictionaries.

    Assumptions not explicitly changed here continue to use the
    default assumptions derived by the forecasting engine.
    """

    base = defaults.copy()

    base.update(
        {
            "revenue_growth": (
                base_inputs[
                    "revenue_growth"
                ]
            ),
            "gross_margin": (
                base_inputs[
                    "gross_margin"
                ]
            ),
            "operating_expense_ratio": (
                base_inputs[
                    "operating_expense_ratio"
                ]
            ),
            "receivable_days": (
                base_inputs[
                    "receivable_days"
                ]
            ),
            "inventory_days": (
                base_inputs[
                    "inventory_days"
                ]
            ),
            "payable_days": (
                base_inputs[
                    "payable_days"
                ]
            ),
        }
    )

    upside = base.copy()

    upside[
        "revenue_growth"
    ] = clamp(
        (
            base["revenue_growth"]
            + upside_adjustments[
                "revenue_growth"
            ]
        ),
        -0.50,
        1.00,
    )

    upside[
        "gross_margin"
    ] = clamp(
        (
            base["gross_margin"]
            + upside_adjustments[
                "gross_margin"
            ]
        ),
        0.00,
        1.00,
    )

    upside[
        "operating_expense_ratio"
    ] = clamp(
        (
            base[
                "operating_expense_ratio"
            ]
            + upside_adjustments[
                "operating_expense_ratio"
            ]
        ),
        0.00,
        1.00,
    )

    upside[
        "receivable_days"
    ] = max(
        0.0,
        (
            base["receivable_days"]
            + upside_adjustments[
                "receivable_days"
            ]
        ),
    )

    upside[
        "inventory_days"
    ] = max(
        0.0,
        (
            base["inventory_days"]
            + upside_adjustments[
                "inventory_days"
            ]
        ),
    )

    upside[
        "payable_days"
    ] = max(
        0.0,
        (
            base["payable_days"]
            + upside_adjustments[
                "payable_days"
            ]
        ),
    )

    downside = base.copy()

    downside[
        "revenue_growth"
    ] = clamp(
        (
            base["revenue_growth"]
            + downside_adjustments[
                "revenue_growth"
            ]
        ),
        -0.50,
        1.00,
    )

    downside[
        "gross_margin"
    ] = clamp(
        (
            base["gross_margin"]
            + downside_adjustments[
                "gross_margin"
            ]
        ),
        0.00,
        1.00,
    )

    downside[
        "operating_expense_ratio"
    ] = clamp(
        (
            base[
                "operating_expense_ratio"
            ]
            + downside_adjustments[
                "operating_expense_ratio"
            ]
        ),
        0.00,
        1.00,
    )

    downside[
        "receivable_days"
    ] = max(
        0.0,
        (
            base["receivable_days"]
            + downside_adjustments[
                "receivable_days"
            ]
        ),
    )

    downside[
        "inventory_days"
    ] = max(
        0.0,
        (
            base["inventory_days"]
            + downside_adjustments[
                "inventory_days"
            ]
        ),
    )

    downside[
        "payable_days"
    ] = max(
        0.0,
        (
            base["payable_days"]
            + downside_adjustments[
                "payable_days"
            ]
        ),
    )

    return {
        "Base": base,
        "Upside": upside,
        "Downside": downside,
    }


# ============================================================
# FORECAST ALL SCENARIOS
# ============================================================

def build_all_scenarios(
    analysis_df,
    scenario_assumptions,
    forecast_years,
):
    forecasts = {}

    for (
        scenario_name,
        assumptions,
    ) in scenario_assumptions.items():

        forecast = build_forecast(
            analysis_df,
            assumptions,
            forecast_years,
        )

        forecast[
            "scenario"
        ] = scenario_name

        forecasts[
            scenario_name
        ] = forecast

    return forecasts


# ============================================================
# ASSUMPTION COMPARISON TABLE
# ============================================================

def build_assumption_comparison(
    scenario_assumptions,
):
    rows = []

    for (
        scenario_name,
        assumptions,
    ) in scenario_assumptions.items():

        rows.append(
            {
                "Scenario": (
                    scenario_name
                ),

                "Revenue Growth": (
                    assumptions[
                        "revenue_growth"
                    ]
                ),

                "Gross Margin": (
                    assumptions[
                        "gross_margin"
                    ]
                ),

                "Operating Expenses": (
                    assumptions[
                        "operating_expense_ratio"
                    ]
                ),

                "Receivable Days": (
                    assumptions[
                        "receivable_days"
                    ]
                ),

                "Inventory Days": (
                    assumptions[
                        "inventory_days"
                    ]
                ),

                "Payable Days": (
                    assumptions[
                        "payable_days"
                    ]
                ),
            }
        )

    output = pd.DataFrame(
        rows
    )

    for column in [
        "Revenue Growth",
        "Gross Margin",
        "Operating Expenses",
    ]:
        output[column] = (
            output[column]
            .map(
                format_percentage
            )
        )

    for column in [
        "Receivable Days",
        "Inventory Days",
        "Payable Days",
    ]:
        output[column] = (
            output[column]
            .map(
                lambda value: (
                    f"{value:.1f}"
                )
            )
        )

    return output


# ============================================================
# SCENARIO SUMMARY
# ============================================================

def build_final_year_summary(
    forecasts,
):
    rows = []

    for (
        scenario_name,
        forecast,
    ) in forecasts.items():

        final = (
            forecast.iloc[-1]
        )

        rows.append(
            {
                "Scenario": (
                    scenario_name
                ),

                "Year": int(
                    final["year"]
                ),

                "Revenue": (
                    final["revenue"]
                ),

                "Net Profit": (
                    final["net_profit"]
                ),

                "Operating Margin": (
                    final[
                        "operating_margin"
                    ]
                ),

                "Closing Cash": (
                    final["cash"]
                ),

                "Free Cash Flow": (
                    final[
                        "free_cash_flow"
                    ]
                ),

                "Current Ratio": (
                    final[
                        "current_ratio"
                    ]
                ),

                "Debt to Equity": (
                    final[
                        "debt_to_equity"
                    ]
                ),

                "Cash Conversion Cycle": (
                    final[
                        "cash_conversion_cycle"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def format_final_year_summary(
    summary_df,
):
    output = summary_df.copy()

    for column in [
        "Revenue",
        "Net Profit",
        "Closing Cash",
        "Free Cash Flow",
    ]:
        output[column] = (
            output[column]
            .map(
                format_currency
            )
        )

    output[
        "Operating Margin"
    ] = output[
        "Operating Margin"
    ].map(
        format_percentage
    )

    for column in [
        "Current Ratio",
        "Debt to Equity",
    ]:
        output[column] = (
            output[column]
            .map(
                format_ratio
            )
        )

    output[
        "Cash Conversion Cycle"
    ] = output[
        "Cash Conversion Cycle"
    ].map(
        lambda value: (
            f"{value:.1f} days"
        )
    )

    return output


# ============================================================
# MODEL CHECK
# ============================================================

def build_model_check(
    forecasts,
):
    rows = []

    for (
        scenario_name,
        forecast,
    ) in forecasts.items():

        maximum_difference = float(
            forecast[
                "balance_check"
            ]
            .abs()
            .max()
        )

        rows.append(
            {
                "Scenario": (
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
# SCENARIO CHART
# ============================================================

def build_scenario_chart(
    forecasts,
    metric,
    metric_type,
):
    frames = []

    for (
        scenario_name,
        forecast,
    ) in forecasts.items():

        frame = forecast[
            [
                "year",
                metric,
            ]
        ].copy()

        frame[
            "Scenario"
        ] = scenario_name

        frames.append(
            frame
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    if metric_type == "percentage":
        combined[
            metric
        ] = (
            combined[
                metric
            ]
            * 100
        )

    chart = combined.pivot(
        index="year",
        columns="Scenario",
        values=metric,
    )

    scenario_order = [
        scenario
        for scenario in [
            "Base",
            "Upside",
            "Downside",
        ]
        if scenario in chart.columns
    ]

    return chart[
        scenario_order
    ]


# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

def build_sensitivity_table(
    analysis_df,
    base_assumptions,
    forecast_years,
    selected_metric,
):
    revenue_offsets = [
        -0.05,
        -0.025,
        0.00,
        0.025,
        0.05,
    ]

    margin_offsets = [
        -0.04,
        -0.02,
        0.00,
        0.02,
        0.04,
    ]

    sensitivity_rows = []

    for margin_offset in margin_offsets:

        row = {}

        gross_margin = clamp(
            (
                base_assumptions[
                    "gross_margin"
                ]
                + margin_offset
            ),
            0.00,
            1.00,
        )

        row_label = (
            f"{gross_margin:.1%}"
        )

        for revenue_offset in revenue_offsets:

            assumptions = (
                base_assumptions.copy()
            )

            revenue_growth = clamp(
                (
                    base_assumptions[
                        "revenue_growth"
                    ]
                    + revenue_offset
                ),
                -0.50,
                1.00,
            )

            assumptions[
                "revenue_growth"
            ] = revenue_growth

            assumptions[
                "gross_margin"
            ] = gross_margin

            forecast = build_forecast(
                analysis_df,
                assumptions,
                forecast_years,
            )

            final = (
                forecast.iloc[-1]
            )

            if (
                selected_metric
                == "Net Profit"
            ):
                value = (
                    final["net_profit"]
                )

            elif (
                selected_metric
                == "Closing Cash"
            ):
                value = (
                    final["cash"]
                )

            else:
                value = (
                    final[
                        "free_cash_flow"
                    ]
                )

            column_label = (
                f"{revenue_growth:.1%}"
            )

            row[
                column_label
            ] = value

        sensitivity_rows.append(
            (
                row_label,
                row,
            )
        )

    sensitivity_df = pd.DataFrame(
        [
            row
            for _, row
            in sensitivity_rows
        ],
        index=[
            label
            for label, _
            in sensitivity_rows
        ],
    )

    sensitivity_df.index.name = (
        "Gross Margin"
    )

    return sensitivity_df


def format_sensitivity_table(
    sensitivity_df,
):
    output = (
        sensitivity_df.copy()
    )

    for column in output.columns:
        output[column] = (
            output[column]
            .map(
                format_currency
            )
        )

    return output


# ============================================================
# COMMENTARY
# ============================================================

def generate_scenario_commentary(
    analysis_df,
    forecasts,
):
    base = (
        forecasts["Base"]
        .iloc[-1]
    )

    upside = (
        forecasts["Upside"]
        .iloc[-1]
    )

    downside = (
        forecasts["Downside"]
        .iloc[-1]
    )

    latest = (
        analysis_df.iloc[-1]
    )

    comments = []

    comments.append(
        (
            "Base case",
            (
                f"The Base scenario produces "
                f"{format_currency(base['revenue'])} "
                f"of revenue and "
                f"{format_currency(base['net_profit'])} "
                f"of net profit by "
                f"{int(base['year'])}."
            ),
        )
    )

    profit_spread = (
        upside["net_profit"]
        - downside["net_profit"]
    )

    comments.append(
        (
            "Scenario range",
            (
                "The difference between Upside and "
                "Downside final-year net profit is "
                f"{format_currency(profit_spread)}. "
                "This illustrates the sensitivity of "
                "forecast performance to the selected "
                "operating assumptions."
            ),
        )
    )

    cash_change = (
        downside["cash"]
        - latest["cash"]
    )

    comments.append(
        (
            "Downside cash",
            (
                "The Downside scenario produces "
                f"{format_currency(downside['cash'])} "
                "of closing cash, representing a change of "
                f"{format_currency(cash_change)} from the "
                "latest historical closing cash balance."
            ),
        )
    )

    if downside["cash"] < 0:
        comments.append(
            (
                "Liquidity warning",
                (
                    "The Downside scenario produces a "
                    "negative closing cash balance. This "
                    "would indicate a potential funding "
                    "requirement within the simplified model."
                ),
            )
        )

    if downside[
        "current_ratio"
    ] < 1:
        comments.append(
            (
                "Current liquidity",
                (
                    "The Downside scenario produces a "
                    "current ratio below 1.00x. This may "
                    "warrant closer review of short-term "
                    "liquidity and funding assumptions."
                ),
            )
        )

    if (
        downside[
            "cash_conversion_cycle"
        ]
        > base[
            "cash_conversion_cycle"
        ]
    ):
        difference = (
            downside[
                "cash_conversion_cycle"
            ]
            - base[
                "cash_conversion_cycle"
            ]
        )

        comments.append(
            (
                "Working capital",
                (
                    "The Downside cash-conversion cycle "
                    f"is {difference:.1f} days longer than "
                    "the Base scenario, indicating more "
                    "cash tied up in operating working "
                    "capital under those assumptions."
                ),
            )
        )

    return comments


# ============================================================
# STREAMLIT TAB
# ============================================================

def render_scenario_analysis_tab(
    analysis_df,
):
    st.header(
        "Scenario Analysis & Sensitivity Testing"
    )

    st.write(
        """
        Compare alternative forecast outcomes by changing key
        operating assumptions. The scenarios are not predictions;
        they show how selected assumptions flow through the
        financial model.
        """
    )

    st.warning(
        """
        Base, Upside and Downside are modelling labels rather than
        probabilities. A Downside case is not a forecast that the
        downside will occur, and an Upside case is not evidence
        that stronger performance is achievable.
        """
    )

    defaults = (
        derive_default_assumptions(
            analysis_df
        )
    )

    # ========================================================
    # HORIZON
    # ========================================================

    forecast_years = st.slider(
        "Scenario forecast horizon (years)",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        key="scenario_forecast_horizon",
    )

    # ========================================================
    # BASE ASSUMPTIONS
    # ========================================================

    st.subheader(
        "1. Base Scenario"
    )

    st.caption(
        """
        The Base scenario starts from assumptions derived from
        the latest historical data. You can edit the main
        operating drivers below.
        """
    )

    base_col1, base_col2 = (
        st.columns(2)
    )

    with base_col1:

        base_revenue_growth = (
            st.number_input(
                "Base revenue growth (%)",
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
                key="scenario_base_revenue",
            )
            / 100
        )

        base_gross_margin = (
            st.number_input(
                "Base gross margin (%)",
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
                key="scenario_base_margin",
            )
            / 100
        )

        base_opex = (
            st.number_input(
                "Base operating expenses (% of revenue)",
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
                key="scenario_base_opex",
            )
            / 100
        )

    with base_col2:

        base_receivable_days = (
            st.number_input(
                "Base receivable days",
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
                key="scenario_base_receivables",
            )
        )

        base_inventory_days = (
            st.number_input(
                "Base inventory days",
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
                key="scenario_base_inventory",
            )
        )

        base_payable_days = (
            st.number_input(
                "Base payable days",
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
                key="scenario_base_payables",
            )
        )

    base_inputs = {
        "revenue_growth": (
            base_revenue_growth
        ),
        "gross_margin": (
            base_gross_margin
        ),
        "operating_expense_ratio": (
            base_opex
        ),
        "receivable_days": (
            base_receivable_days
        ),
        "inventory_days": (
            base_inventory_days
        ),
        "payable_days": (
            base_payable_days
        ),
    }

    # ========================================================
    # UPSIDE ADJUSTMENTS
    # ========================================================

    st.subheader(
        "2. Upside Adjustments"
    )

    st.caption(
        """
        These values are adjustments to the Base assumptions.
        Percentage adjustments are entered in percentage points.
        """
    )

    up_col1, up_col2 = (
        st.columns(2)
    )

    with up_col1:

        upside_revenue = (
            st.number_input(
                "Upside revenue growth adjustment (pp)",
                min_value=-50.0,
                max_value=50.0,
                value=3.0,
                step=0.5,
                key="scenario_up_revenue",
            )
            / 100
        )

        upside_margin = (
            st.number_input(
                "Upside gross margin adjustment (pp)",
                min_value=-50.0,
                max_value=50.0,
                value=1.5,
                step=0.5,
                key="scenario_up_margin",
            )
            / 100
        )

        upside_opex = (
            st.number_input(
                "Upside operating expense adjustment (pp)",
                min_value=-50.0,
                max_value=50.0,
                value=-1.0,
                step=0.5,
                key="scenario_up_opex",
            )
            / 100
        )

    with up_col2:

        upside_receivables = (
            st.number_input(
                "Upside receivable-days adjustment",
                min_value=-100.0,
                max_value=100.0,
                value=-5.0,
                step=1.0,
                key="scenario_up_receivables",
            )
        )

        upside_inventory = (
            st.number_input(
                "Upside inventory-days adjustment",
                min_value=-100.0,
                max_value=100.0,
                value=-5.0,
                step=1.0,
                key="scenario_up_inventory",
            )
        )

        upside_payables = (
            st.number_input(
                "Upside payable-days adjustment",
                min_value=-100.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                key="scenario_up_payables",
            )
        )

    upside_adjustments = {
        "revenue_growth": (
            upside_revenue
        ),
        "gross_margin": (
            upside_margin
        ),
        "operating_expense_ratio": (
            upside_opex
        ),
        "receivable_days": (
            upside_receivables
        ),
        "inventory_days": (
            upside_inventory
        ),
        "payable_days": (
            upside_payables
        ),
    }

    # ========================================================
    # DOWNSIDE ADJUSTMENTS
    # ========================================================

    st.subheader(
        "3. Downside Adjustments"
    )

    down_col1, down_col2 = (
        st.columns(2)
    )

    with down_col1:

        downside_revenue = (
            st.number_input(
                "Downside revenue growth adjustment (pp)",
                min_value=-50.0,
                max_value=50.0,
                value=-5.0,
                step=0.5,
                key="scenario_down_revenue",
            )
            / 100
        )

        downside_margin = (
            st.number_input(
                "Downside gross margin adjustment (pp)",
                min_value=-50.0,
                max_value=50.0,
                value=-2.5,
                step=0.5,
                key="scenario_down_margin",
            )
            / 100
        )

        downside_opex = (
            st.number_input(
                "Downside operating expense adjustment (pp)",
                min_value=-50.0,
                max_value=50.0,
                value=1.5,
                step=0.5,
                key="scenario_down_opex",
            )
            / 100
        )

    with down_col2:

        downside_receivables = (
            st.number_input(
                "Downside receivable-days adjustment",
                min_value=-100.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
                key="scenario_down_receivables",
            )
        )

        downside_inventory = (
            st.number_input(
                "Downside inventory-days adjustment",
                min_value=-100.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
                key="scenario_down_inventory",
            )
        )

        downside_payables = (
            st.number_input(
                "Downside payable-days adjustment",
                min_value=-100.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                key="scenario_down_payables",
            )
        )

    downside_adjustments = {
        "revenue_growth": (
            downside_revenue
        ),
        "gross_margin": (
            downside_margin
        ),
        "operating_expense_ratio": (
            downside_opex
        ),
        "receivable_days": (
            downside_receivables
        ),
        "inventory_days": (
            downside_inventory
        ),
        "payable_days": (
            downside_payables
        ),
    }

    # ========================================================
    # BUILD THE SCENARIOS
    # ========================================================

    scenario_assumptions = (
        build_scenario_assumptions(
            defaults,
            base_inputs,
            upside_adjustments,
            downside_adjustments,
        )
    )

    forecasts = (
        build_all_scenarios(
            analysis_df,
            scenario_assumptions,
            forecast_years,
        )
    )

    # ========================================================
    # ASSUMPTION COMPARISON
    # ========================================================

    st.divider()

    st.subheader(
        "Scenario Assumption Comparison"
    )

    st.dataframe(
        build_assumption_comparison(
            scenario_assumptions
        ),
        hide_index=True,
        width="stretch",
    )

    # ========================================================
    # FINAL YEAR COMPARISON
    # ========================================================

    st.subheader(
        "Final-Year Scenario Comparison"
    )

    summary_df = (
        build_final_year_summary(
            forecasts
        )
    )

    st.dataframe(
        format_final_year_summary(
            summary_df
        ),
        hide_index=True,
        width="stretch",
    )

    base_final = (
        forecasts["Base"]
        .iloc[-1]
    )

    upside_final = (
        forecasts["Upside"]
        .iloc[-1]
    )

    downside_final = (
        forecasts["Downside"]
        .iloc[-1]
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:
        st.metric(
            "Base Net Profit",
            format_currency(
                base_final[
                    "net_profit"
                ]
            ),
        )

    with col2:
        st.metric(
            "Upside Net Profit",
            format_currency(
                upside_final[
                    "net_profit"
                ]
            ),
            format_currency(
                (
                    upside_final[
                        "net_profit"
                    ]
                    - base_final[
                        "net_profit"
                    ]
                )
            ),
        )

    with col3:
        st.metric(
            "Downside Net Profit",
            format_currency(
                downside_final[
                    "net_profit"
                ]
            ),
            format_currency(
                (
                    downside_final[
                        "net_profit"
                    ]
                    - base_final[
                        "net_profit"
                    ]
                )
            ),
        )

    # ========================================================
    # SCENARIO TREND EXPLORER
    # ========================================================

    st.subheader(
        "Scenario Trend Explorer"
    )

    metric_options = {
        "Revenue": (
            "revenue",
            "currency",
        ),

        "Net Profit": (
            "net_profit",
            "currency",
        ),

        "Closing Cash": (
            "cash",
            "currency",
        ),

        "Free Cash Flow": (
            "free_cash_flow",
            "currency",
        ),

        "Operating Margin": (
            "operating_margin",
            "percentage",
        ),

        "Current Ratio": (
            "current_ratio",
            "ratio",
        ),

        "Cash Conversion Cycle": (
            "cash_conversion_cycle",
            "days",
        ),
    }

    selected_metric_label = (
        st.selectbox(
            "Choose scenario metric",
            options=list(
                metric_options.keys()
            ),
            key="scenario_metric",
        )
    )

    (
        selected_metric,
        selected_metric_type,
    ) = metric_options[
        selected_metric_label
    ]

    scenario_chart = (
        build_scenario_chart(
            forecasts,
            selected_metric,
            selected_metric_type,
        )
    )

    st.line_chart(
        scenario_chart
    )

    if (
        selected_metric_type
        == "percentage"
    ):
        st.caption(
            """
            Percentage metrics are displayed as percentage-point
            values on the chart.
            """
        )

    # ========================================================
    # MODEL CHECK
    # ========================================================

    st.subheader(
        "Scenario Model Checks"
    )

    model_check = (
        build_model_check(
            forecasts
        )
    )

    model_check_display = (
        model_check.copy()
    )

    model_check_display[
        "Maximum Difference"
    ] = model_check_display[
        "Maximum Difference"
    ].map(
        lambda value: (
            f"£{value:,.2f}"
        )
    )

    st.dataframe(
        model_check_display,
        hide_index=True,
        width="stretch",
    )

    if (
        model_check[
            "Status"
        ]
        .eq("Passed")
        .all()
    ):
        st.success(
            """
            All three scenario balance-sheet checks passed within
            the £1 tolerance.
            """
        )

    else:
        st.error(
            """
            At least one scenario failed the balance-sheet check.
            Review the model before relying on that scenario.
            """
        )

    # ========================================================
    # SENSITIVITY ANALYSIS
    # ========================================================

    st.divider()

    st.header(
        "Sensitivity Analysis"
    )

    st.write(
        """
        Sensitivity analysis changes revenue growth and gross
        margin simultaneously while holding the other Base
        assumptions constant.
        """
    )

    st.caption(
        """
        Columns show annual revenue-growth assumptions.
        Rows show gross-margin assumptions.
        """
    )

    sensitivity_metric = (
        st.selectbox(
            "Sensitivity output",
            [
                "Net Profit",
                "Closing Cash",
                "Free Cash Flow",
            ],
            key="sensitivity_metric",
        )
    )

    sensitivity_df = (
        build_sensitivity_table(
            analysis_df,
            scenario_assumptions[
                "Base"
            ],
            forecast_years,
            sensitivity_metric,
        )
    )

    st.dataframe(
        format_sensitivity_table(
            sensitivity_df
        ),
        width="stretch",
    )

    numeric_values = (
        sensitivity_df
        .to_numpy()
        .astype(float)
    )

    minimum_result = float(
        np.nanmin(
            numeric_values
        )
    )

    maximum_result = float(
        np.nanmax(
            numeric_values
        )
    )

    base_result = (
        base_final[
            "net_profit"
        ]
        if sensitivity_metric
        == "Net Profit"
        else (
            base_final["cash"]
            if sensitivity_metric
            == "Closing Cash"
            else base_final[
                "free_cash_flow"
            ]
        )
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:
        st.metric(
            "Lowest Sensitivity Result",
            format_currency(
                minimum_result
            ),
        )

    with col2:
        st.metric(
            "Base Scenario Result",
            format_currency(
                base_result
            ),
        )

    with col3:
        st.metric(
            "Highest Sensitivity Result",
            format_currency(
                maximum_result
            ),
        )

    st.info(
        """
        The sensitivity table does not assign probabilities to
        any combination. It shows how the selected output changes
        when two key assumptions move while the remaining Base
        assumptions are held constant.
        """
    )

    # ========================================================
    # SCENARIO COMMENTARY
    # ========================================================

    st.header(
        "Scenario Review"
    )

    for (
        title,
        commentary,
    ) in generate_scenario_commentary(
        analysis_df,
        forecasts,
    ):

        st.write(
            f"**{title}:** {commentary}"
        )

    st.warning(
        """
        Scenario results should be interpreted by reference to
        the assumptions that produced them. A favourable result
        is not evidence that the assumptions are realistic, and
        an adverse result does not establish that financial
        difficulty will occur.
        """
    )