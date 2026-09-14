import numpy as np
import pandas as pd
import streamlit as st

from forecasting_engine import (
    build_forecast,
    derive_default_assumptions,
    format_currency,
    format_percentage,
    format_ratio,
    safe_divide,
)

from scenario_analysis import (
    build_all_scenarios,
    build_scenario_assumptions,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def direction_word(
    value,
):
    if pd.isna(value):
        return "was unavailable"

    if value > 0:
        return "increased"

    if value < 0:
        return "decreased"

    return "was unchanged"


def movement_percentage(
    current,
    previous,
):
    if (
        pd.isna(current)
        or pd.isna(previous)
        or previous == 0
    ):
        return np.nan

    return (
        current
        - previous
    ) / abs(previous)


# ============================================================
# HISTORICAL COMMENTARY
# ============================================================

def build_historical_commentary(
    analysis_df,
):
    latest = (
        analysis_df.iloc[-1]
    )

    previous = (
        analysis_df.iloc[-2]
    )

    sections = []

    # --------------------------------------------------------
    # REVENUE / PROFIT
    # --------------------------------------------------------

    revenue_growth = (
        movement_percentage(
            latest["revenue"],
            previous["revenue"],
        )
    )

    profit_growth = (
        movement_percentage(
            latest["net_profit"],
            previous["net_profit"],
        )
    )

    revenue_text = (
        f"Revenue {direction_word(revenue_growth)} "
        f"from {format_currency(previous['revenue'])} "
        f"to {format_currency(latest['revenue'])}"
    )

    if not pd.isna(
        revenue_growth
    ):
        revenue_text += (
            f", a movement of "
            f"{abs(revenue_growth):.1%}."
        )

    else:
        revenue_text += "."

    profit_text = (
        f"Net profit {direction_word(profit_growth)} "
        f"from {format_currency(previous['net_profit'])} "
        f"to {format_currency(latest['net_profit'])}"
    )

    if not pd.isna(
        profit_growth
    ):
        profit_text += (
            f", a movement of "
            f"{abs(profit_growth):.1%}."
        )

    else:
        profit_text += "."

    sections.append(
        {
            "Area": "Financial performance",
            "Observed": (
                f"{revenue_text} {profit_text}"
            ),
            "Interpretation": (
                "Revenue and profit movements should be "
                "considered together. Revenue growth does not "
                "by itself establish stronger financial "
                "performance if margins, cash conversion or "
                "working capital deteriorate."
            ),
        }
    )

    # --------------------------------------------------------
    # MARGINS
    # --------------------------------------------------------

    gross_margin_change = (
        latest["gross_margin"]
        - previous["gross_margin"]
    )

    operating_margin_change = (
        latest["operating_margin"]
        - previous["operating_margin"]
    )

    sections.append(
        {
            "Area": "Margins",
            "Observed": (
                f"Gross margin is "
                f"{format_percentage(latest['gross_margin'])}, "
                f"a change of "
                f"{gross_margin_change * 100:+.1f} percentage "
                f"points. Operating margin is "
                f"{format_percentage(latest['operating_margin'])}, "
                f"a change of "
                f"{operating_margin_change * 100:+.1f} "
                f"percentage points."
            ),
            "Interpretation": (
                "Margin movements may reflect changes in "
                "pricing, input costs, product mix or operating "
                "cost control. The financial statements alone "
                "do not establish the underlying cause."
            ),
        }
    )

    # --------------------------------------------------------
    # LIQUIDITY
    # --------------------------------------------------------

    current_ratio_change = (
        latest["current_ratio"]
        - previous["current_ratio"]
    )

    quick_ratio_change = (
        latest["quick_ratio"]
        - previous["quick_ratio"]
    )

    sections.append(
        {
            "Area": "Liquidity",
            "Observed": (
                f"Current ratio is "
                f"{format_ratio(latest['current_ratio'])} "
                f"({current_ratio_change:+.2f}x year on year) "
                f"and quick ratio is "
                f"{format_ratio(latest['quick_ratio'])} "
                f"({quick_ratio_change:+.2f}x year on year)."
            ),
            "Interpretation": (
                "Liquidity ratios indicate the relationship "
                "between short-term assets and liabilities but "
                "do not prove that assets are recoverable or "
                "that liabilities can be settled when due."
            ),
        }
    )

    # --------------------------------------------------------
    # WORKING CAPITAL
    # --------------------------------------------------------

    receivable_change = (
        latest["receivable_days"]
        - previous["receivable_days"]
    )

    inventory_change = (
        latest["inventory_days"]
        - previous["inventory_days"]
    )

    payable_change = (
        latest["payable_days"]
        - previous["payable_days"]
    )

    ccc_change = (
        latest["cash_conversion_cycle"]
        - previous["cash_conversion_cycle"]
    )

    sections.append(
        {
            "Area": "Working capital",
            "Observed": (
                f"Receivable days changed by "
                f"{receivable_change:+.1f} days, inventory days "
                f"by {inventory_change:+.1f} days and payable "
                f"days by {payable_change:+.1f} days. The cash "
                f"conversion cycle changed by "
                f"{ccc_change:+.1f} days."
            ),
            "Interpretation": (
                "Longer receivable or inventory holding periods "
                "can absorb cash. However, further evidence would "
                "be required before concluding that collection, "
                "inventory management or recoverability problems "
                "exist."
            ),
        }
    )

    # --------------------------------------------------------
    # LEVERAGE
    # --------------------------------------------------------

    debt_change = (
        latest["debt_to_equity"]
        - previous["debt_to_equity"]
    )

    coverage_change = (
        latest["interest_coverage"]
        - previous["interest_coverage"]
    )

    sections.append(
        {
            "Area": "Leverage & debt servicing",
            "Observed": (
                f"Debt-to-equity is "
                f"{format_ratio(latest['debt_to_equity'])}, "
                f"a year-on-year movement of "
                f"{debt_change:+.2f}x. Interest coverage is "
                f"{format_ratio(latest['interest_coverage'])}, "
                f"a movement of {coverage_change:+.2f}x."
            ),
            "Interpretation": (
                "Leverage and interest coverage indicate aspects "
                "of financing exposure. Loan terms, maturity "
                "profiles, covenant requirements and available "
                "facilities would be needed for a fuller review."
            ),
        }
    )

    # --------------------------------------------------------
    # CASH FLOW
    # --------------------------------------------------------

    operating_cash_change = (
        latest["operating_cash_flow"]
        - previous["operating_cash_flow"]
    )

    free_cash_change = (
        latest["free_cash_flow"]
        - previous["free_cash_flow"]
    )

    sections.append(
        {
            "Area": "Cash generation",
            "Observed": (
                f"Operating cash flow is "
                f"{format_currency(latest['operating_cash_flow'])}, "
                f"a movement of "
                f"{format_currency(operating_cash_change)}. "
                f"Free cash flow is "
                f"{format_currency(latest['free_cash_flow'])}, "
                f"a movement of "
                f"{format_currency(free_cash_change)}."
            ),
            "Interpretation": (
                "Cash generation should be reviewed alongside "
                "profitability and working-capital movements. "
                "Positive accounting profit does not necessarily "
                "mean the same amount of cash has been generated."
            ),
        }
    )

    return pd.DataFrame(
        sections
    )


# ============================================================
# HISTORICAL REVIEW POINTS
# ============================================================

def build_review_points(
    analysis_df,
    movement_threshold,
    current_ratio_threshold,
    quick_ratio_threshold,
    debt_threshold,
    interest_coverage_threshold,
):
    latest = (
        analysis_df.iloc[-1]
    )

    previous = (
        analysis_df.iloc[-2]
    )

    review_points = []

    def add_point(
        area,
        observation,
        why_it_matters,
        evidence_required,
        priority,
    ):
        review_points.append(
            {
                "Area": area,
                "Observation": observation,
                "Why It Matters": why_it_matters,
                "Evidence / Follow-Up": evidence_required,
                "Priority": priority,
                "Status": "Open",
            }
        )

    # --------------------------------------------------------
    # REVENUE
    # --------------------------------------------------------

    revenue_movement = (
        movement_percentage(
            latest["revenue"],
            previous["revenue"],
        )
    )

    if (
        not pd.isna(
            revenue_movement
        )
        and abs(
            revenue_movement
        ) >= movement_threshold
    ):
        add_point(
            "Revenue",
            (
                f"Revenue changed by "
                f"{revenue_movement:+.1%} year on year."
            ),
            (
                "A significant revenue movement may affect "
                "profitability, working capital and forecast "
                "assumptions."
            ),
            (
                "Review sales analysis by product/customer, "
                "pricing, volumes, major new or lost customers "
                "and period cut-off."
            ),
            "Review",
        )

    # --------------------------------------------------------
    # PROFIT
    # --------------------------------------------------------

    profit_movement = (
        movement_percentage(
            latest["net_profit"],
            previous["net_profit"],
        )
    )

    if (
        not pd.isna(
            profit_movement
        )
        and abs(
            profit_movement
        ) >= movement_threshold
    ):
        add_point(
            "Profitability",
            (
                f"Net profit changed by "
                f"{profit_movement:+.1%} year on year."
            ),
            (
                "A significant movement may reflect revenue, "
                "margin, operating-cost, financing or tax changes."
            ),
            (
                "Review profit bridge, major expense movements, "
                "gross-margin drivers, financing costs and tax "
                "calculations."
            ),
            "Review",
        )

    # --------------------------------------------------------
    # MARGINS
    # --------------------------------------------------------

    margin_change = (
        latest["operating_margin"]
        - previous["operating_margin"]
    )

    if abs(
        margin_change
    ) >= 0.02:
        add_point(
            "Operating margin",
            (
                "Operating margin changed by "
                f"{margin_change * 100:+.1f} percentage points."
            ),
            (
                "Margin movement may indicate changing pricing, "
                "sales mix or operating-cost pressure."
            ),
            (
                "Obtain gross-margin bridge, operating-expense "
                "analysis and management explanation of key "
                "drivers."
            ),
            "Review",
        )

    # --------------------------------------------------------
    # LIQUIDITY
    # --------------------------------------------------------

    if (
        latest["current_ratio"]
        < current_ratio_threshold
    ):
        add_point(
            "Liquidity",
            (
                f"Current ratio of "
                f"{latest['current_ratio']:.2f}x is below the "
                f"selected {current_ratio_threshold:.2f}x "
                "review benchmark."
            ),
            (
                "Short-term liquidity may warrant closer review."
            ),
            (
                "Review aged receivables, inventory quality, "
                "creditor ageing, cash forecasts and available "
                "bank facilities."
            ),
            "Higher priority",
        )

    if (
        latest["quick_ratio"]
        < quick_ratio_threshold
    ):
        add_point(
            "Liquidity",
            (
                f"Quick ratio of "
                f"{latest['quick_ratio']:.2f}x is below the "
                f"selected {quick_ratio_threshold:.2f}x "
                "review benchmark."
            ),
            (
                "Liquidity excluding inventory may be weaker "
                "than the selected benchmark."
            ),
            (
                "Review cash, aged receivables, short-term "
                "liabilities and expected settlement dates."
            ),
            "Review",
        )

    # --------------------------------------------------------
    # RECEIVABLES
    # --------------------------------------------------------

    receivable_change = (
        latest["receivable_days"]
        - previous["receivable_days"]
    )

    if receivable_change > 5:
        add_point(
            "Receivables",
            (
                f"Receivable days increased by "
                f"{receivable_change:.1f} days."
            ),
            (
                "Slower collection can increase the amount of "
                "cash tied up in working capital."
            ),
            (
                "Review aged receivables, overdue customers, "
                "subsequent cash receipts, disputes and bad-debt "
                "provisioning."
            ),
            "Review",
        )

    # --------------------------------------------------------
    # INVENTORY
    # --------------------------------------------------------

    inventory_change = (
        latest["inventory_days"]
        - previous["inventory_days"]
    )

    if inventory_change > 5:
        add_point(
            "Inventory",
            (
                f"Inventory days increased by "
                f"{inventory_change:.1f} days."
            ),
            (
                "Longer holding periods may increase working "
                "capital requirements and potential obsolescence "
                "exposure."
            ),
            (
                "Review inventory ageing, slow-moving items, "
                "subsequent sales, write-downs and purchasing "
                "patterns."
            ),
            "Review",
        )

    # --------------------------------------------------------
    # LEVERAGE
    # --------------------------------------------------------

    if (
        latest["debt_to_equity"]
        > debt_threshold
    ):
        add_point(
            "Leverage",
            (
                f"Debt-to-equity of "
                f"{latest['debt_to_equity']:.2f}x exceeds the "
                f"selected {debt_threshold:.2f}x review "
                "benchmark."
            ),
            (
                "Higher leverage may increase financing and "
                "refinancing exposure."
            ),
            (
                "Review loan agreements, maturity profile, "
                "covenants, repayment schedule and available "
                "facilities."
            ),
            "Review",
        )

    # --------------------------------------------------------
    # INTEREST COVERAGE
    # --------------------------------------------------------

    if (
        latest["interest_coverage"]
        < interest_coverage_threshold
    ):
        add_point(
            "Debt servicing",
            (
                f"Interest coverage of "
                f"{latest['interest_coverage']:.2f}x is below "
                f"the selected "
                f"{interest_coverage_threshold:.2f}x review "
                "benchmark."
            ),
            (
                "Lower interest coverage means operating profit "
                "provides less headroom over finance costs."
            ),
            (
                "Review interest calculations, borrowing terms, "
                "debt maturity profile and forecast debt-service "
                "capacity."
            ),
            "Higher priority",
        )

    # --------------------------------------------------------
    # CASH FLOW
    # --------------------------------------------------------

    if (
        latest["free_cash_flow"]
        < 0
    ):
        add_point(
            "Cash flow",
            (
                f"Free cash flow is negative at "
                f"{format_currency(latest['free_cash_flow'])}."
            ),
            (
                "Operating cash generation is insufficient to "
                "cover capital expenditure in the calculated "
                "period."
            ),
            (
                "Review cash-flow forecast, capital-expenditure "
                "commitments, available facilities and funding "
                "plans."
            ),
            "Higher priority",
        )

    if (
        latest["net_profit"] > 0
        and latest["operating_cash_flow"]
        < latest["net_profit"]
    ):
        add_point(
            "Profit to cash conversion",
            (
                "Operating cash flow is below reported net "
                "profit."
            ),
            (
                "This may indicate cash absorption through "
                "working capital or other non-cash accounting "
                "movements."
            ),
            (
                "Review cash-flow bridge, receivables, inventory, "
                "payables and non-cash adjustments."
            ),
            "Monitor",
        )

    if not review_points:
        add_point(
            "General review",
            (
                "No specific review point was generated using "
                "the selected thresholds."
            ),
            (
                "The absence of a rule-based flag does not mean "
                "that the financial statements are free from "
                "error or risk."
            ),
            (
                "Continue normal analytical review and obtain "
                "support for significant balances and movements."
            ),
            "Monitor",
        )

    return pd.DataFrame(
        review_points
    )


# ============================================================
# FORECAST ASSUMPTION CHALLENGE
# ============================================================

def build_forecast_challenges(
    analysis_df,
):
    defaults = (
        derive_default_assumptions(
            analysis_df
        )
    )

    latest = (
        analysis_df.iloc[-1]
    )

    first = (
        analysis_df.iloc[0]
    )

    periods = (
        int(
            latest["year"]
        )
        - int(
            first["year"]
        )
    )

    if (
        periods > 0
        and first["revenue"] > 0
        and latest["revenue"] > 0
    ):
        revenue_cagr = (
            (
                latest["revenue"]
                / first["revenue"]
            )
            ** (
                1
                / periods
            )
            - 1
        )

    else:
        revenue_cagr = np.nan

    challenges = [
        {
            "Assumption": (
                "Revenue growth"
            ),
            "Model Basis": (
                format_percentage(
                    defaults[
                        "revenue_growth"
                    ]
                )
            ),
            "Challenge": (
                "What supports the assumption that historical "
                "revenue growth can continue?"
            ),
            "Evidence to Consider": (
                "Sales pipeline, customer contracts, order book, "
                "pricing, market demand and capacity constraints."
            ),
        },

        {
            "Assumption": (
                "Gross margin"
            ),
            "Model Basis": (
                format_percentage(
                    defaults[
                        "gross_margin"
                    ]
                )
            ),
            "Challenge": (
                "Is the latest gross margin sustainable throughout "
                "the forecast period?"
            ),
            "Evidence to Consider": (
                "Supplier pricing, customer pricing, product mix, "
                "inflation assumptions and purchasing contracts."
            ),
        },

        {
            "Assumption": (
                "Operating expenses"
            ),
            "Model Basis": (
                format_percentage(
                    defaults[
                        "operating_expense_ratio"
                    ]
                )
            ),
            "Challenge": (
                "Can operating costs remain at the modelled "
                "percentage of revenue as the business grows?"
            ),
            "Evidence to Consider": (
                "Payroll plan, rent, technology costs, inflation, "
                "headcount and committed expenditure."
            ),
        },

        {
            "Assumption": (
                "Receivable days"
            ),
            "Model Basis": (
                f"{defaults['receivable_days']:.1f} days"
            ),
            "Challenge": (
                "Is the latest collection period a reasonable "
                "forecast assumption?"
            ),
            "Evidence to Consider": (
                "Customer payment terms, ageing report, subsequent "
                "receipts, customer concentration and overdue debt."
            ),
        },

        {
            "Assumption": (
                "Inventory days"
            ),
            "Model Basis": (
                f"{defaults['inventory_days']:.1f} days"
            ),
            "Challenge": (
                "Does forecast inventory holding reflect expected "
                "sales volumes and purchasing requirements?"
            ),
            "Evidence to Consider": (
                "Demand forecast, lead times, stock ageing, planned "
                "purchases and obsolete-stock review."
            ),
        },

        {
            "Assumption": (
                "Capital expenditure"
            ),
            "Model Basis": (
                format_percentage(
                    defaults[
                        "capex_ratio"
                    ]
                )
            ),
            "Challenge": (
                "Is capital expenditure sufficient to support the "
                "forecast level of operations?"
            ),
            "Evidence to Consider": (
                "Approved capex plan, asset replacement programme, "
                "supplier quotations and capacity requirements."
            ),
        },
    ]

    if not pd.isna(
        revenue_cagr
    ):
        challenges[0][
            "Historical Context"
        ] = (
            f"Historical revenue CAGR: "
            f"{revenue_cagr:.1%}"
        )

    for item in challenges:
        if (
            "Historical Context"
            not in item
        ):
            item[
                "Historical Context"
            ] = ""

    return pd.DataFrame(
        challenges
    )


# ============================================================
# DEFAULT SCENARIO RISK REVIEW
# ============================================================

def build_default_scenario_review(
    analysis_df,
):
    defaults = (
        derive_default_assumptions(
            analysis_df
        )
    )

    base_inputs = {
        "revenue_growth": (
            defaults[
                "revenue_growth"
            ]
        ),
        "gross_margin": (
            defaults[
                "gross_margin"
            ]
        ),
        "operating_expense_ratio": (
            defaults[
                "operating_expense_ratio"
            ]
        ),
        "receivable_days": (
            defaults[
                "receivable_days"
            ]
        ),
        "inventory_days": (
            defaults[
                "inventory_days"
            ]
        ),
        "payable_days": (
            defaults[
                "payable_days"
            ]
        ),
    }

    upside_adjustments = {
        "revenue_growth": 0.03,
        "gross_margin": 0.015,
        "operating_expense_ratio": -0.01,
        "receivable_days": -5,
        "inventory_days": -5,
        "payable_days": 0,
    }

    downside_adjustments = {
        "revenue_growth": -0.05,
        "gross_margin": -0.025,
        "operating_expense_ratio": 0.015,
        "receivable_days": 10,
        "inventory_days": 10,
        "payable_days": 0,
    }

    assumptions = (
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
            assumptions,
            3,
        )
    )

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

    rows = [
        {
            "Review Area": (
                "Profit sensitivity"
            ),
            "Observation": (
                "Final-year net profit ranges from "
                f"{format_currency(downside['net_profit'])} "
                "in the illustrative Downside case to "
                f"{format_currency(upside['net_profit'])} "
                "in the illustrative Upside case."
            ),
            "Review Implication": (
                "Forecast profitability is sensitive to revenue, "
                "margin and operating-cost assumptions."
            ),
        },

        {
            "Review Area": (
                "Cash sensitivity"
            ),
            "Observation": (
                "Illustrative final-year closing cash is "
                f"{format_currency(base['cash'])} in Base and "
                f"{format_currency(downside['cash'])} in Downside."
            ),
            "Review Implication": (
                "Cash headroom should be considered separately "
                "from accounting profit."
            ),
        },

        {
            "Review Area": (
                "Working capital sensitivity"
            ),
            "Observation": (
                "The illustrative Downside cash conversion cycle "
                f"is {downside['cash_conversion_cycle']:.1f} days "
                f"compared with "
                f"{base['cash_conversion_cycle']:.1f} days in Base."
            ),
            "Review Implication": (
                "Changes in collection and inventory assumptions "
                "can materially affect forecast liquidity."
            ),
        },
    ]

    if (
        downside["cash"]
        < 0
    ):
        rows.append(
            {
                "Review Area": (
                    "Potential funding requirement"
                ),
                "Observation": (
                    "Illustrative Downside closing cash becomes "
                    "negative."
                ),
                "Review Implication": (
                    "Further funding, cost reduction, working-"
                    "capital action or revised assumptions may "
                    "need to be considered."
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MANAGEMENT QUESTIONS
# ============================================================

def build_management_questions(
    analysis_df,
):
    latest = (
        analysis_df.iloc[-1]
    )

    previous = (
        analysis_df.iloc[-2]
    )

    questions = []

    questions.append(
        {
            "Area": "Revenue",
            "Question": (
                "What were the main drivers of the latest-year "
                "revenue movement, and are those drivers expected "
                "to continue?"
            ),
        }
    )

    questions.append(
        {
            "Area": "Margins",
            "Question": (
                "What explains the movement in gross and operating "
                "margins, including changes in pricing, input costs "
                "and sales mix?"
            ),
        }
    )

    if (
        latest["receivable_days"]
        > previous["receivable_days"]
    ):
        questions.append(
            {
                "Area": "Receivables",
                "Question": (
                    "Why are customers taking longer to pay, and "
                    "are there specific overdue or disputed "
                    "balances driving the increase?"
                ),
            }
        )

    if (
        latest["inventory_days"]
        > previous["inventory_days"]
    ):
        questions.append(
            {
                "Area": "Inventory",
                "Question": (
                    "Why is inventory being held for longer, and "
                    "is there any slow-moving, damaged or obsolete "
                    "stock requiring review?"
                ),
            }
        )

    questions.append(
        {
            "Area": "Cash flow",
            "Question": (
                "How does management explain the relationship "
                "between reported profit and operating cash flow?"
            ),
        }
    )

    questions.append(
        {
            "Area": "Borrowing",
            "Question": (
                "What are the key repayment dates, interest terms "
                "and covenant requirements attached to existing "
                "borrowing?"
            ),
        }
    )

    questions.append(
        {
            "Area": "Forecast",
            "Question": (
                "What evidence supports the forecast revenue, "
                "margin and working-capital assumptions?"
            ),
        }
    )

    questions.append(
        {
            "Area": "Downside planning",
            "Question": (
                "What actions could management take if growth, "
                "margin or customer collections underperform the "
                "Base forecast?"
            ),
        }
    )

    return pd.DataFrame(
        questions
    )


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

def build_executive_summary(
    analysis_df,
):
    latest = (
        analysis_df.iloc[-1]
    )

    previous = (
        analysis_df.iloc[-2]
    )

    revenue_change = (
        movement_percentage(
            latest["revenue"],
            previous["revenue"],
        )
    )

    profit_change = (
        movement_percentage(
            latest["net_profit"],
            previous["net_profit"],
        )
    )

    ccc_change = (
        latest["cash_conversion_cycle"]
        - previous["cash_conversion_cycle"]
    )

    paragraphs = []

    paragraphs.append(
        (
            f"In {int(latest['year'])}, revenue was "
            f"{format_currency(latest['revenue'])} and net profit "
            f"was {format_currency(latest['net_profit'])}. "
            f"Revenue moved by {revenue_change:+.1%} and net "
            f"profit by {profit_change:+.1%} compared with "
            f"{int(previous['year'])}."
        )
    )

    paragraphs.append(
        (
            f"Operating margin was "
            f"{format_percentage(latest['operating_margin'])}, "
            f"current ratio was "
            f"{format_ratio(latest['current_ratio'])}, and "
            f"interest coverage was "
            f"{format_ratio(latest['interest_coverage'])}."
        )
    )

    paragraphs.append(
        (
            f"The cash conversion cycle changed by "
            f"{ccc_change:+.1f} days year on year, while free cash "
            f"flow was {format_currency(latest['free_cash_flow'])}. "
            "These indicators should be interpreted together rather "
            "than treated as standalone conclusions."
        )
    )

    return paragraphs


# ============================================================
# STREAMLIT TAB
# ============================================================

def render_management_review_tab(
    analysis_df,
):
    st.header(
        "Management Commentary & Review Intelligence"
    )

    st.write(
        """
        Convert financial analysis into structured review points,
        management questions and evidence requests. The output is
        designed to support analytical review rather than make
        unsupported conclusions.
        """
    )

    st.info(
        """
        The commentary below is rule-based and derived from the
        financial model. It does not prove the cause of a movement
        or replace source evidence, management explanations or
        professional judgement.
        """
    )

    # ========================================================
    # REVIEW SETTINGS
    # ========================================================

    with st.expander(
        "Review settings",
        expanded=False,
    ):
        col1, col2 = (
            st.columns(2)
        )

        with col1:

            movement_threshold = (
                st.number_input(
                    "Significant movement threshold (%)",
                    min_value=1.0,
                    max_value=100.0,
                    value=10.0,
                    step=1.0,
                    key=(
                        "management_movement_threshold"
                    ),
                )
                / 100
            )

            current_ratio_threshold = (
                st.number_input(
                    "Current-ratio review benchmark",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.20,
                    step=0.10,
                    key=(
                        "management_current_ratio"
                    ),
                )
            )

            quick_ratio_threshold = (
                st.number_input(
                    "Quick-ratio review benchmark",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.00,
                    step=0.10,
                    key=(
                        "management_quick_ratio"
                    ),
                )
            )

        with col2:

            debt_threshold = (
                st.number_input(
                    "Debt-to-equity review benchmark",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.50,
                    step=0.10,
                    key=(
                        "management_debt_ratio"
                    ),
                )
            )

            interest_threshold = (
                st.number_input(
                    "Interest-coverage review benchmark",
                    min_value=0.0,
                    max_value=50.0,
                    value=3.00,
                    step=0.50,
                    key=(
                        "management_interest_cover"
                    ),
                )
            )

        st.caption(
            """
            These are analytical review settings, not audit
            materiality or universal industry benchmarks.
            """
        )

    # ========================================================
    # EXECUTIVE SUMMARY
    # ========================================================

    st.subheader(
        "Executive Financial Review"
    )

    for paragraph in (
        build_executive_summary(
            analysis_df
        )
    ):
        st.write(
            paragraph
        )

    # ========================================================
    # STRUCTURED COMMENTARY
    # ========================================================

    st.subheader(
        "Structured Management Commentary"
    )

    commentary_df = (
        build_historical_commentary(
            analysis_df
        )
    )

    for _, row in (
        commentary_df.iterrows()
    ):

        with st.expander(
            row["Area"],
            expanded=True,
        ):

            st.write(
                f"**Observed:** "
                f"{row['Observed']}"
            )

            st.write(
                f"**Interpretation:** "
                f"{row['Interpretation']}"
            )

    # ========================================================
    # REVIEW REGISTER
    # ========================================================

    st.divider()

    st.header(
        "Financial Review Register"
    )

    st.write(
        """
        Each item separates the observed financial issue from the
        evidence or follow-up that would be required before a
        conclusion is reached.
        """
    )

    review_df = (
        build_review_points(
            analysis_df,
            movement_threshold,
            current_ratio_threshold,
            quick_ratio_threshold,
            debt_threshold,
            interest_threshold,
        )
    )

    edited_review_df = (
        st.data_editor(
            review_df,
            hide_index=True,
            width="stretch",
            column_config={
                "Status": (
                    st.column_config.SelectboxColumn(
                        "Status",
                        options=[
                            "Open",
                            "In Review",
                            "Resolved",
                        ],
                    )
                )
            },
            key="financial_review_register",
        )
    )

    open_items = (
        edited_review_df[
            edited_review_df[
                "Status"
            ]
            != "Resolved"
        ]
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:
        st.metric(
            "Total Review Points",
            len(
                edited_review_df
            ),
        )

    with col2:
        st.metric(
            "Open / In Review",
            len(
                open_items
            ),
        )

    with col3:
        higher_priority = (
            edited_review_df[
                "Priority"
            ]
            .eq(
                "Higher priority"
            )
            .sum()
        )

        st.metric(
            "Higher-Priority Points",
            int(
                higher_priority
            ),
        )

    st.caption(
        """
        Changing a status in this portfolio interface records a
        review state only. It does not mean evidence has actually
        been obtained or that an issue has been professionally
        cleared.
        """
    )

    # ========================================================
    # ASSUMPTION CHALLENGE
    # ========================================================

    st.divider()

    st.header(
        "Forecast Assumption Challenge"
    )

    st.write(
        """
        A forecast should not be accepted simply because the
        model calculates correctly. The assumptions themselves
        should be challenged and supported.
        """
    )

    challenge_df = (
        build_forecast_challenges(
            analysis_df
        )
    )

    st.dataframe(
        challenge_df,
        hide_index=True,
        width="stretch",
    )

    # ========================================================
    # SCENARIO RISK REVIEW
    # ========================================================

    st.subheader(
        "Illustrative Scenario Risk Review"
    )

    st.caption(
        """
        This section uses the same default Base, Upside and
        Downside structure used in the Scenario Analysis module.
        It is intended to identify areas for review rather than
        assign probabilities.
        """
    )

    scenario_review_df = (
        build_default_scenario_review(
            analysis_df
        )
    )

    st.dataframe(
        scenario_review_df,
        hide_index=True,
        width="stretch",
    )

    # ========================================================
    # MANAGEMENT QUESTIONS
    # ========================================================

    st.divider()

    st.header(
        "Questions for Management"
    )

    st.write(
        """
        These questions demonstrate the type of follow-up that
        would be needed to move from analytical observation to
        evidence-based understanding.
        """
    )

    questions_df = (
        build_management_questions(
            analysis_df
        )
    )

    st.dataframe(
        questions_df,
        hide_index=True,
        width="stretch",
    )

    # ========================================================
    # REVIEW PRINCIPLES
    # ========================================================

    st.subheader(
        "Review Principles Applied"
    )

    st.markdown(
        """
        - **Movement ≠ error:** a significant financial movement
          is a review point, not proof of misstatement.

        - **Ratio ≠ conclusion:** a ratio outside a benchmark
          requires context before it can be interpreted.

        - **Forecast ≠ fact:** projected figures are outputs of
          assumptions, not evidence of future performance.

        - **Explanation ≠ evidence:** a management explanation
          should be supported where appropriate.

        - **Automation ≠ professional judgement:** automated
          review can prioritise attention, but conclusions still
          require judgement and supporting evidence.
        """
    )

    st.warning(
        """
        This portfolio module demonstrates analytical review
        discipline. It is not a statutory audit procedure,
        assurance conclusion, credit assessment or investment
        recommendation.
        """
    )