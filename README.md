# Financial Statement Analysis & Forecasting

An interactive finance portfolio project built with Python and Streamlit to analyse historical financial statements, assess financial performance and create assumption-driven forecasts.

The project combines financial statement analysis, KPI monitoring, forecasting, scenario modelling, sensitivity analysis and structured financial review.

## Project Purpose

The application demonstrates how historical financial information can be transformed into structured analysis while maintaining a clear distinction between:

- historical financial data;
- calculated financial metrics;
- analytical review flags;
- forecasting assumptions;
- projected outcomes; and
- areas requiring professional judgement or supporting evidence.

The project uses synthetic financial data for portfolio demonstration purposes.

## Key Features

### Historical Financial Analysis

- Income statement analysis
- Balance sheet analysis
- Cash-flow analysis
- Year-on-year horizontal analysis
- Common-size / vertical analysis
- CAGR calculations
- Significant movement flags

### Financial Ratios & KPIs

- Gross margin
- Operating margin
- Net margin
- Return on assets
- Return on equity
- Current ratio
- Quick ratio
- Debt-to-equity
- Interest coverage
- Receivable days
- Inventory days
- Payable days
- Cash conversion cycle

### Financial Health Dashboard

- Multi-year KPI monitoring
- Configurable analytical benchmarks
- Financial-health review flags
- Transparent scoring methodology
- DuPont return-on-equity analysis

### Forecasting

- 1–5 year forecast horizon
- Editable revenue-growth assumptions
- Margin assumptions
- Operating-cost assumptions
- Working-capital assumptions
- Capital expenditure
- Depreciation / PPE roll-forward
- Debt assumptions
- Dividend assumptions
- Forecast income statement
- Forecast balance sheet
- Forecast cash flow
- Forecast ratios
- Balance-sheet validation

### Scenario Analysis

- Base scenario
- Upside scenario
- Downside scenario
- Scenario trend comparison
- Profit and cash sensitivity analysis
- Working-capital stress testing
- Scenario-specific accounting checks

### Management Review

- Structured financial commentary
- Significant review points
- Evidence and follow-up requests
- Forecast assumption challenge
- Management questions
- Editable review register
- Explicit separation between observations and conclusions

### Excel Reporting

The application generates a professional Excel analysis pack containing:

- Executive Summary
- Historical Income Statement
- Historical Balance Sheet
- Historical Cash Flow
- Horizontal Analysis
- Common-Size Analysis
- Ratios & KPIs
- Forecast Assumptions
- Forecast Income Statement
- Forecast Balance Sheet
- Forecast Cash Flow
- Forecast Ratios
- Scenario Comparison
- Sensitivity Analysis
- Financial Review Register
- Forecast Assumption Challenge
- Management Questions
- Model Checks

## Accounting & Review Approach

The project deliberately avoids treating analytical indicators as automatic conclusions.

For example:

- a significant movement is not automatically an accounting error;
- a ratio outside a benchmark is not automatically evidence of financial difficulty;
- a forecast is an assumption-driven estimate rather than a statement of future fact;
- a management explanation is not the same as supporting evidence; and
- automated analysis does not replace professional judgement.

## Technology

- Python
- Streamlit
- pandas
- NumPy
- XlsxWriter
- openpyxl

## Data

The included `financial_statements.csv` dataset is synthetic and was created solely for portfolio demonstration.

No confidential company, client or personal financial information is used.

## Limitations

This project is a simplified analytical model. It is not intended to represent a complete corporate forecasting system, statutory financial statement preparation process or audit methodology.

Forecast outputs depend on the assumptions entered into the model. The application does not independently validate whether those assumptions are achievable.

The cash-flow model is a simplified forecasting bridge rather than a statutory cash-flow statement.

## Disclaimer

Portfolio simulation only.

This application does not provide audit assurance, investment advice, credit advice or a guarantee of future financial performance.