# Colorado Rental ROI Analyzer

This interactive dashboard helps investors evaluate rental property returns across Colorado Springs ZIP codes using publicly available Zillow data. Users can adjust key investment assumptions and compare metrics like cash flow, ROI, rent performance, and long-term equity growth.

## Live Demo

[https://co-real-estate.onrender.com/?role=investor](https://co-real-estate.onrender.com/?role=investor)

## Features

### 1. Basic Cash on Cash

* Compares ZIP codes by first-year cash-on-cash return.
* Visualizes rental cash flow relative to initial cash investment (down payment and closing costs).
* Focuses on monthly cash flow without including tax or equity benefits.

### 2. First-Year ROI

* Incorporates tax savings from depreciation and equity gained from mortgage principal paydown.
* Estimates total return in year one.
* Useful for a more complete financial picture than simple cash-on-cash.

### 3. Total Return

* Evaluates investment performance over a multi-year horizon (default is 5 years).
* Includes property appreciation, cumulative rental cash flow, tax savings, and equity buildup.
* Helps identify ZIPs with the highest long-term wealth potential.

### 4. Data Explorer

* Interactive table with filtering and sorting capabilities.
* View metrics such as home price, rent, ROI, and cash invested.
* Export the filtered dataset to CSV for offline analysis.

### 5. Rent Estimator

* Uses a national log-log regression model to estimate rent based on home price.
* Compares actual rents in Colorado Springs ZIPs to the national trend.
* Provides historical rent charts and ZIP-specific rent details.
* Includes a manual 1% rule calculator for cross-checking rent estimates.

### 6. Deal Analyzer

* Allows users to evaluate a specific property by entering its purchase price and ZIP code.
* Predicts rent and calculates monthly cash flow, annual return, tax benefits, and equity gains.
* Summarizes both short-term and multi-year return projections.

## Sidebar Inputs

Global investment parameters that can be adjusted to update all analysis tabs:

* Interest rate (annual percentage)
* Closing cost percentage
* Annual maintenance cost as a percent of home value
* Annual insurance cost (dollars)
* Vacancy rate (percent)
* Property tax rate (percent)
* Property management fee (percent)
* Monthly capital expenditures (dollars)
* Marginal tax rate (percent)
* Structure value percentage (for depreciation)
* Annual appreciation rate (percent)
* Investment horizon (years)

## Technology Stack

* Python 3.12+
* Streamlit (interactive UI)
* Altair (charts)
* Scikit-learn (regression modeling)
* Pandas / NumPy (data analysis)

## Project Structure

```
├── app/
│   ├── dashboard_app.py         # Streamlit entry point
│   ├── real_estate_logic.py     # Core calculations and data logic
│   ├── data_fetcher.py          # Optional utility to auto-fetch latest data
│   └── data/                    # Zillow CSVs (rent and home values)
├── .streamlit/config.toml       # Streamlit UI overrides (optional)
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container definition for deployment
```

## Planned Enhancements

* Add rent growth over time modeling
* Incorporate sale exit cost estimates
* Support refinancing analysis
* Improve mobile responsiveness
* Enable favicon and metadata via Flask landing page

## License

All content © 2025 HeuerHomes. All rights reserved.
