import os
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from sklearn.linear_model import LinearRegression


from data_fetcher import fetch_if_missing


from real_estate_logic import (
    load_data,
    filter_and_label_zips,
    prepare_merged_data,
    get_national_training_data,
    calculate_financial_metrics
)




# --- CONFIG ---
alt.data_transformers.enable("json")
st.set_page_config(page_title="Rental Dashboard", layout="wide")
st.title("📊 Real Estate Rental Dashboard - CoC Analyzer")

# --- CONSTANTS ---
LOAN_TO_VALUE = 0.8
DEPRECIATION_YEARS = 27.5

defaults = {
    "interest_rate": 7.0,
    "closing_cost_pct": 2.0,
    "maintenance_rate": 0.015,
    "insurance_annual": 1300,
    "vacancy_rate": 0.05,
    "marginal_tax_rate": 24.0,
    "structure_pct": 0.85,
    "property_mgmt_pct": 0.08,
    "capex_monthly": 300,
    "annual_appreciation_pct": 3.0,
    "appreciation_years": 5,
    "property_tax_rate": 0.41
}

# --- RESET FUNCTION ---
def reset_to_defaults():
    for key, value in defaults.items():
        st.session_state[key] = value

for key, val in defaults.items():
    st.session_state.setdefault(key, val)

# --- SIDEBAR SETTINGS ---
def create_sidebar():
    st.sidebar.header("Investment Parameters")
    if st.sidebar.button("🔄 Reset to Defaults"):
        reset_to_defaults()

    st.sidebar.markdown("#### Loan Settings")
    st.sidebar.slider("Interest Rate (%)", 2.0, 12.0, value=st.session_state.get("interest_rate", defaults["interest_rate"]), step=0.1, key="interest_rate")
    st.sidebar.slider("Closing Costs (%)", 0.0, 5.0, value=st.session_state.get("closing_cost_pct", defaults["closing_cost_pct"]), step=0.1, key="closing_cost_pct")

    st.sidebar.markdown("#### Property Expenses")
    maintenance_pct = st.sidebar.slider("Annual Maintenance (% of property value)", 0.0, 5.0, value=st.session_state.get("maintenance_rate", defaults["maintenance_rate"]) * 100, step=0.1, key="maintenance_rate_pct")
    st.session_state["maintenance_rate"] = maintenance_pct / 100

    st.sidebar.slider("Annual Insurance ($)", 500, 3000, value=st.session_state.get("insurance_annual", defaults["insurance_annual"]), step=50, key="insurance_annual")

    vacancy_pct = st.sidebar.slider("Vacancy Rate (%)", 0.0, 15.0, value=st.session_state.get("vacancy_rate", defaults["vacancy_rate"]) * 100, step=0.5, key="vacancy_rate_pct")
    st.session_state["vacancy_rate"] = vacancy_pct / 100

    st.sidebar.slider("Property Tax Rate (%)", 0.1, 3.0, value=st.session_state.get("property_tax_rate", defaults["property_tax_rate"]), step=0.01, key="property_tax_rate")

    mgmt_pct = st.sidebar.slider("Property Management Fee (%)", 0.0, 15.0, value=st.session_state.get("property_mgmt_pct", defaults["property_mgmt_pct"]) * 100, step=0.5, key="property_mgmt_pct_pct")
    st.session_state["property_mgmt_pct"] = mgmt_pct / 100

    st.sidebar.slider("Monthly Capital Expenditures ($)", 0, 1000, value=st.session_state.get("capex_monthly", defaults["capex_monthly"]), step=25, key="capex_monthly")

    st.sidebar.markdown("#### Tax & Appreciation")
    st.sidebar.slider("Marginal Tax Rate (%)", 0.0, 50.0, value=st.session_state.get("marginal_tax_rate", defaults["marginal_tax_rate"]), step=1.0, key="marginal_tax_rate")

    structure_pct = st.sidebar.slider("Structure Value (% of property value)", 50.0, 100.0, value=st.session_state.get("structure_pct", defaults["structure_pct"]) * 100, step=1.0, key="structure_pct_pct")
    st.session_state["structure_pct"] = structure_pct / 100

    st.sidebar.slider("Annual Appreciation Rate (%)", 0.0, 10.0, value=st.session_state.get("annual_appreciation_pct", defaults["annual_appreciation_pct"]), step=0.1, key="annual_appreciation_pct")
    st.sidebar.slider("Investment Horizon (years)", 1, 30, value=st.session_state.get("appreciation_years", defaults["appreciation_years"]), step=1, key="appreciation_years")

# --- CHART UTILITY ---
def create_bar_chart(data, x, y, title, y_axis_title):
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X(y, title=y_axis_title),
        y=alt.Y(x, title=None, sort='-x'),
        color=alt.condition(alt.datum[y] > 0, alt.value('darkgreen'), alt.value('darkred')),
        tooltip=[x, y]
    ).properties(
        title=title,
        height=400
    ).interactive()
    return chart

# --- MAIN APP ENTRY ---
def main():
    fetch_if_missing()
    create_sidebar()

    # 📁 Load available files
    all_files = os.listdir("data")
    home_files = sorted([f for f in all_files if "home" in f and f.endswith(".csv")], reverse=True)
    rent_files = sorted([f for f in all_files if "rent" in f and f.endswith(".csv")], reverse=True)
    st.write("🔍 Contents of data/:", all_files)
    st.write("📂 Filtered home_files:", home_files)
    st.write("📂 Filtered rent_files:", rent_files)


    # 🧩 Dropdowns — NOT inside a cached function!
    selected_home = st.sidebar.selectbox("Select Home Value File", home_files, index=0)
    selected_rent = st.sidebar.selectbox("Select Rent Index File", rent_files, index=0)

    st.caption(f"Using home values from: `{selected_home}`")
    st.caption(f"Using rent data from: `{selected_rent}`")

    # 🧠 Cached read from disk
    home_df, rent_df = load_data(selected_home, selected_rent)

    if home_df is None or rent_df is None:
        st.error("Could not load Zillow data.")
        return

    valid_data, latest_month = prepare_merged_data(home_df, rent_df)
    if valid_data is None:
        st.error("Could not merge Zillow data.")
        return

    national_df = get_national_training_data(home_df, rent_df, latest_month)
    params = {k: st.session_state[k] for k in defaults}
    results = calculate_financial_metrics(valid_data, params)

    tab_labels = [
        "Basic Cash on Cash",
        "First-Year ROI",
        "Total Equity Gain",
        "Data Explorer",
        "Rent Estimator"
    ]

    selected = st.radio(
        "View:",
        tab_labels,
        index=tab_labels.index(st.session_state.get("selected_tab", tab_labels[0])),
        horizontal=True,
        key="selected_tab"
    )

    if selected == "Basic Cash on Cash":
        top = results.sort_values("Basic_CoC", ascending=False).head(10)
        st.markdown("**ℹ️ Basic CoC** reflects first-year cash flow only.")
        st.altair_chart(
            create_bar_chart(
                top,
                "Zip_Label",
                "Basic_CoC",
                f"Top ZIPs – Basic CoC",
                "Basic CoC Return (%)"
            ),
            use_container_width=True
        )

    elif selected == "First-Year ROI":
        top = results.sort_values("Advanced_CoC", ascending=False).head(10)
        st.markdown("**ℹ️ First-Year ROI** includes depreciation tax benefits and equity pay-down.")
        st.altair_chart(
            create_bar_chart(
                top,
                "Zip_Label",
                "Advanced_CoC",
                f"Top ZIPs – First-Year ROI",
                "First-Year ROI (%)"
            ),
            use_container_width=True
        )

    elif selected == "Total Equity Gain":
        top = results.sort_values("Total_Equity_Gain", ascending=False).head(10)
        yrs = st.session_state.appreciation_years
        st.markdown(f"**ℹ️ Total Equity Gain** over {yrs} years, including appreciation and paydown.")
        st.altair_chart(
            create_bar_chart(
                top,
                "Zip_Label",
                "Total_Equity_Gain",
                f"Top ZIPs – Total Equity Gain Over {yrs} Years",
                "Total Equity Gain ($)"
            ),
            use_container_width=True
        )

    elif selected == "Data Explorer":
        st.subheader("Data Explorer")
        col1, col2, col3 = st.columns(3)
        with col1:
            min_price = int(results["Home_Price"].min())
            max_price = int(results["Home_Price"].max())
            price_range = st.slider("Home Price Range", min_price, max_price, (min_price, max_price))
        with col2:
            sort_by = st.selectbox(
                "Sort By",
                ["Basic_CoC", "Advanced_CoC", "Total_ROC", "Total_Equity_Gain", "Home_Price", "Rent"]
            )
        with col3:
            ascending = st.checkbox("Ascending Order", False)

        filtered = results[
            (results["Home_Price"] >= price_range[0]) &
            (results["Home_Price"] <= price_range[1])
        ].sort_values(sort_by, ascending=ascending)

        display_df = filtered.rename(columns={"Advanced_CoC": "First-Year ROI"})
        st.dataframe(display_df[[
            "Zip_Label", "Home_Price", "Rent",
            "Basic_CoC", "First-Year ROI",
            "Total_ROC", "Total_Equity_Gain", "Cash_In"
        ]])

        csv = display_df.to_csv(index=False)
        st.download_button(
            "Download Data as CSV",
            csv,
            "rental_analysis.csv",
            "text/csv",
            key="download-csv"
        )

    elif selected == "Rent Estimator":

        if national_df is not None and len(national_df) >= 100:
            X_nat = np.log(national_df[["Home_Price"]].values)
            y_nat = np.log(national_df["Rent"].values)

            national_model = LinearRegression()
            national_model.fit(X_nat, y_nat)

            slope_nat = national_model.coef_[0]
            intercept_nat = national_model.intercept_
            r2_nat = national_model.score(X_nat, y_nat)

            st.markdown("### National Rent Model (Log-Log) - seen below")
            st.caption(f"""
            **Model Equation:**  log(Rent) = {intercept_nat:.2f} + {slope_nat:.3f} × log(Home Price)  
            **R² = {r2_nat:.3f}** based on {len(national_df):,} ZIP codes nationwide.

            This model estimates rent based on the national trend across all U.S. ZIP codes.  
            It's useful as a benchmark to compare against Colorado Springs ZIPs.
            """)
        else:
            st.info("National trend model could not be created — missing or invalid data.")

        st.markdown("""---""")
        st.markdown(
            "<h4 style='text-align: center; margin-top: 2rem; margin-bottom: 2rem;'>📍 Colorado Springs Area ZIPs vs National Rent Model</h4>",
            unsafe_allow_html=True
        )

        citywide_data = results[["Zip_Label", "Home_Price", "Rent"]].dropna()

        if len(citywide_data) >= 10:
            local = citywide_data[(citywide_data["Home_Price"] > 0) & (citywide_data["Rent"] > 0)].copy()
            st.write("🧪 Local ZIPs available for comparison:", len(local))

            local["Predicted_Rent_National"] = np.exp(
                intercept_nat + slope_nat * np.log(local["Home_Price"])
            )
            local["Rent_Difference"] = local["Rent"] - local["Predicted_Rent_National"]

            show_one_percent = st.checkbox("Show 1% Rule Line", value=False)

            scatter = alt.Chart(local).mark_circle(size=80, opacity=0.7).encode(
                x=alt.X("Home_Price:Q", title="Median Home Price ($)", axis=alt.Axis(format="$,.0f")),
                y=alt.Y("Rent:Q", title="Actual Median Rent ($)", axis=alt.Axis(format="$,.0f")),
                color=alt.condition(
                    alt.datum.Rent_Difference > 0,
                    alt.value("seagreen"),
                    alt.value("firebrick")
                ),
                tooltip=["Zip_Label", "Home_Price", "Rent", "Predicted_Rent_National", "Rent_Difference"]
            )

            x_vals = np.linspace(local["Home_Price"].min(), local["Home_Price"].max(), 100)
            line_df = pd.DataFrame({
                "Home_Price": x_vals,
                "Predicted_Rent": np.exp(intercept_nat + slope_nat * np.log(x_vals))
            })
            line = alt.Chart(line_df).mark_line(color="orange", strokeWidth=3).encode(
                x="Home_Price:Q", y="Predicted_Rent:Q"
            )

            if show_one_percent:
                one_percent_df = pd.DataFrame({
                    "Home_Price": x_vals,
                    "OnePercentRent": 0.01 * x_vals
                })
                line_1pct = alt.Chart(one_percent_df).mark_line(
                    color="gray", strokeDash=[4, 4]
                ).encode(
                    x="Home_Price:Q",
                    y="OnePercentRent:Q"
                )
                final_chart = scatter + line + line_1pct
            else:
                final_chart = scatter + line

            st.altair_chart(final_chart.properties(
                title="📊 Colorado Springs ZIPs vs National Rent-Price Trend",
                height=450
            ).configure_title(anchor="start"), use_container_width=True)

            st.markdown("""
            **How to Interpret This Chart:**

            - Each dot is a Colorado Springs ZIP, positioned by its home price and actual median rent.
            - The **orange line** is the national log-log rent model — it shows what rent *should* be based on typical U.S. pricing trends.
            - **Green dots** are ZIPs with *higher-than-expected* rent — may indicate stronger rental demand or above-market pricing power.
            - **Red dots** are ZIPs with *lower-than-expected* rent — may signal underpriced markets or areas with weaker rent growth.
            - If shown, the **dashed gray line** represents the 1% Rule (rent = 1% of home price).
            """)

        else:
            st.info("Not enough local ZIP data available for comparison.")

        st.markdown("""---""")
        st.markdown(
            "<h4 style='text-align: center; margin-top: 2rem; margin-bottom: 2rem;'>📍 ZIP-Specific Rent Details</h4>",
            unsafe_allow_html=True
        )

        zip_list = results["Zip_Label"].sort_values().unique()
        selected_zip = st.selectbox("Select a ZIP Code:", zip_list)

        selected_zip_code = selected_zip.split(" - ")[0]

        rent_history = rent_df.copy()
        rent_history = rent_history[rent_history["RegionName"].astype(str).str.zfill(5) == selected_zip_code]

        rent_ts = rent_history.loc[:, rent_history.columns.str.match(r"\d{4}-\d{2}-\d{2}")].T
        rent_ts.columns = ["Rent"]
        rent_ts.index = pd.to_datetime(rent_ts.index)
        rent_ts = rent_ts.sort_index()

        latest_month = rent_ts.index.max()
        last_month_rent = rent_ts.loc[latest_month, "Rent"]
        avg_12mo_rent = rent_ts.loc[rent_ts.index >= latest_month - pd.DateOffset(months=11), "Rent"].mean()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("📅 Last Month's Rent", f"${last_month_rent:,.0f}")
        with col2:
            st.metric("📊 12-Month Avg Rent", f"${avg_12mo_rent:,.0f}")

        st.markdown("<h4 style='text-align: center;'>📈 Rent Trend Over Time</h4>", unsafe_allow_html=True)

        chart = (
            alt.Chart(rent_ts.reset_index(), height=400)
            .mark_line(point=True)
            .encode(
                x=alt.X("index:T", title="Month", axis=alt.Axis(format="%b %Y")),
                y=alt.Y("Rent:Q", title="Rent ($)"),
                tooltip=[
                    alt.Tooltip("index:T", title="Month", format="%B %Y"),
                    alt.Tooltip("Rent:Q", title="Rent ($)", format=",.0f")
                ]
            )
            .properties(title=f"Zillow Rent Trend: {selected_zip}")
            .configure_title(anchor="middle")
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown("#### 🧮 \"1% Rule\" - Estimated Rent Based on Property Value")

        col1, col2 = st.columns(2)
        with col1:
            user_price = st.number_input("Estimated Home Price ($)", min_value=50000, max_value=2000000, value=450000, step=5000)
        with col2:
            rent_yield_pct = st.slider("Rent Yield (%)", 0.2, 1.5, 1.0, 0.05)

        estimated_rent = user_price * (rent_yield_pct / 100)
        st.metric("💰 Estimated Monthly Rent", f"${estimated_rent:,.0f}")

        if not rent_ts.empty:
            st.markdown(
                f"<p style='text-align: center; font-size: 16px;'>"
                f"📉 <strong>Zillow's last reported rent for {selected_zip_code}:</strong> ${last_month_rent:,.0f} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"🧮 <strong>Your estimate:</strong> ${estimated_rent:,.0f}"
                f"</p>",
                unsafe_allow_html=True
            )
if __name__ == "__main__":
    main()
