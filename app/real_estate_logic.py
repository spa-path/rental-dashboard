# logic.py
# This module contains all non-UI business logic used for the rental ROI dashboard.

import pandas as pd
import numpy as np
import os
import streamlit as st
import pandas as pd


def load_data(home_path, rent_path):
    try:
        home_df = pd.read_csv(home_path)
        rent_df = pd.read_csv(rent_path)
        return home_df, rent_df
    except Exception as e:
        print("❌ Error loading files:", e)
        return None, None

DEPRECIATION_YEARS = 27.5

# Maps Colorado Springs ZIP codes to readable neighborhood labels.
def filter_and_label_zips(df):
    names = {
        "80902": "Fort Carson", "80903": "Downtown", "80904": "Old Colorado City",
        "80905": "Southwest", "80906": "Broadmoor", "80907": "North Central",
        "80908": "Black Forest", "80909": "East Central", "80910": "Southeast",
        "80911": "Security-Widefield", "80915": "Cimarron Hills", "80916": "South Central",
        "80917": "Village Seven", "80918": "Austin Bluffs", "80919": "Rockrimmon",
        "80920": "Briargate", "80921": "Northgate", "80922": "Stetson Hills",
        "80923": "Ridgeview", "80924": "Cordera", "80925": "Schriever Area",
        "80926": "Cheyenne Mountain", "80927": "Banning Lewis", "80928": "SE Rural",
        "80929": "Ellicott", "80930": "East Rural", "80938": "East Springs",
        "80939": "BL North", "80829": "Manitou", "80817": "Fountain"
    }
    df = df.copy()
    df.rename(columns={"RegionName": "Zip_Code"}, inplace=True)
    df["Zip_Code"] = df["Zip_Code"].astype(str)
    df["Zip_Label"] = df["Zip_Code"] + " - " + df["Zip_Code"].map(names)
    return df

# Aligns home and rent data to latest common month, returns merged DataFrame and date.
def prepare_merged_data(home_df, rent_df):
    processed_home_df = filter_and_label_zips(home_df)
    processed_rent_df = filter_and_label_zips(rent_df)

    home_dates = [col for col in processed_home_df.columns if col.count("-") == 2]
    rent_dates = [col for col in processed_rent_df.columns if col.count("-") == 2]
    common_dates = sorted(set(home_dates).intersection(rent_dates))
    if not common_dates:
        return None, None

    latest_month = common_dates[-1]

    home_subset = processed_home_df[["Zip_Code", "Zip_Label", latest_month]].copy()
    rent_subset = processed_rent_df[["Zip_Code", latest_month]].copy()
    merged = pd.merge(home_subset, rent_subset, on="Zip_Code", suffixes=("_Price", "_Rent"))

    merged["Home_Price"] = pd.to_numeric(merged[f"{latest_month}_Price"], errors="coerce")
    merged["Rent"] = pd.to_numeric(merged[f"{latest_month}_Rent"], errors="coerce")

    return merged.dropna().copy(), latest_month

# Extracts national rent-price training data for log-log regression.
def get_national_training_data(home_df, rent_df, latest_month):
    home = home_df.rename(columns={"RegionName": "Zip_Code"}).copy()
    rent = rent_df.rename(columns={"RegionName": "Zip_Code"}).copy()

    home["Zip_Code"] = home["Zip_Code"].astype(str).str.zfill(5)
    rent["Zip_Code"] = rent["Zip_Code"].astype(str).str.zfill(5)

    home_prices = home[["Zip_Code", latest_month]].rename(columns={latest_month: "Home_Price"})
    rents = rent[["Zip_Code", latest_month]].rename(columns={latest_month: "Rent"})

    merged = pd.merge(home_prices, rents, on="Zip_Code").dropna()
    merged = merged[(merged["Home_Price"] > 0) & (merged["Rent"] > 0)]
    return merged

# Core ROI, tax savings, and equity calculations.
def calculate_financial_metrics(valid_data, params):
    data = valid_data.copy()

    int_rate = params["interest_rate"] / 100
    monthly_int = int_rate / 12
    months = 30 * 12
    years = params["appreciation_years"]

    data["Down_Payment"] = data["Home_Price"] * (params["down_payment_pct"] / 100)
    data["Loan_Amount"] = data["Home_Price"] - data["Down_Payment"]

    mortgage = data["Loan_Amount"] * (monthly_int * (1 + monthly_int) ** months) / ((1 + monthly_int) ** months - 1)


    monthly_ins = params["insurance_annual"] / 12
    maint = data["Home_Price"] * params["maintenance_rate"] / 12
    vacancy = data["Rent"] * params["vacancy_rate"]
    mgmt_fee = data["Rent"] * params["property_mgmt_pct"]
    capex = params["capex_monthly"]
    property_tax = data["Home_Price"] * (params["property_tax_rate"] / 100) / 12

    expenses = mortgage + monthly_ins + maint + vacancy + mgmt_fee + capex + property_tax
    data["Monthly_CF"] = data["Rent"] - expenses
    data["Annual_CF"] = data["Monthly_CF"] * 12

    data["Cash_Down"] = data["Down_Payment"]
    data["Closing_Costs"] = data["Home_Price"] * (params["closing_cost_pct"] / 100)
    data["Cash_In"] = data["Cash_Down"] + data["Closing_Costs"]

    data["Structure_Value"] = data["Home_Price"] * params["structure_pct"]
    data["Depreciation"] = data["Structure_Value"] / DEPRECIATION_YEARS
    data["Tax_Savings"] = data["Depreciation"] * (params["marginal_tax_rate"] / 100)

    principal_year1 = []
    for i in range(len(data)):
        bal = data["Loan_Amount"].iloc[i]
        pmt = mortgage.iloc[i]
        paid = 0
        for _ in range(12):
            int_pmt = bal * monthly_int
            princ_pmt = pmt - int_pmt
            paid += princ_pmt
            bal -= princ_pmt
        principal_year1.append(paid)

    data["Basic_CoC"] = (data["Annual_CF"] / data["Cash_In"]) * 100
    data["Advanced_CoC"] = (
        (data["Annual_CF"] + data["Tax_Savings"] + pd.Series(principal_year1, index=data.index))
        / data["Cash_In"]
    ) * 100

    total_principal_paid = []
    for i in range(len(data)):
        bal = data["Loan_Amount"].iloc[i]
        pmt = mortgage.iloc[i]
        paid = 0
        for _ in range(years * 12):
            int_pmt = bal * monthly_int
            princ_pmt = pmt - int_pmt
            paid += princ_pmt
            bal -= princ_pmt
        total_principal_paid.append(paid)

    app_rate = params["annual_appreciation_pct"] / 100
    data["Appreciation_Gain"] = data["Home_Price"] * ((1 + app_rate) ** years) - data["Home_Price"]
    data["Equity_From_Paydown"] = pd.Series(total_principal_paid, index=data.index)

    data["MultiYear_Income_Gain"] = (
        data["Annual_CF"] * years +
        data["Tax_Savings"] * years +
        data["Equity_From_Paydown"]
    )

    data["Total_Return"] = data["Appreciation_Gain"] + data["MultiYear_Income_Gain"]
    data["Total_ROC"] = (data["Total_Return"] / data["Cash_In"]) * 100


    return data
