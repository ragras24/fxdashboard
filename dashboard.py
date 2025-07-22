import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go


# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="FX Insights Dashboard", layout="wide")

#---------- CUSTOM CSS ----------
st.markdown("""
    <style>
    /* Metric box styling (unchanged) */
    [data-testid="stMetric"] {
        background-color: #1c1c1c;
        color: white;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0px 0px 8px rgba(255, 255, 255, 0.05);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Style for st.metric() boxes */
    [data-testid="stMetric"] {
        background-color: #000000; /* pure black */
        color: #d3d3d3;  /* light grey text */
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0px 0px 8px rgba(255, 255, 255, 0.05);
        text-align: center;
    }

    /* Ensure inner text like delta and value is also grey */
    [data-testid="stMetric"] div {
        color: #d3d3d3 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR INPUTS ----------
st.sidebar.title("⚙️ Model Configuration")

country_currency = st.sidebar.selectbox(
    "Country and Currency",
    options=["Japan - JPY", "Australia - AUD", "Canada - CAD", "Eurozone - EUR", "United Kingdom - GBP"]
)

if country_currency == "Japan - JPY":
    data = pd.read_csv(r"Monthly Data/Japan Monthly Data - Final.csv")
    currency_symbol = "¥"
    base_quote = "USD/JPY"
    default_variables = ["Policy Rate Spread", "Japan Debt/GDP", "Relative Inflation", "10 Year Bond Yield Spread", "US Current Account Balance (USD)", "US Trade Policy Uncertainty", "Japan Consumer Sentiment", "Japan % Gold Reserves", "WTI Crude Oil (USD/Barrel)"]
    variable_def = {"Policy Rate Spread": "Difference between US and Japan's policy (overnight interbank lending) rates",
                    "Japan Total FX Reserves": "Total Foreign Currency Reserves held by the Bank of Japan, reported in USD",
                    "Debt/GDP": "Country's federal government debt as a ratio of its GDP, measured quarterly (US) and annually (Japan)",
                    "Relative Inflation": "The difference in inflation rate between the US and Japan",
                    "10 Year Bond Yield Spread": "The yield differential between the US and Japan's 10 Year Bonds",
                    "US Current Account Balance": " The US Current Account Balance, measured quarterly and reported in USD",
                    "Japan Current Account Balance": "Japan's Current Account Balance, reported in Yen",
                    "US Trade Policy Uncertainty": "The US Trade Policy Uncertainty Index quantifies the level of uncertainty surrounding U.S. trade policy based on frequency of newspaper articles that reference trade-related terms alongside words like 'uncertain' or 'uncertainty'", 
                    "Japan Consumer Sentiment": "A survey based percentage measure of Japanese households' who have confidence over pessimism towards the economy",
                    "US Consumer Sentiment": "An index of a survey by the University of Michigan, with 1966-Q1 = 100 that gauges of U.S. household confidence in current and expected economic conditions",
                    "Geopolitical Risk": "A quantitative index measuring geopolitical tensions based on the frequency of newspaper articles referencing geopolitical events",
                    "Japan Investor Sentiment": "A measure of market sentiment based on Japan’s International Investment Position (IIP), showing net purchase of net foreign asset holdings, reported in Yen",
                    "US Stock Market": "The Dow Jones Total Stock Market Index",
                    "Japan % Gold Reserves": "Proportion of total reserves held by Japan in gold, measured quarterly",
                    "Implied PPP": "Purchasing Power Parity exchange rate between the US and Japan, measured annually by the IMF",
                    "Japan Stock Market": "The Nikkei 225 Index",
                    "WTI Crude Oil": "Crude Oil prices per Barrel, reported in USD"}
    
elif country_currency == "Australia - AUD":
    data = pd.read_csv(r"Monthly Data/Australia Monthly Data - Final.csv")
    currency_symbol = "A$"
    base_quote = "USD/AUD"
    default_variables = ["Commodity Exports to China (USD)", "Implied PPP", "Global Economic Policy Uncertainty", "Relative Inflation", "Australia Current Account Balance (USD)","Australia % Gold Reserves", "US Trade Policy Uncertainty", "US Debt/GDP", "10 Year Bond Yield Spread"]
    variable_def = {"Policy Rate Spread": "Difference between US and Australia's policy (overnight interbank lending) rates",
                    "Australia Total FX Reserves": "Total Foreign Currency Reserves held by the Reserve Bank of Australia, reported in USD",
                    "Debt/GDP": "Country's federal government debt as a ratio of its GDP, measured quarterly (US), and annually (Australia)",
                    "Australia % Gold Reserves": "Proportion of total reserves held by Australia in gold",
                    "Australia Gold Reserves": "Value of gold held by the Reseve Bank of Australia, reported in USD",
                    "Relative Inflation": "The difference in inflation rate between the US and Australia, measured quarterly",
                    "10 Year Bond Yield Spread": "The yield differential between the US and Australia's 10 Year Bonds",
                    "US Current Account Balance": "The US's Current Account Balance, measured quarterly and reported in USD",
                    "Australia Current Account Balance": "Australia's Current Account Balance, measured annually and reported in USD",
                    "Implied PPP": "Purchasing Power Parity Exchange Rate between the US and Australia, measured annually by the IMF",
                    "US Trade Policy Uncertainty": "The US Trade Policy Uncertainty Index quantifies the level of uncertainty surrounding U.S. trade policy based on frequency of newspaper articles that reference trade-related terms alongside words like 'uncertain' or 'uncertainty'", 
                    "Commodity Exports to China": "Value of Australian Commoditity Exports to China, reported in USD",
                    "Global Economic Policy Uncertainty": "An indexed measure of global economic uncertainty based on the frequency of policy-related terms in international media",
                    }
    

elif country_currency == "Canada - CAD":
    data = pd.read_csv(r"Monthly Data/Canada Monthly Data - Final.csv")
    currency_symbol = "C$"
    base_quote = "USD/CAD"
    default_variables = ["Relative Inflation", "Implied PPP", "Global Economic Policy Uncertainty", "Policy Rate Spread", "Canada Debt/GDP", "Canada Current Account Balance (CAD)", "Canada % Gold Reserves", "Canada USD Reserves (USD)", "WTI Crude Oil (USD/Barrel)"]
    variable_def = {"Policy Rate Spread": "Difference between US and Canada's policy (overnight interbank lending) rates",
                    "Canada USD Reserves": "USD reserves held by the Bank of Canda, reported in USD",
                    "Canada Other FX Reserves": "Other FX reserves held by the Bank of Canda, reported in USD",
                    "Debt/GDP": "Country's federal government debt as a ratio of its GDP, measured quarterly (US) and annually (Canada)",
                    "Canada % Gold Reserves": "Proportion of total reserves held by Canada in gold, measured quarterly",
                    "Debt/GDP": "Country's federal government debt as a ratio of its GDP, measured quarterly (US and Canada)",
                    "Relative Inflation": "The difference in inflation rate between the US and Canada, measured quarterly",
                    "10 Year Bond Yield Spread": "The yield differential between the US and Canada's 10 Year Bonds",
                    "US Current Account Balance": "The US's Current Account Balance, measured quarterly and reported in USD",
                    "Canada Current Account Balance": "Canada's Current Account Balance, measured quarterly and reported in CAD",
                    "Implied PPP": "Purchasing Power Parity Exchange Rate between the US and Canada, measured annually by the IMF",
                    "Canada Commodities Price Index": "Index of the spot prices of 26 commodities produced in Canada, reported in USD",
                    "WTI Crude Oil": "Crude Oil prices per Barrel, reported in USD",
                    "US Trade Policy Uncertainty": "The US Trade Policy Uncertainty Index quantifies the level of uncertainty surrounding U.S. trade policy based on frequency of newspaper articles that reference trade-related terms alongside words like 'uncertain' or 'uncertainty'",
                    "Global Economic Policy Uncertainty": "An indexed measure of global economic uncertainty based on the frequency of policy-related terms in international media",
                    }
    

elif country_currency == "United Kingdom - GBP":
    data = pd.read_csv(r"Monthly Data/UK Monthly Data - Final.csv")
    currency_symbol = "£"
    base_quote = "USD/GBP"  
    default_variables = ["UK Current Account Balance (USD)", "US Current Account Balance (USD)", "10 Year Bond Yield Spread", "UK Service Exports (GBP)", "UK Debt/GDP", "Policy Rate Spread", "Implied PPP", "UK Economic Policy Uncertainty", "UK Consumer Sentiment", "US Trade Policy Uncertainty", "US Share of Global Trade"]
    variable_def = {"UK Current Account Balance": "The UK's Current Account Balance, measured annually and reported in USD",
                    "US Current Account Balance": "The US's Current Account Balance, measured quarterly and reported in USD",
                    "10 Year Bond Yield Spread": "The yield differential between the US and UK's 10 Year Bonds",
                    "UK Service Exports": "The total value of services the UK sells abroad, measured quarterly and reported in GBP",
                    "UK Service Balance": "The difference in value between UK service exports and imports, measured quarterly and reported in GBP",
                    "Debt/GDP": "Country's federal government debt as a ratio of its GDP, measured quarterly (US and UK)",
                    "Policy Rate Spread": "Difference between US and UK's policy (overnight interbank lending) rates",
                    "UK Total FX Reserves": "Foreign-denominated assets (excluding gold) held by the Bank of England, reported in USD ",
                    "Implied PPP": "Purchasing Power Parity Exchange Rate between the US and the UK, measured annually by the IMF",
                    "UK Economic Policy Uncertainty": "An index measuring uncertainty about UK economic policy based on newspaper mentions.",
                    "UK Consumer Sentiment": "A survey based percentage measure of British households' who have confidence over pessimism towards the economy",
                    "Geopolitical Risk": "A quantitative index measuring geopolitical tensions based on the frequency of newspaper articles referencing geopolitical events",
                    "UK % Gold Reserves": "Proportion of total reserves held by the UK in gold, measured quarterly",
                    "US Trade Policy Uncertainty": "The US Trade Policy Uncertainty Index quantifies the level of uncertainty surrounding U.S. trade policy based on frequency of newspaper articles that reference trade-related terms alongside words like 'uncertain' or 'uncertainty'",
                    "Share of Global GDP": "The country’ GDP as a proportion of global GDP, reflecting its weight in the global economy, measured annually",
                    "Share of Global Trade": "The country’s share of global trade activity, based on a sum of the country's total imports and exports, measured annually",
                    }


elif country_currency == "Eurozone - EUR":
    data = pd.read_csv(r"Monthly Data/EU Monthly Data - Final.csv")
    currency_symbol = "€"
    base_quote = "USD/EUR"
    default_variables = ["Implied PPP", "EU Total FX Reserves (Euro)", "10 Year Bond Yield Spread", "NG1 Natural Gas Futures Price (USD)", "Relative Inflation", "EU Gold Reserves (Euro)", "EU Current Account Balance (USD)", "US Trade Policy Uncertainty", "Geopolitical Risk", "Policy Rate Spread"]
    variable_def = {"10 Year Bond Yield Spread": "The yield differential between the US and Euro 10 Year Bonds",
                    "Euro Area Consumer Sentiment": "A survey based percentage measure of households in the Euro Area who have confidence over pessimism towards the economy",
                    "Debt/GDP": "Country/region's government debt as a ratio of its GDP, measured quarterly (US) and annually (Euro Area)",
                    "Relative Inflation": "The difference in inflation rate between the US and Euro Area, measured annually",
                    "EU Current Account Balance": "The current account balances of the EU, measured annually and reported in USD",
                    "Global Economic Policy Uncertainty": "An indexed measure of global economic uncertainty based on the frequency of policy-related terms in international media",
                    "Geopolitical Risk": "A quantitative index measuring geopolitical tensions based on the frequency of newspaper articles referencing geopolitical events",
                    "Policy Rate Spread": "Difference between US and Euro Area's policy (overnight interbank lending) rates",
                    "EU Total FX Reserves": "The value of total foreign exchange reserves held by European Union central banks, reported in Euro",
                    "EU Gold Reserves": "The value of gold reserves held by European Union central banks, reported in Euro",
                    "Implied PPP": "Purchasing Power Parity Exchange Rate between USD and Euro, measured annually by the IMF",
                    "EU Trade Balance": "The difference between the total value of goods and services exported and imported by the European Union, reported in USD",
                    "US Trade Policy Uncertainty": "The US Trade Policy Uncertainty Index quantifies the level of uncertainty surrounding U.S. trade policy based on frequency of newspaper articles that reference trade-related terms alongside words like 'uncertain' or 'uncertainty'",
                    "NG1 Natural Gas Futures Price": "The price of natural gas futures contracts for the nearest delivery month, reported in USD",
                    "EU Unemployment Rate": "Percentage of workforce in the European Union that is unemployed, measured monthly",
    }

else:
    st.warning("Dataset for this country is not loaded yet.")
    st.stop()

model_choice = st.sidebar.radio(
    "Select Model Type:",
    ["Scorecard", "ARIMA + Gradient Boosting"]
)

if model_choice == "Scorecard":

    data["Date"] = pd.to_datetime(data["Date"], errors='coerce')
    data = data.set_index("Date").sort_index()
    data.index = data.index.date



    #Dates Input
    available_dates = data.index
    start_input = st.sidebar.date_input("Start Date", value=available_dates.min(), min_value=available_dates.min(), max_value=available_dates.max())

    end_input = st.sidebar.date_input("End Date", value=available_dates.max(), min_value=available_dates.min(), max_value=available_dates.max())

    start_input = pd.to_datetime(start_input).date()
    end_input = pd.to_datetime(end_input).date()



    # Filter your data using the adjusted dates
    filtered_data = data.loc[start_input:end_input]

    na_variables = filtered_data.columns[filtered_data.isna().any()].to_list()

    # Drop vraibles that have NA values (insufficient data)
    if na_variables:
        st.warning(f"Dropping the following variables due to insufficient data: {', '.join(na_variables)}")
        filtered_data = filtered_data.drop(columns = na_variables)

    y_raw = filtered_data.iloc[:, 0]  # first column = fx_rate target
    X_raw_all = filtered_data.iloc[:, 1:]  # all other columns as features



    # # Variable descriptions
    # var_descriptions = {
    #     "Interest Rate Differential": "The difference between U.S. and Japan interest rates.",
    #     "Inflation Rate": "CPI or core CPI inflation used as a monetary pressure proxy.",
    #     "GDP Growth": "Quarterly or yearly change in GDP.",
    #     "Money Supply": "Monetary base growth, indicating liquidity.",
    #     "Trade Balance": "Exports minus imports, affecting currency demand.",
    #     # Add all relevant variables here
    # }


    independent_vars = st.sidebar.multiselect(
        "Independent Variables",
        options=X_raw_all.columns.to_list(),
        default= default_variables)

    # # Display descriptions below
    # with st.sidebar.expander("ℹ️ Variable Definitions"):
    #     for var in independent_vars:
    #         if var in var_descriptions:
    #             st.markdown(f"**{var}**: {var_descriptions[var]}")


    if len(independent_vars) == 0:
        st.error("Please select at least one independent variable.")
        st.stop()


    # Subset predictors to selected variables
    X_raw = X_raw_all[independent_vars]

    # 🔍 Drop any predictors with 0 standard deviation (flat series)
    X_std_check = X_raw.std()
    zero_std_vars = X_std_check[X_std_check == 0].index.tolist()

    if zero_std_vars:
        st.warning(f"These variables have no variation and were removed: {', '.join(zero_std_vars)}")
        X_raw = X_raw.drop(columns=zero_std_vars)
        independent_vars = [v for v in independent_vars if v not in zero_std_vars]



    # ---------- HEADER ----------
    st.title("FX Insights Dashboard")



    # ---------- HISTORICAL CHART WITH PREDICTION ----------
    st.subheader(f"📈 {base_quote} Exchange Rate - Historical & Predicted")

    # Create the main chart first (we'll update it with prediction later)
    def create_exchange_rate_chart(historical_data, predicted_value=None, prediction_date=None, actual_value=None):
        fig = go.Figure()
        
        # Historical data line
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data.values,
            mode='lines',
            name=f'Historical {base_quote}',
            line=dict(color="#42cfd4", width=2),
            hovertemplate='<b>Date:</b> %{x|%b %Y}<br><b>Rate:</b> %{y:.4f}<extra></extra>'
        ))
        
        # Add predicted point if provided
        if predicted_value is not None and prediction_date is not None:
            # Add a connecting line from last historical point to prediction
            last_date = historical_data.index[-1]
            last_value = historical_data.iloc[-1]
            
            fig.add_trace(go.Scatter(
                x=[last_date, prediction_date],
                y=[last_value, predicted_value],
                mode='lines',
                name='Prediction Connection',
                line=dict(color='#f65d35', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Add predicted point
            fig.add_trace(go.Scatter(
                x=[prediction_date],
                y=[predicted_value],
                mode='markers',
                name='Predicted Value',
                marker=dict(
                    color='#ff7f0e',
                    size=10,
                    symbol='diamond',
                    line=dict(color='white', width=2)
                ),
                hovertemplate='<b>Predicted Date:</b> %{x|%b %Y}<br><b>Predicted Rate:</b> %{y:.4f}<extra></extra>'
            ))

            if actual_value is not None:
                fig.add_trace(go.Scatter(
                    x=[prediction_date],
                    y=[actual_value],
                    mode='markers',
                    name='Actual Value',
                    marker=dict(
                        color='green',
                        size=10,
                        symbol='circle',
                        line=dict(color='white', width=2)
                    ),
                    hovertemplate='<b>Actual Date:</b> %{x|%b %Y}<br><b>Actual Rate:</b> %{y:.4f}<extra></extra>'
                ))

        # Update layout
        fig.update_layout(
            title=dict(
                text="",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Date",
            yaxis_title=f"Exchange Rate ({base_quote})",
            hovermode='x unified',
            template='plotly_white',
            height=400,
            margin=dict(l=50, r=50, t=60, b=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig

    # Initially show just historical data
    chart_placeholder = st.empty()
    with chart_placeholder.container():
        fig = create_exchange_rate_chart(y_raw)
        st.plotly_chart(fig, use_container_width=True)

    # ---------- MODEL OUTPUT ----------
    st.subheader(f"📊 Results from: {model_choice}")

    # --------------------- SCORECARD MODEL ---------------------

    # 1. Standardize X and y (based on selected variables)
    X_mean = X_raw.mean()
    X_std = X_raw.std()
    X = (X_raw - X_mean) / X_std

    y_mean = y_raw.mean()
    y_std = y_raw.std()
    y = (y_raw - y_mean) / y_std



    # 2. Fit regression with robust SE
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit(cov_type='HC3')
    r_squared = model.rsquared

    betas = model.params
    beta_no_const = betas.drop("const") if "const" in betas else betas
    abs_betas = np.abs(beta_no_const)
    abs_betas_sum = abs_betas.sum()
    normalized_weights = abs_betas / abs_betas_sum
    beta_signs = np.sign(beta_no_const)

    # Calculate model performance metrics on historical data
    y_pred_historical = model.predict(X_with_const)
    y_pred_historical_original_scale = y_pred_historical * y_std + y_mean
    
    # Calculate RMSE and MAE in original scale
    rmse = np.sqrt(mean_squared_error(y_raw, y_pred_historical_original_scale))
    mae = mean_absolute_error(y_raw, y_pred_historical_original_scale)
    # Convert to percentage
    mean_actual = y_raw.mean()
    rmse_pct = (rmse / mean_actual) * 100
    mae_pct = (mae / mean_actual) * 100


    # Convert to DataFrame
    df_definitions = pd.DataFrame(variable_def.items(),columns=["Variable", "Explanation"])
    
    #Variable definitions:
    with st.expander("📘 Predictor Variable Definitions"):
        st.dataframe(df_definitions, use_container_width=True, hide_index = True)

    col_w, col_v, col_s = st.columns([0.95, 0.95, 1.1])


    # 1) Weight sliders (default = normalized weights)
    default_values = X_raw.iloc[-1]
    estimated_display_impact = {}

    for v in independent_vars:
        standardized_val = (default_values[v] - X_mean[v]) / X_std[v]
        beta_est = beta_signs[v] * normalized_weights[v] * abs_betas_sum
        estimated_display_impact[v] = beta_est * standardized_val * y_std

    # Sort ONLY for display
    sorted_display_vars = sorted(estimated_display_impact, key=lambda k: abs(estimated_display_impact[k]), reverse=True)
    top_vars = sorted_display_vars[:4]
    other_vars = sorted_display_vars[4:]
    # Sort or keep as-is depending on context
    # top_vars = independent_vars[:6]
    # other_vars = independent_vars[6:]

    with col_w:
        st.markdown("#### Set Variable Importance")
        st.markdown("")
        weights = {}
        total_weight = 0
        for var in top_vars:
            default_val = round(normalized_weights[var], 3)
            weights[var] = st.slider(f"{var}", 0.0, 1.0, default_val, step=0.001, format ='%.3f' , key = f"weight_slider_{var.replace(' ', '_')}")
            total_weight += weights[var]

        if other_vars:
            with st.expander("Display More Variables"):
                for var in other_vars:
                    default_val = round(normalized_weights[var], 3)
                    weights[var] = st.slider(f"{var}", 0.0, 1.0, default_val, step=0.001, format ='%.3f' , key = f"weight_slider_{var.replace(' ', '_')}")
                    total_weight += weights[var]

        # if abs(total_weight - 1) > 0.001:
        #     st.error(f"⚠️ Weights must sum to 1. Current sum: {round(total_weight,3)}")
        #     st.stop()

    # 2) Predictor values input (raw scale)


    with col_v:
        st.markdown("#### Enter Predicted Values")
        st.markdown("")
        values = {}
        for var in top_vars:
            min_val = float(X_raw[var].min())
            max_val = float(X_raw[var].max())
            default_val = float(X_raw[var].iloc[-1])

            # buffer = (max_val - min_val) * 0.6  # 60% range buffer
            # input_min = min(min_val, default_val) - buffer
            # input_max = max(max_val, default_val) + buffer

            values[var] = st.number_input(f"{var}", value=default_val, step=0.01,key=f"predictor_input_{var.replace(' ', '_')}")
            st.markdown("")

        if other_vars:
            with st.expander("Display More Variables"):
                for var in other_vars:
                    min_val = float(X_raw[var].min())
                    max_val = float(X_raw[var].max())
                    default_val = float(X_raw[var].iloc[-1])

                    # buffer = (max_val - min_val) * 0.6  # 60% range buffer
                    # input_min = min(min_val, default_val) - buffer
                    # input_max = max(max_val, default_val) + buffer

                    values[var] = st.number_input(f"{var}", value=default_val, step=0.01,key=f"predictor_input_{var.replace(' ', '_')}")
                    st.markdown("")
        


    # 3) Score summary and prediction
    with col_s:
        st.markdown("#### 💡 Score Summary")

        # Convert weights back to betas (with sign and magnitude)
        betas_manual = {}
        for v in independent_vars:
            betas_manual[v] = beta_signs[v] * weights[v] * abs_betas_sum

        # Standardize user input values
        values_std = {}
        for v in independent_vars:
            values_std[v] = (values[v] - X_mean[v]) / X_std[v]

        # Predict standardized y
        y_pred_std = sum(betas_manual[v] * values_std[v] for v in independent_vars)

        # Convert predicted y back to original scale
        y_pred = y_pred_std * y_std + y_mean

        # Prepare scorecard dataframe
        scorecard_df = pd.DataFrame({
            "Variable": independent_vars,
            "Weight": [weights[v] for v in independent_vars],
            "Weighted Impact": [(betas_manual[v] * values_std[v])*y_std  for v in independent_vars]
        })

        scorecard_df["Direction"] = scorecard_df["Weighted Impact"].apply(lambda x: "↑" if x > 0 else "↓")

        total_score = scorecard_df["Weighted Impact"].sum()



        # Display predicted FX rate
        st.metric("Predicted FX Rate", f"{currency_symbol}{round(y_pred, 4)}")
        
        # FX Strength Indicator
        if y_pred > (y_raw.iloc[-1]):
            st.success("__The USD is Expected to Appreciate__ 📈")
        elif y_pred < (y_raw.iloc[-1]):
            st.error("__The USD is Expected to Depreciate__ 📉")
        else:
            st.info("__The USD is Expected to Remain Stable__ ⚖️")

        
        
        st.info(f"**Root Mean Square Error:** {rmse_pct:.2f}%")
        st.info(f"**Mean Absolute Error:** {mae_pct:.2f}%")
        st.info(f"**Model R²:** {r_squared * 100:.2f}%")



        st.dataframe(scorecard_df.set_index("Variable"))

        # Update the chart with the prediction
        # Use next business day as prediction date
        last_date = y_raw.index[-1]
        prediction_date = last_date + pd.Timedelta(days=28)

        # Try to get actual FX value from full data (not filtered)
        try:
            # Get the first available date in full data on or after prediction_date
            future_date = min(d for d in data.index if d > prediction_date)
            actual_fx = data.loc[future_date, data.columns[0]]
            show_actual = True
        except (ValueError, KeyError):
            actual_fx = None
            show_actual = False
        
        # Update the chart with prediction
        with chart_placeholder.container():
            fig_updated = create_exchange_rate_chart(y_raw, y_pred, prediction_date, actual_value=actual_fx if show_actual else None)
            st.plotly_chart(fig_updated, use_container_width=True)

else: 

    # Final Model B: SARIMAX + XGBoost (FIXED)

    def rolling_sarimax_plus_xgb_classifier(
        dataset,
        features,
        target_col,
        min_history_for_lags=10,
        sarimax_order=(1, 0, 0),
        residual_threshold=0
    ):
        n = dataset.shape[0]
        initial_train_size = int(n * 0.90)
        preds_sarimax = []
        preds_xgb = []
        actuals = []
        pred_dates = []

        
        # We'll keep track of the last scaler for final forecast scaling
        last_scaler = None

        for i in range(initial_train_size, n):
            train = dataset.iloc[:i]
            test = dataset.iloc[i:i+1]

            # --- Scale features ---
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(train[features]),
                columns=features,
                index=train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(test[features]),
                columns=features,
                index=test.index
            )
            last_scaler = scaler  # update last scaler for forecasting step

            try:
                model = SARIMAX(
                    train[target_col],
                    exog=X_train_scaled,
                    order=sarimax_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                sarimax_res = model.fit(disp=False)


            except Exception as e:
                print(f"SARIMAX fit failed at index {i} with error: {e}")
                preds_sarimax.append(train[target_col].iloc[-1])
                preds_xgb.append(train[target_col].iloc[-1])
                actuals.append(test[target_col].values[0])
                pred_dates.append(test.index[0])
                continue

            sarimax_pred = sarimax_res.forecast(steps=1, exog=X_test_scaled).iloc[0]
            preds_sarimax.append(sarimax_pred)

            residuals = train[target_col] - sarimax_res.fittedvalues

            # XGBoost fit on scaled features and residuals
            xgb_model = XGBRegressor(n_estimators=20, max_depth=3, learning_rate=0.1)
            xgb_model.fit(X_train_scaled, residuals)

            xgb_residual_pred = xgb_model.predict(X_test_scaled)[0]

            final_pred = sarimax_pred + xgb_residual_pred
            preds_xgb.append(final_pred)
            actuals.append(test[target_col].values[0])
            pred_dates.append(test.index[0])

        # === Forecast next unseen point ===
        next_date = dataset.index[-1] + pd.DateOffset(months=1)

        try:
            # Scale full dataset features using last scaler
            X_scaled_full = pd.DataFrame(
                last_scaler.transform(dataset[features]),
                columns=features,
                index=dataset.index
            )

            final_model = SARIMAX(
                dataset[target_col],
                exog=X_scaled_full,
                order=sarimax_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            final_res = final_model.fit(disp=False)

            # Use last row of scaled features as future exog
            future_exog = X_scaled_full.iloc[[-1]]
            sarimax_forecast = final_res.forecast(steps=1, exog=future_exog).iloc[0]

            residuals = dataset[target_col] - final_res.fittedvalues
            xgb_model = XGBRegressor(n_estimators=20, max_depth=3, learning_rate=0.1)
            xgb_model.fit(X_scaled_full, residuals)

            xgb_forecast_resid = xgb_model.predict(future_exog)[0]
            final_forecast = sarimax_forecast + xgb_forecast_resid

            preds_xgb.append(final_forecast)
            actuals.append(np.nan)  # No actual for future
            pred_dates.append(next_date)

        except Exception as e:
            print(f"Final step forecast failed: {e}")

        # Metrics on known predictions only
        valid_mask = ~np.isnan(actuals)
        rmse = np.sqrt(mean_squared_error(np.array(actuals)[valid_mask], np.array(preds_xgb)[valid_mask]))
        mae = mean_absolute_error(np.array(actuals)[valid_mask], np.array(preds_xgb)[valid_mask])
        direction_correct = np.sign(np.diff(np.array(preds_xgb)[valid_mask])) == np.sign(np.diff(np.array(actuals)[valid_mask]))
        directional_accuracy = np.mean(direction_correct) * 100


        # Return the last fitted SARIMAX and XGBoost models for interpretation if needed
        return (
            rmse,
            mae,
            directional_accuracy,
            pd.DatetimeIndex(pred_dates),
            actuals,
            preds_xgb,
            final_res,
            xgb_model
        )

    # Make sure 'Date' is datetime type if you want to use it later
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])

            # Transform target variable: log returns of USD/JPY monthly rate
        data[f"{base_quote} (Monthly)"] = np.log(data[f"{base_quote} (Monthly)"])
        data = data.dropna()
        data = data.set_index('Date')
        data = data.asfreq('M')
        
        available_features = [col for col in data.columns if col not in ["Date", f"{base_quote} (Monthly)"]]

        features = st.sidebar.multiselect("Select Predictor Variables", options=available_features, default=default_variables)  # Select all by default
        target_col = f"{base_quote} (Monthly)"

        print("Using features:", features)
        print("Target:", target_col)

        rmse, mae, directional_accuracy, pred_dates, actuals, preds_xgb, final_res, xgb_model = rolling_sarimax_plus_xgb_classifier(
            data,
            features,
            target_col,
            min_history_for_lags=10,
            sarimax_order=(0, 1, 1),
            residual_threshold=0
        )

        # except FileNotFoundError:
        #     print("Error: Could not find 'Japan Dataset(Exchange Rate (2)).csv'")
        #     print("Please ensure the file exists in your working directory.")
        # except Exception as e:
        #     print(f"Error occurred: {str(e)}")
        #     import traceback
        #     traceback.print_exc()

        # ---------- HEADER ----------
        st.title("FX Insights Dashboard")


        # ---------- HISTORICAL CHART WITH PREDICTION ----------
        st.subheader(f"📈 {base_quote} Exchange Rate - Historical & Predicted")

        fig = go.Figure()

        # Historical actual FX before predictions begin
        historical_actuals = data[target_col].iloc[:len(data) - len(actuals)]
        historical_dates = historical_actuals.index

        # Combine dates for actuals (historical + prediction period actuals)
        combined_dates = historical_dates.append(pred_dates)

        # Combine actual FX values (historical + prediction period actuals)
        combined_actuals = pd.concat([
            historical_actuals,
            pd.Series(actuals, index=pred_dates)
        ])

        # Exponentiate actual values to get back to original scale
        combined_actuals_exp = np.exp(combined_actuals)

        # Plot combined actuals as one connected line (same style as Scorecard)
        fig.add_trace(go.Scatter(
            x=combined_dates,
            y=combined_actuals_exp,
            mode='lines',
            name=f'Historical {base_quote}',
            line=dict(color="#42cfd4", width=2),
            hovertemplate='<b>Date:</b> %{x|%b %Y}<br><b>Rate:</b> %{y:.4f}<extra></extra>'
        ))

        # Final prediction date and value (last in preds_xgb and pred_dates)
        final_pred_date = pred_dates[-1]
        final_pred_value = preds_xgb[-1]

        # Exponentiate the final predicted value to original scale
        final_pred_value_exp = np.exp(final_pred_value)

        # Get last actual (non-NaN) value from the full actuals series
        actual_series = pd.Series(actuals, index=pred_dates).dropna()
        if len(actual_series) > 0:
            last_actual_date = actual_series.index[-1]
            last_actual_value_exp = np.exp(actual_series.iloc[-1])
        else:
            # Fallback to last historical value
            last_actual_date = historical_dates[-1]
            last_actual_value_exp = np.exp(historical_actuals.iloc[-1])

        # Add connecting dashed line from last actual to prediction (same style as Scorecard)
        fig.add_trace(go.Scatter(
            x=[last_actual_date, final_pred_date],
            y=[last_actual_value_exp, final_pred_value_exp],
            mode='lines',
            name='Prediction Connection',
            line=dict(color='#f65d35', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Add predicted point (same style as Scorecard)
        fig.add_trace(go.Scatter(
            x=[final_pred_date],
            y=[final_pred_value_exp],
            mode='markers',
            name='Predicted Value',
            marker=dict(
                color='#ff7f0e',
                size=10,
                symbol='diamond',
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>Predicted Date:</b> %{x|%b %Y}<br><b>Predicted Rate:</b> %{y:.4f}<extra></extra>'
        ))

        # Update layout to match Scorecard styling exactly
        fig.update_layout(
            title=dict(
                text="",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Date",
            yaxis_title=f"Exchange Rate ({base_quote})",
            hovermode='x unified',
            template='plotly_white',  # Changed from plotly_dark to match Scorecard
            height=400,  # Changed from 500 to match Scorecard
            margin=dict(l=50, r=50, t=60, b=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        

        # Convert RMSE and MAE to percentage strings
        rmse_pct = f"{rmse * 100:.2f}%"
        mae_pct = f"{mae * 100:.2f}%"

        # Directional accuracy is already percentage, format nicely
        dir_acc_pct = f"{directional_accuracy:.2f}%"

        # ---------- MODEL OUTPUT ----------
        st.subheader(f"📊 Results from: {model_choice}")

        # Variable Definition Dataframe
        df_definitions = pd.DataFrame(variable_def.items(),columns=["Variable", "Explanation"])
        
        #Variable definitions:
        with st.expander("📘 Predictor Variable Definitions"):
            st.dataframe(df_definitions, use_container_width=True, hide_index = True)



        # Create 3 columns
        col1, col2 = st.columns([1.3,0.7])

        with col1:
            st.markdown("#### Feature Contributions")
            
            # 1. Get SARIMAX coefficients and normalize absolute values
            sarimax_coefs = final_res.params[features]
            sarimax_weights = np.abs(sarimax_coefs)
            sarimax_weights /= sarimax_weights.sum()  # normalize to sum to 1

            # 2. Get XGBoost feature importances (raw)
            xgb_importances = xgb_model.feature_importances_

            # 3. Create comparison DataFrame
            combined_feature_df = pd.DataFrame({
                "Feature": features,
                "ARIMA Model Weight": sarimax_weights.values,
                "Gradient Boosting Importance": xgb_importances
            }).sort_values("ARIMA Model Weight", ascending=False)

            st.dataframe(combined_feature_df.set_index("Feature"), use_container_width=True)
            

        with col2:
            st.markdown("#### 💡 Score Summary")
            st.metric("Predicted FX Rate", f"{currency_symbol}{round(final_pred_value_exp, 4)}")
                
            # FX Strength Indicator
            if final_pred_value_exp > last_actual_value_exp:
                st.success("__The USD is Expected to Appreciate__ 📈")
            elif final_pred_value_exp < last_actual_value_exp:
                st.error("__The USD is Expected to Depreciate__ 📉")
            else:
                st.info("__The USD is Expected to Remain Stable__ ⚖️")

            st.info(f"Root Mean Square Error: {rmse_pct}")
            st.info(f"Mean Absolute Error:    {mae_pct}") 
            st.info(f"Directional Accuracy:   {dir_acc_pct}") 



