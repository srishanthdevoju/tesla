
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Custom CSS for a clean look ---
st.markdown("""
<style>
.main {
    background-color: #f0f2f6;
}
.sidebar .sidebar-content {
    background-color: #ffffff;
    padding: 20px;
    border-right: 1px solid #e0e0e0;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    border-radius: 5px;
    border: none;
    cursor: pointer;
    font-size: 16px;
}
.stButton>button:hover {
    background-color: #45a049;
}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
  font-size:1.2rem;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 5rem;
    padding-right: 5rem;
}
</style>
""", unsafe_allow_html=True)


# --- Page Configuration ---
st.set_page_config(
    page_title="Tesla EV Deliveries Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar ---
st.sidebar.title("🚗 Tesla EV Data Analysis")
st.sidebar.markdown("Explore key insights and model predictions for Tesla EV Deliveries and Production.")

# Navigation
page = st.sidebar.radio(
    "Go to",
    ["Overview", "EDA", "ML Models", "Time Series", "Conclusions"]
)

# Sample Data (Replace with your actual loaded data 'df' from the notebook)
# For demonstration, creating a dummy dataframe
@st.cache_data
def load_sample_data():
    data = {
        'Year': np.random.choice(range(2015, 2026), 1000),
        'Month': np.random.choice(range(1, 13), 1000),
        'Region': np.random.choice(['North America', 'Europe', 'Asia', 'Middle East'], 1000),
        'Model': np.random.choice(['Model S', 'Model 3', 'Model X', 'Model Y', 'Cybertruck'], 1000),
        'Estimated_Deliveries': np.random.randint(1000, 20000, 1000),
        'Production_Units': np.random.randint(1000, 25000, 1000),
        'Avg_Price_USD': np.random.uniform(50000, 120000, 1000),
        'CO2_Saved_tons': np.random.uniform(100, 2000, 1000),
        'Date': pd.to_datetime(np.random.choice(pd.to_datetime(['2015-01-01', '2025-12-31']), 1000))
    }
    df_sample = pd.DataFrame(data)
    return df_sample

df_sample = load_sample_data()

# --- Main Content ---

if page == "Overview":
    st.title("Dashboard Overview")
    st.header("Tesla EV Deliveries and Production Analysis")
    st.markdown("This interactive dashboard provides insights into Tesla's Electric Vehicle (EV) deliveries, production, and related metrics from 2015 to 2025.")

    st.subheader("Dataset Snapshot")
    st.write(df_sample.head())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Estimated Deliveries (Sample)", f"{df_sample['Estimated_Deliveries'].sum():,.0f}")
    with col2:
        st.metric("Average Price (Sample)", f"${df_sample['Avg_Price_USD'].mean():,.2f}")
    with col3:
        st.metric("Total Production Units (Sample)", f"{df_sample['Production_Units'].sum():,.0f}")

    st.subheader("Key Data Insights")
    st.info(
        "The dataset includes information on: Year, Month, Region, Model, Estimated Deliveries, Production Units, Average Price, Battery Capacity, Range, CO2 Saved, Source Type, and Charging Stations."
    )

elif page == "EDA":
    st.title("Exploratory Data Analysis")

    st.subheader("Correlation Matrix")
    st.write("Heatmap showing the correlation between numerical features.")
    # Replace with your actual correlation_matrix if available
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_sample.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f', ax=ax_corr)
    st.pyplot(fig_corr)

    st.subheader("Distribution of Estimated Deliveries")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    sns.histplot(df_sample['Estimated_Deliveries'], kde=True, ax=ax_hist)
    st.pyplot(fig_hist)

    st.subheader("Deliveries by Region and Model")
    region_model_data = df_sample.groupby(['Region', 'Model'])['Estimated_Deliveries'].sum().unstack(fill_value=0)
    fig_region, ax_region = plt.subplots(figsize=(12, 8))
    sns.heatmap(region_model_data, annot=True, fmt='d', cmap='viridis', ax=ax_region)
    st.pyplot(fig_region)

elif page == "ML Models":
    st.title("Machine Learning Model Evaluation")

    st.write("Summary of various regression models trained to predict Estimated Deliveries.")

    st.subheader("Model Performance Metrics")

    # This part should be populated with your actual model results
    model_results = {
        "Model": ["Linear Regression", "Tuned Ridge Regression", "Tuned Lasso Regression"],
        "Train MAE": [315.39, 315.37, 316.07],
        "Train MSE": [149113.46, 149119.20, 149960.73],
        "Train R2": [0.9905, 0.9905, 0.9904],
        "Test MAE": [309.80, 309.70, 309.78],
        "Test MSE": [147422.64, 147434.16, 146047.44],
        "Test R2": [0.9901, 0.9901, 0.9902]
    }
    results_df = pd.DataFrame(model_results)
    st.dataframe(results_df)

    st.subheader("Overfitting Analysis")
    st.success("The models are **not overfitting**. Performance metrics (MAE, MSE, R2) are consistent between training and test sets, and cross-validation showed low standard deviation.")

elif page == "Time Series":
    st.title("Time Series Forecasting")

    st.write("Analysis and forecasting of time-dependent features like Estimated Deliveries, Production Units, and Average Price.")

    st.subheader("Time Series Trends")
    time_series_plot_data = df_sample.groupby('Date').agg({
        'Estimated_Deliveries': 'sum',
        'Production_Units': 'sum',
        'Avg_Price_USD': 'mean'
    }).reset_index()

    fig_ts, ax_ts = plt.subplots(figsize=(15, 7))
    sns.lineplot(x='Date', y='Estimated_Deliveries', data=time_series_plot_data, label='Estimated Deliveries', ax=ax_ts)
    sns.lineplot(x='Date', y='Production_Units', data=time_series_plot_data, label='Production Units', ax=ax_ts)
    ax_ts.set_title('Time Series Trends')
    ax_ts.set_xlabel('Date')
    ax_ts.set_ylabel('Value')
    ax_ts.legend()
    st.pyplot(fig_ts)

    st.subheader("ARIMA Model Forecast (Example)")
    st.info("In your notebook, an ARIMA(5,1,0) model was used for 'Estimated_Deliveries'. The MSE was 169,786,520.58.")

    st.subheader("VAR Model Forecast (Example)")
    st.info("A VAR model was trained on differenced series. For Estimated_Deliveries, MSE was 408,428,477.06.")

elif page == "Conclusions":
    st.title("Key Conclusions")
    st.markdown("""
    - **Data Cleaning**: No missing values or duplicates were found. Outliers were retained as they represent real-world variations.
    - **EDA**: Strong positive correlation between `Production_Units`, `CO2_Saved_tons` and `Estimated_Deliveries`.
    - **ML Models**: Linear, Ridge, and Lasso Regression models showed consistently high R2 scores (approx. 0.99) on both training and test sets, indicating strong predictive power and **no overfitting**.
    - **Time Series**: ARIMA and VAR models were applied, with all main series (`Estimated_Deliveries`, `Production_Units`, `Avg_Price_USD`) found to be stationary.
    """)

    st.subheader("Next Steps")
    st.markdown("""
    - Consider deploying the best performing regression model for real-time predictions.
    - Further investigate the VAR model by inverse differencing to get forecasts in the original scale.
    - Explore more advanced time series models like Prophet or LSTM for potentially better forecasting accuracy.
    """)
