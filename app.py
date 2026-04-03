import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="V.Ger Forecasting Engine", page_icon="🏨", layout="wide")

# 2. Main Header
st.title("V.Ger Hotel Forecasting Dashboard")
st.markdown("Comparing our Operational Baseline (SARIMA) against Strategic Pricing Simulations (TFT).")
st.divider()

# 3. Sidebar for User Inputs
st.sidebar.header("Scenario Controls")
st.sidebar.markdown("Adjust the ADR to see how the Deep Learning model reacts compared to the baseline.")

simulated_adr = st.sidebar.slider(
    "Simulated ADR ($)", 
    min_value=50.0, 
    max_value=250.0, 
    value=100.0, 
    step=5.0
)

# 4. Define BOTH API Endpoints
STRATEGIC_URL = "https://chibikeobi-vger-forecast.hf.space/forecast/strategic" 
OPERATIONAL_URL = "https://chibikeobi-vger-forecast.hf.space/forecast/operational"

if st.sidebar.button("Generate Forecast", type="primary"):
    with st.spinner("Waking up both AI models in the cloud..."):
        try:
            # 5. Fetch from BOTH endpoints
            strat_response = requests.get(STRATEGIC_URL, params={"simulated_adr": simulated_adr})
            op_response = requests.get(OPERATIONAL_URL)
            
            if strat_response.status_code == 200 and op_response.status_code == 200:
                strat_data = strat_response.json()
                op_data = op_response.json()
                
                st.success("Forecasts retrieved and processed successfully!")
                
                # 6. Parse the JSON into Pandas DataFrames
                # Extract dates and values from the Strategic JSON (TFT)
                strat_dates = list(strat_data["simulated_forecast"].keys())
                strat_values = list(strat_data["simulated_forecast"].values())
                df_strat = pd.DataFrame({"Date": strat_dates, "Strategic (TFT)": strat_values})
                df_strat["Date"] = pd.to_datetime(df_strat["Date"])
                
                # Extract dates and values from the Operational JSON (SARIMA)
                # *Note: Adjust "forecast" below if your SARIMA JSON uses a different key!
                op_dates = list(op_data["forecast"].keys())
                op_values = list(op_data["forecast"].values())
                df_op = pd.DataFrame({"Date": op_dates, "Operational (SARIMA)": op_values})
                df_op["Date"] = pd.to_datetime(df_op["Date"])
                
                # Merge both models on the Date column
                df_combined = pd.merge(df_strat, df_op, on="Date")
                
                # 7. Create the Interactive Plotly Chart
                st.subheader(f"10-Week Booking Forecast (Simulated ADR: ${simulated_adr})")
                
                fig = px.line(
                    df_combined, 
                    x="Date", 
                    y=["Operational (SARIMA)", "Strategic (TFT)"],
                    labels={"value": "Predicted Bookings", "variable": "Model"},
                    color_discrete_map={
                        "Operational (SARIMA)": "#808495", # Muted grey for baseline
                        "Strategic (TFT)": "#3b82f6"       # Bright blue for the active simulation
                    },
                    markers=True
                )
                
                # Make the chart look sleek
                fig.update_layout(hovermode="x unified", legend_title_text=None)
                st.plotly_chart(fig, use_container_width=True)
                
                # 8. Show the raw data table below the chart
                with st.expander("View Raw Forecast Data"):
                    st.dataframe(df_combined, use_container_width=True)
                
            else:
                st.error("API Error: One of the models failed to return data.")
                
        except KeyError as e:
            st.error(f"Data Parsing Error: Could not find the expected key in the JSON: {e}")
            st.info("Check your JSON output to ensure the keys match 'simulated_forecast' and 'forecast'.")
        except Exception as e:
            st.error(f"Failed to connect to the cloud engine: {e}")
else:
    st.info("Adjust your parameters and click 'Generate Forecast' to run the simulation.")