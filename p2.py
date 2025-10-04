# streamlit_smart_city_lstm.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---------------------
# Load Data
# ---------------------
st.title("🌆 Smart City Dashboard with AI Predictions")

@st.cache_data
def load_data():
    df = pd.read_csv("smart_city_sample_dataset.csv", parse_dates=["Timestamp"])
    return df

df = load_data()

st.sidebar.header("Filters")
district_filter = st.sidebar.multiselect("Select District(s)", df["District"].unique(), default=df["District"].unique())
df_filtered = df[df["District"].isin(district_filter)]

# ---------------------
# Data Overview
# ---------------------
st.subheader("📊 Data Overview")
st.dataframe(df_filtered)

st.subheader("🔹 Statistics")
st.write(df_filtered.describe())

# ---------------------
# Interactive Graphs
# ---------------------
metrics = ["Traffic_Flow", "Avg_Speed_km_h", "Power_Consumption_kWh",
           "Water_Usage_L", "Crime_Incidents", "Emergency_Calls", "Public_Service_Requests"]

st.subheader("📈 Interactive Metrics Over Time")
for metric in metrics:
    fig = px.line(df_filtered, x="Timestamp", y=metric, color="District", markers=True, title=f"{metric} Over Time")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------
# LSTM Prediction Function
# ---------------------
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape))
    model.add(Dense(len(metrics)))
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_lstm_data(df, metrics, look_back=3):
    data = df[metrics].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:i+look_back])
        y.append(scaled_data[i+look_back])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# ---------------------
# Forecasting Next 6 Hours
# ---------------------
st.subheader("🔮 Multi-Metric Forecast (Next 6 Hours)")

look_back = 3  # past 3 time steps
forecast_horizon = 6  # next 6 steps

for district in df_filtered["District"].unique():
    st.markdown(f"### District {district}")
    df_d = df_filtered[df_filtered["District"] == district].reset_index(drop=True)
    
    if len(df_d) <= look_back:
        st.warning("Not enough data for LSTM prediction for this district.")
        continue
    
    X, y, scaler = prepare_lstm_data(df_d, metrics, look_back)
    model = create_lstm_model((look_back, len(metrics)))
    
    # Train LSTM (for demo purposes, epochs=50)
    model.fit(X, y, epochs=50, batch_size=1, verbose=0)
    
    # Forecast next steps
    last_seq = X[-1]  # last sequence
    preds_scaled = []
    current_seq = last_seq
    for _ in range(forecast_horizon):
        pred = model.predict(current_seq[np.newaxis, :, :], verbose=0)
        preds_scaled.append(pred[0])
        # update sequence
        current_seq = np.vstack([current_seq[1:], pred])
    
    preds_scaled = np.array(preds_scaled)
    preds = scaler.inverse_transform(preds_scaled)
    
    # Build forecast DataFrame
    last_time = df_d["Timestamp"].iloc[-1]
    future_times = pd.date_range(last_time + pd.Timedelta(hours=1), periods=forecast_horizon, freq='H')
    df_forecast = pd.DataFrame(preds, columns=metrics)
    df_forecast["Timestamp"] = future_times
    
    # Combine past and forecast for plotting
    df_plot = pd.concat([df_d[["Timestamp"] + metrics], df_forecast])
    
    # Plot forecast for each metric
    for metric in metrics:
        fig = px.line(df_plot, x="Timestamp", y=metric,
                      title=f"{metric} - District {district} (Forecast Next {forecast_horizon} Hours)")
        st.plotly_chart(fig, use_container_width=True)
