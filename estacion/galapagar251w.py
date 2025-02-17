import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

def load_data(filename):
    """Load CSV data (first two lines are metadata)."""
    with open(filename, 'r', encoding='utf-8') as f:
        meta_header = f.readline().strip().split(',')
        meta_values = f.readline().strip().split(',')
    metadata = dict(zip(meta_header, meta_values))
    data = pd.read_csv(filename, skiprows=2, parse_dates=['time'])
    return metadata, data

# Load the trained model and scaler.

model = tf.keras.models.load_model(
    "modelo_7d_to_72h.h5",
    custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
)
with open("scaler_in_7d.pkl", "rb") as f:
    scaler_in = pickle.load(f)
print("Model and scaler loaded.")

# Load 2025 data.
metadata_2025, data = load_data("galapagarhoraria25.csv")
data.sort_values("time", inplace=True)
data.reset_index(drop=True, inplace=True)
data.set_index("time", inplace=True)

# Define feature columns.
feature_cols = ["temperature_2m (°C)", "relative_humidity_2m (%)", "rain (mm)",
                "pressure_msl (hPa)", "wind_speed_10m (km/h)", "wind_direction_10m (°)"]

# We focus on three target variables:
target_cols = ["temperature_2m (°C)", "pressure_msl (hPa)", "relative_humidity_2m (%)"]

# Parameters.
input_hours = 168  # 7 days of history
output_hours = 72  # forecast 72 hours ahead

all_times = data.index

# Choose a forecast start time (ensure at least 7 days of history available).
forecast_start = pd.Timestamp("2025-01-08 00:00:00")
forecast_times = all_times[all_times >= forecast_start]

# For each forecast issuance time t, extract the previous 168 hours,
# scale the features, and run the model to get a forecast sequence of 72 hours.
predictions = {}  # key: issuance time, value: forecast sequence (72, 3)
for t in forecast_times:
    start_time = t - pd.Timedelta(hours=input_hours)
    window = data.loc[start_time:t].iloc[:-1]  # ensure exactly 168 rows
    if len(window) < input_hours:
        continue
    window = window.iloc[-input_hours:]
    window_arr = window[feature_cols].values.astype(np.float32)
    window_scaled = scaler_in.transform(window_arr)
    window_scaled = np.expand_dims(window_scaled, axis=0)  # shape (1,168, num_features)
    
    # Predict a 72-hour sequence.
    pred_seq = model.predict(window_scaled, verbose=0)[0]  # shape (72, 3)
    predictions[t] = pred_seq

# Now, for evaluation and plotting, we want to compare the actual measurement at time T
# with the predictions that were issued 24, 48, and 72 hours earlier.
# That is, for a target time T (T >= forecast_start + 72 hours),
# the forecast issued at T-24 provides the 24th prediction (index 23),
# the forecast issued at T-48 provides the 48th prediction (index 47),
# and the forecast issued at T-72 provides the 72nd prediction (index 71).

target_start = forecast_start + pd.Timedelta(hours=output_hours)
target_times = all_times[all_times >= target_start]

results = []
for t in target_times:
    issuance_24 = t - pd.Timedelta(hours=24)
    issuance_48 = t - pd.Timedelta(hours=48)
    issuance_72 = t - pd.Timedelta(hours=72)
    if (issuance_24 in predictions) and (issuance_48 in predictions) and (issuance_72 in predictions):
        actual = data.loc[t, target_cols].values  # actual measurement at time t
        results.append({
            'time': t,
            'actual_temp': actual[0],
            'actual_pressure': actual[1],
            'actual_humidity': actual[2],
            'pred24_temp': predictions[issuance_24][23, 0],  # 24th hour (index 23)
            'pred24_pressure': predictions[issuance_24][23, 1],
            'pred24_humidity': predictions[issuance_24][23, 2],
            'pred48_temp': predictions[issuance_48][47, 0],  # 48th hour (index 47)
            'pred48_pressure': predictions[issuance_48][47, 1],
            'pred48_humidity': predictions[issuance_48][47, 2],
            'pred72_temp': predictions[issuance_72][71, 0],  # 72nd hour (index 71)
            'pred72_pressure': predictions[issuance_72][71, 1],
            'pred72_humidity': predictions[issuance_72][71, 2]
        })

results_df = pd.DataFrame(results)
results_df.sort_values('time', inplace=True)
results_df.to_csv("predicciones_comparacion_7d_72h.csv", index=False)
print("Prediction results saved to 'predicciones_comparacion_7d_72h.csv'.")

# Plot the three curves for each parameter.
plt.figure(figsize=(15, 10))

# Temperature
plt.subplot(3,1,1)
plt.plot(results_df['time'], results_df['actual_temp'], label="Actual")
plt.plot(results_df['time'], results_df['pred24_temp'], label="Forecast 24h")
plt.plot(results_df['time'], results_df['pred48_temp'], label="Forecast 48h")
plt.plot(results_df['time'], results_df['pred72_temp'], label="Forecast 72h")
plt.title("Temperature Forecast Comparison")
plt.ylabel("Temperature (°C)")
plt.legend()

# Pressure
plt.subplot(3,1,2)
plt.plot(results_df['time'], results_df['actual_pressure'], label="Actual")
plt.plot(results_df['time'], results_df['pred24_pressure'], label="Forecast 24h")
plt.plot(results_df['time'], results_df['pred48_pressure'], label="Forecast 48h")
plt.plot(results_df['time'], results_df['pred72_pressure'], label="Forecast 72h")
plt.title("Pressure Forecast Comparison")
plt.ylabel("Pressure (hPa)")
plt.legend()

# Relative Humidity
plt.subplot(3,1,3)
plt.plot(results_df['time'], results_df['actual_humidity'], label="Actual")
plt.plot(results_df['time'], results_df['pred24_humidity'], label="Forecast 24h")
plt.plot(results_df['time'], results_df['pred48_humidity'], label="Forecast 48h")
plt.plot(results_df['time'], results_df['pred72_humidity'], label="Forecast 72h")
plt.title("Relative Humidity Forecast Comparison")
plt.ylabel("Humidity (%)")
plt.xlabel("Time")
plt.legend()

plt.tight_layout()
plt.show()
