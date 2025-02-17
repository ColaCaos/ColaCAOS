import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

def load_data(filename):
    """Load data from CSV (first two lines are metadata)."""
    with open(filename, 'r', encoding='utf-8') as f:
        meta_header = f.readline().strip().split(',')
        meta_values = f.readline().strip().split(',')
    metadata = dict(zip(meta_header, meta_values))
    data = pd.read_csv(filename, skiprows=2, parse_dates=['time'])
    return metadata, data

# Load the trained model and scaler
model = tf.keras.models.load_model("modelo_meteo_24h.h5",
                                   custom_objects={'mse': tf.keras.losses.MeanSquaredError(),
                                                   'binary_crossentropy': tf.keras.losses.BinaryCrossentropy()})
with open("scaler_in.pkl", "rb") as f:
    scaler_in = pickle.load(f)
print("Model and scaler loaded.")

# Load 2025 data, sort, and set time as index
metadata_2025, data = load_data("galapagarhoraria25.csv")
data.sort_values("time", inplace=True)
data.reset_index(drop=True, inplace=True)
data.set_index("time", inplace=True)

# Define feature columns (same order as used in training)
feature_cols = ["temperature_2m (°C)", "relative_humidity_2m (%)", "rain (mm)",
                "pressure_msl (hPa)", "wind_speed_10m (km/h)", "wind_direction_10m (°)"]

# Set parameters for forecasting
sequence_length = 72   # input window length
forecast_horizon = 24  # forecast 24 hours ahead

all_times = data.index

# Define forecast start time (ensuring at least 72 hours history are available)
forecast_start = pd.Timestamp("2025-01-04 00:00:00")
forecast_times = all_times[all_times >= forecast_start]

# For each forecast issuance time, use the preceding 72 hours to predict 24 hours ahead.
predictions = {}
for t in forecast_times:
    start_time = t - pd.Timedelta(hours=sequence_length)
    window = data.loc[start_time:t].iloc[:-1]  # ensure exactly 72 rows
    if len(window) < sequence_length:
        continue
    window = window.iloc[-sequence_length:]
    window_arr = window[feature_cols].values.astype(np.float32)
    window_scaled = scaler_in.transform(window_arr)
    window_scaled = np.expand_dims(window_scaled, axis=0)  # shape: (1, 72, num_features)
    
    preds = model.predict(window_scaled, verbose=0)
    # preds[0]: continuous output (shape (1,3)): [temperature, pressure, humidity]
    # preds[1]: rain output (shape (1,1))
    pred_cont = preds[0][0]
    pred_rain = preds[1][0][0]
    
    predictions[t] = {
        'cont_24': pred_cont,  # [temp, pressure, humidity]
        'rain_24': pred_rain
    }

# Compare predictions with actual values: for each target time t, the forecast was issued at t - 24 hours.
target_start = pd.Timestamp("2025-01-05 00:00:00")
target_times = all_times[all_times >= target_start]

results = []
for t in target_times:
    issuance = t - pd.Timedelta(hours=forecast_horizon)
    if issuance in predictions:
        actual = data.loc[t, feature_cols].values
        results.append({
            'time': t,
            'actual_temp': actual[0],
            'actual_pressure': actual[3],
            'actual_humidity': actual[1],
            'actual_rain': actual[2],
            'pred24_temp': predictions[issuance]['cont_24'][0],
            'pred24_pressure': predictions[issuance]['cont_24'][1],
            'pred24_humidity': predictions[issuance]['cont_24'][2],
            'pred24_rain': predictions[issuance]['rain_24']
        })

results_df = pd.DataFrame(results)
results_df.sort_values('time', inplace=True)
results_df.to_csv("predicciones_comparacion_24h.csv", index=False)
print("Prediction results saved to 'predicciones_comparacion_24h.csv'.")

# Plot results for each parameter
plt.figure(figsize=(15, 10))

# Temperature
plt.subplot(2,2,1)
plt.plot(results_df['time'], results_df['actual_temp'], label="Real")
plt.plot(results_df['time'], results_df['pred24_temp'], label="Pred 24h")
plt.title("Temperature (°C)")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()

# Pressure
plt.subplot(2,2,2)
plt.plot(results_df['time'], results_df['actual_pressure'], label="Real")
plt.plot(results_df['time'], results_df['pred24_pressure'], label="Pred 24h")
plt.title("Pressure (hPa)")
plt.xlabel("Time")
plt.ylabel("Pressure")
plt.legend()

# Relative Humidity
plt.subplot(2,2,3)
plt.plot(results_df['time'], results_df['actual_humidity'], label="Real")
plt.plot(results_df['time'], results_df['pred24_humidity'], label="Pred 24h")
plt.title("Relative Humidity (%)")
plt.xlabel("Time")
plt.ylabel("Humidity")
plt.legend()

# Rain
plt.subplot(2,2,4)
plt.plot(results_df['time'], results_df['actual_rain'], label="Real")
plt.plot(results_df['time'], results_df['pred24_rain'], label="Pred 24h")
plt.title("Rain (mm or probability)")
plt.xlabel("Time")
plt.ylabel("Rain")
plt.legend()

plt.tight_layout()
plt.show()
