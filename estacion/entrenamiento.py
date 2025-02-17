import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_galapagar_data(filename):
    """Load data from CSV (first two lines are metadata)."""
    with open(filename, 'r', encoding='utf-8') as f:
        meta_header = f.readline().strip().split(',')
        meta_values = f.readline().strip().split(',')
    metadata = dict(zip(meta_header, meta_values))
    data = pd.read_csv(filename, skiprows=2, parse_dates=['time'])
    return metadata, data

def force_numeric(data, cols):
    """Convert specified columns of the DataFrame to numeric dtype."""
    for col in cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data

def create_sequences(data, sequence_length=72, forecast_horizon=24):
    """
    Create sequences for a 24-hour ahead forecast.
    Input (X): previous 72 hours (features).
    Targets:
      - y_cont: continuous targets at i+24 for [temperature, pressure, humidity]
      - y_rain: binary target (1 if rain > 0, else 0) at i+24.
    """
    feature_cols = ["temperature_2m (°C)", "relative_humidity_2m (%)", "rain (mm)",
                    "pressure_msl (hPa)", "wind_speed_10m (km/h)", "wind_direction_10m (°)"]
    y_cont_cols = ["temperature_2m (°C)", "pressure_msl (hPa)", "relative_humidity_2m (%)"]
    X_list, y_cont_list, y_rain_list = [], [], []
    n = len(data)
    for i in range(sequence_length, n - forecast_horizon):
        X_list.append(data.iloc[i-sequence_length:i][feature_cols].values.astype(np.float32))
        y_cont_list.append(data.iloc[i+forecast_horizon][y_cont_cols].values.astype(np.float32))
        rain_val = data.iloc[i+forecast_horizon]["rain (mm)"]
        y_rain_list.append(1.0 if rain_val > 0 else 0.0)
    X = np.array(X_list, dtype=np.float32)
    y_cont = np.array(y_cont_list, dtype=np.float32)
    y_rain = np.array(y_rain_list, dtype=np.float32).reshape(-1, 1)
    return X, y_cont, y_rain

def build_model_24h(time_steps, num_features, dense_units=32):
    """Build a model that uses 72-hour inputs to forecast 24 hours ahead."""
    inputs = Input(shape=(time_steps, num_features), name='input')
    x = BatchNormalization()(inputs)
    x = LSTM(64, return_sequences=True, name='lstm1')(x)
    x = Dropout(0.3)(x)
    x = LSTM(32, return_sequences=False, name='lstm2')(x)
    x = Dropout(0.3)(x)
    x = Dense(dense_units, activation='relu', name='dense')(x)
    # Continuous outputs: [temperature, pressure, humidity]
    out_cont = Dense(3, activation='linear', name='output_cont')(x)
    # Rain output: binary classification (rain/no rain)
    out_rain = Dense(1, activation='sigmoid', name='output_rain')(x)
    model = Model(inputs=inputs, outputs=[out_cont, out_rain])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss={'output_cont': 'mse', 'output_rain': 'binary_crossentropy'},
                  metrics={'output_cont': 'mae', 'output_rain': 'accuracy'})
    return model

if __name__ == '__main__':
    # Load historical data
    metadata, data = load_galapagar_data("galapagarhoraria.csv")
    print("Data loaded with shape:", data.shape)
    
    # Force numeric conversion for the relevant columns
    num_cols = ["temperature_2m (°C)", "relative_humidity_2m (%)", "rain (mm)",
                "pressure_msl (hPa)", "wind_speed_10m (km/h)", "wind_direction_10m (°)"]
    data = force_numeric(data, num_cols)
    
    # Create sequences (inputs: 72h, target: value 24h ahead)
    sequence_length = 72
    forecast_horizon = 24
    X, y_cont, y_rain = create_sequences(data, sequence_length, forecast_horizon)
    print("Created sequences:", X.shape, y_cont.shape, y_rain.shape)
    
    # Split data into training and validation sets (80%/20%)
    X_train, X_val, y_cont_train, y_cont_val, y_rain_train, y_rain_val = train_test_split(
        X, y_cont, y_rain, test_size=0.2, random_state=42)
    
    # Scale input features
    num_features = X.shape[2]
    scaler_in = StandardScaler()
    X_train_2d = X_train.reshape(-1, num_features)
    scaler_in.fit(X_train_2d)
    X_train_scaled = scaler_in.transform(X_train_2d).reshape(X_train.shape)
    X_val_2d = X_val.reshape(-1, num_features)
    X_val_scaled = scaler_in.transform(X_val_2d).reshape(X_val.shape)
    
    # Build and train the model
    model = build_model_24h(time_steps=sequence_length, num_features=num_features, dense_units=32)
    model.summary()
    history = model.fit(X_train_scaled,
                        {'output_cont': y_cont_train, 'output_rain': y_rain_train},
                        validation_data=(X_val_scaled, {'output_cont': y_cont_val, 'output_rain': y_rain_val}),
                        epochs=50, batch_size=64)
    
    # Save the trained model and input scaler
    model.save("modelo_meteo_24h.h5")
    import pickle
    with open("scaler_in.pkl", "wb") as f:
        pickle.dump(scaler_in, f)
    print("Model and scaler saved.")
