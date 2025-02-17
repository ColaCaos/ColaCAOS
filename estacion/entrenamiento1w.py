import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

def load_data(filename):
    """Load CSV data (first two lines are metadata)."""
    with open(filename, 'r', encoding='utf-8') as f:
        meta_header = f.readline().strip().split(',')
        meta_values = f.readline().strip().split(',')
    metadata = dict(zip(meta_header, meta_values))
    data = pd.read_csv(filename, skiprows=2, parse_dates=['time'])
    return metadata, data

def force_numeric(data, cols):
    """Ensure columns are numeric."""
    for col in cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data

def create_sequences(data, input_hours=168, output_hours=72):
    """
    Create sequences from the data.
    
    Input: previous 168 hours of features.
    Output: next 72 hours of targets (continuous variables).
    
    We use the following columns for features:
      - feature_cols: ["temperature_2m (°C)", "relative_humidity_2m (%)", "rain (mm)",
                        "pressure_msl (hPa)", "wind_speed_10m (km/h)", "wind_direction_10m (°)"]
    
    And for targets (continuous):
      - target_cols: ["temperature_2m (°C)", "pressure_msl (hPa)", "relative_humidity_2m (%)"]
    """
    feature_cols = ["temperature_2m (°C)", "relative_humidity_2m (%)", "rain (mm)",
                    "pressure_msl (hPa)", "wind_speed_10m (km/h)", "wind_direction_10m (°)"]
    target_cols = ["temperature_2m (°C)", "pressure_msl (hPa)", "relative_humidity_2m (%)"]
    
    X_list, y_list = [], []
    total_steps = len(data)
    # We need input_hours past data and output_hours future data.
    for i in range(input_hours, total_steps - output_hours):
        X_seq = data.iloc[i-input_hours:i][feature_cols].values.astype(np.float32)
        y_seq = data.iloc[i:i+output_hours][target_cols].values.astype(np.float32)
        X_list.append(X_seq)
        y_list.append(y_seq)
    X = np.array(X_list)  # shape (n_samples, 168, 6)
    y = np.array(y_list)  # shape (n_samples, 72, 3)
    return X, y

# Load and preprocess the data
metadata, data = load_data("galapagarhoraria.csv")
print("Data loaded with shape:", data.shape)
cols = ["temperature_2m (°C)", "relative_humidity_2m (%)", "rain (mm)",
        "pressure_msl (hPa)", "wind_speed_10m (km/h)", "wind_direction_10m (°)"]
data = force_numeric(data, cols)

# Create sequences using the previous 7 days (168 hours) to forecast next 72 hours.
X, y = create_sequences(data, input_hours=168, output_hours=72)
print("Created sequences:", X.shape, y.shape)

# Split into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale input features (we can fit scaler on all features together).
num_features = X.shape[2]
scaler_in = StandardScaler()
X_train_2d = X_train.reshape(-1, num_features)
scaler_in.fit(X_train_2d)
X_train_scaled = scaler_in.transform(X_train_2d).reshape(X_train.shape)
X_val_2d = X_val.reshape(-1, num_features)
X_val_scaled = scaler_in.transform(X_val_2d).reshape(X_val.shape)

# Optionally, you might want to scale targets too; here we leave targets in raw units.

# Build an encoder-decoder model.
input_seq = Input(shape=(168, num_features), name="encoder_input")
x = BatchNormalization()(input_seq)
encoder_lstm = LSTM(64, return_state=True, name="encoder_lstm")
encoder_outputs, state_h, state_c = encoder_lstm(x)
encoder_states = [state_h, state_c]

# Decoder: use RepeatVector to create a sequence of length 72 from the encoder state.
decoder_inputs = RepeatVector(72, name="repeat_vector")(state_h)  # or use state_h as starting input
decoder_lstm = LSTM(64, return_sequences=True, name="decoder_lstm")
decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = TimeDistributed(Dense(3), name="time_distributed_dense")
decoder_outputs = decoder_dense(decoder_outputs)  # output shape (72, 3)

model = Model(inputs=input_seq, outputs=decoder_outputs)
model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
model.summary()

# Train the model.
history = model.fit(X_train_scaled, y_train,
                    validation_data=(X_val_scaled, y_val),
                    epochs=50, batch_size=64)

# Save the trained model and the input scaler.
model.save("modelo_7d_to_72h.h5")
with open("scaler_in_7d.pkl", "wb") as f:
    pickle.dump(scaler_in, f)
print("Model and scaler saved.")
