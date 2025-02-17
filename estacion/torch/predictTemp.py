import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import timedelta
import os
import math

# -----------------------------
# Device configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device used:", device, flush=True)

# -----------------------------
# Hyperparameters and features
# -----------------------------
INPUT_WINDOW = 168
OUTPUT_WINDOW = 12
FEATURES = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'rain (mm)',
            'pressure_msl (hPa)', 'wind_speed_10m (km/h)', 'wind_direction_10m (°)']
TEMPORAL_FEATURES = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
ALL_FEATURES = FEATURES + TEMPORAL_FEATURES
OUTPUT_FEATURE = 'temperature_2m (°C)'

HIDDEN_SIZE = 64
NUM_LAYERS = 2

# -----------------------------
# Positional Encoding (same as training)
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# -----------------------------
# Model definitions (same as training)
# -----------------------------
class ForecastNetRNNAtt(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, output_window, dropout=0.0, use_sigmoid=False):
        super(ForecastNetRNNAtt, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        self.decoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.proj = nn.Linear(1, num_features)
        self.output_window = output_window
        self.use_sigmoid = use_sigmoid

    def forward(self, src, target_seq=None, teacher_forcing_ratio=0.0):
        batch_size = src.size(0)
        encoder_outputs, (hidden, cell) = self.encoder(src)
        decoder_input = src[:, -1:, 0:1]  # seed: last value of 1st feature
        outputs = []
        for t in range(self.output_window):
            decoder_input_proj = self.proj(decoder_input)
            out, (hidden, cell) = self.decoder(decoder_input_proj, (hidden, cell))
            dec_hidden = out.repeat(1, encoder_outputs.size(1), 1)
            attn_input = torch.cat((encoder_outputs, dec_hidden), dim=2)
            attn_scores = self.attention(attn_input)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = torch.sum(attn_weights * encoder_outputs, dim=1, keepdim=True)
            combined = torch.cat((out, context), dim=2)
            pred = self.fc(combined)
            if self.use_sigmoid:
                pred = torch.sigmoid(pred)
            outputs.append(pred)
            # Always use model prediction (no teacher forcing during inference)
            decoder_input = pred
        outputs = torch.cat(outputs, dim=1)
        return outputs

class ForecastNetTransformer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, output_window, dropout=0.1, use_sigmoid=False):
        super(ForecastNetTransformer, self).__init__()
        self.d_model = d_model
        self.input_linear = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                           num_encoder_layers=num_encoder_layers,
                                           num_decoder_layers=num_decoder_layers,
                                           dim_feedforward=dim_feedforward,
                                           dropout=dropout, batch_first=True)
        self.output_linear = nn.Linear(d_model, 1)
        self.output_window = output_window
        self.use_sigmoid = use_sigmoid
    def forward(self, src, target_seq=None):
        src_emb = self.input_linear(src)
        src_emb = self.pos_encoder(src_emb)
        batch_size = src.size(0)
        tgt = torch.zeros(batch_size, self.output_window, src.size(2), device=src.device)
        tgt_emb = self.input_linear(tgt)
        tgt_emb = self.pos_decoder(tgt_emb)
        output = self.transformer(src_emb, tgt_emb)
        output = self.output_linear(output)
        if self.use_sigmoid:
            output = torch.sigmoid(output)
        return output

# -----------------------------
# Select model type: "rnn" or "transformer"
# -----------------------------
model_type = "rnn"  # Change to "transformer" if needed

if model_type == "rnn":
    model = ForecastNetRNNAtt(num_features=len(ALL_FEATURES), hidden_size=HIDDEN_SIZE,
                              num_layers=NUM_LAYERS, output_window=OUTPUT_WINDOW,
                              dropout=0.2, use_sigmoid=False)
    model_file = "model/forecast_model_temp_rnnatt.pt"
elif model_type == "transformer":
    model = ForecastNetTransformer(num_features=len(ALL_FEATURES), d_model=64, nhead=4,
                                   num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=128,
                                   output_window=OUTPUT_WINDOW, dropout=0.2, use_sigmoid=False)
    model_file = "model/forecast_model_temp_transformer.pt"
else:
    raise ValueError("Invalid model type")

# Load model
model.load_state_dict(torch.load(model_file, map_location=device))
model = model.to(device)
model.eval()

# -----------------------------
# Helper: Add temporal features
# -----------------------------
def add_temporal_features(df, time_col='time'):
    df[time_col] = pd.to_datetime(df[time_col])
    df['hour'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    return df

# -----------------------------
# Load normalization stats and print them
# -----------------------------
stats_df = pd.read_csv("model/normalization_stats_temp.csv", index_col=0)
print("Normalization stats:")
print(stats_df)

# -----------------------------
# Load and normalize prediction data using saved raw-data stats
# -----------------------------
data_path = 'galapagarhoraria25.csv'
df = pd.read_csv(data_path, skiprows=2, parse_dates=['time'])
df.sort_values('time', inplace=True)
df = add_temporal_features(df, time_col='time')
df_norm = df.copy()

for col in ALL_FEATURES:
    mean, std = float(stats_df.loc[0, col]), float(stats_df.loc[1, col])
    df_norm[col] = (df_norm[col] - mean) / std

df_norm.set_index('time', inplace=True)

# -----------------------------
# Prediction loop with debug logging
# -----------------------------
end_pred = df['time'].max() - pd.Timedelta(hours=OUTPUT_WINDOW)
start_pred = pd.Timestamp("2025-01-08 00:00:00")
forecast_dict = {}

# To hold debug info for the first few windows
debug_logs = []
debug_counter = 0
max_debug_logs = 5  # adjust as needed

current_time = start_pred
while current_time <= end_pred:
    window_start = current_time - pd.Timedelta(hours=INPUT_WINDOW)
    window_data = df_norm.loc[window_start: current_time - pd.Timedelta(hours=1), ALL_FEATURES].values
    if window_data.shape[0] != INPUT_WINDOW:
        current_time += pd.Timedelta(hours=1)
        continue

    # For debugging: get raw and normalized temperature values
    if debug_counter < max_debug_logs:
        raw_window = df[(df['time'] >= window_start) & (df['time'] < current_time)][ALL_FEATURES].values
        temp_raw = raw_window[:, 0]
        temp_norm = window_data[:, 0]
    else:
        temp_raw, temp_norm = None, None

    x_input = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x_input)
    pred_np = pred.squeeze(0).cpu().numpy()  # normalized predictions
    forecast_dict[current_time] = pred_np

    # Denormalize predictions using temperature stats from training
    temp_mean = float(stats_df.loc[0, OUTPUT_FEATURE])
    temp_std = float(stats_df.loc[1, OUTPUT_FEATURE])
    pred_denorm = pred_np * temp_std + temp_mean

    try:
        forecast_4 = pred_np[3][0]
        forecast_8 = pred_np[7][0]
        forecast_12 = pred_np[11][0]
        forecast_4_dn = pred_denorm[3][0]
        forecast_8_dn = pred_denorm[7][0]
        forecast_12_dn = pred_denorm[11][0]
    except Exception as e:
        current_time += pd.Timedelta(hours=1)
        continue

    if debug_counter < max_debug_logs:
        debug_logs.append({
            'current_time': current_time,
            'window_start': window_start,
            'temp_raw_input_mean': np.mean(temp_raw) if temp_raw is not None else None,
            'temp_norm_input_mean': np.mean(temp_norm) if temp_norm is not None else None,
            'pred_normalized_4h': forecast_4,
            'pred_normalized_8h': forecast_8,
            'pred_normalized_12h': forecast_12,
            'pred_denorm_4h': forecast_4_dn,
            'pred_denorm_8h': forecast_8_dn,
            'pred_denorm_12h': forecast_12_dn
        })
        debug_counter += 1

    current_time += pd.Timedelta(hours=1)

# -----------------------------
# Save evaluation results
# -----------------------------
eval_start = pd.Timestamp("2025-01-11 00:00:00")
results = []
current_time = eval_start
while current_time in df_norm.index:
    try:
        real = df_norm.loc[current_time, OUTPUT_FEATURE]
        forecast_4 = forecast_dict[current_time - pd.Timedelta(hours=4)][3][0]
        forecast_8 = forecast_dict[current_time - pd.Timedelta(hours=8)][7][0]
        forecast_12 = forecast_dict[current_time - pd.Timedelta(hours=12)][11][0]
    except KeyError:
        current_time += pd.Timedelta(hours=1)
        continue
    results.append({
        'time': current_time,
        'real': real,
        'pred_4': forecast_4,
        'pred_8': forecast_8,
        'pred_12': forecast_12
    })
    current_time += pd.Timedelta(hours=1)

df_res = pd.DataFrame(results)
# Denormalize using temperature stats
df_res['real'] = df_res['real'] * temp_std + temp_mean
df_res['pred_4'] = df_res['pred_4'] * temp_std + temp_mean
df_res['pred_8'] = df_res['pred_8'] * temp_std + temp_mean
df_res['pred_12'] = df_res['pred_12'] * temp_std + temp_mean

os.makedirs("results", exist_ok=True)
plt.figure(figsize=(12, 6))
plt.plot(df_res['time'], df_res['real'], label='Real')
plt.plot(df_res['time'], df_res['pred_4'], label='Prediction 4h')
plt.plot(df_res['time'], df_res['pred_8'], label='Prediction 8h')
plt.plot(df_res['time'], df_res['pred_12'], label='Prediction 12h')
plt.xlabel('Time')
plt.ylabel(OUTPUT_FEATURE)
plt.title('Temperature Predictions Comparison')
plt.legend()
plt.tight_layout()
plt.savefig("results/temperature_predictions.png")
plt.close()
df_res.to_csv("results/temperature_predictions.csv", index=False)
print("Prediction and evaluation complete. Results saved in 'results'.")

# -----------------------------
# Save debug logs for review
# -----------------------------
if debug_logs:
    debug_df = pd.DataFrame(debug_logs)
    debug_df.to_csv("results/debug_intermediates.csv", index=False)
    print("Debug intermediate values saved to 'results/debug_intermediates.csv'.")
